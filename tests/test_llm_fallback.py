"""Fallback chain in llm.py (no network: the per-endpoint call is faked)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import llm
import llm_usage


class Status(Exception):
    def __init__(self, code: int, msg: str = "") -> None:
        super().__init__(msg or f"HTTP {code}")
        self.status_code = code


def ep(role: str, model: str = "m") -> llm.Endpoint:
    return llm.Endpoint("openrouter", role, model, f"key-{role}", llm.OPENROUTER_BASE_URL)


class RunChainTests(unittest.TestCase):
    def run_chain(self, replies, endpoints=None, retries=1):
        """replies: endpoint role -> list of results (str or exception), consumed in order."""
        queue = {k: list(v) for k, v in replies.items()}
        calls = []

        def call(e):
            calls.append(e.role)
            item = queue[e.role].pop(0)
            if isinstance(item, Exception):
                raise item
            return item, (10, 5, None)

        with llm_usage.UsageTracker() as usage:
            try:
                result = llm.run_chain(endpoints or [ep("primary"), ep("fallback", "m2")], call,
                                       retries=retries, sleep=lambda s: None)
            except llm.LLMUnavailableError as exc:
                result = exc
        return result, calls, usage.records

    def test_primary_answers(self) -> None:
        result, calls, records = self.run_chain({"primary": ["ok"]})
        self.assertEqual((result, calls), ("ok", ["primary"]))
        self.assertEqual([(r.role, r.ok, r.prompt_tokens) for r in records], [("primary", True, 10)])

    def test_transient_error_is_retried_on_the_same_endpoint(self) -> None:
        result, calls, _ = self.run_chain({"primary": [Status(503), "ok"]})
        self.assertEqual((result, calls), ("ok", ["primary", "primary"]))

    def test_empty_reply_counts_as_transient(self) -> None:
        result, calls, _ = self.run_chain({"primary": ["  ", "ok"]})
        self.assertEqual((result, calls), ("ok", ["primary", "primary"]))

    def test_outage_moves_to_the_fallback(self) -> None:
        result, calls, records = self.run_chain({"primary": [Status(503), Status(503)], "fallback": ["backup"]})
        self.assertEqual((result, calls), ("backup", ["primary", "primary", "fallback"]))
        self.assertEqual([r.ok for r in records], [False, False, True])
        self.assertEqual(records[-1].role, "fallback")
        self.assertEqual(llm_usage.summarize(records)["total"]["fallback_calls"], 1)

    def test_bad_key_skips_retry_and_falls_back(self) -> None:
        result, calls, _ = self.run_chain({"primary": [Status(401, "invalid key")], "fallback": ["backup"]})
        self.assertEqual((result, calls), ("backup", ["primary", "fallback"]))

    def test_everything_failing_reports_each_endpoint(self) -> None:
        result, _, _ = self.run_chain({"primary": [Status(401)], "fallback": [Status(404, "no such model")]})
        self.assertIsInstance(result, llm.LLMUnavailableError)
        self.assertIn("primary openrouter/m", str(result))
        self.assertIn("no such model", str(result))


class EndpointConfigTests(unittest.TestCase):
    def test_chat_chain_from_env(self) -> None:
        env = {"OPENROUTER_API_KEY": "k1", "LLM_FALLBACK_API_KEY": "k2", "LLM_FALLBACK_MODEL": "backup-model",
               "LLM_FALLBACK_BASE_URL": "https://api.groq.com/openai/v1", "LLM_MODEL": "main-model"}
        with patch.dict(os.environ, env):
            chain = llm.chat_endpoints()
        self.assertEqual([(e.role, e.provider, e.model) for e in chain],
                         [("primary", "openrouter", "main-model"), ("fallback", "api.groq.com", "backup-model")])

    def test_no_fallback_configured(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k1", "LLM_FALLBACK_API_KEY": ""}):
            self.assertEqual([e.role for e in llm.chat_endpoints()], ["primary"])

    def test_gemini_falls_back_to_gemini_backup_then_openrouter(self) -> None:
        env = {"GEMINI_API_KEY": "g1", "GEMINI_FALLBACK_API_KEY": "g2", "OPENROUTER_API_KEY": "k1",
               "LLM_FALLBACK_API_KEY": "", "GEMINI_FALLBACK_TO_OPENROUTER": "true"}
        with patch.dict(os.environ, env):
            chain = llm.gemini_endpoints()
        self.assertEqual([(e.provider, e.role) for e in chain],
                         [("gemini", "primary"), ("gemini", "fallback"), ("openrouter", "fallback")])

    def test_nothing_configured_is_a_clear_error(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "LLM_FALLBACK_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY is not set"):
                llm.chat_completion(system="s", user="u")


class CostTests(unittest.TestCase):
    def test_provider_cost_wins_and_price_table_estimates_otherwise(self) -> None:
        a = llm_usage.record(provider="openrouter", model="meta-llama/llama-4-scout", role="primary", ok=True,
                             latency_s=1, prompt_tokens=1000, completion_tokens=100, cost_usd=0.0002)
        b = llm_usage.record(provider="gemini", model="gemini-3.1-flash-lite-preview", role="primary", ok=True,
                             latency_s=1, prompt_tokens=1_000_000, completion_tokens=1_000_000)
        self.assertEqual((a.cost_usd, a.cost_source), (0.0002, "provider"))
        self.assertEqual((round(b.cost_usd, 2), b.cost_source), (1.75, "price_table"))


if __name__ == "__main__":
    unittest.main()
