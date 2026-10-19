from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from product_matcher import Candidate, atomise, confidence_band, find_candidates, pick_product
from tests.fake_catalogue_search import FakeSearchTool, product

SWITCH = product("Coral Ebony", "16A 2Way Switch")
SWITCH_ORO = product("Oro Metallica", "16A 2Way Switch")


class FindCandidatesTests(unittest.TestCase):
    def test_drops_unrelated_candidates(self) -> None:
        tool = FakeSearchTool({"switch": [(SWITCH, 0.99), (SWITCH_ORO, 0.01)]})
        with patch("product_matcher.get_bu_rag_tool", return_value=tool):
            found = find_candidates("16A 2Way switch")
        self.assertEqual([c.product for c in found], [SWITCH])


class ConfidenceTests(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(confidence_band(0.97), "high")
        self.assertEqual(confidence_band(0.90), "high")
        self.assertEqual(confidence_band(0.70), "medium")
        self.assertEqual(confidence_band(0.30), "low")


class PickProductTests(unittest.TestCase):
    CANDIDATES = [Candidate(SWITCH, 0.99), Candidate(SWITCH_ORO, 0.98)]

    def pick(self, reply):
        with patch("llm_json.chat_completion", side_effect=reply if isinstance(reply, list) else [reply]):
            return pick_product("16A 2Way switch", self.CANDIDATES)

    def test_valid_choice(self) -> None:
        chosen, reason = self.pick(json.dumps({"product_id": SWITCH_ORO.id, "reason": "Oro range"}))
        self.assertEqual(chosen.product, SWITCH_ORO)
        self.assertEqual(reason, "Oro range")

    def test_invented_id_is_rejected(self) -> None:
        chosen, reason = self.pick(json.dumps({"product_id": "made-up", "reason": "x"}))
        self.assertIsNone(chosen)
        self.assertIn("not in the candidate list", reason)

    def test_model_says_none(self) -> None:
        chosen, reason = self.pick(json.dumps({"product_id": None, "reason": "different product"}))
        self.assertIsNone(chosen)
        self.assertEqual(reason, "different product")

    def test_garbage_twice(self) -> None:
        chosen, reason = self.pick(["???", "still not json"])
        self.assertIsNone(chosen)
        self.assertIn("valid product choice", reason)

    def test_no_candidates_skips_the_llm(self) -> None:
        with patch("llm_json.chat_completion") as mock_chat:
            chosen, _ = pick_product("garden hose", [])
        self.assertIsNone(chosen)
        mock_chat.assert_not_called()


class AtomiseTests(unittest.TestCase):
    def test_groups_by_index_and_fills_gaps(self) -> None:
        reply = json.dumps([
            {"index": 0, "items": ["16A switch", "25A switch"]},
            # index 1 missing → kept as is
        ])
        with patch("llm_json.chat_completion", return_value=reply):
            out = atomise(["16A and 25A switches", "TV socket"])
        self.assertEqual(out, [["16A switch", "25A switch"], ["TV socket"]])

    def test_failure_keeps_everything(self) -> None:
        with patch("llm_json.chat_completion", return_value="nope"):
            self.assertEqual(atomise(["a b c", "d e f"]), [["a b c"], ["d e f"]])


if __name__ == "__main__":
    unittest.main()
