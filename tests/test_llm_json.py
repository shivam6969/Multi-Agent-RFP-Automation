from __future__ import annotations

import unittest
from typing import List
from unittest.mock import patch

from pydantic import BaseModel

from llm_json import LLMOutputError, chat_json, extract_json


class _Item(BaseModel):
    name: str
    qty: int


class ExtractJsonTests(unittest.TestCase):
    def test_plain_fenced_and_chatty_replies(self) -> None:
        self.assertEqual(extract_json('[1, 2]'), [1, 2])
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here it is:\n[{"a": 1}]\nHope this helps.'), [{"a": 1}])

    def test_no_json(self) -> None:
        with self.assertRaises(ValueError):
            extract_json("I cannot help with that.")


class ChatJsonTests(unittest.TestCase):
    def call(self):
        return chat_json(system="s", user="u", schema=List[_Item])

    @patch("llm_json.chat_completion", return_value='[{"name": "switch", "qty": 5}]')
    def test_valid_reply(self, mock_chat) -> None:
        self.assertEqual(self.call(), [_Item(name="switch", qty=5)])
        self.assertEqual(mock_chat.call_count, 1)

    @patch("llm_json.chat_completion",
           side_effect=['[{"name": "switch"}]', '[{"name": "switch", "qty": 5}]'])
    def test_retries_once_with_the_error(self, mock_chat) -> None:
        self.assertEqual(self.call()[0].qty, 5)
        retry_prompt = mock_chat.call_args_list[1].kwargs["user"]
        self.assertIn("could not be used", retry_prompt)
        self.assertIn("qty", retry_prompt)

    @patch("llm_json.chat_completion", return_value="not json at all")
    def test_gives_up_after_retry(self, mock_chat) -> None:
        with self.assertRaises(LLMOutputError):
            self.call()
        self.assertEqual(mock_chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
