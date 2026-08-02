import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from trace_record_adapter import prompt_answer, sft_messages  # noqa: E402


class TraceRecordAdapterTests(unittest.TestCase):
    def test_official_trace_prompt_answer_schema(self):
        record = {"prompt": "question", "answer": "answer"}
        self.assertEqual(prompt_answer(record), ("question", "answer"))
        self.assertEqual(sft_messages(record)["messages"][1]["content"], "question")

    def test_slora_conversation_schema(self):
        record = {
            "conversations": [
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "answer"},
            ]
        }
        self.assertEqual(prompt_answer(record), ("question", "answer"))

    def test_invalid_schema_fails_loudly(self):
        with self.assertRaises(ValueError):
            prompt_answer({"prompt": "missing answer"})
