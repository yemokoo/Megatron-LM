#!/usr/bin/env python
"""CPU regression test for TRACE task-specific scoring boundaries."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_eval import normalize_predictions


def check(task, raw, expected):
    actual = normalize_predictions(task, [raw])[0]
    if actual != expected:
        raise AssertionError(
            f"{task}: expected {expected!r}, got {actual!r} from {raw!r}")


def main():
    check("FOMC", "Neutral\nThe stance is unchanged.", "C")
    check("C-STANCE", " B. support\nText:", "B")
    check("FOMC", "Answer: C. neutral\nText:", "C")
    check("NumGLUE-cm", "140\nStep 2 uses 70", "140")
    check("NumGLUE-ds", "1,200\nExplanation: 12 x 100", "1200")
    check("Py150", "import hashlib <EOL> import base64", "import hashlib ")
    check("MeetingBank", "short summary Meeting transcripts: leaked",
          "short summary ")
    check("20Minuten", "simple sentence Paragraph: leaked", "simple sentence ")
    check("ScienceQA", "B\nreasoning Question: leaked", "B\nreasoning ")
    print("TRACE_TASK_SCORING=PASS")


if __name__ == "__main__":
    main()
