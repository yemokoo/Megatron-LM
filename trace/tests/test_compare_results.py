import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_results import summarize  # noqa: E402


class ResultComparisonTests(unittest.TestCase):
    def test_final_average_and_paper_afr(self):
        value = {
            "run_id": "test",
            "provenance": {"classification": "smoke"},
            "tasks": ["a", "b", "c"],
            "score_matrix": [
                [0.8, None, None],
                [0.7, 0.9, None],
                [0.6, 0.8, 1.0],
            ],
        }
        result = summarize(value)
        self.assertAlmostEqual(result["final_average"], 0.8)
        self.assertAlmostEqual(result["per_task_average_forgetting"]["a"], 0.15)
        self.assertAlmostEqual(result["per_task_average_forgetting"]["b"], 0.1)
        self.assertAlmostEqual(result["average_forgetting_rate"], 0.125)


if __name__ == "__main__":
    unittest.main()
