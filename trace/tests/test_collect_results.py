import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_results import (  # noqa: E402
    TASKS, collect_paper_baseline, collect_slora, collect_trace, metric_points, resolve)


class ResultCollectorTests(unittest.TestCase):
    def test_metric_scales(self):
        self.assertEqual(metric_points("C-STANCE", {"accuracy": 0.5}), 50.0)
        self.assertEqual(metric_points("MeetingBank", {"rouge-L": 0.25}), 25.0)
        self.assertEqual(metric_points("Py150", {"similarity": 71}), 71.0)
        self.assertEqual(metric_points("20Minuten", {"sari": 42.5}), 42.5)

    def test_collect_trace_triangle(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            evaluation = run_dir / "evaluation"
            evaluation.mkdir()
            for round_index in range(len(TASKS)):
                for task_index, task in enumerate(TASKS[: round_index + 1]):
                    if task == "MeetingBank":
                        metrics = {"rouge-L": 0.3}
                    elif task == "Py150":
                        metrics = {"similarity": 70}
                    elif task == "20Minuten":
                        metrics = {"sari": 40}
                    else:
                        metrics = {"accuracy": 0.6}
                    path = evaluation / f"results-{round_index}-{task_index}-{task}.json"
                    path.write_text(json.dumps({"eval": metrics}), encoding="utf-8")
            matrix = collect_trace(run_dir)
            self.assertEqual(matrix[0][0], 60.0)
            self.assertIsNone(matrix[0][1])
            self.assertEqual(matrix[-1][2], 30.0)
            self.assertEqual(matrix[-1][3], 70.0)
            self.assertEqual(matrix[-1][-1], 40.0)

    def test_collect_slora_log(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for round_index in range(1, len(TASKS) + 1):
                for task in TASKS[:round_index]:
                    task_dir = run_dir / "evaluation" / f"order{round_index}" / task
                    task_dir.mkdir(parents=True)
                    if task == "MeetingBank":
                        metrics = {"rouge-l": 0.2}
                    elif task == "Py150":
                        metrics = {"similarity": 72}
                    elif task == "20Minuten":
                        metrics = {"sari": 41}
                    else:
                        metrics = {"accuracy": 0.7}
                    (task_dir / "eval.log").write_text(
                        f"In {task}: {metrics}\n", encoding="utf-8"
                    )
            matrix = collect_slora(run_dir)
            self.assertEqual(matrix[0][0], 70.0)
            self.assertIsNone(matrix[0][1])
            self.assertEqual(matrix[-1][2], 20.0)
            self.assertEqual(matrix[-1][-1], 41.0)

    def test_collect_slora_partial_keeps_failed_cell_null(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            task_dir = run_dir / "evaluation" / "order1" / "C-STANCE"
            task_dir.mkdir(parents=True)
            (task_dir / "eval.log").write_text(
                "In C-STANCE: {'accuracy': 0.7}\n", encoding="utf-8")
            matrix = collect_slora(run_dir, sparse_15=True, allow_partial=True)
            self.assertEqual(matrix[0][0], 70.0)
            self.assertIsNone(matrix[1][1])
            with self.assertRaises(FileNotFoundError):
                collect_slora(run_dir, sparse_15=True, allow_partial=False)

    def test_collect_paper_partial_keeps_failed_cell_null(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            task_dir = run_dir / "evaluation" / "order1"
            task_dir.mkdir(parents=True)
            (task_dir / "results-C-STANCE.json").write_text(
                json.dumps({"eval": {"accuracy": 0.7}}), encoding="utf-8")
            matrix = collect_paper_baseline(
                run_dir, sparse_15=True, allow_partial=True)
            self.assertEqual(matrix[0][0], 70.0)
            self.assertIsNone(matrix[1][1])

    def test_resolve_trace_upstream_directories(self):
        root = Path("/tmp/repro")
        self.assertEqual(
            resolve(root, "ewc", "llama31")[1],
            root / "results" / "full_runs" / "llama31" / "ewc_upstream",
        )
        self.assertEqual(
            resolve(root, "lwf", "qwen25_7b")[1],
            root / "results" / "full_runs" / "qwen25_7b" / "lwf_upstream",
        )


if __name__ == "__main__":
    unittest.main()
