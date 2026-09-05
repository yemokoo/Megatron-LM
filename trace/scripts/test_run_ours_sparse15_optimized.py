#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_ours_sparse15_optimized.py")
SPEC = importlib.util.spec_from_file_location("optimized_queue", SCRIPT)
optimized = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = optimized
SPEC.loader.exec_module(optimized)


class OptimizedQueueTest(unittest.TestCase):
    def test_python_launcher_path_is_not_symlink_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "base-python"
            target.touch()
            launcher = root / "venv-python"
            launcher.symlink_to(target)
            self.assertEqual(launcher.absolute(), launcher)
            self.assertNotEqual(launcher.resolve(), launcher)

    def test_sparse15_layout(self):
        cells = optimized.sparse15_cells()
        self.assertEqual(len(cells), 15)
        self.assertEqual(cells[0], optimized.Cell(1, "C-STANCE"))
        self.assertEqual(cells[6], optimized.Cell(7, "NumGLUE-ds"))
        self.assertEqual(cells[7], optimized.Cell(8, "C-STANCE"))
        self.assertEqual(cells[-1], optimized.Cell(8, "20Minuten"))

    def test_lower_triangle_layout(self):
        cells = optimized.lower_triangle_cells()
        self.assertEqual(len(cells), 36)
        self.assertEqual(cells[:3], [
            optimized.Cell(1, "C-STANCE"),
            optimized.Cell(2, "C-STANCE"),
            optimized.Cell(2, "FOMC"),
        ])
        self.assertEqual(cells[-8:], [
            optimized.Cell(8, task) for task in optimized.TASKS])

    def test_expected_indices_preserve_strided_shard_contract(self):
        shards = [optimized.expected_indices(10, i, 4) for i in range(4)]
        self.assertEqual(shards, [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]])
        self.assertEqual(sorted(index for shard in shards for index in shard),
                         list(range(10)))

    def test_progress_parses_tqdm_carriage_return(self):
        text = (
            "loading\n"
            "gen[ScienceQA,max_new=1024]:  12%|xx| 2/16 [00:01<00:08]\r"
            "gen[ScienceQA,max_new=1024]:  50%|xx| 8/16 [00:04<00:04]")
        self.assertEqual(optimized.parse_latest_progress(text), (50, 8, 16))
        self.assertIsNone(optimized.parse_latest_progress("model loading"))

    def test_result_validation_requires_exact_sample_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "part.json"
            path.write_text(json.dumps({
                "sample_indices": [1, 5, 9],
                "results": ["a", "b", "c"],
            }))
            self.assertTrue(optimized.result_is_complete(path, 10, 1, 4))
            self.assertFalse(optimized.result_is_complete(path, 10, 0, 4))

    def test_job_plan_shards_three_long_tasks_in_both_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            run = root / "run"
            for task in optimized.TASKS:
                task_dir = data / task
                task_dir.mkdir(parents=True)
                (task_dir / "test.json").write_text("[{}, {}, {}]")
            jobs, complete = optimized.make_jobs(
                run, data, optimized.DEFAULT_SHARDED_TASKS, 2)
            self.assertFalse(complete)
            # Fifteen cells, and six occurrences of sharded tasks each add one
            # extra GPU job when num_shards=2.
            self.assertEqual(len(jobs), 21)
            science = [job for job in jobs if job.cell.task == "ScienceQA"]
            self.assertEqual(len(science), 4)
            self.assertTrue(all(job.num_shards == 2 for job in science))

    def test_job_plan_accepts_lower_triangle_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            for task in optimized.TASKS:
                task_dir = data / task
                task_dir.mkdir(parents=True)
                (task_dir / "test.json").write_text("[{}]")
            cells = optimized.lower_triangle_cells()
            jobs, complete = optimized.make_jobs(
                root / "run", data, set(), 1, cells)
            self.assertFalse(complete)
            self.assertEqual(len(jobs), 36)

    def test_command_keeps_slora_1024_protocol(self):
        job = optimized.Job(optimized.Cell(5, "ScienceQA"), 2, 4)
        command = optimized.build_eval_command(
            job, Path("/run"), Path("/model"), Path("/data"),
            Path("/out"), 128, Path("/python"), True)
        joined = " ".join(command)
        self.assertIn("--max_ans_len 1024", joined)
        self.assertIn("--no-task_generation_limits", command)
        self.assertIn("--slora_conv_mode llama3", joined)
        self.assertIn("--num_sample_shards 4", joined)
        self.assertIn("--sample_shard_id 2", joined)
        self.assertIn("--result_suffix .shard2-of-4", joined)
        self.assertIn("--trace_generation_stops", command)

    def test_partial_lower_summary_preserves_missing_cell_as_null(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            cells = [
                optimized.Cell(1, "C-STANCE"),
                optimized.Cell(2, "FOMC"),
            ]
            result = run / "evaluation" / "order1" / "results-C-STANCE.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({"eval": {"accuracy": 0.75}}))
            output = optimized.write_partial_lower_triangle_summary(
                run, cells, "method", [{"cell": "order2.FOMC"}])
            payload = json.loads(output.read_text())
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["score_matrix"][0][0], 75.0)
            self.assertIsNone(payload["score_matrix"][1][1])
            self.assertEqual(payload["missing_cells"], ["order2.FOMC"])


if __name__ == "__main__":
    unittest.main()
