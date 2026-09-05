from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "launch_cka_gt_full_census_4gpu.sh"
RUNNER = HERE / "run_cka_gt_full_census_mha.sh"
PRETRAIN = HERE.parents[1] / "Megatron-LM" / "pretrain_gpt.py"
ARGUMENTS = (
    HERE.parents[1]
    / "Megatron-LM"
    / "megatron"
    / "training"
    / "arguments.py"
)
TRAINING = (
    HERE.parents[1]
    / "Megatron-LM"
    / "megatron"
    / "training"
    / "training.py"
)


def plan_launcher(**overrides: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "OUT_ROOT",
        "PILOT_ROOT",
        "ANALYSIS_CONFIG",
        "PILOT_MANIFEST",
        "CODE_PREFIX",
        "BENCHMARK_BATCHES",
        "START_SYSTEMD",
    ):
        environment.pop(name, None)
    environment.update({"PLAN_ONLY": "1", **overrides})
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class FullCensusLauncherContractTest(unittest.TestCase):
    def test_full_census_requests_frozen_reference_teacher(self) -> None:
        text = TRAINING.read_text(encoding="utf-8")
        self.assertIn("getattr(args, 'cka_gt_full_census_path', None)", text)

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_plan_uses_complete_train_and_four_independent_workers(self) -> None:
        result = plan_launcher()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("complete document-bounded Code train manifest", result.stdout)
        self.assertIn("no artificial pilot test holdout", result.stdout)
        self.assertIn("GPU=0 worker=0/4", result.stdout)
        self.assertIn("GPU=3 worker=3/4", result.stdout)
        self.assertIn("no raw hidden", result.stdout)
        self.assertIn("threshold-free B/T/M distributions", result.stdout)
        self.assertIn("5000000 occurrences", result.stdout)

    def test_mandatory_batch_sweep_and_optional_large_batches(self) -> None:
        result = plan_launcher()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("one_process_batches=64,96,128,192", result.stdout)
        self.assertIn(">3% gain from 96 to 128", result.stdout)
        self.assertNotIn("batch=256", result.stdout)

    def test_missing_mandatory_batch_is_rejected(self) -> None:
        result = plan_launcher(BENCHMARK_BATCHES="64,96")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("must be 64,96,128", result.stdout)

    def test_systemd_submission_is_explicit_not_automatic(self) -> None:
        result = plan_launcher()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("START_SYSTEMD=0", result.stdout)
        self.assertIn("[PLAN SYSTEMD] START_SYSTEMD=1", result.stdout)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("systemd-run --user --unit", text)

    def test_runner_plan_binds_world_size_one_and_full_census_args(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "PLAN_ONLY": "1",
                "GPU": "2",
                "WORKER_INDEX": "2",
                "WORKER_COUNT": "4",
                "WINDOW_BATCH_SIZE": "96",
            }
        )
        result = subprocess.run(
            ["bash", str(RUNNER)],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("worker: 2/4", result.stdout)
        self.assertIn("batch: 96", result.stdout)
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--nproc_per_node 1", text)
        self.assertIn("--cka-gt-full-census-worker-count", text)
        self.assertIn("--cka-gt-full-census-benchmark-batch-sizes", text)
        self.assertNotIn("cka-gt-pilot-membership-stats", text)

    def test_model_path_is_read_only_residual_l2_to_l9(self) -> None:
        text = PRETRAIN.read_text(encoding="utf-8")
        start = text.index("def _run_cka_gt_full_census")
        stop = text.index("def _run_cka_gt_pilot", start)
        function = text[start:stop]
        self.assertIn("torch.inference_mode()", function)
        self.assertIn('layer_spec = "2,3,4,5,6,7,8,9"', function)
        self.assertIn("process_batch(", function)
        self.assertIn("forward_seconds=", function)
        self.assertIn("source_dataset_identity(dataset_prefix)", function)
        self.assertIn("checkpoint_identity(reference_load)", function)
        self.assertIn("warmup_batches=3", function)
        self.assertGreaterEqual(function.count("torch.cuda.synchronize()"), 3)
        self.assertNotIn("_capture_moe_router_inputs", function)
        self.assertNotIn("torch.save", function)

    def test_arguments_include_batch_bound_restartable_census(self) -> None:
        text = ARGUMENTS.read_text(encoding="utf-8")
        for flag in (
            "--cka-gt-full-census-path",
            "--cka-gt-full-census-config",
            "--cka-gt-full-census-manifest",
            "--cka-gt-full-census-worker-index",
            "--cka-gt-full-census-worker-count",
            "--cka-gt-full-census-batch-size",
            "--cka-gt-full-census-checkpoint-every-batches",
            "--cka-gt-full-census-histogram-bins",
            "--cka-gt-full-census-reservoir-size",
            "--cka-gt-full-census-benchmark-batch-sizes",
        ):
            self.assertIn(flag, text)
        self.assertIn("default=4096", text)

    def test_launcher_never_terminates_unrelated_jobs(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("pkill", text)
        self.assertNotIn("kill -9", text)
        self.assertIn('kill "$GPU_MONITOR_PID"', text)


if __name__ == "__main__":
    unittest.main()
