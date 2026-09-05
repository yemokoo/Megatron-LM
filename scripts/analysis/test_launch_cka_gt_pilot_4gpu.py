from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).with_name("launch_cka_gt_pilot_4gpu.sh")
BATCH_VARIABLES = (
    "WINDOW_BATCH_SIZE",
    "PASS1_BATCH_SIZE",
    "SMOKE_WINDOW_BATCH_SIZE",
    "SMOKE_PASS1_BATCH_SIZE",
)


def _plan(*, start: str = "prepared_validate", **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (*BATCH_VARIABLES, "OUT_ROOT", "SMOKE_ROOT"):
        environment.pop(name, None)
    environment.update(
        {
            "PLAN_ONLY": "1",
            "START_STAGE": start,
            "STOP_AFTER_STAGE": "analyze",
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class LauncherPlanContractTest(unittest.TestCase):
    def test_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_default_smoke_and_production_batches_are_isolated(self) -> None:
        result = _plan()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("production batches: pass1=32 pass2=32", result.stdout)
        self.assertIn("checkpoint-smoke batches: pass1=4 pass2=4", result.stdout)
        self.assertIn("checkpoint-smoke pass1:", result.stdout)
        self.assertIn("WORKER_COUNT=1 WINDOW_BATCH_SIZE=4", result.stdout)
        self.assertIn("pass1: MODE=pass1", result.stdout)
        self.assertIn("GPU=0 WINDOW_BATCH_SIZE=32", result.stdout)
        self.assertIn("pass2-code worker=3 GPU=3 WINDOW_BATCH_SIZE=32", result.stdout)

    def test_batch_overrides_are_visible_in_plan(self) -> None:
        result = _plan(
            WINDOW_BATCH_SIZE="19",
            PASS1_BATCH_SIZE="17",
            SMOKE_WINDOW_BATCH_SIZE="7",
            SMOKE_PASS1_BATCH_SIZE="5",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("production batches: pass1=17 pass2=19", result.stdout)
        self.assertIn("checkpoint-smoke batches: pass1=5 pass2=7", result.stdout)

    def test_nonpositive_batch_values_are_rejected(self) -> None:
        for name in BATCH_VARIABLES:
            with self.subTest(name=name):
                result = _plan(**{name: "0"})
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(f"{name} must be positive", result.stdout)

    def test_full_chain_plans_deep_stage8_and_shallow_final(self) -> None:
        result = _plan()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("full pre-analysis validate:", result.stdout)
        self.assertIn("--allow-incomplete --deep", result.stdout)
        final_line = next(
            line for line in result.stdout.splitlines()
            if "final after-analysis validate" in line
        )
        self.assertIn("stage8 current => shallow", final_line)
        self.assertNotIn("--deep", final_line)

    def test_analyze_only_plan_keeps_deep_safety_fallback(self) -> None:
        result = _plan(start="analyze")
        self.assertEqual(result.returncode, 0, result.stdout)
        final_line = next(
            line for line in result.stdout.splitlines()
            if "final after-analysis validate" in line
        )
        self.assertIn("analyze-only safety fallback", final_line)
        self.assertIn("--deep", final_line)


if __name__ == "__main__":
    unittest.main()
