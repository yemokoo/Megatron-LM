#!/usr/bin/env python3
"""CPU-only numerical tests for the CKA GT pilot core."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from cka_gt_pilot_core import (  # noqa: E402
    INVALID_NONE,
    INVALID_X_RSM_NORM,
    INVALID_Y_RSM_NORM,
    T_INVALID_CHUNK,
    centered_linear_cka_metrics,
    condition_specific_consensus,
    consensus_strategy_diagnostic,
    permutation_null_metrics,
    random_pair_indices,
    random_pair_null_metrics,
    same_layer_consensus,
    uncentered_token_metrics,
)


class CKAMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)

    def test_identical_cka_and_exact_additive_contribution(self) -> None:
        x = torch.randn(64, 24, dtype=torch.float32)
        result = centered_linear_cka_metrics(x, x.clone())

        self.assertEqual(int(result["invalid_reason"].item()), INVALID_NONE)
        self.assertAlmostEqual(float(result["cka"].item()), 1.0, places=5)
        self.assertTrue(
            torch.allclose(
                result["c_i"].sum(), result["cka"], atol=1.0e-4, rtol=1.0e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                result["s_i"].sum(),
                torch.tensor(float(x.shape[0])),
                atol=1.0e-4,
                rtol=1.0e-4,
            )
        )
        finite_r = result["r_i"][torch.isfinite(result["r_i"])]
        self.assertGreater(finite_r.numel(), 0)
        self.assertTrue(torch.allclose(finite_r, torch.ones_like(finite_r), atol=2.0e-5))

    def test_random_orthogonal_transform_has_cka_one(self) -> None:
        x = torch.randn(96, 32, dtype=torch.float32)
        q, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float32))
        y = x @ q
        result = centered_linear_cka_metrics(x, y)
        self.assertAlmostEqual(float(result["cka"].item()), 1.0, places=5)
        self.assertTrue(
            torch.allclose(result["c_i"].sum(), result["cka"], atol=1.0e-4, rtol=1.0e-4)
        )

    def test_isotropic_scaling_cka_one_but_magnitude_detects_change(self) -> None:
        x = torch.randn(48, 17, dtype=torch.float32)
        y = 2.0 * x
        cka = centered_linear_cka_metrics(x, y)
        magnitude = uncentered_token_metrics(x, y)

        self.assertAlmostEqual(float(cka["cka"].item()), 1.0, places=5)
        self.assertTrue(
            torch.allclose(magnitude["rel_l2"], torch.ones(48), atol=2.0e-6)
        )
        self.assertTrue(
            torch.allclose(
                magnitude["log_r"],
                torch.full((48,), math.log(2.0)),
                atol=2.0e-6,
            )
        )
        self.assertTrue(torch.allclose(magnitude["cosine"], torch.ones(48), atol=2.0e-6))

    def test_permutation_null_is_lower_than_aligned_cka(self) -> None:
        x = torch.randn(128, 24, dtype=torch.float32)
        aligned = centered_linear_cka_metrics(x, x)["cka"]
        null = permutation_null_metrics(x, x, seed=9)["cka"]
        self.assertGreater(float(aligned.item()), 0.9999)
        self.assertLess(float(null.item()), float(aligned.item()) - 0.25)

    def test_random_pair_null_uses_no_fixed_points(self) -> None:
        windows = torch.randn(7, 32, 8, dtype=torch.float32)
        after = windows + 0.01 * torch.randn_like(windows)
        pair = random_pair_indices(7, seed=31)
        self.assertTrue(torch.equal(torch.sort(pair).values, torch.arange(7)))
        self.assertFalse(torch.any(pair == torch.arange(7)))
        result = random_pair_null_metrics(windows, after, pair_indices=pair)
        self.assertTrue(torch.equal(result["pair_indices"].cpu(), pair))
        self.assertEqual(result["cka"].shape, (7,))
        self.assertTrue(torch.isfinite(result["cka"]).all())

    def test_constant_representation_is_invalid_not_zero_score(self) -> None:
        x = torch.ones(32, 12, dtype=torch.float32)
        y = torch.randn(32, 12, dtype=torch.float32)
        x_invalid = centered_linear_cka_metrics(x, y)
        self.assertTrue(int(x_invalid["invalid_reason"].item()) & INVALID_X_RSM_NORM)
        self.assertTrue(torch.isnan(x_invalid["cka"]))
        self.assertTrue(torch.isnan(x_invalid["c_i"]).all())
        self.assertTrue(int(x_invalid["t_invalid_reason"].item()) & T_INVALID_CHUNK)

        both_invalid = centered_linear_cka_metrics(x, torch.full_like(x, 3.0))
        reason = int(both_invalid["invalid_reason"].item())
        self.assertTrue(reason & INVALID_X_RSM_NORM)
        self.assertTrue(reason & INVALID_Y_RSM_NORM)
        self.assertTrue(torch.isnan(both_invalid["cka"]))

    def test_batched_fp16_input_still_returns_fp32_metrics(self) -> None:
        x = torch.randn(3, 20, 11).half()
        y = (x.float() + 0.05 * torch.randn(3, 20, 11)).half()
        result = centered_linear_cka_metrics(x, y)
        self.assertEqual(result["cka"].shape, (3,))
        self.assertEqual(result["c_i"].shape, (3, 20))
        for name in (
            "cka",
            "c_i",
            "c_i_off",
            "cka_off",
            "s_i",
            "diag_ratio",
            "r_i",
            "centered_var_x",
            "centered_var_y",
        ):
            self.assertEqual(result[name].dtype, torch.float32, msg=name)


class ConsensusTest(unittest.TestCase):
    def _conditions(self) -> dict[str, torch.Tensor]:
        # Five token examples, eight layer values each.
        b = torch.ones(5, 8)
        t = torch.ones(5, 8)
        l2 = torch.zeros(5, 8)
        log_r = torch.zeros(5, 8)

        # Token 1: each condition independently passes 7/8, but the two failed
        # conditions are on different layers, leaving only 6 layers that pass
        # all conditions simultaneously.
        b[1, 0] = 0.0
        t[1, 1] = -2.0

        # Token 2: exactly six measured layers and all six pass (the explicit
        # n_valid=6 -> 6/6 pilot rule).
        for value in (b, t, l2, log_r):
            value[2, 6:] = float("nan")

        # Token 3: only five measured layers, hence ineligible.
        for value in (b, t, l2, log_r):
            value[3, 5:] = float("nan")

        # Token 4: two negative T shares must fail even though a deliberately
        # permissive numeric threshold would otherwise accept them.
        t[4, :2] = -0.5
        return {"B": b, "T": t, "L2": l2, "R": log_r}

    def test_condition_specific_and_same_layer_semantics(self) -> None:
        conditions = self._conditions()
        thresholds = {"B": 0.5, "T": -1.0, "L2": 0.2, "R": 0.2}
        comparisons = {"B": "ge", "T": "ge", "L2": "le", "R": "le"}

        independent = condition_specific_consensus(
            conditions,
            thresholds,
            comparisons,
            reject_negative=("T",),
        )
        same_layer = same_layer_consensus(
            conditions,
            thresholds,
            comparisons,
            reject_negative=("T",),
        )

        self.assertEqual(independent["passed"].tolist(), [True, True, True, False, False])
        self.assertEqual(independent["eligible"].tolist(), [True, True, True, False, True])
        self.assertEqual(same_layer["passed"].tolist(), [True, False, True, False, False])
        self.assertEqual(int(independent["by_condition"]["B"]["n_valid"][2]), 6)
        self.assertEqual(int(independent["by_condition"]["B"]["required_pass_count"][2]), 6)
        self.assertEqual(int(independent["by_condition"]["T"]["pass_count"][4]), 6)

        diagnostic = consensus_strategy_diagnostic(
            independent["passed"], same_layer["passed"]
        )
        self.assertEqual(diagnostic["condition_specific_count"], 3)
        self.assertEqual(diagnostic["same_layer_count"], 2)
        self.assertEqual(diagnostic["intersection_count"], 2)
        self.assertAlmostEqual(diagnostic["jaccard"], 2.0 / 3.0)

    def test_consensus_validates_condition_keys(self) -> None:
        with self.assertRaises(ValueError):
            condition_specific_consensus(
                {"B": torch.ones(2, 8)},
                {"wrong": 0.0},
                {"B": "ge"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
