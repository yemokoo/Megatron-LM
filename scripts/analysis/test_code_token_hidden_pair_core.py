#!/usr/bin/env python3
"""Unit tests for paired Code-token hidden metric extraction storage.

The tests intentionally use only tiny CPU tensors.  In particular, the resume
test interrupts after an update that crosses a shard boundary: everything
needed to reproduce committed output must correspond to the committed sample
prefix, rather than to any still-buffered suffix.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from code_token_hidden_pair_core import (  # noqa: E402
    METRIC_NAMES,
    TokenMetricShardWriter,
    compute_token_pair_metrics,
    derive_requested_metrics,
)


LAYERS = (1, 2)
HIDDEN_SIZE = 4
SEQUENCE_LENGTH = 3
PARTITION_START = 10
PARTITION_SAMPLES = 5
SHARD_SAMPLES = 2
RESERVOIR_SIZE = 5


def _assert_close(
    testcase: unittest.TestCase,
    actual: torch.Tensor | np.ndarray,
    expected: torch.Tensor | np.ndarray | float,
    *,
    atol: float = 1e-6,
) -> None:
    if isinstance(actual, torch.Tensor):
        expected_tensor = torch.as_tensor(expected, dtype=actual.dtype)
        testcase.assertTrue(
            torch.allclose(actual.cpu(), expected_tensor.cpu(), atol=atol, rtol=atol),
            msg=f"actual={actual} expected={expected_tensor}",
        )
    else:
        expected_array = np.asarray(expected, dtype=actual.dtype)
        testcase.assertTrue(
            np.allclose(actual, expected_array, atol=atol, rtol=atol),
            msg=f"actual={actual} expected={expected_array}",
        )


def _writer(output_dir: Path) -> TokenMetricShardWriter:
    return TokenMetricShardWriter(
        str(output_dir),
        layers=LAYERS,
        hidden_size=HIDDEN_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        partition_start_sample=PARTITION_START,
        partition_samples=PARTITION_SAMPLES,
        total_samples=100,
        shard_samples=SHARD_SAMPLES,
        reservoir_size=RESERVOIR_SIZE,
        seed=1234,
        metadata={
            "reference_checkpoint": "/reference",
            "current_checkpoint": "/current",
            "dataset_path": "/dataset",
            "worker_index": 0,
            "worker_count": 1,
        },
    )


def _batch(start_sample: int, samples: int) -> dict:
    """Create deterministic aligned inputs and flattened per-layer hidden."""
    sample_ids = torch.arange(start_sample, start_sample + samples, dtype=torch.int64)
    positions = torch.arange(SEQUENCE_LENGTH, dtype=torch.int64)
    input_ids = sample_ids[:, None] * 10 + positions[None, :]
    labels = input_ids + 1
    valid = torch.ones((samples, SEQUENCE_LENGTH), dtype=torch.bool)
    # Exercise invalid-token zero fill in every batch while retaining multiple
    # valid occurrences for deterministic reservoir selection.
    valid[:, -1] = (sample_ids % 2 == 0)

    flat_sample = sample_ids[:, None].expand(-1, SEQUENCE_LENGTH).reshape(-1).float()
    flat_position = positions[None, :].expand(samples, -1).reshape(-1).float()
    reference = {}
    current = {}
    for layer in LAYERS:
        ref = torch.stack(
            (
                flat_sample + layer,
                flat_position + 0.25 * layer,
                flat_sample - flat_position + 0.5,
                torch.ones_like(flat_sample) * (layer + 0.75),
            ),
            dim=-1,
        )
        # Nontrivial, deterministic change: neither pure scaling nor a constant
        # shift, so all canonical metrics contain useful values.
        cur = ref * (1.0 + 0.05 * layer)
        cur[:, 1] += 0.1 * flat_position
        reference[layer] = ref
        current[layer] = cur
    return {
        "sample_ids": sample_ids,
        "input_token_ids": input_ids,
        "label_token_ids": labels,
        "valid_mask": valid,
        "reference_by_layer": reference,
        "current_by_layer": current,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


class PairMetricTest(unittest.TestCase):
    def test_identical_and_scale_metrics_are_float32(self) -> None:
        reference = torch.tensor(
            [[1.0, 2.0, -1.0], [-2.0, 0.5, 4.0]], dtype=torch.float16
        )

        identical = compute_token_pair_metrics(reference, reference.clone())
        self.assertEqual(set(identical), set(METRIC_NAMES))
        for name, values in identical.items():
            self.assertEqual(values.dtype, torch.float32, msg=name)
        _assert_close(self, identical["cosine"], torch.ones(2))
        _assert_close(self, identical["relative_l2"], torch.zeros(2))
        _assert_close(self, identical["symmetric_relative_l2"], torch.zeros(2))
        _assert_close(self, identical["log_norm_ratio"], torch.zeros(2))
        _assert_close(self, identical["delta_mse"], torch.zeros(2))
        _assert_close(self, identical["feature_centered_cosine"], torch.ones(2))

        scaled = compute_token_pair_metrics(reference, 2.0 * reference)
        _assert_close(self, scaled["cosine"], torch.ones(2))
        _assert_close(self, scaled["relative_l2"], torch.ones(2))
        _assert_close(
            self, scaled["symmetric_relative_l2"], torch.full((2,), 2.0 / 3.0)
        )
        _assert_close(self, scaled["log_norm_ratio"], torch.full((2,), math.log(2.0)))
        _assert_close(self, scaled["delta_mse"], reference.float().square().mean(dim=-1))
        _assert_close(self, scaled["feature_centered_cosine"], torch.ones(2))

    def test_orthogonal_and_zero_inputs_are_finite(self) -> None:
        reference = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        current = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        metrics = compute_token_pair_metrics(reference, current)
        for name, values in metrics.items():
            self.assertTrue(torch.isfinite(values).all(), msg=name)
            self.assertEqual(values.dtype, torch.float32, msg=name)

        self.assertAlmostEqual(float(metrics["cosine"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["relative_l2"][0]), math.sqrt(2.0), places=6)
        self.assertAlmostEqual(
            float(metrics["symmetric_relative_l2"][0]), math.sqrt(2.0), places=6
        )
        self.assertAlmostEqual(float(metrics["log_norm_ratio"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["delta_mse"][0]), 1.0, places=6)
        self.assertAlmostEqual(
            float(metrics["feature_centered_cosine"][0]), -1.0, places=6
        )
        self.assertAlmostEqual(float(metrics["relative_l2"][1]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["log_norm_ratio"][1]), 0.0, places=6)

    def test_derived_metric_identities(self) -> None:
        reference = torch.tensor(
            [[1.0, -2.0, 0.5, 3.0], [2.0, 1.0, -1.0, 0.25]], dtype=torch.float32
        )
        current = torch.tensor(
            [[0.5, -1.0, 2.0, 4.0], [-1.0, 2.0, 0.5, 1.0]], dtype=torch.float32
        )
        canonical_torch = compute_token_pair_metrics(reference, current)
        canonical = {name: value.numpy() for name, value in canonical_torch.items()}
        derived = derive_requested_metrics(canonical)

        xnorm = torch.linalg.vector_norm(reference, dim=-1)
        ynorm = torch.linalg.vector_norm(current, dim=-1)
        delta_norm = torch.linalg.vector_norm(current - reference, dim=-1)
        cosine = torch.nn.functional.cosine_similarity(reference, current, dim=-1)
        centered_cosine = torch.nn.functional.cosine_similarity(
            reference - reference.mean(dim=-1, keepdim=True),
            current - current.mean(dim=-1, keepdim=True),
            dim=-1,
        )
        _assert_close(self, derived["cosine"], cosine.numpy())
        _assert_close(self, derived["norm_ratio"], (ynorm / xnorm).numpy())
        _assert_close(
            self,
            derived["symmetric_relative_l2"],
            (2.0 * delta_norm / (xnorm + ynorm)).numpy(),
        )
        _assert_close(self, derived["feature_centered_cosine"], centered_cosine.numpy())
        _assert_close(self, derived["cosine_distance"], 1.0 - cosine.numpy())

    def test_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            compute_token_pair_metrics(torch.zeros(2, 3), torch.zeros(3, 2))
        with self.assertRaises(ValueError):
            compute_token_pair_metrics(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))


class ShardWriterTest(unittest.TestCase):
    def _finish_continuously(self, root: Path) -> TokenMetricShardWriter:
        writer = _writer(root)
        writer.update(**_batch(PARTITION_START, PARTITION_SAMPLES))
        result = writer.finalize()
        self.assertTrue(result["completed"])
        return writer

    def test_roundtrip_partial_final_shard_and_invalid_zero_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            writer = self._finish_continuously(root)
            self.assertEqual(writer.completed_samples, PARTITION_SAMPLES)
            self.assertEqual(writer.next_shard, 3)

            shard_paths = sorted((root / "token_metrics").glob("shard_*.npz"))
            self.assertEqual([path.name for path in shard_paths], [
                "shard_000000.npz",
                "shard_000001.npz",
                "shard_000002.npz",
            ])
            first = _load_npz(shard_paths[0])
            final = _load_npz(shard_paths[-1])
            self.assertEqual(first["sample_ids"].tolist(), [10, 11])
            self.assertEqual(final["sample_ids"].tolist(), [14])
            self.assertEqual(first["input_token_ids"].shape, (2, SEQUENCE_LENGTH))
            self.assertEqual(final["input_token_ids"].shape, (1, SEQUENCE_LENGTH))
            np.testing.assert_array_equal(first["layer_numbers"], np.asarray(LAYERS))

            for name in METRIC_NAMES:
                self.assertEqual(first[name].shape, (2, SEQUENCE_LENGTH, len(LAYERS)))
                self.assertEqual(first[name].dtype, np.float32)
                self.assertTrue(np.isfinite(first[name]).all(), msg=name)
                invalid = ~first["valid_mask"].astype(bool)
                self.assertTrue(np.equal(first[name][invalid], 0.0).all(), msg=name)

            reservoir = _load_npz(root / "reservoir" / "hidden_reservoir.npz")
            self.assertLessEqual(reservoir["sample_ids"].size, RESERVOIR_SIZE)
            self.assertEqual(
                reservoir["reference"].shape,
                (reservoir["sample_ids"].size, len(LAYERS), HIDDEN_SIZE),
            )
            self.assertEqual(reservoir["reference"].dtype, np.float16)
            # reference/current/delta are each quantized to fp16 separately;
            # reconstructing the delta from the two quantized endpoints is
            # therefore only approximately equal to the stored fp32-then-fp16
            # delta.
            np.testing.assert_allclose(
                reservoir["delta"],
                (reservoir["current"].astype(np.float32)
                 - reservoir["reference"].astype(np.float32)).astype(np.float16),
                atol=1e-2,
                rtol=1e-2,
            )

    def test_resume_and_reservoir_are_batching_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            continuous_root = base / "continuous"
            resumed_root = base / "resumed"
            self._finish_continuously(continuous_root)

            # Three samples cross the two-sample shard boundary.  Only the
            # first two are committed; sample 12 must be regenerated after a
            # restart, without being duplicated in reservoir state.
            interrupted = _writer(resumed_root)
            interrupted.update(**_batch(PARTITION_START, 3))
            self.assertEqual(interrupted.completed_samples, 2)
            self.assertEqual(interrupted.buffered_samples, 1)

            resumed = _writer(resumed_root)
            self.assertEqual(resumed.completed_samples, 2)
            self.assertEqual(resumed.next_global_sample, 12)
            resumed.update(**_batch(12, 3))
            resumed.finalize()

            continuous_shards = sorted(
                (continuous_root / "token_metrics").glob("shard_*.npz")
            )
            resumed_shards = sorted((resumed_root / "token_metrics").glob("shard_*.npz"))
            self.assertEqual(len(continuous_shards), len(resumed_shards))
            for expected_path, actual_path in zip(continuous_shards, resumed_shards):
                expected = _load_npz(expected_path)
                actual = _load_npz(actual_path)
                self.assertEqual(set(expected), set(actual))
                for name in expected:
                    np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)

            continuous_reservoir = _load_npz(
                continuous_root / "reservoir" / "hidden_reservoir.npz"
            )
            resumed_reservoir = _load_npz(
                resumed_root / "reservoir" / "hidden_reservoir.npz"
            )
            self.assertEqual(set(continuous_reservoir), set(resumed_reservoir))
            for name in continuous_reservoir:
                np.testing.assert_array_equal(
                    resumed_reservoir[name], continuous_reservoir[name], err_msg=name
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
