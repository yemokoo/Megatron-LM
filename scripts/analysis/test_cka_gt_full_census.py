#!/usr/bin/env python3
"""CPU-only tests for the threshold-free full-train CKA census."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from cka_gt_full_census import (  # noqa: E402
    BTMBatch,
    CHUNK_LAYOUT,
    FullCensusWorker,
    HistogramAccumulator,
    HistogramSpec,
    build_full_train_manifest,
    compute_btm_batch,
    manifest_batch_plan,
    partition_bounds,
    reservoir_candidates,
    token_priorities,
    validate_full_train_manifest,
)
from cka_gt_pilot_core import (  # noqa: E402
    centered_linear_cka_b_t_metrics,
    centered_linear_cka_metrics,
)
from cka_gt_pilot_windows import WINDOW_DTYPE  # noqa: E402


class FakeDataset:
    def __init__(self, lengths: list[int]):
        self.values = [np.arange(length, dtype=np.int32) for length in lengths]
        self.sequence_lengths = np.asarray(lengths, dtype=np.int32)
        self.document_indices = np.arange(len(lengths) + 1, dtype=np.int64)

    def get(self, index: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        value = self.values[int(index)]
        if length is None:
            length = value.size - offset
        return value[offset : offset + length]


def rows_for_lengths(lengths: list[int]) -> np.ndarray:
    rows = np.empty(len(lengths), dtype=WINDOW_DTYPE)
    for index, length in enumerate(lengths):
        eligible = 512 if length == 512 else 384 if length >= 384 else 256
        rows[index] = (
            index, index, index, 0, length, eligible, length - eligible,
            int(length < 512)
        )
    return rows


def hidden_pair(batch: int, length: int, hidden: int = 12):
    generator = torch.Generator().manual_seed(77 + length)
    before, after = {}, {}
    for layer in range(2, 10):
        value = torch.randn(batch, length, hidden, generator=generator)
        before[layer] = value
        after[layer] = value.clone()
    return before, after


class FocusedCoreTest(unittest.TestCase):
    def test_focused_b_t_kernel_matches_full_pilot_kernel(self) -> None:
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(3, 16, 7, generator=generator)
        y = x + 0.1 * torch.randn(3, 16, 7, generator=generator)
        focused = centered_linear_cka_b_t_metrics(x, y)
        full = centered_linear_cka_metrics(x, y)
        torch.testing.assert_close(focused["cka"], full["cka"], rtol=0, atol=1e-6)
        torch.testing.assert_close(focused["s_i"], full["s_i"], rtol=0, atol=1e-5)
        self.assertTrue(torch.equal(focused["chunk_valid"], full["chunk_valid"]))
        self.assertTrue(torch.equal(focused["t_valid"], full["t_valid"]))


class MetricAndHistogramTest(unittest.TestCase):
    def test_identity_pair_has_exact_layout_and_zero_magnitude_change(self) -> None:
        rows = rows_for_lengths([512, 512])
        before, after = hidden_pair(2, 512)
        metrics = compute_btm_batch(
            before_by_layer=before, after_by_layer=after, manifest_rows=rows
        )
        self.assertEqual(tuple(metrics.raw_b.shape), (2, len(CHUNK_LAYOUT), 8))
        torch.testing.assert_close(metrics.raw_b, torch.ones_like(metrics.raw_b), atol=2e-6, rtol=0)
        torch.testing.assert_close(metrics.rel_l2, torch.zeros_like(metrics.rel_l2))
        torch.testing.assert_close(metrics.abs_log_r, torch.zeros_like(metrics.abs_log_r))
        self.assertTrue(torch.isfinite(metrics.t_min[128]).all())
        self.assertTrue(torch.isfinite(metrics.t_min[256]).all())

    def test_metric_baseline_restores_tf32_setting(self) -> None:
        rows = rows_for_lengths([256])
        before, after = hidden_pair(1, 256)
        original = bool(torch.backends.cuda.matmul.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            compute_btm_batch(
                before_by_layer=before, after_by_layer=after, manifest_rows=rows
            )
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original

    def test_tail_suffix_is_ineligible_and_missing_chunk_slots_are_nan(self) -> None:
        rows = rows_for_lengths([300])
        before, after = hidden_pair(1, 300)
        metrics = compute_btm_batch(
            before_by_layer=before, after_by_layer=after, manifest_rows=rows
        )
        self.assertTrue(metrics.eligible[:, :256].all())
        self.assertFalse(metrics.eligible[:, 256:].any())
        self.assertTrue(torch.isnan(metrics.rel_l2[:, 256:]).all())
        # 300-token tails have 3x128 and 1x256 chunks, leaving six fixed slots absent.
        self.assertEqual(int(torch.isfinite(metrics.raw_b[0, :, 0]).sum()), 4)

    def test_histogram_accounts_for_under_over_and_nonfinite(self) -> None:
        accumulator = HistogramAccumulator({"x": HistogramSpec(0.0, 1.0, 4096)})
        values = torch.tensor([[[-0.1, 0.0, 0.5, 1.0, 1.1, float("nan"), 0.2, 0.3]]])
        accumulator.observe("x", values, torch.ones((1, 1), dtype=torch.bool))
        self.assertEqual(accumulator.underflow["x"].tolist(), [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(accumulator.overflow["x"].tolist(), [0, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(accumulator.nonfinite["x"].tolist(), [0, 0, 0, 0, 0, 1, 0, 0])
        self.assertEqual(int(accumulator.counts["x"].sum()), 5)


class ManifestAndReservoirTest(unittest.TestCase):
    def test_full_manifest_is_exhaustive_and_partitions_without_overlap(self) -> None:
        dataset = FakeDataset([255, 256, 512, 1024])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fake_text_document"
            prefix.with_suffix(".idx").write_bytes(b"idx")
            prefix.with_suffix(".bin").write_bytes(b"bin")
            manifest = build_full_train_manifest(
                dataset, dataset_prefix=str(prefix), output_dir=root / "manifest"
            )
            validated = validate_full_train_manifest(
                root / "manifest" / "manifest.json",
                dataset=dataset,
                dataset_prefix=str(prefix),
            )
            self.assertEqual(manifest["statistics"]["window_count"], 4)
            self.assertEqual(validated["validated_counts"]["window_count"], 4)
            bounds = [partition_bounds(4, index, 3) for index in range(3)]
            self.assertEqual(bounds, [(0, 2), (2, 3), (3, 4)])

    def test_priorities_and_reservoir_rows_are_deterministic(self) -> None:
        rows = rows_for_lengths([512])
        before, after = hidden_pair(1, 512)
        metrics = compute_btm_batch(
            before_by_layer=before, after_by_layer=after, manifest_rows=rows
        )
        first_priority = token_priorities(rows, 512, 1234)
        second_priority = token_priorities(rows, 512, 1234)
        np.testing.assert_array_equal(first_priority, second_priority)
        tokens = torch.arange(512).reshape(1, 512)
        first = reservoir_candidates(
            rows, tokens, metrics, priority_cutoff=(1 << 63), seed=1234
        )
        second = reservoir_candidates(
            rows, tokens, metrics, priority_cutoff=(1 << 63), seed=1234
        )
        np.testing.assert_array_equal(first, second)
        self.assertGreater(first.size, 200)
        self.assertLess(first.size, 320)
        self.assertEqual(first["b_min"].shape[1:], (2, 8))


class WorkerResumeTest(unittest.TestCase):
    def test_precomputed_subbatches_equal_direct_logical_batch(self) -> None:
        dataset = FakeDataset([512, 512])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fake_text_document"
            prefix.with_suffix(".idx").write_bytes(b"idx")
            prefix.with_suffix(".bin").write_bytes(b"bin")
            manifest_root = root / "manifest"
            build_full_train_manifest(
                dataset, dataset_prefix=str(prefix), output_dir=manifest_root)
            config = root / "config.json"
            config.write_text("{}")
            direct = FullCensusWorker(
                output_dir=root / "direct", worker_index=0, worker_count=1,
                analysis_config_path=config,
                manifest_path=manifest_root / "manifest.json", batch_size=2,
                checkpoint_every_batches=1, reservoir_size=0)
            split = FullCensusWorker(
                output_dir=root / "split", worker_index=0, worker_count=1,
                analysis_config_path=config,
                manifest_path=manifest_root / "manifest.json", batch_size=2,
                checkpoint_every_batches=1, reservoir_size=0)
            rows = direct.manifest_rows[direct.batch_plan[0]]
            before, after = hidden_pair(2, 512)
            tokens = torch.arange(1024).reshape(2, 512)
            direct.process_batch(
                batch_index=0, rows=rows, token_ids=tokens,
                before_by_layer=before, after_by_layer=after)
            parts = [compute_btm_batch(
                before_by_layer={layer: value[index:index + 1] for layer, value in before.items()},
                after_by_layer={layer: value[index:index + 1] for layer, value in after.items()},
                manifest_rows=rows[index:index + 1]) for index in range(2)]
            combined = BTMBatch(
                raw_b=torch.cat([x.raw_b for x in parts]),
                b_min={s: torch.cat([x.b_min[s] for x in parts]) for s in (128, 256)},
                t_min={s: torch.cat([x.t_min[s] for x in parts]) for s in (128, 256)},
                rel_l2=torch.cat([x.rel_l2 for x in parts]),
                abs_log_r=torch.cat([x.abs_log_r for x in parts]),
                eligible=torch.cat([x.eligible for x in parts]),
            )
            split.process_precomputed_batch(
                batch_index=0, rows=rows, token_ids=tokens, metrics=combined)
            with np.load(direct.progress["shards"][0]["path"], allow_pickle=False) as a, \
                    np.load(split.progress["shards"][0]["path"], allow_pickle=False) as b:
                for key in a.files:
                    if key != "metadata_json":
                        np.testing.assert_array_equal(a[key], b[key])

    def test_atomic_resume_keeps_exact_chunk_b_and_no_gt_output(self) -> None:
        dataset = FakeDataset([512, 512])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fake_text_document"
            prefix.with_suffix(".idx").write_bytes(b"idx")
            prefix.with_suffix(".bin").write_bytes(b"bin")
            manifest_root = root / "manifest"
            build_full_train_manifest(dataset, dataset_prefix=str(prefix), output_dir=manifest_root)
            config = root / "analysis_config.json"
            config.write_text(json.dumps({"candidate_thresholds": {"95": {}}}))
            output = root / "run"
            worker = FullCensusWorker(
                output_dir=output, worker_index=0, worker_count=1,
                analysis_config_path=config, manifest_path=manifest_root / "manifest.json",
                batch_size=1, checkpoint_every_batches=1, reservoir_size=1,
            )
            before, after = hidden_pair(1, 512)
            rows = worker.manifest_rows[worker.batch_plan[0]]
            worker.process_batch(
                batch_index=0, rows=rows, token_ids=torch.zeros((1, 512), dtype=torch.long),
                before_by_layer=before, after_by_layer=after,
            )
            resumed = FullCensusWorker(
                output_dir=output, worker_index=0, worker_count=1,
                analysis_config_path=config, manifest_path=manifest_root / "manifest.json",
                batch_size=1, checkpoint_every_batches=1, reservoir_size=1,
            )
            self.assertEqual(resumed.next_batch_index, 1)
            rows = resumed.manifest_rows[resumed.batch_plan[1]]
            resumed.process_batch(
                batch_index=1, rows=rows, token_ids=torch.zeros((1, 512), dtype=torch.long),
                before_by_layer=before, after_by_layer=after,
            )
            summary = resumed.finalize()
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["full_partition_complete"])
            self.assertFalse(summary["final_gt_created"])
            shard = Path(summary["exact_chunk_b_inventory"][0])
            with np.load(shard, allow_pickle=False) as arrays:
                self.assertIn("raw_b_cka", arrays.files)
                self.assertNotIn("selected", arrays.files)
                self.assertEqual(arrays["raw_b_cka"].shape, (1, len(CHUNK_LAYOUT), 8))

    def test_benchmark_subset_completes_without_claiming_full_partition(self) -> None:
        dataset = FakeDataset([512, 512, 512])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "fake_text_document"
            prefix.with_suffix(".idx").write_bytes(b"idx")
            prefix.with_suffix(".bin").write_bytes(b"bin")
            build_full_train_manifest(dataset, dataset_prefix=str(prefix), output_dir=root / "manifest")
            config = root / "config.json"
            config.write_text("{}")
            worker = FullCensusWorker(
                output_dir=root / "run", worker_index=0, worker_count=1,
                analysis_config_path=config, manifest_path=root / "manifest" / "manifest.json",
                batch_size=1, max_windows=1, full_length_only=True, reservoir_size=0,
            )
            self.assertEqual(len(worker.manifest_rows), 1)
            self.assertEqual(len(manifest_batch_plan(worker.manifest_rows, 1)), 1)


if __name__ == "__main__":
    unittest.main()
