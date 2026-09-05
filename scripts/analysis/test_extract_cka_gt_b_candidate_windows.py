#!/usr/bin/env python3
"""CPU tests for exact/conservative full-census B candidate extraction."""

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
    build_full_train_manifest,
)
from cka_gt_pilot_windows import WINDOW_DTYPE, file_sha256  # noqa: E402
from extract_cka_gt_b_candidate_windows import (  # noqa: E402
    BThresholdBundle,
    candidate_masks_from_raw_b,
    extract_b_candidate_windows,
    load_frozen_b_thresholds,
)


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


def rows(count: int) -> np.ndarray:
    result = np.empty(count, dtype=WINDOW_DTYPE)
    for index in range(count):
        result[index] = (index, index, index, 0, 512, 512, 0, 0)
    return result


def bundle(level: int, threshold: float) -> BThresholdBundle:
    return BThresholdBundle(
        level,
        {
            128: np.full(8, threshold, dtype=np.float32),
            256: np.full(8, threshold, dtype=np.float32),
        },
    )


def write_config(
    path: Path,
    thresholds: dict[int, float],
    *,
    source_identity: dict | None = None,
) -> None:
    if source_identity is None:
        source_identity = {
            "schema": "cka_gt_pilot_source_dataset_identity_v1",
            "storage_kind": "physical_indexed_dataset_files",
            "resolved_prefix": "/tmp/dummy",
            "idx_sha256": "1" * 64,
            "idx_size_bytes": 1,
            "bin_sha256": "2" * 64,
            "bin_size_bytes": 1,
        }
    payload = {
        "schema": "cka_gt_pilot_postprocess_v1",
        "layers": list(range(2, 10)),
        "scales_used_for_gt": [128, 256],
        "consensus": "condition-specific >=7/8, except n_valid=6 requires 6/6; both scales AND",
        "candidate_thresholds": {
            str(level): {
                "level": level,
                "B_lower_threshold": {
                    "128": [value] * 8,
                    "256": [value] * 8,
                },
            }
            for level, value in thresholds.items()
        },
        "prepared_input_provenance": {
            "available": True,
            "checkpoint_identity": {
                "before": {"content_sha256": "a" * 64},
                "after": {"content_sha256": "b" * 64},
            },
            "source_dataset_identity": {"code": source_identity},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class CandidateKernelTest(unittest.TestCase):
    def test_atomic_segment_optimization_matches_slow_token_expansion(self) -> None:
        manifest_rows = np.empty(3, dtype=WINDOW_DTYPE)
        manifest_rows[0] = (0, 0, 0, 0, 512, 512, 0, 0)
        manifest_rows[1] = (1, 1, 1, 0, 420, 384, 36, 1)
        manifest_rows[2] = (2, 2, 2, 0, 300, 256, 44, 1)
        rng = np.random.default_rng(81)
        raw = rng.uniform(0.75, 1.0, (3, len(CHUNK_LAYOUT), 8)).astype(np.float32)
        for row_index, row in enumerate(manifest_rows):
            for slot, (scale, start) in enumerate(CHUNK_LAYOUT):
                if start + scale > int(row["window_length"]):
                    raw[row_index, slot] = np.nan
        raw[rng.random(raw.shape) < 0.04] = np.nan
        bundles = {95: bundle(95, 0.94), 97: bundle(97, 0.90), 99: bundle(99, 0.82)}
        fast = candidate_masks_from_raw_b(raw, manifest_rows, bundles, block_windows=2)

        slow_masks: dict[int, np.ndarray] = {}
        slow_counts: dict[int, int] = {}
        slow_union_tokens = np.zeros((3, 512), dtype=bool)
        for level, item in bundles.items():
            both = np.ones((3, 512), dtype=bool)
            eligible = np.arange(512)[None, :] < manifest_rows["eligible_token_count"][:, None]
            both &= eligible
            for wanted_scale in (128, 256):
                minima = np.full((3, 512, 8), np.nan, dtype=np.float32)
                for slot, (scale, start) in enumerate(CHUNK_LAYOUT):
                    if scale == wanted_scale:
                        minima[:, start : start + scale] = np.fmin(
                            minima[:, start : start + scale], raw[:, slot, None, :]
                        )
                valid = np.isfinite(minima)
                n_valid = valid.sum(axis=-1)
                n_pass = (
                    valid
                    & (minima >= item.thresholds[wanted_scale][None, None, :])
                ).sum(axis=-1)
                both &= (n_valid >= 6) & (
                    (n_pass >= 7) | ((n_valid == 6) & (n_pass == 6))
                )
            slow_masks[level] = both.any(axis=1)
            slow_counts[level] = int(both.sum())
            slow_union_tokens |= both
        for level in bundles:
            np.testing.assert_array_equal(fast.window_masks[level], slow_masks[level])
            self.assertEqual(fast.token_counts[level], slow_counts[level])
        np.testing.assert_array_equal(
            fast.union_window_mask,
            np.logical_or.reduce(list(slow_masks.values())),
        )
        self.assertEqual(fast.union_token_count, int(slow_union_tokens.sum()))

    def test_condition_specific_seven_of_eight_and_six_of_six(self) -> None:
        manifest_rows = rows(3)
        raw = np.ones((3, len(CHUNK_LAYOUT), 8), dtype=np.float32)
        # Seven passing of eight valid layers: pass.
        raw[0, :, 0] = 0.0
        # Exactly six valid, all six passing: pass.
        raw[1, :, :2] = np.nan
        # Only six passing of eight valid: fail.
        raw[2, :, :2] = 0.0
        result = candidate_masks_from_raw_b(
            raw, manifest_rows, {99: bundle(99, 0.9)}, block_windows=1
        )
        np.testing.assert_array_equal(
            result.window_masks[99], np.asarray([True, True, False])
        )
        self.assertEqual(result.token_counts[99], 1024)

    def test_union_is_conservative_when_bundle_thresholds_are_not_monotone(self) -> None:
        manifest_rows = rows(2)
        raw = np.empty((2, len(CHUNK_LAYOUT), 8), dtype=np.float32)
        raw[0].fill(0.85)  # B95-only under this deliberately nonmonotone fixture.
        raw[1].fill(0.95)  # All bundles.
        bundles = {95: bundle(95, 0.8), 97: bundle(97, 0.9), 99: bundle(99, 0.9)}
        result = candidate_masks_from_raw_b(raw, manifest_rows, bundles)
        np.testing.assert_array_equal(result.window_masks[95], [True, True])
        np.testing.assert_array_equal(result.window_masks[97], [False, True])
        np.testing.assert_array_equal(result.window_masks[99], [False, True])
        np.testing.assert_array_equal(result.union_window_mask, [True, True])

    def test_both_scales_are_required(self) -> None:
        manifest_rows = rows(1)
        raw = np.ones((1, len(CHUNK_LAYOUT), 8), dtype=np.float32)
        for slot, (scale, _start) in enumerate(CHUNK_LAYOUT):
            if scale == 256:
                raw[:, slot, :2] = 0.0
        result = candidate_masks_from_raw_b(
            raw, manifest_rows, {99: bundle(99, 0.9)}
        )
        self.assertFalse(result.window_masks[99][0])


class EndToEndExtractorTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        dataset = FakeDataset([512, 512])
        prefix = root / "fake_text_document"
        prefix.with_suffix(".idx").write_bytes(b"idx")
        prefix.with_suffix(".bin").write_bytes(b"bin")
        source_identity = {
            "schema": "cka_gt_pilot_source_dataset_identity_v1",
            "storage_kind": "physical_indexed_dataset_files",
            "resolved_prefix": str(prefix.resolve()),
            "idx_sha256": file_sha256(prefix.with_suffix(".idx")),
            "idx_size_bytes": prefix.with_suffix(".idx").stat().st_size,
            "bin_sha256": file_sha256(prefix.with_suffix(".bin")),
            "bin_size_bytes": prefix.with_suffix(".bin").stat().st_size,
        }
        pilot_manifest = root / "pilot_manifest.json"
        pilot_manifest.write_text(
            json.dumps(
                {
                    "dataset_prefix": str(prefix),
                    "source_dataset_identity": source_identity,
                }
            ),
            encoding="utf-8",
        )
        manifest_dir = root / "manifest"
        build_full_train_manifest(
            dataset,
            dataset_prefix=str(prefix),
            output_dir=manifest_dir,
            pilot_manifest=pilot_manifest,
        )
        config = root / "analysis_config.json"
        write_config(
            config,
            {95: 0.99, 97: 0.97, 99: 0.90},
            source_identity=source_identity,
        )
        census = root / "census"
        for worker_index, raw_value in enumerate((1.0, 0.0)):
            worker = FullCensusWorker(
                output_dir=census,
                worker_index=worker_index,
                worker_count=2,
                analysis_config_path=config,
                manifest_path=manifest_dir / "manifest.json",
                batch_size=1,
                checkpoint_every_batches=1,
                reservoir_size=0,
            )
            batch_rows = worker.manifest_rows[worker.batch_plan[0]]
            raw_b = torch.full(
                (1, len(CHUNK_LAYOUT), 8), raw_value, dtype=torch.float32
            )
            token_shape = (1, 512, 8)
            metrics = BTMBatch(
                raw_b=raw_b,
                b_min={
                    128: torch.full(token_shape, raw_value),
                    256: torch.full(token_shape, raw_value),
                },
                t_min={
                    128: torch.ones(token_shape),
                    256: torch.ones(token_shape),
                },
                rel_l2=torch.zeros(token_shape),
                abs_log_r=torch.zeros(token_shape),
                eligible=torch.ones((1, 512), dtype=torch.bool),
            )
            worker.process_precomputed_batch(
                batch_index=0,
                rows=batch_rows,
                token_ids=torch.zeros((1, 512), dtype=torch.long),
                metrics=metrics,
                metric_seconds=0.0,
            )
            worker.finalize()
            binding = {
                "analysis_config": str(config.resolve()),
                "checkpoint_every_batches": 1,
                "current_checkpoint_identity": "b" * 64,
                "global_token_score_reservoir_size": 0,
                "histogram_bins": 4096,
                "manifest_path": str((manifest_dir / "manifest.json").resolve()),
                "max_windows": 0,
                "partition_windows": 1,
                "raw_hidden_stored": False,
                "reference_checkpoint_identity": "a" * 64,
                "schema": "cka_gt_full_census_model_binding_v2",
                "source_dataset_identity": source_identity,
                "threshold_policy": "none_distribution_census_pilot_lines_overlay_only",
                "window_batch_size": 1,
                "worker_count": 2,
                "worker_index": worker_index,
            }
            (census / f"worker_{worker_index:03d}" / "model_binding.json").write_text(
                json.dumps(binding), encoding="utf-8"
            )
        return census, manifest_dir / "manifest.json", config

    def test_extracts_candidate_and_emits_auditable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            census, manifest, config = self._fixture(root)
            output = root / "candidates"
            result = extract_b_candidate_windows(
                census_root=census,
                manifest_path=manifest,
                analysis_config_path=config,
                output_dir=output,
                worker_count=2,
                block_windows=1,
                require_merged_summary=False,
            )
            selected = np.load(output / "candidate_windows.npy", allow_pickle=False)
            self.assertEqual(selected.dtype, WINDOW_DTYPE)
            self.assertEqual(selected["sample_order"].tolist(), [0])
            stats = result["statistics"]
            self.assertEqual(result["schema"], "cka_gt_b_candidate_windows_v2")
            self.assertEqual(stats["sample_order_coverage"]["seen_once"], 2)
            self.assertTrue(stats["candidate_union_equals_exact_B99_windows"])
            self.assertTrue(all(stats["subset_checks"].values()))
            self.assertTrue((output / "REPORT.md").is_file())
            bindings = result["census_worker_model_bindings"]
            self.assertTrue(bindings["validated"])
            self.assertTrue(bindings["all_workers_agree"])
            self.assertEqual(len(bindings["files"]), 2)
            for item in bindings["files"]:
                path = Path(item["file_identity"]["path"])
                self.assertEqual(item["file_identity"]["sha256"], file_sha256(path))

    def test_duplicate_or_out_of_partition_sample_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            census, manifest, config = self._fixture(root)
            progress_path = census / "worker_001" / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            record = progress["shards"][0]
            shard_path = Path(record["path"])
            with np.load(shard_path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key].copy() for key in loaded.files}
            arrays["window_sample_order"][0] = 0
            with shard_path.open("wb") as handle:
                np.savez(handle, **arrays)
            record["size_bytes"] = shard_path.stat().st_size
            record["sha256"] = file_sha256(shard_path)
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outside partition"):
                extract_b_candidate_windows(
                    census_root=census,
                    manifest_path=manifest,
                    analysis_config_path=config,
                    output_dir=root / "bad_output",
                    worker_count=2,
                    require_merged_summary=False,
                )

    def test_frozen_config_requires_all_three_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_config(path, {95: 0.99, 99: 0.9})
            with self.assertRaisesRegex(ValueError, "missing required bundles"):
                load_frozen_b_thresholds(path)

    def test_corrupt_checkpoint_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            census, manifest, config = self._fixture(root)
            path = census / "worker_001" / "model_binding.json"
            binding = json.loads(path.read_text(encoding="utf-8"))
            binding["current_checkpoint_identity"] = "c" * 64
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "current checkpoint identity differs"):
                extract_b_candidate_windows(
                    census_root=census,
                    manifest_path=manifest,
                    analysis_config_path=config,
                    output_dir=root / "bad_checkpoint",
                    worker_count=2,
                    require_merged_summary=False,
                )

    def test_corrupt_logical_batch_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            census, manifest, config = self._fixture(root)
            path = census / "worker_000" / "model_binding.json"
            binding = json.loads(path.read_text(encoding="utf-8"))
            binding["window_batch_size"] = 2
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "logical window batch differs"):
                extract_b_candidate_windows(
                    census_root=census,
                    manifest_path=manifest,
                    analysis_config_path=config,
                    output_dir=root / "bad_batch",
                    worker_count=2,
                    require_merged_summary=False,
                )

    def test_corrupt_source_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            census, manifest, config = self._fixture(root)
            path = census / "worker_000" / "model_binding.json"
            binding = json.loads(path.read_text(encoding="utf-8"))
            binding["source_dataset_identity"]["bin_sha256"] = "d" * 64
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Code source identity differs"):
                extract_b_candidate_windows(
                    census_root=census,
                    manifest_path=manifest,
                    analysis_config_path=config,
                    output_dir=root / "bad_source",
                    worker_count=2,
                    require_merged_summary=False,
                )


if __name__ == "__main__":
    unittest.main()
