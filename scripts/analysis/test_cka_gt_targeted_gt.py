#!/usr/bin/env python3
"""CPU-only tests for the exact targeted CKA GT pass."""

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

from cka_gt_full_census import BTMBatch, _canonical_json  # noqa: E402
from cka_gt_pilot_windows import WINDOW_DTYPE, file_sha256  # noqa: E402
from cka_gt_targeted_gt import (  # noqa: E402
    BUNDLE_LEVELS,
    ExactTargetedGTWriter,
    evaluate_bundle,
    load_threshold_bundles,
    materialize_locked_bundle,
    unpack_mask,
)


def rows(count: int) -> np.ndarray:
    value = np.empty(count, dtype=WINDOW_DTYPE)
    for index in range(count):
        # Mix lengths so deterministic batch planning is nontrivial.
        length = 512 if index % 2 == 0 else 384
        value[index] = (index, index, 100 + index // 2, index * 64, length, length, 0, int(length < 512))
    return value


def metrics(count: int, length: int, score: float = 1.0) -> BTMBatch:
    shape = (count, length, 8)
    b = {scale: torch.full(shape, score) for scale in (128, 256)}
    t = {scale: torch.full(shape, score) for scale in (128, 256)}
    return BTMBatch(
        raw_b=torch.ones((count, 10, 8)),
        b_min=b,
        t_min=t,
        rel_l2=torch.zeros(shape),
        abs_log_r=torch.zeros(shape),
        eligible=torch.ones((count, length), dtype=torch.bool),
    )


def slice_metrics(value: BTMBatch, index: np.ndarray) -> BTMBatch:
    tensor_index = torch.as_tensor(index)
    return BTMBatch(
        raw_b=value.raw_b.index_select(0, tensor_index),
        b_min={scale: item.index_select(0, tensor_index) for scale, item in value.b_min.items()},
        t_min={scale: item.index_select(0, tensor_index) for scale, item in value.t_min.items()},
        rel_l2=value.rel_l2.index_select(0, tensor_index),
        abs_log_r=value.abs_log_r.index_select(0, tensor_index),
        eligible=value.eligible.index_select(0, tensor_index),
    )


def write_config(path: Path) -> None:
    levels = {95: 0.99, 97: 0.95, 99: 0.80}
    payload = {
        "schema": "cka_gt_pilot_postprocess_v1",
        "candidate_thresholds": {
            str(level): {
                "level": level,
                "B_lower_threshold": {"128": [cut] * 8, "256": [cut] * 8},
                "T_lower_threshold": {"128": [cut] * 8, "256": [cut] * 8},
                "relative_l2_upper_threshold": [1.0 - cut + 0.02] * 8,
                "abs_log_r_upper_threshold": [1.0 - cut + 0.02] * 8,
            }
            for level, cut in levels.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def write_candidate_input(
    root: Path, value: np.ndarray, *, expected_b_offset: int = 0
) -> tuple[Path, Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    candidates = root / "candidate_windows.npy"
    np.save(candidates, value, allow_pickle=False)
    config = root / "analysis_config.json"
    write_config(config)
    full_manifest = root / "full_manifest.json"
    full_manifest.write_text("{}\n", encoding="utf-8")
    manifest_identity = "a" * 64
    census = root / "census_summary.json"
    census.write_text(
        json.dumps(
            {
                "schema": "cka_gt_full_census_merged_summary_v2",
                "complete": True,
                "threshold_free": True,
                "manifest_identity": manifest_identity,
                "processed_eligible_tokens": 10_000,
                "processed_windows": 4,
            }
        ),
        encoding="utf-8",
    )
    reference_identity, current_identity = "b" * 64, "c" * 64
    source_dataset_identity = {
        "schema": "source_v1",
        "resolved_prefix": "/synthetic/code",
        "idx_sha256": "d" * 64,
        "bin_sha256": "e" * 64,
    }
    worker_files = []
    partition_sizes = [1, 1, 1, 1]
    for worker_index in range(4):
        worker_content = {
            "schema": "cka_gt_full_census_model_binding_v2",
            "worker_index": worker_index,
            "worker_count": 4,
            "analysis_config": str(config.resolve()),
            "manifest_path": str(full_manifest.resolve()),
            "reference_checkpoint_identity": reference_identity,
            "current_checkpoint_identity": current_identity,
            "source_dataset_identity": source_dataset_identity,
            "window_batch_size": 192,
            "checkpoint_every_batches": 100,
            "histogram_bins": 4096,
            "global_token_score_reservoir_size": 5_000_000,
            "max_windows": 0,
            "raw_hidden_stored": False,
            "threshold_policy": "none_distribution_census_pilot_lines_overlay_only",
            "partition_windows": partition_sizes[worker_index],
        }
        worker_path = root / f"worker_{worker_index}_model_binding.json"
        worker_path.write_text(json.dumps(worker_content), encoding="utf-8")
        worker_files.append(
            {
                "worker_index": worker_index,
                "file_identity": identity(worker_path),
                "content": worker_content,
            }
        )
    common_content = {
        "binding_schema": "cka_gt_full_census_model_binding_v2",
        "analysis_config": {
            "path": str(config.resolve()),
            "sha256": file_sha256(config),
        },
        "manifest": {
            "path": str(full_manifest.resolve()),
            "content_identity": manifest_identity,
        },
        "reference_checkpoint_identity": reference_identity,
        "current_checkpoint_identity": current_identity,
        "source_dataset_identity": source_dataset_identity,
        "window_batch_size": 192,
        "checkpoint_every_batches": 100,
        "histogram_bins": 4096,
        "global_token_score_reservoir_size": 5_000_000,
        "max_windows": 0,
        "raw_hidden_stored": False,
        "threshold_policy": "none_distribution_census_pilot_lines_overlay_only",
    }
    worker_binding_set = {
        "schema": "cka_gt_full_census_model_binding_set_v1",
        "validated": True,
        "all_workers_agree": True,
        "worker_count": 4,
        "files": worker_files,
        "common_content": common_content,
        "common_content_sha256": __import__("hashlib").sha256(
            _canonical_json(common_content)
        ).hexdigest(),
        "per_worker_content": [
            {"worker_index": index, "partition_windows": count}
            for index, count in enumerate(partition_sizes)
        ],
    }
    worker_binding_set["binding_set_content_sha256"] = __import__("hashlib").sha256(
        _canonical_json(worker_binding_set)
    ).hexdigest()
    expected_b_count = int(value["eligible_token_count"].sum()) + int(expected_b_offset)
    manifest = root / "manifest.json"
    manifest_payload = {
        "schema": "cka_gt_b_candidate_windows_v2",
        "complete": True,
        "not_final_gt": True,
        "source_manifest": {
            "path": str(full_manifest.resolve()),
            "content_identity": manifest_identity,
        },
        "candidate_windows": identity(candidates),
        "analysis_config": identity(config),
        "merged_census_summary": identity(census),
        "census_worker_model_bindings": worker_binding_set,
        "statistics": {
            "source_eligible_token_count": 10_000,
            "candidate_window_count": len(value),
            "per_bundle": {
                str(level): {"B_passing_tokens": expected_b_count}
                for level in BUNDLE_LEVELS
            },
        },
    }
    manifest_payload["manifest_content_sha256"] = __import__("hashlib").sha256(
        _canonical_json(manifest_payload)
    ).hexdigest()
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    authoritative = root / "authoritative_b.npz"
    token_mask = np.zeros((len(value), 512), dtype=np.bool_)
    for index, length in enumerate(value["eligible_token_count"]):
        token_mask[index, : int(length)] = True
    layer_mask = np.repeat(token_mask[..., None], 8, axis=2)
    packed_token = np.packbits(token_mask, axis=1, bitorder="little")
    packed_layer = np.packbits(layer_mask, axis=1, bitorder="little")
    authoritative_metadata = {
        "schema": "cka_gt_authoritative_b_masks_v1",
        "complete": True,
        "raw_hidden_stored": False,
        "sealed_test_opened": False,
        "candidate_window_count": len(value),
        "candidate_windows": identity(candidates),
        "candidate_manifest": {
            **identity(manifest),
            "manifest_content_sha256": manifest_payload["manifest_content_sha256"],
        },
        "analysis_config": identity(config),
        "merged_census_summary": identity(census),
        "census_worker_model_binding_set_sha256": worker_binding_set[
            "binding_set_content_sha256"
        ],
        "bundle_b_token_counts": {
            str(level): expected_b_count for level in BUNDLE_LEVELS
        },
    }
    arrays = {
        "metadata_json": np.frombuffer(
            json.dumps(authoritative_metadata).encode(), dtype=np.uint8
        )
    }
    for level in BUNDLE_LEVELS:
        arrays[f"bundle_{level}_token_packed"] = packed_token
        for scale in (128, 256):
            arrays[f"bundle_{level}_scale_{scale}_pass_packed"] = packed_layer
    for scale in (128, 256):
        arrays[f"scale_{scale}_valid_packed"] = packed_layer
    np.savez(authoritative, **arrays)
    binding = root / "targeted_model_binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "cka_gt_exact_targeted_model_binding_v1",
                "candidate_windows": identity(candidates),
                "candidate_manifest": identity(manifest),
                "analysis_config": identity(config),
                "census_summary": identity(census),
                "authoritative_b": identity(authoritative),
                "reference_checkpoint_identity": reference_identity,
                "current_checkpoint_identity": current_identity,
                "source_dataset_identity": source_dataset_identity,
                "batch_size": 2,
                "checkpoint_every_batches": 1,
            }
        ),
        encoding="utf-8",
    )
    return candidates, manifest, config, census, binding


class SelectorTest(unittest.TestCase):
    def test_seven_of_eight_and_exact_six_of_six(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _candidates, _manifest, config, _census, _binding = write_candidate_input(root, rows(1))
            _identity, bundles = load_threshold_bundles(config)
            value = metrics(3, 8)
            # Seven passing out of eight: pass.
            for condition in (*value.b_min.values(), *value.t_min.values()):
                condition[0, :, 0] = 0.0
            # Exactly six finite and passing: pass.
            for condition in (*value.b_min.values(), *value.t_min.values(), value.rel_l2, value.abs_log_r):
                condition[1, :, :2] = torch.nan
            # Only six passing out of eight: fail.
            for condition in (*value.b_min.values(), *value.t_min.values()):
                condition[2, :, :2] = 0.0
            result = evaluate_bundle(value, bundles[95])
            self.assertTrue(result["selected"][0].all())
            self.assertTrue(result["selected"][1].all())
            self.assertFalse(result["selected"][2].any())

    def test_frozen_bundle_masks_are_nested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _candidates, _manifest, config, _census, _binding = write_candidate_input(root, rows(1))
            _identity, bundles = load_threshold_bundles(config)
            value = metrics(1, 16, score=0.97)
            selected = {
                level: evaluate_bundle(value, bundles[level])["selected"]
                for level in BUNDLE_LEVELS
            }
            self.assertFalse(selected[95].any())
            self.assertTrue(selected[97].all())
            self.assertTrue(selected[99].all())


class WriterTest(unittest.TestCase):
    def _run(self, root: Path, *, resume_after_first: bool) -> ExactTargetedGTWriter:
        source_rows = rows(4)
        candidates, manifest, config, census, binding = write_candidate_input(root, source_rows)

        def new_writer() -> ExactTargetedGTWriter:
            return ExactTargetedGTWriter(
                output_dir=root / "out",
                candidate_windows_path=candidates,
                candidate_manifest_path=manifest,
                analysis_config_path=config,
                census_summary_path=census,
                model_binding_path=binding,
                batch_size=2,
                checkpoint_every_batches=1,
            )

        writer = new_writer()
        for batch_index in range(writer.next_batch_index, writer.expected_batch_count):
            index = writer.plan[batch_index]
            batch_rows = np.asarray(writer.rows[index])
            length = int(batch_rows["window_length"][0])
            token_ids = torch.stack(
                [torch.arange(length) + int(item) * 1000 for item in index]
            )
            writer.process_batch(
                batch_index=batch_index,
                candidate_indices=index,
                rows=batch_rows,
                token_ids=token_ids,
                metrics=metrics(len(index), length),
            )
            if resume_after_first and batch_index == 0:
                writer = new_writer()
        writer.finalize()
        return writer

    def test_batch_resume_equivalence_and_mask_occurrence_identity(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            direct = self._run(Path(first), resume_after_first=False)
            resumed = self._run(Path(second), resume_after_first=True)
            for level in BUNDLE_LEVELS:
                direct_mask = np.load(direct.output_dir / f"bundle_{level}_packed.npy")
                resumed_mask = np.load(resumed.output_dir / f"bundle_{level}_packed.npy")
                np.testing.assert_array_equal(direct_mask, resumed_mask)
                self.assertEqual(int(unpack_mask(direct_mask).sum()), 1792)
                direct_occ = np.load(direct.output_dir / f"bundle_{level}_occurrences.npy")
                resumed_occ = np.load(resumed.output_dir / f"bundle_{level}_occurrences.npy")
                np.testing.assert_array_equal(direct_occ, resumed_occ)
                self.assertEqual(direct_occ.size, 1792)
                self.assertTrue(np.all(direct_occ["sample_order"] == direct_occ["candidate_index"]))
            summary = json.loads(direct.summary_path.read_text())
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["bundle_nesting"]["95_subset_97"])
            self.assertFalse(summary["sealed_test_opened"])

    def test_resume_rejects_candidate_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_rows = rows(2)
            candidates, manifest, config, census, binding = write_candidate_input(root, source_rows)
            writer = ExactTargetedGTWriter(
                output_dir=root / "out",
                candidate_windows_path=candidates,
                candidate_manifest_path=manifest,
                analysis_config_path=config,
                census_summary_path=census,
                model_binding_path=binding,
                batch_size=2,
                checkpoint_every_batches=1,
            )
            index = writer.plan[0]
            batch_rows = np.asarray(writer.rows[index])
            length = int(batch_rows["window_length"][0])
            writer.process_batch(
                batch_index=0,
                candidate_indices=index,
                rows=batch_rows,
                token_ids=torch.arange(length).reshape(1, length),
                metrics=metrics(1, length),
            )
            changed = np.load(candidates)
            changed[0]["document_id"] += 1
            np.save(candidates, changed, allow_pickle=False)
            with self.assertRaises((RuntimeError, ValueError)):
                ExactTargetedGTWriter(
                    output_dir=root / "out",
                    candidate_windows_path=candidates,
                    candidate_manifest_path=manifest,
                    analysis_config_path=config,
                    census_summary_path=census,
                    model_binding_path=binding,
                    batch_size=2,
                    checkpoint_every_batches=1,
                )

    def test_preflight_rejects_manifest_census_and_worker_binding_corruption(self) -> None:
        def construct(paths: tuple[Path, Path, Path, Path, Path], output: Path):
            candidates, manifest, config, census, binding = paths
            return ExactTargetedGTWriter(
                output_dir=output,
                candidate_windows_path=candidates,
                candidate_manifest_path=manifest,
                analysis_config_path=config,
                census_summary_path=census,
                model_binding_path=binding,
                batch_size=2,
                checkpoint_every_batches=1,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_candidate_input(root / "manifest_case", rows(2))
            manifest = paths[1]
            payload = json.loads(manifest.read_text())
            payload["statistics"]["candidate_window_count"] += 1
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical content hash"):
                construct(paths, root / "manifest_out")

            paths = write_candidate_input(root / "census_case", rows(2))
            with paths[3].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "census-summary identity"):
                construct(paths, root / "census_out")

            paths = write_candidate_input(root / "worker_case", rows(2))
            candidate_payload = json.loads(paths[1].read_text())
            worker_path = Path(
                candidate_payload["census_worker_model_bindings"]["files"][0][
                    "file_identity"
                ]["path"]
            )
            with worker_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "model-binding identity changed"):
                construct(paths, root / "worker_out")

    def test_progress_rejects_targeted_model_binding_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates, manifest, config, census, binding = write_candidate_input(
                root, rows(2)
            )
            kwargs = dict(
                output_dir=root / "out",
                candidate_windows_path=candidates,
                candidate_manifest_path=manifest,
                analysis_config_path=config,
                census_summary_path=census,
                model_binding_path=binding,
                batch_size=2,
                checkpoint_every_batches=1,
            )
            ExactTargetedGTWriter(**kwargs)
            with binding.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(RuntimeError, "model_binding"):
                ExactTargetedGTWriter(**kwargs)

    def test_preflight_rejects_authoritative_b_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_rows = rows(4)
            candidates, manifest, config, census, binding = write_candidate_input(
                root, source_rows, expected_b_offset=1
            )
            with self.assertRaisesRegex(ValueError, "unpacked count differs"):
                ExactTargetedGTWriter(
                    output_dir=root / "out",
                    candidate_windows_path=candidates,
                    candidate_manifest_path=manifest,
                    analysis_config_path=config,
                    census_summary_path=census,
                    model_binding_path=binding,
                    batch_size=2,
                    checkpoint_every_batches=1,
                )

    def test_authoritative_b_controls_selection_while_recomputed_b_is_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates, manifest, config, census, binding = write_candidate_input(
                root, rows(2)
            )
            writer = ExactTargetedGTWriter(
                output_dir=root / "out",
                candidate_windows_path=candidates,
                candidate_manifest_path=manifest,
                analysis_config_path=config,
                census_summary_path=census,
                model_binding_path=binding,
                batch_size=2,
                checkpoint_every_batches=1,
            )
            for batch_index in range(writer.expected_batch_count):
                index = writer.plan[batch_index]
                batch_rows = np.asarray(writer.rows[index])
                length = int(batch_rows["window_length"][0])
                value = metrics(len(index), length)
                for item in value.b_min.values():
                    item.zero_()  # Deliberately disagree with authoritative census B.
                writer.process_batch(
                    batch_index=batch_index,
                    candidate_indices=index,
                    rows=batch_rows,
                    token_ids=torch.zeros((len(index), length), dtype=torch.long),
                    metrics=value,
                )
            summary = writer.finalize()
            expected = int(rows(2)["eligible_token_count"].sum())
            self.assertEqual(summary["bundles"]["95"]["selected_occurrences"], expected)
            validation = summary["exact_b_count_validation"]["bundles"]["95"]
            self.assertEqual(validation["observed_authoritative_full_census"], expected)
            self.assertEqual(validation["observed_targeted_gpu_recomputed_diagnostic"], 0)
            self.assertTrue(validation["authoritative_matches"])

    def test_lock_is_create_only_and_preserves_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self._run(root / "measurement", resume_after_first=True)
            locked = root / "locked_bundle_97"
            result = materialize_locked_bundle(
                exact_dir=writer.output_dir,
                bundle_level=97,
                output_dir=locked,
                decision_reason="synthetic exact-result review",
            )
            self.assertEqual(result["bundle_level"], 97)
            metadata = json.loads((locked / "metadata.json").read_text())
            validation = json.loads((locked / "validation.json").read_text())
            lock = json.loads((locked / "threshold_lock.json").read_text())
            self.assertEqual(metadata["axis"], "candidate_window_axis")
            self.assertTrue(validation["passed"])
            self.assertFalse(lock["sealed_test_opened"])
            self.assertEqual(
                lock["authoritative_lineage"]["live_checkpoint_content_identities"][
                    "reference"
                ],
                "b" * 64,
            )
            self.assertEqual(
                validation["census_worker_model_bindings_verified"], 4
            )
            self.assertIn("authoritative_lineage", metadata)
            np.testing.assert_array_equal(
                np.load(locked / "old_like_gt_packed.npy"),
                np.load(writer.output_dir / "bundle_97_packed.npy"),
            )
            with self.assertRaises(FileExistsError):
                materialize_locked_bundle(
                    exact_dir=writer.output_dir,
                    bundle_level=97,
                    output_dir=locked,
                    decision_reason="must not overwrite",
                )

    def test_lock_rejects_post_measurement_worker_binding_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = self._run(root / "measurement", resume_after_first=False)
            summary = json.loads(writer.summary_path.read_text())
            worker_path = Path(
                summary["census_worker_model_bindings"]["files"][2]["file_identity"][
                    "path"
                ]
            )
            with worker_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(RuntimeError, "model-binding hash changed"):
                materialize_locked_bundle(
                    exact_dir=writer.output_dir,
                    bundle_level=99,
                    output_dir=root / "must_not_lock",
                    decision_reason="corruption test",
                )


if __name__ == "__main__":
    unittest.main()
