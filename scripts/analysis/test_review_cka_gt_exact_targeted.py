#!/usr/bin/env python3
"""CPU-only synthetic tests for the exact CKA targeted-result review."""

from __future__ import annotations

import hashlib
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

from cka_gt_full_census import BTMBatch  # noqa: E402
from cka_gt_pilot_windows import WINDOW_DTYPE, file_sha256  # noqa: E402
from cka_gt_targeted_gt import ExactTargetedGTWriter  # noqa: E402
from review_cka_gt_exact_targeted import (  # noqa: E402
    concentration_metrics,
    recommendation,
    run_review,
    token_repetition_metrics,
)


class FakeDataset:
    def __init__(self, documents: list[np.ndarray]) -> None:
        self.documents = [np.asarray(value, dtype=np.int32) for value in documents]
        self.sequence_lengths = np.asarray(
            [len(value) for value in documents], dtype=np.int32
        )
        self.document_indices = np.arange(len(documents) + 1, dtype=np.int64)

    def get(self, index: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        value = self.documents[int(index)]
        if length is None:
            length = len(value) - int(offset)
        return value[int(offset) : int(offset) + int(length)]


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _manifest_content_hash(payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metrics(count: int, length: int) -> BTMBatch:
    shape = (count, length, 8)
    return BTMBatch(
        raw_b=torch.ones((count, 10, 8), dtype=torch.float32),
        b_min={128: torch.ones(shape), 256: torch.ones(shape)},
        t_min={128: torch.ones(shape), 256: torch.ones(shape)},
        rel_l2=torch.zeros(shape),
        abs_log_r=torch.zeros(shape),
        eligible=torch.ones((count, length), dtype=torch.bool),
    )


def _build_exact(root: Path) -> tuple[Path, Path, FakeDataset]:
    source = root / "source"
    source.mkdir(parents=True)
    document_count, length = 12, 16
    documents = [
        np.arange(length, dtype=np.int32) + 1000 * document
        for document in range(document_count)
    ]
    dataset = FakeDataset(documents)
    rows = np.empty(document_count, dtype=WINDOW_DTYPE)
    for index in range(document_count):
        rows[index] = (index, index, index, 0, length, length, 0, 1)
    candidate_path = source / "candidate_windows.npy"
    np.save(candidate_path, rows, allow_pickle=False)
    dataset_prefix = root / "fake_dataset"
    Path(str(dataset_prefix) + ".idx").write_bytes(b"synthetic-index")
    Path(str(dataset_prefix) + ".bin").write_bytes(b"synthetic-bin")
    source_manifest_path = source / "full_manifest.json"
    source_manifest: dict[str, object] = {
        "schema": "cka_gt_full_train_windows_v1",
        "dataset_prefix": str(dataset_prefix.resolve()),
        "dataset_identity_light": {
            "resolved_prefix": str(dataset_prefix.resolve()),
            "idx": {
                "path": str(Path(str(dataset_prefix) + ".idx").resolve()),
                "size_bytes": Path(str(dataset_prefix) + ".idx").stat().st_size,
            },
            "bin": {
                "path": str(Path(str(dataset_prefix) + ".bin").resolve()),
                "size_bytes": Path(str(dataset_prefix) + ".bin").stat().st_size,
            },
        },
        "statistics": {
            "window_count": document_count,
            "eligible_token_count": 10_000,
        },
        "windows": _identity(candidate_path),
    }
    source_manifest["manifest_content_sha256"] = _manifest_content_hash(
        source_manifest
    )
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    config_path = source / "analysis_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "cka_gt_pilot_postprocess_v1",
                "candidate_thresholds": {
                    str(level): {
                        "B_lower_threshold": {
                            "128": [cut] * 8,
                            "256": [cut] * 8,
                        },
                        "T_lower_threshold": {
                            "128": [cut] * 8,
                            "256": [cut] * 8,
                        },
                        "relative_l2_upper_threshold": [1.0] * 8,
                        "abs_log_r_upper_threshold": [1.0] * 8,
                    }
                    for level, cut in ((95, 0.99), (97, 0.97), (99, 0.90))
                },
            }
        ),
        encoding="utf-8",
    )
    census_summary_path = source / "census_summary.json"
    census_summary_path.write_text(
        json.dumps(
            {
                "schema": "cka_gt_full_census_merged_summary_v2",
                "complete": True,
                "threshold_free": True,
                "manifest_identity": source_manifest["manifest_content_sha256"],
                "processed_eligible_tokens": 10_000,
                "processed_windows": document_count,
            }
        ),
        encoding="utf-8",
    )
    reference_identity, current_identity = "b" * 64, "c" * 64
    source_dataset_identity = {
        "schema": "synthetic_source_v1",
        "resolved_prefix": str(dataset_prefix.resolve()),
        "idx_size_bytes": Path(str(dataset_prefix) + ".idx").stat().st_size,
        "bin_size_bytes": Path(str(dataset_prefix) + ".bin").stat().st_size,
    }
    worker_files = []
    for worker_index in range(4):
        content = {
            "schema": "cka_gt_full_census_model_binding_v2",
            "worker_index": worker_index,
            "worker_count": 4,
            "analysis_config": str(config_path.resolve()),
            "manifest_path": str(source_manifest_path.resolve()),
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
            "partition_windows": 3,
        }
        path = source / f"worker_{worker_index}_binding.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        worker_files.append(
            {
                "worker_index": worker_index,
                "file_identity": _identity(path),
                "content": content,
            }
        )
    common_content = {
        "binding_schema": "cka_gt_full_census_model_binding_v2",
        "analysis_config": {
            "path": str(config_path.resolve()),
            "sha256": file_sha256(config_path),
        },
        "manifest": {
            "path": str(source_manifest_path.resolve()),
            "content_identity": source_manifest["manifest_content_sha256"],
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
        "common_content_sha256": hashlib.sha256(
            (json.dumps(common_content, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "per_worker_content": [
            {"worker_index": index, "partition_windows": 3} for index in range(4)
        ],
    }
    worker_binding_set["binding_set_content_sha256"] = _manifest_content_hash(
        worker_binding_set
    )
    manifest_path = source / "manifest.json"
    manifest: dict[str, object] = {
        "schema": "cka_gt_b_candidate_windows_v2",
        "complete": True,
        "not_final_gt": True,
        "source_manifest": {
            "path": str(source_manifest_path.resolve()),
            "content_identity": source_manifest["manifest_content_sha256"],
        },
        "candidate_windows": _identity(candidate_path),
        "analysis_config": _identity(config_path),
        "merged_census_summary": _identity(census_summary_path),
        "census_worker_model_bindings": worker_binding_set,
        "statistics": {
            "source_eligible_token_count": 10_000,
            "candidate_window_count": document_count,
            "per_bundle": {
                str(level): {"B_passing_tokens": document_count * length}
                for level in (95, 97, 99)
            },
        },
    }
    manifest["manifest_content_sha256"] = _manifest_content_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    authoritative_path = source / "authoritative_b.npz"
    token_mask = np.ones((document_count, length), dtype=np.bool_)
    padded = np.zeros((document_count, 512), dtype=np.bool_)
    padded[:, :length] = token_mask
    layer_mask = np.repeat(padded[..., None], 8, axis=2)
    packed_token = np.packbits(padded, axis=1, bitorder="little")
    packed_layer = np.packbits(layer_mask, axis=1, bitorder="little")
    authoritative_metadata = {
        "schema": "cka_gt_authoritative_b_masks_v1",
        "complete": True,
        "raw_hidden_stored": False,
        "sealed_test_opened": False,
        "candidate_window_count": document_count,
        "candidate_windows": _identity(candidate_path),
        "candidate_manifest": {
            **_identity(manifest_path),
            "manifest_content_sha256": manifest["manifest_content_sha256"],
        },
        "analysis_config": _identity(config_path),
        "merged_census_summary": _identity(census_summary_path),
        "census_worker_model_binding_set_sha256": worker_binding_set[
            "binding_set_content_sha256"
        ],
        "bundle_b_token_counts": {
            str(level): document_count * length for level in (95, 97, 99)
        },
    }
    arrays = {
        "metadata_json": np.frombuffer(
            json.dumps(authoritative_metadata).encode(), dtype=np.uint8
        )
    }
    for level in (95, 97, 99):
        arrays[f"bundle_{level}_token_packed"] = packed_token
        for scale in (128, 256):
            arrays[f"bundle_{level}_scale_{scale}_pass_packed"] = packed_layer
    for scale in (128, 256):
        arrays[f"scale_{scale}_valid_packed"] = packed_layer
    np.savez(authoritative_path, **arrays)

    model_binding_path = source / "targeted_model_binding.json"
    model_binding_path.write_text(
        json.dumps(
            {
                "schema": "cka_gt_exact_targeted_model_binding_v1",
                "candidate_windows": _identity(candidate_path),
                "candidate_manifest": _identity(manifest_path),
                "analysis_config": _identity(config_path),
                "census_summary": _identity(census_summary_path),
                "authoritative_b": _identity(authoritative_path),
                "reference_checkpoint_identity": reference_identity,
                "current_checkpoint_identity": current_identity,
                "source_dataset_identity": source_dataset_identity,
                "batch_size": 4,
                "checkpoint_every_batches": 1,
            }
        ),
        encoding="utf-8",
    )

    exact_dir = root / "exact"
    writer = ExactTargetedGTWriter(
        output_dir=exact_dir,
        candidate_windows_path=candidate_path,
        candidate_manifest_path=manifest_path,
        analysis_config_path=config_path,
        census_summary_path=census_summary_path,
        model_binding_path=model_binding_path,
        batch_size=4,
        checkpoint_every_batches=1,
    )
    for batch_index in range(writer.expected_batch_count):
        indices = writer.plan[batch_index]
        batch_rows = np.asarray(writer.rows[indices])
        token_ids = torch.as_tensor(
            np.stack([documents[int(index)] for index in indices]), dtype=torch.long
        )
        writer.process_batch(
            batch_index=batch_index,
            candidate_indices=indices,
            rows=batch_rows,
            token_ids=token_ids,
            metrics=_metrics(len(indices), length),
        )
    writer.finalize()

    threshold_dir = root / "threshold_review"
    threshold_dir.mkdir()
    count = document_count * length
    threshold_dir.joinpath("analysis.json").write_text(
        json.dumps(
            {
                "schema": "cka_gt_full_census_threshold_review_v1",
                "complete": True,
                "threshold_review_only": True,
                "exact_gt_created": False,
                "sealed_test_opened": False,
                "full_eligible_tokens": 10_000,
                "reservoir_rows": 5_000,
                "candidates": {
                    str(level): {"counts": {"selected": 100 + level}}
                    for level in (95, 97, 99)
                },
                "document_bootstrap": {
                    str(level): {
                        "estimated_full_count": count,
                        "estimated_full_count_ci95": [count - 10, count + 10],
                    }
                    for level in (95, 97, 99)
                },
            }
        ),
        encoding="utf-8",
    )
    threshold_dir.joinpath("REPORT.md").write_text("synthetic\n", encoding="utf-8")
    return exact_dir, threshold_dir, dataset


class MetricTest(unittest.TestCase):
    def test_exact_concentration_metrics(self) -> None:
        result = concentration_metrics(np.asarray([1, 2, 3, 3, 4, 4, 4, 4]))
        self.assertEqual(result["active_units"], 4)
        self.assertAlmostEqual(result["max_share"], 0.5)
        self.assertAlmostEqual(result["top5_share"], 1.0)
        self.assertAlmostEqual(result["hhi"], 0.34375)
        self.assertGreater(result["gini_among_active"], 0.0)

    def test_token_entropy_and_within_window_repetition(self) -> None:
        result = token_repetition_metrics(
            np.asarray([7, 7, 8, 7]), np.asarray([1, 1, 1, 2])
        )
        self.assertEqual(result["unique_token_ids"], 2)
        self.assertAlmostEqual(result["top_token_share"], 0.75)
        self.assertAlmostEqual(result["within_window_excess_same_token_fraction"], 0.25)

    def test_recommendation_uses_strictest_passing_bundle(self) -> None:
        rows = []
        for level, count in ((95, 50), (97, 200), (99, 500)):
            rows.append(
                {
                    "bundle": level,
                    "exact_selected_occurrences": count,
                    "active_documents": 20,
                    "same_layer_jaccard": 0.95,
                    "max_document_share": 0.1,
                    "max_window_share": 0.02,
                    "top_token_share": 0.1,
                }
            )
        result = recommendation(
            rows,
            min_selected=100,
            min_documents=10,
            min_same_layer_jaccard=0.9,
            max_document_share=0.25,
            max_window_share=0.05,
            max_token_share=0.25,
        )
        self.assertEqual(result["recommended_bundle"], 97)
        self.assertTrue(result["bundle_95_is_strictest"])
        self.assertTrue(result["bundle_99_is_loosest"])


class IntegrationTest(unittest.TestCase):
    def test_full_review_verifies_exact_axis_and_keeps_test_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exact_dir, threshold_dir, dataset = _build_exact(root)
            output = root / "review"
            payload = run_review(
                exact_dir=exact_dir,
                threshold_review_dir=threshold_dir,
                dataset_prefix=root / "fake_dataset",
                output_dir=output,
                tokenizer_path=None,
                representative_contexts=3,
                top_units=3,
                min_documents=10,
                max_window_share=0.2,
                dataset=dataset,
                decoder=lambda ids: " ".join(str(item) for item in ids),
            )
            self.assertTrue(payload["validation"]["passed"])
            self.assertFalse(payload["sealed_test_opened"])
            self.assertEqual(payload["recommendation"]["recommended_bundle"], 95)
            self.assertEqual(
                payload["bundle_summary"][0]["exact_selected_occurrences"], 192
            )
            self.assertTrue(payload["bundle_summary"][0]["exact_count_inside_reservoir_ci95"])
            self.assertTrue((output / "REPORT.md").is_file())
            self.assertTrue((output / "tables" / "top_documents.csv").is_file())
            contexts = (output / "contexts" / "bundle_95.jsonl").read_text()
            self.assertIn('"token_identity_verified": true', contexts)

    def test_source_token_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exact_dir, threshold_dir, dataset = _build_exact(root)
            dataset.documents[0] = dataset.documents[0].copy()
            dataset.documents[0][0] += 1
            with self.assertRaisesRegex(RuntimeError, "token IDs differ"):
                run_review(
                    exact_dir=exact_dir,
                    threshold_review_dir=threshold_dir,
                    dataset_prefix=root / "fake_dataset",
                    output_dir=root / "review",
                    tokenizer_path=None,
                    representative_contexts=0,
                    top_units=1,
                    dataset=dataset,
                    decoder=lambda ids: "",
                )


if __name__ == "__main__":
    unittest.main()
