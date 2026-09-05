#!/usr/bin/env python3
"""CPU-only tests for deterministic CKA pilot document/window manifests."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from cka_gt_pilot_windows import (  # noqa: E402
    MMapIndexedDatasetLite,
    WINDOW_DTYPE,
    atomic_json,
    atomic_jsonl,
    atomic_npy,
    atomic_parquet,
    build_domain_window_manifest,
    checkpoint_identity,
    chunk_starts,
    document_lengths_from_indexed_dataset,
    eligible_token_count,
    enumerate_document_windows,
    file_sha256,
    load_document_window,
    payload_sha256,
    prepare_pilot,
    ratio_counts,
    sample_enumerated_windows,
    seeded_rng,
    split_document_ids,
    source_dataset_identity,
    upgrade_prepared_input_identities,
    validate_document_splits,
    validate_domain_manifest,
    validate_prepared_pilot,
    window_chunk_layout,
)


class FakeIndexedDataset:
    """Tiny IndexedDataset-compatible object, including multi-sequence docs."""

    def __init__(self, documents: list[list[np.ndarray]]):
        self._sequences: list[np.ndarray] = []
        document_indices = [0]
        for document in documents:
            self._sequences.extend(np.asarray(sequence) for sequence in document)
            document_indices.append(len(self._sequences))
        self.sequence_lengths = np.asarray(
            [sequence.size for sequence in self._sequences], dtype=np.int32
        )
        self.document_indices = np.asarray(document_indices, dtype=np.int64)

    @classmethod
    def from_lengths(cls, lengths: list[int]) -> "FakeIndexedDataset":
        cursor = 0
        documents: list[list[np.ndarray]] = []
        for length in lengths:
            sequence = np.arange(cursor, cursor + length, dtype=np.int32)
            cursor += length
            documents.append([sequence])
        return cls(documents)

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        sequence = self._sequences[int(idx)]
        if length is None:
            length = sequence.size - offset
        return sequence[int(offset) : int(offset) + int(length)]


class DocumentSplitTest(unittest.TestCase):
    def test_floor_counts_overlap_zero_and_complete_coverage(self) -> None:
        document_count = 103
        split = split_document_ids(document_count, base_seed=1234, domain_id=0)
        self.assertEqual(ratio_counts(document_count), {
            "calibration": 41,
            "selection": 30,
            "test": 32,
        })
        report = validate_document_splits(split, document_count)
        self.assertTrue(report["complete_coverage"])
        self.assertEqual(set(report["pairwise_overlap"].values()), {0})
        combined = np.concatenate([split[name] for name in ("calibration", "selection", "test")])
        np.testing.assert_array_equal(np.sort(combined), np.arange(document_count))

    def test_deterministic_and_domain_streams_are_independent(self) -> None:
        first = split_document_ids(200, base_seed=1234, domain_id=0)
        second = split_document_ids(200, base_seed=1234, domain_id=0)
        other_domain = split_document_ids(200, base_seed=1234, domain_id=1)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertFalse(
            np.array_equal(first["calibration"], other_domain["calibration"])
        )

        stream_a = seeded_rng(base_seed=1234, domain_id=0, stage_id=1, split_id=0)
        stream_b = seeded_rng(base_seed=1234, domain_id=0, stage_id=1, split_id=0)
        stream_c = seeded_rng(base_seed=1234, domain_id=0, stage_id=1, split_id=1)
        np.testing.assert_array_equal(stream_a.integers(0, 2**31, 20), stream_b.integers(0, 2**31, 20))
        self.assertFalse(
            np.array_equal(stream_a.integers(0, 2**31, 20), stream_c.integers(0, 2**31, 20))
        )

    def test_overlap_is_rejected(self) -> None:
        split = split_document_ids(10, base_seed=1234, domain_id=0)
        broken = {name: values.copy() for name, values in split.items()}
        broken["selection"][0] = broken["calibration"][0]
        with self.assertRaises(ValueError):
            validate_document_splits(broken, 10)


class TailAndChunkGridTest(unittest.TestCase):
    LENGTHS = np.asarray([255, 256, 257, 300, 511, 512, 513], dtype=np.int64)

    def test_exact_tail_boundaries_and_no_right_alignment(self) -> None:
        windows, stats = enumerate_document_windows(
            np.arange(self.LENGTHS.size), self.LENGTHS
        )
        observed = [
            (
                int(row["document_id"]),
                int(row["window_offset"]),
                int(row["window_length"]),
                int(row["eligible_token_count"]),
                int(row["ineligible_suffix_tokens"]),
            )
            for row in windows
        ]
        self.assertEqual(
            observed,
            [
                (1, 0, 256, 256, 0),
                (2, 0, 257, 256, 1),
                (3, 0, 300, 256, 44),
                (4, 0, 511, 384, 127),
                (5, 0, 512, 512, 0),
                (6, 0, 512, 512, 0),
            ],
        )
        self.assertEqual(stats["window_count"], 6)
        self.assertEqual(stats["full_window_count"], 2)
        self.assertEqual(stats["tail_window_count"], 4)
        self.assertEqual(stats["discarded_tail_token_count"], 256)
        self.assertEqual(stats["discarded_tail_document_count"], 2)
        self.assertEqual(stats["ineligible_suffix_tokens"], 172)
        # The 513-token document creates one fixed 0:512 window and discards
        # its 1-token tail; it must never create a right-aligned 1:513 window.
        doc_513 = windows[windows["document_id"] == 6]
        self.assertEqual(doc_513.size, 1)
        self.assertEqual(int(doc_513[0]["window_offset"]), 0)

    def test_chunk_starts_and_uncovered_suffix(self) -> None:
        expected = {
            # This document is discarded before forward; the pure grid helper
            # still reports the mathematical 128-token chunks it could hold.
            255: ([0, 64], [], 0),
            256: ([0, 64, 128], [0], 256),
            257: ([0, 64, 128], [0], 256),
            300: ([0, 64, 128], [0], 256),
            511: ([0, 64, 128, 192, 256, 320], [0, 128], 384),
            512: ([0, 64, 128, 192, 256, 320, 384], [0, 128, 256], 512),
            513: ([0, 64, 128, 192, 256, 320, 384], [0, 128, 256], 512),
        }
        for length, (starts_128, starts_256, eligible) in expected.items():
            with self.subTest(length=length):
                np.testing.assert_array_equal(
                    chunk_starts(length, 128, 64), np.asarray(starts_128, dtype=np.int32)
                )
                np.testing.assert_array_equal(
                    chunk_starts(length, 256, 128), np.asarray(starts_256, dtype=np.int32)
                )
                self.assertEqual(eligible_token_count(length), eligible)
                layout = window_chunk_layout(length)
                self.assertEqual(layout["eligible_token_count"], eligible)
                self.assertEqual(layout["ineligible_suffix_tokens"], length - eligible)

    def test_uniform_sample_is_deterministic_and_unique(self) -> None:
        lengths = np.full(20, 1024, dtype=np.int64)
        windows, _ = enumerate_document_windows(np.arange(20), lengths)
        first = sample_enumerated_windows(
            windows, 17, base_seed=1234, domain_id=0, split_name="selection"
        )
        second = sample_enumerated_windows(
            windows, 17, base_seed=1234, domain_id=0, split_name="selection"
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first["sample_order"], np.arange(17))
        self.assertEqual(np.unique(first["source_window_index"]).size, 17)


class IndexedDatasetAdapterTest(unittest.TestCase):
    def test_document_lengths_and_cross_sequence_read_within_one_document(self) -> None:
        dataset = FakeIndexedDataset(
            [
                [np.arange(0, 3, dtype=np.int32), np.arange(3, 8, dtype=np.int32)],
                [np.arange(100, 104, dtype=np.int32)],
            ]
        )
        np.testing.assert_array_equal(
            document_lengths_from_indexed_dataset(dataset), np.asarray([8, 4])
        )
        # Starts in sequence 0 and ends in sequence 1 of document 0.
        np.testing.assert_array_equal(
            load_document_window(dataset, 0, 2, 5), np.asarray([2, 3, 4, 5, 6])
        )
        with self.assertRaises(ValueError):
            load_document_window(dataset, 0, 7, 2)

    def test_cpu_only_mmap_reader_matches_megatron_v1_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            prefix = Path(temporary_dir) / "tiny_text_document"
            sequences = [
                np.asarray([10, 11, 12], dtype=np.int32),
                np.asarray([20, 21], dtype=np.int32),
                np.asarray([30, 31, 32, 33], dtype=np.int32),
            ]
            lengths = np.asarray([3, 2, 4], dtype=np.int32)
            pointers = np.asarray([0, 12, 20], dtype=np.int64)
            documents = np.asarray([0, 2, 3], dtype=np.int64)
            with prefix.with_suffix(".idx").open("wb") as handle:
                handle.write(b"MMIDIDX\x00\x00")
                handle.write(struct.pack("<Q", 1))
                handle.write(struct.pack("<B", 4))  # int32 tokens
                handle.write(struct.pack("<Q", len(sequences)))
                handle.write(struct.pack("<Q", documents.size))
                handle.write(lengths.tobytes())
                handle.write(pointers.tobytes())
                handle.write(documents.tobytes())
            with prefix.with_suffix(".bin").open("wb") as handle:
                for sequence in sequences:
                    handle.write(sequence.tobytes())

            dataset = MMapIndexedDatasetLite(str(prefix))
            np.testing.assert_array_equal(dataset.sequence_lengths, lengths)
            np.testing.assert_array_equal(dataset.document_indices, documents)
            np.testing.assert_array_equal(dataset.get(1), sequences[1])
            np.testing.assert_array_equal(dataset.get(2, 1, 2), [31, 32])
            np.testing.assert_array_equal(
                document_lengths_from_indexed_dataset(dataset), [5, 4]
            )
            np.testing.assert_array_equal(
                load_document_window(dataset, 0, 2, 3), [12, 20, 21]
            )


class AtomicManifestTest(unittest.TestCase):
    def test_atomic_json_npy_and_parquet_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            array = np.zeros(2, dtype=WINDOW_DTYPE)
            array["sample_order"] = [0, 1]
            array["source_window_index"] = [10, 20]
            array["document_id"] = [1, 2]
            array["window_length"] = 512
            array["eligible_token_count"] = 512
            json_path = root / "value.json"
            jsonl_path = root / "value.jsonl"
            npy_path = root / "value.npy"
            parquet_path = root / "value.parquet"
            atomic_json(json_path, {"b": 2, "a": [1, 3]})
            atomic_jsonl(jsonl_path, array)
            atomic_npy(npy_path, array)
            try:
                atomic_parquet(parquet_path, array)
            except RuntimeError as error:
                if "pyarrow" in str(error):
                    self.skipTest(str(error))
                raise
            self.assertEqual(json.loads(json_path.read_text()), {"a": [1, 3], "b": 2})
            self.assertEqual(len(jsonl_path.read_text().splitlines()), 2)
            np.testing.assert_array_equal(np.load(npy_path, allow_pickle=False), array)
            self.assertEqual(len(file_sha256(parquet_path)), 64)

    def test_build_validate_and_hash_tamper_detection(self) -> None:
        # Every split has ample windows even with a tiny requested pilot.
        dataset = FakeIndexedDataset.from_lengths([1024 + (index % 3) * 256 for index in range(30)])
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "code"
            payload = build_domain_window_manifest(
                dataset,
                dataset_prefix="/fake/code/train_text_document",
                output_dir=root,
                domain="code",
                domain_id=0,
                total_sample_windows=10,
                base_seed=1234,
                write_parquet=True,
                write_jsonl=True,
            )
            self.assertEqual(payload["requested_total_sample_windows"], 10)
            self.assertEqual(
                [payload["splits"][name]["sampled_window_count"] for name in ("calibration", "selection", "test")],
                [4, 3, 3],
            )
            validated = validate_domain_manifest(root / "manifest.json", dataset=dataset)
            self.assertEqual(validated["manifest_content_sha256"], payload["manifest_content_sha256"])
            # Existing complete output is a validated cache hit, not an overwrite.
            cached = build_domain_window_manifest(
                dataset,
                dataset_prefix="/fake/code/train_text_document",
                output_dir=root,
                domain="code",
                domain_id=0,
                total_sample_windows=10,
                base_seed=1234,
                write_parquet=True,
                write_jsonl=True,
            )
            self.assertEqual(cached["manifest_content_sha256"], payload["manifest_content_sha256"])

            selection_file = root / "selection_documents.npy"
            original = selection_file.read_bytes()
            selection_file.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_domain_manifest(root / "manifest.json", dataset=dataset)

    def test_prepare_canonical_tree_and_validate(self) -> None:
        datasets = {
            "code": FakeIndexedDataset.from_lengths([1024] * 30),
            "wiki": FakeIndexedDataset.from_lengths([1280] * 30),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "pilot"
            payload = prepare_pilot(
                root,
                "/fake/code/train_text_document",
                "/fake/wiki/train_text_document",
                1234,
                before_checkpoint="/fake/before",
                after_checkpoint="/fake/after",
                total_sample_windows_per_domain=10,
                datasets=datasets,
            )
            self.assertEqual(payload["schema"], "cka_gt_pilot_prepared_inputs_v1")
            self.assertTrue((root / "config.json").is_file())
            for domain in ("code", "wiki"):
                for split in ("calibration", "selection", "test"):
                    base = root / "splits" / f"{domain}_{split}"
                    self.assertTrue(
                        base.with_name(base.name + "_documents.npy").is_file()
                    )
                    self.assertTrue(base.with_name(base.name + "_windows.npy").is_file())
                    self.assertTrue(
                        base.with_name(base.name + "_windows.parquet").is_file()
                    )
                    self.assertTrue(
                        base.with_name(base.name + "_windows.jsonl").is_file()
                    )
            validated = validate_prepared_pilot(root, datasets=datasets)
            self.assertEqual(
                validated["config_content_sha256"], payload["config_content_sha256"]
            )

    def test_exact_dataset_and_checkpoint_identity_detect_same_size_mutation(self) -> None:
        datasets = {
            "code": FakeIndexedDataset.from_lengths([1024] * 30),
            "wiki": FakeIndexedDataset.from_lengths([1280] * 30),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            base = Path(temporary_dir)
            prefixes = {}
            for domain in ("code", "wiki"):
                prefix = base / "data" / domain / "train_text_document"
                prefix.parent.mkdir(parents=True, exist_ok=True)
                Path(str(prefix) + ".idx").write_bytes((domain + "-idx").encode())
                Path(str(prefix) + ".bin").write_bytes((domain + "-bin").encode())
                prefixes[domain] = prefix
            checkpoints = {}
            for name, step in (("before", 600), ("after", 1800)):
                checkpoint = base / "checkpoints" / name
                iteration = checkpoint / f"iter_{step:07d}"
                iteration.mkdir(parents=True)
                (checkpoint / "latest_checkpointed_iteration.txt").write_text(
                    f"{step}\n", encoding="utf-8"
                )
                (iteration / "common.pt").write_bytes(b"common-weights")
                (iteration / "__0_0.distcp").write_bytes(b"shard-weights")
                (iteration / ".metadata").write_bytes(b"metadata")
                checkpoints[name] = checkpoint

            root = base / "pilot"
            payload = prepare_pilot(
                root,
                str(prefixes["code"]),
                str(prefixes["wiki"]),
                1234,
                before_checkpoint=str(checkpoints["before"]),
                after_checkpoint=str(checkpoints["after"]),
                total_sample_windows_per_domain=10,
                datasets=datasets,
            )
            self.assertEqual(
                payload["source_dataset_identity"]["code"]["storage_kind"],
                "physical_indexed_dataset_files",
            )
            self.assertEqual(
                payload["checkpoint_identity"]["before"]["tracker_step"], 600
            )
            validate_prepared_pilot(root, datasets=datasets)

            code_bin = Path(str(prefixes["code"]) + ".bin")
            original_bin = code_bin.read_bytes()
            code_bin.write_bytes(bytes([original_bin[0] ^ 1]) + original_bin[1:])
            self.assertEqual(code_bin.stat().st_size, len(original_bin))
            with self.assertRaisesRegex(ValueError, "source_dataset_identity"):
                validate_prepared_pilot(root, datasets=datasets)
            code_bin.write_bytes(original_bin)

            shard = checkpoints["after"] / "iter_0001800" / "__0_0.distcp"
            original_shard = shard.read_bytes()
            shard.write_bytes(bytes([original_shard[0] ^ 1]) + original_shard[1:])
            self.assertEqual(shard.stat().st_size, len(original_shard))
            with self.assertRaisesRegex(ValueError, "checkpoint_identity"):
                validate_prepared_pilot(root, datasets=datasets)

            # Direct helpers are deterministic and content-addressed.
            shard.write_bytes(original_shard)
            self.assertEqual(
                source_dataset_identity(prefixes["wiki"]),
                payload["source_dataset_identity"]["wiki"],
            )
            self.assertEqual(
                checkpoint_identity(checkpoints["after"]),
                payload["checkpoint_identity"]["after"],
            )

            frozen_window_hashes = {
                path.name: file_sha256(path)
                for path in (root / "splits").glob("*_windows.npy")
            }
            legacy_config = json.loads((root / "config.json").read_text())
            legacy_config.pop("source_dataset_identity")
            legacy_config.pop("checkpoint_identity")
            for domain in ("code", "wiki"):
                manifest_path = root / "splits" / f"{domain}_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest.pop("source_dataset_identity")
                manifest.pop("manifest_content_sha256")
                manifest["manifest_content_sha256"] = payload_sha256(manifest)
                atomic_json(manifest_path, manifest)
                legacy_config["artifacts"][f"{domain}_manifest"].update(
                    {
                        "bytes": manifest_path.stat().st_size,
                        "sha256": file_sha256(manifest_path),
                    }
                )
            legacy_config.pop("config_content_sha256")
            legacy_config["config_content_sha256"] = payload_sha256(legacy_config)
            atomic_json(root / "config.json", legacy_config)

            upgraded = upgrade_prepared_input_identities(root, datasets=datasets)
            self.assertIn("source_dataset_identity", upgraded)
            self.assertIn("checkpoint_identity", upgraded)
            self.assertEqual(
                frozen_window_hashes,
                {
                    path.name: file_sha256(path)
                    for path in (root / "splits").glob("*_windows.npy")
                },
            )


if __name__ == "__main__":
    unittest.main()
