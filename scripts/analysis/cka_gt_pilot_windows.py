#!/usr/bin/env python3
"""Deterministic, document-bounded window manifests for the CKA GT pilot.

This module deliberately contains no model code.  It turns Megatron
``IndexedDataset`` document metadata into three disjoint document splits and
uniformly sampled, non-overlapping 512-token windows.  A retained tail is
never shifted to the right: chunks always start on their fixed 64/128-token
grids, and any uncovered suffix is marked ineligible.

The public helpers are also usable by the paired-forward driver:

* :func:`build_domain_window_manifest` writes a complete split manifest;
* :func:`load_document_window` retrieves one window without crossing a
  document boundary, including the generic multi-sequence-per-document case;
* :func:`chunk_starts` and :func:`window_chunk_layout` implement the exact
  chunk grids from the pilot specification.

Only metadata and sampled window coordinates are written.  Token IDs and raw
hidden states are not materialized by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cka_gt_pilot_document_windows_v1"
BASE_SEED = 1234
WINDOW_LENGTH = 512
MIN_TAIL_LENGTH = 256
SPLIT_NAMES = ("calibration", "selection", "test")
SPLIT_IDS = {name: index for index, name in enumerate(SPLIT_NAMES)}
DOMAIN_IDS = {"code": 0, "wiki": 1}
STAGE_DOCUMENT_SPLIT = 0
STAGE_WINDOW_SAMPLE = 1
STAGE_COVARIANCE_SUBSAMPLE = 2
STAGE_KMEANS_RESERVOIR = 3
STAGE_PERMUTATION_NULL = 4
STAGE_RANDOM_PAIR_NULL = 5
STAGE_MATCHED_RANDOM = 6
CHUNK_SPECS = {128: 64, 256: 128}

DEFAULT_OUTPUT_ROOT = Path(
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
    "cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1"
)
DEFAULT_CODE_PREFIX = (
    "/data2/seonghyeonnoh/LLM-continual-learning-data/"
    "flamedata2.data2-verified-backup/code/train/train_text_document"
)
DEFAULT_WIKI_PREFIX = (
    "/data2/seonghyeonnoh/LLM-continual-learning-data/"
    "flamedata2.data2-verified-backup/wiki/train/train_text_document"
)
DEFAULT_BEFORE_CHECKPOINT = (
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/"
    "ffn_experts_only/code/expansion_kd_init/kd_init/full_training/"
    "g2_olddata_kd_9run_20260808__code_e8_to_e16_wiki_kd_step600"
)
DEFAULT_AFTER_CHECKPOINT = (
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/"
    "ffn_experts_only/code/no_replay/lm/full_training/"
    "flame_code_bootstrap_20260810__"
    "g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt-v2"
)

WINDOW_DTYPE = np.dtype(
    [
        ("sample_order", "<i8"),
        ("source_window_index", "<i8"),
        ("document_id", "<i8"),
        ("window_offset", "<i8"),
        ("window_length", "<i4"),
        ("eligible_token_count", "<i4"),
        ("ineligible_suffix_tokens", "<i4"),
        ("is_tail", "u1"),
    ]
)


class MMapIndexedDatasetLite:
    """CPU-only reader for Megatron's standard mmap IndexedDataset format.

    Importing ``megatron.core`` eagerly imports Transformer Engine in this
    repository, which may initialize CUDA even when only ``.idx`` metadata is
    needed.  This small reader follows the same on-disk v1 format and exposes
    exactly the ``sequence_lengths``, ``document_indices``, and ``get`` API
    used by this module.  It never imports torch or touches a GPU.
    """

    _HEADER = b"MMIDIDX\x00\x00"
    _DTYPES = {
        1: np.dtype("u1"),
        2: np.dtype("i1"),
        3: np.dtype("<i2"),
        4: np.dtype("<i4"),
        5: np.dtype("<i8"),
        6: np.dtype("<f8"),
        7: np.dtype("<f4"),
        8: np.dtype("<u2"),
    }

    def __init__(self, path_prefix: str):
        prefix = str(path_prefix)
        if prefix.endswith(".idx") or prefix.endswith(".bin"):
            prefix = prefix.rsplit(".", 1)[0]
        self.path_prefix = prefix
        self.idx_path = Path(prefix + ".idx")
        self.bin_path = Path(prefix + ".bin")
        if not self.idx_path.is_file() or not self.bin_path.is_file():
            raise FileNotFoundError(
                f"IndexedDataset requires both {self.idx_path} and {self.bin_path}"
            )
        with self.idx_path.open("rb") as handle:
            if handle.read(9) != self._HEADER:
                raise ValueError(f"bad IndexedDataset header: {self.idx_path}")
            version = struct.unpack("<Q", handle.read(8))[0]
            if version != 1:
                raise ValueError(f"unsupported IndexedDataset version {version}")
            dtype_code = struct.unpack("<B", handle.read(1))[0]
            if dtype_code not in self._DTYPES:
                raise ValueError(f"unsupported IndexedDataset dtype code {dtype_code}")
            self.dtype = self._DTYPES[dtype_code]
            sequence_count = struct.unpack("<Q", handle.read(8))[0]
            document_index_count = struct.unpack("<Q", handle.read(8))[0]
            offset = handle.tell()

        self._idx_mmap = np.memmap(self.idx_path, mode="r", order="C")
        buffer = memoryview(self._idx_mmap)
        self.sequence_lengths = np.frombuffer(
            buffer, dtype=np.dtype("<i4"), count=sequence_count, offset=offset
        )
        pointer_offset = offset + self.sequence_lengths.nbytes
        self.sequence_pointers = np.frombuffer(
            buffer,
            dtype=np.dtype("<i8"),
            count=sequence_count,
            offset=pointer_offset,
        )
        document_offset = pointer_offset + self.sequence_pointers.nbytes
        self.document_indices = np.frombuffer(
            buffer,
            dtype=np.dtype("<i8"),
            count=document_index_count,
            offset=document_offset,
        )
        if self.document_indices.size == 0:
            raise ValueError("IndexedDataset document index is empty")
        if int(self.document_indices[-1]) != int(sequence_count):
            raise ValueError("IndexedDataset document index does not end at sequence count")
        expected_bin_bytes = 0
        if sequence_count:
            expected_bin_bytes = int(self.sequence_pointers[-1]) + int(
                self.sequence_lengths[-1]
            ) * self.dtype.itemsize
        if self.bin_path.stat().st_size < expected_bin_bytes:
            raise ValueError("IndexedDataset bin file is truncated")
        self._bin_mmap = np.memmap(self.bin_path, mode="r", order="C")

    def __len__(self) -> int:
        return int(self.sequence_lengths.size)

    def get(
        self, idx: int, offset: int = 0, length: int | None = None
    ) -> np.ndarray:
        idx = int(idx)
        offset = int(offset)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"sequence index out of range: {idx}")
        sequence_length = int(self.sequence_lengths[idx])
        if length is None:
            length = sequence_length - offset
        length = int(length)
        if offset < 0 or length < 0 or offset + length > sequence_length:
            raise ValueError(
                f"invalid sequence slice: idx={idx}, offset={offset}, length={length}, "
                f"sequence_length={sequence_length}"
            )
        byte_offset = int(self.sequence_pointers[idx]) + offset * self.dtype.itemsize
        return np.frombuffer(
            self._bin_mmap, dtype=self.dtype, count=length, offset=byte_offset
        )


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays recursively for deterministic JSON."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolved_dataset_prefix(dataset_prefix: str | Path) -> Path:
    value = str(dataset_prefix)
    if value.endswith(".idx") or value.endswith(".bin"):
        value = value.rsplit(".", 1)[0]
    return Path(value).resolve()


def source_dataset_identity(
    dataset_prefix: str | Path, *, dataset: Any | None = None
) -> dict[str, Any]:
    """Return an exact content identity for an IndexedDataset source.

    Production identities hash the complete ``.idx`` and ``.bin`` files.  The
    deterministic in-memory representation exists only for injected CPU-test
    datasets whose fake prefixes have no backing files.
    """

    prefix = _resolved_dataset_prefix(dataset_prefix)
    idx_path = Path(str(prefix) + ".idx")
    bin_path = Path(str(prefix) + ".bin")
    if idx_path.is_file() and bin_path.is_file():
        return {
            "schema": "cka_gt_pilot_source_dataset_identity_v1",
            "storage_kind": "physical_indexed_dataset_files",
            "resolved_prefix": str(prefix),
            "idx": {
                "path": str(idx_path.resolve()),
                "size_bytes": int(idx_path.stat().st_size),
                "sha256": file_sha256(idx_path),
            },
            "bin": {
                "path": str(bin_path.resolve()),
                "size_bytes": int(bin_path.stat().st_size),
                "sha256": file_sha256(bin_path),
            },
        }
    if dataset is None:
        raise FileNotFoundError(
            f"IndexedDataset identity requires both files: {idx_path}, {bin_path}"
        )

    sequence_lengths = np.asarray(dataset.sequence_lengths, dtype="<i8")
    document_indices = np.asarray(dataset.document_indices, dtype="<i8")
    idx_digest = hashlib.sha256()
    idx_digest.update(sequence_lengths.tobytes())
    idx_digest.update(document_indices.tobytes())
    bin_digest = hashlib.sha256()
    logical_bytes = 0
    for sequence_index in range(int(sequence_lengths.size)):
        tokens = np.asarray(dataset.get(sequence_index), dtype="<i8").reshape(-1)
        bin_digest.update(tokens.tobytes())
        logical_bytes += int(tokens.nbytes)
    return {
        "schema": "cka_gt_pilot_source_dataset_identity_v1",
        "storage_kind": "injected_dataset_content_for_cpu_test",
        "resolved_prefix": str(prefix),
        "idx": {
            "path": None,
            "size_bytes": int(sequence_lengths.nbytes + document_indices.nbytes),
            "sha256": idx_digest.hexdigest(),
        },
        "bin": {
            "path": None,
            "size_bytes": logical_bytes,
            "sha256": bin_digest.hexdigest(),
        },
    }


def checkpoint_identity(
    checkpoint_root: str | Path, *, allow_missing_for_cpu_test: bool = False
) -> dict[str, Any]:
    """Hash the tracker and every file in its selected iteration directory."""

    root = Path(checkpoint_root).resolve()
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        if allow_missing_for_cpu_test:
            return {
                "schema": "cka_gt_pilot_checkpoint_identity_v1",
                "storage_kind": "unavailable_injected_cpu_test",
                "resolved_root": str(root),
                "tracker_step": None,
                "iteration_dir": None,
                "files": [],
                "total_bytes": 0,
                "content_sha256": payload_sha256(
                    {"resolved_root": str(root), "cpu_test_placeholder": True}
                ),
            }
        raise FileNotFoundError(f"checkpoint tracker is missing: {tracker}")
    tracker_text = tracker.read_text(encoding="utf-8").strip()
    try:
        step = int(tracker_text)
    except ValueError as error:
        raise ValueError(f"checkpoint tracker is not an integer: {tracker_text!r}") from error
    matches = sorted(
        path
        for path in root.glob("iter_*")
        if path.is_dir()
        and path.name.removeprefix("iter_").isdigit()
        and int(path.name.removeprefix("iter_")) == step
    )
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint step {step} resolves to {len(matches)} iteration directories"
        )
    iteration_dir = matches[0]
    files = [tracker] + sorted(
        (path for path in iteration_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if len(files) <= 1:
        raise ValueError(f"checkpoint iteration directory is empty: {iteration_dir}")
    records = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    identity: dict[str, Any] = {
        "schema": "cka_gt_pilot_checkpoint_identity_v1",
        "storage_kind": "megatron_tracker_and_iteration_files",
        "resolved_root": str(root),
        "tracker_step": step,
        "iteration_dir": iteration_dir.relative_to(root).as_posix(),
        "files": records,
        "total_bytes": sum(record["size_bytes"] for record in records),
    }
    identity["content_sha256"] = payload_sha256(identity)
    return identity


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON after checking that it round-trips."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if loaded != _jsonable(payload):
            raise RuntimeError(f"JSON round-trip mismatch: {path}")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_npy(path: Path, value: np.ndarray) -> None:
    """Atomically write an object-free NPY array with round-trip validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        check = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if check.shape != value.shape or check.dtype != value.dtype:
            raise RuntimeError(f"NPY shape/dtype round-trip mismatch: {path}")
        if not np.array_equal(check, value):
            raise RuntimeError(f"NPY value round-trip mismatch: {path}")
        del check
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _structured_columns(value: np.ndarray) -> dict[str, np.ndarray]:
    if value.dtype.names is None:
        raise ValueError("Parquet manifest values must be a structured array")
    return {name: np.asarray(value[name]) for name in value.dtype.names}


def atomic_parquet(path: Path, value: np.ndarray) -> None:
    """Atomically write a structured array as Parquet.

    PyArrow is imported lazily so pure window/split unit tests do not require
    it unless they exercise the Parquet artifact path.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on runtime env
        raise RuntimeError("pyarrow is required to write Parquet manifests") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    columns = _structured_columns(value)
    try:
        table = pa.table(columns)
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        check = pq.read_table(temporary)
        if check.num_rows != value.shape[0] or check.column_names != list(columns):
            raise RuntimeError(f"Parquet schema/row round-trip mismatch: {path}")
        for name, expected in columns.items():
            actual = np.asarray(check[name].combine_chunks().to_numpy(zero_copy_only=False))
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"Parquet column round-trip mismatch: {path}:{name}")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_jsonl(path: Path, value: np.ndarray) -> None:
    """Atomically write a structured manifest array as one JSON object per row."""

    columns = _structured_columns(value)
    names = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            for row_index in range(value.size):
                row = {name: columns[name][row_index].item() for name in names}
                handle.write(canonical_json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        row_count = 0
        with temporary.open("r", encoding="utf-8") as handle:
            for row_count, line in enumerate(handle, start=1):
                loaded = json.loads(line)
                expected = {
                    name: columns[name][row_count - 1].item() for name in names
                }
                if loaded != expected:
                    raise RuntimeError(
                        f"JSONL row round-trip mismatch: {path}:{row_count - 1}"
                    )
        if row_count != value.size:
            raise RuntimeError(
                f"JSONL row-count mismatch: {path}: {row_count} != {value.size}"
            )
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def seeded_rng(
    *, base_seed: int, domain_id: int, stage_id: int, split_id: int = 0
) -> np.random.Generator:
    """Return an independent PCG64 stream with an explicit SeedSequence."""

    entropy = [int(base_seed), int(domain_id), int(stage_id), int(split_id)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def ratio_counts(total: int) -> dict[str, int]:
    """40/30/30 counts with floor for the first two and remainder for test."""

    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    calibration = (int(total) * 40) // 100
    selection = (int(total) * 30) // 100
    return {
        "calibration": calibration,
        "selection": selection,
        "test": int(total) - calibration - selection,
    }


def split_document_ids(
    document_count: int, *, base_seed: int = BASE_SEED, domain_id: int
) -> dict[str, np.ndarray]:
    """Shuffle dense document IDs once, then partition them 40/30/30."""

    if document_count < 0:
        raise ValueError("document_count must be non-negative")
    rng = seeded_rng(
        base_seed=base_seed,
        domain_id=domain_id,
        stage_id=STAGE_DOCUMENT_SPLIT,
        split_id=0,
    )
    permutation = rng.permutation(document_count).astype(np.int64, copy=False)
    counts = ratio_counts(document_count)
    first = counts["calibration"]
    second = first + counts["selection"]
    result = {
        "calibration": permutation[:first].copy(),
        "selection": permutation[first:second].copy(),
        "test": permutation[second:].copy(),
    }
    validate_document_splits(result, document_count)
    return result


def validate_document_splits(
    splits: Mapping[str, np.ndarray], document_count: int
) -> dict[str, Any]:
    """Assert exact document coverage and zero overlap across all splits."""

    if tuple(splits.keys()) != SPLIT_NAMES:
        raise ValueError(f"split keys/order must be {SPLIT_NAMES}, got {tuple(splits)}")
    arrays = [np.asarray(splits[name], dtype=np.int64) for name in SPLIT_NAMES]
    expected_counts = ratio_counts(document_count)
    for name, values in zip(SPLIT_NAMES, arrays):
        if values.ndim != 1:
            raise ValueError(f"{name} document IDs must be one-dimensional")
        if values.size != expected_counts[name]:
            raise ValueError(
                f"{name} document count mismatch: {values.size} != {expected_counts[name]}"
            )
        if np.any(values < 0) or np.any(values >= document_count):
            raise ValueError(f"{name} contains an out-of-range document ID")
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} contains duplicate document IDs")

    combined = np.concatenate(arrays) if document_count else np.empty(0, dtype=np.int64)
    if combined.size != document_count:
        raise ValueError("document splits do not cover the expected number of documents")
    if not np.array_equal(np.sort(combined), np.arange(document_count, dtype=np.int64)):
        raise ValueError("document splits overlap or omit document IDs")

    pairwise_overlap: dict[str, int] = {}
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            count = int(
                np.intersect1d(
                    splits[left_name], splits[right_name], assume_unique=True
                ).size
            )
            pairwise_overlap[f"{left_name}__{right_name}"] = count
            if count:
                raise ValueError(
                    f"document overlap between {left_name} and {right_name}: {count}"
                )
    return {
        "complete_coverage": True,
        "pairwise_overlap": pairwise_overlap,
        "total_unique_documents": int(document_count),
    }


def document_lengths_from_indexed_dataset(dataset: Any) -> np.ndarray:
    """Compute one concatenated token length per IndexedDataset document."""

    sequence_lengths = np.asarray(dataset.sequence_lengths, dtype=np.int64)
    document_indices = np.asarray(dataset.document_indices, dtype=np.int64)
    if document_indices.ndim != 1 or document_indices.size < 1:
        raise ValueError("document_indices must be a non-empty one-dimensional array")
    if int(document_indices[0]) != 0 or int(document_indices[-1]) != sequence_lengths.size:
        raise ValueError("document_indices must start at 0 and end at sequence_count")
    if np.any(np.diff(document_indices) < 0):
        raise ValueError("document_indices must be non-decreasing")
    prefix = np.empty(sequence_lengths.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(sequence_lengths, dtype=np.int64, out=prefix[1:])
    lengths = prefix[document_indices[1:]] - prefix[document_indices[:-1]]
    if np.any(lengths < 0):
        raise AssertionError("negative document length")
    return lengths


def load_document_window(
    dataset: Any,
    document_id: int,
    window_offset: int,
    window_length: int,
) -> np.ndarray:
    """Read a contiguous window, never crossing the selected document.

    The production datasets currently contain one IndexedDataset sequence per
    document, but supporting multiple sequences here makes the boundary
    guarantee explicit instead of relying on that incidental layout.
    """

    document_indices = np.asarray(dataset.document_indices, dtype=np.int64)
    sequence_lengths = np.asarray(dataset.sequence_lengths, dtype=np.int64)
    document_count = document_indices.size - 1
    document_id = int(document_id)
    window_offset = int(window_offset)
    window_length = int(window_length)
    if document_id < 0 or document_id >= document_count:
        raise IndexError(f"document_id out of range: {document_id}")
    if window_offset < 0 or window_length < 0:
        raise ValueError("window offset/length must be non-negative")

    sequence_start = int(document_indices[document_id])
    sequence_stop = int(document_indices[document_id + 1])
    lengths = sequence_lengths[sequence_start:sequence_stop]
    document_length = int(lengths.sum(dtype=np.int64))
    if window_offset + window_length > document_length:
        raise ValueError(
            f"window [{window_offset}, {window_offset + window_length}) exceeds "
            f"document {document_id} length {document_length}"
        )
    if window_length == 0:
        return np.empty(0, dtype=np.int64)

    pieces: list[np.ndarray] = []
    remaining_offset = window_offset
    remaining_length = window_length
    for sequence_id, sequence_length_value in zip(
        range(sequence_start, sequence_stop), lengths.tolist()
    ):
        sequence_length = int(sequence_length_value)
        if remaining_offset >= sequence_length:
            remaining_offset -= sequence_length
            continue
        take = min(sequence_length - remaining_offset, remaining_length)
        piece = dataset.get(sequence_id, offset=remaining_offset, length=take)
        if isinstance(piece, tuple):
            piece = piece[0]
        pieces.append(np.asarray(piece))
        remaining_length -= take
        remaining_offset = 0
        if remaining_length == 0:
            break
    if remaining_length != 0:
        raise RuntimeError("failed to retrieve the requested document-bounded window")
    result = pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)
    if result.shape != (window_length,):
        raise RuntimeError(
            f"window read shape mismatch: {result.shape} != ({window_length},)"
        )
    return result


def chunk_starts(window_length: int, chunk_length: int, stride: int) -> np.ndarray:
    """Return only fixed-grid chunks fully contained in a window."""

    window_length = int(window_length)
    chunk_length = int(chunk_length)
    stride = int(stride)
    if window_length < 0 or chunk_length <= 0 or stride <= 0:
        raise ValueError("window_length >= 0, chunk_length > 0, stride > 0 required")
    if window_length < chunk_length:
        return np.empty(0, dtype=np.int32)
    return np.arange(0, window_length - chunk_length + 1, stride, dtype=np.int32)


def chunk_coverage_end(window_length: int, chunk_length: int, stride: int) -> int:
    starts = chunk_starts(window_length, chunk_length, stride)
    return 0 if starts.size == 0 else int(starts[-1]) + int(chunk_length)


def eligible_token_count(window_length: int) -> int:
    """Tokens covered by both the 128 and 256 fixed chunk grids."""

    ends = [
        chunk_coverage_end(window_length, chunk_length, stride)
        for chunk_length, stride in CHUNK_SPECS.items()
    ]
    return min(ends) if ends else 0


def window_chunk_layout(window_length: int) -> dict[int, dict[str, Any]]:
    """Describe exact 128/256 grids and the 512/tail control chunk."""

    window_length = int(window_length)
    layout: dict[int, dict[str, Any]] = {}
    for chunk_length, stride in CHUNK_SPECS.items():
        starts = chunk_starts(window_length, chunk_length, stride)
        layout[chunk_length] = {
            "chunk_length": chunk_length,
            "stride": stride,
            "starts": starts,
            "coverage_end": 0 if starts.size == 0 else int(starts[-1]) + chunk_length,
        }
    layout[512] = {
        "chunk_length": window_length,
        "stride": None,
        "starts": np.asarray([0], dtype=np.int32) if window_length > 0 else np.empty(0, np.int32),
        "coverage_end": window_length,
    }
    eligible = eligible_token_count(window_length)
    layout["eligible_token_count"] = eligible  # type: ignore[index]
    layout["ineligible_suffix_tokens"] = window_length - eligible  # type: ignore[index]
    return layout


def _window_counts_and_stats(
    document_ids: np.ndarray, document_lengths: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    selected_lengths = np.asarray(document_lengths, dtype=np.int64)[document_ids]
    full_counts = selected_lengths // WINDOW_LENGTH
    tails = selected_lengths % WINDOW_LENGTH
    keep_tail = tails >= MIN_TAIL_LENGTH
    discard_tail = (tails > 0) & ~keep_tail
    counts = full_counts + keep_tail.astype(np.int64)

    tail_lengths = tails[keep_tail]
    if tail_lengths.size:
        eligible_tail = np.asarray(
            [eligible_token_count(int(length)) for length in tail_lengths], dtype=np.int64
        )
        ineligible_suffix = tail_lengths - eligible_tail
    else:
        eligible_tail = np.empty(0, dtype=np.int64)
        ineligible_suffix = np.empty(0, dtype=np.int64)

    full_window_count = int(full_counts.sum(dtype=np.int64))
    tail_window_count = int(keep_tail.sum())
    retained_token_count = full_window_count * WINDOW_LENGTH + int(
        tail_lengths.sum(dtype=np.int64)
    )
    ineligible_count = int(ineligible_suffix.sum(dtype=np.int64))
    discarded_count = int(tails[discard_tail].sum(dtype=np.int64))
    stats = {
        "document_count": int(document_ids.size),
        "document_token_count": int(selected_lengths.sum(dtype=np.int64)),
        "documents_with_at_least_one_window": int(np.count_nonzero(counts)),
        "full_window_count": full_window_count,
        "tail_window_count": tail_window_count,
        "window_count": full_window_count + tail_window_count,
        "retained_token_count": retained_token_count,
        "eligible_token_count": retained_token_count - ineligible_count,
        "ineligible_suffix_tokens": ineligible_count,
        "ineligible_fraction_of_retained_tokens": (
            float(ineligible_count / retained_token_count) if retained_token_count else 0.0
        ),
        "ineligible_fraction_exceeds_5pct": bool(
            retained_token_count and ineligible_count / retained_token_count > 0.05
        ),
        "discarded_tail_token_count": discarded_count,
        "discarded_tail_document_count": int(discard_tail.sum()),
        "discarded_tail_fraction_of_document_tokens": (
            float(discarded_count / selected_lengths.sum(dtype=np.int64))
            if selected_lengths.sum(dtype=np.int64)
            else 0.0
        ),
    }
    return counts, stats


def enumerate_document_windows(
    document_ids: Sequence[int] | np.ndarray,
    document_lengths: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Enumerate all legal windows for the supplied documents.

    Windows are ordered by the supplied document order, then increasing
    512-aligned offset.  ``source_window_index`` identifies this pre-shuffle
    order and is retained in sampled manifests for auditing.
    """

    document_ids_array = np.asarray(document_ids, dtype=np.int64)
    document_lengths_array = np.asarray(document_lengths, dtype=np.int64)
    if document_ids_array.ndim != 1 or document_lengths_array.ndim != 1:
        raise ValueError("document IDs and lengths must be one-dimensional")
    if np.any(document_ids_array < 0) or np.any(document_ids_array >= document_lengths_array.size):
        raise ValueError("document ID out of range")
    if np.any(document_lengths_array < 0):
        raise ValueError("document lengths must be non-negative")
    if np.unique(document_ids_array).size != document_ids_array.size:
        raise ValueError("document_ids must be unique")

    counts, stats = _window_counts_and_stats(document_ids_array, document_lengths_array)
    result = np.empty(int(counts.sum(dtype=np.int64)), dtype=WINDOW_DTYPE)
    cursor = 0
    for document_id, count in zip(document_ids_array.tolist(), counts.tolist()):
        if count == 0:
            continue
        document_length = int(document_lengths_array[document_id])
        full_count = document_length // WINDOW_LENGTH
        for ordinal in range(int(count)):
            offset = ordinal * WINDOW_LENGTH
            length = WINDOW_LENGTH if ordinal < full_count else document_length % WINDOW_LENGTH
            eligible = eligible_token_count(length)
            result[cursor] = (
                -1,
                cursor,
                document_id,
                offset,
                length,
                eligible,
                length - eligible,
                int(length < WINDOW_LENGTH),
            )
            cursor += 1
    if cursor != result.size:
        raise AssertionError(f"window enumeration mismatch: {cursor} != {result.size}")
    validate_window_rows(result, document_lengths_array)
    return result, stats


def sample_enumerated_windows(
    windows: np.ndarray,
    sample_count: int,
    *,
    base_seed: int = BASE_SEED,
    domain_id: int,
    split_name: str,
) -> np.ndarray:
    """Uniformly shuffle all enumerated windows and take a deterministic prefix."""

    if windows.dtype != WINDOW_DTYPE:
        raise ValueError(f"unexpected window dtype: {windows.dtype}")
    if split_name not in SPLIT_IDS:
        raise ValueError(f"unknown split: {split_name}")
    sample_count = int(sample_count)
    if sample_count < 0 or sample_count > windows.size:
        raise ValueError(
            f"cannot sample {sample_count} windows from {windows.size} in {split_name}"
        )
    rng = seeded_rng(
        base_seed=base_seed,
        domain_id=domain_id,
        stage_id=STAGE_WINDOW_SAMPLE,
        split_id=SPLIT_IDS[split_name],
    )
    indices = rng.permutation(windows.size)[:sample_count]
    sampled = windows[indices].copy()
    sampled["sample_order"] = np.arange(sample_count, dtype=np.int64)
    if np.unique(sampled["source_window_index"]).size != sample_count:
        raise AssertionError("sampled window identities are not unique")
    return sampled


def sampled_window_statistics(windows: np.ndarray) -> dict[str, Any]:
    retained = int(windows["window_length"].sum(dtype=np.int64))
    ineligible = int(windows["ineligible_suffix_tokens"].sum(dtype=np.int64))
    return {
        "window_count": int(windows.size),
        "full_window_count": int(np.count_nonzero(windows["is_tail"] == 0)),
        "tail_window_count": int(np.count_nonzero(windows["is_tail"] != 0)),
        "retained_token_count": retained,
        "eligible_token_count": retained - ineligible,
        "ineligible_suffix_tokens": ineligible,
        "ineligible_fraction_of_retained_tokens": (
            float(ineligible / retained) if retained else 0.0
        ),
        "ineligible_fraction_exceeds_5pct": bool(retained and ineligible / retained > 0.05),
    }


def validate_window_rows(windows: np.ndarray, document_lengths: np.ndarray) -> None:
    """Validate fixed-grid origin, document bounds, and suffix eligibility."""

    if windows.dtype != WINDOW_DTYPE:
        raise ValueError(f"unexpected window dtype: {windows.dtype}")
    document_lengths = np.asarray(document_lengths, dtype=np.int64)
    if windows.size == 0:
        return
    document_ids = windows["document_id"]
    if np.any(document_ids < 0) or np.any(document_ids >= document_lengths.size):
        raise ValueError("window manifest contains out-of-range document ID")
    if np.any(windows["window_offset"] < 0) or np.any(
        windows["window_offset"] % WINDOW_LENGTH != 0
    ):
        raise ValueError("window offsets must lie on the fixed 512-token grid")
    lengths = windows["window_length"].astype(np.int64)
    if np.any(lengths < MIN_TAIL_LENGTH) or np.any(lengths > WINDOW_LENGTH):
        raise ValueError("window length must be in [256, 512]")
    doc_lengths = document_lengths[document_ids]
    if np.any(windows["window_offset"] + lengths > doc_lengths):
        raise ValueError("window crosses a document boundary")

    is_tail = lengths < WINDOW_LENGTH
    if not np.array_equal(windows["is_tail"].astype(bool), is_tail):
        raise ValueError("is_tail does not agree with window length")
    if np.any(is_tail):
        tail_offsets = (doc_lengths[is_tail] // WINDOW_LENGTH) * WINDOW_LENGTH
        tail_lengths = doc_lengths[is_tail] % WINDOW_LENGTH
        if not np.array_equal(windows["window_offset"][is_tail], tail_offsets):
            raise ValueError("tail was right-aligned or is not on the fixed 512 grid")
        if not np.array_equal(lengths[is_tail], tail_lengths):
            raise ValueError("tail length does not equal the document remainder")

    expected_eligible = np.asarray(
        [eligible_token_count(int(length)) for length in lengths], dtype=np.int32
    )
    if not np.array_equal(windows["eligible_token_count"], expected_eligible):
        raise ValueError("eligible-token count mismatch")
    expected_suffix = lengths.astype(np.int32) - expected_eligible
    if not np.array_equal(windows["ineligible_suffix_tokens"], expected_suffix):
        raise ValueError("ineligible suffix count mismatch")


def _artifact(path: Path, *, rows: int, kind: str) -> dict[str, Any]:
    return {
        "file": path.name,
        "format": kind,
        "rows": int(rows),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def build_domain_window_manifest(
    dataset: Any,
    *,
    dataset_prefix: str,
    output_dir: Path,
    domain: str,
    domain_id: int,
    total_sample_windows: int = 100_000,
    base_seed: int = BASE_SEED,
    write_parquet: bool = True,
    write_jsonl: bool = True,
    artifact_prefix: str = "",
    manifest_filename: str = "manifest.json",
) -> dict[str, Any]:
    """Build and atomically write one domain's split/window artifacts."""

    output_dir = Path(output_dir)
    if artifact_prefix and not artifact_prefix.replace("_", "").isalnum():
        raise ValueError(f"unsafe artifact prefix: {artifact_prefix!r}")
    if Path(manifest_filename).name != manifest_filename:
        raise ValueError(f"manifest filename must be a basename: {manifest_filename}")
    manifest_path = output_dir / manifest_filename
    source_identity = source_dataset_identity(dataset_prefix, dataset=dataset)
    if manifest_path.exists():
        cached = validate_domain_manifest(manifest_path, dataset=dataset)
        expected = {
            "domain": domain,
            "domain_id": int(domain_id),
            "dataset_prefix": str(dataset_prefix),
            "source_dataset_identity": source_identity,
            "base_seed": int(base_seed),
            "requested_total_sample_windows": int(total_sample_windows),
            "artifact_prefix": artifact_prefix,
        }
        mismatches = {
            name: (cached.get(name), value)
            for name, value in expected.items()
            if cached.get(name) != value
        }
        if mismatches:
            raise ValueError(f"cached manifest request mismatch: {mismatches}")
        return cached
    output_dir.mkdir(parents=True, exist_ok=True)

    document_lengths = document_lengths_from_indexed_dataset(dataset)
    document_count = int(document_lengths.size)
    splits = split_document_ids(
        document_count, base_seed=base_seed, domain_id=domain_id
    )
    split_validation = validate_document_splits(splits, document_count)
    sample_counts = ratio_counts(total_sample_windows)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "domain": domain,
        "domain_id": int(domain_id),
        "artifact_prefix": artifact_prefix,
        "dataset_prefix": str(dataset_prefix),
        "source_dataset_identity": source_identity,
        "base_seed": int(base_seed),
        "rng": {
            "bit_generator": "PCG64",
            "seed_sequence_entropy_order": [
                "base_seed",
                "domain_id",
                "stage_id",
                "split_id",
            ],
            "stage_ids": {
                "document_split": STAGE_DOCUMENT_SPLIT,
                "window_sample": STAGE_WINDOW_SAMPLE,
                "covariance_subsample": STAGE_COVARIANCE_SUBSAMPLE,
                "kmeans_reservoir": STAGE_KMEANS_RESERVOIR,
                "permutation_null": STAGE_PERMUTATION_NULL,
                "random_pair_null": STAGE_RANDOM_PAIR_NULL,
                "matched_random": STAGE_MATCHED_RANDOM,
            },
            "split_ids": SPLIT_IDS,
        },
        "split_ratios": {"calibration": 0.4, "selection": 0.3, "test": 0.3},
        "document_count": document_count,
        "document_token_count": int(document_lengths.sum(dtype=np.int64)),
        "window_rule": {
            "window_length": WINDOW_LENGTH,
            "stride": WINDOW_LENGTH,
            "minimum_kept_tail_length": MIN_TAIL_LENGTH,
            "right_aligned_tail": False,
            "position_ids_restart_at_zero": True,
            "chunk_grids": {
                "128": {"chunk_length": 128, "stride": 64},
                "256": {"chunk_length": 256, "stride": 128},
                "512_control": {"whole_window_actual_length_for_tail": True},
            },
            "suffix_not_covered_by_both_128_and_256": "ineligible",
        },
        "requested_total_sample_windows": int(total_sample_windows),
        "split_validation": split_validation,
        "splits": {},
    }

    aggregate_stats: dict[str, int] = {
        "full_window_count": 0,
        "tail_window_count": 0,
        "window_count": 0,
        "retained_token_count": 0,
        "eligible_token_count": 0,
        "ineligible_suffix_tokens": 0,
        "discarded_tail_token_count": 0,
        "discarded_tail_document_count": 0,
    }
    for split_name in SPLIT_NAMES:
        document_file = output_dir / f"{artifact_prefix}{split_name}_documents.npy"
        window_npy_file = output_dir / f"{artifact_prefix}{split_name}_windows.npy"
        window_parquet_file = output_dir / f"{artifact_prefix}{split_name}_windows.parquet"
        window_jsonl_file = output_dir / f"{artifact_prefix}{split_name}_windows.jsonl"
        for path in (document_file, window_npy_file):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        if write_parquet and window_parquet_file.exists():
            raise FileExistsError(f"refusing to overwrite {window_parquet_file}")
        if write_jsonl and window_jsonl_file.exists():
            raise FileExistsError(f"refusing to overwrite {window_jsonl_file}")

        document_ids = splits[split_name]
        enumerated, enumeration_stats = enumerate_document_windows(
            document_ids, document_lengths
        )
        sampled = sample_enumerated_windows(
            enumerated,
            sample_counts[split_name],
            base_seed=base_seed,
            domain_id=domain_id,
            split_name=split_name,
        )
        del enumerated
        validate_window_rows(sampled, document_lengths)
        membership = np.zeros(document_count, dtype=bool)
        membership[document_ids] = True
        if sampled.size and not np.all(membership[sampled["document_id"]]):
            raise AssertionError(f"{split_name} sampled a window from another split")

        atomic_npy(document_file, document_ids)
        atomic_npy(window_npy_file, sampled)
        artifacts = {
            "document_ids_npy": _artifact(
                document_file, rows=document_ids.size, kind="npy"
            ),
            "windows_npy": _artifact(window_npy_file, rows=sampled.size, kind="npy"),
        }
        if write_parquet:
            atomic_parquet(window_parquet_file, sampled)
            artifacts["windows_parquet"] = _artifact(
                window_parquet_file, rows=sampled.size, kind="parquet"
            )
        if write_jsonl:
            atomic_jsonl(window_jsonl_file, sampled)
            artifacts["windows_jsonl"] = _artifact(
                window_jsonl_file, rows=sampled.size, kind="jsonl"
            )

        payload["splits"][split_name] = {
            "split_id": SPLIT_IDS[split_name],
            "document_count": int(document_ids.size),
            "available_window_count": int(enumeration_stats["window_count"]),
            "requested_sample_window_count": int(sample_counts[split_name]),
            "sampled_window_count": int(sampled.size),
            "enumeration_stats": enumeration_stats,
            "sampled_stats": sampled_window_statistics(sampled),
            "artifacts": artifacts,
        }
        for name in aggregate_stats:
            aggregate_stats[name] += int(enumeration_stats[name])

    retained = aggregate_stats["retained_token_count"]
    aggregate_stats["ineligible_fraction_of_retained_tokens"] = (
        float(aggregate_stats["ineligible_suffix_tokens"] / retained) if retained else 0.0
    )
    aggregate_stats["ineligible_fraction_exceeds_5pct"] = bool(
        retained and aggregate_stats["ineligible_suffix_tokens"] / retained > 0.05
    )
    document_tokens = int(payload["document_token_count"])
    aggregate_stats["discarded_tail_fraction_of_document_tokens"] = (
        float(aggregate_stats["discarded_tail_token_count"] / document_tokens)
        if document_tokens
        else 0.0
    )
    payload["all_documents_window_stats"] = aggregate_stats
    payload["manifest_content_sha256"] = payload_sha256(payload)
    atomic_json(manifest_path, payload)
    return validate_domain_manifest(manifest_path, dataset=dataset)


def validate_domain_manifest(
    manifest_path: Path,
    *,
    dataset: Any | None = None,
    require_source_identity: bool = True,
) -> dict[str, Any]:
    """Validate manifest hashes, split overlap, Parquet parity, and bounds."""

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported manifest schema: {payload.get('schema')}")
    expected_content_hash = payload.get("manifest_content_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_content_sha256", None)
    if payload_sha256(unhashed) != expected_content_hash:
        raise ValueError("manifest content SHA256 mismatch")
    recorded_source_identity = payload.get("source_dataset_identity")
    if recorded_source_identity is None:
        if require_source_identity:
            raise ValueError("manifest is missing source_dataset_identity")
    else:
        live_source_identity = source_dataset_identity(
            payload["dataset_prefix"], dataset=dataset
        )
        if recorded_source_identity != live_source_identity:
            raise ValueError(
                "source_dataset_identity mismatch: IndexedDataset content changed"
            )

    root = manifest_path.parent
    splits: dict[str, np.ndarray] = {}
    document_lengths = (
        document_lengths_from_indexed_dataset(dataset) if dataset is not None else None
    )
    for split_name in SPLIT_NAMES:
        split_payload = payload["splits"][split_name]
        artifacts = split_payload["artifacts"]
        for artifact in artifacts.values():
            path = root / artifact["file"]
            if not path.is_file():
                raise ValueError(f"missing manifest artifact: {path}")
            if file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"artifact SHA256 mismatch: {path}")
            if int(path.stat().st_size) != int(artifact["bytes"]):
                raise ValueError(f"artifact size mismatch: {path}")

        document_ids = np.load(
            root / artifacts["document_ids_npy"]["file"], allow_pickle=False
        )
        windows = np.load(root / artifacts["windows_npy"]["file"], allow_pickle=False)
        splits[split_name] = document_ids
        if document_ids.dtype != np.int64 or document_ids.ndim != 1:
            raise ValueError(f"bad document ID array for {split_name}")
        if windows.dtype != WINDOW_DTYPE or windows.ndim != 1:
            raise ValueError(f"bad window array for {split_name}")
        if document_ids.size != int(split_payload["document_count"]):
            raise ValueError(f"document row count mismatch for {split_name}")
        if windows.size != int(split_payload["sampled_window_count"]):
            raise ValueError(f"window row count mismatch for {split_name}")
        if not np.array_equal(windows["sample_order"], np.arange(windows.size)):
            raise ValueError(f"sample_order is not contiguous for {split_name}")
        if np.unique(windows["source_window_index"]).size != windows.size:
            raise ValueError(f"duplicate sampled window in {split_name}")
        if windows.size and not np.all(np.isin(windows["document_id"], document_ids)):
            raise ValueError(f"window/document split mismatch in {split_name}")
        if document_lengths is not None:
            validate_window_rows(windows, document_lengths)

        parquet_artifact = artifacts.get("windows_parquet")
        if parquet_artifact is not None:
            try:
                import pyarrow.parquet as pq
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("pyarrow required to validate manifest") from error
            table = pq.read_table(root / parquet_artifact["file"])
            if table.num_rows != windows.size or table.column_names != list(WINDOW_DTYPE.names):
                raise ValueError(f"Parquet/NPY schema mismatch for {split_name}")
            for name in WINDOW_DTYPE.names or ():
                values = np.asarray(
                    table[name].combine_chunks().to_numpy(zero_copy_only=False)
                )
                if not np.array_equal(values, windows[name]):
                    raise ValueError(f"Parquet/NPY value mismatch: {split_name}:{name}")

        jsonl_artifact = artifacts.get("windows_jsonl")
        if jsonl_artifact is not None:
            jsonl_path = root / jsonl_artifact["file"]
            jsonl_rows = 0
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for jsonl_rows, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    expected = {
                        name: windows[name][jsonl_rows - 1].item()
                        for name in WINDOW_DTYPE.names or ()
                    }
                    if row != expected:
                        raise ValueError(
                            f"JSONL/NPY value mismatch: {split_name}:{jsonl_rows - 1}"
                        )
            if jsonl_rows != windows.size:
                raise ValueError(f"JSONL/NPY row mismatch for {split_name}")

    split_validation = validate_document_splits(splits, int(payload["document_count"]))
    if split_validation != payload["split_validation"]:
        raise ValueError("recorded split validation does not match artifacts")
    return payload


PREPARED_PILOT_SCHEMA = "cka_gt_pilot_prepared_inputs_v1"


def prepare_pilot(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    code_prefix: str = DEFAULT_CODE_PREFIX,
    wiki_prefix: str = DEFAULT_WIKI_PREFIX,
    seed: int = BASE_SEED,
    *,
    before_checkpoint: str = DEFAULT_BEFORE_CHECKPOINT,
    after_checkpoint: str = DEFAULT_AFTER_CHECKPOINT,
    total_sample_windows_per_domain: int = 100_000,
    datasets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare the canonical Code+Wiki pilot input tree.

    The public positional signature intentionally matches the hand-off API
    used by the forward driver.  ``datasets`` is an injection point for CPU
    tests; production calls lazily open the two Megatron IndexedDatasets.
    """

    output_root = Path(output_root)
    config_path = output_root / "config.json"
    requested = {
        "code_prefix": str(code_prefix),
        "wiki_prefix": str(wiki_prefix),
        "base_seed": int(seed),
        "before_checkpoint": str(before_checkpoint),
        "after_checkpoint": str(after_checkpoint),
        "sample_windows_per_domain": int(total_sample_windows_per_domain),
    }
    if config_path.exists():
        cached = validate_prepared_pilot(output_root, datasets=datasets)
        mismatches = {
            name: (cached.get(name), value)
            for name, value in requested.items()
            if cached.get(name) != value
        }
        if mismatches:
            raise ValueError(f"cached prepared-pilot request mismatch: {mismatches}")
        return cached

    output_root.mkdir(parents=True, exist_ok=True)
    splits_root = output_root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    opened_datasets = dict(datasets or {})
    if "code" not in opened_datasets:
        opened_datasets["code"] = _load_indexed_dataset(str(code_prefix))
    if "wiki" not in opened_datasets:
        opened_datasets["wiki"] = _load_indexed_dataset(str(wiki_prefix))

    domain_payloads: dict[str, dict[str, Any]] = {}
    for domain, prefix in (("code", code_prefix), ("wiki", wiki_prefix)):
        domain_payloads[domain] = build_domain_window_manifest(
            opened_datasets[domain],
            dataset_prefix=str(prefix),
            output_dir=splits_root,
            domain=domain,
            domain_id=DOMAIN_IDS[domain],
            total_sample_windows=total_sample_windows_per_domain,
            base_seed=seed,
            write_parquet=True,
            write_jsonl=True,
            artifact_prefix=f"{domain}_",
            manifest_filename=f"{domain}_manifest.json",
        )

    allow_missing_checkpoints = datasets is not None
    checkpoint_identities = {
        "before": checkpoint_identity(
            before_checkpoint,
            allow_missing_for_cpu_test=allow_missing_checkpoints,
        ),
        "after": checkpoint_identity(
            after_checkpoint,
            allow_missing_for_cpu_test=allow_missing_checkpoints,
        ),
    }

    payload: dict[str, Any] = {
        "schema": PREPARED_PILOT_SCHEMA,
        **requested,
        "layers": list(range(2, 10)),
        "position_ids": "restart_at_zero_for_each_window",
        "split_policy": {
            "unit": "document",
            "ratios": {"calibration": 0.4, "selection": 0.3, "test": 0.3},
            "counts": "floor40_floor30_remainder_test",
            "overlap_required": 0,
        },
        "test_policy": {
            "metrics_may_be_computed": True,
            "report_v1_must_not_reveal_test": True,
            "open_once_after_human_threshold_choice": True,
        },
        "window_policy": {
            "length": WINDOW_LENGTH,
            "stride": WINDOW_LENGTH,
            "minimum_tail_length": MIN_TAIL_LENGTH,
            "right_align_tail": False,
            "short_tail_action": "discard_and_count",
            "uncovered_fixed_grid_suffix_action": "mark_gt_ineligible",
        },
        "chunk_policy": {
            "128": {"length": 128, "stride": 64},
            "256": {"length": 256, "stride": 128},
            "512": "whole window control; actual tail length when <512",
        },
        "numeric_policy": {
            "metric_accumulation_dtype": "float32",
            "model_forward_dtype": "checkpoint_default",
            "tf32_allowed": False,
            "cka_frobenius_norm_floor": 1e-12,
            "cka_off_invalid_floor": 1e-12,
            "offdiag_warning_fraction_of_chunk_cka": 0.01,
            "offdiag_warning_rule": "CKA_off < 0.01 * CKA",
        },
        "representation_policy": {
            "layers": list(range(2, 10)),
            "hook": "residual_add_completed_transformer_layer_output",
            "raw_hidden_disk_storage": False,
            "chunk_centering": "featurewise_mean_over_tokens_within_each_chunk",
            "linear_cka_diagonal": "included",
            "rsm_row_correlation": "off_diagonal_only_and_diagnostic_only",
        },
        "routing_policy": {
            "router": "standard_mlp_router_read_only_probe",
            "old_expert_ids": list(range(8)),
            "top_k": 4,
            "per_token_per_moe_layer_fields": [
                "top4_expert_ids",
                "top4_weight",
                "old_full_mass",
                "old_selected_mass",
            ],
            "checkpoints": ["before", "after"],
            "probe_must_be_verified_on_one_window": True,
        },
        "membership_pass1": {
            "source": "before_checkpoint_wiki_calibration_only",
            "mean": {
                "estimator": "all_valid_wiki_calibration_tokens",
                "per_layer": True,
            },
            "covariance": {
                "estimator": "LedoitWolf",
                "per_layer": True,
                "uniform_subsample_cap_tokens": 2_000_000,
                "distance_output": "square_root_shrinkage_whitened_mahalanobis",
                "rng_stage_id": STAGE_COVARIANCE_SUBSAMPLE,
            },
            "prototypes": {
                "estimator": "MiniBatchKMeans-compatible",
                "implementation": "repository_deterministic_minibatch_kmeans_v1",
                "implementation_version": 1,
                "requires_sklearn": False,
                "clusters": 64,
                "per_layer_uniform_reservoir_tokens": 200_000,
                "rng_stage_id": STAGE_KMEANS_RESERVOIR,
                "random_state": int(seed),
                "init": "k-means++",
                "n_init": 1,
                "batch_size": 4096,
                "max_iter": 100,
                "reassignment_ratio": 0.01,
            },
            "membership_is_gt_condition": False,
        },
        "threshold_candidates": {
            "wiki_per_condition_recall_levels": [0.95, 0.97, 0.99],
            "cka_and_s_lower_quantiles": [0.05, 0.03, 0.01],
            "rel_l2_and_abs_log_r_upper_quantiles": [0.95, 0.97, 0.99],
            "mixed_recall_bundles": False,
            "human_selects_final_bundle": True,
        },
        "consensus_policy": {
            "primary": "condition_specific",
            "conditions_counted_independently": ["B", "T", "rel_l2", "abs_log_r"],
            "normal_rule": "at_least_7_of_8_layers_per_condition",
            "minimum_valid_layers": 6,
            "six_valid_layers_rule": "require_6_of_6",
            "scales": [128, 256],
            "scale_composition": "AND",
            "overlapping_chunk_reduction": "minimum",
            "same_layer_all_conditions": {
                "selector_role": "diagnostic_only",
                "selection_split_report": ["selected_count", "jaccard_vs_primary"],
                "human_review_if_jaccard_below": 0.9,
            },
        },
        "null_diagnostics": {
            "selector_condition": False,
            "types": {
                "permutation": {
                    "operation": "shuffle_after_token_order_within_chunk",
                    "rng_stage_id": STAGE_PERMUTATION_NULL,
                },
                "random_pair": {
                    "operation": "pair_before_with_after_from_another_window",
                    "rng_stage_id": STAGE_RANDOM_PAIR_NULL,
                },
            },
            "report_per_scale_layer": ["median", "p95", "real_distribution_overlay"],
        },
        "legacy_cosine_policy": {
            "primary_threshold_source": "code_calibration_split_per_layer_top_1pct",
            "primary_application_split": "selection",
            "all_layers_required": list(range(2, 10)),
            "selection_in_split_top_1pct": "parallel_reproduction_only",
        },
        "matched_random_policy": {
            "source": "selection_eligible_tokens",
            "count_matches": "cka_plus_m_95_bundle",
            "rng_stage_id": STAGE_MATCHED_RANDOM,
        },
        "artifacts": {},
        "source_dataset_identity": {},
        "checkpoint_identity": checkpoint_identities,
        "tail_and_ineligible_stats": {},
    }
    for domain in ("code", "wiki"):
        manifest_path = splits_root / f"{domain}_manifest.json"
        payload["artifacts"][f"{domain}_manifest"] = {
            "file": str(manifest_path.relative_to(output_root)),
            "bytes": int(manifest_path.stat().st_size),
            "sha256": file_sha256(manifest_path),
        }
        payload["tail_and_ineligible_stats"][domain] = domain_payloads[domain][
            "all_documents_window_stats"
        ]
        payload["source_dataset_identity"][domain] = domain_payloads[domain][
            "source_dataset_identity"
        ]
    payload["config_content_sha256"] = payload_sha256(payload)
    atomic_json(config_path, payload)
    return validate_prepared_pilot(output_root, datasets=opened_datasets)


def validate_prepared_pilot(
    output_root: str | Path, *, datasets: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the complete canonical Code+Wiki prepared input tree."""

    output_root = Path(output_root)
    config_path = output_root / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PREPARED_PILOT_SCHEMA:
        raise ValueError(f"unsupported prepared-pilot schema: {payload.get('schema')}")
    expected_hash = payload.get("config_content_sha256")
    unhashed = dict(payload)
    unhashed.pop("config_content_sha256", None)
    if payload_sha256(unhashed) != expected_hash:
        raise ValueError("prepared-pilot config content SHA256 mismatch")

    allow_missing_checkpoints = datasets is not None
    expected_checkpoint_identities = {
        "before": checkpoint_identity(
            payload["before_checkpoint"],
            allow_missing_for_cpu_test=allow_missing_checkpoints,
        ),
        "after": checkpoint_identity(
            payload["after_checkpoint"],
            allow_missing_for_cpu_test=allow_missing_checkpoints,
        ),
    }
    if payload.get("checkpoint_identity") != expected_checkpoint_identities:
        raise ValueError("checkpoint_identity mismatch: checkpoint content changed")

    opened_datasets = dict(datasets or {})
    domain_payloads: dict[str, dict[str, Any]] = {}
    for domain in ("code", "wiki"):
        artifact = payload["artifacts"][f"{domain}_manifest"]
        manifest_path = output_root / artifact["file"]
        if not manifest_path.is_file():
            raise ValueError(f"missing domain manifest: {manifest_path}")
        if file_sha256(manifest_path) != artifact["sha256"]:
            raise ValueError(f"domain manifest SHA256 mismatch: {domain}")
        if int(manifest_path.stat().st_size) != int(artifact["bytes"]):
            raise ValueError(f"domain manifest size mismatch: {domain}")
        if domain not in opened_datasets:
            prefix = payload[f"{domain}_prefix"]
            opened_datasets[domain] = _load_indexed_dataset(prefix)
        domain_payload = validate_domain_manifest(
            manifest_path, dataset=opened_datasets[domain]
        )
        if domain_payload["domain"] != domain:
            raise ValueError(f"domain identity mismatch: {domain}")
        if domain_payload["base_seed"] != payload["base_seed"]:
            raise ValueError(f"domain seed mismatch: {domain}")
        if (
            domain_payload["requested_total_sample_windows"]
            != payload["sample_windows_per_domain"]
        ):
            raise ValueError(f"domain sample-count mismatch: {domain}")
        if (
            domain_payload["all_documents_window_stats"]
            != payload["tail_and_ineligible_stats"][domain]
        ):
            raise ValueError(f"domain tail/ineligible stats mismatch: {domain}")
        if payload.get("source_dataset_identity", {}).get(domain) != domain_payload.get(
            "source_dataset_identity"
        ):
            raise ValueError(f"config/domain source_dataset_identity mismatch: {domain}")
        domain_payloads[domain] = domain_payload

    for split_name, expected_count in ratio_counts(
        int(payload["sample_windows_per_domain"])
    ).items():
        for domain in ("code", "wiki"):
            actual = domain_payloads[domain]["splits"][split_name][
                "sampled_window_count"
            ]
            if int(actual) != int(expected_count):
                raise ValueError(
                    f"canonical sample count mismatch: {domain}/{split_name}: "
                    f"{actual} != {expected_count}"
                )
    return payload


def upgrade_prepared_input_identities(
    output_root: str | Path, *, datasets: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Crash-recoverably add exact source identities to a prepared-only root.

    Frozen split/window arrays are preserved byte-for-byte.  The operation is
    refused once any runtime, membership, or metric artifact exists.  A small
    transaction journal makes a crash between the two domain-manifest renames
    and the config rename safely resumable.
    """

    output_root = Path(output_root).resolve()
    journal_path = output_root / ".source_identity_upgrade.json.inprogress"
    protected_roots = (
        output_root / "runtime",
        output_root / "membership",
        output_root / "token_metrics",
        output_root / "chunk_metrics",
        output_root / "sealed_test" / "raw",
    )
    material = [
        str(path.relative_to(output_root))
        for root in protected_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    if material:
        raise RuntimeError(
            "source-identity upgrade is forbidden after runtime artifacts exist: "
            f"{material[:10]}"
        )

    if journal_path.is_file():
        transaction = json.loads(journal_path.read_text(encoding="utf-8"))
        if transaction.get("schema") != "cka_gt_pilot_source_identity_upgrade_v1":
            raise RuntimeError("unsupported source-identity upgrade journal")
    else:
        config_path = output_root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("source_dataset_identity") and config.get(
            "checkpoint_identity"
        ):
            return validate_prepared_pilot(output_root, datasets=datasets)
        expected_hash = config.get("config_content_sha256")
        unhashed = dict(config)
        unhashed.pop("config_content_sha256", None)
        if payload_sha256(unhashed) != expected_hash:
            raise ValueError("prepared config is corrupt before identity upgrade")

        opened = dict(datasets or {})
        domain_manifests: dict[str, dict[str, Any]] = {}
        for domain in ("code", "wiki"):
            if domain not in opened:
                opened[domain] = _load_indexed_dataset(config[f"{domain}_prefix"])
            artifact = config["artifacts"][f"{domain}_manifest"]
            manifest_path = output_root / artifact["file"]
            if file_sha256(manifest_path) != artifact["sha256"]:
                raise ValueError(f"pre-upgrade domain manifest hash mismatch: {domain}")
            manifest = validate_domain_manifest(
                manifest_path,
                dataset=opened[domain],
                require_source_identity=bool(
                    config.get("source_dataset_identity", {}).get(domain)
                ),
            )
            identity = source_dataset_identity(
                config[f"{domain}_prefix"], dataset=opened[domain]
            )
            manifest = dict(manifest)
            manifest["source_dataset_identity"] = identity
            manifest.pop("manifest_content_sha256", None)
            manifest["manifest_content_sha256"] = payload_sha256(manifest)
            domain_manifests[domain] = manifest

        upgraded_config = dict(config)
        upgraded_config["source_dataset_identity"] = {
            domain: domain_manifests[domain]["source_dataset_identity"]
            for domain in ("code", "wiki")
        }
        upgraded_config["checkpoint_identity"] = {
            "before": checkpoint_identity(
                upgraded_config["before_checkpoint"],
                allow_missing_for_cpu_test=datasets is not None,
            ),
            "after": checkpoint_identity(
                upgraded_config["after_checkpoint"],
                allow_missing_for_cpu_test=datasets is not None,
            ),
        }
        for domain in ("code", "wiki"):
            manifest_path = (
                output_root
                / upgraded_config["artifacts"][f"{domain}_manifest"]["file"]
            )
            encoded = canonical_json_bytes(domain_manifests[domain])
            upgraded_config["artifacts"][f"{domain}_manifest"].update(
                {
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        upgraded_config.pop("config_content_sha256", None)
        upgraded_config["config_content_sha256"] = payload_sha256(upgraded_config)
        transaction = {
            "schema": "cka_gt_pilot_source_identity_upgrade_v1",
            "domain_manifests": domain_manifests,
            "config": upgraded_config,
        }
        atomic_json(journal_path, transaction)

    for domain in ("code", "wiki"):
        manifest_file = transaction["config"]["artifacts"][f"{domain}_manifest"][
            "file"
        ]
        atomic_json(
            output_root / manifest_file,
            transaction["domain_manifests"][domain],
        )
    atomic_json(output_root / "config.json", transaction["config"])
    journal_path.unlink()
    return validate_prepared_pilot(output_root, datasets=datasets)


# Backward-compatible name used while the audit initially covered datasets
# only; the transaction now upgrades both datasets and checkpoints.
upgrade_prepared_source_identities = upgrade_prepared_input_identities


def _load_indexed_dataset(dataset_prefix: str) -> Any:
    return MMapIndexedDatasetLite(dataset_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare canonical deterministic Code+Wiki CKA pilot manifests"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--code-prefix", default=DEFAULT_CODE_PREFIX)
    parser.add_argument("--wiki-prefix", default=DEFAULT_WIKI_PREFIX)
    parser.add_argument("--before-checkpoint", default=DEFAULT_BEFORE_CHECKPOINT)
    parser.add_argument("--after-checkpoint", default=DEFAULT_AFTER_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--sample-windows-per-domain", type=int, default=100_000)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an already prepared output instead of creating it",
    )
    parser.add_argument(
        "--upgrade-source-identities",
        action="store_true",
        help=(
            "upgrade a prepared-only root with exact dataset/checkpoint identities; "
            "refuses roots containing runtime metrics"
        ),
    )
    args = parser.parse_args()

    if args.validate_only and args.upgrade_source_identities:
        raise SystemExit("choose only one of --validate-only/--upgrade-source-identities")
    if args.upgrade_source_identities:
        payload = upgrade_prepared_input_identities(args.output_root)
    elif args.validate_only:
        payload = validate_prepared_pilot(args.output_root)
    else:
        payload = prepare_pilot(
            args.output_root,
            args.code_prefix,
            args.wiki_prefix,
            args.seed,
            before_checkpoint=args.before_checkpoint,
            after_checkpoint=args.after_checkpoint,
            total_sample_windows_per_domain=args.sample_windows_per_domain,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
