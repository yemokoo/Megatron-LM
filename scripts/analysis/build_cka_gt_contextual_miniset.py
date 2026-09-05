"""Build a standalone contextual replay miniset from a locked CKA GT bundle.

Unlike the cosine-era token-only miniset, every GT occurrence keeps the exact
512-token document-bounded window it was measured in.  The miniset is a single
ordered document whose length is an exact multiple of the sequence length, so
GPTDataset sample ``i`` reproduces source window ``i`` byte for byte in
positions ``0..seq_len-1``.  Only GT positions carry a replay loss; every other
position is context.

The GT axis therefore becomes the miniset's own outer sample index, which is
what ``OldLikeGTDataset`` expects.  No occurrence is concatenated, reordered, or
stripped of its context.

Read-only with respect to the locked bundle: the lock's own SHA256 records are
verified before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "cka_gt_contextual_window_miniset_v1"
DEFAULT_LOCK = (
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816"
    "/analysis/cka_gt_full_census_v1/full_census/gt_locked_bundle_95_v1"
)
DEFAULT_SOURCE_PREFIX = (
    "/data2/seonghyeonnoh/LLM-continual-learning-data"
    "/flamedata2.data2-verified-backup/code/train/train_text_document"
)
_IDX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {
    1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
    5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_index(prefix: str) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    """Return (sizes, byte pointers, dtype) of a Megatron MMapIndexedDataset."""
    path = Path(f"{prefix}.idx")
    with path.open("rb") as handle:
        if handle.read(9) != _IDX_MAGIC:
            raise ValueError(f"not a Megatron .idx file: {path}")
        (version,) = struct.unpack("<Q", handle.read(8))
        if version != 1:
            raise ValueError(f"unsupported .idx version {version}: {path}")
        (code,) = struct.unpack("<B", handle.read(1))
        (sequence_count,) = struct.unpack("<Q", handle.read(8))
        struct.unpack("<Q", handle.read(8))  # document count, unused here
        offset = handle.tell()
    dtype = _DTYPE_CODES[code]
    sizes = np.memmap(path, dtype=np.int32, mode="r", offset=offset, shape=(sequence_count,))
    pointers = np.memmap(
        path, dtype=np.int64, mode="r",
        offset=offset + sequence_count * 4, shape=(sequence_count,),
    )
    return np.asarray(sizes), np.asarray(pointers), np.dtype(dtype)


def verify_lock(lock_dir: Path) -> dict[str, Any]:
    """Re-verify the immutable lock's own SHA256 records before reading it."""
    validation = json.loads((lock_dir / "validation.json").read_text())
    if not validation.get("passed"):
        raise ValueError("locked bundle validation did not pass")
    recorded = validation["output_files_before_validation"]
    for name, entry in recorded.items():
        actual = _sha256(lock_dir / name)
        if actual != entry["sha256"]:
            raise ValueError(f"lock SHA256 mismatch for {name}: {actual} != {entry['sha256']}")
    return validation


def write_gt_root(
    *,
    gt_root: Path,
    packed: np.ndarray,
    sequence_length: int,
    window_count: int,
    miniset_prefix: str,
    shard_identity: dict[str, Any],
) -> None:
    """Emit the mask in the layout ``OldLikeGTDataset`` validates and mmaps.

    One partition is enough: the miniset is a single ordered stream and the GT
    axis is its own outer sample index, so there is no rank-sharded paired
    extraction to mirror.
    """
    rank_name = "rank_000"
    mask_file = "old_like_gt_packed.npy"
    rank_dir = gt_root / rank_name
    rank_dir.mkdir(parents=True, exist_ok=True)
    np.save(rank_dir / mask_file, packed)

    config = {
        "selector": "cka_locked_bundle_95",
        "sequence_length": sequence_length,
        "total_samples": window_count,
        "context_preserved": True,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    common = {
        "schema": SCHEMA,
        "complete": True,
        "sequence_length": sequence_length,
        "config_sha256": config_sha256,
    }
    (rank_dir / "metadata.json").write_text(json.dumps({
        **common,
        "partition_start_sample": 0,
        "partition_samples": window_count,
        "mask_file": mask_file,
        "mask_shape": [window_count, packed.shape[1]],
        "mask_dtype": "uint8",
        "bitorder": "little",
    }, indent=2, sort_keys=True))

    (gt_root / "metadata.json").write_text(json.dumps({
        **common,
        "total_samples": window_count,
        "config": config,
        "partitions": [{
            "rank": rank_name,
            "partition_start_sample": 0,
            "partition_samples": window_count,
            "mask_file": mask_file,
        }],
        # The replay sample axis is this miniset, not the full Code blend.
        # Declaring it exhaustively is what keeps sample i bound to window i.
        "dataset_identity": {
            "sequence_length": sequence_length,
            "total_samples": window_count,
            "split": "100,0,0",
            "blend_mode": "exhaustive",
            "shards": [{"prefix": miniset_prefix, **shard_identity}],
        },
    }, indent=2, sort_keys=True))

    # Every miniset window contains at least one GT occurrence, so the replay
    # positive subset is the whole axis.  The runtime still requires it to be
    # declared explicitly rather than inferred.
    positive_ids = np.arange(window_count, dtype=np.int32)
    subset_path = gt_root / "replay_positive_sample_ids.npy"
    np.save(subset_path, positive_ids)
    (gt_root / "replay_subset_metadata.json").write_text(json.dumps({
        "schema": "old_like_gt_positive_sample_subset_v1",
        "complete": True,
        "source_gt_schema": SCHEMA,
        "source_gt_config_sha256": config_sha256,
        "source_total_samples": window_count,
        "subset_file": "replay_positive_sample_ids.npy",
        "dtype": "int32",
        "positive_sample_count": window_count,
        "subset_sha256": _sha256(subset_path),
    }, indent=2, sort_keys=True))


def build(
    *,
    lock_dir: Path,
    source_prefix: str,
    output_dir: Path,
    output_prefix: str,
    sequence_length: int,
    gt_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    validation = verify_lock(lock_dir)
    expected_occurrences = int(validation["selected_occurrences"])

    documents = np.load(lock_dir / "occurrence_document_ids.npy")
    offsets = np.load(lock_dir / "occurrence_window_offsets.npy")
    window_ids = np.load(lock_dir / "occurrence_source_window_indices.npy")
    positions = np.load(lock_dir / "occurrence_positions.npy")
    token_ids = np.load(lock_dir / "occurrence_token_ids.npy")
    if token_ids.size != expected_occurrences:
        raise ValueError("occurrence count disagrees with the lock validation record")

    # Deterministic miniset order: ascending source window index.
    unique_windows, first_index = np.unique(window_ids, return_index=True)
    order = np.argsort(unique_windows)
    unique_windows = unique_windows[order]
    first_index = first_index[order]
    window_documents = documents[first_index]
    window_offsets = offsets[first_index]
    window_count = int(unique_windows.size)

    # A window's document and offset must be single-valued; the lock is per
    # occurrence, so disagreement would mean the axis is not what we assume.
    slot_of_window = {int(w): i for i, w in enumerate(unique_windows)}
    slot = np.fromiter((slot_of_window[int(w)] for w in window_ids), dtype=np.int64, count=window_ids.size)
    if not np.array_equal(window_documents[slot], documents):
        raise ValueError("a source window maps to more than one document id")
    if not np.array_equal(window_offsets[slot], offsets):
        raise ValueError("a source window maps to more than one window offset")
    if int(positions.max()) >= sequence_length or int(positions.min()) < 0:
        raise ValueError("occurrence position outside the sequence length")

    sizes, pointers, dtype = read_index(source_prefix)
    source_bin = np.memmap(f"{source_prefix}.bin", dtype=dtype, mode="r")

    tokens = np.empty(window_count * sequence_length, dtype=dtype)
    for slot_index, (document, offset) in enumerate(zip(window_documents, window_offsets)):
        start = int(pointers[int(document)]) // dtype.itemsize + int(offset)
        window = source_bin[start:start + sequence_length]
        if window.size != sequence_length:
            raise ValueError(f"source window {slot_index} is short: {window.size}")
        tokens[slot_index * sequence_length:(slot_index + 1) * sequence_length] = window

    # Every GT occurrence must land on the token the census actually recorded.
    flat = slot * sequence_length + positions.astype(np.int64)
    mismatches = int(np.count_nonzero(tokens[flat] != token_ids))
    if mismatches:
        raise ValueError(f"{mismatches} GT occurrences do not match the recorded token id")

    packed_bytes = (sequence_length + 7) // 8
    mask = np.zeros((window_count, sequence_length), dtype=bool)
    mask[slot, positions.astype(np.int64)] = True
    packed = np.packbits(mask, axis=1, bitorder="little")
    if packed.shape != (window_count, packed_bytes):
        raise ValueError(f"unexpected packed mask shape {packed.shape}")
    popcount = int(np.unpackbits(packed, axis=1, bitorder="little")[:, :sequence_length].sum())
    if popcount != expected_occurrences:
        raise ValueError(f"packed popcount {popcount} != {expected_occurrences}")

    report = {
        "schema": SCHEMA,
        "lock_dir": str(lock_dir),
        "source_prefix": source_prefix,
        "sequence_length": sequence_length,
        "window_count": window_count,
        "document_count": int(np.unique(window_documents).size),
        "miniset_tokens": int(tokens.size),
        "gt_occurrences": expected_occurrences,
        "gt_density": expected_occurrences / float(tokens.size),
        "occurrences_per_window_min": int(np.bincount(slot, minlength=window_count).min()),
        "occurrences_per_window_max": int(np.bincount(slot, minlength=window_count).max()),
        "token_identity_comparisons": expected_occurrences,
        "token_identity_mismatches": 0,
        "packed_popcount": popcount,
        "lock_sha256_verified": True,
        "context_preserved": True,
        "tokens_concatenated_out_of_context": False,
    }
    if dry_run:
        report["written"] = False
        return report

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Megatron-LM"))
    import torch

    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{output_prefix}.bin"
    idx_path = output_dir / f"{output_prefix}.idx"
    # IndexedDatasetBuilder keys its type code off the numpy scalar class, not
    # a dtype instance.
    builder = IndexedDatasetBuilder(str(bin_path), dtype=dtype.type, multimodal=False)
    # One ordered document keeps miniset sample i aligned with window i while
    # letting GPTDataset form contiguous samples exactly as it does upstream.
    builder.add_document(torch.from_numpy(tokens.astype(np.int64)), [int(tokens.size)])
    builder.finalize(str(idx_path))

    np.save(output_dir / "old_like_gt_packed.npy", packed)
    # Physical fingerprints of the miniset the GT sample axis refers to; the
    # runtime refuses to attach the mask to any other shard.
    write_gt_root(
        gt_root=gt_root,
        packed=packed,
        sequence_length=sequence_length,
        window_count=window_count,
        miniset_prefix=str((output_dir / output_prefix).resolve()),
        shard_identity={
            "idx_bytes": idx_path.stat().st_size,
            "bin_bytes": bin_path.stat().st_size,
            "idx_sha256": _sha256(idx_path),
        },
    )
    report["gt_root"] = str(gt_root)
    report["written"] = True
    report["files"] = {
        name: {"sha256": _sha256(output_dir / name), "size_bytes": (output_dir / name).stat().st_size}
        for name in (f"{output_prefix}.bin", f"{output_prefix}.idx", "old_like_gt_packed.npy")
    }
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-dir", default=DEFAULT_LOCK)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="train_text_document")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--gt-root", help="defaults to <output-dir>/../gt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = build(
        lock_dir=Path(args.lock_dir),
        source_prefix=args.source_prefix,
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        sequence_length=args.sequence_length,
        gt_root=Path(args.gt_root) if args.gt_root else Path(args.output_dir).parent / "gt",
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
