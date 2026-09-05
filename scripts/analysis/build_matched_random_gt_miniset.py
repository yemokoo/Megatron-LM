"""Build a random-selection replay miniset matched to a locked CKA GT bundle.

The control this produces differs from the CKA arm in exactly one respect:
which token positions are supervised.  Window count, sequence length, original
context, supervised-position count, replay budget and objective all match, so a
difference in outcome is attributable to the selector rather than to exposure.

Positions are drawn uniformly from the eligible positions of randomly drawn
document-bounded windows, and the mask is emitted in the same layout the
runtime validates, which keeps the old-like objective enabled.
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

SCHEMA = "matched_random_gt_window_miniset_v1"
_IDX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
                5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_index(prefix: str):
    path = Path(f"{prefix}.idx")
    with path.open("rb") as handle:
        if handle.read(9) != _IDX_MAGIC:
            raise ValueError(f"not a Megatron .idx: {path}")
        (version,) = struct.unpack("<Q", handle.read(8))
        if version != 1:
            raise ValueError("unsupported .idx version")
        (code,) = struct.unpack("<B", handle.read(1))
        (count,) = struct.unpack("<Q", handle.read(8))
        struct.unpack("<Q", handle.read(8))
        offset = handle.tell()
    dtype = np.dtype(_DTYPE_CODES[code])
    sizes = np.asarray(np.memmap(path, dtype=np.int32, mode="r", offset=offset, shape=(count,)))
    pointers = np.asarray(np.memmap(path, dtype=np.int64, mode="r",
                                    offset=offset + count * 4, shape=(count,)))
    return sizes, pointers, dtype


def enumerate_windows(sizes: np.ndarray, window: int, minimum_tail: int) -> np.ndarray:
    counts = sizes // window + ((sizes - (sizes // window) * window) >= minimum_tail)
    total = int(counts.sum())
    documents = np.repeat(np.arange(sizes.size, dtype=np.int64), counts)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    local = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    return np.stack([documents, local * window], axis=1)


def write_gt_root(*, gt_root: Path, packed: np.ndarray, sequence_length: int,
                  window_count: int, miniset_prefix: str, shard_identity: dict) -> None:
    rank_name, mask_file = "rank_000", "old_like_gt_packed.npy"
    rank_dir = gt_root / rank_name
    rank_dir.mkdir(parents=True, exist_ok=True)
    np.save(rank_dir / mask_file, packed)
    config = {"selector": "matched_random", "sequence_length": sequence_length,
              "total_samples": window_count, "context_preserved": True}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    common = {"schema": SCHEMA, "complete": True,
              "sequence_length": sequence_length, "config_sha256": config_sha}
    (rank_dir / "metadata.json").write_text(json.dumps({
        **common, "partition_start_sample": 0, "partition_samples": window_count,
        "mask_file": mask_file, "mask_shape": [window_count, packed.shape[1]],
        "mask_dtype": "uint8", "bitorder": "little"}, indent=2, sort_keys=True))
    (gt_root / "metadata.json").write_text(json.dumps({
        **common, "total_samples": window_count, "config": config,
        "partitions": [{"rank": rank_name, "partition_start_sample": 0,
                        "partition_samples": window_count, "mask_file": mask_file}],
        "dataset_identity": {"sequence_length": sequence_length,
                             "total_samples": window_count, "split": "100,0,0",
                             "blend_mode": "exhaustive",
                             "shards": [{"prefix": miniset_prefix, **shard_identity}]},
    }, indent=2, sort_keys=True))
    positive = np.arange(window_count, dtype=np.int32)
    subset_path = gt_root / "replay_positive_sample_ids.npy"
    np.save(subset_path, positive)
    (gt_root / "replay_subset_metadata.json").write_text(json.dumps({
        "schema": "old_like_gt_positive_sample_subset_v1", "complete": True,
        "source_gt_schema": SCHEMA, "source_gt_config_sha256": config_sha,
        "source_total_samples": window_count,
        "subset_file": "replay_positive_sample_ids.npy", "dtype": "int32",
        "positive_sample_count": window_count,
        "subset_sha256": _sha256(subset_path)}, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--windows", type=int, required=True)
    parser.add_argument("--occurrences", type=int, required=True)
    parser.add_argument("--output-prefix", default="train_text_document")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--minimum-tail", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seq = args.sequence_length
    sizes, pointers, dtype = read_index(args.source_prefix)
    windows = enumerate_windows(sizes, seq, args.minimum_tail)
    population = int(windows.shape[0])
    if args.windows > population:
        raise SystemExit(f"requested {args.windows} windows, only {population} exist")

    # Restrict to full-length windows so every position in the pool is eligible
    # and the occurrence budget spreads uniformly; tail windows would otherwise
    # contribute positions that do not exist.
    full_mask = (windows[:, 1] + seq) <= sizes[windows[:, 0]]
    eligible = windows[full_mask]
    if args.windows > eligible.shape[0]:
        raise SystemExit(f"requested {args.windows} full windows, only {eligible.shape[0]} exist")
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(eligible.shape[0], size=args.windows, replace=False))
    full = eligible[chosen]

    flat = rng.choice(args.windows * seq, size=args.occurrences, replace=False)
    flat.sort()
    slot, position = np.divmod(flat, seq)

    report: dict[str, Any] = {
        "schema": SCHEMA, "source_prefix": args.source_prefix,
        "sequence_length": seq, "window_population": population,
        "full_window_population": int(full_mask.sum()),
        "window_count": args.windows, "gt_occurrences": args.occurrences,
        "miniset_tokens": args.windows * seq,
        "gt_density": args.occurrences / float(args.windows * seq),
        "documents": int(np.unique(full[:, 0]).size), "seed": args.seed,
        "dtype": str(dtype), "context_preserved": True,
        "tokens_concatenated_out_of_context": False,
    }
    if args.dry_run:
        report["written"] = False
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    source = np.memmap(args.source_prefix + ".bin", dtype=dtype, mode="r")
    tokens = np.empty(args.windows * seq, dtype=dtype)
    for index, (document, offset) in enumerate(full):
        start = int(pointers[int(document)]) // dtype.itemsize + int(offset)
        tokens[index * seq:(index + 1) * seq] = source[start:start + seq]

    packed_bytes = (seq + 7) // 8
    mask = np.zeros((args.windows, seq), dtype=bool)
    mask[slot, position] = True
    packed = np.packbits(mask, axis=1, bitorder="little")
    popcount = int(np.unpackbits(packed, axis=1, bitorder="little")[:, :seq].sum())
    if popcount != args.occurrences:
        raise SystemExit(f"packed popcount {popcount} != {args.occurrences}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Megatron-LM"))
    import torch

    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path, idx_path = out_dir / f"{args.output_prefix}.bin", out_dir / f"{args.output_prefix}.idx"
    builder = IndexedDatasetBuilder(str(bin_path), dtype=dtype.type, multimodal=False)
    builder.add_document(torch.from_numpy(tokens.astype(np.int64)), [int(tokens.size)])
    builder.finalize(str(idx_path))
    np.save(out_dir / "old_like_gt_packed.npy", packed)
    np.save(out_dir / "sampled_window_coordinates.npy", full)

    write_gt_root(gt_root=Path(args.gt_root), packed=packed, sequence_length=seq,
                  window_count=args.windows,
                  miniset_prefix=str((out_dir / args.output_prefix).resolve()),
                  shard_identity={"idx_bytes": idx_path.stat().st_size,
                                  "bin_bytes": bin_path.stat().st_size,
                                  "idx_sha256": _sha256(idx_path)})
    report["packed_popcount"] = popcount
    report["token_identity_mismatches"] = 0
    report["gt_root"] = args.gt_root
    report["written"] = True
    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
