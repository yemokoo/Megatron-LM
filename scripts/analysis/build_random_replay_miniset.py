"""Build a random-sample replay miniset held out as its own dataset.

This is the control arm for selector-based replay: instead of choosing tokens
by a criterion, take a uniformly random slice of the task's own training data
and replay it.  Sampling is by document-bounded window so every replayed token
keeps a real context, and the whole miniset is supervised (no GT mask).

Supports single-prefix datasets (Code) and sharded ones (Conversation), and
preserves the source dtype so token IDs round-trip exactly.

    build_random_replay_miniset.py --source-prefix <prefix> --output-dir <dir>
    build_random_replay_miniset.py --source-dir <dir with shard_*.idx> ...
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

SCHEMA = "random_replay_window_miniset_v1"
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
    path = Path(f"{prefix}.idx")
    with path.open("rb") as handle:
        if handle.read(9) != _IDX_MAGIC:
            raise ValueError(f"not a Megatron .idx file: {path}")
        (version,) = struct.unpack("<Q", handle.read(8))
        if version != 1:
            raise ValueError(f"unsupported .idx version {version}: {path}")
        (code,) = struct.unpack("<B", handle.read(1))
        (sequence_count,) = struct.unpack("<Q", handle.read(8))
        struct.unpack("<Q", handle.read(8))
        offset = handle.tell()
    dtype = np.dtype(_DTYPE_CODES[code])
    sizes = np.memmap(path, dtype=np.int32, mode="r", offset=offset, shape=(sequence_count,))
    pointers = np.memmap(
        path, dtype=np.int64, mode="r",
        offset=offset + sequence_count * 4, shape=(sequence_count,),
    )
    return np.asarray(sizes), np.asarray(pointers), dtype


def resolve_shards(source_prefix: str | None, source_dir: str | None) -> list[str]:
    if source_prefix:
        return [source_prefix]
    shards = sorted(str(p)[: -len(".idx")] for p in Path(source_dir).glob("*.idx"))
    if not shards:
        raise ValueError(f"no .idx shards under {source_dir}")
    return shards


def enumerate_windows(sizes: np.ndarray, window: int, minimum_tail: int) -> np.ndarray:
    """Document-bounded windows: offsets 0, W, 2W, ... plus a long-enough tail.

    Same rule the CKA census used, so the random control and the selector
    control sample from an identical window population.
    """
    counts = sizes // window + ((sizes - (sizes // window) * window) >= minimum_tail)
    total = int(counts.sum())
    documents = np.repeat(np.arange(sizes.size, dtype=np.int64), counts)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    local = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    return np.stack([documents, local * window], axis=1)


def build(
    *,
    shards: list[str],
    output_dir: Path,
    output_prefix: str,
    fraction: float,
    windows_requested: int | None,
    window: int,
    minimum_tail: int,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    per_shard: list[dict[str, Any]] = []
    dtype: np.dtype | None = None
    total_tokens = 0
    for prefix in shards:
        sizes, pointers, shard_dtype = read_index(prefix)
        if dtype is None:
            dtype = shard_dtype
        elif shard_dtype != dtype:
            raise ValueError(f"shard dtype mismatch: {shard_dtype} != {dtype}")
        windows = enumerate_windows(sizes, window, minimum_tail)
        per_shard.append({"prefix": prefix, "pointers": pointers, "windows": windows})
        total_tokens += int(sizes.sum(dtype=np.int64))

    shard_of = np.concatenate([
        np.full(entry["windows"].shape[0], index, dtype=np.int64)
        for index, entry in enumerate(per_shard)
    ])
    all_windows = np.concatenate([entry["windows"] for entry in per_shard], axis=0)
    population = int(all_windows.shape[0])

    # A replay pool is normally sized by the training budget it has to fill
    # (budget / epochs / window), not by a corpus fraction, so an explicit
    # window count takes precedence. The achieved fraction is always reported.
    if windows_requested is not None:
        sample_count = int(windows_requested)
    else:
        target_tokens = int(round(total_tokens * fraction))
        sample_count = max(1, int(round(target_tokens / window)))
    if sample_count > population:
        raise ValueError(f"requested {sample_count} windows but only {population} exist")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(population, size=sample_count, replace=False))

    report = {
        "schema": SCHEMA,
        "shards": [entry["prefix"] for entry in per_shard],
        "source_documents": int(sum(len(read_index(p)[0]) for p in shards)),
        "source_tokens": total_tokens,
        "window": window,
        "minimum_tail": minimum_tail,
        "window_population": population,
        "requested_fraction": None if windows_requested is not None else fraction,
        "requested_windows": windows_requested,
        "sampled_windows": sample_count,
        "miniset_tokens": sample_count * window,
        "achieved_fraction": sample_count * window / total_tokens,
        "seed": seed,
        "dtype": str(dtype),
        "all_positions_supervised": True,
        "context_preserved": True,
    }
    if dry_run:
        report["written"] = False
        return report

    tokens = np.empty(sample_count * window, dtype=dtype)
    for slot, index in enumerate(chosen):
        entry = per_shard[int(shard_of[index])]
        document, offset = all_windows[index]
        source = np.memmap(entry["prefix"] + ".bin", dtype=dtype, mode="r")
        start = int(entry["pointers"][int(document)]) // dtype.itemsize + int(offset)
        piece = source[start:start + window]
        if piece.size != window:
            raise ValueError(f"short window at shard slot {slot}")
        tokens[slot * window:(slot + 1) * window] = piece

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Megatron-LM"))
    import torch

    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{output_prefix}.bin"
    idx_path = output_dir / f"{output_prefix}.idx"
    builder = IndexedDatasetBuilder(str(bin_path), dtype=dtype.type, multimodal=False)
    builder.add_document(torch.from_numpy(tokens.astype(np.int64)), [int(tokens.size)])
    builder.finalize(str(idx_path))

    np.save(output_dir / "sampled_window_coordinates.npy",
            np.stack([shard_of[chosen], all_windows[chosen, 0], all_windows[chosen, 1]], axis=1))
    report["written"] = True
    report["files"] = {
        name: {"sha256": _sha256(output_dir / name), "size_bytes": (output_dir / name).stat().st_size}
        for name in (f"{output_prefix}.bin", f"{output_prefix}.idx", "sampled_window_coordinates.npy")
    }
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-prefix")
    group.add_argument("--source-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="train_text_document")
    parser.add_argument("--fraction", type=float, default=0.001)
    parser.add_argument("--windows", type=int,
                        help="exact window count; overrides --fraction")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--minimum-tail", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = build(
        shards=resolve_shards(args.source_prefix, args.source_dir),
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        fraction=args.fraction,
        windows_requested=args.windows,
        window=args.window,
        minimum_tail=args.minimum_tail,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
