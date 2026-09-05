"""Concatenate the sharded Conversation train split into one IndexedDataset.

The CKA census addresses windows by (document id, offset) against a single
dataset prefix.  Merging the 42 shards in sorted order gives Conversation the
same single-prefix shape Code already has, so the census, GT builder and
miniset builder all work unchanged.  Originals are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

_IDX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
                5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def read_index(prefix: str):
    path = Path(f"{prefix}.idx")
    with path.open("rb") as handle:
        if handle.read(9) != _IDX_MAGIC:
            raise ValueError(f"not a Megatron .idx: {path}")
        (version,) = struct.unpack("<Q", handle.read(8))
        if version != 1:
            raise ValueError(f"unsupported .idx version {version}")
        (code,) = struct.unpack("<B", handle.read(1))
        (count,) = struct.unpack("<Q", handle.read(8))
        struct.unpack("<Q", handle.read(8))
        offset = handle.tell()
    dtype = np.dtype(_DTYPE_CODES[code])
    sizes = np.asarray(np.memmap(path, dtype=np.int32, mode="r", offset=offset, shape=(count,)))
    pointers = np.asarray(np.memmap(path, dtype=np.int64, mode="r",
                                    offset=offset + count * 4, shape=(count,)))
    return sizes, pointers, dtype


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="train_text_document")
    args = parser.parse_args(argv)

    shards = sorted(str(p)[: -len(".idx")] for p in Path(args.source_dir).glob("*.idx"))
    if not shards:
        raise SystemExit(f"no shards under {args.source_dir}")

    dtype = None
    lengths: list[np.ndarray] = []
    for prefix in shards:
        sizes, _, shard_dtype = read_index(prefix)
        if dtype is None:
            dtype = shard_dtype
        elif shard_dtype != dtype:
            raise SystemExit(f"dtype mismatch in {prefix}")
        lengths.append(sizes)
    all_lengths = np.concatenate(lengths)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Megatron-LM"))
    import torch

    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{args.output_prefix}.bin"
    idx_path = out_dir / f"{args.output_prefix}.idx"
    builder = IndexedDatasetBuilder(str(bin_path), dtype=dtype.type, multimodal=False)
    for prefix in shards:
        sizes, pointers, _ = read_index(prefix)
        source = np.memmap(prefix + ".bin", dtype=dtype, mode="r")
        for index in range(sizes.size):
            start = int(pointers[index]) // dtype.itemsize
            length = int(sizes[index])
            builder.add_document(
                torch.from_numpy(np.asarray(source[start:start + length]).astype(np.int64)),
                [length],
            )
    builder.finalize(str(idx_path))

    merged_sizes, _, merged_dtype = read_index(str(out_dir / args.output_prefix))
    report = {
        "schema": "conversation_merged_train_v1",
        "shards": shards,
        "shard_count": len(shards),
        "documents": int(all_lengths.size),
        "merged_documents": int(merged_sizes.size),
        "tokens": int(all_lengths.sum(dtype=np.int64)),
        "merged_tokens": int(merged_sizes.sum(dtype=np.int64)),
        "dtype": str(dtype),
        "merged_dtype": str(merged_dtype),
        "document_lengths_identical": bool(np.array_equal(all_lengths, merged_sizes)),
        "idx_sha256": hashlib.sha256(idx_path.read_bytes()).hexdigest(),
        "bin_bytes": bin_path.stat().st_size,
    }
    if not report["document_lengths_identical"]:
        raise SystemExit("merged document lengths differ from the shard sequence")
    if report["tokens"] != report["merged_tokens"]:
        raise SystemExit("merged token count differs")
    (out_dir / "merge_metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "shards"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
