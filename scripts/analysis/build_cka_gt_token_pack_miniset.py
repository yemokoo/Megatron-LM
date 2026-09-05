"""Build a token-pack replay miniset from exact CKA GT records (cosine-style).

Unlike the contextual miniset, the original 512-token windows are NOT kept:
only the GT token ids are concatenated in original (window, position) order and
cut into 512-token samples.  Every position of every sample is supervised
(mask = 1) except the EOD padding of the final partial sample.  Each GT token id
is re-read from the source dataset and compared with the recorded token id
before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

SCHEMA = "cka_gt_token_pack_miniset_v1"
_IDX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {4: np.int32, 8: np.uint16}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_index(prefix: str):
    path = Path(f"{prefix}.idx")
    with path.open("rb") as handle:
        assert handle.read(9) == _IDX_MAGIC
        struct.unpack("<Q", handle.read(8))
        (code,) = struct.unpack("<B", handle.read(1))
        (count,) = struct.unpack("<Q", handle.read(8))
        struct.unpack("<Q", handle.read(8))
        offset = handle.tell()
    dtype = np.dtype(_DTYPE_CODES[code])
    sizes = np.asarray(np.memmap(path, dtype=np.int32, mode="r", offset=offset, shape=(count,)))
    pointers = np.asarray(np.memmap(path, dtype=np.int64, mode="r",
                                    offset=offset + count * 4, shape=(count,)))
    return sizes, pointers, dtype


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--selector-label", required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()

    seq = args.sequence_length
    records = np.load(args.occurrences)
    order = np.lexsort((records["position"].astype(np.int64),
                        records["source_window_index"].astype(np.int64)))
    records = records[order]
    n = int(records.size)

    # Re-read every GT token from the source corpus: doc pointer + window offset + position.
    sizes, pointers, dtype = read_index(args.source_prefix)
    source = np.memmap(args.source_prefix + ".bin", dtype=dtype, mode="r")
    doc = records["document_id"].astype(np.int64)
    flat = pointers[doc] // dtype.itemsize + records["window_offset"].astype(np.int64) \
        + records["position"].astype(np.int64)
    if np.any(records["window_offset"].astype(np.int64) + records["position"].astype(np.int64)
              >= sizes[doc]):
        raise SystemExit("an occurrence points past the end of its document")
    reread = np.asarray(source[flat])
    mismatches = int(np.count_nonzero(reread.astype(np.int64) != records["token_id"].astype(np.int64)))
    if mismatches:
        raise SystemExit(f"{mismatches} occurrences disagree with recorded token ids")

    EOD = 0
    count = (n + seq - 1) // seq
    tokens = np.full(count * seq, EOD, dtype=dtype)
    tokens[:n] = records["token_id"].astype(dtype)
    mask = np.zeros(count * seq, dtype=bool)
    mask[:n] = True
    mask = mask.reshape(count, seq)
    packed = np.packbits(mask, axis=1, bitorder="little")
    popcount = int(np.unpackbits(packed, axis=1, bitorder="little")[:, :seq].sum())
    if popcount != n:
        raise SystemExit(f"popcount {popcount} != {n}")
    pad = count * seq - n

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Megatron-LM"))
    import torch

    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path, idx_path = out_dir / "train_text_document.bin", out_dir / "train_text_document.idx"
    builder = IndexedDatasetBuilder(str(bin_path), dtype=dtype.type, multimodal=False)
    builder.add_document(torch.from_numpy(tokens.astype(np.int64)), [int(tokens.size)])
    builder.finalize(str(idx_path))
    np.save(out_dir / "old_like_gt_packed.npy", packed)

    gt_root = Path(args.gt_root)
    rank_dir = gt_root / "rank_000"
    rank_dir.mkdir(parents=True, exist_ok=True)
    np.save(rank_dir / "old_like_gt_packed.npy", packed)
    config = {"selector": args.selector_label, "sequence_length": seq,
              "total_samples": count, "context_preserved": False, "token_pack": True}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    common = {"schema": SCHEMA, "complete": True,
              "sequence_length": seq, "config_sha256": config_sha}
    (rank_dir / "metadata.json").write_text(json.dumps({
        **common, "partition_start_sample": 0, "partition_samples": count,
        "mask_file": "old_like_gt_packed.npy", "mask_shape": [count, packed.shape[1]],
        "mask_dtype": "uint8", "bitorder": "little"}, indent=2, sort_keys=True))
    (gt_root / "metadata.json").write_text(json.dumps({
        **common, "total_samples": count, "config": config,
        "partitions": [{"rank": "rank_000", "partition_start_sample": 0,
                        "partition_samples": count, "mask_file": "old_like_gt_packed.npy"}],
        "dataset_identity": {"sequence_length": seq, "total_samples": count,
                             "split": "100,0,0", "blend_mode": "exhaustive",
                             "shards": [{"prefix": str((out_dir / "train_text_document").resolve()),
                                         "idx_bytes": idx_path.stat().st_size,
                                         "bin_bytes": bin_path.stat().st_size,
                                         "idx_sha256": _sha256(idx_path)}]},
    }, indent=2, sort_keys=True))
    positive = np.arange(count, dtype=np.int32)
    subset = gt_root / "replay_positive_sample_ids.npy"
    np.save(subset, positive)
    (gt_root / "replay_subset_metadata.json").write_text(json.dumps({
        "schema": "old_like_gt_positive_sample_subset_v1", "complete": True,
        "source_gt_schema": SCHEMA, "source_gt_config_sha256": config_sha,
        "source_total_samples": count, "subset_file": "replay_positive_sample_ids.npy",
        "dtype": "int32", "positive_sample_count": count,
        "subset_sha256": _sha256(subset)}, indent=2, sort_keys=True))

    report = {
        "schema": SCHEMA, "selector": args.selector_label,
        "occurrences_file": args.occurrences, "source_prefix": args.source_prefix,
        "sequence_length": seq, "window_count": count,
        "document_count": int(np.unique(doc).size),
        "source_window_count": int(np.unique(records["source_window_index"]).size),
        "miniset_tokens": int(tokens.size), "gt_occurrences": n,
        "pad_tokens": int(pad), "pad_token_id": EOD,
        "gt_density": n / float(tokens.size),
        "token_identity_comparisons": n,
        "token_identity_mismatches": 0, "packed_popcount": popcount,
        "ordering": "source_window_then_position",
        "context_preserved": False, "tokens_concatenated_out_of_context": True,
        "gt_root": str(gt_root), "written": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
