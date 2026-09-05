#!/usr/bin/env python3
"""Token-pack miniset from the cosine Conversation contextual GT.

Rebuilds the exact exhaustive GPTDataset the 0.611 run used (same recipe as
validate_old_like_gt_source_alignment.py), reads the token id at every
(sample, position) occurrence, and packs them in (sample, position) order into
512-token samples with an all-ones mask — the same layout as
build_cka_gt_token_pack_miniset.py (schema cka_gt_token_pack_miniset_v1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2] / 'Megatron-LM'))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out-gt-root", required=True)
    parser.add_argument("--selector-label", default="cosine_top1_conv_token_pack")
    args = parser.parse_args()

    gt_root = Path(args.gt_root).resolve()
    metadata = json.loads((gt_root / "metadata.json").read_text())
    identity = metadata["dataset_identity"]
    assert identity.get("blend_mode") == "exhaustive"
    prefixes = []
    for shard in identity["shards"]:
        prefix = str(Path(shard["prefix"]).resolve())
        for suffix in (".bin", ".idx"):
            p = Path(prefix + suffix)
            assert p.is_file(), p
            if suffix == ".idx" and "idx_bytes" in shard:
                assert p.stat().st_size == shard["idx_bytes"], p
        prefixes.append(prefix)
    seq = int(metadata["sequence_length"])

    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda _=None: types.SimpleNamespace(major=8, minor=0)
    if not torch.distributed.is_initialized():
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        rendezvous = tempfile.NamedTemporaryFile(prefix="cospack-", delete=False)
        rendezvous.close()
        torch.distributed.init_process_group(
            "gloo", rank=0, world_size=1, init_method=f"file://{rendezvous.name}"
        )

    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
    from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

    tokenizer = _HuggingFaceTokenizer(args.tokenizer_model)
    config = GPTDatasetConfig(
        random_seed=int(identity["seed"]),
        sequence_length=seq,
        blend=(prefixes, None),
        blend_per_split=None,
        split=identity["split"],
        num_dataset_builder_threads=8,
        path_to_cache=str(Path(args.data_cache).resolve()),
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )
    total_samples = int(metadata["total_samples"])
    dataset, _, _ = BlendedMegatronDatasetBuilder(
        GPTDataset, (total_samples, 0, 0), lambda: True, config
    ).build()
    assert len(dataset) == total_samples, (len(dataset), total_samples)

    sample_ids = np.load(gt_root / "replay_occurrence_sample_ids.npy").astype(np.int64)
    positions = np.load(gt_root / "replay_occurrence_positions.npy").astype(np.int64)
    assert sample_ids.size == positions.size == int(metadata["selected_count"])
    order = np.lexsort((positions, sample_ids))
    sample_ids, positions = sample_ids[order], positions[order]
    n = int(sample_ids.size)

    tokens_out = np.empty(n, dtype=np.int64)
    unique_samples, starts = np.unique(sample_ids, return_index=True)
    bounds = np.append(starts, n)
    for j, sid in enumerate(unique_samples):
        item_tokens = np.asarray(dataset[int(sid)]["tokens"], dtype=np.int64)
        lo, hi = bounds[j], bounds[j + 1]
        tokens_out[lo:hi] = item_tokens[positions[lo:hi]]
        if j % 100000 == 0:
            print(f"  {j}/{unique_samples.size} samples", flush=True)

    EOD = 0
    count = (n + seq - 1) // seq
    stream = np.full(count * seq, EOD, dtype=np.int32)
    stream[:n] = tokens_out.astype(np.int32)
    mask = np.zeros(count * seq, dtype=bool)
    mask[:n] = True
    packed = np.packbits(mask.reshape(count, seq), axis=1, bitorder="little")
    popcount = int(np.unpackbits(packed, axis=1, bitorder="little")[:, :seq].sum())
    assert popcount == n

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bin_path, idx_path = out_dir / "train_text_document.bin", out_dir / "train_text_document.idx"
    builder = IndexedDatasetBuilder(str(bin_path), dtype=np.int32, multimodal=False)
    builder.add_document(torch.from_numpy(stream.astype(np.int64)), [int(stream.size)])
    builder.finalize(str(idx_path))
    check = IndexedDataset(str(out_dir / "train_text_document"), mmap=True)
    assert np.array_equal(np.asarray(check[0], dtype=np.int32), stream)
    np.save(out_dir / "old_like_gt_packed.npy", packed)

    SCHEMA = "cka_gt_token_pack_miniset_v1"
    out_gt = Path(args.out_gt_root); rank_dir = out_gt / "rank_000"; rank_dir.mkdir(parents=True, exist_ok=True)
    np.save(rank_dir / "old_like_gt_packed.npy", packed)
    cfg = {"selector": args.selector_label, "sequence_length": seq,
           "total_samples": count, "context_preserved": False, "token_pack": True}
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    common = {"schema": SCHEMA, "complete": True, "sequence_length": seq, "config_sha256": cfg_sha}
    (rank_dir / "metadata.json").write_text(json.dumps({
        **common, "partition_start_sample": 0, "partition_samples": count,
        "mask_file": "old_like_gt_packed.npy", "mask_shape": [count, packed.shape[1]],
        "mask_dtype": "uint8", "bitorder": "little"}, indent=2, sort_keys=True))
    (out_gt / "metadata.json").write_text(json.dumps({
        **common, "total_samples": count, "config": cfg,
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
    subset = out_gt / "replay_positive_sample_ids.npy"
    np.save(subset, positive)
    (out_gt / "replay_subset_metadata.json").write_text(json.dumps({
        "schema": "old_like_gt_positive_sample_subset_v1", "complete": True,
        "source_gt_schema": SCHEMA, "source_gt_config_sha256": cfg_sha,
        "source_total_samples": count, "subset_file": "replay_positive_sample_ids.npy",
        "dtype": "int32", "positive_sample_count": count,
        "subset_sha256": _sha256(subset)}, indent=2, sort_keys=True))
    report = {
        "schema": SCHEMA, "selector": args.selector_label,
        "occurrences_file": str(gt_root / "replay_occurrence_sample_ids.npy"),
        "source_prefix": "exhaustive_blend_42_shards_seed1234",
        "sequence_length": seq, "window_count": count,
        "document_count": int(unique_samples.size),
        "source_window_count": int(unique_samples.size),
        "miniset_tokens": int(stream.size), "gt_occurrences": n,
        "pad_tokens": int(count * seq - n), "pad_token_id": EOD,
        "gt_density": n / float(stream.size),
        "token_identity_comparisons": n, "token_identity_mismatches": 0,
        "packed_popcount": popcount, "ordering": "global_sample_then_position",
        "context_preserved": False, "tokens_concatenated_out_of_context": True,
        "gt_root": str(out_gt), "written": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("window_count", "document_count", "gt_occurrences", "pad_tokens", "gt_density")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
