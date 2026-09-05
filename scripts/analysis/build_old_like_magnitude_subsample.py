#!/usr/bin/env python3
"""Build a fixed CPU-only token subsample from paired metric shards.

The resulting bundle is reused by every norm-gate/old-like analysis. Code uses
equal-probability two-stage sampling over equal-size full shards; Wiki uses an
exact uniform sample over all dense token occurrences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


LAYERS = np.arange(2, 10, dtype=np.int16)
METRICS = ("cosine", "relative_l2", "log_norm_ratio", "delta_mse")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def discover(root: Path) -> list[dict]:
    rows = []
    for shard in sorted(root.glob("rank_[0-9][0-9][0-9]/token_metrics/shard_*.npz")):
        sidecar_path = shard.with_suffix(".json")
        with sidecar_path.open(encoding="utf-8") as handle:
            sidecar = json.load(handle)
        rows.append(
            {
                "path": shard,
                "rank": shard.parents[1].name,
                "file": shard.name,
                "start_sample": int(sidecar["start_sample"]),
                "end_sample": int(sidecar["end_sample"]),
                "samples": int(sidecar["samples"]),
                "valid_tokens": int(sidecar["valid_tokens"]),
            }
        )
    if not rows:
        raise RuntimeError(f"no shards below {root}")
    return rows


def choose_code(shards: list[dict], seed: int, target: int, shard_count: int) -> dict[int, np.ndarray]:
    dense = np.asarray([row["samples"] * 512 for row in shards], dtype=np.int64)
    if not np.all(dense == dense[0]):
        raise RuntimeError("Code two-stage sampler requires equal dense shard sizes")
    if target % shard_count:
        raise ValueError("Code target must divide evenly by shard count")
    per_shard = target // shard_count
    if per_shard > int(dense[0]):
        raise ValueError("too many tokens requested per shard")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(shards), size=shard_count, replace=False))
    result = {}
    for ordinal in chosen:
        local_rng = np.random.default_rng(seed ^ ((int(ordinal) + 1) * 0x9E3779B1))
        result[int(ordinal)] = np.sort(
            local_rng.choice(int(dense[ordinal]), size=per_shard, replace=False).astype(np.int64)
        )
    return result


def choose_uniform(shards: list[dict], seed: int, target: int) -> dict[int, np.ndarray]:
    sizes = np.asarray([row["samples"] * 512 for row in shards], dtype=np.int64)
    total = int(sizes.sum())
    if target > total:
        raise ValueError(f"target {target} exceeds total {total}")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=target, replace=False).astype(np.int64))
    cumulative = np.cumsum(sizes)
    result = {}
    for ordinal, end in enumerate(cumulative):
        start = 0 if ordinal == 0 else int(cumulative[ordinal - 1])
        left = int(np.searchsorted(selected, start, side="left"))
        right = int(np.searchsorted(selected, int(end), side="left"))
        if right > left:
            result[ordinal] = selected[left:right] - start
    return result


def extract(shards: list[dict], selections: dict[int, np.ndarray], domain: str) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "sample_ids": [], "positions": [], "token_ids": [], "shard_ordinals": [],
        **{name: [] for name in METRICS},
    }
    for ordinal in sorted(selections):
        row = shards[ordinal]
        flat = selections[ordinal]
        with np.load(row["path"], allow_pickle=False) as data:
            if not np.array_equal(data["layer_numbers"], np.arange(1, 10)):
                raise RuntimeError(f"layer mismatch in {row['path']}")
            valid = data["valid_mask"].reshape(-1)
            if not np.all(valid[flat] == 1):
                raise RuntimeError(f"sampled invalid token in {row['path']}")
            sequence = data["valid_mask"].shape[1]
            local_sample = flat // sequence
            chunks["sample_ids"].append(data["sample_ids"][local_sample].astype(np.int64))
            chunks["positions"].append((flat % sequence).astype(np.uint16))
            chunks["token_ids"].append(data["input_token_ids"].reshape(-1)[flat].astype(np.int32))
            chunks["shard_ordinals"].append(np.full(flat.size, ordinal, dtype=np.uint16))
            for metric in METRICS:
                values = data[metric].reshape(-1, 9)[flat, 1:]
                chunks[metric].append(values.astype(np.float32))
        print(f"[{domain}] shard {ordinal + 1}/{len(shards)} selected={flat.size:,}", flush=True)
    result = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    count = result["sample_ids"].size
    for name in METRICS:
        if result[name].shape != (count, 8) or not np.isfinite(result[name]).all():
            raise RuntimeError(f"invalid extracted {name}: {result[name].shape}")
    relative = result["relative_l2"]
    delta = result["delta_mse"]
    valid_proxy = (relative > 1e-12) & (delta > 1e-24)
    reference_rms = np.full(relative.shape, np.nan, dtype=np.float32)
    reference_rms[valid_proxy] = np.sqrt(delta[valid_proxy]) / relative[valid_proxy]
    if not np.isfinite(reference_rms).all() or np.any(reference_rms <= 0):
        raise RuntimeError("reference RMS proxy contains invalid entries")
    result["reference_rms"] = reference_rms
    result["symmetric_relative_l2"] = (
        2.0 * relative / (1.0 + np.exp(result["log_norm_ratio"].astype(np.float64)))
    ).astype(np.float32)
    result["layer_numbers"] = LAYERS
    return result


def shard_manifest(shards: list[dict], selections: dict[int, np.ndarray]) -> list[dict]:
    return [
        {
            "ordinal": ordinal,
            "path": str(shards[ordinal]["path"]),
            "rank": shards[ordinal]["rank"],
            "file": shards[ordinal]["file"],
            "start_sample": shards[ordinal]["start_sample"],
            "end_sample": shards[ordinal]["end_sample"],
            "dense_tokens": shards[ordinal]["samples"] * 512,
            "selected_tokens": int(selections[ordinal].size),
            "selected_flat_index_sha256": hashlib.sha256(selections[ordinal].tobytes()).hexdigest(),
        }
        for ordinal in sorted(selections)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--wiki-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--code-tokens", type=int, default=2_000_000)
    parser.add_argument("--wiki-tokens", type=int, default=2_000_000)
    parser.add_argument("--code-shards", type=int, default=16)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "subsample_metadata.json"
    code_path = output / "code_subsample.npz"
    wiki_path = output / "wiki_subsample.npz"
    if metadata_path.is_file() and code_path.is_file() and wiki_path.is_file():
        metadata = json.load(metadata_path.open(encoding="utf-8"))
        expected = {
            "seed": args.seed, "code_tokens": args.code_tokens, "wiki_tokens": args.wiki_tokens,
            "code_shards_selected": args.code_shards,
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            print(json.dumps({"cached": True, "metadata": str(metadata_path)}, indent=2))
            return
        raise RuntimeError("existing subsample metadata differs; use a new output directory")

    code_shards = discover(args.code_root.resolve())
    wiki_shards = discover(args.wiki_root.resolve())
    code_selection = choose_code(code_shards, args.seed, args.code_tokens, args.code_shards)
    wiki_selection = choose_uniform(wiki_shards, args.seed + 1, args.wiki_tokens)
    code = extract(code_shards, code_selection, "Code")
    wiki = extract(wiki_shards, wiki_selection, "Wiki")
    atomic_npz(code_path, code)
    atomic_npz(wiki_path, wiki)
    metadata = {
        "schema": "old_like_magnitude_subsample_v1",
        "complete": True,
        "seed": args.seed,
        "code_root": str(args.code_root.resolve()),
        "wiki_root": str(args.wiki_root.resolve()),
        "code_tokens": int(code["sample_ids"].size),
        "wiki_tokens": int(wiki["sample_ids"].size),
        "code_total_shards": len(code_shards),
        "wiki_total_shards": len(wiki_shards),
        "code_shards_selected": args.code_shards,
        "sampling": {
            "code": "uniform shards without replacement, then equal uniform tokens without replacement per equal-size shard",
            "wiki": "exact uniform tokens without replacement over all dense token occurrences",
        },
        "layers": LAYERS.tolist(),
        "hidden_size": 1024,
        "reference_rms_formula": "sqrt(delta_mse)/relative_l2 = ||reference||/sqrt(hidden_size)",
        "code_shard_manifest": shard_manifest(code_shards, code_selection),
        "wiki_shard_manifest": shard_manifest(wiki_shards, wiki_selection),
        "gt_assigned": False,
    }
    atomic_json(metadata_path, metadata)
    print(json.dumps({"complete": True, "code_tokens": args.code_tokens, "wiki_tokens": args.wiki_tokens,
                      "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
