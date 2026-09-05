#!/usr/bin/env python3
"""Merge data-parallel generation shards into one corpus directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    mats, texts, routing, stats = [], [], None, None
    for sd in a.shards:
        sd = Path(sd)
        with np.load(sd / "tokens.npz") as z:
            mats.append(z["tokens"])
        texts.extend(json.loads(l) for l in (sd / "text.jsonl").open())
        r = json.loads((sd / "routing.json").read_text())
        if routing is None:
            routing = r
            routing["per_layer_top1_counts"] = {k: list(v) for k, v in r["per_layer_top1_counts"].items()}
            routing["total_top1_counts"] = list(r["total_top1_counts"])
        else:
            for k, v in r["per_layer_top1_counts"].items():
                routing["per_layer_top1_counts"][k] = [
                    x + y for x, y in zip(routing["per_layer_top1_counts"][k], v)]
            routing["total_top1_counts"] = [
                x + y for x, y in zip(routing["total_top1_counts"], r["total_top1_counts"])]
        s = json.loads((sd / "stats.json").read_text())
        if stats is None:
            stats = dict(s)
        else:
            stats["num_seqs"] += s["num_seqs"]
            stats["tokens_generated"] += s["tokens_generated"]
            stats["wall_seconds"] = max(stats["wall_seconds"], s["wall_seconds"])

    width = max(m.shape[1] for m in mats)
    pad = int(mats[0][0, -1]) if False else None
    # pad value: reuse the value the sampler padded with (last column of a short row is pad)
    padv = int(np.bincount(np.concatenate([m[:, -1] for m in mats])).argmax())
    merged = np.full((sum(m.shape[0] for m in mats), width), padv, dtype=np.int32)
    row = 0
    for m in mats:
        merged[row:row + m.shape[0], :m.shape[1]] = m
        row += m.shape[0]
    np.savez_compressed(out / "tokens.npz", tokens=merged)

    with (out / "text.jsonl").open("w") as fh:
        for i, rec in enumerate(texts):
            rec["i"] = i
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = np.asarray(routing["total_top1_counts"], dtype=float)
    routing["total_top1_fraction"] = (total / max(total.sum(), 1)).tolist()
    routing["experts_used"] = int((total > 0).sum())
    (out / "routing.json").write_text(json.dumps(routing, indent=1))
    stats["shards"] = [str(s) for s in a.shards]
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[merge] {len(a.shards)} shards -> {merged.shape[0]} seqs, "
          f"{stats['tokens_generated']:,} tokens -> {out}")


if __name__ == "__main__":
    main()
