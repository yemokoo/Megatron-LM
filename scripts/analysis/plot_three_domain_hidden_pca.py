#!/usr/bin/env python3
"""Joint PCA of wiki / code / conversation hidden states from one checkpoint.

Answers a single question visually: where does the conversation cloud sit
relative to the code and wiki clouds in the model's own hidden space, layer by
layer?  If conversation barely overlaps code, no anchor selector working inside
conversation can find code-aligned positions; if it does overlap, the selector
has something to find.  Also reports a nearest-neighbour overlap number so the
picture is not the only evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"wiki": "#2563eb", "code": "#f97316", "conversation": "#16a34a"}


def load(path: str):
    with np.load(path, allow_pickle=False) as d:
        return d["hidden_layers"].astype(np.float32), d["layer_numbers"].astype(np.int64)


def pca2(x: np.ndarray, seed: int):
    mu = x.mean(0, keepdims=True)
    xc = x - mu
    # Randomized SVD is enough for a 2-D projection and keeps this fast.
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((xc.shape[1], 8)).astype(np.float32)
    for _ in range(3):
        q, _ = np.linalg.qr(xc.T @ (xc @ q))
    b = xc @ q
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    comps = (vt[:2] @ q.T)
    var = (s[:2] ** 2) / (xc ** 2).sum()
    return xc @ comps.T, var


def knn_overlap(x: np.ndarray, labels: np.ndarray, k: int, rng):
    """Fraction of each domain's k nearest neighbours that belong to each domain."""
    n = x.shape[0]
    idx = rng.choice(n, size=min(n, 3000), replace=False)
    q = x[idx]
    sq = (x ** 2).sum(1)
    d = sq[None, :] - 2 * q @ x.T + sq[idx][:, None]
    d[np.arange(len(idx)), idx] = np.inf
    nn = np.argpartition(d, k, axis=1)[:, :k]
    out = {}
    doms = sorted(set(labels.tolist()))
    for a in doms:
        rows = nn[labels[idx] == a]
        if rows.size == 0:
            continue
        neigh = labels[rows].ravel()
        out[a] = {b: float((neigh == b).mean()) for b in doms}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--conversation", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="2,5,7,9")
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    dumps = {"wiki": load(args.wiki), "code": load(args.code), "conversation": load(args.conversation)}
    layer_ids = dumps["wiki"][1]
    for k, (_, l) in dumps.items():
        assert np.array_equal(l, layer_ids), f"layer axis differs for {k}"
    want = [int(v) for v in args.layers.split(",")]
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(want), figsize=(5.2 * len(want), 5.0), squeeze=False)
    report = {}
    for ax, L in zip(axes[0], want):
        li = int(np.where(layer_ids == L)[0][0])
        parts, labs = [], []
        for name, (h, _) in dumps.items():
            x = h[li]
            sel = rng.choice(x.shape[0], size=min(args.max_points, x.shape[0]), replace=False)
            parts.append(x[sel]); labs.append(np.full(len(sel), name))
        X = np.concatenate(parts); Y = np.concatenate(labs)
        Z, var = pca2(X, args.seed)
        for name in ("wiki", "code", "conversation"):
            m = Y == name
            ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=0.35, c=COLORS[name], label=name, rasterized=True)
        ax.set_title(f"L{L}  (PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%)")
        ax.set_xticks([]); ax.set_yticks([])
        ov = knn_overlap(X, Y, k=10, rng=rng)
        report[f"L{L}"] = {"explained_var": [float(v) for v in var], "knn10_neighbor_fraction": ov}
        c2c = ov.get("conversation", {}).get("code", 0.0)
        c2w = ov.get("conversation", {}).get("wiki", 0.0)
        ax.text(0.02, 0.02, f"conv→code {c2c*100:.1f}%  conv→wiki {c2w*100:.1f}%",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0))
    axes[0][0].legend(loc="upper left", fontsize=8, markerscale=4)
    fig.suptitle("Joint PCA of hidden states — conv-chain KD-init (24E), before conversation training", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "three_domain_pca.png", dpi=150)
    (out / "three_domain_pca.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
