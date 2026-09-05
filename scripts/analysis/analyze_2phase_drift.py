#!/usr/bin/env python3
"""Hidden-drift histograms for the 2-phase A/B/C/D set (+ F, G reference arms).

Per token, per stage, per probe: D(t) = mean over layers 2-9 of the raw (full
1024-d) L2 distance between that stage's hidden state and the wiki-only
model's hidden state for the same token.  No sampling: every dumped token goes
into the histogram.  Layer 1 is frozen everywhere and excluded (see
analyze_hidden_drift.py for why).

Methods:
  A = ffn/kd0, B = hyb/kd0, C = ffn/kd1, D = hyb/kd1   (5-checkpoint 2-phase chains)
  F = HF kd_1phase one_phase (real full-data 1-phase)   -- s2 (code) / s4 (conv) only
  G = selfgen_replay_Gnew_20260828 (replay-free 1-phase) -- s2 / s4 only

Input: dump_2phase_drift.sh output (<root>/hidden/<id>.npz), id = <label>_<stage>_<probe>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STAGE_LABEL = {"s1": "+code (phase-1)", "s2": "+code router FT", "s3": "+conv (phase-1)", "s4": "+conv router FT"}
STAGES = ["s1", "s2", "s3", "s4"]
PROBES = ["wiki", "code"]
METHODS = ["A", "B", "C", "D"]
METHOD_BASE = {"A": "ffn", "B": "hyb", "C": "ffn", "D": "hyb"}          # which s0 reference each uses
REFS = ["F", "G"]                                                       # only at s2 (code) / s4 (conv)
REF_STAGE = {"F": {"s2": "code/one_phase", "s4": "conversation/one_phase"},
             "G": {"s2": "code_1phase", "s4": "conv_1phase"}}
COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#9467bd", "F": "#555555", "G": "#d62728"}
SUMMARY_LAYERS_FROM = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data2/seonghyeonnoh/LLM-continual-learning-runs/hidden_drift_2phase_20260829")
    p.add_argument("--out", default=None)
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--clip-percentile", type=float, default=99.5)
    return p.parse_args()


def load_dump(path: Path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        return {
            "hidden": z["hidden_layers"].astype(np.float32),
            "layers": z["layer_numbers"].astype(np.int64),
            "token_ids": z["token_ids"].astype(np.int64),
            "positions": z["positions"].astype(np.int64),
            "samples": z["sample_indices"].astype(np.int64),
            "meta": meta,
        }


def assert_aligned(a, b, name_a, name_b):
    for key in ("token_ids", "positions", "samples"):
        if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
            raise RuntimeError(f"probe stream mismatch on {key}: {name_a} vs {name_b}")
    if not np.array_equal(a["layers"], b["layers"]):
        raise RuntimeError(f"layer sets differ: {name_a}={a['layers']} {name_b}={b['layers']}")


def token_score(before, after):
    """D(t) = mean over layers 2..9 of ||h_after(t) - h_before(t)||_2."""
    layers = before["layers"]
    keep = layers >= SUMMARY_LAYERS_FROM
    diff = after["hidden"][keep] - before["hidden"][keep]          # [L', N, D]
    l2 = np.linalg.norm(diff, axis=-1)                              # [L', N]
    return l2.mean(axis=0), l2                                       # ([N], [L', N]) token score, per-layer


def try_load(root: Path, cache: dict, sid: str):
    if sid in cache:
        return cache[sid]
    p = root / "hidden" / f"{sid}.npz"
    if not p.is_file():
        cache[sid] = None
        return None
    try:
        cache[sid] = load_dump(p)
    except Exception as exc:
        print(f"[WARN] unreadable dump {p.name} ({type(exc).__name__}); treated as missing")
        cache[sid] = None
    return cache[sid]


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    # scores[probe][stage][label] = token D(t) array; layer_curve[probe][stage][label] = per-layer median
    scores = {p: {s: {} for s in STAGES} for p in PROBES}
    layer_curve = {p: {s: {} for s in STAGES} for p in PROBES}
    layers_seen = None

    for probe in PROBES:
        for m in METHODS:
            base = try_load(root, cache, f"{METHOD_BASE[m]}_s0_{probe}")
            if base is None:
                print(f"[WARN] missing s0 reference for {m}/{probe}"); continue
            for stage in STAGES:
                ck = try_load(root, cache, f"{m}_{stage}_{probe}")
                if ck is None:
                    continue
                assert_aligned(base, ck, f"{METHOD_BASE[m]}_s0", f"{m}_{stage}")
                tok, per_layer = token_score(base, ck)
                scores[probe][stage][m] = tok
                layer_curve[probe][stage][m] = np.median(per_layer, axis=1)
                layers_seen = ck["layers"][ck["layers"] >= SUMMARY_LAYERS_FROM]
        for ref in REFS:
            base = try_load(root, cache, f"hyb_s0_{probe}")
            if base is None:
                continue
            for stage, _ in REF_STAGE[ref].items():
                ck = try_load(root, cache, f"{ref}_{stage}_{probe}")
                if ck is None:
                    continue
                assert_aligned(base, ck, f"hyb_s0", f"{ref}_{stage}")
                tok, per_layer = token_score(base, ck)
                scores[probe][stage][ref] = tok
                layer_curve[probe][stage][ref] = np.median(per_layer, axis=1)

    if all(not scores[p][s] for p in PROBES for s in STAGES):
        raise SystemExit("nothing to analyse -- no complete pairs found under " + str(root))

    # ---- shared histogram range per probe (percentile across everything at that probe) ----
    xmax = {}
    for probe in PROBES:
        allvals = np.concatenate([v for s in STAGES for v in scores[probe][s].values()]) if any(scores[probe].values()) else np.array([0.0])
        xmax[probe] = float(np.percentile(allvals, args.clip_percentile)) if allvals.size else 1.0

    # ---- MAIN figure: one per probe, 4 panels (s1..s4), A/B/C/D (+F/G at s2/s4) overlaid ----
    for probe in PROBES:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True, sharey=True)
        bins = np.linspace(0, xmax[probe], args.bins)
        for ax, stage in zip(axes, STAGES):
            for label in METHODS + REFS:
                d = scores[probe][stage].get(label)
                if d is None:
                    continue
                ax.hist(d, bins=bins, histtype="step", lw=2, density=True, color=COLORS[label], label=label)
                ax.axvline(np.median(d), color=COLORS[label], lw=1, ls="--", alpha=0.7)
            ax.set_title(f"{STAGE_LABEL[stage]}"); ax.set_xlabel("token drift D(t) = mean L2, layers 2-9")
        axes[0].set_ylabel("density"); axes[0].legend(fontsize=9)
        fig.suptitle(f"Hidden-state drift from wiki-only model -- {probe} probe, full token histograms (no sampling)", fontsize=13)
        fig.tight_layout(); fig.savefig(out / f"main_{probe}.png", dpi=130); plt.close(fig)

    # ---- per-method figure: stages overlaid ----
    for probe in PROBES:
        fig, axes = plt.subplots(1, len(METHODS), figsize=(5.2 * len(METHODS), 4.6), sharex=True, sharey=True)
        bins = np.linspace(0, xmax[probe], args.bins)
        stage_colors = {"s1": "#1f77b4", "s2": "#2ca02c", "s3": "#ff7f0e", "s4": "#d62728"}
        for ax, m in zip(axes, METHODS):
            for stage in STAGES:
                d = scores[probe][stage].get(m)
                if d is None:
                    continue
                ax.hist(d, bins=bins, histtype="step", lw=2, density=True, color=stage_colors[stage], label=STAGE_LABEL[stage])
            ax.set_title(m); ax.set_xlabel("D(t)")
        axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
        fig.suptitle(f"Per-method drift over stages -- {probe} probe", fontsize=13)
        fig.tight_layout(); fig.savefig(out / f"per_method_{probe}.png", dpi=130); plt.close(fig)

    # ---- layer curve ----
    if layers_seen is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, probe in zip(axes, PROBES):
            for label in METHODS + REFS:
                stage = "s4" if label in REFS else "s4"
                lc = layer_curve[probe].get(stage, {}).get(label)
                if lc is None:
                    continue
                ax.plot(layers_seen, lc, "o-", color=COLORS[label], label=label)
            ax.set_xlabel("layer"); ax.set_ylabel("median per-layer L2"); ax.set_title(f"{probe} probe, final stage (s4)")
        axes[0].legend(fontsize=9)
        fig.tight_layout(); fig.savefig(out / "layer_curve_s4.png", dpi=130); plt.close(fig)

    # ---- summary table ----
    lines = [f"{'method':<8}{'stage':<6}{'probe':<6}{'median':>10}{'p90':>10}{'n_tokens':>10}"]
    for probe in PROBES:
        for stage in STAGES:
            for label in METHODS + REFS:
                d = scores[probe][stage].get(label)
                if d is None:
                    continue
                lines.append(f"{label:<8}{stage:<6}{probe:<6}{np.median(d):>10.4f}{np.percentile(d,90):>10.4f}{len(d):>10d}")
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[OUT] {out}")


if __name__ == "__main__":
    main()
