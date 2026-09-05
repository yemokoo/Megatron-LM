#!/usr/bin/env python3
"""Per-token hidden drift from the wiki-only model, in the full 1024-d space.

Input: the .npz dumps written by dump_wiki_drift_all.sh (hidden_layers
[L, N, D], layer_numbers, token_ids, positions, sample_indices) plus the
wiki next_token_acc files.  Every dump was produced from the same probe stream,
and that is asserted here before anything is compared: if token identity
differs, the pair is refused rather than silently compared.

For each (method, stage) pair against its wiki-only reference and each layer:
  d_i      = ||h_i^after - h_i^before||_2          primary, histogrammed
  cos_i    = cos(h_i^after, h_i^before)              secondary
  centroid = ||mean(h^after) - mean(h^before)||     rigid shift
Raw L2 is the headline -- the question is how far each token moved, not how
large it started; a relative column (d_i / ||h_i^before||) is emitted alongside
for the methods whose wiki-only reference is a different model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, reference id, after-code id, after-conv id)
METHODS = [
    ("Seq",        "dense_wiki", "seq_code",       "seq_conv"),
    ("EWC",        "dense_wiki", "ewc_code",       "ewc_conv"),
    ("GEM",        "dense_wiki", "gem_code",       "gem_conv"),
    ("SLoRA-r16",  "dense_wiki", "slora16_code",   "slora16_conv"),
    ("SLoRA-r64",  "dense_wiki", "slora64_code",   "slora64_conv"),
    ("O-LoRA",     "olora_wiki", "olora_code",     "olora_conv"),
    ("fixed-MoE",  "fmoe_wiki",  "fmoe_code",      "fmoe_conv"),
    ("Ours",       "ours_wiki",  "ours_code_hmse", "ours_conv"),
    ("Ours(code=lm)",     "ours_wiki", "ours_code_lm", None),
    # HF kd_1phase/ffn_attn_shared_router: FFN + QKVO attention experts, shared
    # router, KD-init + one-phase lm replay.  Its own wiki e8 reference
    # (a different wiki pretrain from the FFN-only e8); frozen parts verified
    # bit-identical across wiki -> code -> conv.
    ("Ours-attn",  "oursattn_wiki", "oursattn_code", "oursattn_conv"),
    # MoE-LPR trained from the same FFN-only e8 (same lineage as ours_wiki):
    # expansion + new-experts-only training, then a supervised task-group
    # router retune (LPR loss) on the old+new mix.  "pre" = before retune.
    ("MoE-LPR",        "ours_wiki", "lpr_code",     "lpr_conv"),
    ("MoE-LPR(pre)",   "ours_wiki", "lpr_code_pre", "lpr_conv_pre"),
    # MoE-LPR with the Ours-attn expert layout (FFN + attention experts, shared
    # router), from the same HF hybrid e8 as Ours-attn.
    ("MoE-LPR-attn",      "oursattn_wiki", "lprhyb_code",     "lprhyb_conv"),
    ("MoE-LPR-attn(pre)", "oursattn_wiki", "lprhyb_code_pre", "lprhyb_conv_pre"),
    # HF baseline/ffn_only (expansion + fine-tune, then router retune) is NOT
    # paired: it was expanded from the a100 g2matched e8 (mb128), which is
    # neither on disk nor in the HF repo, and its attention differs from the
    # FFN-only e8 here by ~2e-4.  Dumps exist (moeft_*) but have no reference.
    # ours_conv_norep (fingerprint_router_geometry_20260809) descends from a
    # different e8 run: its embedding, layer-0 FFN, attention and even the
    # original experts differ from ours_wiki by ~1e-3, so drift against
    # ours_wiki would mix lineage with forgetting.  Kept as a dump, not a pair.
]
HIGHLIGHT = {"Ours"}
# Layer 1 is frozen in every method (baselines train layers 2-9; the MoE stack
# is dense at layer 1 and never trained after wiki), so its drift is exactly 0
# and its cosine exactly 1.  Averaging it in only adds a constant floor.
SUMMARY_LAYERS_FROM = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data2/seonghyeonnoh/LLM-continual-learning-runs/hidden_drift_wiki_20260826")
    p.add_argument("--out", default=None, help="defaults to <root>/analysis")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--clip-percentile", type=float, default=99.5,
                   help="shared histogram range: this percentile of the largest drift across methods")
    return p.parse_args()


def load_dump(path: Path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        return {
            "hidden": z["hidden_layers"].astype(np.float32),   # [L, N, D]
            "layers": z["layer_numbers"].astype(np.int64),
            "token_ids": z["token_ids"].astype(np.int64),
            "positions": z["positions"].astype(np.int64),
            "samples": z["sample_indices"].astype(np.int64),
            "meta": meta,
        }


def assert_aligned(a, b, name_a, name_b):
    for key in ("token_ids", "positions", "samples"):
        if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
            raise RuntimeError(
                f"probe stream mismatch on {key}: {name_a} vs {name_b} "
                f"({a[key].shape} vs {b[key].shape}); the dumps were not produced "
                "under the same seed/seq/batch/iters contract")
    if not np.array_equal(a["layers"], b["layers"]):
        raise RuntimeError(f"layer sets differ: {name_a}={a['layers']} {name_b}={b['layers']}")


def read_acc(path: Path):
    if not path.is_file():
        return None
    m = re.search(r"next_token_acc: ([0-9.]+)", path.read_text())
    return float(m.group(1)) if m else None


def per_token_metrics(before, after):
    """Returns dict of [L, N] arrays plus per-layer scalars."""
    hb, ha = before["hidden"], after["hidden"]
    diff = ha - hb
    l2 = np.linalg.norm(diff, axis=-1)                                   # [L, N]
    nb = np.linalg.norm(hb, axis=-1)
    na = np.linalg.norm(ha, axis=-1)
    cos = np.einsum("lnd,lnd->ln", ha, hb) / np.maximum(na * nb, 1e-8)
    rel = l2 / np.maximum(nb, 1e-8)
    centroid = np.linalg.norm(ha.mean(axis=1) - hb.mean(axis=1), axis=-1)  # [L]
    return {"l2": l2, "cos": cos, "rel": rel, "centroid": centroid, "norm_before": nb}


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    dumps, missing = {}, []
    for _, ref, code, conv in METHODS:
        for i in (ref, code, conv):
            if i and i not in dumps:
                p = root / "hidden" / f"{i}.npz"
                if not p.is_file():
                    missing.append(i)
                    continue
                try:
                    dumps[i] = load_dump(p)
                except Exception as exc:   # a dump still being written, or truncated
                    print(f"[WARN] unreadable dump {p.name} ({type(exc).__name__}); treated as missing")
                    missing.append(i)
    if missing:
        print("[WARN] missing dumps, pairs using them are skipped:", sorted(set(missing)))

    # ---- pairwise metrics -------------------------------------------------
    rows, hist_data = [], {}      # hist_data[(label, stage)] = l2 [L, N]
    layers = None
    for label, ref, code, conv in METHODS:
        if ref not in dumps:
            continue
        for stage, sid in (("code", code), ("conv", conv)):
            if not sid or sid not in dumps:
                continue
            assert_aligned(dumps[ref], dumps[sid], ref, sid)
            m = per_token_metrics(dumps[ref], dumps[sid])
            layers = dumps[ref]["layers"]
            hist_data[(label, stage)] = m["l2"]
            acc_ref = read_acc(root / "acc" / f"{ref}.txt")
            acc_now = read_acc(root / "acc" / f"{sid}.txt")
            for li, layer in enumerate(layers):
                rows.append({
                    "method": label, "stage": stage, "ref": ref, "ckpt": sid, "layer": int(layer),
                    "l2_median": float(np.median(m["l2"][li])),
                    "l2_mean": float(m["l2"][li].mean()),
                    "l2_p90": float(np.percentile(m["l2"][li], 90)),
                    "rel_median": float(np.median(m["rel"][li])),
                    "cos_median": float(np.median(m["cos"][li])),
                    "cos_gt_0.99": float((m["cos"][li] > 0.99).mean()),
                    "centroid_shift": float(m["centroid"][li]),
                    "norm_before_median": float(np.median(m["norm_before"][li])),
                    "acc_wiki_ref": acc_ref, "acc_wiki_now": acc_now,
                    "acc_drop": (acc_ref - acc_now) if (acc_ref is not None and acc_now is not None) else None,
                })
    if not rows:
        raise SystemExit("nothing to analyse -- no complete pairs")

    with (out / "drift_per_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- summary table (layer-averaged) ------------------------------------
    summary = {}
    for r in rows:
        if r["layer"] < SUMMARY_LAYERS_FROM:
            continue
        k = (r["method"], r["stage"])
        s = summary.setdefault(k, {"l2": [], "rel": [], "cos99": [], "cent": [],
                                   "acc_ref": r["acc_wiki_ref"], "acc_now": r["acc_wiki_now"]})
        s["l2"].append(r["l2_median"]); s["rel"].append(r["rel_median"])
        s["cos99"].append(r["cos_gt_0.99"]); s["cent"].append(r["centroid_shift"])
    lines = [f"{'method':<16}{'stage':<6}{'L2 med (layers 2-9)':>20}{'rel med':>10}{'cos>.99':>9}"
             f"{'centroid':>10}{'acc ref':>9}{'acc now':>9}{'Δacc':>8}"]
    lines.append("-" * len(lines[0]))
    scatter = []
    for (label, stage), s in summary.items():
        l2 = float(np.mean(s["l2"])); rel = float(np.mean(s["rel"]))
        c99 = float(np.mean(s["cos99"])); cent = float(np.mean(s["cent"]))
        ar, an = s["acc_ref"], s["acc_now"]
        d = (ar - an) if (ar is not None and an is not None) else None
        lines.append(f"{label:<16}{stage:<6}{l2:>20.4f}{rel:>10.4f}{c99:>9.3f}{cent:>10.4f}"
                     f"{(f'{ar:.4f}' if ar is not None else '-'):>9}"
                     f"{(f'{an:.4f}' if an is not None else '-'):>9}"
                     f"{(f'{d:+.4f}' if d is not None else '-'):>8}")
        if d is not None and stage == "conv":
            scatter.append((label, l2, d))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- drift vs acc drop (conv stage) -------------------------------------
    if len(scatter) >= 3:
        from scipy.stats import spearmanr
        xs = np.array([s[1] for s in scatter]); ys = np.array([s[2] for s in scatter])
        rho, pval = spearmanr(xs, ys)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for label, x, y in scatter:
            ax.scatter(x, y, s=60 if label in HIGHLIGHT else 30, c="#d62728" if label in HIGHLIGHT else "#1f77b4")
            ax.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("median L2 drift from wiki-only (layer avg)")
        ax.set_ylabel("wiki next_token_acc drop (wiki-only − after conv)")
        ax.set_title(f"drift vs forgetting  (Spearman ρ={rho:.2f}, p={pval:.3f})")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "drift_vs_acc_drop.png", dpi=160); plt.close(fig)
        (out / "spearman.json").write_text(json.dumps({"rho": float(rho), "p": float(pval), "n": len(scatter)}, indent=1))
        print(f"\nSpearman(drift, acc drop) over {len(scatter)} methods: rho={rho:.3f} p={pval:.3f}")

    # ---- per-layer histograms, one figure per stage, methods overlaid ------
    n_layers = len(layers)
    for stage in ("code", "conv"):
        keys = [k for k in hist_data if k[1] == stage]
        if not keys:
            continue
        hi = np.percentile(np.concatenate([hist_data[k].ravel() for k in keys]), args.clip_percentile)
        edges = np.linspace(0.0, hi, args.bins + 1)
        ncols = 3; nrows = int(np.ceil(n_layers / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
        for li, layer in enumerate(layers):
            ax = axes[li // ncols][li % ncols]
            for label, _ in keys:
                v = np.clip(hist_data[(label, stage)][li], 0, hi)
                ours = label in HIGHLIGHT
                ax.hist(v, bins=edges, density=True, histtype="step",
                        linewidth=2.2 if ours else 1.1, alpha=1.0 if ours else 0.75,
                        color="#d62728" if ours else None, label=label)
            ax.set_title(f"layer {layer}"); ax.set_xlim(0, hi)
            if li == 0:
                ax.legend(fontsize=7, frameon=False)
        for j in range(n_layers, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"per-token L2 drift from wiki-only, after {stage} (wiki test probe, {hist_data[keys[0]].shape[1]:,} tokens)")
        fig.tight_layout(); fig.savefig(out / f"hist_l2_{stage}.png", dpi=150); plt.close(fig)

    # ---- layer curve: median drift by depth ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, stage in zip(axes, ("code", "conv")):
        for label, _ in [k for k in hist_data if k[1] == stage]:
            med = np.median(hist_data[(label, stage)], axis=1)
            ours = label in HIGHLIGHT
            ax.plot(layers, med, marker="o", linewidth=2.4 if ours else 1.2,
                    color="#d62728" if ours else None, label=label)
        ax.set_title(f"after {stage}"); ax.set_xlabel("layer"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("median L2 drift from wiki-only")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / "layer_curve_l2.png", dpi=160); plt.close(fig)

    # ---- trajectory: wiki→code→conv per method (layer-avg median) -----------
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for label, ref, code, conv in METHODS:
        pts = [0.0]
        keep = layers >= SUMMARY_LAYERS_FROM
        for stage, sid in (("code", code), ("conv", conv)):
            if (label, stage) in hist_data:
                pts.append(float(np.mean(np.median(hist_data[(label, stage)][keep], axis=1))))
        if len(pts) > 1:
            ours = label in HIGHLIGHT
            ax.plot(range(len(pts)), pts, marker="o", linewidth=2.4 if ours else 1.2,
                    color="#d62728" if ours else None, label=label)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["wiki", "+code", "+conv"])
    ax.set_ylabel("median L2 drift from wiki-only"); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / "trajectory_l2.png", dpi=160); plt.close(fig)

    print(f"\n[OUT] {out}")


if __name__ == "__main__":
    main()
