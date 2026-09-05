#!/usr/bin/env python3
"""Clean per-layer B32 histograms: distribution + valley marker only."""
import argparse, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def hist(z, m, li):
    return z[f"hist__{m}__counts"][li].astype(float), z[f"hist__{m}__edges"]

def find_valley(c, e, smooth=25):
    mid = 0.5 * (e[:-1] + e[1:]); d = np.convolve(c, np.ones(smooth) / smooth, mode="same")
    peak = int(np.argmax(d)); tail_peak = int(len(d) - 30 + np.argmax(d[-30:]))
    if tail_peak <= peak: return None
    j = peak + int(np.argmin(d[peak:tail_peak + 1]))
    if d[peak] / max(d[j], 1e-12) > 2 and d[tail_peak] / max(d[j], 1e-12) > 1.5:
        return float(mid[j]), float(c[mid >= mid[j]].sum() / c.sum() * 100)
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hist", required=True); ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--metric", default="token_min_b_32"); a = ap.parse_args()
    z = np.load(a.hist); m = a.metric
    fig, axes = plt.subplots(4, 2, figsize=(11, 16)); axes = axes.ravel(); rep = {}
    for li, L in enumerate(range(2, 10)):
        ax = axes[li]; c, e = hist(z, m, li)
        ax.bar(e[:-1], c / c.sum(), width=np.diff(e), color="#7c8ba1", align="edge")
        v = find_valley(c, e)
        if v:
            ax.axvline(v[0], color="black", lw=2.0, label=f"valley = {v[0]:.4f}  (top {v[1]:.2f}%)")
            ax.legend(loc="upper left", fontsize=9, frameon=False)
            rep[f"L{L}"] = {"valley": v[0], "top_pct": v[1]}
        ax.set_yscale("log"); ax.set_xlim(0.6, 1.0); ax.set_ylim(1e-7, 1)
        ax.set_title(f"Layer {L}", fontsize=12)
        ax.set_xlabel("token-level CKA (min over covering 32-token chunks)", fontsize=9)
        ax.set_ylabel("fraction of tokens (log)", fontsize=9)
    fig.suptitle(a.title, fontsize=13)
    fig.tight_layout(); fig.savefig(a.out, dpi=140)
    print(json.dumps(rep, indent=2))
if __name__ == "__main__": main()
