#!/usr/bin/env python3
"""Per-layer B32 token histograms with every cut candidate drawn on the same axes.

Candidates: the natural valley of the (bimodal) conversation distribution,
conversation self-quantiles, and the wiki / code old-domain bands measured at
the same scale.  Everything is read from merged census histograms; nothing is
recomputed from hidden states.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def hist(z, m, li):
    c = z[f"hist__{m}__counts"][li].astype(float); e = z[f"hist__{m}__edges"]
    uf = float(z[f"hist__{m}__underflow"][li]); of = float(z[f"hist__{m}__overflow"][li])
    return c, e, uf, of

def quant(z, m, li, frac):
    c, e, uf, of = hist(z, m, li); tot = c.sum() + uf + of
    cum = uf + np.cumsum(c); i = min(int(np.searchsorted(cum, frac * tot)), c.size - 1)
    below = uf + (c[:i].sum() if i else 0.0); w = c[i]
    f = 0.0 if w <= 0 else min(max((frac * tot - below) / w, 0.0), 1.0)
    return float(e[i] + f * (e[i + 1] - e[i]))

def valley(z, m, li, lo=0.90, hi=0.999, smooth=15):
    """Lowest smoothed density between the main mode and the near-1 mode."""
    c, e, _, _ = hist(z, m, li); mid = 0.5 * (e[:-1] + e[1:])
    k = np.ones(smooth) / smooth; d = np.convolve(c, k, mode="same")
    band = (mid >= lo) & (mid <= hi)
    if not band.any(): return None
    # Only meaningful if there is a rise after the minimum (a second mode).
    idx = np.where(band)[0]; j = idx[np.argmin(d[idx])]
    after = d[j:idx[-1] + 1]
    return float(mid[j]) if after.max() > 1.5 * d[j] else None

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conv", required=True); ap.add_argument("--wiki", required=True); ap.add_argument("--code", required=True)
    ap.add_argument("--metric", default="token_min_b_32"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zc, zw, zk = (np.load(p) for p in (a.conv, a.wiki, a.code))
    m = a.metric
    fig, axes = plt.subplots(2, 4, figsize=(21, 8.5)); axes = axes.ravel()
    table = {}
    for li, L in enumerate(range(2, 10)):
        ax = axes[li]
        c, e, _, _ = hist(zc, m, li); ax.bar(e[:-1], c / c.sum(), width=np.diff(e), color="#94a3b8", align="edge", label="conv (all tokens)")
        for z_, col, lab in ((zw, "#2563eb", "wiki"), (zk, "#f97316", "code")):
            cc, ee, _, _ = hist(z_, m, li)
            ax.step(ee[:-1], cc / cc.sum(), where="post", color=col, lw=1.0, alpha=0.9, label=f"{lab} (calibration)")
        v = valley(zc, m, li)
        cuts = {
            "valley": v,
            "conv top1%": quant(zc, m, li, 0.99),
            "conv top0.1%": quant(zc, m, li, 0.999),
            "wiki b95": quant(zw, m, li, 0.05),
            "code b95": quant(zk, m, li, 0.05),
        }
        style = {"valley": ("#000000", "-", 2.2), "conv top1%": ("#ef4444", "-", 1.3), "conv top0.1%": ("#7f1d1d", "-", 1.3),
                 "wiki b95": ("#2563eb", "--", 1.4), "code b95": ("#f97316", "--", 1.4)}
        for k, x in cuts.items():
            if x is None: continue
            col, ls, lw = style[k]; ax.axvline(x, color=col, ls=ls, lw=lw, label=f"{k} {x:.4f}")
        ax.set_yscale("log"); ax.set_xlim(0.6, 1.0); ax.set_ylim(1e-7, 1)
        ax.set_title(f"L{L}"); ax.legend(fontsize=6.5, loc="upper left")
        # fraction of conv tokens passing each cut, from the same histogram
        tot = c.sum(); mids = 0.5 * (e[:-1] + e[1:])
        table[f"L{L}"] = {k: {"cut": x, "conv_pass": float(c[mids >= x].sum() / tot)} for k, x in cuts.items() if x is not None}
    fig.suptitle(f"{m}: conversation vs old-domain bands at scale 32 — cut candidates per layer", fontsize=12)
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    Path(a.out).with_suffix(".json").write_text(json.dumps(table, indent=2))
    print(f'{"L":<4}' + "".join(f"{k:>22}" for k in ("valley", "conv top1%", "conv top0.1%", "wiki b95", "code b95")))
    for L in range(2, 10):
        row = table[f"L{L}"]; print(f"L{L:<3}" + "".join(f"{(row[k]['cut'] if k in row else float('nan')):>10.4f} {(row[k]['conv_pass']*100 if k in row else float('nan')):>9.3f}% " for k in ("valley", "conv top1%", "conv top0.1%", "wiki b95", "code b95")))
if __name__ == "__main__":
    main()
