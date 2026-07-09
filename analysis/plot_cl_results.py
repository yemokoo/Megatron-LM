#!/usr/bin/env python3
"""Two-panel grouped bar chart: Average Accuracy (AA) and Forgetting Measure (FM)
across 4 models x 3 continual-learning training regimes.

Ours = purple, Dense = blue, Life-long MoE = red, Fixed-24 MoE = aqua.
AA panel: 0-anchored with the B/C upper-bound band. FM panel: 0-baseline, negative
bars = backward transfer, C = undefined (n/a).

Run:  python analysis/plot_cl_results.py
Out:  cl_results.png / cl_results.pdf  (in the repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- palette (light mode, validated CVD-safe) ----
COL = {
    "ours":     "#4a3aa7",  # purple  (hero)
    "lifelong": "#e34948",  # red
    "dense":    "#2a78d6",  # blue
    "fixed24":  "#1baf7a",  # aqua
}
INK, MUTED = "#0b0b0b", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

MODELS = [  # (key, full label, short legend label)
    ("ours",     "Ours (FFN expand + router-FT)",       "Ours"),
    ("lifelong", "Life-long MoE (FFN + attn-unfreeze)",  "Life-long MoE"),
    ("dense",    "Dense FFN",                            "Dense FFN"),
    ("fixed24",  "Fixed-24 MoE",                         "Fixed-24 MoE"),
]
METHODS = [
    ("A", "A · Sequential",        "Wiki→Code→Conv", "6.3B tok"),
    ("B", "B · Cumulative replay", "CL upper bound",           "12.6B tok"),
    ("C", "C · Joint mix",         "FT ceiling",               "6.3B tok"),
]
D = {
    "A": {"ours": (0.4954, 0.0103), "lifelong": (0.3588, 0.2221),
          "dense": (0.3532, 0.2509), "fixed24": (0.3608, 0.2596)},
    "B": {"dense": (0.5260, -0.0276), "fixed24": (0.5404, -0.0314)},
    "C": {"dense": (0.5151, None),    "fixed24": (0.5306, None)},
}
CEIL_LO, CEIL_HI = 0.5151, 0.5404

# ---- bar x-positions (grouped) ----
BAR_W, INTRA, INTER = 0.86, 1.0, 1.9
bars, groups = [], []
x = 0.0
for mkey, *_ in METHODS:
    present = [k for k, *_ in MODELS if k in D[mkey]]
    gx0 = x
    for k in present:
        bars.append({"method": mkey, "model": k, "x": x})
        x += INTRA
    x -= INTRA
    groups.append({"cx": (gx0 + x) / 2})
    x += INTER
XMAX = x - INTER + BAR_W
for g, (mk, ml, ms, mt) in zip(groups, METHODS):
    g.update(label=ml, sub=ms, tok=mt)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

fig, (axA, axF) = plt.subplots(2, 1, figsize=(8.6, 9.6))
fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.05, hspace=0.72)

LEG_HANDLES = [Patch(facecolor=COL[k], edgecolor=(INK if k == "ours" else "none"),
                     linewidth=(1.2 if k == "ours" else 0), label=short)
               for k, _, short in MODELS]


def draw(ax, metric, ylim, yticks, title, hint):
    ax.set_xlim(-0.7, XMAX + 0.2)
    ax.set_ylim(*ylim)
    rng = ylim[1] - ylim[0]

    for yv in yticks:
        zero = abs(yv) < 1e-9
        ax.axhline(yv, color=(AXIS if zero else GRID), lw=(1.4 if zero else 0.9), zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels([("0" if abs(t) < 1e-9 else f"{t:.2f}" if metric == "fm" else f"{t:.1f}")
                        for t in yticks], fontsize=10, color=MUTED)

    for b in bars:
        val = D[b["method"]][b["model"]][0 if metric == "aa" else 1]
        cx = b["x"]
        if metric == "fm" and val is None:
            ax.text(cx, rng * 0.02, "n/a", ha="center", va="bottom", fontsize=9.5, color=MUTED)
            continue
        ours = b["model"] == "ours"
        ax.bar(cx, val, width=BAR_W, color=COL[b["model"]],
               edgecolor=(INK if ours else "none"), linewidth=(1.3 if ours else 0), zorder=3)
        off = rng * 0.018
        if val >= 0:
            ax.text(cx, val + off, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK, zorder=4)
        else:
            ax.text(cx, val - off, f"−{abs(val):.3f}", ha="center", va="top",
                    fontsize=9.5, fontweight="bold", color=INK, zorder=4)
        if ours:
            ax.text(cx, val + off * 3.6, "★ Ours", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=COL["ours"], zorder=4)

    for g in groups:  # group labels + tokens below axis
        ax.text(g["cx"], -0.085, g["label"], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(g["cx"], -0.135, g["sub"], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9.5, color=MUTED)
        ax.text(g["cx"], -0.185, g["tok"], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9.5, color=MUTED)

    ax.set_xticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)

    # header: title, hint, legend — all stacked ABOVE the plot (no overlap with bars)
    ax.text(0.0, 1.28, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14, fontweight="bold", color=INK)
    ax.text(0.0, 1.205, hint, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.5, color=MUTED)
    ax.legend(handles=LEG_HANDLES, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=4, frameon=False, fontsize=9.8, handlelength=1.1,
              columnspacing=1.5, handletextpad=0.5)


draw(axA, "aa", (0, 0.60), [0, 0.1, 0.2, 0.3, 0.4, 0.5],
     "Average Accuracy (AA)", "higher is better")
draw(axF, "fm", (-0.06, 0.30), [-0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25],
     "Forgetting Measure (FM)", "lower is better")

fig.savefig("cl_results.png", dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig("cl_results.pdf", facecolor="white", bbox_inches="tight")
print("saved cl_results.png / cl_results.pdf")
