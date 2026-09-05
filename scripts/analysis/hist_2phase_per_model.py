#!/usr/bin/env python
"""Per-token hidden-drift histograms for ONE 2-phase run at a time.

analyze_2phase_drift.py overlays the methods against each other; this draws one
figure per run instead, overlaying that run's own stages, to read how far a
stage pushed the wiki representation and how much the router FT pulled back.

  d_i(layer) = ||h_i^stage - h_i^s0||_2      (raw L2 in the 1024-d space)
  D(i)       = mean over the trained layers of d_i(layer)      (token score)

Per-layer panels show d_i; the pooled panel, the combined figure and the summary
use D(i), the same token score analyze_2phase_drift.py reports.  Pooling the raw
d_i across layers instead would mix eight differently-scaled distributions (the
median runs 0.68 at layer 2 to 4.58 at layer 9), so a chunk of the spread would
be layer scale rather than token-to-token variation.

Layer 1 is frozen in every run, so its drift is identically zero; it is dropped
from the panels and its slot carries the pooled distribution over the layers
that do move.  The s0 stage is the wiki-only reference itself and is likewise
zero everywhere, so it is drawn as a marker at the origin rather than a delta
that would own the y-axis.  Every figure in one invocation shares an x range,
so runs can be compared side by side.

  python scripts/analysis/hist_2phase_per_model.py --models A,C --probe wiki
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# label -> (run name, which s0 dump it is referenced against).  A-D are the
# 2-phase runs; F (HF kd_1phase one_phase) and G (selfgen_replay_Gnew) are
# 1-phase chains that exist only at s2 and s4, and both descend from the hybrid
# e8, so they are referenced against hyb_s0 like B and D.
RUNS = {"A": ("ffn_kd0", "ffn"), "B": ("hyb_kd0", "hyb"),
        "C": ("ffn_kd1", "ffn"), "D": ("hyb_kd1", "hyb"),
        "F": ("hf_1phase", "hyb"), "G": ("selfgen_Gnew", "hyb")}
STAGE_LABEL = {"s0": "wiki only (reference)", "s1": "+code (phase-1)",
               "s2": "+code router FT", "s3": "+conv (phase-1)",
               "s4": "+conv router FT"}
TRAINED_LAYERS_FROM = 2      # layer 1 is frozen: zero drift, no panel, no summary


def binned(v, edges, hi):
    """Raw density per bin, with the clip pile-up bin dropped.

    Everything past the percentile cut lands in the final bin, so it is an
    artefact of clipping rather than part of the distribution.  The counts are
    NOT smoothed: a Gaussian blur over these bins moves the crest (it merged
    A's 0.78 and C's 0.56 into a common 0.99), so the drawn curve has to stay
    the histogram itself.
    """
    counts, _ = np.histogram(np.clip(v, 0, hi), bins=edges, density=True)
    return counts[:-1]


def marker(v, counts, edges, how):
    """(x, height) for the plumb line, or None.

    Defaults to the median.  The mode is available but is not trustworthy here:
    on these steeply-rising unimodal drifts its position -- and even the A/C
    ordering -- flips with the bin width (A/C read 1.09/1.34 at 50 bins and
    1.03/0.91 at 100), while the median is stable to three decimals.
    """
    if how == "none":
        return None
    if how == "mode":
        k = int(np.argmax(counts))
    else:
        x = float(np.median(v))
        k = min(int(np.searchsorted(edges, x, side="right")) - 1,
                len(counts) - 1)
        return x, float(counts[max(k, 0)])
    return 0.5 * (edges[k] + edges[k + 1]), float(counts[k])


def annotate_markers(ax, peaks, fontsize, headroom=1.26):
    """Draw each plumb line and park its value on its own tier above the curves.

    Anchoring the text to the curve height made labels collide with each other,
    with the panel title, and with the crest they belong to.  Instead the panel
    gains headroom, every marker gets a tier of its own counting down from the
    top, and the plumb line is extended to that tier so line and number stay
    visibly attached.  Labels near the right edge flip to the left of the line
    so they cannot run off the panel.
    """
    if not peaks:
        return
    lo, top = ax.get_ylim()
    top *= headroom
    ax.set_ylim(lo, top)
    x_lo, x_hi = ax.get_xlim()
    for rank, (xpk, _ypk, color) in enumerate(sorted(peaks)):
        tier = top * (0.96 - 0.10 * rank)
        ax.vlines(xpk, 0, tier, color=color, linewidth=1.2, alpha=0.85)
        ax.plot([xpk], [0], marker="v", markersize=7, color=color, clip_on=False)
        right_edge = (xpk - x_lo) / (x_hi - x_lo) > 0.75
        ax.annotate(f"{xpk:.2f}", xy=(xpk, tier),
                    xytext=(-5 if right_edge else 5, 0),
                    textcoords="offset points",
                    ha="right" if right_edge else "left", va="center",
                    color=color, fontsize=fontsize, fontweight="bold")


def load(root, name):
    p = root / "hidden" / f"{name}.npz"
    if not p.exists():
        raise SystemExit(f"missing dump: {p}")
    return np.load(p)


def drift(ref, ref_h, ck, ref_name, ck_name):
    """Per-layer, per-token L2 between two dumps, after asserting alignment."""
    for key in ("token_ids", "positions", "sample_indices"):
        if not np.array_equal(ref[key], ck[key]):
            raise SystemExit(
                f"probe misalignment on {key}: {ref_name} vs {ck_name}. "
                "The two dumps did not see the same tokens; re-dump with the "
                "same seed/mb/gbs/probe-iters before comparing them.")
    return np.linalg.norm(ck["hidden_layers"].astype(np.float32) - ref_h, axis=2)


# A combined figure carries model x stage curves in one axes; every curve is a
# solid line, so colour alone separates them.
COMBO_PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd",
                 "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]
ZERO_COLOR = "#999999"          # s0 against itself, and any frozen layer


# When the runs are the only axis being compared, each keeps its own colour in
# every figure, so D reads the same in "C vs D" as in "D vs F".
MODEL_COLOR = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c",
               "D": "#9467bd", "F": "#d62728", "G": "#8c564b"}


def curve_colors(models, stages):
    """One colour per (model, stage), shared by the per-model and combined figures.

    With several stages drawn, the colour is keyed on the pair, so a run keeps
    its colours between the two kinds of figure: without this, A and C both drew
    their +code curve in blue per-model while the combined figure gave C red.
    With a single stage the run is the only thing that varies, so the fixed
    per-run colour is used instead and stays put across figures.
    """
    drawn = [s for s in stages if s != "s0"]
    by_model = len(drawn) <= 1
    colors = {}
    i = 0
    for model in models:
        for stage in stages:
            if stage == "s0":
                colors[(model, stage)] = ZERO_COLOR
                continue
            if by_model:
                colors[(model, stage)] = MODEL_COLOR.get(
                    model, COMBO_PALETTE[i % len(COMBO_PALETTE)])
            else:
                colors[(model, stage)] = COMBO_PALETTE[i % len(COMBO_PALETTE)]
            i += 1
    return colors


def combined_figure(args, out, models, stages, token_score, layers_kept, edges, hi):
    """One axes: the token score D(t) for every model x stage."""
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    peaks = []
    colors = curve_colors(models, stages)
    for model in models:
        run_dir, _ = RUNS[model]
        for stage in stages:
            v = token_score[model][stage]
            label = f"{model} ({run_dir})  {STAGE_LABEL[stage]}"
            color = colors[(model, stage)]
            if not np.any(v > 0):
                ax.axvline(0.0, color=color, linewidth=2.4,
                           label=f"{label} (=0)")
                continue
            counts = binned(v, edges, hi)
            ax.stairs(counts, edges[:-1], linewidth=2.2, color=color,
                      label=label)
            mk = marker(v, counts, edges, args.mark)
            if mk:
                peaks.append((mk[0], mk[1], color))

    ax.set_xlim(0, hi)
    annotate_markers(ax, peaks, fontsize=15)
    ax.set_xlabel("token score D(t) = mean over layers "
                  f"{int(layers_kept[0])}-{int(layers_kept[-1])} of the "
                  "per-token L2 drift from wiki-only")
    ax.set_ylabel("density")
    ax.legend(fontsize=11, frameon=False)
    n_tok = token_score[models[0]][stages[-1]].shape[0]
    ax.set_title(
        f"{' vs '.join(models)}: token score D(t) vs the wiki-only model "
        f"({args.probe} probe, {n_tok:,} tokens)")
    fig.tight_layout()
    # stages belong in the name: one invocation's s1/s2 figure must not clobber
    # the next one's s4 figure
    path = (out / f"hist_combined_{'_'.join(models)}_"
                  f"{''.join(stages)}_{args.probe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def combined_layer_figure(args, out, models, stages, per_model, token_score,
                          layers_kept, edges_layer, hi_layer, edges, hi):
    """The combined comparison broken out per layer, plus the D(t) panel."""
    colors = curve_colors(models, stages)
    n_panels = len(layers_kept) + 1
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows),
                             squeeze=False)
    panels = [(f"layer {int(l)}", i) for i, l in enumerate(layers_kept)]
    panels.append((f"token score D(t): mean over layers "
                   f"{int(layers_kept[0])}-{int(layers_kept[-1])}", None))
    for pi, (title, li) in enumerate(panels):
        ax = axes[pi // ncols][pi % ncols]
        peaks = []
        p_edges, p_hi = ((edges_layer, hi_layer) if li is not None
                         else (edges, hi))
        for model in models:
            run_dir, _ = RUNS[model]
            for stage in stages:
                v = (per_model[model][stage][li] if li is not None
                     else token_score[model][stage])
                color = colors[(model, stage)]
                label = f"{model} ({run_dir}) {STAGE_LABEL[stage]}"
                if not np.any(v > 0):
                    ax.axvline(0.0, color=color, linewidth=2.4, alpha=0.9,
                               label=f"{label} (=0)")
                    continue
                counts = binned(v, p_edges, p_hi)
                ax.stairs(counts, p_edges[:-1], linewidth=1.8, color=color,
                          label=label)
                mk = marker(v, counts, p_edges, args.mark)
                if mk:
                    peaks.append((mk[0], mk[1], color))
        ax.set_xlim(0, p_hi)
        annotate_markers(ax, peaks, fontsize=13)
        ax.set_title(title, fontweight="bold" if li is None else "normal",
                     pad=10)
        ax.set_xlabel("token score D(t)" if li is None
                      else "per-token L2 drift from wiki-only")
        if pi == 0:
            ax.legend(fontsize=9, frameon=False)
    for j in range(n_panels, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    n_tok = token_score[models[0]][stages[-1]].shape[0]
    fig.suptitle(
        f"{' vs '.join(models)}: drift from the wiki-only model, per layer "
        f"({args.probe} probe, {n_tok:,} tokens)")
    fig.tight_layout()
    path = (out / f"hist_combined_{'_'.join(models)}_"
                  f"{''.join(stages)}_{args.probe}_bylayer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/data2/seonghyeonnoh/LLM-continual-learning-runs"
                                 "/hidden_drift_2phase_20260829"))
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <root>/analysis")
    ap.add_argument("--models", default="A,C")
    ap.add_argument("--stages", default="s0,s1,s2")
    ap.add_argument("--probe", default="wiki", choices=["wiki", "code"])
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--clip-percentile", type=float, default=99.5,
                    help="shared x range: this percentile over every drawn "
                         "stage of every model in this invocation")
    ap.add_argument("--mark", default="median",
                    choices=["median", "mode", "none"],
                    help="what the plumb line marks (default: median; the mode "
                         "moves with the bin width on these distributions)")
    ap.add_argument("--combined", action="store_true",
                    help="one figure, pooled over the trained layers, with "
                         "every model x stage overlaid instead of per-model "
                         "per-layer grids")
    ap.add_argument("--by-layer", action="store_true",
                    help="with --combined: break the comparison out into one "
                         "panel per trained layer plus the D(t) panel")
    args = ap.parse_args()

    out = args.out or (args.root / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for m in models:
        if m not in RUNS:
            raise SystemExit(f"unknown model {m}; expected one of {list(RUNS)}")

    # ---- pass 1: drift arrays for every model, so the x range can be shared ----
    per_model = {}            # model -> {stage: [trained_layers, tokens]}
    layers_kept = None
    ref_cache = {}
    for model in models:
        _, base = RUNS[model]
        if base not in ref_cache:
            ref = load(args.root, f"{base}_s0_{args.probe}")
            ref_cache[base] = (ref, ref["hidden_layers"].astype(np.float32))
        ref, ref_h = ref_cache[base]
        keep = ref["layer_numbers"] >= TRAINED_LAYERS_FROM
        layers_kept = ref["layer_numbers"][keep]

        data = {}
        for stage in stages:
            if stage == "s0":
                data[stage] = np.zeros((int(keep.sum()), ref_h.shape[1]),
                                       dtype=np.float32)
                continue
            ck = load(args.root, f"{model}_{stage}_{args.probe}")
            data[stage] = drift(ref, ref_h, ck, f"{base}_s0",
                                f"{model}_{stage}")[keep]
        per_model[model] = data

    # D(i): one score per token, the mean of its per-layer drifts
    token_score = {m: {s: v.mean(axis=0) for s, v in per_model[m].items()}
                   for m in models}

    # two x ranges: per-layer panels live on the raw d_i scale, everything that
    # aggregates over layers lives on the narrower D(i) scale.  Each is shared
    # across the models in this invocation so the figures stay comparable.
    drawn = [s for s in stages if s != "s0"]

    def span(arrays):
        return float(np.percentile(np.concatenate(arrays),
                                   args.clip_percentile)) if drawn else 1.0

    hi_layer = span([per_model[m][s].ravel() for m in models for s in drawn])
    hi = span([token_score[m][s] for m in models for s in drawn])
    edges_layer = np.linspace(0.0, hi_layer, args.bins + 1)
    edges = np.linspace(0.0, hi, args.bins + 1)

    rows = []
    for model in models:
        run_dir, _ = RUNS[model]
        for stage in stages:
            v = token_score[model][stage]
            rows.append({"model": model, "run": run_dir, "stage": stage,
                         "label": STAGE_LABEL[stage], "probe": args.probe,
                         "median": float(np.median(v)),
                         "p90": float(np.percentile(v, 90)),
                         "mean": float(np.mean(v)),
                         "n_tokens": int(v.shape[0])})

    if args.combined:
        if args.by_layer:
            combined_layer_figure(args, out, models, stages, per_model,
                                  token_score, layers_kept, edges_layer,
                                  hi_layer, edges, hi)
        else:
            combined_figure(args, out, models, stages, token_score, layers_kept,
                            edges, hi)
        report(rows, layers_kept, args, hi, out)
        return

    # ---- pass 2: one figure per model, panels = trained layers + pooled ----
    colors = curve_colors(models, stages)
    n_panels = len(layers_kept) + 1
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    for model in models:
        run_dir, _ = RUNS[model]
        data = per_model[model]
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows),
                                 squeeze=False)
        panels = [(f"layer {int(l)}", i) for i, l in enumerate(layers_kept)]
        panels.append((f"token score D(t): mean over layers "
                       f"{int(layers_kept[0])}-{int(layers_kept[-1])}", None))
        for pi, (title, li) in enumerate(panels):
            ax = axes[pi // ncols][pi % ncols]
            peaks = []
            # the D(t) panel lives on its own, narrower scale than the raw d_i
            # panels, so it carries its own bins and x limit
            p_edges, p_hi = ((edges_layer, hi_layer) if li is not None
                             else (edges, hi))
            for stage in stages:
                v = (data[stage][li] if li is not None
                     else token_score[model][stage])
                color = colors[(model, stage)]
                if not np.any(v > 0):
                    # frozen or self-referenced: mark the spike at 0 instead of
                    # drawing a delta that would rescale the whole panel
                    ax.axvline(0.0, color=color, linewidth=2.4,
                               alpha=0.9, label=f"{STAGE_LABEL[stage]} (=0)")
                    continue
                counts = binned(v, p_edges, p_hi)
                ax.stairs(counts, p_edges[:-1], linewidth=1.8,
                          color=color, label=STAGE_LABEL[stage])
                # a plumb line, so a panel can be read off the x axis directly
                mk = marker(v, counts, p_edges, args.mark)
                if mk:
                    peaks.append((mk[0], mk[1], color))
            ax.set_xlim(0, p_hi)
            annotate_markers(ax, peaks, fontsize=13)
            ax.set_title(title, fontweight="bold" if li is None else "normal",
                         pad=10)
            ax.set_xlabel("token score D(t)" if li is None
                          else "per-token L2 drift from wiki-only")
            if pi == 0:
                ax.legend(fontsize=8, frameon=False)
        for j in range(n_panels, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(
            f"{model} = {run_dir}: per-token drift from the wiki-only model "
            f"({args.probe} probe, {data[stages[-1]].shape[1]:,} tokens, "
            f"x ranges shared across {'/'.join(models)})")
        fig.tight_layout()
        path = out / f"hist_{model}_{run_dir}_{args.probe}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")

    report(rows, layers_kept, args, hi, out)


def report(rows, layers_kept, args, hi, out):
    width = max(len(r["label"]) for r in rows)
    print(f"\nlayers {int(layers_kept[0])}-{int(layers_kept[-1])}, "
          f"{args.probe} probe, x range 0-{hi:.2f}\n")
    print(f"{'model':6s} {'run':9s} {'stage':6s} {'what':{width}s} "
          f"{'median':>9s} {'p90':>9s} {'mean':>9s}")
    for r in rows:
        print(f"{r['model']:6s} {r['run']:9s} {r['stage']:6s} {r['label']:{width}s} "
              f"{r['median']:9.4f} {r['p90']:9.4f} {r['mean']:9.4f}")
    stage_tag = "".join(s.strip() for s in args.stages.split(",") if s.strip())
    summary = out / f"per_model_stage_summary_{stage_tag}_{args.probe}.json"
    summary.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
