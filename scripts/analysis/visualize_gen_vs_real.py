#!/usr/bin/env python3
"""Generated (BoS+anchor) vs real wiki/code: token-level and routing-level comparison figures."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "Megatron-LM"))
from megatron.core.datasets.indexed_dataset import IndexedDataset  # noqa: E402
V = 50277
def load_seqs(prefix, max_tokens, seed=0, min_len=128, trunc=512):
    ds = IndexedDataset(prefix); rng = np.random.default_rng(seed); out, tot = [], 0
    for i in rng.permutation(len(ds)):
        s = np.asarray(ds.get(int(i)), dtype=np.int64); s = s[s < V]
        if len(s) < min_len: continue
        s = s[:trunc]; out.append(s); tot += len(s)
        if tot >= max_tokens: break
    return out
def unigram(seqs):
    c = np.zeros(V); [np.add.at(c, s, 1) for s in seqs]; return c / c.sum()
def js(p, q, eps=1e-12):
    p = p + eps; q = q + eps; p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))
def rep4(s):
    seen = set(); r = 0
    for i in range(len(s) - 3):
        g = tuple(s[i:i + 4]); r += g in seen; seen.add(g)
    return r / max(len(s) - 3, 1)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True); ap.add_argument("--code", required=True)
    ap.add_argument("--gen-wiki", required=True); ap.add_argument("--gen-code", required=True)
    ap.add_argument("--router-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1_000_000)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sets = {"real wiki": load_seqs(a.wiki, a.max_tokens, 0), "gen wiki": load_seqs(a.gen_wiki, a.max_tokens, 0),
            "real code": load_seqs(a.code, a.max_tokens, 0), "gen code": load_seqs(a.gen_code, a.max_tokens, 0)}
    wiki2, code2 = load_seqs(a.wiki, a.max_tokens, 1), load_seqs(a.code, a.max_tokens, 1)
    P = {k: unigram(v) for k, v in sets.items()}
    with np.errstate(divide="ignore"):
        lr = np.log(P["real code"] + 1e-12) - np.log(P["real wiki"] + 1e-12)
    markers = np.zeros(V, bool); markers[(P["real code"] >= 2e-5) & (lr >= 3)] = True
    lw, lc = np.log(P["real wiki"] + 1e-9), np.log(P["real code"] + 1e-9)
    stats = {k: {"marker": np.array([markers[s].mean() for s in v]),
                 "llr": np.array([float((lw[s] - lc[s]).sum()) / len(s) for s in v]),   # per-token LLR
                 "uniq": np.array([len(np.unique(s)) / len(s) for s in v]),
                 "rep4": np.array([rep4(s.tolist()) for s in v])} for k, v in sets.items()}
    jsm = {"gen wiki→real wiki": js(P["gen wiki"], P["real wiki"]), "gen code→real code": js(P["gen code"], P["real code"]),
           "gen wiki→real code": js(P["gen wiki"], P["real code"]), "gen code→real wiki": js(P["gen code"], P["real wiki"]),
           "real wiki↔real code": js(P["real wiki"], P["real code"]),
           "real wiki↔real wiki′ (noise floor)": js(P["real wiki"], unigram(wiki2)), "real code↔real code′ (noise floor)": js(P["real code"], unigram(code2))}
    col = {"real wiki": "#1f77b4", "gen wiki": "#17becf", "real code": "#d62728", "gen code": "#ff7f0e"}
    # ---------------- figure 1: token level ----------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for k in sets:
        ax[0, 0].hist(stats[k]["marker"], bins=np.linspace(0, 0.8, 60), histtype="step", lw=2, density=True, color=col[k], label=k)
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("code-marker token rate per sequence"); ax[0, 0].set_ylabel("density"); ax[0, 0].set_title("(a) code-marker rate"); ax[0, 0].legend()
    for k in sets:
        ax[0, 1].hist(stats[k]["llr"], bins=np.linspace(-5, 5, 80), histtype="step", lw=2, density=True, color=col[k], label=k)
    ax[0, 1].set_xlabel("per-token log P_wiki(x)/P_code(x)   (>0 wiki-like, <0 code-like)"); ax[0, 1].set_title("(b) naive-Bayes wiki-vs-code score per sequence"); ax[0, 1].legend()
    names = list(jsm); vals = [jsm[n] for n in names]
    ax[0, 2].barh(range(len(names)), vals, color=["#17becf", "#ff7f0e", "#999", "#999", "#444", "#bbb", "#bbb"]); ax[0, 2].set_yticks(range(len(names))); ax[0, 2].set_yticklabels(names, fontsize=9); ax[0, 2].invert_yaxis()
    for i, v in enumerate(vals): ax[0, 2].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax[0, 2].set_xlim(0, 0.8); ax[0, 2].set_xlabel("unigram Jensen-Shannon divergence (bits)"); ax[0, 2].set_title("(c) token-distribution distance")
    for pair, axx, ttl in ((("real wiki", "gen wiki"), ax[1, 0], "(d) unigram freq: gen wiki vs real wiki"), (("real code", "gen code"), ax[1, 1], "(e) unigram freq: gen code vs real code")):
        x, y = P[pair[0]], P[pair[1]]; m = (x > 0) | (y > 0)
        axx.scatter(x[m] + 1e-8, y[m] + 1e-8, s=3, alpha=0.3, color=col[pair[1]]); axx.plot([1e-8, 0.2], [1e-8, 0.2], "k--", lw=1)
        axx.set_xscale("log"); axx.set_yscale("log"); axx.set_xlabel(f"P({pair[0]})"); axx.set_ylabel(f"P({pair[1]})"); axx.set_title(ttl)
        r = np.corrcoef(np.log(x[m] + 1e-8), np.log(y[m] + 1e-8))[0, 1]; axx.text(0.05, 0.9, f"log-freq corr = {r:.3f}\nJS = {jsm[pair[1] + '→' + pair[0]]:.3f}", transform=axx.transAxes)
    pos = np.arange(4); w = 0.35
    for j, (key, lab) in enumerate((("uniq", "unique-token ratio"), ("rep4", "repeated 4-gram rate"))):
        vals_ = [stats[k][key] for k in sets]
        bp = ax[1, 2].boxplot(vals_, positions=pos + (j - 0.5) * w, widths=0.3, showfliers=False, patch_artist=True)
        for patch, k in zip(bp["boxes"], sets): patch.set_facecolor(col[k]); patch.set_alpha(0.45 if j == 0 else 0.9)
    ax[1, 2].set_xticks(pos); ax[1, 2].set_xticklabels(list(sets)); ax[1, 2].set_title("(f) per-sequence diversity: unique ratio (light) / 4-gram repeat (dark)")
    fig.suptitle("Self-generated replay (BoS + 1 anchor token, code-stage model) vs real wiki / code test  — token level", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "gen_vs_real_tokens.png", dpi=130); plt.close(fig)
    # ---------------- figure 2: routing level ----------------
    R = {k: json.loads((Path(a.router_dir) / f).read_text()) for k, f in (("real wiki", "wiki_test.json"), ("gen wiki", "replay1pct_wiki.json"), ("real code", "code_test.json"), ("gen code", "replay1pct_code.json"))}
    layers = sorted(R["real wiki"]["layers"], key=int); S = 8
    def combo_js(x, y):
        keys = list(set(x) | set(y)); return js(np.array([x.get(k, 0) for k in keys], float), np.array([y.get(k, 0) for k in keys], float))
    def cov(ref, g): return sum(v for k, v in ref.items() if k in g) / max(sum(ref.values()), 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k in R:
        ax[0].plot([int(l) for l in layers], [sum(R[k]["layers"][l]["usage"][:S]) for l in layers], "o-", color=col[k], label=k, lw=2)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("share of top-4 slots on wiki experts 0-7"); ax[0].set_ylim(0, 1.05); ax[0].set_title("(a) old-expert share per layer"); ax[0].legend()
    ax[1].plot([int(l) for l in layers], [combo_js(R["gen wiki"]["layers"][l]["combos"], R["real wiki"]["layers"][l]["combos"]) for l in layers], "o-", color=col["gen wiki"], lw=2, label="gen wiki → real wiki")
    ax[1].plot([int(l) for l in layers], [combo_js(R["gen code"]["layers"][l]["combos"], R["real code"]["layers"][l]["combos"]) for l in layers], "o-", color=col["gen code"], lw=2, label="gen code → real code")
    ax[1].plot([int(l) for l in layers], [combo_js(R["real wiki"]["layers"][l]["combos"], R["real code"]["layers"][l]["combos"]) for l in layers], "s--", color="#444", lw=1.5, label="real wiki ↔ real code (reference)")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("JS divergence of top-4 combination distributions (bits)"); ax[1].set_ylim(0, 1); ax[1].set_title("(b) routing-combination distance"); ax[1].legend()
    U = np.array([[R[k]["layers"][l]["usage"] for l in layers] for k in R])  # [4, L, 16]
    im = ax[2].imshow(np.concatenate([U[i] for i in range(4)], axis=0), aspect="auto", cmap="viridis", vmin=0, vmax=U.max())
    ax[2].set_yticks([len(layers) * i + len(layers) / 2 - 0.5 for i in range(4)]); ax[2].set_yticklabels(list(R)); ax[2].set_xlabel("expert (0-7 wiki, 8-15 code)"); ax[2].set_title("(c) per-layer expert usage (rows = layers 2-9 per set)")
    for i in range(1, 4): ax[2].axhline(len(layers) * i - 0.5, color="w", lw=1.5)
    ax[2].axvline(7.5, color="w", lw=1, ls="--"); plt.colorbar(im, ax=ax[2], fraction=0.04, label="fraction of top-4 slots")
    covs = {k: float(np.mean([cov(R[r]["layers"][l]["combos"], R[k]["layers"][l]["combos"]) for l in layers])) for k, r in (("gen wiki", "real wiki"), ("gen code", "real code"))}
    fig.suptitle(f"Routing through the code-stage model — generated vs real   (combo coverage of real by gen: wiki {covs['gen wiki']:.3f}, code {covs['gen code']:.3f})", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "gen_vs_real_routing.png", dpi=130); plt.close(fig)
    summary = {"js": jsm, "combo_coverage": covs, "per_seq_medians": {k: {m: float(np.median(stats[k][m])) for m in stats[k]} for k in stats}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))
if __name__ == "__main__":
    main()
