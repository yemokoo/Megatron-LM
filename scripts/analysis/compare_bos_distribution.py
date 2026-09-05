#!/usr/bin/env python3
"""Token-distribution comparison: BoS-generated corpus vs wiki test vs code test.

Reads Megatron indexed datasets and compares unigram / bigram distributions with
Jensen-Shannon divergence (base 2, so 0 = identical, 1 = disjoint), plus a few
surface statistics.  The wiki-vs-code divergence is printed as the reference
scale: a generated set that sits at ~0 to wiki and ~wiki-vs-code to code is
"wiki-like", one halfway between is a mixture.

Also reports, per generated sequence, which reference it is closer to under a
unigram naive-Bayes score, which gives a rough wiki/code mixture fraction.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Megatron-LM"))
from megatron.core.datasets.indexed_dataset import IndexedDataset  # noqa: E402

VOCAB = 50277


def load_tokens(prefix: str, max_tokens: int, seed: int = 0):
    ds = IndexedDataset(prefix)
    n = len(ds)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    out, idxs, total = [], [], 0
    for i in order:
        s = np.asarray(ds.get(int(i)), dtype=np.int64)
        out.append(s); idxs.append(int(i))
        total += len(s)
        if total >= max_tokens:
            break
    load_tokens.last_indices = idxs
    return out


def unigram(seqs):
    c = np.zeros(VOCAB, dtype=np.float64)
    for s in seqs:
        np.add.at(c, s[s < VOCAB], 1)
    return c / c.sum()


def bigram_counter(seqs):
    c = Counter()
    for s in seqs:
        a = s[:-1] * VOCAB + s[1:]
        vals, cnts = np.unique(a, return_counts=True)
        c.update(dict(zip(vals.tolist(), cnts.tolist())))
    return c


def js_div(p, q, eps=1e-12):
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_div_counter(a: Counter, b: Counter):
    keys = list(set(a) | set(b))
    pa = np.array([a.get(k, 0) for k in keys], dtype=np.float64)
    pb = np.array([b.get(k, 0) for k in keys], dtype=np.float64)
    return js_div(pa, pb)


def code_marker_set(p_wiki, p_code, min_code_prob=2e-5, log_ratio=3.0):
    """Tokens far more frequent in code than in wiki (data-driven 'code markers')."""
    with np.errstate(divide="ignore"):
        lr = np.log(p_code + 1e-12) - np.log(p_wiki + 1e-12)
    return np.where((p_code >= min_code_prob) & (lr >= log_ratio))[0]


def marker_rate(seqs, markers):
    m = np.zeros(VOCAB, dtype=bool); m[markers] = True
    return np.array([float(m[s[s < VOCAB]].mean()) if len(s) else 0.0 for s in seqs])


def nb_assign(seqs, p_wiki, p_code, eps=1e-9):
    """Per-sequence log-likelihood ratio under the two reference unigram models."""
    lw = np.log(p_wiki + eps); lc = np.log(p_code + eps)
    llr = np.array([float((lw[s[s < VOCAB]] - lc[s[s < VOCAB]]).sum()) for s in seqs])
    return llr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True, help="generated dataset prefixes (label=prefix)")
    ap.add_argument("--wiki", default="/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/wiki/test/test_text_document")
    ap.add_argument("--code", default="/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/test/test_text_document")
    ap.add_argument("--max-tokens", type=int, default=1_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wiki = load_tokens(args.wiki, args.max_tokens, 0)
    code = load_tokens(args.code, args.max_tokens, 0)
    # second, disjoint-seed halves of the references give the sampling-noise floor
    wiki2 = load_tokens(args.wiki, args.max_tokens, 1)
    code2 = load_tokens(args.code, args.max_tokens, 1)
    pw, pc = unigram(wiki), unigram(code)
    bw, bc = bigram_counter(wiki), bigram_counter(code)
    ref = {
        "unigram_js/wiki_vs_code": js_div(pw, pc),
        "unigram_js/wiki_vs_wiki2 (noise floor)": js_div(pw, unigram(wiki2)),
        "unigram_js/code_vs_code2 (noise floor)": js_div(pc, unigram(code2)),
        "bigram_js/wiki_vs_code": js_div_counter(bw, bc),
        "bigram_js/wiki_vs_wiki2 (noise floor)": js_div_counter(bw, bigram_counter(wiki2)),
    }
    markers = code_marker_set(pw, pc)
    mr_w, mr_c = marker_rate(wiki, markers), marker_rate(code, markers)
    wiki_p95 = float(np.percentile(mr_w, 95))
    ref.update({
        "code_marker_tokens": int(markers.size),
        "marker_rate/wiki_mean": float(mr_w.mean()), "marker_rate/wiki_p95": wiki_p95,
        "marker_rate/code_mean": float(mr_c.mean()),
        "marker_rate/code_frac_above_wiki_p95": float((mr_c > wiki_p95).mean()),
    })
    print("reference")
    for k, v in ref.items():
        print(f"  {k:<42} {v:.4f}" if isinstance(v, float) else f"  {k:<42} {v}")

    results = {"reference": ref, "generated": {}}
    for item in args.gen:
        label, prefix = item.split("=", 1) if "=" in item else (Path(item).parent.name, item)
        g = load_tokens(prefix, args.max_tokens, 0)
        pg, bg = unigram(g), bigram_counter(g)
        llr = nb_assign(g, pw, pc)
        gidx = list(load_tokens.last_indices)
        mr_g = marker_rate(g, markers)
        flat = np.concatenate(g)
        r = {
            "marker_rate/mean": float(mr_g.mean()),
            "marker_rate/frac_seqs_above_wiki_p95 (code-like)": float((mr_g > wiki_p95).mean()),
            "llr_deciles(wiki>0>code)": [round(float(x), 1) for x in np.percentile(llr, [10, 30, 50, 70, 90])],
            "tokens": int(flat.size), "seqs": len(g),
            "unigram_js/to_wiki": js_div(pg, pw), "unigram_js/to_code": js_div(pg, pc),
            "bigram_js/to_wiki": js_div_counter(bg, bw), "bigram_js/to_code": js_div_counter(bg, bc),
            "nb_frac_wiki_like": float((llr > 0).mean()),
            "nb_frac_code_like": float((llr < 0).mean()),
            "eod_rate": float((flat == 0).mean()),
            "unique_token_ratio": float(np.unique(flat).size / VOCAB),
        }
        results["generated"][label] = r
        print(f"\n{label}  (tokens={r['tokens']:,}, seqs={r['seqs']})")
        for k in ("unigram_js/to_wiki", "unigram_js/to_code", "bigram_js/to_wiki", "bigram_js/to_code",
                  "nb_frac_wiki_like", "nb_frac_code_like", "marker_rate/mean",
                  "marker_rate/frac_seqs_above_wiki_p95 (code-like)", "eod_rate", "unique_token_ratio"):
            print(f"  {k:<50} {r[k]:.4f}")
        print(f"  {'llr_deciles(wiki>0>code) p10/30/50/70/90':<50} {r['llr_deciles(wiki>0>code)']}")
        # eyeball the extremes if the sampler left text.jsonl next to the dataset
        tj = Path(prefix).parent / "text.jsonl"
        if tj.is_file():
            texts = {}
            for line in tj.open():
                d = json.loads(line); texts[d["i"]] = d["text"]
            order = np.argsort(llr)
            ex = Path(args.out or (Path(prefix).parent / "compare.json")).parent / f"examples_{label}.txt"
            with ex.open("w") as f:
                for title, sel in (("MOST CODE-LIKE (lowest llr)", order[:5]), ("MOST WIKI-LIKE (highest llr)", order[-5:][::-1])):
                    f.write(f"===== {title} =====\n")
                    for j in sel:
                        f.write(f"--- seq {gidx[j]}  llr={llr[j]:.1f}  marker_rate={mr_g[j]:.3f}\n{texts.get(gidx[j], '')[:1200]}\n\n")
            print(f"  examples -> {ex}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\n[OUT] {args.out}")


if __name__ == "__main__":
    main()
