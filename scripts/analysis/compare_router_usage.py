#!/usr/bin/env python3
"""Compare routing histograms (router_usage_hist.py JSONs): generated vs wiki vs code.

Per layer:
  usage JS       Jensen-Shannon (bits) between per-expert top-k usage vectors
  combo JS       JS between top-k combination distributions
  wiki coverage  mass of wiki's combo distribution whose combos also occur in the
                 generated set (1.0 = every routing pattern wiki uses was exercised)
  old-expert use fraction of top-k slots on experts < SOURCE (the wiki experts)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def js(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps; q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def combo_js(a: dict, b: dict):
    keys = list(set(a) | set(b))
    return js([a.get(k, 0) for k in keys], [b.get(k, 0) for k in keys])


def coverage(ref: dict, gen: dict):
    tot = sum(ref.values())
    return sum(v for k, v in ref.items() if k in gen) / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True); ap.add_argument("--code", required=True)
    ap.add_argument("--gen", nargs="+", required=True, help="label=path.json")
    ap.add_argument("--source-experts", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    W = json.loads(Path(a.wiki).read_text()); C = json.loads(Path(a.code).read_text())
    layers = sorted(W["layers"], key=int)
    S = a.source_experts
    res = {"reference": {}, "generated": {}}

    def old_frac(d, l): return float(np.sum(d["layers"][l]["usage"][:S]))
    print(f"{'layer':<6}{'JS(wiki,code)':>14}{'cov(code→wiki)':>16}{'old% wiki':>11}{'old% code':>11}")
    for l in layers:
        r = {"usage_js": js(W["layers"][l]["usage"], C["layers"][l]["usage"]),
             "combo_js": combo_js(W["layers"][l]["combos"], C["layers"][l]["combos"]),
             "wiki_cov_by_code": coverage(W["layers"][l]["combos"], C["layers"][l]["combos"]),
             "old_frac_wiki": old_frac(W, l), "old_frac_code": old_frac(C, l)}
        res["reference"][l] = r
        print(f"{l:<6}{r['combo_js']:>14.3f}{r['wiki_cov_by_code']:>16.3f}{r['old_frac_wiki']:>11.3f}{r['old_frac_code']:>11.3f}")

    for item in a.gen:
        label, path = item.split("=", 1)
        G = json.loads(Path(path).read_text()); res["generated"][label] = {}
        print(f"\n{label}  (tokens={G['tokens']:,})")
        print(f"{'layer':<6}{'usageJS→wiki':>13}{'usageJS→code':>13}{'comboJS→wiki':>13}{'comboJS→code':>13}{'wiki cov':>10}{'code cov':>10}{'old%':>7}{'#combos':>9}")
        for l in layers:
            g = G["layers"][l]
            r = {"usage_js_wiki": js(g["usage"], W["layers"][l]["usage"]),
                 "usage_js_code": js(g["usage"], C["layers"][l]["usage"]),
                 "combo_js_wiki": combo_js(g["combos"], W["layers"][l]["combos"]),
                 "combo_js_code": combo_js(g["combos"], C["layers"][l]["combos"]),
                 "wiki_coverage": coverage(W["layers"][l]["combos"], g["combos"]),
                 "code_coverage": coverage(C["layers"][l]["combos"], g["combos"]),
                 "old_frac": old_frac(G, l), "distinct_combos": g["distinct_combos"]}
            res["generated"][label][l] = r
            print(f"{l:<6}{r['usage_js_wiki']:>13.3f}{r['usage_js_code']:>13.3f}{r['combo_js_wiki']:>13.3f}{r['combo_js_code']:>13.3f}"
                  f"{r['wiki_coverage']:>10.3f}{r['code_coverage']:>10.3f}{r['old_frac']:>7.3f}{r['distinct_combos']:>9}")
        m = lambda k: float(np.mean([res["generated"][label][l][k] for l in layers]))
        print(f"  layer-avg: comboJS→wiki {m('combo_js_wiki'):.3f}  →code {m('combo_js_code'):.3f}   wiki coverage {m('wiki_coverage'):.3f}   code coverage {m('code_coverage'):.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1)); print(f"[OUT] {a.out}")


if __name__ == "__main__":
    main()
