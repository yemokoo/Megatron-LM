"""Count tokens whose token-min B32 is in the top X% at EVERY layer.

Cuts come from the merged full-census histogram (per layer); membership is
evaluated exactly on the stored per-chunk B by taking, for each token, the min
over covering chunks.  Reports counts for all-8-layers and, for reference,
the 7-of-8 relaxation.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
LAYERS = tuple(range(2, 10))

def cut_top(z, m, li, pct):
    c = z[f"hist__{m}__counts"][li].astype(float); e = z[f"hist__{m}__edges"]
    uf = float(z[f"hist__{m}__underflow"][li]); of = float(z[f"hist__{m}__overflow"][li])
    tot = c.sum() + uf + of; target = (1 - pct / 100.0) * tot
    cum = uf + np.cumsum(c); i = min(int(np.searchsorted(cum, target)), c.size - 1)
    below = uf + (c[:i].sum() if i else 0.0); w = c[i]
    f = 0.0 if w <= 0 else min(max((target - below) / w, 0.0), 1.0)
    return float(e[i] + f * (e[i + 1] - e[i]))

def token_min(raw_b, layout, scale, seq=512):
    slots = [i for i, (s, _) in enumerate(layout) if int(s) == scale]
    out = np.full((raw_b.shape[0], seq, len(LAYERS)), np.inf, np.float32)
    for s in slots:
        a = int(layout[s][1]); b = min(a + scale, seq)
        np.minimum(out[:, a:b, :], raw_b[:, s, :][:, None, :], out=out[:, a:b, :])
    out[~np.isfinite(out)] = np.nan
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census-root", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--pct", type=float, default=4.0); ap.add_argument("--scale", type=int, default=32)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    z = np.load(f"{a.census_root}/histograms.npz")
    cuts = np.array([cut_top(z, f"token_min_b_{a.scale}", li, a.pct) for li in range(8)], np.float32)
    rows = np.load(Path(a.manifest).parent / "windows.npy")
    elig = rows["eligible_token_count"].astype(np.int64)
    doc = rows["document_id"].astype(np.int64); off = rows["window_offset"].astype(np.int64)
    n_all = n_7 = 0; per_layer = np.zeros(8, np.int64); occ_w, occ_p = [], []; scanned = 0
    for p in sorted(glob.glob(f"{a.census_root}/worker_*/shards/*.npz")):
        s = np.load(p); tm = token_min(s["raw_b_cka"].astype(np.float32), s["chunk_layout_scale_start"], a.scale)
        order = s["window_sample_order"]; valid = np.arange(512)[None, :] < elig[order][:, None]
        passes = np.nan_to_num(tm, nan=-np.inf) >= cuts[None, None, :]
        per_layer += (passes & valid[..., None]).sum((0, 1))
        cnt = passes.sum(-1); all8 = (cnt >= 8) & valid; n_all += int(all8.sum()); n_7 += int(((cnt >= 7) & valid).sum())
        w, q = np.nonzero(all8); occ_w.append(order[w]); occ_p.append(q.astype(np.int16)); scanned += order.size
    win = np.concatenate(occ_w); pos = np.concatenate(occ_p); o = np.lexsort((pos, win)); win, pos = win[o], pos[o]
    E = int(elig.sum())
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    occ = np.zeros(win.size, dtype=[("source_window_index", "<i8"), ("document_id", "<i8"), ("window_offset", "<i8"), ("position", "<i2")])
    occ["source_window_index"] = win; occ["document_id"] = doc[win]; occ["window_offset"] = off[win]; occ["position"] = pos
    np.save(out / "occurrences_all8.npy", occ)
    rep = {"rule": f"token-min B{a.scale} in top {a.pct}% at all 8 layers", "cuts_per_layer": cuts.tolist(),
           "windows_scanned": scanned, "eligible_tokens": E,
           "per_layer_pass": per_layer.tolist(), "per_layer_pass_pct": (per_layer / E * 100).tolist(),
           "all8": n_all, "all8_pct": n_all / E * 100, "any7of8": n_7, "any7of8_pct": n_7 / E * 100,
           "windows": int(np.unique(win).size), "documents": int(np.unique(doc[win]).size)}
    (out / "summary.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))
if __name__ == "__main__": main()
