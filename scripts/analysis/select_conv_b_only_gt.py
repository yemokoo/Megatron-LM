"""B-only GT over the full Conversation census: chunk-CKA structure, nothing else.

Mirrors the cosine-era rule in spirit (one quantity, per-layer cuts, layer
consensus) but with the relational CKA instead of per-token direction.  T and
the magnitude conditions are intentionally dropped so the comparison isolates
"structure preserved" from "structure + contribution + scale preserved".
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

LAYERS = tuple(range(2, 10))


def token_min_over_chunks(raw_b, layout, scale, seq):
    slots = [i for i, (s, _) in enumerate(layout) if int(s) == scale]
    out = np.full((raw_b.shape[0], seq, len(LAYERS)), np.inf, dtype=np.float32)
    for slot in slots:
        start = int(layout[slot][1]); stop = min(start + scale, seq)
        np.minimum(out[:, start:stop, :], raw_b[:, slot, :][:, None, :], out=out[:, start:stop, :])
    out[~np.isfinite(out)] = np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--thresholds", required=True, help="derived thresholds json")
    ap.add_argument("--domain", required=True, help="which per_domain cut to use, e.g. wiki")
    ap.add_argument("--bundle", type=int, default=95)
    ap.add_argument("--scale", type=int, default=128)
    ap.add_argument("--consensus", type=int, default=7)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    th = json.loads(Path(args.thresholds).read_text())
    cut = np.array([[r["per_domain"][args.domain] for r in th["thresholds"]
                     if r["bundle"] == args.bundle and r["metric"] == f"raw_b_{args.scale}"
                     and r["layer"] == L][0] for L in LAYERS], dtype=np.float32)

    rows = np.load(Path(args.manifest).parent / "windows.npy")
    elig = rows["eligible_token_count"].astype(np.int64)
    doc = rows["document_id"].astype(np.int64); off = rows["window_offset"].astype(np.int64)

    occ_win, occ_pos = [], []
    scanned = 0
    for path in sorted(glob.glob(f"{args.census_root}/worker_*/shards/*.npz")):
        z = np.load(path)
        raw_b = z["raw_b_cka"].astype(np.float32); order = z["window_sample_order"]
        tmin = token_min_over_chunks(raw_b, z["chunk_layout_scale_start"], args.scale, 512)
        ok = (np.nan_to_num(tmin, nan=-np.inf) >= cut[None, None, :]).sum(-1) >= args.consensus
        ok &= np.arange(512)[None, :] < elig[order][:, None]
        w, p = np.nonzero(ok)
        occ_win.append(order[w]); occ_pos.append(p.astype(np.int16))
        scanned += order.size

    win = np.concatenate(occ_win); pos = np.concatenate(occ_pos)
    o = np.lexsort((pos, win)); win, pos = win[o], pos[o]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    occ = np.zeros(win.size, dtype=[("source_window_index", "<i8"), ("document_id", "<i8"),
                                    ("window_offset", "<i8"), ("position", "<i2")])
    occ["source_window_index"] = win; occ["document_id"] = doc[win]
    occ["window_offset"] = off[win]; occ["position"] = pos
    np.save(out / f"bundle_{args.bundle}_occurrences.npy", occ)

    E = int(elig.sum())
    report = {
        "schema": "conv_b_only_gt_v1", "rule": f"B{args.scale} only, >={args.consensus}/8 layers",
        "cut_domain": args.domain, "bundle": args.bundle, "cut": cut.tolist(),
        "windows_scanned": int(scanned), "eligible_tokens": E,
        "selected": int(win.size), "coverage": win.size / E,
        "windows": int(np.unique(win).size), "documents": int(np.unique(doc[win]).size),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("cut_domain", "bundle", "selected", "coverage", "windows", "documents")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
