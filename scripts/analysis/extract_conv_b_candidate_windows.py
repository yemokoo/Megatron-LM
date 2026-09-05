"""Select candidate windows for the Conversation GT from the stored chunk CKA.

The Code extractor validates that the manifest is hash-bound to the Code pilot,
which Conversation has no counterpart for.  The selection itself needs none of
that: the census already stored exact per-chunk CKA for every window, so the
token-level B rule can be reapplied directly and reproducibly.

A window is a candidate when at least one eligible token passes the B
consensus at both chunk scales, evaluated at the loosest bundle so the later
exact pass can still tighten to any stricter bundle.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

LAYERS = tuple(range(2, 10))


def token_min_over_chunks(raw_b: np.ndarray, layout: np.ndarray, scale: int,
                          sequence_length: int) -> np.ndarray:
    """Return [windows, tokens, layers] minima over the chunks covering a token.

    A token only counts as stable when every chunk containing it is stable, so
    the covering minimum — not the best chunk — is the quantity the rule uses.
    """
    slots = [i for i, (s, _) in enumerate(layout) if int(s) == scale]
    out = np.full((raw_b.shape[0], sequence_length, len(LAYERS)), np.inf, dtype=np.float32)
    for slot in slots:
        start = int(layout[slot][1])
        stop = min(start + scale, sequence_length)
        chunk = raw_b[:, slot, :][:, None, :]
        np.minimum(out[:, start:stop, :], chunk, out=out[:, start:stop, :])
    out[~np.isfinite(out)] = np.nan
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle", type=int, default=99)
    parser.add_argument("--consensus", type=int, default=7)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()

    config = json.loads(Path(args.analysis_config).read_text())
    thresholds = config["candidate_thresholds"][str(args.bundle)]["B_lower_threshold"]
    cut = {int(scale): np.asarray(values, dtype=np.float32) for scale, values in thresholds.items()}

    manifest = json.loads(Path(args.manifest).read_text())
    windows = np.load(Path(args.manifest).parent / "windows.npy")
    eligible_counts = windows["eligible_token_count"].astype(np.int64)

    shards = sorted(glob.glob(f"{args.census_root}/worker_*/shards/*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {args.census_root}")

    candidates: list[np.ndarray] = []
    scanned = 0
    for path in shards:
        z = np.load(path)
        raw_b = z["raw_b_cka"].astype(np.float32)
        order = z["window_sample_order"]
        layout = z["chunk_layout_scale_start"]
        scanned += order.size

        passes = None
        for scale, cuts in cut.items():
            token_min = token_min_over_chunks(raw_b, layout, scale, args.sequence_length)
            layer_pass = np.nan_to_num(token_min, nan=-np.inf) >= cuts[None, None, :]
            scale_pass = layer_pass.sum(axis=-1) >= args.consensus
            passes = scale_pass if passes is None else (passes & scale_pass)

        # Tokens past a window's eligible count are padding, not candidates.
        positions = np.arange(args.sequence_length)[None, :]
        passes &= positions < eligible_counts[order][:, None]
        hit = passes.any(axis=1)
        if hit.any():
            candidates.append(order[hit])

    candidate_windows = np.sort(np.concatenate(candidates)) if candidates else np.empty(0, np.int64)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "candidate_windows.npy", candidate_windows)

    report = {
        "schema": "conv_b_candidate_windows_v1",
        "census_root": args.census_root,
        "analysis_config": args.analysis_config,
        "manifest": args.manifest,
        "bundle": args.bundle,
        "consensus": f">={args.consensus}/8 per scale, both scales AND",
        "scales": sorted(cut),
        "windows_scanned": int(scanned),
        "manifest_window_count": int(manifest["statistics"]["window_count"]),
        "candidate_windows": int(candidate_windows.size),
        "candidate_fraction": candidate_windows.size / float(scanned) if scanned else 0.0,
        "not_final_gt": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
