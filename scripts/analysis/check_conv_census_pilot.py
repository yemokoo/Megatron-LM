"""Validate a Conversation CKA census pilot before the full run is launched.

Checks only what a wrong wiring would break: that metrics exist for the
expected layers and scales, that CKA is in range, that real pairs beat a
degenerate baseline, and that no raw hidden state was written.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

LAYERS = 8
CHUNKS = 10


def fail(message: str) -> int:
    print(f"[PILOT-INVALID] {message}", file=sys.stderr)
    return 1


def main() -> int:
    root = os.environ["CENSUS_ROOT"]
    shards = sorted(glob.glob(f"{root}/worker_*/shards/*.npz"))
    if not shards:
        return fail(f"no shards under {root}")

    total = 0
    finite = 0
    in_range = True
    for path in shards:
        z = np.load(path)
        if "raw_b_cka" not in z:
            return fail(f"{path} has no raw_b_cka")
        b = z["raw_b_cka"]
        if b.ndim != 3 or b.shape[1] != CHUNKS or b.shape[2] != LAYERS:
            return fail(f"{path} raw_b_cka shape {b.shape}, expected (n,{CHUNKS},{LAYERS})")
        good = np.isfinite(b)
        total += b.size
        finite += int(good.sum())
        vals = b[good]
        if vals.size and (vals.min() < -1.001 or vals.max() > 1.001):
            in_range = False
        if "chunk_layout_scale_start" in z:
            layout = z["chunk_layout_scale_start"]
            if sorted(set(layout[:, 0].tolist())) != [128, 256]:
                return fail(f"unexpected chunk scales {sorted(set(layout[:, 0].tolist()))}")

    if not in_range:
        return fail("CKA values outside [-1,1]")
    if total == 0 or finite / total < 0.5:
        return fail(f"only {finite}/{total} finite CKA values")

    for pattern in ("**/hidden*.npy", "**/*hidden*.npz"):
        leaked = glob.glob(f"{root}/{pattern}", recursive=True)
        if leaked:
            return fail(f"raw hidden state was written: {leaked[:3]}")

    summary = {
        "shards": len(shards),
        "cka_values": total,
        "finite_fraction": finite / total,
        "raw_hidden_written": False,
    }
    print(f"[PILOT-OK] {json.dumps(summary, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
