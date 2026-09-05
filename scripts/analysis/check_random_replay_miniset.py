"""Preflight a random-control replay miniset.

The random arm has no selector and therefore no mask, so the invariants worth
asserting are that the sample really is window-aligned, keeps its context, and
supervises every position.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    meta_path = os.environ["RANDOM_META"]
    seq = int(os.environ["SEQ_CHECK"])
    meta = json.load(open(meta_path))
    if meta.get("schema") != "random_replay_window_miniset_v1":
        print(f"unexpected schema: {meta.get('schema')!r}", file=sys.stderr)
        return 1
    if meta.get("all_positions_supervised") is not True:
        print("random miniset does not supervise every position", file=sys.stderr)
        return 1
    if meta.get("context_preserved") is not True:
        print("random miniset did not preserve context", file=sys.stderr)
        return 1
    if meta["miniset_tokens"] != meta["sampled_windows"] * seq:
        print("random miniset is not window-aligned", file=sys.stderr)
        return 1
    print(
        f"[PREFLIGHT] random control: windows={meta['sampled_windows']} "
        f"tokens={meta['miniset_tokens']} fraction={meta['achieved_fraction']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
