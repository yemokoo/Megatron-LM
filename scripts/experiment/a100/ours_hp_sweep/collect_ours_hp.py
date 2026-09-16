#!/usr/bin/env python3
"""Collect AA / FM / LA for the Ours HP sweep cells.

Reads the per-stage probe lines Megatron writes
("probe <name> at iteration N | ... | next_token_acc: X") and reduces each cell
to the same three numbers the DoF table uses:

    LA = (a11 + a22 + a33) / 3     learned-after accuracy
    FM = mean over old tasks of (best earlier - final)
    AA = final 3-task average      (identity check: AA = LA - (2/3) FM)

a11 is the wiki source accuracy. No sweep stage produces it: the ours wiki
source is an HF-migrated checkpoint directory with no probe log, which is why
the DoF extractor cannot read it from the stage tree either. It comes instead
from the dedicated probe pass in wiki_source_probe_20260908, whose hybrid
(ffn+attn shared router) reading is 0.467627 -- the default below. The ffn-only
source is 0.460047 and is the wrong number for this sweep. The printed identity
residual (AA - (LA - 2/3 FM)) is the check: a wrong a11 shows up there.

  python collect_ours_hp.py
  python collect_ours_hp.py --json out.json
"""
import argparse
import json
import os
import re

# ffn+attn shared-router wiki source, from
# wiki_source_probe_20260908/hybrid.probe at iteration 1800.
WIKI_SOURCE_ACC_HYBRID = 0.467627
CENTRE = ("r0p1_kd360",
          "/data2/seonghyeonnoh/LLM-continual-learning-runs/ours_hyb_kd360_sub0p1_20260908")
DEFAULT_SWEEP = "/data2/seonghyeonnoh/LLM-continual-learning-runs/ours_hp_sweep_20260912"
CELLS = [
    ("r0p01_kd180", "0.01%", 180), ("r0p01_kd360", "0.01%", 360),
    ("r0p01_kd720", "0.01%", 720), ("r0p1_kd180", "0.1%", 180),
    ("r0p1_kd360", "0.1%", 360), ("r0p1_kd720", "0.1%", 720),
    ("r1_kd360", "1%", 360),
]
PROBE = re.compile(r"probe (\S+) at iteration (\d+).*?next_token_acc: ([0-9.]+)")


def last_probes(directory):
    """Final probe reading per name across every log in a stage directory."""
    out = {}
    logs = os.path.join(directory, "logs")
    if not os.path.isdir(logs):
        return out
    for name in sorted(os.listdir(logs)):
        if not name.endswith(".log"):
            continue
        with open(os.path.join(logs, name), errors="ignore") as handle:
            for line in handle:
                match = PROBE.search(line)
                if match:
                    out[match.group(1)] = float(match.group(3))
    return out


def cell_metrics(root, a11):
    code = last_probes(os.path.join(root, "code_1phase"))
    conv = last_probes(os.path.join(root, "conv_1phase"))
    if not code or not conv:
        return None
    a22 = code.get("code_probe")
    a21 = code.get("wiki_probe")
    final = {task: conv.get(f"{task}_probe")
             for task in ("wiki", "code", "conversation")}
    if a22 is None or None in final.values():
        return None
    a33 = final["conversation"]
    la = (a11 + a22 + a33) / 3
    fm_wiki = max(a11, a21 if a21 is not None else a11) - final["wiki"]
    fm_code = a22 - final["code"]
    fm = (fm_wiki + fm_code) / 2
    aa = sum(final.values()) / 3
    return {"LA": la, "FM": fm, "AA": aa, "identity_residual": aa - (la - 2 / 3 * fm),
            "final": final, "a22": a22, "a21": a21}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default=DEFAULT_SWEEP)
    parser.add_argument("--wiki-source-acc", type=float,
                        default=WIKI_SOURCE_ACC_HYBRID,
                        help="a11 for the ffn_attn_shared_router wiki source "
                             f"(default {WIKI_SOURCE_ACC_HYBRID}, from "
                             "wiki_source_probe_20260908/hybrid.probe)")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    rows = {}
    print(f"{'cell':<14}{'replay':>8}{'kd':>6}{'AA':>9}{'FM':>10}{'LA':>9}"
          f"{'wiki':>8}{'code':>8}{'conv':>8}{'id.res':>9}")
    for cell, replay, kd in CELLS:
        root = CENTRE[1] if cell == CENTRE[0] else os.path.join(args.sweep_root, cell)
        metrics = cell_metrics(root, args.wiki_source_acc)
        if metrics is None:
            print(f"{cell:<14}{replay:>8}{kd:>6}{'  (incomplete)':>36}")
            continue
        rows[cell] = dict(metrics, replay=replay, kd_iters=kd, root=root)
        final = metrics["final"]
        print(f"{cell:<14}{replay:>8}{kd:>6}{metrics['AA']:9.4f}{metrics['FM']:+10.4f}"
              f"{metrics['LA']:9.4f}{final['wiki']:8.4f}{final['code']:8.4f}"
              f"{final['conversation']:8.4f}{metrics['identity_residual']:+9.1e}")
    worst = max((abs(r["identity_residual"]) for r in rows.values()), default=0.0)
    print(f"\nAA = LA - (2/3)FM residual, max |.| = {worst:.1e} "
          f"(a11 = {args.wiki_source_acc}; a large residual means the wrong a11)")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
