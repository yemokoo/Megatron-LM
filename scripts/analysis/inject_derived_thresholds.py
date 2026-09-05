"""Write derived thresholds into the analysis-config shape the tools expect.

The census-derived thresholds are computed per metric per layer; the candidate
extractor and targeted pass read them from an analysis config under
``candidate_thresholds``.  This rewrites one into the other so the Conversation
selector uses Conversation-calibrated numbers instead of the Code-pilot values
carried along for histogram overlays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = tuple(range(2, 10))
METRIC_SLOTS = {
    "raw_b_128": ("B_lower_threshold", "128"),
    "raw_b_256": ("B_lower_threshold", "256"),
    "token_min_t_128": ("T_lower_threshold", "128"),
    "token_min_t_256": ("T_lower_threshold", "256"),
    "relative_l2": ("relative_l2_upper_threshold", None),
    "abs_log_r": ("abs_log_r_upper_threshold", None),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="analysis config to rewrite")
    parser.add_argument("--thresholds", required=True, help="derived thresholds json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    derived = json.loads(Path(args.thresholds).read_text())

    by_bundle: dict[int, dict] = {}
    for row in derived["thresholds"]:
        slot = METRIC_SLOTS.get(row["metric"])
        if slot is None:
            raise SystemExit(f"unmapped metric {row['metric']!r}")
        key, scale = slot
        entry = by_bundle.setdefault(int(row["bundle"]), {"level": int(row["bundle"])})
        if scale is None:
            entry.setdefault(key, [None] * len(LAYERS))
            entry[key][LAYERS.index(int(row["layer"]))] = float(row["threshold"])
        else:
            entry.setdefault(key, {}).setdefault(scale, [None] * len(LAYERS))
            entry[key][scale][LAYERS.index(int(row["layer"]))] = float(row["threshold"])

    for bundle, entry in by_bundle.items():
        for key, value in entry.items():
            if key == "level":
                continue
            arrays = value.values() if isinstance(value, dict) else [value]
            for array in arrays:
                if any(item is None for item in array):
                    raise SystemExit(f"bundle {bundle} {key} has missing layers")

    config["candidate_thresholds"] = {str(b): by_bundle[b] for b in sorted(by_bundle)}
    config["threshold_provenance"] = derived.get("rule", "derived from old-domain census")
    config["threshold_source_file"] = str(Path(args.thresholds).resolve())
    config["threshold_old_domains"] = sorted(derived.get("old_domains", {}))

    Path(args.output).write_text(json.dumps(config, indent=2, sort_keys=True))
    print(json.dumps({
        "output": args.output,
        "bundles": sorted(config["candidate_thresholds"]),
        "old_domains": config["threshold_old_domains"],
        "b128_bundle95": config["candidate_thresholds"]["95"]["B_lower_threshold"]["128"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
