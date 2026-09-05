"""Write the census analysis config for the Conversation checkpoint pair.

The census refuses to run unless the config's bound dataset and checkpoint
identities match what it recomputes at startup.  Building the config with the
very same identity helpers is what makes that check meaningful instead of
something to be worked around.

Thresholds are carried over from the Code pilot only so the census can draw its
histogram overlay lines; the census itself is threshold-free, and the
Conversation selector thresholds are derived later from Wiki/Code calibration
histograms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cka_gt_pilot_windows import checkpoint_identity, source_dataset_identity


def normalized_source(prefix: str) -> dict:
    live = source_dataset_identity(prefix)
    return {
        "schema": live.get("schema"),
        "storage_kind": live.get("storage_kind"),
        "resolved_prefix": live.get("resolved_prefix"),
        "idx_size_bytes": live.get("idx", {}).get("size_bytes"),
        "idx_sha256": live.get("idx", {}).get("sha256"),
        "bin_size_bytes": live.get("bin", {}).get("size_bytes"),
        "bin_sha256": live.get("bin", {}).get("sha256"),
    }


# Exactly the key set the census recomputes and compares against, in the same
# shape: a missing or extra key makes the comparison fail even when the
# checkpoint is the right one.
CHECKPOINT_KEYS = ("schema", "storage_kind", "resolved_root", "tracker_step",
                   "iteration_dir", "total_bytes", "content_sha256")


def normalized_checkpoint(root: str) -> dict:
    live = checkpoint_identity(root)
    return {key: live.get(key) for key in CHECKPOINT_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="Code pilot analysis_config.json")
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.template).read_text())
    config["schema"] = "cka_gt_pilot_postprocess_v1"
    config["task"] = "conversation"
    config["threshold_provenance"] = (
        "candidate_thresholds are Code-pilot values retained for histogram overlay "
        "only; Conversation selector thresholds come from Wiki+Code calibration"
    )
    provenance = config.setdefault("prepared_input_provenance", {})
    provenance["available"] = True
    provenance["source_dataset_identity"] = {"code": normalized_source(args.dataset_prefix)}
    provenance["checkpoint_identity"] = {
        "before": normalized_checkpoint(args.before),
        "after": normalized_checkpoint(args.after),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, sort_keys=True))
    print(json.dumps({
        "output": str(out),
        "dataset_prefix": provenance["source_dataset_identity"]["code"]["resolved_prefix"],
        "before_step": provenance["checkpoint_identity"]["before"].get("tracker_step"),
        "after_step": provenance["checkpoint_identity"]["after"].get("tracker_step"),
        "layers": config.get("layers"),
        "scales_used_for_gt": config.get("scales_used_for_gt"),
        "threshold_keys": sorted(config.get("candidate_thresholds", {})),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
