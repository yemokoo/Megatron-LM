#!/usr/bin/env python3
"""Materialize one human-chosen exact CKA bundle as create-only final GT."""

from __future__ import annotations

import argparse
import json

try:
    from .cka_gt_targeted_gt import materialize_locked_bundle
except ImportError:
    from cka_gt_targeted_gt import materialize_locked_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--bundle", required=True, type=int, choices=(95, 97, 99))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--acknowledge-human-lock",
        action="store_true",
        help="Required guard acknowledging that this creates a final threshold lock.",
    )
    args = parser.parse_args()
    if not args.acknowledge_human_lock:
        parser.error("--acknowledge-human-lock is required")
    result = materialize_locked_bundle(
        exact_dir=args.exact_dir,
        bundle_level=args.bundle,
        output_dir=args.output_dir,
        decision_reason=args.reason,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
