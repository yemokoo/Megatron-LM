"""Derive selector thresholds from old-domain census histograms.

The rule is the one already used for Code: a bundle level N means "the value
that N% of the old domain reaches", read per metric per layer.  Conversation
has two old domains (Wiki and Code), so a threshold must be passed by the
old data on *both* sides; the conservative choice is taken for each metric.

Lower-bound metrics (chunk CKA, token contribution) take the lower quantile;
upper-bound magnitude metrics take the upper quantile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOWER_METRICS = ("raw_b_128", "raw_b_256", "token_min_t_128", "token_min_t_256")
UPPER_METRICS = ("relative_l2", "abs_log_r")
LAYERS = tuple(range(2, 10))


def load_histograms(root: Path) -> dict:
    path = root / "histograms.npz"
    if not path.is_file():
        raise SystemExit(f"missing histograms: {path}")
    return dict(np.load(path))


def quantile(hist: dict, metric: str, layer_index: int, q: float) -> float:
    counts = hist[f"hist__{metric}__counts"][layer_index].astype(np.float64)
    edges = hist[f"hist__{metric}__edges"]
    under = float(hist[f"hist__{metric}__underflow"][layer_index])
    over = float(hist[f"hist__{metric}__overflow"][layer_index])
    total = counts.sum() + under + over
    if total <= 0:
        raise SystemExit(f"empty histogram for {metric} layer index {layer_index}")
    # Interpolate inside the crossing bin; 4096 bins bound the resolution, which
    # is reported next to every threshold rather than hidden.
    target = q * total
    cumulative = under + np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target))
    index = min(index, counts.size - 1)
    below = under + (counts[:index].sum() if index else 0.0)
    within = counts[index]
    frac = 0.0 if within <= 0 else min(max((target - below) / within, 0.0), 1.0)
    return float(edges[index] + frac * (edges[index + 1] - edges[index]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-census", action="append", required=True,
                        metavar="NAME=ROOT", help="repeatable old-domain census root")
    parser.add_argument("--bundle", type=int, action="append", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("permissive", "strict"), default="permissive")
    args = parser.parse_args()
    bundles = args.bundle or [95, 97, 99]
    mode = args.mode

    domains = {}
    for entry in args.old_census:
        name, _, root = entry.partition("=")
        if not root:
            raise SystemExit(f"expected NAME=ROOT, got {entry!r}")
        domains[name] = load_histograms(Path(root))

    rows = []
    for bundle in bundles:
        lower_q, upper_q = (100 - bundle) / 100.0, bundle / 100.0
        for metric in LOWER_METRICS + UPPER_METRICS:
            upper = metric in UPPER_METRICS
            for layer_index, layer in enumerate(LAYERS):
                per_domain = {
                    name: quantile(hist, metric, layer_index, upper_q if upper else lower_q)
                    for name, hist in domains.items()
                }
                if mode == "strict":
                    # An anchor must sit inside EVERY old domain's band, so the
                    # binding cut is the strict side: highest lower-bound,
                    # lowest upper-bound.
                    value = min(per_domain.values()) if upper else max(per_domain.values())
                else:
                    # Permissive union: inside at least the widest band.
                    value = max(per_domain.values()) if upper else min(per_domain.values())
                rows.append({
                    "bundle": bundle, "metric": metric, "layer": layer,
                    "comparison": "<=" if upper else ">=",
                    "threshold": value, "per_domain": per_domain,
                })

    payload = {
        "schema": "cka_thresholds_from_old_domain_census_v1",
        "old_domains": {name: str(entry) for name, entry in zip(domains, args.old_census)},
        "bundles": bundles,
        "histogram_bins": int(next(iter(domains.values()))["hist__raw_b_128__counts"].shape[1]),
        "mode": mode,
        "rule": ("bundle N = value reached by N% of each old domain; strict mode "
                 "requires every domain band (highest lower cut, lowest upper cut), "
                 "permissive mode requires only the widest band"),
        "thresholds": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    for bundle in bundles:
        b = [r for r in rows if r["bundle"] == bundle and r["metric"] == "raw_b_128"]
        print(f"bundle {bundle} B128 lower: " + " ".join(f"{r['threshold']:.6f}" for r in b))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
