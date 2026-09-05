#!/usr/bin/env python3
"""Fail-fast integrity checks for the 10M-token paired streaming study."""

import argparse
import hashlib
import json
import os

import numpy as np


LABELS = (
    "code_only_no_replay_no_router_ft_after_expansion_kd_init",
    "code_lm_plus_wiki_layer_output_hidden_kl_router_gradient",
    "code_only_then_wiki_code_router_only_lm_ft",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"root": os.path.abspath(args.root), "domains": {}, "passed": False}
    expected_layers = list(range(2, 10))
    for domain in ("wiki", "code"):
        domain_rows = {}
        canonical_blocks = None
        canonical_samples = None
        for label in LABELS:
            path = os.path.join(args.root, "streaming_stats", domain, label)
            with open(os.path.join(path, "metadata.json"), encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata["target_tokens"] != 10_000_000 or metadata["block_tokens"] != 2_000_000:
                raise RuntimeError(f"wrong token geometry: {path}")
            if metadata["layers"] != expected_layers or metadata["representation"] != "residual_included_transformer_layer_output":
                raise RuntimeError(f"wrong representation/layers: {path}")
            blocks = [(row["tokens"], row["sha256"]) for row in metadata["blocks"]]
            if len(blocks) != 5 or any(count != 2_000_000 for count, _ in blocks):
                raise RuntimeError(f"wrong blocks: {path}")
            samples_hash = file_sha256(os.path.join(path, "samples.jsonl"))
            if canonical_blocks is None:
                canonical_blocks, canonical_samples = blocks, samples_hash
            if blocks != canonical_blocks or samples_hash != canonical_samples:
                raise RuntimeError(f"aligned sample identity mismatch: {path}")
            max_symmetry_error = 0.0
            for index in range(5):
                with np.load(os.path.join(path, f"block_{index:03d}.npz")) as block:
                    if int(block["count"]) != 2_000_000:
                        raise RuntimeError(f"block count mismatch: {path}/block_{index:03d}.npz")
                    for key in ("sum_x", "sum_y", "sum_delta", "xx", "yy", "xy", "scalar_sums"):
                        if not np.isfinite(block[key]).all():
                            raise RuntimeError(f"non-finite {key}: {path}/block_{index:03d}.npz")
                    for key in ("xx", "yy"):
                        matrix = block[key]
                        error = float(np.max(np.abs(matrix - matrix.transpose(0, 2, 1))))
                        max_symmetry_error = max(max_symmetry_error, error)
            with np.load(os.path.join(path, "delta_reservoir.npz")) as reservoir:
                if reservoir["delta"].shape != (8, 4096, 1024):
                    raise RuntimeError(f"reservoir shape mismatch: {path}")
                if not np.isfinite(reservoir["delta"]).all():
                    raise RuntimeError(f"non-finite reservoir: {path}")
            domain_rows[label] = {
                "tokens": 10_000_000,
                "blocks": 5,
                "block_hashes": [digest for _, digest in blocks],
                "samples_jsonl_sha256": samples_hash,
                "max_raw_moment_symmetry_error": max_symmetry_error,
                "bytes": sum(
                    os.path.getsize(os.path.join(directory, filename))
                    for directory, _dirs, files in os.walk(path) for filename in files
                ),
            }
        report["domains"][domain] = domain_rows
    report["passed"] = True
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output + ".inprogress", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    os.replace(args.output + ".inprogress", args.output)
    print(args.output)


if __name__ == "__main__":
    main()
