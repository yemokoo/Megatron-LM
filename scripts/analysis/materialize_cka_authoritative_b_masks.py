#!/usr/bin/env python3
"""Materialize full-census B decisions on the compact candidate-window axis.

The exhaustive census is the authoritative source for B.  A later targeted
forward may move a handful of BF16 values across a frozen threshold when its
batch grouping differs, so this artifact carries both the token consensus and
the per-scale/per-layer valid/pass states needed for an exact same-layer
diagnostic.  No hidden states or sealed-test data are read or stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analysis.cka_gt_pilot_windows import file_sha256
from scripts.analysis.extract_cka_gt_b_candidate_windows import (
    CHUNK_LAYOUT_ARRAY,
    LAYERS,
    SCALES,
    SEGMENT_STOPS,
    SEGMENT_WIDTHS,
    WINDOW_DTYPE,
    _condition_consensus,
    _segment_min_b_for_scale,
    _validated_worker_records,
    load_frozen_b_thresholds,
    load_full_train_manifest,
    validate_census_model_bindings,
)


SCHEMA = "cka_gt_authoritative_b_masks_v1"
PACKED_BYTES = 64


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _pack_tokens(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.bool_)
    if value.ndim not in (2, 3) or value.shape[1] != 8:
        raise ValueError("segment mask must be [N,8] or [N,8,L]")
    expanded = np.repeat(value, 64, axis=1)
    return np.packbits(expanded, axis=1, bitorder="little")


def unpack_tokens(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim not in (2, 3) or value.shape[1] != 64:
        raise ValueError("packed mask must be uint8 [N,64] or [N,64,L]")
    return np.unpackbits(value, axis=1, bitorder="little")[:, :512].astype(bool)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize(
    *,
    census_root: str | Path,
    manifest_path: str | Path,
    analysis_config_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_windows_path: str | Path,
    output_path: str | Path,
    worker_count: int = 4,
) -> dict[str, Any]:
    census_root = Path(census_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    analysis_config_path = Path(analysis_config_path).resolve()
    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    candidate_windows_path = Path(candidate_windows_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite authoritative B artifact: {output_path}")

    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if (
        candidate_manifest.get("schema") != "cka_gt_b_candidate_windows_v2"
        or not candidate_manifest.get("complete")
        or not candidate_manifest.get("not_final_gt")
    ):
        raise RuntimeError("candidate manifest is not authoritative v2")
    recorded_manifest_hash = candidate_manifest.get("manifest_content_sha256")
    unsigned = dict(candidate_manifest)
    unsigned.pop("manifest_content_sha256", None)
    if recorded_manifest_hash != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise RuntimeError("candidate manifest canonical hash differs")
    candidate_identity = _identity(candidate_windows_path)
    if any(
        candidate_manifest.get("candidate_windows", {}).get(key) != candidate_identity[key]
        for key in ("path", "size_bytes", "sha256")
    ):
        raise RuntimeError("candidate-window identity differs from manifest")
    candidates = np.load(candidate_windows_path, mmap_mode="r", allow_pickle=False)
    if candidates.dtype != WINDOW_DTYPE or candidates.ndim != 1 or not len(candidates):
        raise RuntimeError("candidate-window axis is malformed")
    if not np.all(np.diff(candidates["sample_order"]) > 0):
        raise RuntimeError("candidate-window sample_order is not strictly increasing")

    source_manifest, windows = load_full_train_manifest(manifest_path)
    config_identity, bundles = load_frozen_b_thresholds(analysis_config_path)
    analysis_config = json.loads(analysis_config_path.read_text(encoding="utf-8"))
    model_bindings = validate_census_model_bindings(
        census_root=census_root,
        worker_count=int(worker_count),
        manifest_path=manifest_path,
        source_manifest=source_manifest,
        analysis_config_path=analysis_config_path,
        analysis_config_identity=config_identity,
        analysis_config=analysis_config,
    )
    records, merged = _validated_worker_records(
        census_root,
        worker_count=int(worker_count),
        manifest_identity=source_manifest["manifest_content_sha256"],
        analysis_config_sha256=config_identity["sha256"],
        total_windows=int(len(windows)),
        require_merged_summary=True,
    )
    merged_identity = _identity(census_root / "summary.json")
    if any(
        candidate_manifest.get("merged_census_summary", {}).get(key)
        != merged_identity[key]
        for key in ("path", "size_bytes", "sha256")
    ):
        raise RuntimeError("candidate manifest and live merged census differ")
    if candidate_manifest.get("census_worker_model_bindings") != model_bindings:
        raise RuntimeError("candidate manifest and live census model bindings differ")
    if candidate_manifest.get("source_manifest", {}).get("content_identity") != source_manifest.get(
        "manifest_content_sha256"
    ):
        raise RuntimeError("candidate manifest and source manifest differ")

    n = int(len(candidates))
    candidate_orders = np.asarray(candidates["sample_order"], dtype=np.int64)
    seen = np.zeros(n, dtype=np.bool_)
    token_segments = {
        level: np.zeros((n, 8), dtype=np.bool_) for level in bundles
    }
    scale_valid = {
        scale: np.zeros((n, 8, len(LAYERS)), dtype=np.bool_) for scale in SCALES
    }
    scale_pass = {
        (level, scale): np.zeros((n, 8, len(LAYERS)), dtype=np.bool_)
        for level in bundles
        for scale in SCALES
    }
    inventory: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["path"]).resolve()
        observed_sha = file_sha256(path)
        if path.stat().st_size != int(record["size_bytes"]) or observed_sha != record["sha256"]:
            raise RuntimeError(f"raw-B shard identity changed: {path}")
        inventory.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": observed_sha,
                "worker_index": int(record["worker_index"]),
                "index": int(record["index"]),
            }
        )
        with np.load(path, allow_pickle=False) as shard:
            orders = np.asarray(shard["window_sample_order"])
            raw_b = np.asarray(shard["raw_b_cka"])
            layout = np.asarray(shard["chunk_layout_scale_start"])
        if orders.dtype != np.int64 or raw_b.dtype != np.float32:
            raise RuntimeError(f"raw-B shard dtype differs: {path}")
        if raw_b.shape != (len(orders), len(CHUNK_LAYOUT_ARRAY), len(LAYERS)):
            raise RuntimeError(f"raw-B shard shape differs: {path}")
        if not np.array_equal(layout, CHUNK_LAYOUT_ARRAY):
            raise RuntimeError(f"raw-B chunk layout differs: {path}")
        candidate_index = np.searchsorted(candidate_orders, orders)
        within = candidate_index < n
        matched = np.zeros(len(orders), dtype=np.bool_)
        matched[within] = candidate_orders[candidate_index[within]] == orders[within]
        if not matched.any():
            continue
        target = candidate_index[matched]
        if seen[target].any():
            raise RuntimeError("candidate window appears in multiple raw-B shards")
        rows = np.asarray(windows[orders[matched]])
        if not np.array_equal(rows, np.asarray(candidates[target])):
            raise RuntimeError("candidate rows differ from full manifest")
        selected_raw = raw_b[matched]
        eligible_segments = SEGMENT_STOPS[None, :] <= rows["eligible_token_count"][:, None]
        minima = {
            scale: _segment_min_b_for_scale(
                selected_raw, scale=scale, maximum_token_count=512
            )
            for scale in SCALES
        }
        for scale in SCALES:
            scale_valid[scale][target] = (
                np.isfinite(minima[scale]) & eligible_segments[:, :, None]
            )
        for level, bundle in bundles.items():
            token = eligible_segments.copy()
            for scale in SCALES:
                passed = scale_valid[scale][target] & (
                    minima[scale] >= bundle.thresholds[scale][None, None, :]
                )
                scale_pass[(level, scale)][target] = passed
                token &= _condition_consensus(minima[scale], bundle.thresholds[scale])
            token_segments[level][target] = token
        seen[target] = True

    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise RuntimeError(f"authoritative B omitted candidate rows: {missing[:20].tolist()}")
    arrays: dict[str, np.ndarray] = {}
    observed_counts: dict[str, int] = {}
    for level in sorted(bundles):
        arrays[f"bundle_{level}_token_packed"] = _pack_tokens(token_segments[level])
        observed = int((token_segments[level] * SEGMENT_WIDTHS[None, :]).sum(dtype=np.int64))
        expected = int(
            candidate_manifest["statistics"]["per_bundle"][str(level)]["B_passing_tokens"]
        )
        if observed != expected:
            raise RuntimeError(f"authoritative B{level} count differs: {observed} != {expected}")
        observed_counts[str(level)] = observed
        for scale in SCALES:
            arrays[f"bundle_{level}_scale_{scale}_pass_packed"] = _pack_tokens(
                scale_pass[(level, scale)]
            )
    for scale in SCALES:
        arrays[f"scale_{scale}_valid_packed"] = _pack_tokens(scale_valid[scale])
    if not np.all(unpack_tokens(arrays["bundle_99_token_packed"]).any(axis=1)):
        raise RuntimeError("candidate union contains a window without authoritative B99")

    inventory_digest = hashlib.sha256(_canonical_json(inventory)).hexdigest()
    metadata = {
        "schema": SCHEMA,
        "complete": True,
        "role": "authoritative full-census B state on candidate-window axis",
        "candidate_manifest": {
            **_identity(candidate_manifest_path),
            "manifest_content_sha256": recorded_manifest_hash,
        },
        "candidate_windows": candidate_identity,
        "analysis_config": config_identity,
        "merged_census_summary": merged_identity,
        "source_manifest_content_sha256": source_manifest["manifest_content_sha256"],
        "census_worker_model_binding_set_sha256": model_bindings[
            "binding_set_content_sha256"
        ],
        "raw_b_inventory_count": len(inventory),
        "raw_b_inventory_total_bytes": int(sum(item["size_bytes"] for item in inventory)),
        "raw_b_inventory_digest_sha256": inventory_digest,
        "candidate_window_count": n,
        "bundle_b_token_counts": observed_counts,
        "layers": list(LAYERS),
        "scales": list(SCALES),
        "packed_token_axis_bytes": PACKED_BYTES,
        "raw_hidden_stored": False,
        "sealed_test_opened": False,
    }
    arrays["metadata_json"] = np.frombuffer(_canonical_json(metadata), dtype=np.uint8)
    _atomic_npz(output_path, arrays)
    result = {
        **metadata,
        "artifact": _identity(output_path),
    }
    sidecar = output_path.with_suffix(".json")
    _atomic_json(sidecar, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    args = parser.parse_args()
    result = materialize(
        census_root=args.census_root,
        manifest_path=args.manifest,
        analysis_config_path=args.analysis_config,
        candidate_manifest_path=args.candidate_manifest,
        candidate_windows_path=args.candidate_windows,
        output_path=args.output,
        worker_count=args.worker_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
