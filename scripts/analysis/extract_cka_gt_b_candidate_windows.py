#!/usr/bin/env python3
"""Extract a conservative B-only window set for the targeted CKA GT pass.

The full census deliberately stores exact chunk CKA (``raw_b_cka``), but not
wide token metrics.  This CPU-only post-process reconstructs, for every
eligible token, the minimum CKA over overlapping chunks at scales 128 and
256.  It applies the frozen pilot B thresholds with the production
condition-specific consensus rule at each scale::

    (pass >= 7 of 8 valid layers) OR (exactly 6 valid and pass 6 of 6)

and requires both scales to pass.  A window is a candidate when at least one
eligible token passes B for *any* frozen threshold bundle.  Consequently the
output is a conservative superset of every B95/B97/B99 token candidate.  For
the authoritative monotone thresholds the union is also exactly the B99
window set.  T and magnitude metrics are intentionally not used here; they
are recomputed only for these windows in the later targeted GPU pass.

Before emitting anything, every worker journal and shard is authenticated and
the shard ``sample_order`` values are proved to cover the full manifest once,
without gaps or duplicates.  No model, dataset, or GPU is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .cka_gt_full_census import (
        CHUNK_LAYOUT_ARRAY,
        LAYERS,
        MERGED_SCHEMA,
        PROGRESS_SCHEMA,
        SHARD_SCHEMA,
        SUMMARY_SCHEMA,
        load_full_train_manifest,
        partition_bounds,
    )
    from .cka_gt_pilot_windows import WINDOW_DTYPE, file_sha256
except ImportError:  # Direct script execution.
    from cka_gt_full_census import (
        CHUNK_LAYOUT_ARRAY,
        LAYERS,
        MERGED_SCHEMA,
        PROGRESS_SCHEMA,
        SHARD_SCHEMA,
        SUMMARY_SCHEMA,
        load_full_train_manifest,
        partition_bounds,
    )
    from cka_gt_pilot_windows import WINDOW_DTYPE, file_sha256


SCHEMA = "cka_gt_b_candidate_windows_v2"
REPORT_SCHEMA = "cka_gt_b_candidate_extraction_report_v1"
MODEL_BINDING_SET_SCHEMA = "cka_gt_full_census_model_binding_set_v1"
MODEL_BINDING_SCHEMA = "cka_gt_full_census_model_binding_v2"
SCALES = (128, 256)
REQUIRED_LEVELS = (95, 97, 99)


def _atomic_segments() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return intervals on which membership in every chunk is constant."""

    boundaries = {0, 512}
    for scale, start in CHUNK_LAYOUT_ARRAY.tolist():
        boundaries.add(int(start))
        boundaries.add(int(start) + int(scale))
    ordered = np.asarray(sorted(boundaries), dtype=np.int32)
    return ordered[:-1], ordered[1:], np.diff(ordered)


SEGMENT_STARTS, SEGMENT_STOPS, SEGMENT_WIDTHS = _atomic_segments()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as handle:
            check = np.load(handle, allow_pickle=False)
            if check.dtype != array.dtype or check.shape != array.shape:
                raise RuntimeError("candidate NPY round-trip validation failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


@dataclass(frozen=True)
class BThresholdBundle:
    level: int
    thresholds: dict[int, np.ndarray]

    def serializable(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "B_lower_threshold": {
                str(scale): self.thresholds[scale].tolist() for scale in SCALES
            },
        }


def load_frozen_b_thresholds(
    analysis_config_path: str | Path,
) -> tuple[dict[str, Any], dict[int, BThresholdBundle]]:
    """Load and strictly validate every frozen B threshold bundle."""

    path = Path(analysis_config_path)
    raw = path.read_bytes()
    config = json.loads(raw)
    if list(config.get("layers", ())) != list(LAYERS):
        raise ValueError(f"analysis config must use residual layers {LAYERS}")
    if set(config.get("scales_used_for_gt", ())) != set(SCALES):
        raise ValueError(f"analysis config must use scales {SCALES} for GT")
    threshold_payload = config.get("candidate_thresholds")
    if not isinstance(threshold_payload, Mapping) or not threshold_payload:
        raise ValueError("analysis config has no candidate_thresholds")
    bundles: dict[int, BThresholdBundle] = {}
    for label, payload in threshold_payload.items():
        try:
            level = int(label)
        except (TypeError, ValueError) as error:
            raise ValueError(f"candidate level is not an integer: {label!r}") from error
        if level in bundles or not isinstance(payload, Mapping):
            raise ValueError(f"invalid/duplicate candidate bundle {label!r}")
        raw_b = payload.get("B_lower_threshold")
        if not isinstance(raw_b, Mapping):
            raise ValueError(f"bundle {level} has no B_lower_threshold")
        by_scale: dict[int, np.ndarray] = {}
        for scale in SCALES:
            values = raw_b.get(str(scale), raw_b.get(scale))
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (len(LAYERS),) or not np.isfinite(array).all():
                raise ValueError(
                    f"bundle {level} scale {scale} threshold must be finite [8]"
                )
            if np.any(array < -1.0) or np.any(array > 1.0):
                raise ValueError(f"bundle {level} scale {scale} is outside [-1,1]")
            by_scale[scale] = array
        bundles[level] = BThresholdBundle(level, by_scale)
    missing = set(REQUIRED_LEVELS) - set(bundles)
    if missing:
        raise ValueError(f"analysis config is missing required bundles {sorted(missing)}")
    identity = {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema": config.get("schema"),
        "consensus_from_config": config.get("consensus"),
    }
    return identity, dict(sorted(bundles.items()))


def _normalized_source_dataset_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize flattened binding and nested manifest source identities."""

    def component(name: str, field: str) -> Any:
        direct = value.get(f"{name}_{field}")
        nested = value.get(name)
        if direct is not None:
            return direct
        return nested.get(field) if isinstance(nested, Mapping) else None

    result = {
        "schema": value.get("schema"),
        "storage_kind": value.get("storage_kind"),
        "resolved_prefix": str(Path(str(value.get("resolved_prefix"))).resolve()),
        "idx_sha256": component("idx", "sha256"),
        "idx_size_bytes": component("idx", "size_bytes"),
        "bin_sha256": component("bin", "sha256"),
        "bin_size_bytes": component("bin", "size_bytes"),
    }
    for key in ("idx_sha256", "bin_sha256"):
        digest = result[key]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"source dataset {key} is not a SHA-256 digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"source dataset {key} is not hexadecimal") from error
    for key in ("idx_size_bytes", "bin_size_bytes"):
        result[key] = int(result[key])
        if result[key] < 0:
            raise ValueError(f"source dataset {key} is negative")
    return result


def _checkpoint_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} checkpoint identity is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} checkpoint identity is not hexadecimal") from error
    return value


def validate_census_model_bindings(
    *,
    census_root: str | Path,
    worker_count: int,
    manifest_path: str | Path,
    source_manifest: Mapping[str, Any],
    analysis_config_path: str | Path,
    analysis_config_identity: Mapping[str, Any],
    analysis_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless all census workers bind to one authoritative run."""

    census_root = Path(census_root)
    manifest_path = Path(manifest_path).resolve()
    analysis_config_path = Path(analysis_config_path).resolve()
    prepared = analysis_config.get("prepared_input_provenance")
    if not isinstance(prepared, Mapping) or not prepared.get("available"):
        raise RuntimeError("analysis config lacks authoritative prepared-input provenance")
    checkpoints = prepared.get("checkpoint_identity")
    sources = prepared.get("source_dataset_identity")
    if not isinstance(checkpoints, Mapping) or not isinstance(sources, Mapping):
        raise RuntimeError("analysis config lacks checkpoint/source identities")
    before = checkpoints.get("before")
    after = checkpoints.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise RuntimeError("analysis config checkpoint identities are malformed")
    reference_identity = _checkpoint_digest(
        before.get("content_sha256"), label="reference"
    )
    current_identity = _checkpoint_digest(after.get("content_sha256"), label="current")
    code_source = sources.get("code")
    if not isinstance(code_source, Mapping):
        raise RuntimeError("analysis config lacks the Code source identity")
    expected_source = _normalized_source_dataset_identity(code_source)

    pilot = source_manifest.get("pilot_manifest_identity")
    pilot_source = pilot.get("source_dataset_identity") if isinstance(pilot, Mapping) else None
    if not isinstance(pilot_source, Mapping):
        raise RuntimeError("full manifest is not hash-bound to a pilot Code source identity")
    if _normalized_source_dataset_identity(pilot_source) != expected_source:
        raise RuntimeError("full manifest and analysis config bind different Code sources")
    light = source_manifest.get("dataset_identity_light")
    if not isinstance(light, Mapping):
        raise RuntimeError("full manifest lacks its light dataset identity")
    if str(Path(str(light.get("resolved_prefix"))).resolve()) != expected_source["resolved_prefix"]:
        raise RuntimeError("full manifest dataset prefix differs from the Code source")
    for name in ("idx", "bin"):
        item = light.get(name)
        if not isinstance(item, Mapping) or int(item.get("size_bytes", -1)) != expected_source[f"{name}_size_bytes"]:
            raise RuntimeError(f"full manifest Code {name} size differs")

    file_records: list[dict[str, Any]] = []
    common_raw: dict[str, Any] | None = None
    per_worker: list[dict[str, int]] = []
    for worker in range(int(worker_count)):
        worker_dir = census_root / f"worker_{worker:03d}"
        binding_path = worker_dir / "model_binding.json"
        if not binding_path.is_file():
            raise RuntimeError(f"worker {worker} model binding is missing")
        raw = binding_path.read_bytes()
        try:
            binding = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"worker {worker} model binding is invalid JSON") from error
        if binding.get("schema") != MODEL_BINDING_SCHEMA:
            raise RuntimeError(f"worker {worker} model-binding schema differs")
        if int(binding.get("worker_index", -1)) != worker:
            raise RuntimeError(f"worker {worker} model-binding index differs")
        if int(binding.get("worker_count", -1)) != int(worker_count):
            raise RuntimeError(f"worker {worker} model-binding worker-count differs")
        start, stop = partition_bounds(
            int(source_manifest["statistics"]["window_count"]), worker, int(worker_count)
        )
        if int(binding.get("partition_windows", -1)) != stop - start:
            raise RuntimeError(f"worker {worker} model-binding partition size differs")
        if Path(str(binding.get("manifest_path"))).resolve() != manifest_path:
            raise RuntimeError(f"worker {worker} binds a different full manifest")
        if Path(str(binding.get("analysis_config"))).resolve() != analysis_config_path:
            raise RuntimeError(f"worker {worker} binds a different analysis config")
        if binding.get("reference_checkpoint_identity") != reference_identity:
            raise RuntimeError(f"worker {worker} reference checkpoint identity differs")
        if binding.get("current_checkpoint_identity") != current_identity:
            raise RuntimeError(f"worker {worker} current checkpoint identity differs")
        source = binding.get("source_dataset_identity")
        if not isinstance(source, Mapping) or _normalized_source_dataset_identity(source) != expected_source:
            raise RuntimeError(f"worker {worker} Code source identity differs")
        if binding.get("raw_hidden_stored") is not False:
            raise RuntimeError(f"worker {worker} unexpectedly claims raw-hidden storage")
        if binding.get("threshold_policy") != "none_distribution_census_pilot_lines_overlay_only":
            raise RuntimeError(f"worker {worker} threshold policy is not census-only")
        if int(binding.get("max_windows", -1)) != 0:
            raise RuntimeError(f"worker {worker} was not a full-corpus census")

        progress = json.loads((worker_dir / "progress.json").read_text(encoding="utf-8"))
        summary = json.loads((worker_dir / "summary.json").read_text(encoding="utf-8"))
        logical_batch = int(binding.get("window_batch_size", -1))
        if logical_batch <= 0 or any(
            int(payload.get("batch_size", -1)) != logical_batch
            for payload in (progress, summary)
        ):
            raise RuntimeError(f"worker {worker} logical window batch differs")
        histogram_bins = int(binding.get("histogram_bins", -1))
        reservoir_size = int(binding.get("global_token_score_reservoir_size", -1))
        if any(int(payload.get("histogram_bins", -1)) != histogram_bins for payload in (progress, summary)):
            raise RuntimeError(f"worker {worker} histogram-bin binding differs")
        if any(int(payload.get("reservoir_size", -1)) != reservoir_size for payload in (progress, summary)):
            raise RuntimeError(f"worker {worker} reservoir-size binding differs")
        checkpoint_batches = int(binding.get("checkpoint_every_batches", -1))
        journal = progress.get("shards", ())
        if checkpoint_batches <= 0 or not journal:
            raise RuntimeError(f"worker {worker} checkpoint cadence is invalid")
        spans = [int(item["batch_stop"]) - int(item["batch_start"]) for item in journal]
        if any(span != checkpoint_batches for span in spans[:-1]) or not (0 < spans[-1] <= checkpoint_batches):
            raise RuntimeError(f"worker {worker} checkpoint cadence differs from its journal")
        if progress.get("max_windows") is not None or summary.get("max_windows") is not None:
            raise RuntimeError(f"worker {worker} progress is not a full census")

        common = dict(binding)
        common.pop("worker_index")
        common.pop("partition_windows")
        if common_raw is None:
            common_raw = common
        elif _canonical_json(common) != _canonical_json(common_raw):
            raise RuntimeError("census worker model bindings disagree on common identity")
        per_worker.append(
            {"worker_index": worker, "partition_windows": stop - start}
        )
        file_records.append(
            {
                "worker_index": worker,
                "file_identity": {
                    "path": str(binding_path.resolve()),
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "content": binding,
            }
        )
    if common_raw is None:
        raise RuntimeError("no census model bindings were found")
    common_content = {
        "binding_schema": MODEL_BINDING_SCHEMA,
        "analysis_config": {
            "path": str(analysis_config_path),
            "sha256": analysis_config_identity["sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "content_identity": source_manifest["manifest_content_sha256"],
        },
        "reference_checkpoint_identity": reference_identity,
        "current_checkpoint_identity": current_identity,
        "source_dataset_identity": expected_source,
        "window_batch_size": int(common_raw["window_batch_size"]),
        "checkpoint_every_batches": int(common_raw["checkpoint_every_batches"]),
        "histogram_bins": int(common_raw["histogram_bins"]),
        "global_token_score_reservoir_size": int(
            common_raw["global_token_score_reservoir_size"]
        ),
        "max_windows": int(common_raw["max_windows"]),
        "raw_hidden_stored": bool(common_raw["raw_hidden_stored"]),
        "threshold_policy": common_raw["threshold_policy"],
    }
    binding_set = {
        "schema": MODEL_BINDING_SET_SCHEMA,
        "validated": True,
        "all_workers_agree": True,
        "worker_count": int(worker_count),
        "files": file_records,
        "common_content": common_content,
        "common_content_sha256": hashlib.sha256(_canonical_json(common_content)).hexdigest(),
        "per_worker_content": per_worker,
    }
    binding_set["binding_set_content_sha256"] = hashlib.sha256(
        _canonical_json(binding_set)
    ).hexdigest()
    return binding_set


def _condition_consensus(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Return the production condition-specific lower-bound consensus mask."""

    if values.ndim != 3 or values.shape[-1] != len(LAYERS):
        raise ValueError(f"expected [windows,tokens,8], got {values.shape}")
    valid = np.isfinite(values)
    n_valid = valid.sum(axis=-1, dtype=np.int16)
    passed = (valid & (values >= threshold[None, None, :])).sum(
        axis=-1, dtype=np.int16
    )
    return (n_valid >= 6) & ((passed >= 7) | ((n_valid == 6) & (passed == 6)))


def _segment_min_b_for_scale(
    raw_b: np.ndarray, *, scale: int, maximum_token_count: int
) -> np.ndarray:
    """Compute minima on exact atomic token intervals.

    Chunk membership changes only at a chunk start/end.  The fixed 128/256
    grids therefore induce eight 64-token intervals over a full window.  A
    score computed once per interval is exactly the score of every token in
    it, cutting CPU work and temporary memory by 64x versus a 512-token
    expansion.
    """

    count = raw_b.shape[0]
    segment_mask = SEGMENT_STOPS <= int(maximum_token_count)
    starts, stops = SEGMENT_STARTS[segment_mask], SEGMENT_STOPS[segment_mask]
    result = np.full(
        (count, int(starts.size), len(LAYERS)), np.nan, dtype=np.float32
    )
    for slot, (slot_scale, start) in enumerate(CHUNK_LAYOUT_ARRAY.tolist()):
        if int(slot_scale) != int(scale):
            continue
        covered = (starts >= int(start)) & (stops <= int(start) + int(scale))
        if not covered.any():
            continue
        # np.fmin deliberately preserves a finite value when the other input
        # is NaN.  Thus absent tail chunks cannot erase a valid overlapping
        # chunk, while intervals with no valid chunk remain NaN.
        # Boolean indexing returns a copy, so assign the fmin result back
        # explicitly rather than relying on an ``out=`` view.
        result[:, covered] = np.fmin(
            result[:, covered], raw_b[:, slot, None, :]
        )
    return result


@dataclass
class ShardCandidateResult:
    window_masks: dict[int, np.ndarray]
    union_window_mask: np.ndarray
    token_counts: dict[int, int]
    union_token_count: int


def candidate_masks_from_raw_b(
    raw_b: np.ndarray,
    rows: np.ndarray,
    bundles: Mapping[int, BThresholdBundle],
    *,
    block_windows: int = 2048,
) -> ShardCandidateResult:
    """Compute exact per-bundle and conservative-union candidate masks."""

    raw_b, rows = np.asarray(raw_b), np.asarray(rows)
    if raw_b.dtype != np.float32 or raw_b.shape != (
        len(rows), len(CHUNK_LAYOUT_ARRAY), len(LAYERS)
    ):
        raise ValueError(
            "raw_b must be float32 [windows,fixed_chunk_slots,8] aligned to rows"
        )
    if rows.dtype != WINDOW_DTYPE or rows.ndim != 1:
        raise ValueError("rows must be one-dimensional WINDOW_DTYPE")
    if int(block_windows) <= 0:
        raise ValueError("block_windows must be positive")
    window_masks = {level: np.zeros(len(rows), dtype=np.bool_) for level in bundles}
    token_counts = {level: 0 for level in bundles}
    union_window_mask = np.zeros(len(rows), dtype=np.bool_)
    union_token_count = 0
    for begin in range(0, len(rows), int(block_windows)):
        end = min(len(rows), begin + int(block_windows))
        block_rows, block_raw = rows[begin:end], raw_b[begin:end]
        maximum = int(block_rows["eligible_token_count"].max(initial=0))
        if maximum == 0:
            continue
        if np.any(block_rows["eligible_token_count"] % SEGMENT_WIDTHS[0] != 0):
            raise ValueError("eligible-token boundary does not align to the fixed chunk grid")
        segment_mask = SEGMENT_STOPS <= maximum
        segment_stops = SEGMENT_STOPS[segment_mask]
        segment_widths = SEGMENT_WIDTHS[segment_mask]
        eligible = segment_stops[None, :] <= block_rows["eligible_token_count"][:, None]
        minima = {
            scale: _segment_min_b_for_scale(
                block_raw, scale=scale, maximum_token_count=maximum
            )
            for scale in SCALES
        }
        union_tokens = np.zeros(eligible.shape, dtype=np.bool_)
        for level, bundle in bundles.items():
            token_mask = eligible.copy()
            for scale in SCALES:
                token_mask &= _condition_consensus(
                    minima[scale], bundle.thresholds[scale]
                )
            window = token_mask.any(axis=1)
            window_masks[level][begin:end] = window
            union_window_mask[begin:end] |= window
            token_counts[level] += int(
                (token_mask * segment_widths[None, :]).sum(dtype=np.int64)
            )
            union_tokens |= token_mask
        union_token_count += int(
            (union_tokens * segment_widths[None, :]).sum(dtype=np.int64)
        )
    return ShardCandidateResult(
        window_masks, union_window_mask, token_counts, union_token_count
    )


def _validated_worker_records(
    census_root: Path,
    *,
    worker_count: int,
    manifest_identity: str,
    analysis_config_sha256: str,
    total_windows: int,
    require_merged_summary: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    expected_total = 0
    for worker in range(worker_count):
        worker_dir = census_root / f"worker_{worker:03d}"
        summary_path, progress_path = (
            worker_dir / "summary.json",
            worker_dir / "progress.json",
        )
        if not summary_path.is_file() or not progress_path.is_file():
            raise RuntimeError(f"worker {worker} is not finalized")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if summary.get("schema") != SUMMARY_SCHEMA:
            raise RuntimeError(f"worker {worker} summary schema differs")
        if progress.get("schema") != PROGRESS_SCHEMA:
            raise RuntimeError(f"worker {worker} progress schema differs")
        for payload_name, payload in (("summary", summary), ("progress", progress)):
            if int(payload.get("worker_index", -1)) != worker:
                raise RuntimeError(f"worker {worker} {payload_name} index differs")
            if int(payload.get("worker_count", -1)) != worker_count:
                raise RuntimeError(f"worker {worker} {payload_name} worker-count differs")
            if payload.get("manifest_identity") != manifest_identity:
                raise RuntimeError(f"worker {worker} {payload_name} manifest identity differs")
            if payload.get("pilot_overlay", {}).get("sha256") != analysis_config_sha256:
                raise RuntimeError(f"worker {worker} {payload_name} analysis-config binding differs")
        if not summary.get("complete") or not summary.get("full_partition_complete"):
            raise RuntimeError(f"worker {worker} did not finish its full partition")
        if not progress.get("finalized") or not progress.get("complete"):
            raise RuntimeError(f"worker {worker} progress was not atomically finalized")
        start, stop = partition_bounds(total_windows, worker, worker_count)
        expected_windows = stop - start
        if int(summary.get("processed_windows", -1)) != expected_windows:
            raise RuntimeError(f"worker {worker} processed-window count differs")
        journal = progress.get("shards", ())
        expected_batch = 0
        journal_windows = 0
        for index, record in enumerate(journal):
            if (
                int(record.get("index", -1)) != index
                or int(record.get("batch_start", -1)) != expected_batch
            ):
                raise RuntimeError(f"worker {worker} shard journal is not contiguous")
            expected_batch = int(record["batch_stop"])
            journal_windows += int(record["processed_windows"])
            records.append({**record, "worker_index": worker, "partition": [start, stop]})
        if expected_batch != int(progress.get("next_batch_index", -1)):
            raise RuntimeError(f"worker {worker} journal endpoint differs")
        if journal_windows != expected_windows:
            raise RuntimeError(f"worker {worker} shard rows do not cover its partition")
        journal_paths = [str(Path(record["path"]).resolve()) for record in journal]
        summary_paths = [
            str(Path(value).resolve()) for value in summary.get("exact_chunk_b_inventory", ())
        ]
        if journal_paths != summary_paths:
            raise RuntimeError(f"worker {worker} summary/journal shard inventories differ")
        expected_total += expected_windows
    if expected_total != total_windows:
        raise AssertionError("worker partitions do not add up to the manifest")

    merged_path = census_root / "summary.json"
    merged: dict[str, Any] | None = None
    if merged_path.is_file():
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        if merged.get("schema") != MERGED_SCHEMA or not merged.get("complete"):
            raise RuntimeError("merged census summary is not complete")
        if merged.get("manifest_identity") != manifest_identity:
            raise RuntimeError("merged census manifest identity differs")
        if int(merged.get("processed_windows", -1)) != total_windows:
            raise RuntimeError("merged census window count differs")
        if merged.get("pilot_overlay", {}).get("sha256") != analysis_config_sha256:
            raise RuntimeError("merged census analysis-config binding differs")
        expected_paths = [str(Path(record["path"]).resolve()) for record in records]
        observed_paths = [str(Path(value).resolve()) for value in merged["exact_chunk_b_shards"]]
        if observed_paths != expected_paths:
            raise RuntimeError("merged exact-chunk inventory differs from worker journals")
    elif require_merged_summary:
        raise RuntimeError(
            "merged census summary is absent; merge workers first or explicitly allow "
            "complete unmerged workers"
        )
    return records, merged


def _threshold_monotonicity(
    bundles: Mapping[int, BThresholdBundle]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    monotone = True
    # Higher Wiki recall means a lower/more permissive lower-bound threshold.
    for lower, higher in ((95, 97), (97, 99), (95, 99)):
        for scale in SCALES:
            passed = bool(
                np.all(
                    bundles[higher].thresholds[scale]
                    <= bundles[lower].thresholds[scale]
                )
            )
            checks[f"B{higher}_no_stricter_than_B{lower}_scale{scale}"] = passed
            monotone &= passed
    checks["all_expected_monotone"] = monotone
    return checks


def extract_b_candidate_windows(
    *,
    census_root: str | Path,
    manifest_path: str | Path,
    analysis_config_path: str | Path,
    output_dir: str | Path,
    worker_count: int = 4,
    block_windows: int = 2048,
    verify_shard_sha256: bool = True,
    require_merged_summary: bool = True,
) -> dict[str, Any]:
    """Validate the full census and emit the conservative candidate window set."""

    census_root, output_dir = Path(census_root), Path(output_dir)
    targets = [
        output_dir / "candidate_windows.npy",
        output_dir / "manifest.json",
        output_dir / "REPORT.md",
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite candidate extraction artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    source_manifest, windows = load_full_train_manifest(manifest_path)
    total_windows = int(len(windows))
    config_identity, bundles = load_frozen_b_thresholds(analysis_config_path)
    analysis_config = json.loads(Path(analysis_config_path).read_text(encoding="utf-8"))
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
        total_windows=total_windows,
        require_merged_summary=bool(require_merged_summary),
    )

    seen = np.zeros(total_windows, dtype=np.bool_)
    union_mask = np.zeros(total_windows, dtype=np.bool_)
    per_level_masks = {
        level: np.zeros(total_windows, dtype=np.bool_) for level in bundles
    }
    per_level_token_counts = {level: 0 for level in bundles}
    union_token_count = 0
    bytes_read = 0
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"missing/size-changed shard: {path}")
        if verify_shard_sha256 and file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"shard sha256 differs: {path}")
        bytes_read += int(path.stat().st_size)
        with np.load(path, allow_pickle=False) as shard:
            required = {
                "metadata_json",
                "window_sample_order",
                "raw_b_cka",
                "chunk_layout_scale_start",
            }
            if not required.issubset(shard.files):
                raise RuntimeError(f"shard is missing exact-B arrays: {path}")
            metadata = json.loads(
                np.asarray(shard["metadata_json"], dtype=np.uint8)
                .tobytes()
                .decode("utf-8")
            )
            if metadata.get("schema") != SHARD_SCHEMA:
                raise RuntimeError(f"shard metadata schema differs: {path}")
            if int(metadata.get("worker_index", -1)) != int(record["worker_index"]):
                raise RuntimeError(f"shard worker identity differs: {path}")
            for key in ("index", "batch_start", "batch_stop", "processed_windows"):
                if int(metadata.get(key, -1)) != int(record[key]):
                    raise RuntimeError(f"shard metadata/journal {key} differs: {path}")
            if metadata.get("manifest_identity") != source_manifest["manifest_content_sha256"]:
                raise RuntimeError(f"shard manifest identity differs: {path}")
            sample_order = np.asarray(shard["window_sample_order"])
            raw_b = np.asarray(shard["raw_b_cka"])
            layout = np.asarray(shard["chunk_layout_scale_start"])
            if sample_order.dtype != np.int64 or sample_order.ndim != 1:
                raise RuntimeError(f"sample_order dtype/shape differs: {path}")
            if not np.array_equal(layout, CHUNK_LAYOUT_ARRAY):
                raise RuntimeError(f"fixed chunk layout differs: {path}")
            if raw_b.dtype != np.float32 or raw_b.shape != (
                sample_order.size,
                len(CHUNK_LAYOUT_ARRAY),
                len(LAYERS),
            ):
                raise RuntimeError(f"raw_b_cka dtype/shape differs: {path}")
            if sample_order.size != int(record["processed_windows"]):
                raise RuntimeError(f"shard journal/sample row count differs: {path}")
            start, stop = record["partition"]
            if (
                np.any(sample_order < int(start))
                or np.any(sample_order >= int(stop))
                or np.unique(sample_order).size != sample_order.size
            ):
                raise RuntimeError(f"shard sample_order is duplicate/outside partition: {path}")
            if seen[sample_order].any():
                raise RuntimeError(f"sample_order appears in more than one shard: {path}")
            aligned_rows = np.asarray(windows[sample_order])
            if not np.array_equal(aligned_rows["sample_order"], sample_order):
                raise RuntimeError(f"shard rows do not align to the manifest: {path}")
            lengths = aligned_rows["window_length"].astype(np.int32)
            for slot, (scale, chunk_start) in enumerate(CHUNK_LAYOUT_ARRAY.tolist()):
                coordinate_valid = int(chunk_start) + int(scale) <= lengths
                if np.isfinite(raw_b[~coordinate_valid, slot]).any():
                    raise RuntimeError(f"invalid tail chunk contains finite B values: {path}")
            result = candidate_masks_from_raw_b(
                raw_b,
                aligned_rows,
                bundles,
                block_windows=int(block_windows),
            )
            seen[sample_order] = True
            union_mask[sample_order] = result.union_window_mask
            for level in bundles:
                per_level_masks[level][sample_order] = result.window_masks[level]
                per_level_token_counts[level] += result.token_counts[level]
            union_token_count += result.union_token_count

    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise RuntimeError(
            f"full sample_order coverage failed: {missing.size} missing; "
            f"first={missing[:10].tolist()}"
        )
    subset_checks = {
        f"B{level}_windows_subset_candidate_union": bool(
            np.all(~per_level_masks[level] | union_mask)
        )
        for level in bundles
    }
    if not all(subset_checks.values()):
        raise AssertionError("candidate union omitted a frozen B bundle")
    monotonicity = _threshold_monotonicity(bundles)
    exact_b99 = per_level_masks[99]
    union_equals_b99 = bool(np.array_equal(union_mask, exact_b99))
    if monotonicity["all_expected_monotone"] and not union_equals_b99:
        raise AssertionError("monotone frozen thresholds did not yield a B99 superset")

    candidates = np.asarray(windows[union_mask]).copy()
    if candidates.dtype != WINDOW_DTYPE:
        raise AssertionError("candidate window dtype changed")
    if not np.all(np.diff(candidates["sample_order"]) > 0):
        raise AssertionError("candidate sample_order is not strictly increasing")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_windows.npy"
    _atomic_npy(candidate_path, candidates)

    total_eligible = int(source_manifest["statistics"]["eligible_token_count"])
    counts = {
        str(level): {
            "candidate_windows": int(per_level_masks[level].sum(dtype=np.int64)),
            "candidate_window_fraction": float(per_level_masks[level].mean()),
            "B_passing_tokens": int(per_level_token_counts[level]),
            "B_passing_token_fraction_of_full_eligible": float(
                per_level_token_counts[level] / max(1, total_eligible)
            ),
        }
        for level in bundles
    }
    report_payload = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "role": "B-only conservative prefilter for targeted GPU T/M/GT recomputation",
        "not_final_gt": True,
        "candidate_rule": (
            "any eligible token passes condition-specific B consensus at scale128 "
            "AND scale256 for any frozen bundle; per scale >=7/8, except 6 valid "
            "requires 6/6"
        ),
        "source_window_count": total_windows,
        "source_eligible_token_count": total_eligible,
        "candidate_window_count": int(candidates.size),
        "candidate_window_fraction": float(candidates.size / max(1, total_windows)),
        "candidate_window_retained_token_upper_bound": int(
            candidates["window_length"].sum(dtype=np.int64)
        ),
        "candidate_window_eligible_token_upper_bound": int(
            candidates["eligible_token_count"].sum(dtype=np.int64)
        ),
        "candidate_union_B_passing_token_count": int(union_token_count),
        "per_bundle": counts,
        "threshold_monotonicity": monotonicity,
        "subset_checks": subset_checks,
        "candidate_union_equals_exact_B99_windows": union_equals_b99,
        "extra_union_windows_beyond_exact_B99": int(
            np.count_nonzero(union_mask & ~exact_b99)
        ),
        "sample_order_coverage": {
            "expected": total_windows,
            "seen_once": int(seen.sum(dtype=np.int64)),
            "missing": int((~seen).sum(dtype=np.int64)),
            "duplicates": 0,
        },
        "worker_count": int(worker_count),
        "shard_count": len(records),
        "validated_shard_bytes": bytes_read,
        "shard_sha256_recomputed": bool(verify_shard_sha256),
        "merged_summary_required": bool(require_merged_summary),
        "merged_summary_present": merged is not None,
        "census_worker_model_bindings_validated": True,
        "census_worker_model_binding_set_sha256": model_bindings[
            "binding_set_content_sha256"
        ],
    }
    markdown = [
        "# Full-census B candidate window extraction",
        "",
        "This is a **B-only conservative prefilter**, not final old-like GT. T, rel-L2,",
        "and |log-r| must be recomputed in the targeted GPU pass.",
        "",
        f"- Full window coverage: {total_windows:,}/{total_windows:,} exactly once",
        f"- Candidate union: {candidates.size:,} windows ({candidates.size / max(1, total_windows):.6%})",
        f"- Eligible-token upper bound in candidate windows: {report_payload['candidate_window_eligible_token_upper_bound']:,}",
        f"- Union equals exact B99 window set: {union_equals_b99}",
        f"- Recomputed shard SHA-256: {bool(verify_shard_sha256)}",
        f"- Census worker model bindings: {model_bindings['worker_count']}/{worker_count} validated and hash-bound",
        "",
        "| Bundle | Candidate windows | Window fraction | B-passing tokens |",
        "|---:|---:|---:|---:|",
    ]
    for level in bundles:
        item = counts[str(level)]
        markdown.append(
            f"| B{level} | {item['candidate_windows']:,} | "
            f"{item['candidate_window_fraction']:.6%} | {item['B_passing_tokens']:,} |"
        )
    markdown.extend(
        [
            "",
            "The emitted window union contains every per-bundle candidate by construction.",
            "With the frozen monotone 95/97/99 thresholds it is identical to exact B99.",
            "",
        ]
    )
    report_path = output_dir / "REPORT.md"
    _atomic_text(report_path, "\n".join(markdown))

    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "not_final_gt": True,
        "source_manifest": {
            "path": str(Path(manifest_path).resolve()),
            "content_identity": source_manifest["manifest_content_sha256"],
        },
        "census_root": str(census_root.resolve()),
        "merged_census_summary": (
            _file_identity(census_root / "summary.json") if merged is not None else None
        ),
        "analysis_config": config_identity,
        "census_worker_model_bindings": model_bindings,
        "frozen_B_thresholds": {
            str(level): bundle.serializable() for level, bundle in bundles.items()
        },
        "candidate_windows": _file_identity(candidate_path),
        "report": _file_identity(report_path),
        "statistics": report_payload,
    }
    content = dict(manifest)
    manifest["manifest_content_sha256"] = hashlib.sha256(
        _canonical_json(content)
    ).hexdigest()
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--block-windows", type=int, default=2048)
    parser.add_argument(
        "--skip-shard-sha256",
        action="store_true",
        help="Trust journal hashes instead of recomputing them (not recommended).",
    )
    parser.add_argument(
        "--allow-unmerged-complete-workers",
        action="store_true",
        help="Allow extraction before the merged summary exists; workers must still be complete.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = extract_b_candidate_windows(
        census_root=args.census_root,
        manifest_path=args.manifest,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
        worker_count=args.worker_count,
        block_windows=args.block_windows,
        verify_shard_sha256=not args.skip_shard_sha256,
        require_merged_summary=not args.allow_unmerged_complete_workers,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BThresholdBundle",
    "ShardCandidateResult",
    "candidate_masks_from_raw_b",
    "extract_b_candidate_windows",
    "load_frozen_b_thresholds",
    "validate_census_model_bindings",
]
