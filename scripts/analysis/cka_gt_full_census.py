#!/usr/bin/env python3
"""Threshold-free, streaming full-train CKA B/T/M census.

The paired-forward driver supplies aligned residual-included layer outputs for
layers 2--9.  This module computes fp32 CKA-derived B/T scores and magnitude
scores, then immediately releases every reference to caller-owned hidden
states.  It deliberately creates no final old-like GT: the human chooses
thresholds after inspecting these distributions, and an exact targeted pass
then reruns only B-candidate windows.

Persisted artifacts are intentionally compact:

* exact per-window/per-chunk/layer B (CKA) values for scales 128 and 256;
* >=4096-bin exact marginal counts for raw B, token-min B/T, rel-L2, abs-log-r;
* a deterministic uniform token-score reservoir (5M by default after merge);
* window/document/token repetition summaries derived from that reservoir.

No hidden states, router/membership/null diagnostics, scale-512 metrics, wide
token Parquet, or threshold-selected GT occurrences are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from .cka_gt_pilot_core import (
        centered_linear_cka_b_t_metrics,
        uncentered_token_metrics,
    )
    from .cka_gt_pilot_windows import (
        DEFAULT_CODE_PREFIX,
        MMapIndexedDatasetLite,
        WINDOW_DTYPE,
        chunk_starts,
        document_lengths_from_indexed_dataset,
        enumerate_document_windows,
        file_sha256,
        validate_window_rows,
    )
except ImportError:  # Direct script execution.
    from cka_gt_pilot_core import (
        centered_linear_cka_b_t_metrics,
        uncentered_token_metrics,
    )
    from cka_gt_pilot_windows import (
        DEFAULT_CODE_PREFIX,
        MMapIndexedDatasetLite,
        WINDOW_DTYPE,
        chunk_starts,
        document_lengths_from_indexed_dataset,
        enumerate_document_windows,
        file_sha256,
        validate_window_rows,
    )


MANIFEST_SCHEMA = "cka_gt_full_train_windows_v1"
PROGRESS_SCHEMA = "cka_gt_full_census_worker_progress_v2"
SHARD_SCHEMA = "cka_gt_full_census_worker_shard_v2"
SUMMARY_SCHEMA = "cka_gt_full_census_worker_summary_v2"
MERGED_SCHEMA = "cka_gt_full_census_merged_summary_v2"
LAYERS = tuple(range(2, 10))
# Chunk scales are a measurement choice, not a fixed property of the census.
# CKA_CENSUS_SCALES="32" (or "32,64") overrides the historical 128/256 pair;
# stride is always half the scale, matching the original grid rule.
_SCALES_ENV = os.environ.get("CKA_CENSUS_SCALES", "128,256")
SCALES = tuple(int(v) for v in _SCALES_ENV.split(",") if v.strip())
if not SCALES or any(s <= 0 or 512 % s for s in SCALES):
    raise RuntimeError(f"CKA_CENSUS_SCALES must divide 512: {_SCALES_ENV!r}")
CHUNK_LAYOUT = tuple(
    (scale, start)
    for scale in SCALES
    for start in range(0, 512 - scale + 1, scale // 2)
)
CHUNK_LAYOUT_ARRAY = np.asarray(CHUNK_LAYOUT, dtype=np.int16)
KNOWN_CODE_COUNTS = {
    "window_count": 4_161_493,
    "retained_token_count": 2_083_238_288,
    "eligible_token_count": 2_058_949_248,
}
DEFAULT_ANALYSIS_CONFIG = Path(
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
    "cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1/analysis_config.json"
)
RESERVOIR_OVERSAMPLE_FACTOR = 1.25
UINT64_MODULUS = 1 << 64

# One candidate row is 224 bytes: 48 fp32 scores plus occurrence identity.
RESERVOIR_DTYPE = np.dtype(
    [
        ("priority", "<u8"),
        ("sample_order", "<i8"),
        ("source_window_index", "<i8"),
        ("document_id", "<i8"),
        ("window_offset", "<i8"),
        ("position", "<i2"),
        ("token_id", "<i4"),
        # One slot per configured chunk scale; a fixed (2, 8) would leave an
        # uninitialised second slot whenever a single scale is measured.
        ("b_min", "<f4", (len(SCALES), 8)),
        ("t_min", "<f4", (len(SCALES), 8)),
        ("rel_l2", "<f4", (8,)),
        ("abs_log_r", "<f4", (8,)),
    ],
    align=False,
)


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
            _jsonable(payload), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as handle:
            check = np.load(handle, allow_pickle=False)
        if check.dtype != value.dtype or check.shape != value.shape:
            raise RuntimeError("NPY round-trip validation failed")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, compress: bool) -> None:
    """Atomically write an NPZ; production shards are deliberately uncompressed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer = np.savez_compressed if compress else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as check:
            if set(check.files) != set(arrays):
                raise RuntimeError("NPZ key round-trip validation failed")
            for key, expected in arrays.items():
                if check[key].shape != expected.shape or check[key].dtype != expected.dtype:
                    raise RuntimeError(f"NPZ array round-trip failed for {key}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_array(payload: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(_canonical_json(payload), dtype=np.uint8).copy()


def _decode_json_array(value: np.ndarray) -> dict[str, Any]:
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode())


def _resolved_prefix(value: str | Path) -> str:
    text = str(value)
    if text.endswith(".idx") or text.endswith(".bin"):
        text = text.rsplit(".", 1)[0]
    return str(Path(text).resolve())


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _dataset_identity_light(prefix: str | Path) -> dict[str, Any]:
    resolved = _resolved_prefix(prefix)
    result: dict[str, Any] = {"resolved_prefix": resolved}
    for suffix in ("idx", "bin"):
        path = Path(f"{resolved}.{suffix}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[suffix] = {
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
        }
    return result


def build_full_train_manifest(
    dataset: Any,
    *,
    dataset_prefix: str,
    output_dir: str | Path,
    pilot_manifest: str | Path | None = None,
    enforce_known_code_counts: bool = False,
) -> dict[str, Any]:
    """Write exhaustive, document-bounded, unsplit train coordinates."""

    output_dir = Path(output_dir)
    manifest_path, windows_path = output_dir / "manifest.json", output_dir / "windows.npy"
    if manifest_path.exists() or windows_path.exists():
        if manifest_path.exists() and windows_path.exists():
            return validate_full_train_manifest(
                manifest_path, dataset=dataset, dataset_prefix=dataset_prefix,
                enforce_known_code_counts=enforce_known_code_counts
            )
        raise RuntimeError("partial full-train manifest exists")
    lengths = document_lengths_from_indexed_dataset(dataset)
    windows, statistics = enumerate_document_windows(np.arange(lengths.size), lengths)
    identity = np.arange(windows.size, dtype=np.int64)
    windows["sample_order"], windows["source_window_index"] = identity, identity
    validate_window_rows(windows, lengths)
    observed = {name: int(statistics[name]) for name in KNOWN_CODE_COUNTS}
    if enforce_known_code_counts and observed != KNOWN_CODE_COUNTS:
        raise RuntimeError(f"known Code counts differ: {observed} != {KNOWN_CODE_COUNTS}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npy(windows_path, windows)
    pilot_identity = None
    if pilot_manifest:
        pilot_path = Path(pilot_manifest)
        pilot = json.loads(pilot_path.read_text())
        if _resolved_prefix(pilot["dataset_prefix"]) != _resolved_prefix(dataset_prefix):
            raise ValueError("pilot/full manifests refer to different datasets")
        pilot_identity = _file_identity(pilot_path)
        pilot_identity["source_dataset_identity"] = pilot.get("source_dataset_identity")
    content = {
        "schema": MANIFEST_SCHEMA,
        "dataset_prefix": _resolved_prefix(dataset_prefix),
        "dataset_identity_light": _dataset_identity_light(dataset_prefix),
        "pilot_manifest_identity": pilot_identity,
        "document_count": int(lengths.size),
        "document_token_count": int(lengths.sum(dtype=np.int64)),
        "window_rule": {
            "length": 512, "stride": 512, "minimum_tail": 256,
            "right_aligned_tail": False, "scales": list(SCALES)
        },
        "statistics": statistics,
        "known_code_counts_enforced": bool(enforce_known_code_counts),
        "windows": _file_identity(windows_path),
    }
    content["manifest_content_sha256"] = hashlib.sha256(_canonical_json(content)).hexdigest()
    _atomic_json(manifest_path, content)
    return validate_full_train_manifest(
        manifest_path, dataset=dataset, dataset_prefix=dataset_prefix,
        enforce_known_code_counts=enforce_known_code_counts
    )


def validate_full_train_manifest(
    manifest_path: str | Path,
    *,
    dataset: Any | None = None,
    dataset_prefix: str | Path | None = None,
    enforce_known_code_counts: bool = False,
) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported full-train manifest schema")
    content = dict(payload)
    claimed = content.pop("manifest_content_sha256")
    if hashlib.sha256(_canonical_json(content)).hexdigest() != claimed:
        raise RuntimeError("manifest content hash mismatch")
    if dataset_prefix is not None:
        if _resolved_prefix(dataset_prefix) != payload["dataset_prefix"]:
            raise ValueError("dataset prefix differs from manifest")
        current = _dataset_identity_light(dataset_prefix)
        for suffix in ("idx", "bin"):
            for key in ("path", "size_bytes"):
                if current[suffix][key] != payload["dataset_identity_light"][suffix][key]:
                    raise RuntimeError(f"dataset {suffix} {key} changed")
    windows_path = Path(payload["windows"]["path"])
    if not windows_path.exists():
        windows_path = path.parent / "windows.npy"
    if windows_path.stat().st_size != payload["windows"]["size_bytes"]:
        raise RuntimeError("windows artifact size mismatch")
    if file_sha256(windows_path) != payload["windows"]["sha256"]:
        raise RuntimeError("windows artifact hash mismatch")
    windows = np.load(windows_path, mmap_mode="r", allow_pickle=False)
    if windows.dtype != WINDOW_DTYPE or windows.ndim != 1:
        raise RuntimeError("windows artifact dtype/shape mismatch")
    for start in range(0, windows.size, 1_000_000):
        stop = min(windows.size, start + 1_000_000)
        expected = np.arange(start, stop, dtype=np.int64)
        if not np.array_equal(windows["sample_order"][start:stop], expected):
            raise RuntimeError("sample_order is not exhaustive")
        if not np.array_equal(windows["source_window_index"][start:stop], expected):
            raise RuntimeError("source_window_index is not exhaustive")
    observed = {
        "window_count": int(windows.size),
        "retained_token_count": int(windows["window_length"].sum(dtype=np.int64)),
        "eligible_token_count": int(windows["eligible_token_count"].sum(dtype=np.int64)),
    }
    if any(observed[name] != int(payload["statistics"][name]) for name in observed):
        raise RuntimeError("manifest aggregate counts differ from windows")
    if enforce_known_code_counts and observed != KNOWN_CODE_COUNTS:
        raise RuntimeError(f"known Code counts differ: {observed} != {KNOWN_CODE_COUNTS}")
    if dataset is not None:
        lengths = document_lengths_from_indexed_dataset(dataset)
        validate_window_rows(np.asarray(windows), lengths)
    result = dict(payload)
    result["resolved_windows_path"] = str(windows_path.resolve())
    result["validated_counts"] = observed
    return result


def load_full_train_manifest(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = validate_full_train_manifest(path)
    return payload, np.load(payload["resolved_windows_path"], mmap_mode="r", allow_pickle=False)


def partition_bounds(total: int, index: int, count: int) -> tuple[int, int]:
    total, index, count = int(total), int(index), int(count)
    if total < 0 or count <= 0 or not 0 <= index < count:
        raise ValueError("invalid worker partition")
    quotient, remainder = divmod(total, count)
    start = index * quotient + min(index, remainder)
    return start, start + quotient + int(index < remainder)


def partition_manifest_rows(rows: np.ndarray, index: int, count: int) -> np.ndarray:
    start, stop = partition_bounds(len(rows), index, count)
    return rows[start:stop]


def manifest_batch_plan(rows: np.ndarray, batch_size: int) -> tuple[np.ndarray, ...]:
    rows, batch_size = np.asarray(rows), int(batch_size)
    if rows.dtype != WINDOW_DTYPE or rows.ndim != 1 or batch_size <= 0:
        raise ValueError("batch plan requires 1-D WINDOW_DTYPE and batch_size > 0")
    plan: list[np.ndarray] = []
    for length in np.unique(rows["window_length"])[::-1].tolist():
        indices = np.flatnonzero(rows["window_length"] == length)
        indices = indices[np.argsort(rows["sample_order"][indices], kind="stable")]
        plan.extend(indices[start : start + batch_size] for start in range(0, indices.size, batch_size))
    return tuple(np.asarray(value, dtype=np.int64) for value in plan)


@dataclass
class BTMBatch:
    raw_b: torch.Tensor  # [window, fixed chunk slot, layer]
    b_min: dict[int, torch.Tensor]  # [window, token, layer]
    t_min: dict[int, torch.Tensor]
    rel_l2: torch.Tensor
    abs_log_r: torch.Tensor
    eligible: torch.Tensor


def _nanmin_update(target: torch.Tensor, candidate: torch.Tensor) -> None:
    replace = torch.isfinite(candidate) & (~torch.isfinite(target) | (candidate < target))
    target.copy_(torch.where(replace, candidate, target))


@torch.no_grad()
def _compute_btm_batch_impl(
    *, before_by_layer: Mapping[int, torch.Tensor],
    after_by_layer: Mapping[int, torch.Tensor], manifest_rows: np.ndarray,
) -> BTMBatch:
    """Compute raw B plus token-min B/T and M; return no hidden references."""

    if tuple(sorted(before_by_layer)) != LAYERS or tuple(sorted(after_by_layer)) != LAYERS:
        raise ValueError(f"census requires exactly residual layers {LAYERS}")
    first = before_by_layer[2]
    if first.ndim != 3:
        raise ValueError("hidden tensors must be [batch,tokens,hidden]")
    batch, length, hidden = first.shape
    rows = np.asarray(manifest_rows)
    if rows.dtype != WINDOW_DTYPE or rows.shape != (batch,) or np.any(rows["window_length"] != length):
        raise ValueError("manifest rows do not align with unpadded hidden batch")
    for layer in LAYERS:
        if before_by_layer[layer].shape != (batch, length, hidden):
            raise ValueError(f"before layer {layer} shape differs")
        if after_by_layer[layer].shape != (batch, length, hidden):
            raise ValueError(f"after layer {layer} shape differs")
        if before_by_layer[layer].device != first.device or after_by_layer[layer].device != first.device:
            raise ValueError("all hidden tensors must share one metric device")
    device = first.device
    token_shape = (batch, length, len(LAYERS))
    raw_b = torch.full(
        (batch, len(CHUNK_LAYOUT), len(LAYERS)), float("nan"),
        device=device, dtype=torch.float32
    )
    b_min = {scale: torch.full(token_shape, float("nan"), device=device) for scale in SCALES}
    t_min = {scale: torch.full(token_shape, float("nan"), device=device) for scale in SCALES}
    rel_l2 = torch.empty(token_shape, device=device, dtype=torch.float32)
    abs_log_r = torch.empty_like(rel_l2)
    for layer_index, layer in enumerate(LAYERS):
        magnitude = uncentered_token_metrics(before_by_layer[layer], after_by_layer[layer])
        rel_l2[..., layer_index] = magnitude["rel_l2"]
        abs_log_r[..., layer_index] = magnitude["log_r"].abs()
    slot = 0
    for scale in SCALES:
        for start_value in chunk_starts(length, scale, scale // 2).tolist():
            start, stop = int(start_value), int(start_value) + scale
            expected_slot = CHUNK_LAYOUT.index((scale, start))
            if expected_slot < slot:
                raise AssertionError("chunk layout order regressed")
            slot = expected_slot
            for layer_index, layer in enumerate(LAYERS):
                values = centered_linear_cka_b_t_metrics(
                    before_by_layer[layer][:, start:stop], after_by_layer[layer][:, start:stop]
                )
                raw_b[:, slot, layer_index] = values["cka"]
                _nanmin_update(
                    b_min[scale][:, start:stop, layer_index],
                    values["cka"][:, None].expand(-1, scale),
                )
                _nanmin_update(t_min[scale][:, start:stop, layer_index], values["s_i"])
    positions = torch.arange(length, device=device)[None, :]
    eligible_counts = torch.as_tensor(rows["eligible_token_count"].astype(np.int64), device=device)
    eligible = positions < eligible_counts[:, None]
    for value in (*b_min.values(), *t_min.values(), rel_l2, abs_log_r):
        value.masked_fill_((~eligible).unsqueeze(-1), float("nan"))
    return BTMBatch(raw_b, b_min, t_min, rel_l2, abs_log_r, eligible)


@torch.no_grad()
def compute_btm_batch(
    *, before_by_layer: Mapping[int, torch.Tensor],
    after_by_layer: Mapping[int, torch.Tensor], manifest_rows: np.ndarray,
) -> BTMBatch:
    """Strict-fp32 B/T/M baseline, restoring the caller's TF32 setting."""

    previous = bool(torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _compute_btm_batch_impl(
            before_by_layer=before_by_layer,
            after_by_layer=after_by_layer,
            manifest_rows=manifest_rows,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@dataclass(frozen=True)
class HistogramSpec:
    minimum: float
    maximum: float
    bins: int

    def edges(self) -> np.ndarray:
        return np.linspace(self.minimum, self.maximum, self.bins + 1)


def default_histogram_specs(bins: int) -> dict[str, HistogramSpec]:
    # CKA candidate cuts differ at roughly 1e-3 near one.  4096 bins over
    # [0,1] preserve ~2.4e-4 resolution while remaining only tiny counters.
    bins = max(4096, int(bins))
    specs: dict[str, HistogramSpec] = {}
    # One B/T histogram family per configured chunk scale.
    for scale in SCALES:
        specs[f"raw_b_{scale}"] = HistogramSpec(0.0, 1.0, bins)
        specs[f"token_min_b_{scale}"] = HistogramSpec(0.0, 1.0, bins)
        specs[f"token_min_t_{scale}"] = HistogramSpec(-10.0, 10.0, bins)
    specs["relative_l2"] = HistogramSpec(0.0, 4.0, bins)
    specs["abs_log_r"] = HistogramSpec(0.0, 2.0, bins)
    return specs


class HistogramAccumulator:
    """GPU-binned exact counts with under/overflow/nonfinite per layer."""

    def __init__(self, specs: Mapping[str, HistogramSpec]) -> None:
        self.specs = dict(specs)
        self.counts = {key: np.zeros((8, value.bins), np.int64) for key, value in specs.items()}
        self.underflow = {key: np.zeros(8, np.int64) for key in specs}
        self.overflow = {key: np.zeros(8, np.int64) for key in specs}
        self.nonfinite = {key: np.zeros(8, np.int64) for key in specs}

    def observe(self, key: str, values: torch.Tensor, eligible: torch.Tensor) -> None:
        if values.shape[-1] != 8 or eligible.shape != values.shape[:-1]:
            raise ValueError(f"histogram {key} value/eligibility shape mismatch")
        spec = self.specs[key]
        eligible3, finite = eligible.unsqueeze(-1), torch.isfinite(values)
        under = eligible3 & finite & (values < spec.minimum)
        over = eligible3 & finite & (values > spec.maximum)
        inside = eligible3 & finite & (values >= spec.minimum) & (values <= spec.maximum)
        reduce_dims = tuple(range(values.ndim - 1))
        self.underflow[key] += under.sum(dim=reduce_dims).cpu().numpy()
        self.overflow[key] += over.sum(dim=reduce_dims).cpu().numpy()
        self.nonfinite[key] += (eligible3 & ~finite).sum(dim=reduce_dims).cpu().numpy()
        index = torch.floor(
            (values - spec.minimum) * spec.bins / (spec.maximum - spec.minimum)
        ).to(torch.int64).clamp_(0, spec.bins - 1)
        layer = torch.arange(8, device=values.device).reshape(*([1] * (values.ndim - 1)), 8)
        combined = index + layer * spec.bins
        counts = torch.bincount(combined[inside], minlength=8 * spec.bins).reshape(8, spec.bins)
        self.counts[key] += counts.cpu().numpy()

    def arrays(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key, spec in self.specs.items():
            result[f"hist__{key}__counts"] = self.counts[key]
            result[f"hist__{key}__underflow"] = self.underflow[key]
            result[f"hist__{key}__overflow"] = self.overflow[key]
            result[f"hist__{key}__nonfinite"] = self.nonfinite[key]
            result[f"hist__{key}__edges"] = spec.edges()
        return result


def _splitmix64(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.uint64).copy()
    value += np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def token_priorities(rows: np.ndarray, length: int, seed: int) -> np.ndarray:
    sample = rows["sample_order"].astype(np.uint64)[:, None]
    position = np.arange(length, dtype=np.uint64)[None, :]
    # Every window is at most 512 tokens, so this packing is injective before
    # the bijective SplitMix64 permutation (unlike an ad-hoc multiply/xor key).
    identity = (sample << np.uint64(9)) | position
    identity ^= np.uint64(seed)
    return _splitmix64(identity)


def reservoir_candidates(
    rows: np.ndarray,
    token_ids: torch.Tensor,
    metrics: BTMBatch,
    *,
    priority_cutoff: int,
    seed: int,
) -> np.ndarray:
    if priority_cutoff <= 0:
        return np.empty(0, dtype=RESERVOIR_DTYPE)
    batch, length = token_ids.shape
    priorities = token_priorities(rows, length, seed)
    eligible = np.arange(length)[None, :] < rows["eligible_token_count"][:, None]
    if priority_cutoff >= UINT64_MODULUS - 1:
        # A cutoff at the modulus means "keep every eligible token".  The
        # comparison below would be a no-op anyway, and converting a full
        # 64-bit Python int through np.uint64 overflows C long.
        mask = eligible
    else:
        mask = eligible & (priorities <= np.uint64(priority_cutoff))
    batch_ids, positions = np.nonzero(mask)
    result = np.empty(batch_ids.size, dtype=RESERVOIR_DTYPE)
    if not batch_ids.size:
        return result
    selected_rows = rows[batch_ids]
    result["priority"] = priorities[batch_ids, positions]
    for key in ("sample_order", "source_window_index", "document_id", "window_offset"):
        result[key] = selected_rows[key]
    result["position"] = positions.astype(np.int16)
    device_batch = torch.as_tensor(batch_ids, device=token_ids.device)
    device_position = torch.as_tensor(positions, device=token_ids.device)
    result["token_id"] = token_ids[device_batch, device_position].detach().cpu().numpy()
    for scale_index, scale in enumerate(SCALES):
        result["b_min"][:, scale_index] = (
            metrics.b_min[scale][device_batch, device_position].detach().cpu().numpy()
        )
        result["t_min"][:, scale_index] = (
            metrics.t_min[scale][device_batch, device_position].detach().cpu().numpy()
        )
    result["rel_l2"] = metrics.rel_l2[device_batch, device_position].detach().cpu().numpy()
    result["abs_log_r"] = metrics.abs_log_r[device_batch, device_position].detach().cpu().numpy()
    return result


class _PendingShard:
    def __init__(self, specs: Mapping[str, HistogramSpec], start_batch: int) -> None:
        self.start_batch = self.next_batch = int(start_batch)
        self.hist = HistogramAccumulator(specs)
        self.window_ids: list[np.ndarray] = []
        self.raw_b: list[np.ndarray] = []
        self.reservoir: list[np.ndarray] = []
        self.windows = self.retained = self.eligible = 0
        self.forward_seconds = self.metric_seconds = 0.0

    @property
    def batch_count(self) -> int:
        return self.next_batch - self.start_batch


class FullCensusWorker:
    """Atomic, resumable worker over one contiguous full-train partition."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        worker_index: int,
        worker_count: int,
        analysis_config_path: str | Path,
        manifest_path: str | Path,
        batch_size: int,
        histogram_bins: int = 4096,
        checkpoint_every_batches: int = 100,
        reservoir_size: int = 5_000_000,
        reservoir_seed: int = 1234,
        max_windows: int | None = None,
        full_length_only: bool = False,
    ) -> None:
        self.output_dir, self.worker_index, self.worker_count = Path(output_dir), int(worker_index), int(worker_count)
        self.batch_size, self.histogram_bins = int(batch_size), max(4096, int(histogram_bins))
        self.checkpoint_every_batches = int(checkpoint_every_batches)
        self.reservoir_size, self.reservoir_seed = int(reservoir_size), int(reservoir_seed)
        self.max_windows, self.full_length_only = max_windows, bool(full_length_only)
        if self.batch_size <= 0 or self.checkpoint_every_batches <= 0 or self.reservoir_size < 0:
            raise ValueError("invalid worker batch/checkpoint/reservoir configuration")
        manifest, all_rows = load_full_train_manifest(manifest_path)
        self.manifest, self.manifest_path = manifest, str(Path(manifest_path).resolve())
        self.full_partition_rows = partition_manifest_rows(all_rows, self.worker_index, self.worker_count)
        rows = self.full_partition_rows
        if self.full_length_only:
            rows = rows[rows["window_length"] == 512]
        if max_windows is not None:
            if int(max_windows) <= 0:
                raise ValueError("max_windows must be positive")
            rows = rows[: int(max_windows)]
        self.manifest_rows = rows
        self.batch_plan = manifest_batch_plan(rows, self.batch_size)
        config_path = Path(analysis_config_path)
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        self.pilot_overlay = {
            "path": str(config_path.resolve()),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "candidate_thresholds": config.get("candidate_thresholds", {}),
            "role": "diagnostic histogram overlay only; never applied as GT in this census",
        }
        total_eligible = int(manifest["statistics"]["eligible_token_count"])
        expected_candidates = self.reservoir_size * RESERVOIR_OVERSAMPLE_FACTOR
        probability = min(1.0, expected_candidates / max(1, total_eligible))
        self.priority_cutoff = int(math.floor(probability * (UINT64_MODULUS - 1)))
        self.specs = default_histogram_specs(self.histogram_bins)
        self.worker_dir = self.output_dir / f"worker_{self.worker_index:03d}"
        self.shard_dir = self.worker_dir / "shards"
        self.progress_path, self.summary_path = self.worker_dir / "progress.json", self.worker_dir / "summary.json"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.progress = self._load_progress()
        self._next_batch = int(self.progress["next_batch_index"])
        self.pending = _PendingShard(self.specs, self._next_batch)

    @property
    def next_batch_index(self) -> int:
        return self._next_batch

    @property
    def expected_batch_count(self) -> int:
        return len(self.batch_plan)

    def _identity(self) -> dict[str, Any]:
        # Pilot overlay thresholds are deliberately absent: they are not a
        # selector or resume invariant in this threshold-free run.
        return {
            "manifest_identity": self.manifest["manifest_content_sha256"],
            "worker_index": self.worker_index, "worker_count": self.worker_count,
            "batch_size": self.batch_size, "histogram_bins": self.histogram_bins,
            "reservoir_size": self.reservoir_size, "reservoir_seed": self.reservoir_seed,
            "priority_cutoff": self.priority_cutoff,
            "max_windows": self.max_windows, "full_length_only": self.full_length_only,
        }

    def _load_progress(self) -> dict[str, Any]:
        if not self.progress_path.exists():
            payload = {
                "schema": PROGRESS_SCHEMA, **self._identity(),
                "manifest_path": self.manifest_path, "pilot_overlay": self.pilot_overlay,
                "expected_windows": int(len(self.manifest_rows)),
                "expected_batches": int(len(self.batch_plan)), "next_batch_index": 0,
                "processed_windows": 0, "processed_retained_tokens": 0,
                "processed_eligible_tokens": 0, "forward_seconds": 0.0,
                "metric_seconds": 0.0, "shards": [], "finalized": False,
            }
            _atomic_json(self.progress_path, payload)
            return payload
        payload = json.loads(self.progress_path.read_text())
        if payload.get("schema") != PROGRESS_SCHEMA:
            raise RuntimeError("unsupported threshold-free progress schema")
        for key, value in self._identity().items():
            if payload.get(key) != value:
                raise RuntimeError(f"resume identity mismatch for {key}")
        expected_batch = 0
        for index, record in enumerate(payload["shards"]):
            if record["index"] != index or record["batch_start"] != expected_batch:
                raise RuntimeError("non-contiguous shard journal")
            path = Path(record["path"])
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"missing/corrupt census shard: {path}")
            expected_batch = record["batch_stop"]
        if expected_batch != payload["next_batch_index"]:
            raise RuntimeError("journal next batch differs from shards")
        return payload

    def process_batch(
        self,
        *,
        batch_index: int,
        rows: np.ndarray,
        token_ids: torch.Tensor,
        before_by_layer: Mapping[int, torch.Tensor],
        after_by_layer: Mapping[int, torch.Tensor],
        forward_seconds: float = 0.0,
    ) -> dict[str, Any]:
        batch_index, rows = int(batch_index), np.asarray(rows)
        if self.progress.get("finalized"):
            raise RuntimeError("cannot append after finalize")
        if batch_index != self._next_batch:
            raise RuntimeError(f"expected batch {self._next_batch}, got {batch_index}")
        if not math.isfinite(forward_seconds) or forward_seconds < 0:
            raise ValueError("forward_seconds must be finite and non-negative")
        expected = self.manifest_rows[self.batch_plan[batch_index]]
        if rows.dtype != WINDOW_DTYPE or not np.array_equal(rows["sample_order"], expected["sample_order"]):
            raise RuntimeError("batch rows differ from deterministic plan")
        started = time.perf_counter()
        metrics = compute_btm_batch(
            before_by_layer=before_by_layer, after_by_layer=after_by_layer,
            manifest_rows=rows
        )
        return self.process_precomputed_batch(
            batch_index=batch_index, rows=rows, token_ids=token_ids,
            metrics=metrics, forward_seconds=forward_seconds,
            metric_started=started,
        )

    def process_precomputed_batch(
        self,
        *,
        batch_index: int,
        rows: np.ndarray,
        token_ids: torch.Tensor,
        metrics: BTMBatch,
        forward_seconds: float = 0.0,
        metric_started: float | None = None,
        metric_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Commit an already-computed logical batch (possibly sub-forwarded)."""
        batch_index, rows = int(batch_index), np.asarray(rows)
        if self.progress.get("finalized"):
            raise RuntimeError("cannot append after finalize")
        if batch_index != self._next_batch:
            raise RuntimeError(f"expected batch {self._next_batch}, got {batch_index}")
        if not math.isfinite(forward_seconds) or forward_seconds < 0:
            raise ValueError("forward_seconds must be finite and non-negative")
        expected = self.manifest_rows[self.batch_plan[batch_index]]
        if rows.dtype != WINDOW_DTYPE or not np.array_equal(
            rows["sample_order"], expected["sample_order"]
        ):
            raise RuntimeError("batch rows differ from deterministic plan")
        started = time.perf_counter() if metric_started is None else metric_started
        raw_scales = torch.as_tensor(CHUNK_LAYOUT_ARRAY[:, 0], device=metrics.raw_b.device)
        for scale in SCALES:
            slot_mask = raw_scales == scale
            raw = metrics.raw_b[:, slot_mask]
            scale_starts = torch.as_tensor(
                CHUNK_LAYOUT_ARRAY[CHUNK_LAYOUT_ARRAY[:, 0] == scale, 1],
                device=metrics.raw_b.device,
            )
            window_lengths = torch.as_tensor(
                rows["window_length"].astype(np.int64), device=metrics.raw_b.device
            )
            # Coordinate eligibility is independent of metric validity, so a
            # completely degenerate chunk contributes eight nonfinite counts.
            raw_eligible = (
                scale_starts.unsqueeze(0) + int(scale)
                <= window_lengths.unsqueeze(1)
            )
            self.pending.hist.observe(f"raw_b_{scale}", raw, raw_eligible)
            self.pending.hist.observe(f"token_min_b_{scale}", metrics.b_min[scale], metrics.eligible)
            self.pending.hist.observe(f"token_min_t_{scale}", metrics.t_min[scale], metrics.eligible)
        self.pending.hist.observe("relative_l2", metrics.rel_l2, metrics.eligible)
        self.pending.hist.observe("abs_log_r", metrics.abs_log_r, metrics.eligible)
        candidates = reservoir_candidates(
            rows, token_ids, metrics, priority_cutoff=self.priority_cutoff,
            seed=self.reservoir_seed
        )
        self.pending.window_ids.append(rows["sample_order"].astype(np.int64, copy=True))
        self.pending.raw_b.append(metrics.raw_b.detach().cpu().numpy().astype(np.float32))
        if candidates.size:
            self.pending.reservoir.append(candidates)
        self.pending.windows += len(rows)
        self.pending.retained += int(rows["window_length"].sum(dtype=np.int64))
        self.pending.eligible += int(rows["eligible_token_count"].sum(dtype=np.int64))
        self.pending.forward_seconds += float(forward_seconds)
        measured_metric_seconds = (
            time.perf_counter() - started
            if metric_seconds is None else float(metric_seconds)
        )
        if not math.isfinite(measured_metric_seconds) or measured_metric_seconds < 0:
            raise ValueError("metric_seconds must be finite and non-negative")
        self.pending.metric_seconds += measured_metric_seconds
        self.pending.next_batch = self._next_batch = batch_index + 1
        result = {
            "batch_index": batch_index, "windows": len(rows),
            "eligible_tokens": int(rows["eligible_token_count"].sum()),
            "reservoir_candidates": int(candidates.size),
        }
        if self.pending.batch_count >= self.checkpoint_every_batches:
            self.checkpoint()
        return result

    def checkpoint(self) -> dict[str, Any]:
        if self.pending.batch_count == 0:
            return self.progress
        index = len(self.progress["shards"])
        metadata = {
            "schema": SHARD_SCHEMA, **self._identity(), "index": index,
            "batch_start": self.pending.start_batch, "batch_stop": self.pending.next_batch,
            "processed_windows": self.pending.windows,
            "processed_retained_tokens": self.pending.retained,
            "processed_eligible_tokens": self.pending.eligible,
            "forward_seconds": self.pending.forward_seconds,
            "metric_seconds": self.pending.metric_seconds,
        }
        arrays = self.pending.hist.arrays()
        arrays["window_sample_order"] = np.concatenate(self.pending.window_ids)
        arrays["raw_b_cka"] = np.concatenate(self.pending.raw_b)
        arrays["chunk_layout_scale_start"] = CHUNK_LAYOUT_ARRAY
        arrays["reservoir_candidates"] = (
            np.concatenate(self.pending.reservoir)
            if self.pending.reservoir else np.empty(0, dtype=RESERVOIR_DTYPE)
        )
        arrays["metadata_json"] = _json_array(metadata)
        path = self.shard_dir / (
            f"shard_{index:06d}_b{self.pending.start_batch:07d}_{self.pending.next_batch:07d}.npz"
        )
        _atomic_npz(path, arrays, compress=False)
        record = {
            "index": index, "batch_start": self.pending.start_batch,
            "batch_stop": self.pending.next_batch, "path": str(path.resolve()),
            "size_bytes": path.stat().st_size, "sha256": file_sha256(path),
            "processed_windows": self.pending.windows,
            "processed_retained_tokens": self.pending.retained,
            "processed_eligible_tokens": self.pending.eligible,
            "reservoir_candidates": int(arrays["reservoir_candidates"].size),
            "forward_seconds": self.pending.forward_seconds,
            "metric_seconds": self.pending.metric_seconds,
        }
        updated = dict(self.progress)
        updated["shards"] = [*updated["shards"], record]
        updated["next_batch_index"] = self.pending.next_batch
        for key, delta in (
            ("processed_windows", self.pending.windows),
            ("processed_retained_tokens", self.pending.retained),
            ("processed_eligible_tokens", self.pending.eligible),
            ("forward_seconds", self.pending.forward_seconds),
            ("metric_seconds", self.pending.metric_seconds),
        ):
            updated[key] = updated[key] + delta
        _atomic_json(self.progress_path, updated)
        self.progress = updated
        self.pending = _PendingShard(self.specs, self._next_batch)
        return updated

    def finalize(self) -> dict[str, Any]:
        if self.summary_path.exists():
            return json.loads(self.summary_path.read_text())
        self.checkpoint()
        requested_complete = self.progress["next_batch_index"] == len(self.batch_plan)
        full_partition_complete = (
            requested_complete and not self.full_length_only and self.max_windows is None
            and self.progress["processed_windows"] == len(self.full_partition_rows)
        )
        summary = {
            "schema": SUMMARY_SCHEMA, **self._identity(), "complete": requested_complete,
            "full_partition_complete": full_partition_complete,
            "threshold_free": True, "final_gt_created": False,
            "exact_gt_requires_targeted_second_pass_after_threshold_lock": True,
            "pilot_overlay": self.pilot_overlay,
            "processed_windows": self.progress["processed_windows"],
            "processed_retained_tokens": self.progress["processed_retained_tokens"],
            "processed_eligible_tokens": self.progress["processed_eligible_tokens"],
            "next_batch_index": self.progress["next_batch_index"],
            "expected_batches": len(self.batch_plan), "shard_count": len(self.progress["shards"]),
            "reservoir_candidate_count": sum(x["reservoir_candidates"] for x in self.progress["shards"]),
            "forward_seconds": self.progress["forward_seconds"],
            "metric_seconds": self.progress["metric_seconds"],
            "active_seconds": self.progress["forward_seconds"] + self.progress["metric_seconds"],
            "exact_chunk_b_inventory": [x["path"] for x in self.progress["shards"]],
        }
        _atomic_json(self.summary_path, summary)
        updated = dict(self.progress)
        updated.update({"finalized": True, "complete": requested_complete,
                        "summary_path": str(self.summary_path.resolve())})
        _atomic_json(self.progress_path, updated)
        self.progress = updated
        return summary


def _merge_histograms(records: Sequence[Mapping[str, Any]], bins: int) -> dict[str, np.ndarray]:
    specs = default_histogram_specs(bins)
    result = HistogramAccumulator(specs).arrays()
    for key in list(result):
        if not key.endswith("__edges"):
            result[key].fill(0)
    for record in records:
        with np.load(record["path"], allow_pickle=False) as shard:
            for key in result:
                if not key.endswith("__edges"):
                    result[key] += shard[key]
    return result


def _write_histogram_report(
    output_dir: Path,
    histograms: Mapping[str, np.ndarray],
    pilot_overlay: Mapping[str, Any],
) -> dict[str, Any]:
    """Write dependency-free layer plots and quantiles after the census merge."""

    report_dir = output_dir / "histogram_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#2563eb", "#dc2626", "#16a34a")
    candidates = pilot_overlay.get("candidate_thresholds", {})
    csv_lines = ["metric,layer,count,p01,p05,p10,p25,p50,p75,p90,p95,p99"]
    artifacts: list[str] = []

    def thresholds(metric: str, layer_index: int) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        for level in ("95", "97", "99"):
            item = candidates.get(level, {})
            scale_suffix = metric.rsplit("_", 1)[-1]
            if metric.startswith(("raw_b_", "token_min_b_")):
                vector = item.get("B_lower_threshold", {}).get(scale_suffix)
            elif metric.startswith("token_min_t_"):
                vector = item.get("T_lower_threshold", {}).get(scale_suffix)
            elif metric == "relative_l2":
                vector = item.get("relative_l2_upper_threshold")
            elif metric == "abs_log_r":
                vector = item.get("abs_log_r_upper_threshold")
            else:
                vector = None
            if vector is not None:
                values.append((level, float(vector[layer_index])))
        return values

    for metric in default_histogram_specs(4096):
        counts = np.asarray(histograms[f"hist__{metric}__counts"], dtype=np.float64)
        edges = np.asarray(histograms[f"hist__{metric}__edges"], dtype=np.float64)
        centers = (edges[:-1] + edges[1:]) * 0.5
        for layer_index in range(8):
            row, total = counts[layer_index], counts[layer_index].sum()
            cdf = np.cumsum(row)
            quantiles = []
            for q in (.01, .05, .10, .25, .50, .75, .90, .95, .99):
                index = int(np.searchsorted(cdf, q * total, side="left")) if total else 0
                quantiles.append(float(centers[min(index, centers.size - 1)]))
            csv_lines.append(
                ",".join([metric, str(layer_index + 2), str(int(total))] + [f"{x:.9g}" for x in quantiles])
            )

        for log_y in (False, True):
            width, height, panel_w, panel_h = 1200, 900, 380, 270
            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="600" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{metric} ({"log" if log_y else "linear"} probability/bin)</text>',
            ]
            for layer_index in range(8):
                col, row_index = layer_index % 3, layer_index // 3
                x0, y0 = 45 + col * panel_w, 50 + row_index * panel_h
                plot_w, plot_h = 320, 205
                mass = counts[layer_index] / max(counts[layer_index].sum(), 1.0)
                values = np.log10(np.maximum(mass, 1e-12)) if log_y else mass
                low = -12.0 if log_y else 0.0
                high = float(values.max()) if values.size else 1.0
                high = max(high, low + 1e-12)
                step = max(1, int(math.ceil(values.size / 800)))
                points = []
                for index in range(0, values.size, step):
                    x = x0 + plot_w * index / max(values.size - 1, 1)
                    y = y0 + plot_h * (1.0 - (float(values[index]) - low) / (high - low))
                    points.append(f"{x:.2f},{y:.2f}")
                parts.extend([
                    f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#777"/>',
                    f'<polyline points="{" ".join(points)}" fill="none" stroke="#111827" stroke-width="1.2"/>',
                    f'<text x="{x0 + 4}" y="{y0 + 17}" font-family="sans-serif" font-size="14">Layer {layer_index + 2}</text>',
                    f'<text x="{x0}" y="{y0 + plot_h + 18}" font-family="sans-serif" font-size="11">{edges[0]:.3g}</text>',
                    f'<text x="{x0 + plot_w}" y="{y0 + plot_h + 18}" text-anchor="end" font-family="sans-serif" font-size="11">{edges[-1]:.3g}</text>',
                ])
                for threshold_index, (level, value) in enumerate(thresholds(metric, layer_index)):
                    if edges[0] <= value <= edges[-1]:
                        x = x0 + plot_w * (value - edges[0]) / (edges[-1] - edges[0])
                        parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + plot_h}" stroke="{colors[threshold_index]}" stroke-width="1" stroke-dasharray="4,3"><title>pilot {level}: {value:.8g}</title></line>')
            parts.append('</svg>')
            path = report_dir / f"{metric}_{'log' if log_y else 'linear'}.svg"
            _atomic_text(path, "\n".join(parts) + "\n")
            artifacts.append(str(path))
    quantile_path = report_dir / "layer_quantiles.csv"
    _atomic_text(quantile_path, "\n".join(csv_lines) + "\n")
    artifacts.append(str(quantile_path))
    return {"directory": str(report_dir), "artifacts": artifacts}


def merge_worker_outputs(output_dir: str | Path, worker_count: int) -> dict[str, Any]:
    """Validate full coverage and materialize exact 5M bottom-hash reservoir."""

    output_dir, worker_count = Path(output_dir), int(worker_count)
    summaries, records = [], []
    for worker in range(worker_count):
        directory = output_dir / f"worker_{worker:03d}"
        summary = json.loads((directory / "summary.json").read_text())
        progress = json.loads((directory / "progress.json").read_text())
        if not summary.get("full_partition_complete"):
            raise RuntimeError(f"worker {worker} did not complete its full partition")
        summaries.append(summary)
        records.extend(progress["shards"])
    identity = {summary["manifest_identity"] for summary in summaries}
    if len(identity) != 1:
        raise RuntimeError("worker manifest identities differ")
    total = {
        "window_count": sum(x["processed_windows"] for x in summaries),
        "retained_token_count": sum(x["processed_retained_tokens"] for x in summaries),
        "eligible_token_count": sum(x["processed_eligible_tokens"] for x in summaries),
    }
    if total != KNOWN_CODE_COUNTS:
        raise RuntimeError(f"merged coverage differs from known full Code counts: {total}")
    bins = summaries[0]["histogram_bins"]
    if any(x["histogram_bins"] != bins for x in summaries):
        raise RuntimeError("worker histogram specifications differ")
    histograms = _merge_histograms(records, bins)
    hist_path = output_dir / "histograms.npz"
    _atomic_npz(hist_path, histograms, compress=True)
    histogram_report = _write_histogram_report(
        output_dir, histograms, summaries[0]["pilot_overlay"]
    )
    candidate_parts = []
    for record in records:
        with np.load(record["path"], allow_pickle=False) as shard:
            if shard["reservoir_candidates"].size:
                candidate_parts.append(shard["reservoir_candidates"].copy())
    candidates = np.concatenate(candidate_parts) if candidate_parts else np.empty(0, RESERVOIR_DTYPE)
    target = summaries[0]["reservoir_size"]
    if any(x["reservoir_size"] != target or x["reservoir_seed"] != summaries[0]["reservoir_seed"] for x in summaries):
        raise RuntimeError("worker reservoir configurations differ")
    if candidates.size < target:
        raise RuntimeError(
            f"hash candidate stream ({candidates.size}) is smaller than reservoir target ({target})"
        )
    if target:
        keep = np.argpartition(candidates["priority"], target - 1)[:target]
        reservoir = candidates[keep]
        reservoir = reservoir[np.argsort(reservoir["priority"], kind="stable")]
    else:
        reservoir = np.empty(0, RESERVOIR_DTYPE)
    reservoir_path = output_dir / "token_score_reservoir.npz"
    token_ids, token_counts = np.unique(reservoir["token_id"], return_counts=True)
    window_ids, window_counts = np.unique(reservoir["sample_order"], return_counts=True)
    document_ids, document_counts = np.unique(reservoir["document_id"], return_counts=True)
    _atomic_npz(
        reservoir_path,
        {
            "token_scores": reservoir,
            "token_ids": token_ids.astype(np.int32), "token_counts": token_counts.astype(np.int64),
            "window_ids": window_ids.astype(np.int64), "window_counts": window_counts.astype(np.int64),
            "document_ids": document_ids.astype(np.int64), "document_counts": document_counts.astype(np.int64),
        },
        compress=False,
    )
    summary = {
        "schema": MERGED_SCHEMA, "complete": True, "threshold_free": True,
        "final_gt_created": False,
        "exact_gt_requires_targeted_second_pass_after_threshold_lock": True,
        "worker_count": worker_count, "manifest_identity": next(iter(identity)),
        "processed_windows": total["window_count"],
        "processed_retained_tokens": total["retained_token_count"],
        "processed_eligible_tokens": total["eligible_token_count"],
        "histograms": _file_identity(hist_path),
        "histogram_report": histogram_report,
        "token_score_reservoir": _file_identity(reservoir_path),
        "reservoir_rows": int(reservoir.size),
        "reservoir_unique_windows": int(window_ids.size),
        "reservoir_unique_documents": int(document_ids.size),
        "reservoir_unique_token_ids": int(token_ids.size),
        "exact_chunk_b_shards": [x["path"] for x in records],
        "exact_chunk_b_bytes": sum(x["size_bytes"] for x in records),
        "pilot_overlay": summaries[0]["pilot_overlay"],
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--dataset-prefix", default=DEFAULT_CODE_PREFIX)
    build.add_argument("--pilot-manifest")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--enforce-known-code-counts", action="store_true")
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--dataset-prefix")
    validate.add_argument("--enforce-known-code-counts", action="store_true")
    merge = commands.add_parser("merge-workers")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--worker-count", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-manifest":
        payload = build_full_train_manifest(
            MMapIndexedDatasetLite(args.dataset_prefix), dataset_prefix=args.dataset_prefix,
            output_dir=args.output_dir, pilot_manifest=args.pilot_manifest,
            enforce_known_code_counts=args.enforce_known_code_counts
        )
    elif args.command == "validate-manifest":
        dataset = MMapIndexedDatasetLite(args.dataset_prefix) if args.dataset_prefix else None
        payload = validate_full_train_manifest(
            args.manifest, dataset=dataset, dataset_prefix=args.dataset_prefix,
            enforce_known_code_counts=args.enforce_known_code_counts
        )
    else:
        payload = merge_worker_outputs(args.output_dir, args.worker_count)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BTMBatch", "CHUNK_LAYOUT", "FullCensusWorker", "HistogramAccumulator",
    "HistogramSpec", "KNOWN_CODE_COUNTS", "LAYERS", "RESERVOIR_DTYPE", "SCALES",
    "build_full_train_manifest", "compute_btm_batch", "default_histogram_specs",
    "load_full_train_manifest", "manifest_batch_plan", "merge_worker_outputs",
    "partition_bounds", "partition_manifest_rows", "reservoir_candidates",
    "token_priorities", "validate_full_train_manifest",
]
