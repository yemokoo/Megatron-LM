#!/usr/bin/env python3
"""Exact targeted CKA selector and compact token-GT writer.

This is the second pass after the threshold-free full census.  Its input is
the exact union of windows containing at least one token that passes a frozen
B threshold bundle.  The caller forwards those windows through the same
before/after checkpoint pair and supplies :class:`BTMBatch` values.  This
module then applies all frozen B/T/magnitude thresholds exactly and writes
only packed masks and selected occurrence identities -- never hidden states.

The three Wiki-calibrated bundles (95/97/99) remain separate.  This module
does not inspect the sealed pilot test split and does not claim that any one
bundle is the human-locked final GT.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    from .cka_gt_full_census import (
        BTMBatch,
        _canonical_json,
        _atomic_json,
        _atomic_npy,
        _atomic_npz,
        _decode_json_array,
        _json_array,
        manifest_batch_plan,
    )
    from .cka_gt_pilot_core import (
        condition_specific_consensus,
        same_layer_consensus,
    )
    from .cka_gt_pilot_windows import WINDOW_DTYPE, file_sha256
except ImportError:  # Direct script/test execution.
    from cka_gt_full_census import (
        BTMBatch,
        _canonical_json,
        _atomic_json,
        _atomic_npy,
        _atomic_npz,
        _decode_json_array,
        _json_array,
        manifest_batch_plan,
    )
    from cka_gt_pilot_core import (
        condition_specific_consensus,
        same_layer_consensus,
    )
    from cka_gt_pilot_windows import WINDOW_DTYPE, file_sha256


SCHEMA = "cka_gt_exact_targeted_v2"
PROGRESS_SCHEMA = "cka_gt_exact_targeted_progress_v2"
SHARD_SCHEMA = "cka_gt_exact_targeted_shard_v2"
SUMMARY_SCHEMA = "cka_gt_exact_targeted_summary_v2"
AUTHORITATIVE_B_SCHEMA = "cka_gt_authoritative_b_masks_v1"
BUNDLE_LEVELS = (95, 97, 99)
SCALES = (128, 256)
LAYERS = tuple(range(2, 10))
PACKED_BYTES_PER_WINDOW = 512 // 8
CONDITION_COUNT_NAMES = (
    "eligible",
    "B128",
    "B256",
    "B",
    "T128",
    "T256",
    "T",
    "L2",
    "R",
    "M",
    "B+T",
    "B+M",
    "B+T+M",
    "same_layer",
)

OCCURRENCE_DTYPE = np.dtype(
    [
        ("candidate_index", "<i8"),
        ("sample_order", "<i8"),
        ("source_window_index", "<i8"),
        ("document_id", "<i8"),
        ("window_offset", "<i8"),
        ("position", "<i2"),
        ("token_id", "<i4"),
    ],
    align=False,
)


@dataclass(frozen=True)
class ThresholdBundle:
    level: int
    b: Mapping[int, torch.Tensor]
    t: Mapping[int, torch.Tensor]
    rel_l2: torch.Tensor
    abs_log_r: torch.Tensor


def _file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _threshold_vector(value: Any, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if result.shape != (len(LAYERS),) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly eight finite values")
    return result


def load_threshold_bundles(
    path: str | Path,
) -> tuple[dict[str, Any], dict[int, ThresholdBundle]]:
    """Load the frozen Wiki-derived candidate lines without touching test data."""

    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "cka_gt_pilot_postprocess_v1":
        raise ValueError(f"unsupported analysis config schema: {config.get('schema')}")
    raw = config.get("candidate_thresholds", {})
    bundles: dict[int, ThresholdBundle] = {}
    for level in BUNDLE_LEVELS:
        item = raw.get(str(level), raw.get(level))
        if not isinstance(item, Mapping):
            raise ValueError(f"analysis config is missing threshold bundle {level}")
        b_raw, t_raw = item.get("B_lower_threshold"), item.get("T_lower_threshold")
        if not isinstance(b_raw, Mapping) or not isinstance(t_raw, Mapping):
            raise ValueError(f"threshold bundle {level} is missing B/T scales")
        bundles[level] = ThresholdBundle(
            level=level,
            b={
                scale: _threshold_vector(
                    b_raw.get(str(scale), b_raw.get(scale)), f"bundle{level}.B{scale}"
                )
                for scale in SCALES
            },
            t={
                scale: _threshold_vector(
                    t_raw.get(str(scale), t_raw.get(scale)), f"bundle{level}.T{scale}"
                )
                for scale in SCALES
            },
            rel_l2=_threshold_vector(
                item.get("relative_l2_upper_threshold"), f"bundle{level}.rel_l2"
            ),
            abs_log_r=_threshold_vector(
                item.get("abs_log_r_upper_threshold"), f"bundle{level}.abs_log_r"
            ),
        )
    return _file_identity(path), bundles


def _bundle_tensors(
    bundle: ThresholdBundle, *, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    thresholds = {
        "B128": bundle.b[128].to(device=device),
        "B256": bundle.b[256].to(device=device),
        "T128": bundle.t[128].to(device=device),
        "T256": bundle.t[256].to(device=device),
        "L2": bundle.rel_l2.to(device=device),
        "R": bundle.abs_log_r.to(device=device),
    }
    comparisons = {
        "B128": "ge",
        "B256": "ge",
        "T128": "ge",
        "T256": "ge",
        "L2": "le",
        "R": "le",
    }
    return thresholds, comparisons


@torch.no_grad()
def evaluate_bundle(
    metrics: BTMBatch, bundle: ThresholdBundle
) -> dict[str, torch.Tensor]:
    """Apply the frozen condition-specific selector and same-layer diagnostic."""

    conditions = {
        "B128": metrics.b_min[128],
        "B256": metrics.b_min[256],
        "T128": metrics.t_min[128],
        "T256": metrics.t_min[256],
        "L2": metrics.rel_l2,
        "R": metrics.abs_log_r,
    }
    device = metrics.rel_l2.device
    thresholds, comparisons = _bundle_tensors(bundle, device=device)
    primary = condition_specific_consensus(
        conditions,
        thresholds,
        comparisons,
        reject_negative=("T128", "T256"),
        min_valid_layers=6,
        required_pass_layers=7,
    )
    same = same_layer_consensus(
        conditions,
        thresholds,
        comparisons,
        reject_negative=("T128", "T256"),
        min_valid_layers=6,
        required_pass_layers=7,
    )
    by = primary["by_condition"]
    masks = {
        name: by[name]["passed"] & metrics.eligible
        for name in ("B128", "B256", "T128", "T256", "L2", "R")
    }
    masks["eligible"] = primary["eligible"] & metrics.eligible
    masks["B"] = masks["B128"] & masks["B256"]
    masks["T"] = masks["T128"] & masks["T256"]
    masks["M"] = masks["L2"] & masks["R"]
    masks["B+T"] = masks["B"] & masks["T"]
    masks["B+M"] = masks["B"] & masks["M"]
    masks["B+T+M"] = masks["B"] & masks["T"] & masks["M"]
    masks["same_layer"] = same["passed"] & metrics.eligible
    masks["selected"] = masks["B+T+M"]
    return masks


def _pack_mask(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[1] > 512:
        raise ValueError("token mask must be [batch,length<=512]")
    value = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    padded = np.zeros((value.shape[0], 512), dtype=np.bool_)
    padded[:, : value.shape[1]] = value
    return np.packbits(padded, axis=1, bitorder="little")


def unpack_mask(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != PACKED_BYTES_PER_WINDOW:
        raise ValueError("packed mask must be [windows,64] uint8")
    return np.unpackbits(value.astype(np.uint8, copy=False), axis=1, bitorder="little")[:, :512].astype(bool)


def _unpack_layer_mask(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if (
        value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[1:] != (PACKED_BYTES_PER_WINDOW, len(LAYERS))
    ):
        raise ValueError("packed layer mask must be uint8 [windows,64,8]")
    return np.unpackbits(value, axis=1, bitorder="little")[:, :512].astype(bool)


def _condition_consensus_from_layer_states(
    valid: torch.Tensor, passed: torch.Tensor
) -> torch.Tensor:
    """Match the approved 7/8 rule, with exactly six valid requiring 6/6."""

    valid = valid.to(dtype=torch.bool)
    passed = passed.to(dtype=torch.bool) & valid
    n_valid = valid.sum(dim=-1)
    n_pass = passed.sum(dim=-1)
    required = torch.where(n_valid == 6, 6, 7)
    return (n_valid >= 6) & (n_pass >= required)


@torch.no_grad()
def apply_authoritative_b(
    metrics: BTMBatch,
    bundle: ThresholdBundle,
    recomputed: dict[str, torch.Tensor],
    *,
    authoritative_b: torch.Tensor,
    authoritative_scale_valid: Mapping[int, torch.Tensor],
    authoritative_scale_pass: Mapping[int, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Combine census-exact B with targeted T/magnitude measurements.

    The targeted forward still recomputes B as a numerical diagnostic, but it
    never overrides the exhaustive full-census B decision used for selection.
    """

    eligible = metrics.eligible.to(dtype=torch.bool)
    authoritative_b = authoritative_b.to(device=eligible.device, dtype=torch.bool)
    if authoritative_b.shape != eligible.shape:
        raise ValueError("authoritative B does not align with targeted token axis")
    result = dict(recomputed)
    for scale in SCALES:
        valid = authoritative_scale_valid[scale].to(
            device=eligible.device, dtype=torch.bool
        )
        passed = authoritative_scale_pass[scale].to(
            device=eligible.device, dtype=torch.bool
        )
        if valid.shape != (*eligible.shape, len(LAYERS)) or passed.shape != valid.shape:
            raise ValueError(f"authoritative B{scale} layer state shape differs")
        result[f"B{scale}"] = _condition_consensus_from_layer_states(
            valid, passed
        ) & eligible
    result["B"] = authoritative_b & eligible
    if not torch.equal(result["B"], result["B128"] & result["B256"]):
        raise RuntimeError("authoritative token B differs from its scale consensuses")

    # Rebuild the approved same-layer diagnostic with census-exact B states
    # and the targeted T/magnitude states on the same layer.
    layer_valid = None
    layer_pass = None
    for scale in SCALES:
        value = metrics.t_min[scale]
        threshold = bundle.t[scale].to(device=value.device)
        valid = torch.isfinite(value) & eligible[..., None]
        passed = valid & (value >= threshold) & (value >= 0)
        b_valid = authoritative_scale_valid[scale].to(value.device, torch.bool)
        b_pass = authoritative_scale_pass[scale].to(value.device, torch.bool)
        valid = valid & b_valid
        passed = passed & b_pass
        layer_valid = valid if layer_valid is None else layer_valid & valid
        layer_pass = passed if layer_pass is None else layer_pass & passed
    for value, threshold in (
        (metrics.rel_l2, bundle.rel_l2),
        (metrics.abs_log_r, bundle.abs_log_r),
    ):
        cut = threshold.to(device=value.device)
        valid = torch.isfinite(value) & eligible[..., None]
        passed = valid & (value <= cut)
        layer_valid &= valid
        layer_pass &= passed
    result["same_layer"] = _condition_consensus_from_layer_states(
        layer_valid, layer_pass
    ) & eligible
    result["B+T"] = result["B"] & result["T"]
    result["B+M"] = result["B"] & result["M"]
    result["B+T+M"] = result["B"] & result["T"] & result["M"]
    result["selected"] = result["B+T+M"]
    result["B_recomputed"] = recomputed["B"]
    return result


def _occurrences(
    mask: torch.Tensor,
    token_ids: torch.Tensor,
    rows: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    selected = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    batch_ids, positions = np.nonzero(selected)
    result = np.empty(batch_ids.size, dtype=OCCURRENCE_DTYPE)
    if not batch_ids.size:
        return result
    chosen_rows = rows[batch_ids]
    result["candidate_index"] = candidate_indices[batch_ids]
    for key in ("sample_order", "source_window_index", "document_id", "window_offset"):
        result[key] = chosen_rows[key]
    result["position"] = positions.astype(np.int16)
    token_cpu = token_ids.detach().to(device="cpu").numpy()
    result["token_id"] = token_cpu[batch_ids, positions].astype(np.int32)
    return result


class _Pending:
    def __init__(self, start_batch: int) -> None:
        self.start_batch = self.next_batch = int(start_batch)
        self.candidate_indices: list[np.ndarray] = []
        self.rows: list[np.ndarray] = []
        self.packed = {level: [] for level in BUNDLE_LEVELS}
        self.same_packed = {level: [] for level in BUNDLE_LEVELS}
        self.b_packed = {level: [] for level in BUNDLE_LEVELS}
        self.b_recomputed_packed = {level: [] for level in BUNDLE_LEVELS}
        self.occurrences = {level: [] for level in BUNDLE_LEVELS}
        self.condition_counts = {
            level: np.zeros(len(CONDITION_COUNT_NAMES), dtype=np.int64)
            for level in BUNDLE_LEVELS
        }

    @property
    def batch_count(self) -> int:
        return self.next_batch - self.start_batch


class ExactTargetedGTWriter:
    """Atomic/resumable writer for exact candidate-window paired forwards."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        candidate_windows_path: str | Path,
        candidate_manifest_path: str | Path,
        analysis_config_path: str | Path,
        census_summary_path: str | Path,
        model_binding_path: str | Path,
        authoritative_b_path: str | Path | None = None,
        batch_size: int,
        checkpoint_every_batches: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.output_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.json"
        self.summary_path = self.output_dir / "summary.json"
        self.batch_size = int(batch_size)
        self.checkpoint_every_batches = int(checkpoint_every_batches)
        if self.batch_size <= 0 or self.checkpoint_every_batches <= 0:
            raise ValueError("batch/checkpoint cadence must be positive")

        candidate_windows_path = Path(candidate_windows_path).resolve()
        rows = np.load(candidate_windows_path, mmap_mode="r", allow_pickle=False)
        if rows.dtype != WINDOW_DTYPE or rows.ndim != 1 or rows.size == 0:
            raise ValueError("candidate windows must be a nonempty 1-D WINDOW_DTYPE array")
        if not np.all(rows["sample_order"][1:] > rows["sample_order"][:-1]):
            raise ValueError("candidate windows must be strictly sample_order sorted")
        self.rows = rows
        self.candidate_windows_identity = _file_identity(candidate_windows_path)
        candidate_manifest_path = Path(candidate_manifest_path).resolve()
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        if candidate_manifest.get("schema") != "cka_gt_b_candidate_windows_v2":
            raise ValueError("candidate manifest has unsupported schema")
        if not candidate_manifest.get("complete") or not candidate_manifest.get("not_final_gt"):
            raise ValueError("candidate manifest must be a complete, B-only non-final GT")
        recorded_content_hash = candidate_manifest.get("manifest_content_sha256")
        content_without_hash = dict(candidate_manifest)
        content_without_hash.pop("manifest_content_sha256", None)
        observed_content_hash = hashlib.sha256(
            _canonical_json(content_without_hash)
        ).hexdigest()
        if recorded_content_hash != observed_content_hash:
            raise ValueError("candidate manifest canonical content hash differs")
        bound_candidate = candidate_manifest.get("candidate_windows", {})
        for key in ("size_bytes", "sha256"):
            if bound_candidate.get(key) != self.candidate_windows_identity[key]:
                raise ValueError(f"candidate manifest file identity differs for {key}")
        if Path(bound_candidate.get("path", "")).resolve() != candidate_windows_path:
            raise ValueError("candidate manifest binds a different candidate-windows path")
        statistics = candidate_manifest.get("statistics", {})
        if int(statistics.get("candidate_window_count", -1)) != int(len(rows)):
            raise ValueError("candidate manifest window count differs")
        self.candidate_manifest_identity = _file_identity(candidate_manifest_path)
        self.analysis_config_identity, self.bundles = load_threshold_bundles(
            analysis_config_path
        )
        bound_config = candidate_manifest.get("analysis_config", {})
        for key in ("size_bytes", "sha256"):
            if bound_config.get(key) != self.analysis_config_identity[key]:
                raise ValueError(f"candidate manifest analysis-config identity differs for {key}")
        if Path(bound_config.get("path", "")).resolve() != Path(analysis_config_path).resolve():
            raise ValueError("candidate manifest binds a different analysis-config path")
        census_summary_path = Path(census_summary_path).resolve()
        census_summary = json.loads(census_summary_path.read_text(encoding="utf-8"))
        if (
            census_summary.get("schema") != "cka_gt_full_census_merged_summary_v2"
            or not census_summary.get("complete")
            or not census_summary.get("threshold_free")
        ):
            raise ValueError("census summary is not a complete threshold-free merge")
        self.census_summary_identity = _file_identity(census_summary_path)
        bound_census = candidate_manifest.get("merged_census_summary")
        if not isinstance(bound_census, Mapping):
            raise ValueError("candidate manifest does not bind a merged census summary")
        for key in ("size_bytes", "sha256"):
            if bound_census.get(key) != self.census_summary_identity[key]:
                raise ValueError(f"candidate manifest census-summary identity differs for {key}")
        if Path(bound_census.get("path", "")).resolve() != census_summary_path:
            raise ValueError("candidate manifest binds a different census-summary path")
        if (
            candidate_manifest.get("source_manifest", {}).get("content_identity")
            != census_summary.get("manifest_identity")
        ):
            raise ValueError("candidate manifest and census summary source identities differ")
        self.full_eligible_token_count = int(census_summary.get("processed_eligible_tokens", 0))
        if self.full_eligible_token_count <= 0:
            raise ValueError("full eligible-token denominator must be positive")
        if int(statistics.get("source_eligible_token_count", -1)) != self.full_eligible_token_count:
            raise ValueError("candidate/full-census eligible-token denominator differs")
        expected_b_counts: dict[int, int] = {}
        for level in BUNDLE_LEVELS:
            count = statistics.get("per_bundle", {}).get(str(level), {}).get(
                "B_passing_tokens"
            )
            if count is None or int(count) < 0:
                raise ValueError(f"candidate manifest has no valid B{level} token count")
            expected_b_counts[level] = int(count)
        self.expected_b_token_counts = expected_b_counts

        model_binding_path = Path(model_binding_path).resolve()
        model_binding = json.loads(model_binding_path.read_text(encoding="utf-8"))
        if model_binding.get("schema") != "cka_gt_exact_targeted_model_binding_v1":
            raise ValueError("targeted model binding has unsupported schema")
        if authoritative_b_path is None:
            bound_authoritative = model_binding.get("authoritative_b")
            if not isinstance(bound_authoritative, Mapping):
                raise ValueError("targeted model binding has no authoritative B artifact")
            authoritative_b_path = bound_authoritative.get("path")
        authoritative_b_path = Path(authoritative_b_path).resolve()
        self.authoritative_b_identity = _file_identity(authoritative_b_path)
        authoritative_files = {
            "candidate_windows": self.candidate_windows_identity,
            "candidate_manifest": self.candidate_manifest_identity,
            "analysis_config": self.analysis_config_identity,
            "census_summary": self.census_summary_identity,
            "authoritative_b": self.authoritative_b_identity,
        }
        for name, observed in authoritative_files.items():
            bound = model_binding.get(name)
            if not isinstance(bound, Mapping):
                raise ValueError(f"model binding does not bind {name}")
            for key in ("path", "size_bytes", "sha256"):
                if bound.get(key) != observed[key]:
                    raise ValueError(f"model binding {name} identity differs for {key}")
        for key in ("reference_checkpoint_identity", "current_checkpoint_identity"):
            value = model_binding.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"model binding has no authoritative {key}")
        if not isinstance(model_binding.get("source_dataset_identity"), Mapping):
            raise ValueError("model binding has no source dataset identity")
        if int(model_binding.get("batch_size", -1)) != self.batch_size:
            raise ValueError("model binding logical batch size differs")
        if int(model_binding.get("checkpoint_every_batches", -1)) != self.checkpoint_every_batches:
            raise ValueError("model binding checkpoint cadence differs")
        self.model_binding_identity = _file_identity(model_binding_path)
        worker_bindings = candidate_manifest.get("census_worker_model_bindings")
        if not isinstance(worker_bindings, Mapping):
            raise ValueError("candidate manifest does not carry census worker bindings")
        if (
            worker_bindings.get("schema") != "cka_gt_full_census_model_binding_set_v1"
            or worker_bindings.get("validated") is not True
            or worker_bindings.get("all_workers_agree") is not True
            or int(worker_bindings.get("worker_count", -1)) != 4
        ):
            raise ValueError("candidate census-worker binding set is not authoritative")
        recorded_binding_set_hash = worker_bindings.get("binding_set_content_sha256")
        worker_binding_content = dict(worker_bindings)
        worker_binding_content.pop("binding_set_content_sha256", None)
        if recorded_binding_set_hash != hashlib.sha256(
            _canonical_json(worker_binding_content)
        ).hexdigest():
            raise ValueError("candidate census-worker binding-set canonical hash differs")
        binding_files = worker_bindings.get("files")
        if not isinstance(binding_files, list) or len(binding_files) != 4:
            raise ValueError("candidate census-worker binding set must contain four files")
        seen_workers: set[int] = set()
        parsed_worker_bindings: dict[int, Mapping[str, Any]] = {}
        for record in binding_files:
            worker_index = int(record.get("worker_index", -1))
            if worker_index not in range(4) or worker_index in seen_workers:
                raise ValueError("candidate census-worker binding indices are invalid")
            seen_workers.add(worker_index)
            identity = record.get("file_identity")
            content = record.get("content")
            if not isinstance(identity, Mapping) or not isinstance(content, Mapping):
                raise ValueError("candidate census-worker binding record is malformed")
            path = Path(identity.get("path", "")).resolve()
            observed = _file_identity(path)
            if any(observed[key] != identity.get(key) for key in ("path", "size_bytes", "sha256")):
                raise ValueError(f"census worker {worker_index} model-binding identity changed")
            live_content = json.loads(path.read_text(encoding="utf-8"))
            if live_content != content:
                raise ValueError(f"census worker {worker_index} model-binding content changed")
            if (
                int(content.get("worker_index", -1)) != worker_index
                or int(content.get("worker_count", -1)) != 4
            ):
                raise ValueError(f"census worker {worker_index} binding index/count differs")
            parsed_worker_bindings[worker_index] = content
        if seen_workers != set(range(4)):
            raise ValueError("candidate census-worker binding coverage is incomplete")

        common = worker_bindings.get("common_content")
        if not isinstance(common, Mapping):
            raise ValueError("candidate census-worker binding set has no common content")
        if worker_bindings.get("common_content_sha256") != hashlib.sha256(
            _canonical_json(common)
        ).hexdigest():
            raise ValueError("candidate census-worker common-content hash differs")
        common_config = common.get("analysis_config", {})
        if (
            Path(common_config.get("path", "")).resolve()
            != Path(self.analysis_config_identity["path"])
            or common_config.get("sha256") != self.analysis_config_identity["sha256"]
        ):
            raise ValueError("census-worker common analysis-config identity differs")
        common_manifest = common.get("manifest", {})
        if (
            common_manifest.get("content_identity") != census_summary.get("manifest_identity")
            or Path(common_manifest.get("path", "")).resolve()
            != Path(candidate_manifest.get("source_manifest", {}).get("path", "")).resolve()
        ):
            raise ValueError("census-worker common source-manifest identity differs")
        if (
            common.get("reference_checkpoint_identity")
            != model_binding["reference_checkpoint_identity"]
            or common.get("current_checkpoint_identity")
            != model_binding["current_checkpoint_identity"]
            or common.get("source_dataset_identity")
            != model_binding["source_dataset_identity"]
        ):
            raise ValueError("targeted and census-worker model/data lineage differs")
        common_to_worker_key = {
            "binding_schema": "schema",
            "reference_checkpoint_identity": "reference_checkpoint_identity",
            "current_checkpoint_identity": "current_checkpoint_identity",
            "source_dataset_identity": "source_dataset_identity",
            "window_batch_size": "window_batch_size",
            "checkpoint_every_batches": "checkpoint_every_batches",
            "histogram_bins": "histogram_bins",
            "global_token_score_reservoir_size": "global_token_score_reservoir_size",
            "max_windows": "max_windows",
            "raw_hidden_stored": "raw_hidden_stored",
            "threshold_policy": "threshold_policy",
        }
        for worker_index, content in parsed_worker_bindings.items():
            for common_key, worker_key in common_to_worker_key.items():
                if common.get(common_key) != content.get(worker_key):
                    raise ValueError(
                        f"census worker {worker_index} differs from common {common_key}"
                    )
            if Path(content.get("analysis_config", "")).resolve() != Path(
                self.analysis_config_identity["path"]
            ):
                raise ValueError(f"census worker {worker_index} analysis config path differs")
            if Path(content.get("manifest_path", "")).resolve() != Path(
                common_manifest.get("path", "")
            ).resolve():
                raise ValueError(f"census worker {worker_index} manifest path differs")
        per_worker = worker_bindings.get("per_worker_content")
        if not isinstance(per_worker, list) or len(per_worker) != 4:
            raise ValueError("candidate census-worker partition content is incomplete")
        partition_by_worker = {
            int(item.get("worker_index", -1)): int(item.get("partition_windows", -1))
            for item in per_worker
        }
        if set(partition_by_worker) != set(range(4)) or any(value <= 0 for value in partition_by_worker.values()):
            raise ValueError("candidate census-worker partition metadata is invalid")
        if sum(partition_by_worker.values()) != int(census_summary.get("processed_windows", -1)):
            raise ValueError("census-worker partition total differs from merged census")
        for worker_index, content in parsed_worker_bindings.items():
            if int(content.get("partition_windows", -1)) != partition_by_worker[worker_index]:
                raise ValueError(f"census worker {worker_index} partition size differs")
        self.census_worker_model_bindings = dict(worker_bindings)
        self.census_worker_model_bindings_identity = {
            "schema": worker_bindings["schema"],
            "worker_count": 4,
            "binding_set_content_sha256": recorded_binding_set_hash,
            "common_content_sha256": worker_bindings["common_content_sha256"],
            "per_file_sha256": {
                str(record["worker_index"]): record["file_identity"]["sha256"]
                for record in binding_files
            },
        }

        with np.load(authoritative_b_path, allow_pickle=False) as artifact:
            if "metadata_json" not in artifact.files:
                raise ValueError("authoritative B artifact has no metadata")
            authoritative_metadata = json.loads(
                np.asarray(artifact["metadata_json"], dtype=np.uint8).tobytes().decode(
                    "utf-8"
                )
            )
            if (
                authoritative_metadata.get("schema") != AUTHORITATIVE_B_SCHEMA
                or authoritative_metadata.get("complete") is not True
                or authoritative_metadata.get("raw_hidden_stored") is not False
                or authoritative_metadata.get("sealed_test_opened") is not False
            ):
                raise ValueError("authoritative B artifact metadata is invalid")
            if int(authoritative_metadata.get("candidate_window_count", -1)) != len(rows):
                raise ValueError("authoritative B candidate-window count differs")
            for name, observed in (
                ("candidate_windows", self.candidate_windows_identity),
                ("analysis_config", self.analysis_config_identity),
                ("merged_census_summary", self.census_summary_identity),
            ):
                bound = authoritative_metadata.get(name)
                if not isinstance(bound, Mapping) or any(
                    bound.get(key) != observed[key]
                    for key in ("path", "size_bytes", "sha256")
                ):
                    raise ValueError(f"authoritative B {name} identity differs")
            bound_manifest = authoritative_metadata.get("candidate_manifest", {})
            if any(
                bound_manifest.get(key) != self.candidate_manifest_identity[key]
                for key in ("path", "size_bytes", "sha256")
            ) or bound_manifest.get("manifest_content_sha256") != recorded_content_hash:
                raise ValueError("authoritative B candidate-manifest identity differs")
            if authoritative_metadata.get(
                "census_worker_model_binding_set_sha256"
            ) != self.census_worker_model_bindings_identity[
                "binding_set_content_sha256"
            ]:
                raise ValueError("authoritative B census-worker lineage differs")
            if authoritative_metadata.get("bundle_b_token_counts") != {
                str(level): self.expected_b_token_counts[level]
                for level in BUNDLE_LEVELS
            }:
                raise ValueError("authoritative B token counts differ from candidates")
            self.authoritative_b = {
                level: unpack_mask(artifact[f"bundle_{level}_token_packed"])
                for level in BUNDLE_LEVELS
            }
            self.authoritative_scale_valid = {
                scale: _unpack_layer_mask(artifact[f"scale_{scale}_valid_packed"])
                for scale in SCALES
            }
            self.authoritative_scale_pass = {
                (level, scale): _unpack_layer_mask(
                    artifact[f"bundle_{level}_scale_{scale}_pass_packed"]
                )
                for level in BUNDLE_LEVELS
                for scale in SCALES
            }
        self.authoritative_b_metadata = authoritative_metadata
        for level in BUNDLE_LEVELS:
            if int(self.authoritative_b[level].sum()) != self.expected_b_token_counts[level]:
                raise ValueError(f"authoritative B{level} unpacked count differs")
        self.plan = manifest_batch_plan(rows, self.batch_size)
        self.progress = self._load_progress()
        self._next_batch = int(self.progress["next_batch_index"])
        self.pending = _Pending(self._next_batch)

    @property
    def next_batch_index(self) -> int:
        return self._next_batch

    @property
    def expected_batch_count(self) -> int:
        return len(self.plan)

    def _identity(self) -> dict[str, Any]:
        return {
            "candidate_windows": self.candidate_windows_identity,
            "candidate_manifest": self.candidate_manifest_identity,
            "analysis_config": self.analysis_config_identity,
            "census_summary": self.census_summary_identity,
            "model_binding": self.model_binding_identity,
            "authoritative_b": self.authoritative_b_identity,
            "batch_size": self.batch_size,
            "checkpoint_every_batches": self.checkpoint_every_batches,
            "full_eligible_token_count": self.full_eligible_token_count,
            "candidate_window_count": int(len(self.rows)),
            "bundle_levels": list(BUNDLE_LEVELS),
            "expected_b_token_counts": {
                str(level): self.expected_b_token_counts[level]
                for level in BUNDLE_LEVELS
            },
            "census_worker_model_bindings_identity": self.census_worker_model_bindings_identity,
        }

    def _load_progress(self) -> dict[str, Any]:
        identity = self._identity()
        if not self.progress_path.exists():
            payload = {
                "schema": PROGRESS_SCHEMA,
                **identity,
                "expected_batches": int(len(self.plan)),
                "next_batch_index": 0,
                "processed_windows": 0,
                "shards": [],
                "finalized": False,
                "raw_hidden_stored": False,
                "sealed_test_opened": False,
            }
            _atomic_json(self.progress_path, payload)
            return payload
        payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if payload.get("schema") != PROGRESS_SCHEMA:
            raise RuntimeError("unsupported targeted-GT progress schema")
        for key, value in identity.items():
            if payload.get(key) != value:
                raise RuntimeError(f"targeted-GT resume identity mismatch: {key}")
        expected_batch = 0
        processed = 0
        for index, record in enumerate(payload.get("shards", ())):
            if record.get("index") != index or record.get("batch_start") != expected_batch:
                raise RuntimeError("targeted-GT shard journal is non-contiguous")
            path = Path(record["path"])
            if not path.is_file() or file_sha256(path) != record.get("sha256"):
                raise RuntimeError(f"targeted-GT shard is missing/corrupt: {path}")
            expected_batch = int(record["batch_stop"])
            processed += int(record["window_count"])
        if expected_batch != int(payload.get("next_batch_index", -1)):
            raise RuntimeError("targeted-GT journal next batch differs")
        if processed != int(payload.get("processed_windows", -1)):
            raise RuntimeError("targeted-GT journal processed-window count differs")
        return payload

    def process_batch(
        self,
        *,
        batch_index: int,
        candidate_indices: np.ndarray,
        rows: np.ndarray,
        token_ids: torch.Tensor,
        metrics: BTMBatch,
    ) -> dict[int, int]:
        batch_index = int(batch_index)
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        rows = np.asarray(rows)
        if self.progress.get("finalized"):
            raise RuntimeError("cannot append to a finalized targeted pass")
        if batch_index != self._next_batch:
            raise RuntimeError(f"expected batch {self._next_batch}, got {batch_index}")
        expected_indices = self.plan[batch_index]
        if not np.array_equal(candidate_indices, expected_indices):
            raise RuntimeError("candidate indices differ from deterministic plan")
        if rows.dtype != WINDOW_DTYPE or not np.array_equal(rows, self.rows[candidate_indices]):
            raise RuntimeError("batch rows differ from candidate manifest")
        if token_ids.ndim != 2 or token_ids.shape != metrics.eligible.shape:
            raise ValueError("token IDs do not align with exact metrics")
        if token_ids.shape[0] != len(rows) or np.any(rows["window_length"] != token_ids.shape[1]):
            raise ValueError("token tensor does not align with unpadded candidate rows")

        self.pending.candidate_indices.append(candidate_indices.copy())
        self.pending.rows.append(rows.copy())
        selected_counts: dict[int, int] = {}
        for level in BUNDLE_LEVELS:
            recomputed = evaluate_bundle(metrics, self.bundles[level])
            batch_slice = candidate_indices
            result = apply_authoritative_b(
                metrics,
                self.bundles[level],
                recomputed,
                authoritative_b=torch.as_tensor(
                    self.authoritative_b[level][batch_slice, : token_ids.shape[1]],
                    device=metrics.rel_l2.device,
                ),
                authoritative_scale_valid={
                    scale: torch.as_tensor(
                        self.authoritative_scale_valid[scale][
                            batch_slice, : token_ids.shape[1]
                        ],
                        device=metrics.rel_l2.device,
                    )
                    for scale in SCALES
                },
                authoritative_scale_pass={
                    scale: torch.as_tensor(
                        self.authoritative_scale_pass[(level, scale)][
                            batch_slice, : token_ids.shape[1]
                        ],
                        device=metrics.rel_l2.device,
                    )
                    for scale in SCALES
                },
            )
            selected = result["selected"]
            self.pending.packed[level].append(_pack_mask(selected))
            self.pending.same_packed[level].append(_pack_mask(result["same_layer"]))
            self.pending.b_packed[level].append(_pack_mask(result["B"]))
            self.pending.b_recomputed_packed[level].append(
                _pack_mask(result["B_recomputed"])
            )
            self.pending.occurrences[level].append(
                _occurrences(selected, token_ids, rows, candidate_indices)
            )
            for index, name in enumerate(CONDITION_COUNT_NAMES):
                self.pending.condition_counts[level][index] += int(result[name].sum().item())
            selected_counts[level] = int(selected.sum().item())
        self.pending.next_batch += 1
        self._next_batch += 1
        if self.pending.batch_count >= self.checkpoint_every_batches:
            self._commit_pending()
        return selected_counts

    def _commit_pending(self) -> None:
        if self.pending.batch_count == 0:
            return
        shard_index = len(self.progress["shards"])
        candidate_indices = np.concatenate(self.pending.candidate_indices)
        rows = np.concatenate(self.pending.rows)
        arrays: dict[str, np.ndarray] = {
            "candidate_indices": candidate_indices,
            "rows": rows,
            "condition_count_names": np.asarray(CONDITION_COUNT_NAMES, dtype="S16"),
        }
        selected_counts: dict[str, int] = {}
        for level in BUNDLE_LEVELS:
            arrays[f"bundle_{level}_packed"] = np.concatenate(self.pending.packed[level])
            arrays[f"bundle_{level}_same_layer_packed"] = np.concatenate(
                self.pending.same_packed[level]
            )
            arrays[f"bundle_{level}_b_packed"] = np.concatenate(self.pending.b_packed[level])
            arrays[f"bundle_{level}_b_recomputed_packed"] = np.concatenate(
                self.pending.b_recomputed_packed[level]
            )
            occurrences = np.concatenate(self.pending.occurrences[level])
            arrays[f"bundle_{level}_occurrences"] = occurrences
            arrays[f"bundle_{level}_condition_counts"] = self.pending.condition_counts[level]
            selected_counts[str(level)] = int(occurrences.size)
        metadata = {
            "schema": SHARD_SCHEMA,
            "index": shard_index,
            "batch_start": self.pending.start_batch,
            "batch_stop": self.pending.next_batch,
            "window_count": int(candidate_indices.size),
            "selected_counts": selected_counts,
        }
        arrays["metadata_json"] = _json_array(metadata)
        path = self.shard_dir / f"shard_{shard_index:06d}.npz"
        _atomic_npz(path, arrays, compress=False)
        record = {
            **metadata,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        progress = dict(self.progress)
        progress["shards"] = [*progress["shards"], record]
        progress["next_batch_index"] = self.pending.next_batch
        progress["processed_windows"] = int(progress["processed_windows"]) + int(
            candidate_indices.size
        )
        _atomic_json(self.progress_path, progress)
        self.progress = progress
        self.pending = _Pending(self._next_batch)

    def _validate_occurrences(
        self, occurrences: np.ndarray, packed: np.ndarray, level: int
    ) -> None:
        if occurrences.dtype != OCCURRENCE_DTYPE:
            raise RuntimeError(f"bundle {level} occurrence dtype differs")
        masks = unpack_mask(packed)
        if int(masks.sum()) != int(occurrences.size):
            raise RuntimeError(f"bundle {level} mask/occurrence count differs")
        if not occurrences.size:
            return
        candidate = occurrences["candidate_index"]
        position = occurrences["position"].astype(np.int64)
        if np.any(candidate < 0) or np.any(candidate >= len(self.rows)):
            raise RuntimeError(f"bundle {level} occurrence candidate index is invalid")
        if np.any(position < 0) or np.any(position >= self.rows["eligible_token_count"][candidate]):
            raise RuntimeError(f"bundle {level} occurrence position is ineligible")
        if not masks[candidate, position].all():
            raise RuntimeError(f"bundle {level} occurrence is absent from packed mask")
        for key in ("sample_order", "source_window_index", "document_id", "window_offset"):
            if not np.array_equal(occurrences[key], self.rows[key][candidate]):
                raise RuntimeError(f"bundle {level} occurrence identity differs for {key}")
        keys = np.empty(occurrences.size, dtype=[("candidate", "<i8"), ("position", "<i2")])
        keys["candidate"], keys["position"] = candidate, occurrences["position"]
        if np.unique(keys).size != occurrences.size:
            raise RuntimeError(f"bundle {level} contains duplicate occurrences")

    @staticmethod
    def _concentration(occurrences: np.ndarray) -> dict[str, Any]:
        count = int(occurrences.size)
        if count == 0:
            return {
                "selected_windows": 0,
                "selected_documents": 0,
                "unique_token_ids": 0,
                "max_window_share": 0.0,
                "max_document_share": 0.0,
                "top_token_share": 0.0,
                "token_id_entropy_bits": 0.0,
                "excess_repeat_fraction": 0.0,
            }
        _windows, window_counts = np.unique(occurrences["candidate_index"], return_counts=True)
        _docs, doc_counts = np.unique(occurrences["document_id"], return_counts=True)
        _tokens, token_counts = np.unique(occurrences["token_id"], return_counts=True)
        probabilities = token_counts.astype(np.float64) / count
        return {
            "selected_windows": int(window_counts.size),
            "selected_documents": int(doc_counts.size),
            "unique_token_ids": int(token_counts.size),
            "max_window_share": float(window_counts.max() / count),
            "max_document_share": float(doc_counts.max() / count),
            "top_token_share": float(token_counts.max() / count),
            "token_id_entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
            "excess_repeat_fraction": float((count - token_counts.size) / count),
        }

    def finalize(self) -> dict[str, Any]:
        if self.progress.get("finalized"):
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        self._commit_pending()
        if self._next_batch != len(self.plan):
            raise RuntimeError(
                f"targeted pass incomplete: {self._next_batch}/{len(self.plan)} batches"
            )
        packed = {
            level: np.zeros((len(self.rows), PACKED_BYTES_PER_WINDOW), dtype=np.uint8)
            for level in BUNDLE_LEVELS
        }
        same_packed = {level: np.zeros_like(packed[level]) for level in BUNDLE_LEVELS}
        b_packed = {level: np.zeros_like(packed[level]) for level in BUNDLE_LEVELS}
        b_recomputed_packed = {
            level: np.zeros_like(packed[level]) for level in BUNDLE_LEVELS
        }
        occurrences = {level: [] for level in BUNDLE_LEVELS}
        condition_counts = {
            level: np.zeros(len(CONDITION_COUNT_NAMES), dtype=np.int64)
            for level in BUNDLE_LEVELS
        }
        seen = np.zeros(len(self.rows), dtype=np.bool_)
        for record in self.progress["shards"]:
            path = Path(record["path"])
            with np.load(path, allow_pickle=False) as shard:
                metadata = _decode_json_array(shard["metadata_json"])
                if metadata.get("schema") != SHARD_SCHEMA:
                    raise RuntimeError(f"wrong targeted shard schema: {path}")
                indices = shard["candidate_indices"]
                if np.any(seen[indices]):
                    raise RuntimeError("candidate window appears in multiple targeted shards")
                if not np.array_equal(shard["rows"], self.rows[indices]):
                    raise RuntimeError("targeted shard rows differ from candidates")
                seen[indices] = True
                for level in BUNDLE_LEVELS:
                    packed[level][indices] = shard[f"bundle_{level}_packed"]
                    same_packed[level][indices] = shard[f"bundle_{level}_same_layer_packed"]
                    b_packed[level][indices] = shard[f"bundle_{level}_b_packed"]
                    b_recomputed_packed[level][indices] = shard[
                        f"bundle_{level}_b_recomputed_packed"
                    ]
                    occurrences[level].append(shard[f"bundle_{level}_occurrences"])
                    condition_counts[level] += shard[f"bundle_{level}_condition_counts"]
        if not seen.all():
            raise RuntimeError("targeted shard coverage is incomplete")

        exact_occurrences: dict[int, np.ndarray] = {}
        bundle_summaries: dict[str, Any] = {}
        b_count_validation = {
            "schema": "cka_gt_exact_b_count_validation_v2",
            "passed": True,
            "authoritative_source": self.authoritative_b_identity,
            "bundles": {},
        }
        for level in BUNDLE_LEVELS:
            observed_authoritative_b = int(
                condition_counts[level][CONDITION_COUNT_NAMES.index("B")]
            )
            observed_recomputed_b = int(unpack_mask(b_recomputed_packed[level]).sum())
            expected_b = int(self.expected_b_token_counts[level])
            matches = observed_authoritative_b == expected_b
            b_count_validation["bundles"][str(level)] = {
                "expected_from_candidate_manifest": expected_b,
                "observed_authoritative_full_census": observed_authoritative_b,
                "authoritative_matches": matches,
                "observed_targeted_gpu_recomputed_diagnostic": observed_recomputed_b,
                "targeted_minus_authoritative": observed_recomputed_b
                - observed_authoritative_b,
            }
            b_count_validation["passed"] = bool(
                b_count_validation["passed"] and matches
            )
        _atomic_json(self.output_dir / "b_count_validation.json", b_count_validation)
        if not b_count_validation["passed"]:
            raise RuntimeError(
                "materialized authoritative B token counts differ from the "
                f"candidate manifest: {b_count_validation['bundles']}"
            )
        for level in BUNDLE_LEVELS:
            value = np.concatenate(occurrences[level]) if occurrences[level] else np.empty(0, dtype=OCCURRENCE_DTYPE)
            if value.size:
                value = value[np.lexsort((value["position"], value["candidate_index"]))]
            self._validate_occurrences(value, packed[level], level)
            exact_occurrences[level] = value
            primary_mask, same_mask = unpack_mask(packed[level]), unpack_mask(same_packed[level])
            intersection = int((primary_mask & same_mask).sum())
            union = int((primary_mask | same_mask).sum())
            bundle_summaries[str(level)] = {
                "selected_occurrences": int(value.size),
                "full_train_eligible_coverage": float(
                    value.size / self.full_eligible_token_count
                ),
                "candidate_token_coverage": float(
                    value.size / max(1, int(self.rows["eligible_token_count"].sum()))
                ),
                "condition_counts": {
                    name: int(condition_counts[level][index])
                    for index, name in enumerate(CONDITION_COUNT_NAMES)
                },
                "same_layer": {
                    "condition_specific_count": int(primary_mask.sum()),
                    "same_layer_count": int(same_mask.sum()),
                    "intersection_count": intersection,
                    "union_count": union,
                    "jaccard": float(intersection / union) if union else 1.0,
                    "manual_review_required": bool(union and intersection / union < 0.9),
                },
                "concentration": self._concentration(value),
            }

        masks = {level: unpack_mask(packed[level]) for level in BUNDLE_LEVELS}
        nesting = {
            "95_subset_97": bool(np.all(~masks[95] | masks[97])),
            "97_subset_99": bool(np.all(~masks[97] | masks[99])),
        }
        if not all(nesting.values()):
            raise RuntimeError(f"frozen bundle selection is not nested: {nesting}")
        b99 = unpack_mask(b_packed[99])
        b99_candidate_qualified = b99.any(axis=1)
        if not b99_candidate_qualified.all():
            missing = np.flatnonzero(~b99_candidate_qualified)[:20].tolist()
            raise RuntimeError(
                "input contains windows that fail recomputed B99 candidate qualification: "
                f"{missing}"
            )

        for level in BUNDLE_LEVELS:
            _atomic_npy(self.output_dir / f"bundle_{level}_packed.npy", packed[level])
            _atomic_npy(
                self.output_dir / f"bundle_{level}_occurrences.npy", exact_occurrences[level]
            )
        summary = {
            "schema": SUMMARY_SCHEMA,
            **self._identity(),
            "complete": True,
            "not_human_locked_final_gt": True,
            "sealed_test_opened": False,
            "raw_hidden_stored": False,
            "layers": list(LAYERS),
            "scales": list(SCALES),
            "census_worker_model_bindings": self.census_worker_model_bindings,
            "authoritative_b_metadata": self.authoritative_b_metadata,
            "selector": (
                "full-census-authoritative condition-specific B128 7/8 & "
                "B256 7/8 & targeted T128 7/8 & "
                "T256 7/8 & rel-L2 7/8 & abs-log-r 7/8; exactly six valid layers uses 6/6"
            ),
            "candidate_windows_revalidated_by_authoritative_B99": int(
                b99_candidate_qualified.sum()
            ),
            "candidate_eligible_tokens": int(self.rows["eligible_token_count"].sum()),
            "bundle_nesting": nesting,
            "exact_b_count_validation": {
                **b_count_validation,
                "artifact": _file_identity(
                    self.output_dir / "b_count_validation.json"
                ),
            },
            "bundles": bundle_summaries,
            "outputs": {
                str(level): {
                    "packed_mask": _file_identity(
                        self.output_dir / f"bundle_{level}_packed.npy"
                    ),
                    "occurrences": _file_identity(
                        self.output_dir / f"bundle_{level}_occurrences.npy"
                    ),
                }
                for level in BUNDLE_LEVELS
            },
        }
        _atomic_json(self.summary_path, summary)
        progress = dict(self.progress)
        progress["finalized"] = True
        progress["summary"] = _file_identity(self.summary_path)
        _atomic_json(self.progress_path, progress)
        self.progress = progress
        return summary


def materialize_locked_bundle(
    *,
    exact_dir: str | Path,
    bundle_level: int,
    output_dir: str | Path,
    decision_reason: str,
) -> dict[str, Any]:
    """Create a new, immutable-by-policy final GT directory from exact outputs.

    The destination must not already exist.  This is deliberately separate
    from the exact measurement pass: looking at exact 95/97/99 summaries does
    not silently lock a threshold, and the sealed pilot test remains closed.
    """

    exact_dir, output_dir = Path(exact_dir).resolve(), Path(output_dir).resolve()
    bundle_level = int(bundle_level)
    if bundle_level not in BUNDLE_LEVELS:
        raise ValueError(f"bundle level must be one of {BUNDLE_LEVELS}")
    decision_reason = str(decision_reason).strip()
    if not decision_reason:
        raise ValueError("a nonempty human threshold-lock reason is required")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final GT directory: {output_dir}")
    summary_path = exact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or not summary.get("complete")
        or not summary.get("not_human_locked_final_gt")
        or summary.get("sealed_test_opened")
    ):
        raise RuntimeError("exact targeted summary is incomplete, already locked, or test-opened")

    def verify_summary_bound_file(key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        bound = summary.get(key)
        if not isinstance(bound, Mapping):
            raise RuntimeError(f"exact summary does not bind {key}")
        observed = _file_identity(Path(bound.get("path", "")))
        if any(observed[item] != bound.get(item) for item in ("path", "size_bytes", "sha256")):
            raise RuntimeError(f"exact summary {key} content identity changed")
        return observed, json.loads(Path(observed["path"]).read_text(encoding="utf-8"))

    observed_binding, model_binding = verify_summary_bound_file("model_binding")
    if model_binding.get("schema") != "cka_gt_exact_targeted_model_binding_v1":
        raise RuntimeError("targeted model binding schema changed")
    observed_census, census_summary = verify_summary_bound_file("census_summary")
    if (
        census_summary.get("schema") != "cka_gt_full_census_merged_summary_v2"
        or not census_summary.get("complete")
        or not census_summary.get("threshold_free")
    ):
        raise RuntimeError("bound census summary is not authoritative")
    observed_candidate_manifest, candidate_manifest = verify_summary_bound_file(
        "candidate_manifest"
    )
    recorded_candidate_hash = candidate_manifest.get("manifest_content_sha256")
    candidate_content = dict(candidate_manifest)
    candidate_content.pop("manifest_content_sha256", None)
    if recorded_candidate_hash != hashlib.sha256(_canonical_json(candidate_content)).hexdigest():
        raise RuntimeError("bound candidate manifest canonical hash changed")
    worker_bindings = summary.get("census_worker_model_bindings")
    worker_binding_identity = summary.get("census_worker_model_bindings_identity", {})
    worker_binding_content = dict(worker_bindings) if isinstance(worker_bindings, Mapping) else {}
    recorded_worker_binding_hash = worker_binding_content.pop(
        "binding_set_content_sha256", None
    )
    if (
        not isinstance(worker_bindings, Mapping)
        or worker_bindings != candidate_manifest.get("census_worker_model_bindings")
        or hashlib.sha256(_canonical_json(worker_binding_content)).hexdigest()
        != recorded_worker_binding_hash
        or recorded_worker_binding_hash
        != worker_binding_identity.get("binding_set_content_sha256")
        or worker_bindings.get("common_content_sha256")
        != worker_binding_identity.get("common_content_sha256")
        or worker_bindings.get("validated") is not True
        or worker_bindings.get("all_workers_agree") is not True
        or int(worker_bindings.get("worker_count", -1)) != 4
    ):
        raise RuntimeError("census-worker model-binding lineage changed")
    files = worker_bindings.get("files")
    if not isinstance(files, list) or len(files) != 4:
        raise RuntimeError("census-worker binding file set is incomplete")
    observed_workers: set[int] = set()
    for record in files:
        worker_index = int(record.get("worker_index", -1))
        identity = record.get("file_identity", {})
        content = record.get("content")
        if worker_index not in range(4) or worker_index in observed_workers:
            raise RuntimeError("census-worker binding index set changed")
        observed_workers.add(worker_index)
        observed = _file_identity(Path(identity.get("path", "")))
        if any(observed[key] != identity.get(key) for key in ("path", "size_bytes", "sha256")):
            raise RuntimeError(f"census worker {worker_index} model-binding hash changed")
        if json.loads(Path(observed["path"]).read_text(encoding="utf-8")) != content:
            raise RuntimeError(f"census worker {worker_index} model-binding content changed")
    common_worker_lineage = worker_bindings.get("common_content", {})
    if (
        common_worker_lineage.get("reference_checkpoint_identity")
        != model_binding.get("reference_checkpoint_identity")
        or common_worker_lineage.get("current_checkpoint_identity")
        != model_binding.get("current_checkpoint_identity")
        or common_worker_lineage.get("source_dataset_identity")
        != model_binding.get("source_dataset_identity")
        or common_worker_lineage.get("manifest", {}).get("content_identity")
        != census_summary.get("manifest_identity")
    ):
        raise RuntimeError("locked GT checkpoint/dataset/census lineage is inconsistent")
    authoritative_lineage = {
        "targeted_model_binding": {
            "file_identity": observed_binding,
            "content": model_binding,
        },
        "live_checkpoint_content_identities": {
            "reference": model_binding["reference_checkpoint_identity"],
            "current": model_binding["current_checkpoint_identity"],
        },
        "source_dataset_identity": model_binding["source_dataset_identity"],
        "census_summary": {
            "file_identity": observed_census,
            "manifest_identity": census_summary["manifest_identity"],
            "processed_windows": census_summary["processed_windows"],
            "processed_eligible_tokens": census_summary["processed_eligible_tokens"],
        },
        "candidate_manifest": {
            "file_identity": observed_candidate_manifest,
            "manifest_content_sha256": recorded_candidate_hash,
        },
        "census_worker_model_bindings": worker_bindings,
    }
    expected = summary.get("outputs", {}).get(str(bundle_level), {})
    source_packed = exact_dir / f"bundle_{bundle_level}_packed.npy"
    source_occurrences = exact_dir / f"bundle_{bundle_level}_occurrences.npy"
    for name, path in (("packed_mask", source_packed), ("occurrences", source_occurrences)):
        observed = _file_identity(path)
        binding = expected.get(name, {})
        if any(observed[key] != binding.get(key) for key in ("size_bytes", "sha256")):
            raise RuntimeError(f"exact targeted {name} content identity changed")

    candidate_identity = summary.get("candidate_windows", {})
    candidate_path = Path(candidate_identity.get("path", ""))
    observed_candidate = _file_identity(candidate_path)
    if any(
        observed_candidate[key] != candidate_identity.get(key)
        for key in ("size_bytes", "sha256")
    ):
        raise RuntimeError("candidate-window axis content identity changed")
    candidate_rows = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    packed = np.load(source_packed, allow_pickle=False)
    occurrences = np.load(source_occurrences, allow_pickle=False)
    if candidate_rows.dtype != WINDOW_DTYPE or candidate_rows.ndim != 1:
        raise RuntimeError("candidate-window axis is malformed")
    if packed.dtype != np.uint8 or packed.shape != (
        len(candidate_rows), PACKED_BYTES_PER_WINDOW
    ):
        raise RuntimeError("exact packed mask has wrong dtype/shape")
    if occurrences.dtype != OCCURRENCE_DTYPE:
        raise RuntimeError("exact occurrence array has wrong dtype")
    unpacked = unpack_mask(packed)
    eligible = np.arange(512)[None, :] < candidate_rows["eligible_token_count"][:, None]
    if np.any(unpacked & ~eligible):
        raise RuntimeError("exact packed mask selects an ineligible suffix token")
    if int(unpacked.sum()) != int(occurrences.size):
        raise RuntimeError("exact packed-mask/occurrence count differs")
    if occurrences.size:
        candidate_index = occurrences["candidate_index"]
        position = occurrences["position"].astype(np.int64)
        if (
            np.any(candidate_index < 0)
            or np.any(candidate_index >= len(candidate_rows))
            or not unpacked[candidate_index, position].all()
        ):
            raise RuntimeError("exact occurrence-to-mask identity validation failed")
        for key in ("sample_order", "source_window_index", "document_id", "window_offset"):
            if not np.array_equal(occurrences[key], candidate_rows[key][candidate_index]):
                raise RuntimeError(f"exact occurrence {key} identity validation failed")
    positive_window_indices = np.flatnonzero(unpacked.any(axis=1)).astype(np.int64)

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".inprogress", dir=parent))
    try:
        _atomic_npy(temporary / "old_like_gt_packed.npy", packed)
        _atomic_npy(temporary / "occurrences.npy", occurrences)
        _atomic_npy(temporary / "positive_window_indices.npy", positive_window_indices)
        _atomic_npy(temporary / "candidate_windows.npy", np.asarray(candidate_rows))
        occurrence_fields = {
            "occurrence_candidate_indices.npy": occurrences["candidate_index"],
            "occurrence_sample_orders.npy": occurrences["sample_order"],
            "occurrence_source_window_indices.npy": occurrences["source_window_index"],
            "occurrence_document_ids.npy": occurrences["document_id"],
            "occurrence_window_offsets.npy": occurrences["window_offset"],
            "occurrence_positions.npy": occurrences["position"],
            "occurrence_token_ids.npy": occurrences["token_id"],
        }
        for name, value in occurrence_fields.items():
            _atomic_npy(temporary / name, np.asarray(value))

        config_identity = summary.get("analysis_config", {})
        config_path = Path(config_identity.get("path", ""))
        observed_config = _file_identity(config_path)
        if any(
            observed_config[key] != config_identity.get(key)
            for key in ("size_bytes", "sha256")
        ):
            raise RuntimeError("frozen threshold config identity changed")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        frozen_bundle = config["candidate_thresholds"][str(bundle_level)]
        lock = {
            "schema": "cka_gt_threshold_lock_v1",
            "bundle_level": bundle_level,
            "decision_reason": decision_reason,
            "thresholds": frozen_bundle,
            "selector": summary["selector"],
            "source_exact_summary": _file_identity(summary_path),
            "source_bundle_packed": _file_identity(source_packed),
            "source_bundle_occurrences": _file_identity(source_occurrences),
            "authoritative_lineage": authoritative_lineage,
            "sealed_test_opened": False,
            "immutable_create_only": True,
        }
        _atomic_json(temporary / "threshold_lock.json", lock)
        metadata = {
            "schema": "cka_gt_locked_bundle_metadata_v1",
            "complete": True,
            "bundle_level": bundle_level,
            "axis": "candidate_window_axis",
            "axis_row_dtype": str(WINDOW_DTYPE.descr),
            "source_candidate_windows": observed_candidate,
            "locked_candidate_windows": {
                "relative_path": "candidate_windows.npy",
                "size_bytes": int((temporary / "candidate_windows.npy").stat().st_size),
                "sha256": file_sha256(temporary / "candidate_windows.npy"),
            },
            "global_window_identity": (
                "candidate row sample_order maps to the exhaustive document-window manifest; "
                "position is zero-based within the original unpadded document window"
            ),
            "layers": list(LAYERS),
            "scales": list(SCALES),
            "authoritative_lineage": authoritative_lineage,
            "selected_occurrences": int(occurrences.size),
            "positive_candidate_windows": int(positive_window_indices.size),
            "full_train_eligible_coverage": float(
                occurrences.size / int(summary["full_eligible_token_count"])
            ),
            "original_document_context_required_for_training": True,
            "loss_mask_only_at_selected_positions": True,
            "raw_hidden_stored": False,
            "sealed_test_opened": False,
        }
        _atomic_json(temporary / "metadata.json", metadata)
        output_identities = {
            path.name: {
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        validation = {
            "schema": "cka_gt_locked_bundle_validation_v1",
            "passed": True,
            "source_hashes_verified": True,
            "authoritative_model_data_census_lineage_verified": True,
            "census_worker_model_bindings_verified": 4,
            "mask_occurrence_bijection": True,
            "occurrence_window_identity_verified": True,
            "ineligible_suffix_selected": 0,
            "duplicate_occurrences": int(
                occurrences.size
                - np.unique(
                    occurrences[["candidate_index", "position"]]
                ).size
            ) if occurrences.size else 0,
            "selected_occurrences": int(occurrences.size),
            "packed_selected_bits": int(unpacked.sum()),
            "output_files_before_validation": output_identities,
        }
        if validation["duplicate_occurrences"]:
            raise RuntimeError("locked bundle contains duplicate occurrences")
        _atomic_json(temporary / "validation.json", validation)
        # os.rename is atomic within this filesystem and fails if another
        # process materialized the final directory first.
        os.rename(temporary, output_dir)
    except Exception:
        # Preserve the owned in-progress directory for forensic inspection;
        # never remove or overwrite a possibly useful partial result.
        raise
    return {
        "output_dir": str(output_dir),
        "bundle_level": bundle_level,
        "selected_occurrences": int(occurrences.size),
        "positive_candidate_windows": int(positive_window_indices.size),
        "validation": str(output_dir / "validation.json"),
    }


__all__ = [
    "BUNDLE_LEVELS",
    "CONDITION_COUNT_NAMES",
    "ExactTargetedGTWriter",
    "OCCURRENCE_DTYPE",
    "ThresholdBundle",
    "evaluate_bundle",
    "load_threshold_bundles",
    "materialize_locked_bundle",
    "unpack_mask",
]
