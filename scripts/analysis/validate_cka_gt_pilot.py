#!/usr/bin/env python3
"""Validate prepared inputs and scalar runtime artifacts for the CKA GT pilot.

The validator is read-only.  Its shallow mode verifies manifests, hashes,
Arrow schemas, atomic shard sequences, row counts, Pass-1 array shapes, and
the sealed-test contract.  ``--deep`` additionally scans every scalar row for
finite/range/shape errors and checks exact prepared-window identity plus token
and chunk coverage.  It never imports model code and never touches a GPU.

Incomplete extraction directories are errors by default.  During an active
pilot, ``--allow-incomplete`` downgrades only *missing/in-progress* conditions
to warnings; corrupt committed shards, duplicate windows, hash failures, and
invalid scalar values remain errors.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from cka_gt_pilot_windows import (  # noqa: E402
    SPLIT_NAMES,
    WINDOW_DTYPE,
    chunk_starts,
    file_sha256,
    prepare_pilot,
    validate_prepared_pilot,
)


LAYERS = tuple(range(2, 10))
SCALES = (128, 256, 512)
TOKEN_STREAM_KIND = "token_metrics_wide"
CHUNK_STREAM_KIND = "chunk_metrics_long"
RUNTIME_STREAM_KINDS = (TOKEN_STREAM_KIND, CHUNK_STREAM_KIND)
PAIRED_STREAM_KIND = "paired_token_chunk_parquet"
RUNTIME_SCHEMA_VERSION = 1
DEFAULT_HIDDEN_SIZE = 1024
RANGE_EPS = 2.0e-5

TOKEN_REQUIRED = {
    "domain",
    "split",
    "sample_order",
    "document_id",
    "window_offset",
    "position",
    "token_id",
    "eligible",
    "cosine",
    "log_r",
    "ref_rms",
    "maha_mean",
    "proto_mean",
}
TOKEN_SCALAR_FLOATS = {"maha_mean", "proto_mean"}
TOKEN_ALIASES = {
    "rel_l2": ("rel_l2", "relative_l2"),
    "sym_rel_l2": ("sym_rel_l2", "symmetric_relative_l2"),
}
TOKEN_LAYER_FLOATS = {
    "cosine",
    "rel_l2",
    "relative_l2",
    "sym_rel_l2",
    "symmetric_relative_l2",
    "log_r",
    "ref_rms",
    "maha",
    "proto",
    "cka_min_128",
    "cka_min_256",
    "s_min_128",
    "s_min_256",
    "r_min_128",
    "r_mean_128",
    "r_max_128",
    "r_min_256",
    "r_mean_256",
    "r_max_256",
    "worst_diag_ratio_128",
    "worst_diag_ratio_256",
    "s_at_worst_cka_128",
    "s_at_worst_cka_256",
    "before_old_full_mass",
    "after_old_full_mass",
    "before_old_selected_mass",
    "after_old_selected_mass",
}
TOKEN_LAYER_INTS = {
    "layer_numbers",
    "before_router_layer_numbers",
    "after_router_layer_numbers",
    "valid_cka_128",
    "valid_cka_256",
    "valid_s_128",
    "valid_s_256",
    "worst_chunk_id_128",
    "worst_chunk_id_256",
    "worst_s_chunk_id_128",
    "worst_s_chunk_id_256",
    "neg_contrib_128",
    "neg_contrib_256",
    "offdiag_warning_128",
    "offdiag_warning_256",
}
ROUTER_ID_COLUMNS = ("before_top4_id", "before_top4_ids", "after_top4_id", "after_top4_ids")
ROUTER_WEIGHT_COLUMNS = ("before_top4_weight", "after_top4_weight")
ROUTER_MASS_COLUMNS = (
    "before_old_full_mass",
    "after_old_full_mass",
    "before_old_selected_mass",
    "after_old_selected_mass",
)

CHUNK_REQUIRED = {
    "domain",
    "split",
    "sample_order",
    "document_id",
    "window_offset",
    "scale",
    "chunk_start",
    "chunk_length",
    "layer",
    "cka",
    "cka_off",
    "diag_ratio",
    "invalid_reason",
    "random_pair_invalid_reason",
    "random_pair_donor_window_uid",
}

FORBIDDEN_RAW_NAMES = {
    "hidden",
    "raw_hidden",
    "reference_hidden",
    "current_hidden",
    "before_hidden",
    "after_hidden",
    "delta_hidden",
    "hidden_states",
    "layer_output_hidden",
}


def _import_arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - production environment has Arrow
        raise RuntimeError("pyarrow is required to validate runtime shards") from error
    return pa, pq


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass
class ValidationReport:
    root: str
    allow_incomplete: bool
    deep: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.errors.append(str(message))

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def incomplete(self, message: str) -> None:
        (self.warning if self.allow_incomplete else self.error)(str(message))

    def checked(self, message: str) -> None:
        self.checks.append(str(message))

    @property
    def ok(self) -> bool:
        return not self.errors

    def serializable(self) -> dict[str, Any]:
        return {
            "schema": "cka_gt_pilot_validation_v1",
            "root": self.root,
            "allow_incomplete": self.allow_incomplete,
            "deep": self.deep,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "stats": self.stats,
        }


@dataclass
class PreparedIndex:
    config: dict[str, Any]
    windows: dict[tuple[str, str], np.ndarray]

    def row(self, domain: str, split: str, sample_order: int) -> np.void | None:
        values = self.windows.get((domain, split))
        if values is None or sample_order < 0 or sample_order >= values.size:
            return None
        row = values[int(sample_order)]
        if int(row["sample_order"]) != int(sample_order):
            return None
        return row

    @property
    def expected_keys(self) -> set[tuple[str, str, int]]:
        return {
            (domain, split, int(sample_order))
            for (domain, split), rows in self.windows.items()
            for sample_order in rows["sample_order"].tolist()
        }


def _load_prepared_index(root: Path, config: Mapping[str, Any]) -> PreparedIndex:
    windows: dict[tuple[str, str], np.ndarray] = {}
    for domain in ("code", "wiki"):
        manifest_entry = config["artifacts"][f"{domain}_manifest"]
        manifest_path = root / manifest_entry["file"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_root = manifest_path.parent
        for split in SPLIT_NAMES:
            artifact = manifest["splits"][split]["artifacts"]["windows_npy"]
            rows = np.load(manifest_root / artifact["file"], mmap_mode="r", allow_pickle=False)
            if rows.dtype != WINDOW_DTYPE:
                raise ValueError(f"unexpected window dtype: {domain}/{split}")
            windows[(domain, split)] = rows
    return PreparedIndex(dict(config), windows)


def _validate_prepared(
    root: Path,
    report: ValidationReport,
    datasets: Mapping[str, Any] | None,
) -> PreparedIndex | None:
    config_path = root / "config.json"
    if not config_path.is_file():
        report.incomplete(f"prepared config missing: {config_path}")
        return None
    try:
        config = validate_prepared_pilot(root, datasets=datasets)
        prepared = _load_prepared_index(root, config)
    except Exception as error:
        report.error(f"prepared input validation failed: {type(error).__name__}: {error}")
        return None

    policy = config.get("test_policy", {})
    required_policy = {
        "metrics_may_be_computed": True,
        "report_v1_must_not_reveal_test": True,
        "open_once_after_human_threshold_choice": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) is not expected:
            report.error(f"config test policy changed: {key}={policy.get(key)!r}")
    if config.get("layers") != list(LAYERS):
        report.error(f"prepared config layers must be {list(LAYERS)}")
    numeric = config.get("numeric_policy", {})
    if numeric.get("metric_accumulation_dtype") != "float32":
        report.error("metric accumulation dtype is not frozen to float32")
    if numeric.get("tf32_allowed") is not False:
        report.error("TF32 must be disabled for CKA metric computation")
    report.checked("prepared config, document split, window hashes, and format parity")
    report.stats["prepared_windows"] = {
        f"{domain}/{split}": int(rows.size)
        for (domain, split), rows in prepared.windows.items()
    }
    return prepared


def _forbidden_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return bool(
        normalized in FORBIDDEN_RAW_NAMES
        or ("hidden" in normalized and normalized not in {"hidden_size"})
        or normalized.endswith("_layer_output")
        or normalized.endswith("_layer_outputs")
    )


def _validate_no_raw_hidden_files(root: Path, report: ValidationReport) -> None:
    scan_roots = [
        root / "runtime",
        root / "membership",
        root / "token_metrics",
        root / "chunk_metrics",
        root / "sealed_test" / "raw",
    ]
    forbidden: list[str] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            stem = path.stem.lower().replace("-", "_")
            if (
                _forbidden_name(stem)
                and path.suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".bin", ".arrow", ".parquet"}
            ):
                forbidden.append(str(path.relative_to(root)))
    if forbidden:
        report.error(f"raw hidden artifacts are forbidden: {forbidden[:20]}")
    else:
        report.checked("no raw-hidden artifact filenames")


def _schema_list_like(pa: Any, value_type: Any) -> bool:
    return bool(
        pa.types.is_list(value_type)
        or pa.types.is_large_list(value_type)
        or pa.types.is_fixed_size_list(value_type)
    )


def _leaf_type(pa: Any, value_type: Any) -> Any:
    while _schema_list_like(pa, value_type):
        value_type = value_type.value_type
    return value_type


def _validate_stream_schema(kind: str, schema: Any, report: ValidationReport, label: str) -> None:
    pa, _ = _import_arrow()
    names = set(schema.names)
    required = TOKEN_REQUIRED if kind == TOKEN_STREAM_KIND else CHUNK_REQUIRED
    missing = sorted(required - names)
    if kind == TOKEN_STREAM_KIND:
        for canonical, aliases in TOKEN_ALIASES.items():
            if not any(alias in names for alias in aliases):
                missing.append(f"{canonical} (one of {aliases})")
    if missing:
        report.error(f"{label}: missing required {kind} columns: {missing}")
    forbidden = sorted(name for name in names if _forbidden_name(name))
    if forbidden:
        report.error(f"{label}: raw hidden columns are forbidden: {forbidden}")

    for name in ("domain", "split"):
        if name in names and not (
            pa.types.is_string(schema.field(name).type)
            or pa.types.is_large_string(schema.field(name).type)
            or pa.types.is_dictionary(schema.field(name).type)
        ):
            report.error(f"{label}: {name} must be a string/dictionary column")
    integer_names = {
        "window_uid",
        "chunk_uid",
        "sample_order",
        "source_window_index",
        "document_id",
        "window_offset",
        "window_length",
        "position",
        "document_token_offset",
        "token_id",
        "random_pair_donor_window_uid",
        "scale",
        "chunk_start",
        "chunk_length",
        "layer",
    }
    for name in names & integer_names:
        if not pa.types.is_integer(schema.field(name).type):
            report.error(f"{label}: {name} must be integer")
    if kind == TOKEN_STREAM_KIND and "eligible" in names and not pa.types.is_boolean(
        schema.field("eligible").type
    ):
        report.error(f"{label}: eligible must be boolean")

    if kind == TOKEN_STREAM_KIND:
        for name in names & TOKEN_SCALAR_FLOATS:
            if schema.field(name).type != pa.float32():
                report.error(f"{label}: scalar metric {name} dtype must be float32")
        for name in names & (TOKEN_LAYER_FLOATS | TOKEN_LAYER_INTS):
            field_type = schema.field(name).type
            if not _schema_list_like(pa, field_type):
                report.error(f"{label}: token layer column {name} is not a list type")
            elif name in TOKEN_LAYER_FLOATS and _leaf_type(pa, field_type) != pa.float32():
                report.error(f"{label}: metric {name} leaf dtype must be float32")
        for name in names & set(ROUTER_ID_COLUMNS + ROUTER_WEIGHT_COLUMNS):
            if not _schema_list_like(pa, schema.field(name).type):
                report.error(f"{label}: router column {name} is not list-like")
        for name in names & set(ROUTER_WEIGHT_COLUMNS):
            if _leaf_type(pa, schema.field(name).type) != pa.float32():
                report.error(f"{label}: router weight {name} leaf dtype must be float32")
    else:
        for name in names & {
            "cka",
            "cka_permutation",
            "cka_random_pair",
            "cka_off",
            "diag_ratio",
            "centered_variance_x",
            "centered_variance_y",
            "centered_var_x",
            "centered_var_y",
            "k_norm",
            "l_norm",
        }:
            if schema.field(name).type != pa.float32():
                report.error(f"{label}: chunk metric {name} dtype must be float32")
        for name in names & {"c_i", "c_i_off", "s_i", "r_i"}:
            field_type = schema.field(name).type
            if not _schema_list_like(pa, field_type):
                report.error(f"{label}: chunk contribution {name} is not list-like")
            elif _leaf_type(pa, field_type) != pa.float32():
                report.error(f"{label}: chunk contribution {name} leaf dtype must be float32")
        for name in names & {
            "invalid_reason",
            "t_invalid_reason",
            "random_pair_invalid_reason",
        }:
            if not (
                pa.types.is_integer(schema.field(name).type)
                or pa.types.is_string(schema.field(name).type)
                or pa.types.is_large_string(schema.field(name).type)
            ):
                report.error(f"{label}: {name} must be integer or string")
        if "offdiag_warning" in names and not pa.types.is_boolean(
            schema.field("offdiag_warning").type
        ):
            report.error(f"{label}: offdiag_warning must be boolean")


def _validate_pass2_token_extras(schema: Any, report: ValidationReport, label: str) -> None:
    names = set(schema.names)
    required = {
        "window_uid",
        "source_window_index",
        "window_length",
        "document_token_offset",
        "layer_numbers",
        "maha",
        "proto",
        "maha_mean",
        "proto_mean",
        "cka_min_128",
        "cka_min_256",
        "s_min_128",
        "s_min_256",
        "r_min_128",
        "r_mean_128",
        "r_max_128",
        "r_min_256",
        "r_mean_256",
        "r_max_256",
        "worst_diag_ratio_128",
        "worst_diag_ratio_256",
        "neg_contrib_128",
        "neg_contrib_256",
        "offdiag_warning_128",
        "offdiag_warning_256",
        "worst_chunk_id_128",
        "worst_chunk_id_256",
        "s_at_worst_cka_128",
        "s_at_worst_cka_256",
        "worst_s_chunk_id_128",
        "worst_s_chunk_id_256",
        "valid_cka_128",
        "valid_cka_256",
        "valid_s_128",
        "valid_s_256",
        "before_router_layer_numbers",
        "after_router_layer_numbers",
        "before_top4_weight",
        "after_top4_weight",
        "before_old_full_mass",
        "after_old_full_mass",
        "before_old_selected_mass",
        "after_old_selected_mass",
    }
    missing = sorted(required - names)
    for checkpoint in ("before", "after"):
        if not any(name in names for name in (f"{checkpoint}_top4_id", f"{checkpoint}_top4_ids")):
            missing.append(f"{checkpoint}_top4_id(s)")
    if missing:
        report.error(f"{label}: pass2 token schema missing required diagnostics: {missing}")


def _validate_pass2_chunk_extras(schema: Any, report: ValidationReport, label: str) -> None:
    names = set(schema.names)
    required = {
        "chunk_uid",
        "window_uid",
        "cka_permutation",
        "cka_random_pair",
        "random_pair_invalid_reason",
        "random_pair_donor_window_uid",
        "k_norm",
        "l_norm",
        "t_invalid_reason",
        "offdiag_warning",
        "c_i",
        "c_i_off",
        "s_i",
        "r_i",
    }
    missing = sorted(required - names)
    for side, aliases in {
        "before_centered_variance": (
            "centered_var_x",
            "centered_variance_x",
            "before_centered_variance",
        ),
        "after_centered_variance": (
            "centered_var_y",
            "centered_variance_y",
            "after_centered_variance",
        ),
    }.items():
        if not any(alias in names for alias in aliases):
            missing.append(f"{side} (one of {aliases})")
    if missing:
        report.error(f"{label}: pass2 chunk schema missing null/variance diagnostics: {missing}")


def _array_numpy(batch: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    array = batch.column(batch.schema.get_field_index(name))
    values = array.to_numpy(zero_copy_only=False)
    return values.astype(dtype, copy=False) if dtype is not None else values


def _list_numpy(batch: Any, name: str, width: int, dtype: Any) -> np.ndarray:
    array = batch.column(batch.schema.get_field_index(name))
    # Production diagnostics are fixed-size lists.  Flatten their primitive
    # Arrow buffer directly instead of constructing Python objects per scalar.
    # Nullable/variable legacy arrays retain the original row-wise fallback.
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    if getattr(array.type, "list_size", None) == width and int(array.null_count) == 0:
        flat = array.flatten()
        if int(flat.null_count) == 0 and getattr(flat.type, "value_type", None) is None:
            values = flat.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
            if values.size == len(array) * width:
                return values.reshape(len(array), width)
    rows = array.to_pylist()
    result = np.empty((len(rows), width), dtype=dtype)
    for index, row in enumerate(rows):
        flat = np.asarray(row).reshape(-1) if row is not None else np.empty(0)
        if flat.size != width:
            raise ValueError(f"{name} row {index} width {flat.size}, expected {width}")
        result[index] = flat.astype(dtype, copy=False)
    return result


def _nested_router_numpy(batch: Any, name: str, dtype: Any) -> np.ndarray:
    array = batch.column(batch.schema.get_field_index(name))
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    if getattr(array.type, "list_size", None) == 8 and int(array.null_count) == 0:
        inner = array.flatten()
        if getattr(inner.type, "list_size", None) == 4 and int(inner.null_count) == 0:
            flat = inner.flatten()
            if int(flat.null_count) == 0 and getattr(flat.type, "value_type", None) is None:
                values = flat.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
                if values.size == len(array) * 32:
                    return values.reshape(len(array), 8, 4)
    rows = array.to_pylist()
    result = np.empty((len(rows), 8, 4), dtype=dtype)
    for index, row in enumerate(rows):
        values = np.asarray(row)
        if values.size != 32:
            raise ValueError(f"{name} row {index} has {values.size} entries, expected 8x4")
        result[index] = values.reshape(8, 4).astype(dtype, copy=False)
    return result


def _strings(batch: Any, name: str) -> np.ndarray:
    return np.asarray(
        batch.column(batch.schema.get_field_index(name)).to_pylist(), dtype=object
    )


def _finite(values: np.ndarray, name: str, report: ValidationReport, label: str) -> bool:
    ok = bool(np.isfinite(values).all())
    if not ok:
        report.error(f"{label}: {name} contains NaN/Inf where finite values are required")
    return ok


@dataclass
class TokenCoverage:
    prepared: PreparedIndex | None
    report: ValidationReport
    by_window: dict[tuple[str, str, int], tuple[int, int]] = field(default_factory=dict)
    # value = (position bitset, observed row count)
    identity: dict[tuple[str, str, int], tuple[int, int, int]] = field(default_factory=dict)

    def add_batch(self, batch: Any, label: str) -> set[tuple[str, str, int]]:
        domains = _strings(batch, "domain")
        splits = _strings(batch, "split")
        orders = _array_numpy(batch, "sample_order", np.int64)
        document_ids = _array_numpy(batch, "document_id", np.int64)
        offsets = _array_numpy(batch, "window_offset", np.int64)
        positions = _array_numpy(batch, "position", np.int64)
        eligible = _array_numpy(batch, "eligible", bool)
        if "window_length" in batch.schema.names:
            lengths = _array_numpy(batch, "window_length", np.int64)
        else:
            lengths = np.full(len(batch), -1, dtype=np.int64)

        groups: dict[tuple[str, str, int], list[int]] = {}
        for index, (domain, split, order) in enumerate(zip(domains, splits, orders)):
            key = (str(domain), str(split), int(order))
            groups.setdefault(key, []).append(index)
        for key, indices_list in groups.items():
            indices = np.asarray(indices_list, dtype=np.int64)
            domain, split, order = key
            if domain not in ("code", "wiki") or split not in SPLIT_NAMES:
                self.report.error(f"{label}: invalid domain/split in token row: {key[:2]}")
                continue
            prepared_row = self.prepared.row(domain, split, order) if self.prepared else None
            if self.prepared and prepared_row is None:
                self.report.error(f"{label}: token window not present in prepared manifest: {key}")
                continue
            expected_document = int(prepared_row["document_id"]) if prepared_row is not None else int(document_ids[indices[0]])
            expected_offset = int(prepared_row["window_offset"]) if prepared_row is not None else int(offsets[indices[0]])
            expected_length = int(prepared_row["window_length"]) if prepared_row is not None else int(lengths[indices[0]])
            if np.any(document_ids[indices] != expected_document) or np.any(offsets[indices] != expected_offset):
                self.report.error(f"{label}: document identity mismatch for {key}")
            if np.any(lengths[indices] >= 0) and np.any(lengths[indices] != expected_length):
                self.report.error(f"{label}: window length mismatch for {key}")
            pos = positions[indices]
            if np.any(pos < 0) or np.any(pos >= expected_length):
                self.report.error(f"{label}: out-of-range token position for {key}")
                continue
            if np.unique(pos).size != pos.size:
                self.report.error(f"{label}: duplicate token positions within batch for {key}")
            if prepared_row is not None:
                expected_eligible_count = int(prepared_row["eligible_token_count"])
                expected_eligible = pos < expected_eligible_count
                if not np.array_equal(eligible[indices], expected_eligible):
                    self.report.error(f"{label}: eligible mask differs from prepared grid for {key}")
            mask = np.zeros(expected_length, dtype=np.uint8)
            mask[pos] = 1
            bitset = int.from_bytes(np.packbits(mask, bitorder="little").tobytes(), "little")
            previous_bits, previous_count = self.by_window.get(key, (0, 0))
            if previous_bits & bitset:
                self.report.error(f"{label}: duplicate token coverage across shards for {key}")
            self.by_window[key] = (previous_bits | bitset, previous_count + int(pos.size))
            prior_identity = self.identity.setdefault(
                key, (expected_document, expected_offset, expected_length)
            )
            if prior_identity != (expected_document, expected_offset, expected_length):
                self.report.error(f"{label}: inconsistent repeated window identity for {key}")
        return set(groups)

    def finalize(self) -> set[tuple[str, str, int]]:
        for key, (bits, count) in self.by_window.items():
            expected_length = self.identity[key][2]
            expected_bits = (1 << expected_length) - 1
            if count != expected_length or bits != expected_bits:
                self.report.error(
                    f"token coverage incomplete for {key}: rows={count}, expected={expected_length}"
                )
        return set(self.by_window)


def _hash64(scale: np.ndarray, start: np.ndarray, layer: np.ndarray, salt: int) -> np.ndarray:
    values = (
        scale.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
        ^ start.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ layer.astype(np.uint64) * np.uint64(0x165667B19E3779F9)
        ^ np.uint64(salt)
    )
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _expected_chunk_fingerprint(length: int) -> tuple[int, int, int]:
    rows: list[tuple[int, int, int]] = []
    for scale, stride in ((128, 64), (256, 128)):
        for start in chunk_starts(length, scale, stride).tolist():
            rows.extend((scale, int(start), layer) for layer in LAYERS)
    rows.extend((512, 0, layer) for layer in LAYERS)
    values = np.asarray(rows, dtype=np.uint64)
    hashes = _hash64(values[:, 0], values[:, 1], values[:, 2], 0xA11CE)
    return int(values.shape[0]), int(hashes.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(hashes))


@dataclass
class ChunkCoverage:
    prepared: PreparedIndex | None
    report: ValidationReport
    summary: dict[tuple[str, str, int], tuple[int, int, int]] = field(default_factory=dict)

    def add_batch(self, batch: Any, label: str) -> None:
        domains = _strings(batch, "domain")
        splits = _strings(batch, "split")
        orders = _array_numpy(batch, "sample_order", np.int64)
        document_ids = _array_numpy(batch, "document_id", np.int64)
        offsets = _array_numpy(batch, "window_offset", np.int64)
        scales = _array_numpy(batch, "scale", np.int64)
        starts = _array_numpy(batch, "chunk_start", np.int64)
        lengths = _array_numpy(batch, "chunk_length", np.int64)
        layers = _array_numpy(batch, "layer", np.int64)
        if np.any(~np.isin(layers, LAYERS)):
            self.report.error(f"{label}: chunk layers are not exactly within 2..9")
        if np.any(~np.isin(scales, SCALES)):
            self.report.error(f"{label}: unsupported chunk scale")

        hashes = _hash64(scales, starts, layers, 0xA11CE)
        groups: dict[tuple[str, str, int], list[int]] = {}
        for index, (domain, split, order) in enumerate(zip(domains, splits, orders)):
            groups.setdefault((str(domain), str(split), int(order)), []).append(index)
        for key, indices_list in groups.items():
            indices = np.asarray(indices_list, dtype=np.int64)
            domain, split, order = key
            prepared_row = self.prepared.row(domain, split, order) if self.prepared else None
            if self.prepared and prepared_row is None:
                self.report.error(f"{label}: chunk window not in prepared manifest: {key}")
                continue
            window_length = int(prepared_row["window_length"]) if prepared_row is not None else 512
            if prepared_row is not None and (
                np.any(document_ids[indices] != int(prepared_row["document_id"]))
                or np.any(offsets[indices] != int(prepared_row["window_offset"]))
            ):
                self.report.error(f"{label}: chunk document identity mismatch for {key}")
            local_scales = scales[indices]
            local_starts = starts[indices]
            local_lengths = lengths[indices]
            bad_128 = (local_scales == 128) & (
                (local_lengths != 128)
                | (local_starts % 64 != 0)
                | (local_starts + 128 > window_length)
            )
            bad_256 = (local_scales == 256) & (
                (local_lengths != 256)
                | (local_starts % 128 != 0)
                | (local_starts + 256 > window_length)
            )
            bad_512 = (local_scales == 512) & (
                (local_starts != 0) | (local_lengths != window_length)
            )
            if np.any(bad_128 | bad_256 | bad_512):
                self.report.error(f"{label}: non-canonical fixed chunk grid for {key}")
            count, total, xor = self.summary.get(key, (0, 0, 0))
            local_hash = hashes[indices]
            self.summary[key] = (
                count + int(indices.size),
                (total + int(local_hash.sum(dtype=np.uint64))) & ((1 << 64) - 1),
                xor ^ int(np.bitwise_xor.reduce(local_hash)),
            )

    def finalize(self) -> None:
        for key, actual in self.summary.items():
            prepared_row = self.prepared.row(*key) if self.prepared else None
            length = int(prepared_row["window_length"]) if prepared_row is not None else 512
            expected = _expected_chunk_fingerprint(length)
            if actual != expected:
                self.report.error(
                    f"chunk layer/grid coverage mismatch for {key}: {actual} != {expected}"
                )


def _validate_token_batch(batch: Any, report: ValidationReport, label: str) -> None:
    names = set(batch.schema.names)
    for name in names & TOKEN_SCALAR_FLOATS:
        values = _array_numpy(batch, name, np.float32)
        if _finite(values, name, report, label) and np.any(values < -RANGE_EPS):
            report.error(f"{label}: {name} contains negative values")
    layer_values: dict[str, np.ndarray] = {}
    for name in names & TOKEN_LAYER_FLOATS:
        try:
            values = _list_numpy(batch, name, 8, np.float32)
        except Exception as error:
            report.error(f"{label}: bad {name} shape: {error}")
            continue
        layer_values[name] = values
        if name.startswith(
            (
                "cka_min_",
                "s_min_",
                "r_min_",
                "r_mean_",
                "r_max_",
                "worst_diag_ratio_",
                "s_at_worst_cka_",
            )
        ):
            # Missing CKA/T measurements are represented by NaN for ineligible
            # tokens.  Other infinities are never legal.
            if np.isinf(values).any():
                report.error(f"{label}: {name} contains infinity")
        else:
            _finite(values, name, report, label)
    integer_layers: dict[str, np.ndarray] = {}
    for name in names & TOKEN_LAYER_INTS:
        try:
            integer_layers[name] = _list_numpy(batch, name, 8, np.int64)
        except Exception as error:
            report.error(f"{label}: bad {name} shape: {error}")

    for name in ("layer_numbers", "before_router_layer_numbers", "after_router_layer_numbers"):
        values = integer_layers.get(name)
        if values is not None and not np.all(values == np.asarray(LAYERS, dtype=np.int64)):
            report.error(f"{label}: {name} is not exactly layer 2..9")
    for name in (
        "valid_cka_128",
        "valid_cka_256",
        "valid_s_128",
        "valid_s_256",
        "neg_contrib_128",
        "neg_contrib_256",
        "offdiag_warning_128",
        "offdiag_warning_256",
    ):
        values = integer_layers.get(name)
        if values is not None and np.any((values != 0) & (values != 1)):
            report.error(f"{label}: {name} contains values outside {{0,1}}")
    # Runtime stores the fixed-grid chunk *start offset*, not a zero-based
    # ordinal.  Accept only offsets that can occur in a full 512-token window;
    # shorter tail windows are checked more tightly by the coverage tracker.
    valid_chunk_starts = {
        128: np.asarray([-1, 0, 64, 128, 192, 256, 320, 384], dtype=np.int64),
        256: np.asarray([-1, 0, 128, 256], dtype=np.int64),
    }
    for scale, allowed in valid_chunk_starts.items():
        for prefix in ("worst_chunk_id", "worst_s_chunk_id"):
            name = f"{prefix}_{scale}"
            values = integer_layers.get(name)
            if values is not None and np.any(~np.isin(values, allowed)):
                observed = np.unique(values[~np.isin(values, allowed)]).tolist()
                report.error(
                    f"{label}: {name} contains non-grid chunk starts {observed}"
                )

    cosine = layer_values.get("cosine")
    if cosine is not None and np.any((cosine < -1.0 - RANGE_EPS) | (cosine > 1.0 + RANGE_EPS)):
        report.error(f"{label}: cosine outside [-1,1]")
    for name in ("cka_min_128", "cka_min_256"):
        values = layer_values.get(name)
        if values is not None:
            finite = np.isfinite(values)
            if np.any(
                (values[finite] < -RANGE_EPS) | (values[finite] > 1.0 + RANGE_EPS)
            ):
                report.error(f"{label}: {name} outside [0,1]")
    for name in (
        "r_min_128",
        "r_mean_128",
        "r_max_128",
        "r_min_256",
        "r_mean_256",
        "r_max_256",
    ):
        values = layer_values.get(name)
        if values is not None:
            finite = np.isfinite(values)
            if np.any(
                (values[finite] < -1.0 - RANGE_EPS)
                | (values[finite] > 1.0 + RANGE_EPS)
            ):
                report.error(f"{label}: {name} outside [-1,1]")
    for name in ("rel_l2", "relative_l2", "ref_rms", "maha", "proto"):
        values = layer_values.get(name)
        if values is not None and np.any(values < -RANGE_EPS):
            report.error(f"{label}: {name} contains negative values")
    for name in ("sym_rel_l2", "symmetric_relative_l2"):
        values = layer_values.get(name)
        if values is not None and np.any((values < -RANGE_EPS) | (values > 2.0 + RANGE_EPS)):
            report.error(f"{label}: {name} outside [0,2]")

    for name in names & set(ROUTER_ID_COLUMNS):
        try:
            values = _nested_router_numpy(batch, name, np.int64)
            if np.any((values < 0) | (values >= 16)):
                report.error(f"{label}: {name} expert ID outside [0,15]")
            if np.any(np.diff(np.sort(values, axis=2), axis=2) == 0):
                report.error(f"{label}: {name} contains duplicate top-4 IDs")
        except Exception as error:
            report.error(f"{label}: bad {name} shape: {error}")
    for name in names & set(ROUTER_WEIGHT_COLUMNS):
        try:
            values = _nested_router_numpy(batch, name, np.float32)
            if _finite(values, name, report, label) and np.any(
                (values < -RANGE_EPS) | (values > 1.0 + RANGE_EPS)
            ):
                report.error(f"{label}: {name} outside [0,1]")
        except Exception as error:
            report.error(f"{label}: bad {name} shape: {error}")
    for name in names & set(ROUTER_MASS_COLUMNS):
        values = layer_values.get(name)
        if values is not None and np.any((values < -RANGE_EPS) | (values > 1.0 + RANGE_EPS)):
            report.error(f"{label}: {name} outside [0,1]")


def _validate_chunk_batch(batch: Any, report: ValidationReport, label: str) -> None:
    names = set(batch.schema.names)
    reasons = _strings(batch, "invalid_reason")
    valid = np.asarray(
        [reason in (None, "", "valid", "none", "0", 0) for reason in reasons], dtype=bool
    )
    if "t_invalid_reason" in names:
        t_reasons = _strings(batch, "t_invalid_reason")
        t_valid = np.asarray(
            [reason in (None, "", "valid", "none", "0", 0) for reason in t_reasons],
            dtype=bool,
        )
    else:
        t_valid = valid.copy()
    for name in (
        "cka",
        "cka_permutation",
        "cka_random_pair",
        "cka_off",
        "diag_ratio",
        "centered_variance_x",
        "centered_variance_y",
        "centered_var_x",
        "centered_var_y",
    ):
        if name not in names:
            continue
        values = _array_numpy(batch, name, np.float64)
        is_null = name in ("cka_permutation", "cka_random_pair")
        if is_null and np.isinf(values[valid]).any():
            report.error(f"{label}: {name} contains infinity")
        elif not is_null and np.any(~np.isfinite(values[valid])):
            report.error(f"{label}: valid chunk has non-finite {name}")
        finite_valid = valid & np.isfinite(values)
        if name in ("cka", "cka_permutation", "cka_random_pair") and np.any(
            (values[finite_valid] < -RANGE_EPS)
            | (values[finite_valid] > 1.0 + RANGE_EPS)
        ):
            report.error(f"{label}: {name} outside [0,1]")
        if name.startswith(("centered_variance", "centered_var")) and np.any(
            values[valid] < -RANGE_EPS
        ):
            report.error(f"{label}: negative centered variance")
    layers = _array_numpy(batch, "layer", np.int64)
    if np.any(~np.isin(layers, LAYERS)):
        report.error(f"{label}: layer outside 2..9")

    if "invalid_reason" in names and np.issubdtype(
        _array_numpy(batch, "invalid_reason").dtype, np.integer
    ):
        values = _array_numpy(batch, "invalid_reason", np.int64)
        if np.any((values < 0) | (values > 7)):
            report.error(f"{label}: invalid_reason has unsupported bit(s)")
    if "t_invalid_reason" in names and np.issubdtype(
        _array_numpy(batch, "t_invalid_reason").dtype, np.integer
    ):
        values = _array_numpy(batch, "t_invalid_reason", np.int64)
        if np.any((values < 0) | (values > 3)):
            report.error(f"{label}: t_invalid_reason has unsupported bit(s)")
    if {
        "random_pair_invalid_reason",
        "random_pair_donor_window_uid",
        "window_uid",
        "cka_random_pair",
    }.issubset(names):
        pair_reasons = _array_numpy(
            batch, "random_pair_invalid_reason", np.int64
        )
        donor_uids = _array_numpy(
            batch, "random_pair_donor_window_uid", np.int64
        )
        target_uids = _array_numpy(batch, "window_uid", np.int64)
        pair_cka = _array_numpy(batch, "cka_random_pair", np.float64)
        if np.any((pair_reasons < 0) | (pair_reasons > 3)):
            report.error(f"{label}: random_pair_invalid_reason outside [0,3]")
        pair_valid = pair_reasons == 0
        if np.any(~np.isfinite(pair_cka[pair_valid])):
            report.error(f"{label}: valid random-pair null has non-finite CKA")
        if np.any(donor_uids[pair_valid] < 0):
            report.error(f"{label}: valid random-pair null omits donor UID")
        if np.any(donor_uids[pair_valid] == target_uids[pair_valid]):
            report.error(f"{label}: random-pair null contains a self donor")
        no_donor = np.isin(pair_reasons, (1, 2))
        if np.any(donor_uids[no_donor] != -1) or np.any(
            np.isfinite(pair_cka[no_donor])
        ):
            report.error(
                f"{label}: no-donor random-pair reason must store donor=-1 and CKA=NaN"
            )
        metric_invalid = pair_reasons == 3
        if np.any(donor_uids[metric_invalid] < 0):
            report.error(
                f"{label}: metric-invalid random pair must retain its donor UID"
            )

    if not {"c_i", "c_i_off", "s_i", "r_i"}.issubset(names):
        return
    chunk_lengths = _array_numpy(batch, "chunk_length", np.int64)
    cka = _array_numpy(batch, "cka", np.float64)
    cka_off = _array_numpy(batch, "cka_off", np.float64)
    # These four ragged columns dominate production validation (billions of
    # scalar values).  Converting every row through ``to_pylist`` and looping
    # in Python made the 1,000+1,000-window smoke take minutes and projected to
    # hours at full scale.  Arrow offsets let us preserve the exact same
    # all-row identities with vectorized prefix sums.
    contribution_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("c_i", "c_i_off", "s_i", "r_i"):
        array = batch.column(batch.schema.get_field_index(name))
        try:
            lengths = np.asarray(array.value_lengths().to_numpy(zero_copy_only=False), dtype=np.int64)
            offsets = np.empty(lengths.size + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(lengths, out=offsets[1:])
            flat = np.asarray(
                array.flatten().to_numpy(zero_copy_only=False), dtype=np.float64
            )
        except Exception as error:
            report.error(f"{label}: cannot flatten contribution {name}: {error}")
            return
        if flat.size != int(offsets[-1]):
            report.error(
                f"{label}: flattened contribution {name} size mismatch: "
                f"{flat.size} != {int(offsets[-1])}"
            )
            return
        bad_length = lengths != chunk_lengths
        if np.any(bad_length):
            indices = np.flatnonzero(bad_length)
            preview = indices[:8].tolist()
            report.error(
                f"{label}: contribution length mismatch for {name} at "
                f"{indices.size} row(s), first={preview}"
            )
        if np.isinf(flat).any():
            report.error(f"{label}: {name} contains infinity")
        contribution_arrays[name] = (flat, offsets)

    def row_sum(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        cumulative = np.empty(values.size + 1, dtype=np.float64)
        cumulative[0] = 0.0
        np.cumsum(values, dtype=np.float64, out=cumulative[1:])
        return cumulative[offsets[1:]] - cumulative[offsets[:-1]]

    def row_nonfinite(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        bad = (~np.isfinite(values)).astype(np.int64, copy=False)
        cumulative = np.empty(bad.size + 1, dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(bad, dtype=np.int64, out=cumulative[1:])
        return cumulative[offsets[1:]] - cumulative[offsets[:-1]]

    c_values, c_offsets = contribution_arrays["c_i"]
    co_values, co_offsets = contribution_arrays["c_i_off"]
    s_values, s_offsets = contribution_arrays["s_i"]
    r_values, _ = contribution_arrays["r_i"]

    c_nonfinite = row_nonfinite(c_values, c_offsets)
    co_nonfinite = row_nonfinite(co_values, co_offsets)
    bad_valid = valid & ((c_nonfinite != 0) | (co_nonfinite != 0))
    if np.any(bad_valid):
        indices = np.flatnonzero(bad_valid)
        report.error(
            f"{label}: valid chunk has non-finite c_i/c_i_off at "
            f"{indices.size} row(s), first={indices[:8].tolist()}"
        )

    finite_identity = valid & (c_nonfinite == 0) & (co_nonfinite == 0)
    c_sums = row_sum(c_values, c_offsets)
    co_sums = row_sum(co_values, co_offsets)
    bad_c = finite_identity & ~np.isclose(c_sums, cka, atol=1e-4, rtol=1e-4)
    bad_co = finite_identity & ~np.isclose(co_sums, cka_off, atol=1e-4, rtol=1e-4)
    if np.any(bad_c):
        indices = np.flatnonzero(bad_c)
        report.error(
            f"{label}: sum(c_i) != CKA at {indices.size} row(s), "
            f"first={indices[:8].tolist()}"
        )
    if np.any(bad_co):
        indices = np.flatnonzero(bad_co)
        report.error(
            f"{label}: sum(c_i_off) != CKA_off at {indices.size} row(s), "
            f"first={indices[:8].tolist()}"
        )

    s_nonfinite = row_nonfinite(s_values, s_offsets)
    bad_s_finite = t_valid & (s_nonfinite != 0)
    if np.any(bad_s_finite):
        indices = np.flatnonzero(bad_s_finite)
        report.error(
            f"{label}: T-valid chunk has non-finite s_i at {indices.size} "
            f"row(s), first={indices[:8].tolist()}"
        )
    finite_s_rows = t_valid & (s_nonfinite == 0)
    s_means = row_sum(s_values, s_offsets) / np.maximum(chunk_lengths, 1)
    bad_s_mean = finite_s_rows & ~np.isclose(s_means, 1.0, atol=1e-4, rtol=1e-4)
    if np.any(bad_s_mean):
        indices = np.flatnonzero(bad_s_mean)
        report.error(
            f"{label}: mean(s_i) != 1 at {indices.size} row(s), "
            f"first={indices[:8].tolist()}"
        )

    finite_r = r_values[np.isfinite(r_values)]
    if np.any((finite_r < -1.0 - RANGE_EPS) | (finite_r > 1.0 + RANGE_EPS)):
        report.error(f"{label}: r_i outside [-1,1]")


def _recognized_runtime_manifests(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    runtime_root = root / "runtime"
    if not runtime_root.exists():
        return manifests
    for path in sorted(runtime_root.rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("stream_kind") in RUNTIME_STREAM_KINDS:
            manifests.append((path, payload))
    for path in sorted(runtime_root.rglob("progress.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("stream_kind") == PAIRED_STREAM_KIND:
            manifests.append((path, payload))
    return manifests


def _resolve_inside_pilot(
    root: Path,
    value: str | os.PathLike[str],
    *,
    relative_to: Path,
    report: ValidationReport,
    label: str,
) -> Path | None:
    """Resolve a manifest-owned path without ever following it outside the pilot."""

    try:
        raw = Path(value)
        resolved = (raw if raw.is_absolute() else relative_to / raw).resolve()
        resolved.relative_to(root.resolve())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        report.error(f"{label}: artifact path escapes pilot root: {value!r} ({error})")
        return None
    return resolved


def _paired_recorded_dir(
    root: Path,
    worker_root: Path,
    payload: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    *,
    directory_key: str,
    artifact_key: str,
    default_name: str,
    report: ValidationReport,
    label: str,
) -> Path | None:
    """Resolve a paired stream directory, including pre-directory-field logs."""

    raw_directory = payload.get(directory_key)
    if raw_directory is None:
        metadata = payload.get("metadata", {})
        if isinstance(metadata, Mapping):
            raw_directory = metadata.get(directory_key)
    if raw_directory is not None:
        # Canonical progress paths are rooted at the pilot.  Production writes
        # absolute paths, while root-relative paths remain portable.
        return _resolve_inside_pilot(
            root,
            raw_directory,
            relative_to=root,
            report=report,
            label=f"{label}/{directory_key}",
        )

    # Backward compatibility: the original writer omitted token_dir/chunk_dir
    # but wrote absolute artifact paths.  Infer a directory only when every
    # committed record agrees; otherwise use the local worker default.
    absolute_parents = {
        Path(str(record.get(artifact_key))).parent.resolve()
        for record in shards
        if record.get(artifact_key) is not None
        and Path(str(record.get(artifact_key))).is_absolute()
    }
    if len(absolute_parents) > 1:
        report.error(f"{label}: {artifact_key} records disagree on their directory")
        return None
    inferred = next(iter(absolute_parents), worker_root / default_name)
    return _resolve_inside_pilot(
        root,
        inferred,
        relative_to=worker_root,
        report=report,
        label=f"{label}/{directory_key}",
    )


def _paired_artifact_path(
    root: Path,
    directory: Path,
    recorded: Any,
    *,
    report: ValidationReport,
    label: str,
) -> Path | None:
    if not isinstance(recorded, str) or not recorded:
        report.error(f"{label}: artifact filename is missing")
        return None
    path = _resolve_inside_pilot(
        root,
        recorded,
        relative_to=directory,
        report=report,
        label=label,
    )
    if path is not None and path.parent != directory.resolve():
        report.error(
            f"{label}: artifact parent {path.parent} differs from recorded directory {directory}"
        )
        return None
    return path


def _expected_paired_stream(
    root: Path,
    progress_path: Path,
    payload: Mapping[str, Any],
    prepared: PreparedIndex | None,
    report: ValidationReport,
    label: str,
) -> dict[str, Any] | None:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        report.error(f"{label}: paired metadata is not an object")
        return None
    if prepared is not None:
        if metadata.get("source_dataset_identity") != prepared.config.get(
            "source_dataset_identity"
        ):
            report.error(f"{label}: runtime source_dataset_identity differs from config")
        if metadata.get("checkpoint_identity") != prepared.config.get(
            "checkpoint_identity"
        ):
            report.error(f"{label}: runtime checkpoint_identity differs from config")
    stream = metadata.get("metric_stream")
    if stream not in ("open", "test"):
        report.error(f"{label}: metric_stream must be open/test, got {stream!r}")
        return None
    domain = metadata.get("domain")
    requested_split = metadata.get("requested_split")
    if domain not in ("code", "wiki") or requested_split not in (
        "calibration",
        "selection",
        "test",
        "all",
    ):
        report.error(f"{label}: invalid domain/requested_split metadata")
        return None
    requested_names = (
        ("calibration", "selection", "test")
        if requested_split == "all"
        else (str(requested_split),)
    )
    expected_splits = (
        [name for name in requested_names if name != "test"]
        if stream == "open"
        else [name for name in requested_names if name == "test"]
    )
    if not expected_splits:
        report.error(f"{label}: {stream} stream has no split in request {requested_split}")
    if metadata.get("stream_splits") != expected_splits:
        report.error(
            f"{label}: stream_splits={metadata.get('stream_splits')!r}, "
            f"expected={expected_splits!r}"
        )
    if metadata.get("test_metrics_sealed") is not (stream == "test"):
        report.error(f"{label}: test_metrics_sealed disagrees with metric_stream")

    worker_index = int(metadata.get("worker_index", -1))
    worker_count = int(metadata.get("worker_count", 0))
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        report.error(f"{label}: invalid worker index/count")
        return None
    expected_progress = (
        root
        / "runtime"
        / "pass2"
        / str(domain)
        / str(requested_split)
        / f"worker_{worker_index:03d}"
        / "paired_progress"
        / str(stream)
        / "progress.json"
    ).resolve()
    if progress_path.resolve() != expected_progress:
        report.error(
            f"{label}: non-canonical progress path; expected "
            f"{expected_progress.relative_to(root)}"
        )

    label_name = f"{domain}_{requested_split}_{stream}_worker_{worker_index:03d}"
    if stream == "open":
        token_dir = root / "token_metrics" / label_name
        chunk_dir = root / "chunk_metrics" / label_name
    else:
        token_dir = root / "sealed_test" / "raw" / "token_metrics" / label_name
        chunk_dir = root / "sealed_test" / "raw" / "chunk_metrics" / label_name

    result = {
        "stream": stream,
        "domain": str(domain),
        "splits": expected_splits,
        "token_dir": token_dir.resolve(),
        "chunk_dir": chunk_dir.resolve(),
        "window_uids": set(),
        "window_keys": set(),
        "token_count": 0,
        "chunk_count": 0,
    }
    pair_policy = metadata.get("random_pair_null")
    required_pair_policy = {
        "policy_version": "deterministic_full_after_donor_cache_v1",
        "batch_ge_2": "within_batch_seeded_cyclic_derangement_no_fixed_points",
        "singleton": (
            "sha256(window_uid,scale,chunk_start)_indexed_nonself_donor_from_"
            "first_8_full_after_windows"
        ),
        "donor_cache_scope": "per_split_per_worker",
        "donor_cache_size": 8,
        "donor_cache_storage": "memory_only_rebuilt_by_forward_on_resume",
        "reason_codes": {
            "0": "valid",
            "1": "singleton_no_donor_cache",
            "2": "singleton_no_nonself_full_window_donor",
            "3": "paired_cka_metric_invalid",
        },
        "coverage_columns": [
            "random_pair_invalid_reason",
            "random_pair_donor_window_uid",
            "cka_random_pair",
        ],
        "selector_dependency": False,
    }
    if not isinstance(pair_policy, Mapping):
        report.error(f"{label}: random_pair_null policy metadata is missing")
    else:
        for key, expected in required_pair_policy.items():
            if pair_policy.get(key) != expected:
                report.error(
                    f"{label}: random_pair_null {key}={pair_policy.get(key)!r}, "
                    f"expected={expected!r}"
                )
    if prepared is None:
        return result
    cap = int(metadata.get("max_windows", 0) or 0)
    domain_id = {"code": 0, "wiki": 1}[str(domain)]
    split_ids = {"calibration": 0, "selection": 1, "test": 2}

    expected_donor_candidates: dict[str, list[int]] = {}
    for split in requested_names:
        candidate_rows = prepared.windows[(str(domain), split)]
        quotient, remainder = divmod(int(candidate_rows.shape[0]), worker_count)
        start = worker_index * quotient + min(worker_index, remainder)
        count = quotient + int(worker_index < remainder)
        candidate_rows = candidate_rows[start : start + count]
        if cap > 0:
            candidate_rows = candidate_rows[:cap]
        candidate_rows = candidate_rows[candidate_rows["window_length"] == 512]
        if candidate_rows.size:
            candidate_rows = candidate_rows[
                np.argsort(
                    candidate_rows["sample_order"].astype(np.int64), kind="stable"
                )
            ]
        candidate_rows = candidate_rows[:8]
        expected_donor_candidates[split] = (
            np.int64(domain_id) * np.int64(1_000_000_000)
            + np.int64(split_ids[split]) * np.int64(100_000_000)
            + candidate_rows["sample_order"].astype(np.int64)
        ).tolist()
    if isinstance(pair_policy, Mapping) and pair_policy.get(
        "candidate_window_uids_by_split"
    ) != expected_donor_candidates:
        report.error(
            f"{label}: random-pair donor candidate UIDs differ from prepared worker partition"
        )

    for split in expected_splits:
        rows = prepared.windows[(str(domain), split)]
        quotient, remainder = divmod(int(rows.shape[0]), worker_count)
        start = worker_index * quotient + min(worker_index, remainder)
        count = quotient + int(worker_index < remainder)
        rows = rows[start : start + count]
        if cap > 0:
            rows = rows[:cap]
        orders = rows["sample_order"].astype(np.int64)
        result["window_uids"].update(
            (
                np.int64(domain_id) * np.int64(1_000_000_000)
                + np.int64(split_ids[split]) * np.int64(100_000_000)
                + orders
            ).tolist()
        )
        result["window_keys"].update(
            (str(domain), split, int(order)) for order in orders.tolist()
        )
        result["token_count"] += int(rows["window_length"].sum(dtype=np.int64))
        result["chunk_count"] += sum(
            _expected_chunk_fingerprint(int(length))[0]
            for length in rows["window_length"].tolist()
        )
    if int(metadata.get("stream_window_count", -1)) != len(result["window_uids"]):
        report.error(f"{label}: stream_window_count differs from prepared worker partition")
    if int(metadata.get("stream_token_count", -1)) != result["token_count"]:
        report.error(f"{label}: stream_token_count differs from prepared worker partition")
    return result


def _validate_paired_progress(
    root: Path,
    progress_path: Path,
    payload: Mapping[str, Any],
    report: ValidationReport,
    token_coverage: TokenCoverage,
    chunk_coverage: ChunkCoverage,
    pass2_token_window_owner: dict[tuple[str, str, int], str],
    membership_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the production paired token/chunk Parquet transaction log."""

    _, pq = _import_arrow()
    label = str(progress_path.relative_to(root))
    worker_root = progress_path.parent
    is_pass2 = "pass2" in progress_path.parts or payload.get("metadata", {}).get("mode") == "pass2"
    stream_expected = (
        _expected_paired_stream(root, progress_path, payload, token_coverage.prepared, report, label)
        if is_pass2
        else None
    )
    metadata = payload.get("metadata", {})
    if is_pass2:
        if membership_identity is None:
            report.error(f"{label}: cannot bind Pass2 stream without membership identity")
        elif not isinstance(metadata, Mapping) or metadata.get(
            "membership_identity"
        ) != membership_identity:
            report.error(f"{label}: Pass2 membership_identity differs from canonical Pass1")
        worker_metadata_path = progress_path.parents[2] / "metadata.json"
        if not worker_metadata_path.is_file():
            report.incomplete(f"{label}: completed Pass2 worker metadata is missing")
        else:
            try:
                worker_metadata = json.loads(
                    worker_metadata_path.read_text(encoding="utf-8")
                )
                if worker_metadata.get("completed") is not True:
                    report.error(f"{label}: Pass2 worker metadata is not completed")
                if membership_identity is not None and worker_metadata.get(
                    "membership_identity"
                ) != membership_identity:
                    report.error(
                        f"{label}: worker metadata membership_identity differs from Pass1"
                    )
                paired = worker_metadata.get("paired_manifests")
                stream = metadata.get("metric_stream") if isinstance(metadata, Mapping) else None
                if not isinstance(paired, Mapping) or paired.get(stream) != payload:
                    report.error(
                        f"{label}: worker paired_manifests[{stream!r}] does not match progress"
                    )
                if isinstance(paired, Mapping):
                    expected_cumulative = sum(
                        float(value.get("cumulative_elapsed_seconds", 0.0))
                        for value in paired.values()
                        if isinstance(value, Mapping)
                    )
                    worker_cumulative = float(
                        worker_metadata.get(
                            "cumulative_elapsed_seconds", float("nan")
                        )
                    )
                    if not math.isfinite(worker_cumulative) or not math.isclose(
                        worker_cumulative,
                        expected_cumulative,
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    ):
                        report.error(
                            f"{label}: worker cumulative timing differs from paired journals"
                        )
            except Exception as error:
                report.error(
                    f"{label}: cannot validate Pass2 worker metadata: "
                    f"{type(error).__name__}: {error}"
                )
    if payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        report.error(f"{label}: unsupported paired progress schema version")
    if payload.get("complete") is not True:
        report.incomplete(f"{label}: paired progress is not complete")
    target_rows = int(payload.get("target_token_rows_per_shard", 0))
    if target_rows <= 0:
        report.error(f"{label}: invalid target_token_rows_per_shard")
    shards = payload.get("shards", [])
    if not isinstance(shards, list):
        report.error(f"{label}: paired shards is not a list")
        return {"manifest": label, "kind": PAIRED_STREAM_KIND, "rows": 0}
    if payload.get("complete") is True and not shards:
        report.error(f"{label}: complete paired stream has no committed shards")

    mapping_shards = [record for record in shards if isinstance(record, Mapping)]
    token_dir = _paired_recorded_dir(
        root,
        worker_root,
        payload,
        mapping_shards,
        directory_key="token_dir",
        artifact_key="token_file",
        default_name="token_metrics",
        report=report,
        label=label,
    )
    chunk_dir = _paired_recorded_dir(
        root,
        worker_root,
        payload,
        mapping_shards,
        directory_key="chunk_dir",
        artifact_key="chunk_file",
        default_name="chunk_metrics",
        report=report,
        label=label,
    )
    if token_dir is None or chunk_dir is None:
        return {"manifest": label, "kind": PAIRED_STREAM_KIND, "rows": 0}
    if stream_expected is not None:
        if token_dir.resolve() != stream_expected["token_dir"]:
            report.error(
                f"{label}: token_dir is not canonical for {stream_expected['stream']} stream"
            )
        if chunk_dir.resolve() != stream_expected["chunk_dir"]:
            report.error(
                f"{label}: chunk_dir is not canonical for {stream_expected['stream']} stream"
            )
    external_transients = sorted(
        str(path.relative_to(root))
        for directory in (token_dir, chunk_dir)
        if directory.exists()
        for path in directory.glob("*.inprogress")
    )
    if external_transients:
        report.incomplete(
            f"{label}: atomic temporary shard files remain: {external_transients}"
        )

    listed_token = {Path(str(record.get("token_file"))).name for record in mapping_shards}
    listed_chunk = {Path(str(record.get("chunk_file"))).name for record in mapping_shards}
    listed_sidecars = {Path(str(record.get("sidecar_file"))).name for record in mapping_shards}
    orphan_token = sorted({path.name for path in token_dir.glob("shard_*.parquet")} - listed_token)
    orphan_chunk = sorted({path.name for path in chunk_dir.glob("shard_*.parquet")} - listed_chunk)
    orphan_sidecars = sorted({path.name for path in worker_root.glob("shard_*.json")} - listed_sidecars)
    if orphan_token or orphan_chunk or orphan_sidecars:
        report.incomplete(
            f"{label}: uncommitted paired artifacts token={orphan_token} "
            f"chunk={orphan_chunk} sidecar={orphan_sidecars}"
        )

    committed_windows: set[int] = set()
    stream_keys: set[tuple[str, str, int]] = set()
    observed_stream_keys: set[tuple[str, str, int]] = set()
    total_token_rows = 0
    total_chunk_rows = 0
    total_batches = 0
    total_active_seconds = 0.0
    random_pair_reason_counts = {str(reason): 0 for reason in range(4)}
    token_schema: Any | None = None
    chunk_schema: Any | None = None
    for index, record in enumerate(shards):
        if not isinstance(record, dict):
            report.error(f"{label}: malformed paired record {index}")
            continue
        expected = f"shard_{index:06d}"
        expected_names = {
            "token_file": expected + ".parquet",
            "chunk_file": expected + ".parquet",
            "sidecar_file": expected + ".json",
        }
        for key, expected_name in expected_names.items():
            if Path(str(record.get(key))).name != expected_name:
                report.error(f"{label}: {key} sequence mismatch at shard {index}")
        token_path = _paired_artifact_path(
            root,
            token_dir,
            record.get("token_file"),
            report=report,
            label=f"{label}/shard{index}/token_file",
        )
        chunk_path = _paired_artifact_path(
            root,
            chunk_dir,
            record.get("chunk_file"),
            report=report,
            label=f"{label}/shard{index}/chunk_file",
        )
        sidecar_path = _paired_artifact_path(
            root,
            worker_root,
            record.get("sidecar_file"),
            report=report,
            label=f"{label}/shard{index}/sidecar_file",
        )
        if token_path is None or chunk_path is None or sidecar_path is None:
            continue
        artifacts = (
            (token_path, "token_sha256"),
            (chunk_path, "chunk_sha256"),
            (sidecar_path, "sidecar_sha256"),
        )
        corrupt = False
        for path, digest_key in artifacts:
            if not path.is_file():
                report.error(f"{label}: missing committed paired artifact {path.name}")
                corrupt = True
            elif file_sha256(path) != record.get(digest_key):
                report.error(f"{label}: hash mismatch for {path.name}")
                corrupt = True
        if corrupt:
            continue
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            for key in (
                "schema_version",
                "shard_index",
                "batch_count",
                "active_seconds",
                "window_uids",
                "token_file",
                "token_rows",
                "token_sha256",
                "chunk_file",
                "chunk_rows",
                "chunk_sha256",
            ):
                if sidecar.get(key) != record.get(key):
                    report.error(f"{label}: sidecar/record mismatch for {key} at shard {index}")
            token_table = pq.read_table(token_path)
            chunk_table = pq.read_table(chunk_path)
        except Exception as error:
            report.error(f"{label}: cannot read paired shard {index}: {type(error).__name__}: {error}")
            continue
        if token_table.num_rows != int(record.get("token_rows", -1)):
            report.error(f"{label}: token row count mismatch at shard {index}")
        if chunk_table.num_rows != int(record.get("chunk_rows", -1)):
            report.error(f"{label}: chunk row count mismatch at shard {index}")
        total_token_rows += token_table.num_rows
        total_chunk_rows += chunk_table.num_rows
        if "random_pair_invalid_reason" in chunk_table.column_names:
            pair_reasons = _array_numpy(
                chunk_table, "random_pair_invalid_reason", np.int64
            )
            for reason in range(4):
                random_pair_reason_counts[str(reason)] += int(
                    np.count_nonzero(pair_reasons == reason)
                )
        total_batches += int(record.get("batch_count", 0))
        active_seconds = float(record.get("active_seconds", float("nan")))
        if not math.isfinite(active_seconds) or active_seconds < 0.0:
            report.error(f"{label}: invalid active_seconds at shard {index}")
        else:
            total_active_seconds += active_seconds
        if int(record.get("batch_count", 0)) <= 0:
            report.error(f"{label}: non-positive batch_count at shard {index}")
        if token_table.num_rows <= 0 or chunk_table.num_rows <= 0:
            report.error(f"{label}: empty committed paired table at shard {index}")

        if stream_expected is not None:
            allowed_splits = set(stream_expected["splits"])
            expected_domain = stream_expected["domain"]
            token_domains = set(str(value) for value in token_table["domain"].to_pylist())
            chunk_domains = set(str(value) for value in chunk_table["domain"].to_pylist())
            token_splits = set(str(value) for value in token_table["split"].to_pylist())
            chunk_splits = set(str(value) for value in chunk_table["split"].to_pylist())
            if token_domains != {expected_domain} or chunk_domains != {expected_domain}:
                report.error(f"{label}: stream contains a different domain")
            if (
                token_splits != chunk_splits
                or not token_splits
                or not token_splits.issubset(allowed_splits)
            ):
                report.error(
                    f"{label}: physical split isolation violated: token={sorted(token_splits)} "
                    f"chunk={sorted(chunk_splits)} allowed={sorted(allowed_splits)}"
                )
            token_orders = _array_numpy(token_table, "sample_order", np.int64)
            token_domain_rows = _strings(token_table, "domain")
            token_split_rows = _strings(token_table, "split")
            observed_stream_keys.update(
                (str(domain_value), str(split_value), int(order))
                for domain_value, split_value, order in zip(
                    token_domain_rows, token_split_rows, token_orders
                )
            )

        if token_schema is None:
            token_schema = token_table.schema
            chunk_schema = chunk_table.schema
        elif not token_schema.equals(token_table.schema) or not chunk_schema.equals(
            chunk_table.schema
        ):
            report.error(f"{label}: paired Parquet schema changed at shard {index}")

        _validate_stream_schema(TOKEN_STREAM_KIND, token_table.schema, report, label)
        _validate_stream_schema(CHUNK_STREAM_KIND, chunk_table.schema, report, label)
        if is_pass2:
            _validate_pass2_token_extras(token_table.schema, report, label)
            _validate_pass2_chunk_extras(chunk_table.schema, report, label)
        recorded_uids = [int(value) for value in record.get("window_uids", [])]
        if len(recorded_uids) != len(set(recorded_uids)):
            report.error(f"{label}: duplicate window UID inside paired record {index}")
        token_uids = set(int(value) for value in token_table["window_uid"].to_pylist())
        chunk_uids = set(int(value) for value in chunk_table["window_uid"].to_pylist())
        if token_uids != set(recorded_uids) or chunk_uids != set(recorded_uids):
            report.error(f"{label}: paired token/chunk/sidecar window UID mismatch at shard {index}")
        if committed_windows & set(recorded_uids):
            report.error(f"{label}: window UID duplicated across committed paired shards")
        committed_windows.update(recorded_uids)

        if report.deep:
            for batch in token_table.to_batches(max_chunksize=65_536):
                _validate_token_batch(batch, report, label)
                if is_pass2:
                    keys = token_coverage.add_batch(batch, label)
                    stream_keys.update(keys)
            for batch in chunk_table.to_batches(max_chunksize=65_536):
                _validate_chunk_batch(batch, report, label)
                if is_pass2:
                    chunk_coverage.add_batch(batch, label)

    if total_batches != int(payload.get("committed_batches", -1)):
        report.error(f"{label}: committed batch count mismatch")
    recorded_elapsed = float(
        payload.get("cumulative_elapsed_seconds", float("nan"))
    )
    if not math.isfinite(recorded_elapsed) or not math.isclose(
        recorded_elapsed, total_active_seconds, rel_tol=1e-12, abs_tol=1e-9
    ):
        report.error(f"{label}: cumulative elapsed seconds mismatch")
    if len(committed_windows) != int(payload.get("committed_window_count", -1)):
        report.error(f"{label}: committed window count mismatch")
    if stream_expected is not None:
        expected_uids = stream_expected["window_uids"]
        expected_keys = stream_expected["window_keys"]
        extra_uids = committed_windows - expected_uids
        missing_uids = expected_uids - committed_windows
        extra_keys = observed_stream_keys - expected_keys
        missing_keys = expected_keys - observed_stream_keys
        if extra_uids or extra_keys:
            report.error(
                f"{label}: stream contains unprepared/foreign windows: "
                f"uid_extra={len(extra_uids)} key_extra={len(extra_keys)}"
            )
        if missing_uids or missing_keys:
            message = (
                f"{label}: stream window coverage incomplete: "
                f"uid_missing={len(missing_uids)} key_missing={len(missing_keys)}"
            )
            if payload.get("complete") is True:
                report.error(message)
            else:
                report.incomplete(message)
        if total_token_rows != int(stream_expected["token_count"]):
            message = (
                f"{label}: token row count {total_token_rows} != prepared stream "
                f"{stream_expected['token_count']}"
            )
            if payload.get("complete") is True:
                report.error(message)
            else:
                report.incomplete(message)
        if total_chunk_rows != int(stream_expected["chunk_count"]):
            message = (
                f"{label}: chunk row count {total_chunk_rows} != prepared stream "
                f"{stream_expected['chunk_count']}"
            )
            if payload.get("complete") is True:
                report.error(message)
            else:
                report.incomplete(message)
    if payload.get("complete") is True and shards:
        for index, record in enumerate(shards[:-1]):
            if int(record.get("token_rows", -1)) < target_rows:
                report.error(
                    f"{label}: non-final token shard {index} is below target rows"
                )
    if report.deep and is_pass2:
        for key in stream_keys:
            owner = pass2_token_window_owner.setdefault(key, label)
            if owner != label:
                report.error(f"duplicate pass2 window across workers: {key}: {owner}, {label}")
    return {
        "manifest": label,
        "kind": PAIRED_STREAM_KIND,
        "complete": payload.get("complete") is True,
        "shards": len(shards),
        "token_rows": total_token_rows,
        "chunk_rows": total_chunk_rows,
        "windows": len(committed_windows),
        "cumulative_elapsed_seconds": total_active_seconds,
        "metric_stream": stream_expected["stream"] if stream_expected else None,
        "stream_splits": stream_expected["splits"] if stream_expected else None,
        "test_metrics_sealed": bool(
            stream_expected and stream_expected["stream"] == "test"
        ),
        "random_pair_coverage": {
            "reason_counts": random_pair_reason_counts,
            "finite_fraction": (
                random_pair_reason_counts["0"] / total_chunk_rows
                if total_chunk_rows
                else None
            ),
        },
    }


def _validate_runtime(
    root: Path,
    report: ValidationReport,
    prepared: PreparedIndex | None,
    membership_identity: Mapping[str, Any] | None = None,
) -> None:
    manifests = _recognized_runtime_manifests(root)
    if not manifests:
        report.incomplete("no runtime Parquet stream manifests/progress files found")
        return
    pa, pq = _import_arrow()
    token_coverage = TokenCoverage(prepared, report)
    chunk_coverage = ChunkCoverage(prepared, report)
    pass2_token_window_owner: dict[tuple[str, str, int], str] = {}
    stream_stats: list[dict[str, Any]] = []

    for manifest_path, payload in manifests:
        label = str(manifest_path.relative_to(root))
        kind = payload.get("stream_kind")
        if kind == PAIRED_STREAM_KIND:
            stream_stats.append(
                _validate_paired_progress(
                    root,
                    manifest_path,
                    payload,
                    report,
                    token_coverage,
                    chunk_coverage,
                    pass2_token_window_owner,
                    membership_identity,
                )
            )
            continue
        is_pass2 = (
            "pass2" in manifest_path.parts
            or payload.get("metadata", {}).get("mode") == "pass2"
        )
        if payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            report.error(f"{label}: unsupported runtime schema version")
            continue
        if payload.get("complete") is not True:
            report.incomplete(f"{label}: stream manifest is not complete")
        rows_per_shard = int(payload.get("rows_per_shard", 0))
        if rows_per_shard <= 0:
            report.error(f"{label}: invalid rows_per_shard")
        shards = payload.get("shards")
        if not isinstance(shards, list):
            report.error(f"{label}: shards is not a list")
            continue
        if int(payload.get("next_shard", -1)) != len(shards):
            report.error(f"{label}: next_shard does not equal shard count")
        encoded_schema = payload.get("schema_base64")
        try:
            manifest_schema = (
                pa.ipc.read_schema(pa.BufferReader(base64.b64decode(encoded_schema)))
                if encoded_schema
                else None
            )
        except Exception as error:
            report.error(f"{label}: invalid serialized Arrow schema: {error}")
            manifest_schema = None
        if manifest_schema is not None:
            _validate_stream_schema(kind, manifest_schema, report, label)
            if kind == TOKEN_STREAM_KIND and is_pass2:
                _validate_pass2_token_extras(manifest_schema, report, label)
            if kind == CHUNK_STREAM_KIND and is_pass2:
                _validate_pass2_chunk_extras(manifest_schema, report, label)

        listed = {
            Path(str(record.get("file"))).name
            for record in shards
            if isinstance(record, dict)
        }
        actual = {path.name for path in manifest_path.parent.glob("shard_*.parquet")}
        extras = sorted(actual - listed)
        missing_names = sorted(listed - actual)
        if extras or missing_names:
            report.incomplete(
                f"{label}: uncommitted/missing shards extras={extras} missing={missing_names}"
            )
        total_rows = 0
        stream_window_keys: set[tuple[str, str, int]] = set()
        for index, record in enumerate(shards):
            if not isinstance(record, dict):
                report.error(f"{label}: malformed shard record {index}")
                continue
            expected_name = f"shard_{index:06d}.parquet"
            if Path(str(record.get("file"))).name != expected_name:
                report.error(f"{label}: non-contiguous shard name {record.get('file')} != {expected_name}")
            shard_path = _paired_artifact_path(
                root,
                manifest_path.parent,
                record.get("file"),
                report=report,
                label=f"{label}/shard{index}/file",
            )
            if shard_path is None:
                continue
            if not shard_path.is_file():
                continue
            if int(record.get("bytes", -1)) != shard_path.stat().st_size:
                report.error(f"{label}: shard byte-size mismatch: {shard_path.name}")
            if record.get("sha256") != file_sha256(shard_path):
                report.error(f"{label}: shard SHA256 mismatch: {shard_path.name}")
                continue
            sidecar_name = record.get("sidecar_file")
            sidecar_path = _paired_artifact_path(
                root,
                manifest_path.parent,
                sidecar_name,
                report=report,
                label=f"{label}/shard{index}/sidecar_file",
            )
            if sidecar_path is None:
                continue
            if not sidecar_path.is_file() or file_sha256(sidecar_path) != record.get(
                "sidecar_sha256"
            ):
                report.error(f"{label}: sidecar missing/corrupt for {shard_path.name}")
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                for key in ("schema_version", "stream_kind", "file", "rows", "bytes", "sha256"):
                    expected_value = (
                        payload.get(key)
                        if key in ("schema_version", "stream_kind")
                        else record.get(key)
                    )
                    if sidecar.get(key) != expected_value:
                        report.error(f"{label}: standalone sidecar mismatch for {key}")
                table = pq.read_table(shard_path)
                schema = table.schema
                if manifest_schema is not None and not schema.equals(manifest_schema):
                    report.error(f"{label}: shard schema differs from manifest")
                if manifest_schema is None:
                    _validate_stream_schema(kind, schema, report, label)
                    if kind == TOKEN_STREAM_KIND and is_pass2:
                        _validate_pass2_token_extras(schema, report, label)
                    if kind == CHUNK_STREAM_KIND and is_pass2:
                        _validate_pass2_chunk_extras(schema, report, label)
                shard_rows = table.num_rows
                if shard_rows != int(record.get("rows", -1)):
                    report.error(f"{label}: shard row count mismatch: {shard_path.name}")
                total_rows += shard_rows
                if report.deep:
                    for batch in table.to_batches(max_chunksize=65_536):
                        if kind == TOKEN_STREAM_KIND:
                            _validate_token_batch(batch, report, label)
                            if is_pass2:
                                stream_window_keys.update(
                                    token_coverage.add_batch(batch, label)
                                )
                        else:
                            _validate_chunk_batch(batch, report, label)
                            if is_pass2:
                                chunk_coverage.add_batch(batch, label)
            except Exception as error:
                report.error(f"{label}: cannot read Parquet shard {shard_path.name}: {type(error).__name__}: {error}")
        if total_rows != int(payload.get("committed_rows", -1)):
            report.error(f"{label}: committed_rows mismatch {total_rows} != {payload.get('committed_rows')}")
        if payload.get("complete") is True and shards:
            for record in shards[:-1]:
                if int(record.get("rows", -1)) < rows_per_shard:
                    report.error(f"{label}: non-final complete shard is below target rows")
        if report.deep and kind == TOKEN_STREAM_KIND and is_pass2:
            for key in stream_window_keys:
                owner = pass2_token_window_owner.setdefault(key, label)
                if owner != label:
                    report.error(f"duplicate pass2 window across workers: {key}: {owner}, {label}")
        stream_stats.append(
            {
                "manifest": label,
                "kind": kind,
                "complete": payload.get("complete") is True,
                "shards": len(shards),
                "rows": total_rows,
            }
        )

    if report.deep:
        observed_token_windows = token_coverage.finalize()
        chunk_coverage.finalize()
        missing_chunk_windows = set(pass2_token_window_owner) - set(chunk_coverage.summary)
        extra_chunk_windows = set(chunk_coverage.summary) - set(pass2_token_window_owner)
        if missing_chunk_windows:
            report.incomplete(
                f"chunk metrics missing for {len(missing_chunk_windows)} pass2 token windows"
            )
        if extra_chunk_windows:
            report.error(
                f"chunk metrics contain {len(extra_chunk_windows)} windows without token metrics"
            )
        if prepared is not None:
            expected = prepared.expected_keys
            missing = expected - set(pass2_token_window_owner)
            extra = set(pass2_token_window_owner) - expected
            if missing:
                report.incomplete(f"pass2 window coverage missing {len(missing)} prepared windows")
            if extra:
                report.error(f"pass2 window coverage has {len(extra)} unprepared windows")
            report.stats["pass2_window_coverage"] = {
                "expected": len(expected),
                "observed_pass2": len(pass2_token_window_owner),
                "all_token_stream_windows": len(observed_token_windows),
                "missing": len(missing),
                "extra": len(extra),
                "missing_chunk_windows": len(missing_chunk_windows),
                "extra_chunk_windows": len(extra_chunk_windows),
            }
    report.stats["runtime_streams"] = stream_stats
    report.checked("runtime Parquet progress/manifests, atomic paired shards, hashes, schemas, and row counts")
    if report.deep:
        report.checked("deep scalar finiteness/ranges, router shapes, and token/chunk coverage")


def _find_membership_npz(root: Path) -> Path | None:
    preferred = root / "membership" / "wiki_calibration_membership.npz"
    if preferred.is_file():
        return preferred
    candidates = sorted((root / "membership").glob("*.npz")) if (root / "membership").exists() else []
    return candidates[0] if len(candidates) == 1 else None


def _membership_identity_from_npz(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the runtime's immutable membership identity without importing torch."""

    path = path.resolve()
    with np.load(path, allow_pickle=False) as data:
        layers = data["layers"].astype(np.int16).tolist()
        metadata = json.loads(str(data["metadata_json"].item()))
    source = metadata.get("metadata", {})
    if not isinstance(source, Mapping):
        raise ValueError("membership metadata_json.metadata must be an object")
    reference = source.get("reference_load")
    summary = {
        "analysis": source.get("analysis"),
        "mode": source.get("mode"),
        "source_domain": source.get("domain"),
        "source_split": source.get("requested_split"),
        "source_dataset_prefix": source.get("dataset_prefix"),
        "reference_checkpoint": os.path.realpath(str(reference)) if reference else None,
        "reference_step": source.get("reference_step"),
        "prepared_config_content_sha256": source.get(
            "prepared_config_content_sha256"
        ),
        "source_window_count": source.get("total_windows"),
        "source_token_count": source.get("total_tokens"),
        "seed": source.get("seed"),
        "max_windows": source.get("max_windows"),
        "layers": [int(layer) for layer in layers],
        "sampling_provenance": metadata.get("sampling_provenance"),
    }
    identity = {
        "schema": "cka_gt_pilot_membership_identity_v1",
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "saved_metadata_summary": summary,
    }
    return identity, metadata


def _validate_membership_binding(
    root: Path,
    report: ValidationReport,
    prepared: PreparedIndex | None,
) -> dict[str, Any] | None:
    """Bind the canonical NPZ, Pass1 worker metadata, and prepared source."""

    path = _find_membership_npz(root)
    if path is None:
        report.incomplete("Pass-1 membership NPZ is missing or ambiguous")
        return None
    canonical_path = (root / "membership" / "wiki_calibration_membership.npz").resolve()
    if path.resolve() != canonical_path:
        report.error(f"membership NPZ is not at canonical path: {path}")
    try:
        identity, _ = _membership_identity_from_npz(path)
    except Exception as error:
        report.error(f"cannot reconstruct membership identity: {type(error).__name__}: {error}")
        return None

    pass1_path = (
        root
        / "runtime"
        / "pass1"
        / "wiki"
        / "calibration"
        / "worker_000"
        / "metadata.json"
    )
    if not pass1_path.is_file():
        report.incomplete(f"canonical Pass1 worker metadata is missing: {pass1_path}")
    else:
        try:
            pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
            if pass1.get("completed") is not True:
                report.error("canonical Pass1 worker metadata is not completed")
            if pass1.get("membership_identity") != identity:
                report.error("Pass1 worker membership_identity differs from canonical NPZ")
            if Path(str(pass1.get("membership_statistics"))).resolve() != canonical_path:
                report.error("Pass1 worker metadata points at a different membership NPZ")
            if prepared is not None and pass1.get(
                "source_dataset_identity"
            ) != prepared.config.get("source_dataset_identity"):
                report.error("Pass1 worker source_dataset_identity differs from config")
            if prepared is not None and pass1.get(
                "checkpoint_identity"
            ) != prepared.config.get("checkpoint_identity"):
                report.error("Pass1 worker checkpoint_identity differs from config")
        except Exception as error:
            report.error(f"cannot validate Pass1 worker metadata: {type(error).__name__}: {error}")

    summary = identity["saved_metadata_summary"]
    if prepared is not None:
        cap = int(summary.get("max_windows") or 0)
        source_rows = prepared.windows[("wiki", "calibration")]
        if cap > 0:
            source_rows = source_rows[:cap]
        expected = {
            "analysis": "cka_gt_pilot_v1",
            "mode": "pass1",
            "source_domain": "wiki",
            "source_split": "calibration",
            "source_dataset_prefix": str(prepared.config["wiki_prefix"]),
            "reference_checkpoint": os.path.realpath(
                str(prepared.config["before_checkpoint"])
            ),
            "reference_step": 600,
            "prepared_config_content_sha256": prepared.config.get(
                "config_content_sha256"
            ),
            "source_window_count": int(source_rows.shape[0]),
            "source_token_count": int(
                source_rows["window_length"].sum(dtype=np.int64)
            ),
            "seed": int(prepared.config["base_seed"]),
            "max_windows": cap,
            "layers": list(LAYERS),
        }
        for key, wanted in expected.items():
            if summary.get(key) != wanted:
                report.error(
                    f"membership source provenance mismatch: {key}={summary.get(key)!r}, "
                    f"expected={wanted!r}"
                )
    sampling = summary.get("sampling_provenance")
    if not isinstance(sampling, Mapping):
        report.error("membership identity omits sampling provenance")
    else:
        frozen_seed = (
            int(prepared.config["base_seed"])
            if prepared is not None
            else int(summary.get("seed", 1234))
        )
        frozen_sampling = {
            "base_seed": frozen_seed,
            "domain_id": 1,
            "split_id": 0,
            "covariance_stage_id": 2,
            "kmeans_reservoir_stage_id": 3,
            "covariance_entropy": [frozen_seed, 1, 2, 0],
            "kmeans_reservoir_entropy": [frozen_seed, 1, 3, 0],
        }
        for key, wanted in frozen_sampling.items():
            if sampling.get(key) != wanted:
                report.error(
                    f"membership sampling provenance mismatch: {key}="
                    f"{sampling.get(key)!r}, expected={wanted!r}"
                )
    report.stats["membership_identity"] = identity
    report.checked("membership NPZ identity, canonical Pass1 metadata, and prepared-source binding")
    return identity


def _pick_array(data: Any, aliases: Sequence[str]) -> tuple[str, np.ndarray] | None:
    for name in aliases:
        if name in data.files:
            return name, np.asarray(data[name])
    return None


def _whitening_probe_error(covariance: np.ndarray, whitener: np.ndarray) -> float:
    dimension = covariance.shape[0]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1234, 0xCA])))
    probes = rng.standard_normal((dimension, min(16, dimension))).astype(np.float64)
    cov = covariance.astype(np.float64, copy=False)
    white = whitener.astype(np.float64, copy=False)
    left = white @ (cov @ (white.T @ probes))
    right = white.T @ (cov @ (white @ probes))
    denominator = max(float(np.linalg.norm(probes)), 1e-12)
    return min(
        float(np.linalg.norm(left - probes) / denominator),
        float(np.linalg.norm(right - probes) / denominator),
    )


def _validate_membership_stats(
    root: Path,
    report: ValidationReport,
    *,
    expected_hidden_size: int = DEFAULT_HIDDEN_SIZE,
) -> None:
    path = _find_membership_npz(root)
    if path is None:
        report.incomplete("Pass-1 membership NPZ is missing or ambiguous")
        return
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as error:
        report.error(f"cannot load Pass-1 membership stats without pickle: {error}")
        return
    forbidden = sorted(name for name in data.files if _forbidden_name(name))
    if forbidden:
        report.error(f"Pass-1 NPZ contains raw hidden arrays: {forbidden}")

    mean_item = _pick_array(data, ("mean", "means", "mu", "wiki_mean"))
    covariance_item = _pick_array(data, ("covariance", "covariances", "sigma", "cov"))
    whitener_item = _pick_array(
        data,
        ("whitener", "whiteners", "cov_inv_sqrt", "inverse_sqrt_covariance"),
    )
    prototype_item = _pick_array(data, ("prototypes", "centroids", "kmeans_centers"))
    metadata: dict[str, Any] = {}
    if "metadata_json" in data.files:
        try:
            metadata = json.loads(str(data["metadata_json"].item()))
        except Exception as error:
            report.error(f"Pass-1 metadata_json is invalid: {error}")
    # Production schema stores each layer separately to support direct lookup
    # without materializing a second stacked copy.
    if mean_item is None and all(f"layer_{layer}_mean" in data.files for layer in LAYERS):
        mean_item = (
            "layer_*_mean",
            np.stack([data[f"layer_{layer}_mean"] for layer in LAYERS]),
        )
    if covariance_item is None and all(
        f"layer_{layer}_covariance" in data.files for layer in LAYERS
    ):
        covariance_item = (
            "layer_*_covariance",
            np.stack([data[f"layer_{layer}_covariance"] for layer in LAYERS]),
        )
    if whitener_item is None and all(
        f"layer_{layer}_inverse_sqrt" in data.files for layer in LAYERS
    ):
        whitener_item = (
            "layer_*_inverse_sqrt",
            np.stack([data[f"layer_{layer}_inverse_sqrt"] for layer in LAYERS]),
        )
    if prototype_item is None and all(
        f"layer_{layer}_prototypes" in data.files for layer in LAYERS
    ):
        prototype_item = (
            "layer_*_prototypes",
            np.stack([data[f"layer_{layer}_prototypes"] for layer in LAYERS]),
        )
    if mean_item is None:
        report.error("Pass-1 NPZ missing layer means")
    if covariance_item is None:
        report.error("Pass-1 NPZ missing Ledoit-Wolf covariance")
    if whitener_item is None:
        report.error("Pass-1 NPZ missing covariance whitener")
    if prototype_item is None:
        report.error("Pass-1 NPZ missing K=64 prototypes")
    if any(item is None for item in (mean_item, covariance_item, whitener_item, prototype_item)):
        data.close()
        return

    _, means = mean_item
    _, covariance = covariance_item
    _, whitener = whitener_item
    _, prototypes = prototype_item
    expected_mean = (8, expected_hidden_size)
    expected_matrix = (8, expected_hidden_size, expected_hidden_size)
    expected_prototypes = (8, 64, expected_hidden_size)
    if means.shape != expected_mean:
        report.error(f"Pass-1 mean shape {means.shape}, expected {expected_mean}")
    if covariance.shape != expected_matrix:
        report.error(f"Pass-1 covariance shape {covariance.shape}, expected {expected_matrix}")
    if whitener.shape != expected_matrix:
        report.error(f"Pass-1 whitener shape {whitener.shape}, expected {expected_matrix}")
    if prototypes.shape != expected_prototypes:
        report.error(f"Pass-1 prototype shape {prototypes.shape}, expected {expected_prototypes}")
    for name, values in (("mean", means), ("covariance", covariance), ("whitener", whitener), ("prototypes", prototypes)):
        if not np.issubdtype(values.dtype, np.floating):
            report.error(f"Pass-1 {name} dtype is not floating: {values.dtype}")
        if not np.isfinite(values).all():
            report.error(f"Pass-1 {name} contains NaN/Inf")

    if covariance.shape == expected_matrix:
        symmetry_error = float(
            np.max(np.abs(covariance - np.swapaxes(covariance, -1, -2)))
        )
        if symmetry_error > 2e-4:
            report.error(f"Pass-1 covariance is not symmetric: max error {symmetry_error}")
        if np.any(np.diagonal(covariance, axis1=1, axis2=2) <= 0):
            report.error("Pass-1 covariance has non-positive diagonal")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1234, 0xC0])))
        probes = rng.standard_normal((expected_hidden_size, min(16, expected_hidden_size)))
        for layer_index in range(8):
            quadratic = np.sum(probes * (covariance[layer_index] @ probes), axis=0)
            if np.any(quadratic <= 0):
                report.error(f"Pass-1 covariance fails positive-definite probes at layer {layer_index + 2}")
    whitening_errors: list[float] = []
    if covariance.shape == expected_matrix and whitener.shape == expected_matrix:
        for layer_index in range(8):
            error = _whitening_probe_error(covariance[layer_index], whitener[layer_index])
            whitening_errors.append(error)
            if not math.isfinite(error) or error > 0.10:
                report.error(
                    f"Pass-1 whitening probe error at layer {layer_index + 2}: {error:.6g}"
                )
    if prototypes.shape == expected_prototypes:
        for layer_index in range(8):
            centers = prototypes[layer_index].astype(np.float64, copy=False)
            squared_norm = np.sum(centers * centers, axis=1)
            squared_distance = (
                squared_norm[:, None]
                + squared_norm[None, :]
                - 2.0 * (centers @ centers.T)
            )
            np.fill_diagonal(squared_distance, np.inf)
            if np.any(squared_distance <= 1e-20):
                report.error(
                    f"Pass-1 prototypes contain duplicate centers at layer {layer_index + 2}"
                )
    if "layers" in data.files and not np.array_equal(data["layers"], np.asarray(LAYERS)):
        report.error("Pass-1 layer list is not exactly 2..9")
    layer_metadata = metadata.get("layers", {}) if isinstance(metadata, dict) else {}
    for layer in LAYERS:
        values = layer_metadata.get(str(layer))
        if values is None:
            if metadata:
                report.error(f"Pass-1 metadata omits layer {layer}")
            continue
        all_count = int(values.get("all_token_count", 0))
        covariance_count = int(values.get("covariance_sample_count", 0))
        kmeans_count = int(values.get("kmeans_sample_count", 0))
        if all_count <= 0 or covariance_count <= 0 or kmeans_count <= 0:
            report.error(f"Pass-1 sample counts must be positive at layer {layer}")
        if covariance_count > min(all_count, 2_000_000):
            report.error(f"Pass-1 covariance sample cap exceeded at layer {layer}")
        if kmeans_count > min(all_count, 200_000):
            report.error(f"Pass-1 k-means reservoir cap exceeded at layer {layer}")
        shrinkage = float(values.get("ledoit_wolf_shrinkage", float("nan")))
        if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
            report.error(f"Pass-1 Ledoit-Wolf shrinkage invalid at layer {layer}")
        kmeans = values.get("kmeans_report", {})
        expected_kmeans = {
            "algorithm": "repository_deterministic_minibatch_kmeans_v1",
            "version": 1,
            "n_clusters": 64,
            "batch_size": 4096,
            "n_init": 1,
            "max_iter": 100,
            "reassignment_ratio": 0.01,
            "seed": 1234,
            "init": "k-means++",
            "init_size": min(kmeans_count, max(3 * 4096, 3 * 64)),
            "init_pool_policy": "min(n_samples,max(3*batch_size,3*n_clusters))",
            "initialization_compute": "shared_cpu_numpy_float32_d_squared",
        }
        for key, expected in expected_kmeans.items():
            if kmeans.get(key) != expected:
                report.error(
                    f"Pass-1 k-means parameter mismatch layer {layer}: "
                    f"{key}={kmeans.get(key)!r}, expected {expected!r}"
                )
        initial_runs = kmeans.get("initial_indices_by_run")
        if not (
            isinstance(initial_runs, list)
            and len(initial_runs) == 1
            and isinstance(initial_runs[0], list)
            and len(initial_runs[0]) == 64
            and len(set(initial_runs[0])) == 64
            and all(0 <= int(index) < kmeans_count for index in initial_runs[0])
        ):
            report.error(
                f"Pass-1 k-means++ initial centers malformed at layer {layer}"
            )
        pool_hashes = kmeans.get("init_pool_indices_sha256_by_run")
        if not (
            isinstance(pool_hashes, list)
            and len(pool_hashes) == 1
            and isinstance(pool_hashes[0], str)
            and len(pool_hashes[0]) == 64
        ):
            report.error(f"Pass-1 k-means init-pool provenance malformed at layer {layer}")
        cluster_counts = kmeans.get("cluster_update_counts")
        if not (
            isinstance(cluster_counts, list)
            and len(cluster_counts) == 64
            and all(int(value) >= 0 for value in cluster_counts)
        ):
            report.error(f"Pass-1 k-means update counts malformed at layer {layer}")
        eigen_key = f"layer_{layer}_covariance_eigenvalues"
        empirical_key = f"layer_{layer}_empirical_covariance"
        if eigen_key in data.files:
            eigenvalues = np.asarray(data[eigen_key])
            if eigenvalues.shape != (expected_hidden_size,) or not np.isfinite(
                eigenvalues
            ).all():
                report.error(f"Pass-1 covariance eigenvalues malformed at layer {layer}")
        if empirical_key in data.files:
            empirical = np.asarray(data[empirical_key])
            if empirical.shape != (expected_hidden_size, expected_hidden_size) or not np.isfinite(
                empirical
            ).all():
                report.error(f"Pass-1 empirical covariance malformed at layer {layer}")
    report.stats["membership"] = {
        "file": str(path.relative_to(root)),
        "mean_shape": list(means.shape),
        "covariance_shape": list(covariance.shape),
        "whitener_shape": list(whitener.shape),
        "prototype_shape": list(prototypes.shape),
        "whitening_probe_max_relative_error": max(whitening_errors) if whitening_errors else None,
    }
    report.checked("Pass-1 means, covariance, whitener, and K=64 prototypes")
    data.close()


def _validate_test_seal(root: Path, report: ValidationReport) -> None:
    sealed_root = root / "sealed_test"
    manifest_path = sealed_root / "manifest.json"
    if not manifest_path.is_file():
        report.incomplete("sealed-test manifest is missing")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        report.error(f"cannot read sealed-test manifest: {error}")
        return
    if manifest.get("schema") != "cka_gt_pilot_sealed_manifest_v1":
        report.error("sealed-test manifest schema mismatch")
    if manifest.get("included_in_report_v1") is not False:
        report.error("sealed test is marked as included in REPORT v1")
    metrics_file = manifest.get("metrics_file")
    if not isinstance(metrics_file, str) or Path(metrics_file).name != metrics_file:
        report.error("sealed metrics filename is unsafe or missing")
        return
    metrics_path = sealed_root / metrics_file
    if not metrics_path.is_file():
        report.error("sealed metrics file is missing")
        return
    if file_sha256(metrics_path) != manifest.get("sha256"):
        report.error("sealed metrics SHA256 mismatch")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as error:
        report.error(f"cannot parse sealed metrics: {error}")
        return
    if metrics.get("schema") != "cka_gt_pilot_sealed_test_v1" or metrics.get("sealed") is not True:
        report.error("sealed metrics payload is not marked sealed")
    report_path = root / "REPORT.md"
    if not report_path.is_file():
        report.incomplete("REPORT.md is missing while sealed test exists")
    else:
        text = report_path.read_text(encoding="utf-8")
        if "REPORT v1 deliberately contains no test" not in text:
            report.error("REPORT.md lacks the explicit test-exclusion declaration")
        # A byte-for-byte embedded payload is an unambiguous leak.
        if metrics_path.read_text(encoding="utf-8").strip() in text:
            report.error("sealed test payload was embedded in REPORT.md")
    for path in root.rglob("*test_metrics*.json"):
        if sealed_root not in path.parents:
            report.error(f"test metrics exist outside sealed_test/: {path.relative_to(root)}")
    report.checked("sealed-test hash and REPORT-v1 exclusion policy")


def validate_pilot(
    root: str | Path,
    *,
    allow_incomplete: bool = False,
    deep: bool = False,
    datasets: Mapping[str, Any] | None = None,
    expected_hidden_size: int = DEFAULT_HIDDEN_SIZE,
) -> ValidationReport:
    root = Path(root).resolve()
    report = ValidationReport(str(root), allow_incomplete, deep)
    if not root.is_dir():
        report.error(f"pilot root is not a directory: {root}")
        return report
    prepared = _validate_prepared(root, report, datasets)
    _validate_no_raw_hidden_files(root, report)

    transient_roots = [
        root / "runtime",
        root / "membership",
        root / "splits",
        root / "token_metrics",
        root / "chunk_metrics",
        root / "sealed_test" / "raw",
    ]
    transients = [
        str(path.relative_to(root))
        for scan_root in transient_roots
        if scan_root.exists()
        for path in scan_root.rglob("*")
        if path.is_file() and (path.name.endswith(".inprogress") or path.name.endswith(".tmp"))
    ]
    if transients:
        report.incomplete(f"atomic temporary files remain: {transients[:20]}")
    else:
        report.checked("no stale atomic temporary files")

    membership_identity = _validate_membership_binding(root, report, prepared)
    _validate_runtime(root, report, prepared, membership_identity)
    _validate_membership_stats(
        root, report, expected_hidden_size=int(expected_hidden_size)
    )
    _validate_test_seal(root, report)
    return report


def _fixture_table(columns: Mapping[str, Any]) -> Any:
    pa, _ = _import_arrow()
    arrays: dict[str, Any] = {}
    for name, values in columns.items():
        if isinstance(values, np.ndarray) and values.ndim > 1:
            nested = pa.array(values.reshape(-1))
            for width in reversed(values.shape[1:]):
                nested = pa.FixedSizeListArray.from_arrays(nested, int(width))
            arrays[name] = nested
        elif (
            isinstance(values, (list, tuple))
            and values
            and isinstance(values[0], np.ndarray)
        ):
            leaf_dtype = np.asarray(values[0]).dtype
            leaf_type = pa.float32() if leaf_dtype == np.float32 else None
            arrays[name] = pa.array(
                [np.asarray(value).tolist() for value in values],
                type=pa.list_(leaf_type) if leaf_type is not None else None,
            )
        else:
            arrays[name] = pa.array(values)
    return pa.table(arrays)


def _write_arrow_fixture(
    output_dir: Path,
    kind: str,
    columns: Mapping[str, Any],
    *,
    complete: bool = True,
) -> None:
    _, pq = _import_arrow()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = _fixture_table(columns)
    shard = output_dir / "shard_000000.parquet"
    with shard.open("wb") as handle:
        pq.write_table(table, handle, compression="zstd")
    record = {
        "file": shard.name,
        "rows": table.num_rows,
        "bytes": shard.stat().st_size,
        "sha256": file_sha256(shard),
    }
    sidecar = output_dir / "shard_000000.json"
    sidecar_payload = {
        "schema_version": 1,
        "stream_kind": kind,
        "shard_index": 0,
        **record,
        "column_names": table.column_names,
    }
    sidecar.write_text(json.dumps(sidecar_payload, sort_keys=True) + "\n", encoding="utf-8")
    record["sidecar_file"] = sidecar.name
    record["sidecar_sha256"] = file_sha256(sidecar)
    manifest = {
        "schema_version": 1,
        "stream_kind": kind,
        "rows_per_shard": max(1, table.num_rows),
        "committed_rows": table.num_rows,
        "next_shard": 1,
        "schema_base64": base64.b64encode(table.schema.serialize().to_pybytes()).decode("ascii"),
        "shards": [record],
        "complete": complete,
        "metadata": {"mode": "pass2"},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_paired_fixture(
    root: Path,
    worker_root: Path,
    token_dir: Path,
    chunk_dir: Path,
    token_columns: Mapping[str, Any],
    chunk_columns: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> None:
    """Write one production-shaped paired transaction for the self-test."""

    _, pq = _import_arrow()
    worker_root.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    token_table = _fixture_table(token_columns)
    chunk_table = _fixture_table(chunk_columns)
    token_path = token_dir / "shard_000000.parquet"
    chunk_path = chunk_dir / "shard_000000.parquet"
    pq.write_table(token_table, token_path, compression="zstd")
    pq.write_table(chunk_table, chunk_path, compression="zstd")
    window_uids = sorted(set(int(value) for value in token_columns["window_uid"]))
    sidecar_path = worker_root / "shard_000000.json"
    sidecar_payload = {
        "schema_version": 1,
        "shard_index": 0,
        "batch_count": 1,
        "active_seconds": 1.0,
        "window_uids": window_uids,
        "token_file": str(token_path.resolve()),
        "token_rows": token_table.num_rows,
        "token_sha256": file_sha256(token_path),
        "chunk_file": str(chunk_path.resolve()),
        "chunk_rows": chunk_table.num_rows,
        "chunk_sha256": file_sha256(chunk_path),
    }
    sidecar_path.write_text(
        json.dumps(sidecar_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = {
        **sidecar_payload,
        "sidecar_file": str(sidecar_path.resolve()),
        "sidecar_sha256": file_sha256(sidecar_path),
    }
    progress = {
        "schema_version": 1,
        "stream_kind": PAIRED_STREAM_KIND,
        "token_dir": str(token_dir.resolve()),
        "chunk_dir": str(chunk_dir.resolve()),
        "target_token_rows_per_shard": max(1, token_table.num_rows),
        "committed_batches": 1,
        "committed_window_count": len(window_uids),
        "cumulative_elapsed_seconds": 1.0,
        "shards": [record],
        "complete": True,
        "metadata": dict(metadata),
    }
    (worker_root / "progress.json").write_text(
        json.dumps(progress, sort_keys=True) + "\n", encoding="utf-8"
    )


def _self_test() -> None:
    from test_cka_gt_pilot_windows import FakeIndexedDataset

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir) / "pilot"
        datasets = {
            "code": FakeIndexedDataset.from_lengths([1024] * 20),
            "wiki": FakeIndexedDataset.from_lengths([1024] * 20),
        }
        prepare_pilot(
            root,
            "/fake/code/train_text_document",
            "/fake/wiki/train_text_document",
            1234,
            before_checkpoint="/fake/before",
            after_checkpoint="/fake/after",
            total_sample_windows_per_domain=4,
            datasets=datasets,
        )
        prepared = _load_prepared_index(
            root, json.loads((root / "config.json").read_text(encoding="utf-8"))
        )
        row = prepared.windows[("code", "calibration")][0]
        length = int(row["window_length"])
        token_rows = length
        window_uid = 123456789
        base = {
            "domain": ["code"] * token_rows,
            "split": ["calibration"] * token_rows,
            "window_uid": np.full(token_rows, window_uid, dtype=np.int64),
            "sample_order": np.zeros(token_rows, dtype=np.int64),
            "source_window_index": np.full(
                token_rows, int(row["source_window_index"]), dtype=np.int64
            ),
            "document_id": np.full(token_rows, int(row["document_id"]), dtype=np.int64),
            "window_offset": np.full(token_rows, int(row["window_offset"]), dtype=np.int64),
            "window_length": np.full(token_rows, length, dtype=np.int32),
            "position": np.arange(token_rows, dtype=np.int32),
            "document_token_offset": int(row["window_offset"]) + np.arange(
                token_rows, dtype=np.int64
            ),
            "token_id": np.arange(token_rows, dtype=np.int32),
            "eligible": np.ones(token_rows, dtype=bool),
            "layer_numbers": np.tile(np.asarray(LAYERS, dtype=np.int16), (token_rows, 1)),
            "cosine": np.full((token_rows, 8), 0.99, dtype=np.float32),
            "relative_l2": np.full((token_rows, 8), 0.05, dtype=np.float32),
            "symmetric_relative_l2": np.full((token_rows, 8), 0.049, dtype=np.float32),
            "log_r": np.zeros((token_rows, 8), dtype=np.float32),
            "ref_rms": np.ones((token_rows, 8), dtype=np.float32),
            "maha": np.ones((token_rows, 8), dtype=np.float32),
            "proto": np.ones((token_rows, 8), dtype=np.float32),
            "maha_mean": np.ones(token_rows, dtype=np.float32),
            "proto_mean": np.ones(token_rows, dtype=np.float32),
            "cka_min_128": np.full((token_rows, 8), 0.9, dtype=np.float32),
            "cka_min_256": np.full((token_rows, 8), 0.9, dtype=np.float32),
            "s_min_128": np.ones((token_rows, 8), dtype=np.float32),
            "s_min_256": np.ones((token_rows, 8), dtype=np.float32),
            "r_min_128": np.full((token_rows, 8), 0.9, dtype=np.float32),
            "r_mean_128": np.full((token_rows, 8), 0.95, dtype=np.float32),
            "r_max_128": np.ones((token_rows, 8), dtype=np.float32),
            "r_min_256": np.full((token_rows, 8), 0.9, dtype=np.float32),
            "r_mean_256": np.full((token_rows, 8), 0.95, dtype=np.float32),
            "r_max_256": np.ones((token_rows, 8), dtype=np.float32),
            "worst_diag_ratio_128": np.full((token_rows, 8), 0.2, dtype=np.float32),
            "worst_diag_ratio_256": np.full((token_rows, 8), 0.2, dtype=np.float32),
            "neg_contrib_128": np.zeros((token_rows, 8), dtype=bool),
            "neg_contrib_256": np.zeros((token_rows, 8), dtype=bool),
            "offdiag_warning_128": np.zeros((token_rows, 8), dtype=bool),
            "offdiag_warning_256": np.zeros((token_rows, 8), dtype=bool),
            "worst_chunk_id_128": np.zeros((token_rows, 8), dtype=np.int16),
            "worst_chunk_id_256": np.zeros((token_rows, 8), dtype=np.int16),
            "s_at_worst_cka_128": np.ones((token_rows, 8), dtype=np.float32),
            "s_at_worst_cka_256": np.ones((token_rows, 8), dtype=np.float32),
            "worst_s_chunk_id_128": np.zeros((token_rows, 8), dtype=np.int16),
            "worst_s_chunk_id_256": np.zeros((token_rows, 8), dtype=np.int16),
            "valid_cka_128": np.ones((token_rows, 8), dtype=bool),
            "valid_cka_256": np.ones((token_rows, 8), dtype=bool),
            "valid_s_128": np.ones((token_rows, 8), dtype=bool),
            "valid_s_256": np.ones((token_rows, 8), dtype=bool),
            "before_router_layer_numbers": np.tile(
                np.asarray(LAYERS, dtype=np.int16), (token_rows, 1)
            ),
            "after_router_layer_numbers": np.tile(
                np.asarray(LAYERS, dtype=np.int16), (token_rows, 1)
            ),
            "before_top4_id": np.tile(np.arange(4), (token_rows, 8, 1)),
            "after_top4_id": np.tile(np.arange(4), (token_rows, 8, 1)),
            "before_top4_weight": np.full((token_rows, 8, 4), 0.25, dtype=np.float32),
            "after_top4_weight": np.full((token_rows, 8, 4), 0.25, dtype=np.float32),
            "before_old_full_mass": np.full((token_rows, 8), 0.5, dtype=np.float32),
            "after_old_full_mass": np.full((token_rows, 8), 0.5, dtype=np.float32),
            "before_old_selected_mass": np.ones((token_rows, 8), dtype=np.float32),
            "after_old_selected_mass": np.ones((token_rows, 8), dtype=np.float32),
        }
        # Exercise the production convention: these are chunk start offsets,
        # not chunk ordinals.
        base["worst_chunk_id_128"][64:, :] = 64
        base["worst_s_chunk_id_128"][64:, :] = 64
        base["worst_chunk_id_256"][128:, :] = 128
        base["worst_s_chunk_id_256"][128:, :] = 128
        chunk_rows: list[tuple[int, int, int, int]] = []
        for scale, stride in ((128, 64), (256, 128)):
            for start in chunk_starts(length, scale, stride).tolist():
                chunk_rows.extend((scale, int(start), scale, layer) for layer in LAYERS)
        chunk_rows.extend((512, 0, length, layer) for layer in LAYERS)
        chunks = np.asarray(chunk_rows, dtype=np.int64)
        count = chunks.shape[0]
        chunk_columns = {
            "domain": ["code"] * count,
            "split": ["calibration"] * count,
            "chunk_uid": np.arange(count, dtype=np.int64) + 987654321,
            "window_uid": np.full(count, window_uid, dtype=np.int64),
            "sample_order": np.zeros(count, dtype=np.int64),
            "document_id": np.full(count, int(row["document_id"]), dtype=np.int64),
            "window_offset": np.full(count, int(row["window_offset"]), dtype=np.int64),
            "scale": chunks[:, 0],
            "chunk_start": chunks[:, 1],
            "chunk_length": chunks[:, 2],
            "layer": chunks[:, 3],
            "cka": np.full(count, 0.9, dtype=np.float32),
            "cka_permutation": np.full(count, 0.1, dtype=np.float32),
            "cka_random_pair": np.full(count, 0.1, dtype=np.float32),
            "random_pair_invalid_reason": np.zeros(count, dtype=np.uint8),
            "random_pair_donor_window_uid": np.full(
                count, 999_999_999, dtype=np.int64
            ),
            "cka_off": np.full(count, 0.7, dtype=np.float32),
            "diag_ratio": np.full(count, 0.2, dtype=np.float32),
            "centered_variance_x": np.ones(count, dtype=np.float32),
            "centered_variance_y": np.ones(count, dtype=np.float32),
            "k_norm": np.ones(count, dtype=np.float32),
            "l_norm": np.ones(count, dtype=np.float32),
            "invalid_reason": np.zeros(count, dtype=np.uint8),
            "t_invalid_reason": np.zeros(count, dtype=np.uint8),
            "offdiag_warning": np.zeros(count, dtype=bool),
            "c_i": [
                np.full(int(chunk_length), 0.9 / float(chunk_length), dtype=np.float32)
                for chunk_length in chunks[:, 2]
            ],
            "c_i_off": [
                np.full(int(chunk_length), 0.7 / float(chunk_length), dtype=np.float32)
                for chunk_length in chunks[:, 2]
            ],
            "s_i": [
                np.ones(int(chunk_length), dtype=np.float32)
                for chunk_length in chunks[:, 2]
            ],
            "r_i": [
                np.full(int(chunk_length), 0.9, dtype=np.float32)
                for chunk_length in chunks[:, 2]
            ],
        }
        # Build the canonical Pass-1 artifact first.  Pass-2 progress journals
        # are cryptographically bound to this exact identity.
        from cka_gt_pilot_runtime import (
            load_membership_statistics,
            save_membership_statistics_atomic,
        )

        dimension = 16
        wiki_calibration = prepared.windows[("wiki", "calibration")]
        source_windows = int(wiki_calibration.shape[0])
        source_tokens = int(
            wiki_calibration["window_length"].sum(dtype=np.int64)
        )
        sampled = np.arange(source_tokens, dtype="<i8")
        sampling_provenance = {
            "sampler": "numpy.PCG64_SeedSequence_choice_without_replacement_sorted",
            "seed_sequence_entropy_order": [
                "base_seed",
                "domain_id",
                "stage_id",
                "split_id",
            ],
            "base_seed": 1234,
            "domain": "wiki",
            "domain_id": 1,
            "split": "calibration",
            "split_id": 0,
            "covariance_stage_id": 2,
            "kmeans_reservoir_stage_id": 3,
            "covariance_entropy": [1234, 1, 2, 0],
            "kmeans_reservoir_entropy": [1234, 1, 3, 0],
            "covariance_sample_count": source_tokens,
            "kmeans_reservoir_count": source_tokens,
            "covariance_indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
            "kmeans_indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
        }
        init_size = min(source_tokens, 3 * 4096)
        initial_indices = list(range(64))
        init_pool = np.arange(init_size, dtype="<i8")
        statistics: dict[int, dict[str, Any]] = {}
        for layer in LAYERS:
            identity_matrix = np.eye(dimension, dtype=np.float32)
            prototypes = (
                np.arange(64 * dimension, dtype=np.float32).reshape(64, dimension)
                + np.float32(layer * 100_000)
            )
            statistics[layer] = {
                "mean": np.zeros(dimension, dtype=np.float32),
                "covariance": identity_matrix,
                "empirical_covariance": identity_matrix.copy(),
                "inverse_sqrt": identity_matrix.copy(),
                "covariance_eigenvalues": np.ones(dimension, dtype=np.float32),
                "prototypes": prototypes,
                "ledoit_wolf_shrinkage": 0.5,
                "all_token_count": source_tokens,
                "covariance_sample_count": source_tokens,
                "kmeans_sample_count": source_tokens,
                "kmeans_report": {
                    "algorithm": "repository_deterministic_minibatch_kmeans_v1",
                    "version": 1,
                    "n_clusters": 64,
                    "batch_size": 4096,
                    "n_init": 1,
                    "max_iter": 100,
                    "reassignment_ratio": 0.01,
                    "seed": 1234,
                    "init": "k-means++",
                    "init_size": init_size,
                    "init_pool_policy": "min(n_samples,max(3*batch_size,3*n_clusters))",
                    "initial_indices_by_run": [initial_indices],
                    "init_pool_indices_sha256_by_run": [
                        hashlib.sha256(init_pool.tobytes()).hexdigest()
                    ],
                    "initialization_compute": "shared_cpu_numpy_float32_d_squared",
                    "inertia": 1.0,
                    "cluster_update_counts": [1] * 64,
                    "compute_device": "cpu",
                },
                "sampling_provenance": sampling_provenance,
            }
        membership_path = root / "membership" / "wiki_calibration_membership.npz"
        pass1_source_metadata = {
            "analysis": "cka_gt_pilot_v1",
            "mode": "pass1",
            "domain": "wiki",
            "requested_split": "calibration",
            "dataset_prefix": str(prepared.config["wiki_prefix"]),
            "reference_load": str(prepared.config["before_checkpoint"]),
            "reference_step": 600,
            "prepared_config_content_sha256": prepared.config[
                "config_content_sha256"
            ],
            "total_windows": source_windows,
            "total_tokens": source_tokens,
            "seed": 1234,
            "max_windows": 0,
            "source_dataset_identity": prepared.config["source_dataset_identity"],
            "checkpoint_identity": prepared.config["checkpoint_identity"],
        }
        save_membership_statistics_atomic(
            membership_path, statistics, metadata=pass1_source_metadata
        )
        loaded_membership = load_membership_statistics(membership_path)
        membership_identity = loaded_membership.membership_identity
        pass1_worker = (
            root
            / "runtime"
            / "pass1"
            / "wiki"
            / "calibration"
            / "worker_000"
        )
        pass1_worker.mkdir(parents=True, exist_ok=True)
        (pass1_worker / "metadata.json").write_text(
            json.dumps(
                {
                    **pass1_source_metadata,
                    "completed": True,
                    "membership_statistics": str(membership_path.resolve()),
                    "membership_identity": membership_identity,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        def concatenate_columns(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name in parts[0]:
                values = [part[name] for part in parts]
                if isinstance(values[0], np.ndarray):
                    result[name] = np.concatenate(values, axis=0)
                else:
                    result[name] = [item for value in values for item in value]
            return result

        def columns_for_window(
            split: str, prepared_row: np.void
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            local_length = int(prepared_row["window_length"])
            if local_length != token_rows:
                raise AssertionError("self-test fixture assumes fixed 512-token windows")
            order = int(prepared_row["sample_order"])
            split_id = {"calibration": 0, "selection": 1, "test": 2}[split]
            uid = split_id * 100_000_000 + order
            token_part = {
                name: (value.copy() if isinstance(value, np.ndarray) else list(value))
                for name, value in base.items()
            }
            token_part.update(
                {
                    "domain": ["code"] * local_length,
                    "split": [split] * local_length,
                    "window_uid": np.full(local_length, uid, dtype=np.int64),
                    "sample_order": np.full(local_length, order, dtype=np.int64),
                    "source_window_index": np.full(
                        local_length,
                        int(prepared_row["source_window_index"]),
                        dtype=np.int64,
                    ),
                    "document_id": np.full(
                        local_length, int(prepared_row["document_id"]), dtype=np.int64
                    ),
                    "window_offset": np.full(
                        local_length, int(prepared_row["window_offset"]), dtype=np.int64
                    ),
                    "window_length": np.full(
                        local_length, local_length, dtype=np.int32
                    ),
                    "position": np.arange(local_length, dtype=np.int32),
                    "document_token_offset": int(prepared_row["window_offset"])
                    + np.arange(local_length, dtype=np.int64),
                    "eligible": np.arange(local_length)
                    < int(prepared_row["eligible_token_count"]),
                }
            )
            chunk_part = {
                name: (value.copy() if isinstance(value, np.ndarray) else list(value))
                for name, value in chunk_columns.items()
            }
            chunk_part.update(
                {
                    "domain": ["code"] * count,
                    "split": [split] * count,
                    "chunk_uid": np.int64(uid) * np.int64(10_000)
                    + np.arange(count, dtype=np.int64),
                    "window_uid": np.full(count, uid, dtype=np.int64),
                    "sample_order": np.full(count, order, dtype=np.int64),
                    "document_id": np.full(
                        count, int(prepared_row["document_id"]), dtype=np.int64
                    ),
                    "window_offset": np.full(
                        count, int(prepared_row["window_offset"]), dtype=np.int64
                    ),
                }
            )
            return token_part, chunk_part

        open_token_parts: list[dict[str, Any]] = []
        open_chunk_parts: list[dict[str, Any]] = []
        test_token_parts: list[dict[str, Any]] = []
        test_chunk_parts: list[dict[str, Any]] = []
        for split in SPLIT_NAMES:
            for prepared_row in prepared.windows[("code", split)]:
                token_part, chunk_part = columns_for_window(split, prepared_row)
                destination_tokens = test_token_parts if split == "test" else open_token_parts
                destination_chunks = test_chunk_parts if split == "test" else open_chunk_parts
                destination_tokens.append(token_part)
                destination_chunks.append(chunk_part)

        common_metadata = {
            "analysis": "cka_gt_pilot_v1",
            "mode": "pass2",
            "domain": "code",
            "requested_split": "all",
            "worker_index": 0,
            "worker_count": 1,
            "max_windows": 0,
            "membership_identity": membership_identity,
            "source_dataset_identity": prepared.config["source_dataset_identity"],
            "checkpoint_identity": prepared.config["checkpoint_identity"],
            "random_pair_null": {
                "policy_version": "deterministic_full_after_donor_cache_v1",
                "batch_ge_2": "within_batch_seeded_cyclic_derangement_no_fixed_points",
                "singleton": (
                    "sha256(window_uid,scale,chunk_start)_indexed_nonself_donor_from_"
                    "first_8_full_after_windows"
                ),
                "donor_cache_scope": "per_split_per_worker",
                "donor_cache_size": 8,
                "donor_cache_storage": "memory_only_rebuilt_by_forward_on_resume",
                "candidate_window_uids_by_split": {
                    split: (
                        np.int64({"calibration": 0, "selection": 1, "test": 2}[split])
                        * np.int64(100_000_000)
                        + prepared.windows[("code", split)][
                            prepared.windows[("code", split)]["window_length"] == 512
                        ]["sample_order"].astype(np.int64)[:8]
                    ).tolist()
                    for split in SPLIT_NAMES
                },
                "reason_codes": {
                    "0": "valid",
                    "1": "singleton_no_donor_cache",
                    "2": "singleton_no_nonself_full_window_donor",
                    "3": "paired_cka_metric_invalid",
                },
                "coverage_columns": [
                    "random_pair_invalid_reason",
                    "random_pair_donor_window_uid",
                    "cka_random_pair",
                ],
                "selector_dependency": False,
            },
        }
        worker_base = root / "runtime" / "pass2" / "code" / "all" / "worker_000"
        paired_manifests: dict[str, Any] = {"open": None, "test": None}
        for stream, token_parts, chunk_parts, splits, sealed in (
            (
                "open",
                open_token_parts,
                open_chunk_parts,
                ["calibration", "selection"],
                False,
            ),
            ("test", test_token_parts, test_chunk_parts, ["test"], True),
        ):
            stream_tokens = concatenate_columns(token_parts)
            stream_chunks = concatenate_columns(chunk_parts)
            label_name = f"code_all_{stream}_worker_000"
            if stream == "open":
                token_dir = root / "token_metrics" / label_name
                chunk_dir = root / "chunk_metrics" / label_name
            else:
                token_dir = root / "sealed_test" / "raw" / "token_metrics" / label_name
                chunk_dir = root / "sealed_test" / "raw" / "chunk_metrics" / label_name
            _write_paired_fixture(
                root,
                worker_base / "paired_progress" / stream,
                token_dir,
                chunk_dir,
                stream_tokens,
                stream_chunks,
                metadata={
                    **common_metadata,
                    "metric_stream": stream,
                    "stream_splits": splits,
                    "test_metrics_sealed": sealed,
                    "stream_window_count": len(token_parts),
                    "stream_token_count": int(len(stream_tokens["window_uid"])),
                },
            )
            paired_manifests[stream] = json.loads(
                (
                    worker_base
                    / "paired_progress"
                    / stream
                    / "progress.json"
                ).read_text(encoding="utf-8")
            )
        (worker_base / "metadata.json").write_text(
            json.dumps(
                {
                    **common_metadata,
                    "completed": True,
                    "paired_manifests": paired_manifests,
                    "elapsed_seconds": 2.0,
                    "cumulative_elapsed_seconds": 2.0,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        report = validate_pilot(
            root,
            allow_incomplete=True,
            deep=True,
            datasets=datasets,
            expected_hidden_size=dimension,
        )
        if not report.ok:
            raise AssertionError(json.dumps(report.serializable(), indent=2))
        if not any("missing" in warning for warning in report.warnings):
            raise AssertionError("self-test expected incomplete coverage/seal warnings")

        # Regression coverage for the vectorized ragged-contribution audit.
        # Keep this independent of the runtime writer so a future Arrow/list
        # refactor cannot silently drop the exact CKA/T identities.
        valid_chunk_batch = _fixture_table(chunk_columns).to_batches()[0]
        contribution_report = ValidationReport("synthetic-contributions", True, True)
        _validate_chunk_batch(
            valid_chunk_batch, contribution_report, "synthetic-contributions"
        )
        if not contribution_report.ok:
            raise AssertionError(
                "self-test rejected valid vectorized contributions: "
                + json.dumps(contribution_report.serializable(), indent=2)
            )

        corrupt_contributions = {
            name: (value.copy() if isinstance(value, np.ndarray) else list(value))
            for name, value in chunk_columns.items()
        }
        corrupt_contributions["c_i"] = [
            np.asarray(value, dtype=np.float32).copy()
            for value in chunk_columns["c_i"]
        ]
        corrupt_contributions["s_i"] = [
            np.asarray(value, dtype=np.float32).copy()
            for value in chunk_columns["s_i"]
        ]
        corrupt_contributions["c_i"][0][0] += np.float32(0.1)
        corrupt_contributions["s_i"][1][0] += np.float32(1.0)
        corrupt_batch = _fixture_table(corrupt_contributions).to_batches()[0]
        corrupt_contribution_report = ValidationReport(
            "corrupt-contributions", True, True
        )
        _validate_chunk_batch(
            corrupt_batch,
            corrupt_contribution_report,
            "corrupt-contributions",
        )
        if corrupt_contribution_report.ok or not any(
            "sum(c_i) != CKA" in error
            for error in corrupt_contribution_report.errors
        ) or not any(
            "mean(s_i) != 1" in error
            for error in corrupt_contribution_report.errors
        ):
            raise AssertionError(
                "self-test failed to reject corrupt vectorized contributions: "
                + json.dumps(corrupt_contribution_report.serializable(), indent=2)
            )

        # A corrupt committed value must remain an error even when incomplete
        # output is allowed.
        bad = dict(base)
        bad["cosine"] = np.asarray(base["cosine"]).copy()
        bad["cosine"][0, 0] = np.nan
        corrupt_root = Path(temporary_dir) / "corrupt"
        corrupt_dir = corrupt_root / "runtime" / "pass2" / "code" / "calibration" / "worker_000" / "token_metrics"
        _write_arrow_fixture(corrupt_dir, TOKEN_STREAM_KIND, bad)
        corrupt_report = ValidationReport(str(corrupt_root), True, True)
        _validate_runtime(corrupt_root, corrupt_report, prepared)
        if corrupt_report.ok or not any("NaN/Inf" in error for error in corrupt_report.errors):
            raise AssertionError("self-test failed to reject non-finite committed token metrics")
    print("validate_cka_gt_pilot self-test: PASS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--expected-hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    if args.root is None:
        raise SystemExit("root is required unless --self-test is used")
    report = validate_pilot(
        args.root,
        allow_incomplete=args.allow_incomplete,
        deep=args.deep,
        expected_hidden_size=args.expected_hidden_size,
    )
    payload = report.serializable()
    if args.output_json is not None:
        _json_dump(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
