#!/usr/bin/env python3
"""Analyze the merged, threshold-free full Code CKA census.

This is deliberately a *threshold review* stage, not a GT writer.  It applies
the three Wiki-calibrated candidate bundles (95/97/99) to the deterministic
uniform token reservoir and combines those joint estimates with exact merged
histograms.  It never opens the physically sealed pilot test split.

Primary selection follows the frozen pilot definition:

* B128, B256, T128, T256, rel-L2 and |log norm ratio| are each evaluated with
  their own layer consensus (>=7/8; 6/6 when exactly six layers are valid);
* B = B128 AND B256, T = T128 AND T256, M = rel-L2 AND |log-r|;
* the candidate is B AND T AND M;
* requiring all six conditions on the same layer is diagnostic only.

The output includes selection/marginal tables, pairwise overlaps, a document
cluster-bootstrap confidence interval for full-corpus counts, concentration
and token-ID repetition audits, threshold-overlay histograms, machine JSON,
and REPORT.md.  Exact GT occurrences still require a targeted second pass
after a human locks one candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cka_gt_full_census_threshold_review_v1"
EXPECTED_CENSUS_SCHEMA = "cka_gt_full_census_merged_summary_v2"
EXPECTED_CONFIG_SCHEMA = "cka_gt_pilot_postprocess_v1"
LEVELS = (95, 97, 99)
LAYERS = tuple(range(2, 10))
SCALES = (128, 256)
REQUIRED_RESERVOIR_FIELDS = {
    "priority", "sample_order", "source_window_index", "document_id",
    "window_offset", "position", "token_id", "b_min", "t_min",
    "rel_l2", "abs_log_r",
}
METRIC_SPECS = {
    "raw_b_128": ("B", "ge", 128),
    "raw_b_256": ("B", "ge", 256),
    "token_min_b_128": ("B", "ge", 128),
    "token_min_b_256": ("B", "ge", 256),
    "token_min_t_128": ("T", "ge", 128),
    "token_min_t_256": ("T", "ge", 256),
    "relative_l2": ("L2", "le", None),
    "abs_log_r": ("R", "le", None),
}


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(
        _jsonable(payload), ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False,
    ) + "\n"
    _atomic_text(path, text)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: _jsonable(row.get(key)) for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    result = {"path": str(path.resolve()), "size_bytes": int(path.stat().st_size)}
    if hash_file:
        result["sha256"] = _sha256(path)
    return result


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


@dataclass(frozen=True)
class CandidateBundle:
    level: int
    b: Mapping[int, np.ndarray]
    t: Mapping[int, np.ndarray]
    rel_l2: np.ndarray
    abs_log_r: np.ndarray


def _threshold_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise RuntimeError(f"{name} must be eight finite layer thresholds")
    return result


def load_frozen_bundles(config_path: str | Path) -> tuple[dict[str, Any], dict[int, CandidateBundle]]:
    """Load only open, frozen calibration cuts from analysis_config.

    Deliberately do not follow token/chunk/sealed-test paths present in the
    config.  Candidate thresholds were derived from Wiki calibration by the
    pilot's exact-disk-backed raw-unit quantile builder.
    """

    path = Path(config_path)
    config = json.loads(path.read_text())
    schema = config.get("schema")
    if schema not in (EXPECTED_CONFIG_SCHEMA, None):
        raise RuntimeError(f"unsupported pilot analysis_config schema: {schema}")
    raw = config.get("candidate_thresholds")
    if not isinstance(raw, Mapping):
        raise RuntimeError("analysis_config has no frozen candidate_thresholds")
    bundles: dict[int, CandidateBundle] = {}
    for level in LEVELS:
        item = raw.get(str(level), raw.get(level))
        if not isinstance(item, Mapping):
            raise RuntimeError(f"missing candidate bundle {level}")
        b_raw, t_raw = item.get("B_lower_threshold"), item.get("T_lower_threshold")
        if not isinstance(b_raw, Mapping) or not isinstance(t_raw, Mapping):
            raise RuntimeError(f"bundle {level} is missing B/T scale thresholds")
        bundles[level] = CandidateBundle(
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
                item.get("relative_l2_upper_threshold"), f"bundle{level}.relative_l2"
            ),
            abs_log_r=_threshold_vector(
                item.get("abs_log_r_upper_threshold"), f"bundle{level}.abs_log_r"
            ),
        )

    # Higher Wiki recall must be weakly less strict.  Fail closed if a frozen
    # table was accidentally reordered or overwritten.
    for strict, loose in zip(LEVELS, LEVELS[1:]):
        left, right = bundles[strict], bundles[loose]
        for scale in SCALES:
            if np.any(left.b[scale] < right.b[scale] - 1e-10):
                raise RuntimeError("B thresholds are not nested 95 -> 97 -> 99")
            if np.any(left.t[scale] < right.t[scale] - 1e-10):
                raise RuntimeError("T thresholds are not nested 95 -> 97 -> 99")
        if np.any(left.rel_l2 > right.rel_l2 + 1e-10):
            raise RuntimeError("rel-L2 thresholds are not nested 95 -> 97 -> 99")
        if np.any(left.abs_log_r > right.abs_log_r + 1e-10):
            raise RuntimeError("|log-r| thresholds are not nested 95 -> 97 -> 99")
    return config, bundles


def bind_open_calibration_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the exact open raw-quantile artifact without scanning sealed data."""

    raw_value = config.get("raw_quantiles")
    if not raw_value:
        raise RuntimeError("analysis_config does not declare its open raw_quantiles artifact")
    raw_path = Path(str(raw_value))
    manifest_path = raw_path.with_name("raw_unit_quantile_accumulator_manifest.json")
    if not raw_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"frozen open Wiki-calibration artifact/manifest missing: {raw_path}, {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "cka_gt_raw_unit_exact_quantiles_v1":
        raise RuntimeError("unsupported raw Wiki-calibration quantile manifest")
    raw_sha = _sha256(raw_path)
    if manifest.get("output_sha256") != raw_sha:
        raise RuntimeError("raw Wiki-calibration quantile SHA256 differs from its manifest")
    expected_inventory = config.get("input_inventory_digest_sha256")
    if expected_inventory and manifest.get("input_inventory_digest_sha256") != expected_inventory:
        raise RuntimeError("calibration quantile/input inventory binding differs from analysis_config")
    return {
        "raw_quantiles": _file_identity(raw_path),
        "raw_quantile_manifest": _file_identity(manifest_path),
        "raw_quantile_schema": manifest["schema"],
        "method": "exact_disk_backed",
        "wiki_calibration_groups_bound": sum(
            1 for row in manifest.get("groups", [])
            if row.get("domain") == "wiki" and row.get("split") == "calibration"
        ),
        "sealed_test_content_read": False,
    }


def _serializable_bundle(bundle: CandidateBundle) -> dict[str, Any]:
    return {
        "level": bundle.level,
        "B_lower_threshold": {str(scale): bundle.b[scale].tolist() for scale in SCALES},
        "T_lower_threshold": {str(scale): bundle.t[scale].tolist() for scale in SCALES},
        "relative_l2_upper_threshold": bundle.rel_l2.tolist(),
        "abs_log_r_upper_threshold": bundle.abs_log_r.tolist(),
    }


@dataclass(frozen=True)
class Consensus:
    eligible: np.ndarray
    passed: np.ndarray
    n_valid: np.ndarray
    pass_count: np.ndarray
    valid_by_layer: np.ndarray
    pass_by_layer: np.ndarray


def condition_consensus(
    values: np.ndarray,
    thresholds: np.ndarray,
    *,
    comparison: str,
    reject_negative: bool = False,
) -> Consensus:
    values = np.asarray(values, dtype=np.float64)
    thresholds = _threshold_vector(thresholds, "condition threshold")
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("condition values must have shape [token,8]")
    valid = np.isfinite(values)
    if comparison == "ge":
        passed = valid & (values >= thresholds[None, :])
    elif comparison == "le":
        passed = valid & (values <= thresholds[None, :])
    else:
        raise ValueError("comparison must be ge or le")
    if reject_negative:
        passed &= values >= 0.0
    n_valid = valid.sum(axis=1, dtype=np.int16)
    pass_count = passed.sum(axis=1, dtype=np.int16)
    required = np.minimum(n_valid, 7)
    eligible = n_valid >= 6
    selected = eligible & (pass_count >= required)
    return Consensus(eligible, selected, n_valid, pass_count, valid, passed)


def evaluate_candidate(scores: np.ndarray, bundle: CandidateBundle) -> dict[str, np.ndarray]:
    """Evaluate one reservoir slice with the exact frozen pilot semantics."""

    b_values = {scale: scores["b_min"][:, index] for index, scale in enumerate(SCALES)}
    t_values = {scale: scores["t_min"][:, index] for index, scale in enumerate(SCALES)}
    b = {
        scale: condition_consensus(b_values[scale], bundle.b[scale], comparison="ge")
        for scale in SCALES
    }
    t = {
        scale: condition_consensus(
            t_values[scale], bundle.t[scale], comparison="ge", reject_negative=True
        )
        for scale in SCALES
    }
    l2 = condition_consensus(scores["rel_l2"], bundle.rel_l2, comparison="le")
    logr = condition_consensus(scores["abs_log_r"], bundle.abs_log_r, comparison="le")
    b_eligible = b[128].eligible & b[256].eligible
    t_eligible = t[128].eligible & t[256].eligible
    b_pass = b_eligible & b[128].passed & b[256].passed
    t_pass = t_eligible & t[128].passed & t[256].passed
    m_eligible = l2.eligible & logr.eligible
    m_pass = m_eligible & l2.passed & logr.passed
    full_eligible = b_eligible & t_eligible & m_eligible
    full = full_eligible & b_pass & t_pass & m_pass

    same_valid = np.ones((len(scores), 8), dtype=np.bool_)
    same_pass = np.ones((len(scores), 8), dtype=np.bool_)
    for result in (*b.values(), *t.values(), l2, logr):
        same_valid &= result.valid_by_layer
        same_pass &= result.pass_by_layer
    same_n_valid = same_valid.sum(axis=1, dtype=np.int16)
    same_pass_count = (same_pass & same_valid).sum(axis=1, dtype=np.int16)
    same_required = np.minimum(same_n_valid, 7)
    same_eligible = same_n_valid >= 6
    same_selected = same_eligible & (same_pass_count >= same_required)

    return {
        "B128_eligible": b[128].eligible,
        "B128": b[128].passed,
        "B256_eligible": b[256].eligible,
        "B256": b[256].passed,
        "B_eligible": b_eligible,
        "B": b_pass,
        "T128_eligible": t[128].eligible,
        "T128": t[128].passed,
        "T256_eligible": t[256].eligible,
        "T256": t[256].passed,
        "T_eligible": t_eligible,
        "T": t_pass,
        "L2_eligible": l2.eligible,
        "L2": l2.passed,
        "R_eligible": logr.eligible,
        "R": logr.passed,
        "M_eligible": m_eligible,
        "M": m_pass,
        "B+T": b_pass & t_pass,
        "B+M": b_pass & m_pass,
        "B+T+M": full,
        "full_eligible": full_eligible,
        "selected": full,
        "same_eligible": same_eligible,
        "same_selected": same_selected,
    }


def _validate_reservoir_dtype(scores: np.ndarray) -> None:
    if scores.ndim != 1 or scores.dtype.names is None:
        raise RuntimeError("token_scores must be a 1-D structured array")
    missing = REQUIRED_RESERVOIR_FIELDS - set(scores.dtype.names)
    if missing:
        raise RuntimeError(f"token_scores missing fields: {sorted(missing)}")
    if scores.dtype["b_min"].shape != (2, 8) or scores.dtype["t_min"].shape != (2, 8):
        raise RuntimeError("b_min/t_min reservoir shapes must be [2,8]")
    if scores.dtype["rel_l2"].shape != (8,) or scores.dtype["abs_log_r"].shape != (8,):
        raise RuntimeError("magnitude reservoir shapes must be [8]")


def analyze_reservoir(
    scores: np.ndarray,
    bundles: Mapping[int, CandidateBundle],
    *,
    chunk_rows: int = 250_000,
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    _validate_reservoir_dtype(scores)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    masks = {level: np.zeros(len(scores), dtype=np.bool_) for level in LEVELS}
    same_masks = {level: np.zeros(len(scores), dtype=np.bool_) for level in LEVELS}
    count_names = (
        "B128_eligible", "B128", "B256_eligible", "B256", "B_eligible", "B",
        "T128_eligible", "T128", "T256_eligible", "T256", "T_eligible", "T",
        "L2_eligible", "L2", "R_eligible", "R", "M_eligible", "M",
        "B+T", "B+M", "B+T+M", "full_eligible", "selected",
        "same_eligible", "same_selected",
    )
    counts = {level: {name: 0 for name in count_names} for level in LEVELS}
    for start in range(0, len(scores), chunk_rows):
        stop = min(len(scores), start + chunk_rows)
        part = scores[start:stop]
        for level in LEVELS:
            result = evaluate_candidate(part, bundles[level])
            for name in count_names:
                counts[level][name] += int(np.count_nonzero(result[name]))
            masks[level][start:stop] = result["selected"]
            same_masks[level][start:stop] = result["same_selected"]

    summaries: dict[int, dict[str, Any]] = {}
    total = len(scores)
    for level in LEVELS:
        selected, same = masks[level], same_masks[level]
        intersection = int(np.count_nonzero(selected & same))
        union = int(np.count_nonzero(selected | same))
        summary = {
            "level": level,
            "reservoir_rows": total,
            "counts": counts[level],
            "fractions_of_all": {
                name: _ratio(value, total) for name, value in counts[level].items()
            },
            "fractions_of_eligible": {
                "B": _ratio(counts[level]["B"], counts[level]["B_eligible"]),
                "T": _ratio(counts[level]["T"], counts[level]["T_eligible"]),
                "L2": _ratio(counts[level]["L2"], counts[level]["L2_eligible"]),
                "R": _ratio(counts[level]["R"], counts[level]["R_eligible"]),
                "M": _ratio(counts[level]["M"], counts[level]["M_eligible"]),
                "selected": _ratio(counts[level]["selected"], counts[level]["full_eligible"]),
            },
            "same_layer": {
                "condition_specific_count": int(selected.sum()),
                "same_layer_count": int(same.sum()),
                "intersection": intersection,
                "union": union,
                "jaccard": _ratio(intersection, union) if union else 1.0,
                "manual_review_required": bool(union and intersection / union < 0.9),
            },
        }
        summaries[level] = summary
    return summaries, masks


def pairwise_jaccard_rows(masks: Mapping[int, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, left in enumerate(LEVELS):
        for right in LEVELS[index + 1:]:
            intersection = int(np.count_nonzero(masks[left] & masks[right]))
            union = int(np.count_nonzero(masks[left] | masks[right]))
            rows.append({
                "left_bundle": left,
                "right_bundle": right,
                "left_count": int(masks[left].sum()),
                "right_count": int(masks[right].sum()),
                "intersection": intersection,
                "union": union,
                "jaccard": _ratio(intersection, union) if union else 1.0,
                "left_subset_of_right": bool(np.all(~masks[left] | masks[right])),
            })
    return rows


def _gini_positive(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    values = np.sort(values[values > 0])
    if not values.size or values.sum() == 0:
        return float("nan")
    index = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2.0 * np.dot(index, values) / values.sum() - (values.size + 1)) / values.size)


def concentration_metrics(ids: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(ids)
    if not ids.size:
        return {
            "selected_rows": 0, "active_units": 0, "max_share": None,
            "top5_share": None, "top10_share": None, "hhi": None,
            "effective_units": None, "gini_among_active": None,
        }
    unique_ids, counts = np.unique(ids, return_counts=True)
    ordered = np.sort(counts.astype(np.float64))[::-1]
    shares = ordered / ordered.sum()
    hhi = float(np.square(shares).sum())
    top = np.argsort(counts)[-10:][::-1]
    return {
        "selected_rows": int(ids.size),
        "active_units": int(unique_ids.size),
        "max_share": float(shares[0]),
        "top5_share": float(shares[:5].sum()),
        "top10_share": float(shares[:10].sum()),
        "hhi": hhi,
        "effective_units": float(1.0 / hhi),
        "gini_among_active": _gini_positive(counts),
        "median_rows_per_active_unit": float(np.median(counts)),
        "p95_rows_per_active_unit": float(np.quantile(counts, 0.95)),
        "top_units": [
            {"id": int(unique_ids[item]), "count": int(counts[item]),
             "share": float(counts[item] / ids.size)}
            for item in top
        ],
        "definition": "shares/Gini are among selected reservoir occurrences and active units only",
    }


def token_repetition_metrics(token_ids: np.ndarray, window_ids: np.ndarray) -> dict[str, Any]:
    token_ids, window_ids = np.asarray(token_ids), np.asarray(window_ids)
    if token_ids.size != window_ids.size:
        raise ValueError("token/window identity lengths differ")
    if not token_ids.size:
        return {"selected_rows": 0, "unique_token_ids": 0, "entropy_bits": None}
    unique, counts = np.unique(token_ids, return_counts=True)
    probabilities = counts.astype(np.float64) / token_ids.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    top = np.argsort(counts)[-10:][::-1]

    # Reservoir rows are sparse; this is explicitly a same-window token-ID
    # duplicate diagnostic, not an n-gram statistic.
    pair = np.empty(token_ids.size, dtype=[("window", "<i8"), ("token", "<i8")])
    pair["window"] = window_ids.astype(np.int64, copy=False)
    pair["token"] = token_ids.astype(np.int64, copy=False)
    within_unique = np.unique(pair).size
    repeated_occurrences = int(counts[counts > 1].sum())
    return {
        "selected_rows": int(token_ids.size),
        "unique_token_ids": int(unique.size),
        "unique_token_ratio": float(unique.size / token_ids.size),
        "entropy_bits": entropy,
        "normalized_entropy": float(entropy / math.log2(unique.size)) if unique.size > 1 else 0.0,
        "effective_token_vocabulary": float(2.0 ** entropy),
        "token_id_hhi": float(np.square(probabilities).sum()),
        "top_token_share": float(probabilities.max()),
        "top10_token_share": float(np.sort(probabilities)[-10:].sum()),
        "occurrences_whose_token_id_repeats": repeated_occurrences,
        "repeated_token_occurrence_fraction": float(repeated_occurrences / token_ids.size),
        "excess_token_repeat_fraction": float((token_ids.size - unique.size) / token_ids.size),
        "within_window_excess_same_token_fraction": float((token_ids.size - within_unique) / token_ids.size),
        "top_token_ids": [
            {"token_id": int(unique[item]), "count": int(counts[item]),
             "share": float(counts[item] / token_ids.size)}
            for item in top
        ],
        "limitation": "uniform sparse reservoir supports token-ID repetition, not exact contiguous n-gram repetition",
    }


def document_bootstrap_intervals(
    document_ids: np.ndarray,
    masks: Mapping[int, np.ndarray],
    *,
    full_eligible_tokens: int,
    repetitions: int,
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Ordinary document-cluster bootstrap of token-weighted coverage."""

    if repetitions < 20:
        raise ValueError("document bootstrap requires at least 20 repetitions")
    documents, inverse, totals = np.unique(
        np.asarray(document_ids, dtype=np.int64), return_inverse=True, return_counts=True
    )
    selected_by_doc = np.stack(
        [np.bincount(inverse, weights=masks[level].astype(np.int8), minlength=len(documents))
         for level in LEVELS],
        axis=1,
    ).astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, len(LEVELS)), dtype=np.float64)
    for repeat in range(repetitions):
        sampled = rng.integers(0, len(documents), size=len(documents), dtype=np.int64)
        weights = np.bincount(sampled, minlength=len(documents)).astype(np.float64, copy=False)
        denominator = float(np.dot(weights, totals))
        samples[repeat] = weights @ selected_by_doc / denominator if denominator else np.nan
    result: dict[int, dict[str, Any]] = {}
    for column, level in enumerate(LEVELS):
        point = float(masks[level].mean()) if len(document_ids) else float("nan")
        low, high = np.nanquantile(samples[:, column], [0.025, 0.975])
        result[level] = {
            "reservoir_coverage": point,
            "coverage_ci95": [float(low), float(high)],
            "estimated_full_count": int(round(point * full_eligible_tokens)),
            "estimated_full_count_ci95": [
                int(round(float(low) * full_eligible_tokens)),
                int(round(float(high) * full_eligible_tokens)),
            ],
            "full_eligible_tokens": int(full_eligible_tokens),
            "bootstrap_documents": int(len(documents)),
            "bootstrap_repetitions": int(repetitions),
            "bootstrap_seed": int(seed),
            "method": "ordinary cluster bootstrap: resample observed document IDs with replacement; token-weighted ratio",
        }
    return result


def threshold_rows(bundles: Mapping[int, CandidateBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        bundle = bundles[level]
        for layer_index, layer in enumerate(LAYERS):
            for scale in SCALES:
                rows.append({"bundle": level, "condition": "B", "scale": scale,
                             "layer": layer, "comparison": ">=", "threshold": bundle.b[scale][layer_index]})
                rows.append({"bundle": level, "condition": "T", "scale": scale,
                             "layer": layer, "comparison": ">= and value >= 0",
                             "threshold": bundle.t[scale][layer_index]})
            rows.append({"bundle": level, "condition": "relative_l2", "scale": "",
                         "layer": layer, "comparison": "<=", "threshold": bundle.rel_l2[layer_index]})
            rows.append({"bundle": level, "condition": "abs_log_r", "scale": "",
                         "layer": layer, "comparison": "<=", "threshold": bundle.abs_log_r[layer_index]})
    return rows


def _metric_threshold(bundle: CandidateBundle, metric: str, layer_index: int) -> float:
    kind, _comparison, scale = METRIC_SPECS[metric]
    if kind == "B":
        return float(bundle.b[int(scale)][layer_index])
    if kind == "T":
        return float(bundle.t[int(scale)][layer_index])
    if kind == "L2":
        return float(bundle.rel_l2[layer_index])
    return float(bundle.abs_log_r[layer_index])


def histogram_threshold_bounds(
    histograms: Mapping[str, np.ndarray], bundles: Mapping[int, CandidateBundle]
) -> list[dict[str, Any]]:
    """Conservative/liberal exact-count bounds for threshold-crossing bins."""

    rows: list[dict[str, Any]] = []
    for metric, (_kind, comparison, _scale) in METRIC_SPECS.items():
        required = [
            f"hist__{metric}__counts", f"hist__{metric}__edges",
            f"hist__{metric}__underflow", f"hist__{metric}__overflow",
            f"hist__{metric}__nonfinite",
        ]
        if any(key not in histograms for key in required):
            continue
        counts = np.asarray(histograms[required[0]], dtype=np.int64)
        edges = np.asarray(histograms[required[1]], dtype=np.float64)
        under = np.asarray(histograms[required[2]], dtype=np.int64)
        over = np.asarray(histograms[required[3]], dtype=np.int64)
        nonfinite = np.asarray(histograms[required[4]], dtype=np.int64)
        if counts.ndim != 2 or counts.shape[0] != 8 or edges.size != counts.shape[1] + 1:
            raise RuntimeError(f"malformed merged histogram for {metric}")
        for level in LEVELS:
            for layer_index, layer in enumerate(LAYERS):
                threshold = _metric_threshold(bundles[level], metric, layer_index)
                row = counts[layer_index]
                left, right = edges[:-1], edges[1:]
                if comparison == "ge":
                    guaranteed = int(row[left >= threshold].sum())
                    possible = int(row[right > threshold].sum())
                    if threshold <= edges[-1]:
                        guaranteed += int(over[layer_index])
                        possible += int(over[layer_index])
                    if threshold < edges[0]:
                        possible += int(under[layer_index])
                else:
                    guaranteed = int(row[right <= threshold].sum())
                    possible = int(row[left <= threshold].sum())
                    if threshold >= edges[0]:
                        guaranteed += int(under[layer_index])
                        possible += int(under[layer_index])
                    if threshold > edges[-1]:
                        possible += int(over[layer_index])
                finite = int(row.sum() + under[layer_index] + over[layer_index])
                eligible = finite + int(nonfinite[layer_index])
                rows.append({
                    "metric": metric, "bundle": level, "layer": layer,
                    "comparison": comparison, "threshold": threshold,
                    "finite_count": finite, "nonfinite_count": int(nonfinite[layer_index]),
                    "pass_count_conservative": guaranteed,
                    "pass_count_liberal": possible,
                    "pass_fraction_conservative": _ratio(guaranteed, eligible),
                    "pass_fraction_liberal": _ratio(possible, eligible),
                    "ambiguous_crossing_bin_count": possible - guaranteed,
                    "note": "bounds differ only by the threshold-crossing histogram bin/tail",
                })
    return rows


def _svg_histogram_panels(
    path: Path,
    *,
    metric: str,
    counts: np.ndarray,
    edges: np.ndarray,
    bundles: Mapping[int, CandidateBundle],
    zoom: bool,
) -> None:
    width, height = 1240, 930
    panel_w, panel_h = 395, 275
    colors = {95: "#dc2626", 97: "#2563eb", 99: "#16a34a"}
    all_thresholds = [
        _metric_threshold(bundles[level], metric, layer_index)
        for level in LEVELS for layer_index in range(8)
    ]
    if zoom:
        low_t, high_t = min(all_thresholds), max(all_thresholds)
        pad = max((high_t - low_t) * 1.5, (edges[-1] - edges[0]) * 0.015)
        xmin, xmax = max(edges[0], low_t - pad), min(edges[-1], high_t + pad)
    else:
        xmin, xmax = float(edges[0]), float(edges[-1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="620" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(metric)}: full Code histogram + frozen Wiki cuts ({"threshold zoom" if zoom else "full range"})</text>',
    ]
    for layer_index, layer in enumerate(LAYERS):
        col, row_index = layer_index % 3, layer_index // 3
        x0, y0 = 50 + col * panel_w, 55 + row_index * panel_h
        plot_w, plot_h = 330, 205
        centers = (edges[:-1] + edges[1:]) * 0.5
        visible = (centers >= xmin) & (centers <= xmax)
        values = counts[layer_index].astype(np.float64)
        total = max(values.sum(), 1.0)
        fractions = values / total
        peak = max(float(fractions[visible].max(initial=0.0)), 1e-15)
        sampled = np.flatnonzero(visible)
        if sampled.size > 700:
            sampled = sampled[:: int(math.ceil(sampled.size / 700))]
        points = []
        for index in sampled:
            x = x0 + plot_w * (centers[index] - xmin) / max(xmax - xmin, 1e-12)
            y = y0 + plot_h * (1.0 - fractions[index] / peak)
            points.append(f"{x:.2f},{y:.2f}")
        parts.extend([
            f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#777"/>',
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#111827" stroke-width="1.2"/>',
            f'<text x="{x0+5}" y="{y0+17}" font-family="sans-serif" font-size="14">Layer {layer}</text>',
            f'<text x="{x0}" y="{y0+plot_h+18}" font-family="sans-serif" font-size="11">{xmin:.5g}</text>',
            f'<text x="{x0+plot_w}" y="{y0+plot_h+18}" text-anchor="end" font-family="sans-serif" font-size="11">{xmax:.5g}</text>',
        ])
        for level in LEVELS:
            value = _metric_threshold(bundles[level], metric, layer_index)
            if xmin <= value <= xmax:
                x = x0 + plot_w * (value - xmin) / max(xmax - xmin, 1e-12)
                parts.append(
                    f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0+plot_h}" stroke="{colors[level]}" stroke-width="1.2" stroke-dasharray="4,3"><title>Wiki bundle {level}: {value:.9g}</title></line>'
                )
    for index, level in enumerate(LEVELS):
        x = 790 + index * 135
        parts.append(f'<line x1="{x}" y1="900" x2="{x+25}" y2="900" stroke="{colors[level]}" stroke-width="2"/>')
        parts.append(f'<text x="{x+31}" y="904" font-family="sans-serif" font-size="12">Wiki {level}</text>')
    parts.append("</svg>\n")
    _atomic_text(path, "\n".join(parts))


def write_histogram_overlays(
    output_dir: Path,
    histograms: Mapping[str, np.ndarray],
    bundles: Mapping[int, CandidateBundle],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    for metric in METRIC_SPECS:
        counts_key, edges_key = f"hist__{metric}__counts", f"hist__{metric}__edges"
        if counts_key not in histograms or edges_key not in histograms:
            continue
        counts, edges = np.asarray(histograms[counts_key]), np.asarray(histograms[edges_key])
        for zoom in (False, True):
            path = output_dir / f"{metric}_{'zoom' if zoom else 'full'}.svg"
            _svg_histogram_panels(
                path, metric=metric, counts=counts, edges=edges,
                bundles=bundles, zoom=zoom,
            )
            artifacts.append(str(path))
    return artifacts


def _recommendation(
    summaries: Mapping[int, Mapping[str, Any]],
    bootstrap: Mapping[int, Mapping[str, Any]],
    concentrations: Mapping[int, Mapping[str, Any]],
    *,
    min_hits: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        selected = int(summaries[level]["counts"]["selected"])
        same_j = float(summaries[level]["same_layer"]["jaccard"])
        docs = concentrations[level]["document"]
        ci = bootstrap[level]["coverage_ci95"]
        point = float(bootstrap[level]["reservoir_coverage"])
        relative_width = (ci[1] - ci[0]) / point if point > 0 else float("inf")
        gates = {
            "reservoir_hits_at_least_min": selected >= min_hits,
            "same_layer_jaccard_at_least_0_9": same_j >= 0.9,
            "active_documents_at_least_20": int(docs["active_units"]) >= 20,
            "max_document_share_at_most_0_25": docs["max_share"] is not None and float(docs["max_share"]) <= 0.25,
            "effective_documents_at_least_10": docs["effective_units"] is not None and float(docs["effective_units"]) >= 10,
            "bootstrap_relative_width_at_most_1": math.isfinite(relative_width) and relative_width <= 1.0,
        }
        rows.append({
            "bundle": level, "passed_gate_count": sum(gates.values()),
            "all_gates_pass": all(gates.values()), "gates": gates,
            "bootstrap_relative_width": relative_width,
        })
    passing = [row for row in rows if row["all_gates_pass"]]
    if passing:
        chosen, status = min(passing, key=lambda row: int(row["bundle"])), "READY_FOR_HUMAN_THRESHOLD_LOCK"
        rationale = "strictest Wiki-calibrated bundle passing all precision/diversity/stability gates"
    else:
        chosen = max(rows, key=lambda row: (int(row["passed_gate_count"]), -int(row["bundle"])))
        status = "DO_NOT_LOCK_HUMAN_REVIEW_REQUIRED"
        rationale = "best-supported candidate, but at least one pre-lock quality gate failed"
    return {
        "recommended_bundle": int(chosen["bundle"]),
        "status": status,
        "rationale": rationale,
        "candidate_gate_rows": rows,
        "gate_policy": {
            "min_reservoir_hits": int(min_hits), "same_layer_jaccard": 0.9,
            "min_active_documents": 20, "max_document_share": 0.25,
            "min_effective_documents": 10, "max_bootstrap_relative_width": 1.0,
        },
        "not_a_final_gt": True,
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "NA" if not math.isfinite(value) else f"{value:.6g}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |\n")
    return header + separator + "".join(body)


def _report(payload: Mapping[str, Any]) -> str:
    candidate_rows = []
    marginal_rows = []
    concentration_rows = []
    repetition_rows = []
    for level in LEVELS:
        summary = payload["candidates"][str(level)]
        boot = payload["document_bootstrap"][str(level)]
        same = summary["same_layer"]
        candidate_rows.append({
            "bundle": level,
            "reservoir selected": summary["counts"]["selected"],
            "coverage %": 100.0 * boot["reservoir_coverage"],
            "estimated full count": boot["estimated_full_count"],
            "count CI95": f"{boot['estimated_full_count_ci95'][0]}–{boot['estimated_full_count_ci95'][1]}",
            "same-layer Jaccard": same["jaccard"],
        })
        fractions = summary["fractions_of_all"]
        marginal_rows.append({
            "bundle": level, "B": fractions["B"], "T": fractions["T"],
            "M": fractions["M"], "B+T": fractions["B+T"],
            "B+M": fractions["B+M"], "B+T+M": fractions["B+T+M"],
        })
        concentration = payload["concentration"][str(level)]
        concentration_rows.append({
            "bundle": level,
            "windows": concentration["window"]["active_units"],
            "documents": concentration["document"]["active_units"],
            "max window share": concentration["window"]["max_share"],
            "window HHI": concentration["window"]["hhi"],
            "max document share": concentration["document"]["max_share"],
            "document HHI": concentration["document"]["hhi"],
        })
        repetition = payload["token_repetition"][str(level)]
        repetition_rows.append({
            "bundle": level, "unique token IDs": repetition.get("unique_token_ids"),
            "unique ratio": repetition.get("unique_token_ratio"),
            "entropy bits": repetition.get("entropy_bits"),
            "top token share": repetition.get("top_token_share"),
            "excess repeat fraction": repetition.get("excess_token_repeat_fraction"),
        })
    recommendation = payload["recommendation"]
    recommended_bundle = payload["candidate_thresholds"][str(recommendation["recommended_bundle"])]
    recommended_threshold_rows = []
    for layer_index, layer in enumerate(LAYERS):
        recommended_threshold_rows.append({
            "layer": layer,
            "B128 >=": recommended_bundle["B_lower_threshold"]["128"][layer_index],
            "B256 >=": recommended_bundle["B_lower_threshold"]["256"][layer_index],
            "T128 >=": recommended_bundle["T_lower_threshold"]["128"][layer_index],
            "T256 >=": recommended_bundle["T_lower_threshold"]["256"][layer_index],
            "rel-L2 <=": recommended_bundle["relative_l2_upper_threshold"][layer_index],
            "|log-r| <=": recommended_bundle["abs_log_r_upper_threshold"][layer_index],
        })
    return f"""# Full Code CKA census — threshold review

> This report does **not** create exact GT and does **not** open the sealed test split.
> Counts are estimates from the deterministic 5M uniform token reservoir; exact
> occurrences require a targeted second pass after a human threshold lock.

## Recommendation

- Recommended candidate: **bundle {recommendation['recommended_bundle']}**
- Status: **{recommendation['status']}**
- Reason: {recommendation['rationale']}
- Label semantics remain **replay-stable relational anchor candidate**, not a
  proven semantic Wiki-like token.

### Recommended candidate's exact frozen thresholds

{_markdown_table(recommended_threshold_rows, tuple(recommended_threshold_rows[0]))}

All 95/97/99 values are in `tables/candidate_thresholds.csv` and
`analysis.json:candidate_thresholds`.

## Candidate coverage and full-count estimate

{_markdown_table(candidate_rows, tuple(candidate_rows[0]))}

The CI is an ordinary document-cluster bootstrap: observed document IDs are
resampled with replacement and the token-weighted selected/total ratio is
recomputed. It captures observed document clustering; it is not an exact GT
count interval.

## Condition contributions

Fractions below use all reservoir tokens as the denominator. `B` and `T` each
AND both scales; `M` ANDs rel-L2 and |log-r|. Each primitive condition first
uses its own 7/8 layer consensus (6/6 if exactly six valid layers).

{_markdown_table(marginal_rows, tuple(marginal_rows[0]))}

## Condition-specific versus same-layer

The primary rule lets different layers supply each condition's 7/8 consensus.
The diagnostic requires all B/T/M conditions on the same layer before the
7/8 consensus. Jaccard below 0.9 requires human review; see
`tables/same_layer_diagnostic.csv`.

## Concentration audit

{_markdown_table(concentration_rows, tuple(concentration_rows[0]))}

HHI, Gini and max share are computed among selected reservoir occurrences and
active units. A small number of active/effective windows or documents indicates
the repeated-list/table failure mode seen in the pilot, even when token count is
large.

## Token-ID repetition audit

{_markdown_table(repetition_rows, tuple(repetition_rows[0]))}

The reservoir is sparse, so this audit can measure token-ID repetition and
same-window duplicate IDs but cannot recover exact contiguous repeated n-grams.

## Exact histogram evidence

`tables/histogram_threshold_pass_bounds.csv` gives conservative/liberal exact
full-census marginal pass counts; the only ambiguity is the single histogram
bin crossed by a threshold. `histograms/*_full.svg` and `*_zoom.svg` overlay
the three frozen Wiki-calibration cuts on full Code distributions.

## Calibration and sealing

- analysis_config SHA256: `{payload['calibration_binding']['analysis_config']['sha256']}`
- thresholds: exact Wiki-calibration candidate cuts frozen by the pilot config
- open exact-quantile artifact SHA256: `{payload['calibration_binding']['open_calibration_artifacts']['raw_quantiles']['sha256']}`
- sealed test opened: **false**
- no referenced token/chunk/test path in analysis_config was traversed

## Decision boundary

This analysis recommends a threshold candidate, not exact GT. Lock the chosen
bundle in a separate immutable decision record before any sealed-test opening.
Then run the targeted exact occurrence pass and re-audit concentration on exact
selected windows/documents before using the labels for KD.
"""


def run_analysis(
    *,
    census_root: str | Path,
    analysis_config: str | Path,
    output_dir: str | Path,
    bootstrap_repetitions: int = 200,
    bootstrap_seed: int = 20260816,
    chunk_rows: int = 250_000,
    min_recommendation_hits: int = 200,
) -> dict[str, Any]:
    root, config_path, output = Path(census_root), Path(analysis_config), Path(output_dir)
    summary_path = root / "summary.json"
    hist_path, reservoir_path = root / "histograms.npz", root / "token_score_reservoir.npz"
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != EXPECTED_CENSUS_SCHEMA or summary.get("complete") is not True:
        raise RuntimeError("full census summary is absent, incomplete, or unsupported")
    if summary.get("threshold_free") is not True or summary.get("final_gt_created") is not False:
        raise RuntimeError("input is not the threshold-free full census")
    for path in (hist_path, reservoir_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen_overlay = summary.get("pilot_overlay")
    if not isinstance(frozen_overlay, Mapping) or not frozen_overlay.get("sha256"):
        raise RuntimeError("merged census does not bind a frozen pilot analysis_config")
    config_sha256 = _sha256(config_path)
    if frozen_overlay["sha256"] != config_sha256:
        raise RuntimeError(
            "analysis_config SHA256 differs from the configuration frozen into the census"
        )
    for key, path in (("histograms", hist_path), ("token_score_reservoir", reservoir_path)):
        identity = summary.get(key, {})
        if identity.get("size_bytes") != path.stat().st_size:
            raise RuntimeError(f"{key} size differs from merged summary")
        if identity.get("sha256") and identity["sha256"] != _sha256(path):
            raise RuntimeError(f"{key} SHA256 differs from merged summary")

    config, bundles = load_frozen_bundles(config_path)
    open_calibration = bind_open_calibration_artifacts(config)
    # Do not dereference any paths in config.  In particular, test metrics and
    # sealed_test_path (if present) stay unopened by construction.
    calibration_binding = {
        "analysis_config": _file_identity(config_path),
        "census_frozen_analysis_config_sha256": frozen_overlay["sha256"],
        "matches_census_frozen_analysis_config": True,
        "analysis_config_schema": config.get("schema"),
        "candidate_levels": list(LEVELS),
        "threshold_source": "candidate_thresholds frozen from exact Wiki calibration quantiles",
        "open_calibration_artifacts": open_calibration,
        "declared_input_inventory_digest": config.get("input_inventory_digest_sha256"),
        "sealed_test_opened": False,
        "allowlisted_referenced_paths_traversed": [
            open_calibration["raw_quantiles"]["path"],
            open_calibration["raw_quantile_manifest"]["path"],
        ],
        "non_allowlisted_referenced_paths_traversed": False,
    }

    with np.load(hist_path, allow_pickle=False) as archive:
        histograms = {key: archive[key].copy() for key in archive.files}
    with np.load(reservoir_path, allow_pickle=False) as archive:
        if "token_scores" not in archive.files:
            raise RuntimeError("reservoir NPZ has no token_scores")
        scores = archive["token_scores"]
        if int(summary.get("reservoir_rows", -1)) != len(scores):
            raise RuntimeError("token_scores row count differs from merged summary")
        if len(scores) > 1 and np.any(scores["priority"][1:] < scores["priority"][:-1]):
            raise RuntimeError("deterministic bottom-hash reservoir is not priority-sorted")
        summaries, masks = analyze_reservoir(scores, bundles, chunk_rows=chunk_rows)
        bootstrap = document_bootstrap_intervals(
            scores["document_id"], masks,
            full_eligible_tokens=int(summary["processed_eligible_tokens"]),
            repetitions=bootstrap_repetitions, seed=bootstrap_seed,
        )
        concentration: dict[int, dict[str, Any]] = {}
        repetition: dict[int, dict[str, Any]] = {}
        for level in LEVELS:
            selected = masks[level]
            concentration[level] = {
                "window": concentration_metrics(scores["sample_order"][selected]),
                "document": concentration_metrics(scores["document_id"][selected]),
            }
            repetition[level] = token_repetition_metrics(
                scores["token_id"][selected], scores["sample_order"][selected]
            )

    overlap_rows = pairwise_jaccard_rows(masks)
    if not all(row["left_subset_of_right"] for row in overlap_rows):
        raise RuntimeError("nested Wiki candidate thresholds produced non-nested selections")
    threshold_table = threshold_rows(bundles)
    histogram_bounds = histogram_threshold_bounds(histograms, bundles)
    recommendation = _recommendation(
        summaries, bootstrap, concentration, min_hits=min_recommendation_hits
    )
    histogram_artifacts = write_histogram_overlays(output / "histograms", histograms, bundles)

    marginal_rows: list[dict[str, Any]] = []
    sequential_rows: list[dict[str, Any]] = []
    same_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    repetition_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        counts, fractions = summaries[level]["counts"], summaries[level]["fractions_of_all"]
        for condition in ("B128", "B256", "B", "T128", "T256", "T", "L2", "R", "M"):
            marginal_rows.append({
                "bundle": level, "condition": condition, "pass_count": counts[condition],
                "pass_fraction_all": fractions[condition],
                "eligible_count": counts.get(condition + "_eligible", counts.get("full_eligible")),
            })
        for stage in ("B", "B+T", "B+M", "B+T+M"):
            sequential_rows.append({
                "bundle": level, "stage": stage, "pass_count": counts[stage],
                "pass_fraction_all": fractions[stage],
            })
        same_rows.append({"bundle": level, **summaries[level]["same_layer"]})
        for unit in ("window", "document"):
            concentration_rows.append({"bundle": level, "unit": unit, **concentration[level][unit]})
        repetition_rows.append({"bundle": level, **repetition[level]})

    payload = {
        "schema": SCHEMA,
        "complete": True,
        "threshold_review_only": True,
        "exact_gt_created": False,
        "sealed_test_opened": False,
        "census_summary": _file_identity(summary_path),
        "histograms_input": _file_identity(hist_path),
        "reservoir_input": _file_identity(reservoir_path),
        "calibration_binding": calibration_binding,
        "candidate_thresholds": {
            str(level): _serializable_bundle(bundles[level]) for level in LEVELS
        },
        "reservoir_rows": int(len(scores)),
        "full_eligible_tokens": int(summary["processed_eligible_tokens"]),
        "consensus": "condition-specific 7/8 (6/6 at n_valid=6); B128&B256&T128&T256&L2&R",
        "candidates": {str(level): summaries[level] for level in LEVELS},
        "document_bootstrap": {str(level): bootstrap[level] for level in LEVELS},
        "concentration": {str(level): concentration[level] for level in LEVELS},
        "token_repetition": {str(level): repetition[level] for level in LEVELS},
        "pairwise_jaccard": overlap_rows,
        "recommendation": recommendation,
        "histogram_artifacts": histogram_artifacts,
        "limitations": [
            "reservoir joint counts and concentration are estimates, not exact selected occurrences",
            "histogram joint rules cannot be recovered exactly from marginal bins",
            "sparse token reservoir cannot measure exact contiguous n-gram repetition",
            "semantic Wiki-likeness and causal preservation are not proven by CKA stability",
        ],
        "next_required_step": "human threshold lock, then exact targeted occurrence pass",
    }

    _write_csv(output / "tables" / "candidate_thresholds.csv", threshold_table)
    _write_csv(output / "tables" / "marginal_pass_fractions.csv", marginal_rows)
    _write_csv(output / "tables" / "sequential_B_T_M.csv", sequential_rows)
    _write_csv(output / "tables" / "same_layer_diagnostic.csv", same_rows)
    _write_csv(output / "tables" / "pairwise_jaccard.csv", overlap_rows)
    _write_csv(output / "tables" / "concentration.csv", concentration_rows)
    _write_csv(output / "tables" / "token_id_repetition.csv", repetition_rows)
    _write_csv(output / "tables" / "histogram_threshold_pass_bounds.csv", histogram_bounds)
    _atomic_json(output / "calibration_binding.json", calibration_binding)
    _atomic_json(output / "analysis.json", payload)
    _atomic_text(output / "REPORT.md", _report(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--min-recommendation-hits", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_analysis(
        census_root=args.census_root,
        analysis_config=args.analysis_config,
        output_dir=args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        chunk_rows=args.chunk_rows,
        min_recommendation_hits=args.min_recommendation_hits,
    )
    print(json.dumps({
        "schema": payload["schema"], "complete": payload["complete"],
        "recommendation": payload["recommendation"],
        "sealed_test_opened": payload["sealed_test_opened"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateBundle", "analyze_reservoir", "concentration_metrics",
    "condition_consensus", "document_bootstrap_intervals", "evaluate_candidate",
    "histogram_threshold_bounds", "load_frozen_bundles", "pairwise_jaccard_rows",
    "run_analysis", "token_repetition_metrics",
]
