#!/usr/bin/env python3
"""Full-census cross-layer stability analysis for paired Code-token metrics.

The input shards contain contextual token occurrences and residual-included
Transformer layer outputs compared between the reference and current models.
Layer 1 is intentionally excluded.  This script assigns no GT threshold: it
measures cross-layer agreement and diagnoses direction-only, scaling, and
low-reference-norm cases using Layers 2--9.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "code_token_cross_layer_stability_v1"
LAYERS = np.arange(2, 10, dtype=np.int16)
TOP_PERCENTS = np.asarray([1, 5, 10, 20], dtype=np.int16)
TOP_THRESHOLDS = 1.0 - TOP_PERCENTS.astype(np.float64) / 100.0
PERCENTILE_BINS = 200
METRIC_BINS = 256
AGGREGATE_BINS = 1000
CHECKPOINT_EVERY_SHARDS = 12
CANDIDATES_PER_RANK_PER_TOP_P = 512
FINAL_CANDIDATES_PER_TOP_P = 2048
METRIC_NAMES = (
    "relative_l2_log10",
    "log_norm_ratio",
    "symmetric_relative_l2",
    "reference_rms_log10",
)
METRIC_RANGES = np.asarray(
    [
        [-5.0, 2.0],
        [-4.0, 4.0],
        [0.0, 2.0],
        [-4.0, 3.0],
    ],
    dtype=np.float64,
)
AGGREGATE_NAMES = (
    "percentile_mean",
    "percentile_median",
    "percentile_min",
    "late_l7_l9_percentile_mean",
    "raw_cosine_mean",
)
STABLE_DIAGNOSTIC_NAMES = (
    "mean_relative_l2",
    "late_l7_l9_relative_l2",
    "max_abs_log_norm_ratio",
    "min_reference_rms_log10",
)
EXPECTED_TOTAL_TOKENS = 2_123_366_400


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64, copy=False) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _discover(root: Path, limit_shards_per_rank: int | None) -> list[tuple[Path, list[Path]]]:
    result: list[tuple[Path, list[Path]]] = []
    for rank_dir in sorted(root.glob("rank_[0-9][0-9][0-9]")):
        shards = sorted((rank_dir / "token_metrics").glob("shard_*.npz"))
        if limit_shards_per_rank is not None:
            shards = shards[:limit_shards_per_rank]
        if shards:
            result.append((rank_dir, shards))
    if not result:
        raise RuntimeError(f"no token metric shards below {root}")
    return result


def _config_hash(
    root: Path,
    ranks: list[tuple[Path, list[Path]]],
    cosine_counts_path: Path,
) -> str:
    digest = hashlib.sha256()
    settings = {
        "schema": SCHEMA,
        "root": str(root.resolve()),
        "layers": LAYERS.tolist(),
        "top_percents": TOP_PERCENTS.tolist(),
        "percentile_bins": PERCENTILE_BINS,
        "metric_bins": METRIC_BINS,
        "metric_ranges": METRIC_RANGES.tolist(),
        "aggregate_bins": AGGREGATE_BINS,
        "cosine_counts_path": str(cosine_counts_path.resolve()),
        "cosine_counts_bytes": cosine_counts_path.stat().st_size,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for rank_dir, shards in ranks:
        digest.update(rank_dir.name.encode())
        for shard in shards:
            digest.update(f"{shard.name}:{shard.stat().st_size}".encode())
    return digest.hexdigest()


def build_percentile_lookup(cosine_counts_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(cosine_counts_path, allow_pickle=False) as data:
        counts_all = data["counts"].astype(np.uint64)
        edges = data["bin_edges"].astype(np.float64)
        layer_numbers = data["layer_numbers"]
    if not np.array_equal(layer_numbers, np.arange(1, 10)):
        raise RuntimeError(f"unexpected cosine histogram layers: {layer_numbers.tolist()}")
    if counts_all.shape != (9, 20_000) or edges.shape != (20_001,):
        raise RuntimeError(f"unexpected cosine histogram shapes: {counts_all.shape}, {edges.shape}")
    counts = counts_all[1:]
    totals = counts.sum(axis=1, dtype=np.uint64)
    if not np.all(totals == totals[0]):
        raise RuntimeError(f"cosine histogram layer totals differ: {totals.tolist()}")
    cumulative = np.cumsum(counts, axis=1, dtype=np.uint64)
    lookup = (cumulative.astype(np.float64) - 0.5 * counts.astype(np.float64)) / totals[:, None]
    max_midrank_error = counts.max(axis=1).astype(np.float64) / (2.0 * totals)
    return lookup, {
        "histogram_value_bin_width": float(edges[1] - edges[0]),
        "percentile_midrank_max_tie_error_by_layer": {
            str(int(layer)): float(error) for layer, error in zip(LAYERS, max_midrank_error)
        },
        "tokens_per_layer": int(totals[0]),
    }


def cosine_to_percentiles(cosine: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    if cosine.ndim != 2 or cosine.shape[1] != 8:
        raise ValueError(f"cosine must be [tokens,8], got {cosine.shape}")
    indices = np.floor((cosine.astype(np.float64) + 1.0) * 10_000.0).astype(np.int32)
    np.clip(indices, 0, 19_999, out=indices)
    result = np.empty(cosine.shape, dtype=np.float32)
    for layer_index in range(8):
        result[:, layer_index] = lookup[layer_index, indices[:, layer_index]]
    return result


def membership_pattern_counts(percentiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pattern_counts = np.zeros((len(TOP_PERCENTS), 256), dtype=np.uint64)
    stable_counts = np.empty((percentiles.shape[0], len(TOP_PERCENTS)), dtype=np.uint8)
    for top_index, threshold in enumerate(TOP_THRESHOLDS):
        membership = percentiles >= threshold
        stable_counts[:, top_index] = membership.sum(axis=1, dtype=np.uint8)
        patterns = np.zeros(percentiles.shape[0], dtype=np.uint8)
        for layer_index in range(8):
            patterns |= membership[:, layer_index].astype(np.uint8) << np.uint8(layer_index)
        pattern_counts[top_index] = np.bincount(patterns, minlength=256).astype(np.uint64)
    return pattern_counts, stable_counts


def reference_rms_proxy(delta_mse: np.ndarray, relative_l2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (relative_l2 > 1e-12) & (delta_mse > 1e-24)
    proxy = np.full(relative_l2.shape, np.nan, dtype=np.float32)
    proxy[valid] = np.sqrt(delta_mse[valid]).astype(np.float32) / relative_l2[valid]
    return proxy, valid


def _metric_bin(values: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    under = finite & (values < lower)
    over = finite & (values > upper)
    scaled = np.nan_to_num((values - lower) / (upper - lower), nan=0.0, neginf=0.0, posinf=1.0)
    indices = np.floor(scaled * METRIC_BINS).astype(np.int32)
    np.clip(indices, 0, METRIC_BINS - 1, out=indices)
    return indices, under, over


def _empty_accumulator() -> dict[str, np.ndarray]:
    return {
        "tokens": np.asarray(0, dtype=np.uint64),
        "percentile_sum": np.zeros(8, dtype=np.float64),
        "percentile_cross": np.zeros((8, 8), dtype=np.float64),
        "pattern_counts": np.zeros((4, 256), dtype=np.uint64),
        "aggregate_hist": np.zeros((len(AGGREGATE_NAMES), AGGREGATE_BINS), dtype=np.uint64),
        "aggregate_sum": np.zeros(len(AGGREGATE_NAMES), dtype=np.float64),
        "aggregate_sum_sq": np.zeros(len(AGGREGATE_NAMES), dtype=np.float64),
        "aggregate_min": np.full(len(AGGREGATE_NAMES), np.inf, dtype=np.float64),
        "aggregate_max": np.full(len(AGGREGATE_NAMES), -np.inf, dtype=np.float64),
        "loo_sum": np.zeros(8, dtype=np.float64),
        "loo_sum_sq": np.zeros(8, dtype=np.float64),
        "full_mean_sum": np.asarray(0.0, dtype=np.float64),
        "full_mean_sum_sq": np.asarray(0.0, dtype=np.float64),
        "full_loo_cross": np.zeros(8, dtype=np.float64),
        "joint_metric": np.zeros((len(METRIC_NAMES), 8, PERCENTILE_BINS, METRIC_BINS), dtype=np.uint64),
        "metric_underflow": np.zeros((len(METRIC_NAMES), 8), dtype=np.uint64),
        "metric_overflow": np.zeros((len(METRIC_NAMES), 8), dtype=np.uint64),
        "metric_nonfinite": np.zeros((len(METRIC_NAMES), 8), dtype=np.uint64),
        "stable_diagnostic_hist": np.zeros(
            (len(TOP_PERCENTS), len(STABLE_DIAGNOSTIC_NAMES), 9, METRIC_BINS), dtype=np.uint64
        ),
    }


def _candidate_empty() -> dict[str, np.ndarray]:
    return {
        "priority": np.empty(0, dtype=np.uint64),
        "sample_ids": np.empty(0, dtype=np.int64),
        "positions": np.empty(0, dtype=np.uint16),
        "token_ids": np.empty(0, dtype=np.int32),
        "cosine": np.empty((0, 8), dtype=np.float32),
        "percentile": np.empty((0, 8), dtype=np.float32),
        "stable_counts": np.empty((0, 4), dtype=np.uint8),
        "relative_l2": np.empty((0, 8), dtype=np.float32),
        "log_norm_ratio": np.empty((0, 8), dtype=np.float32),
        "reference_rms": np.empty((0, 8), dtype=np.float32),
    }


def _candidate_take(candidate: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[keep] for name, values in candidate.items()}


def _candidate_merge(
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    limit: int,
) -> dict[str, np.ndarray]:
    if old["priority"].size == 0:
        combined = {name: values.copy() for name, values in new.items()}
    elif new["priority"].size == 0:
        return old
    else:
        combined = {name: np.concatenate((old[name], new[name]), axis=0) for name in old}
    if combined["priority"].size <= limit:
        order = np.argsort(combined["priority"], kind="stable")
    else:
        keep = np.argpartition(combined["priority"], limit - 1)[:limit]
        order = keep[np.argsort(combined["priority"][keep], kind="stable")]
    return _candidate_take(combined, order)


def _accumulate_percentile_moments(acc: dict[str, np.ndarray], percentiles: np.ndarray) -> None:
    acc["percentile_sum"] += percentiles.sum(axis=0, dtype=np.float64)
    for start in range(0, percentiles.shape[0], 262_144):
        block = percentiles[start : start + 262_144].astype(np.float64)
        acc["percentile_cross"] += block.T @ block


def _accumulate_aggregates(
    acc: dict[str, np.ndarray],
    percentiles: np.ndarray,
    cosine: np.ndarray,
) -> None:
    percentile_sum = percentiles.sum(axis=1, dtype=np.float32)
    values = (
        percentile_sum / 8.0,
        np.median(percentiles, axis=1),
        percentiles.min(axis=1),
        percentiles[:, 5:8].mean(axis=1, dtype=np.float32),
        cosine.mean(axis=1, dtype=np.float32),
    )
    for index, value in enumerate(values):
        value64 = value.astype(np.float64)
        acc["aggregate_sum"][index] += value64.sum(dtype=np.float64)
        acc["aggregate_sum_sq"][index] += np.square(value64).sum(dtype=np.float64)
        acc["aggregate_min"][index] = min(acc["aggregate_min"][index], float(value.min()))
        acc["aggregate_max"][index] = max(acc["aggregate_max"][index], float(value.max()))
        value_range = (0.0, 1.0) if index < 4 else (-1.0, 1.0)
        hist, _ = np.histogram(value, bins=AGGREGATE_BINS, range=value_range)
        acc["aggregate_hist"][index] += hist.astype(np.uint64)

    full_mean = values[0].astype(np.float64)
    acc["full_mean_sum"] += full_mean.sum(dtype=np.float64)
    acc["full_mean_sum_sq"] += np.square(full_mean).sum(dtype=np.float64)
    for layer_index in range(8):
        loo = ((percentile_sum - percentiles[:, layer_index]) / 7.0).astype(np.float64)
        acc["loo_sum"][layer_index] += loo.sum(dtype=np.float64)
        acc["loo_sum_sq"][layer_index] += np.square(loo).sum(dtype=np.float64)
        acc["full_loo_cross"][layer_index] += np.multiply(full_mean, loo).sum(dtype=np.float64)


def _accumulate_joint(
    acc: dict[str, np.ndarray],
    percentile_bins: np.ndarray,
    metric_index: int,
    values: np.ndarray,
) -> np.ndarray:
    lower, upper = METRIC_RANGES[metric_index]
    metric_bins, under, over = _metric_bin(values, float(lower), float(upper))
    finite = np.isfinite(values)
    for layer_index in range(8):
        acc["metric_underflow"][metric_index, layer_index] += int(under[:, layer_index].sum())
        acc["metric_overflow"][metric_index, layer_index] += int(over[:, layer_index].sum())
        acc["metric_nonfinite"][metric_index, layer_index] += int((~finite[:, layer_index]).sum())
        flat = percentile_bins[:, layer_index] * METRIC_BINS + metric_bins[:, layer_index]
        acc["joint_metric"][metric_index, layer_index] += np.bincount(
            flat, minlength=PERCENTILE_BINS * METRIC_BINS
        ).reshape(PERCENTILE_BINS, METRIC_BINS).astype(np.uint64)
    return metric_bins


def _accumulate_stable_diagnostics(
    acc: dict[str, np.ndarray],
    stable_counts: np.ndarray,
    relative_l2: np.ndarray,
    log_norm_ratio: np.ndarray,
    reference_rms: np.ndarray,
) -> None:
    diagnostics = (
        np.log10(relative_l2.mean(axis=1, dtype=np.float32) + 1e-30),
        np.log10(relative_l2[:, 5:8].mean(axis=1, dtype=np.float32) + 1e-30),
        np.abs(log_norm_ratio).max(axis=1),
        np.log10(np.nanmin(reference_rms, axis=1) + 1e-30),
    )
    ranges = (METRIC_RANGES[0], METRIC_RANGES[0], np.asarray([0.0, 4.0]), METRIC_RANGES[3])
    for diagnostic_index, (values, value_range) in enumerate(zip(diagnostics, ranges)):
        bins, _, _ = _metric_bin(values, float(value_range[0]), float(value_range[1]))
        for top_index in range(len(TOP_PERCENTS)):
            flat = stable_counts[:, top_index].astype(np.int32) * METRIC_BINS + bins
            acc["stable_diagnostic_hist"][top_index, diagnostic_index] += np.bincount(
                flat, minlength=9 * METRIC_BINS
            ).reshape(9, METRIC_BINS).astype(np.uint64)


def _progress_payload(
    config_hash: str,
    next_shard: int,
    acc: dict[str, np.ndarray],
    candidates: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    payload = {
        "schema": np.asarray(SCHEMA),
        "config_hash": np.asarray(config_hash),
        "next_shard": np.asarray(next_shard, dtype=np.int64),
    }
    payload.update({f"acc_{name}": value for name, value in acc.items()})
    for top_index, candidate in enumerate(candidates):
        for name, value in candidate.items():
            payload[f"candidate_{top_index}_{name}"] = value
    return payload


def _restore_progress(
    path: Path,
    config_hash: str,
) -> tuple[int, dict[str, np.ndarray], list[dict[str, np.ndarray]]] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"].item()) != SCHEMA or str(data["config_hash"].item()) != config_hash:
                return None
            acc = _empty_accumulator()
            for name in acc:
                acc[name] = data[f"acc_{name}"].copy()
            candidates = []
            for top_index in range(len(TOP_PERCENTS)):
                candidate = _candidate_empty()
                for name in candidate:
                    candidate[name] = data[f"candidate_{top_index}_{name}"].copy()
                candidates.append(candidate)
            return int(data["next_shard"].item()), acc, candidates
    except Exception:
        return None


def _scan_rank(
    rank_dir_string: str,
    shard_strings: list[str],
    output_dir_string: str,
    config_hash: str,
    percentile_lookup: np.ndarray,
    force: bool,
) -> dict[str, Any]:
    rank_dir = Path(rank_dir_string)
    shards = [Path(value) for value in shard_strings]
    output_dir = Path(output_dir_string)
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    progress_path = partial_dir / f"{rank_dir.name}.progress.npz"
    final_path = partial_dir / f"{rank_dir.name}.npz"
    if force:
        for path in (progress_path, final_path):
            if path.exists():
                path.unlink()
    restored = _restore_progress(final_path, config_hash)
    if restored is not None and restored[0] == len(shards):
        return {"rank": rank_dir.name, "cached": True, "tokens": int(restored[1]["tokens"]), "shards": len(shards)}
    restored = _restore_progress(progress_path, config_hash)
    if restored is None:
        next_shard = 0
        acc = _empty_accumulator()
        candidates = [_candidate_empty() for _ in TOP_PERCENTS]
    else:
        next_shard, acc, candidates = restored

    started = time.monotonic()
    for shard_index in range(next_shard, len(shards)):
        shard = shards[shard_index]
        with np.load(shard, allow_pickle=False) as data:
            if not np.array_equal(data["layer_numbers"], np.arange(1, 10)):
                raise RuntimeError(f"layer mismatch in {shard}")
            valid_mask = data["valid_mask"].astype(bool, copy=False)
            sample_ids = data["sample_ids"]
            input_token_ids = data["input_token_ids"]
            cosine_dense = data["cosine"][..., 1:]
            if cosine_dense.shape[:2] != valid_mask.shape or cosine_dense.shape[2] != 8:
                raise RuntimeError(f"cosine/mask shape mismatch in {shard}")
            cosine = cosine_dense.reshape(-1, 8) if valid_mask.all() else cosine_dense[valid_mask]
            if not np.isfinite(cosine).all():
                raise RuntimeError(f"non-finite cosine in {shard}")
            token_count = cosine.shape[0]
            percentiles = cosine_to_percentiles(cosine, percentile_lookup)
            pattern_counts, stable_counts = membership_pattern_counts(percentiles)
            acc["pattern_counts"] += pattern_counts
            _accumulate_percentile_moments(acc, percentiles)
            _accumulate_aggregates(acc, percentiles, cosine)
            percentile_bins = np.floor(percentiles * PERCENTILE_BINS).astype(np.int32)
            np.clip(percentile_bins, 0, PERCENTILE_BINS - 1, out=percentile_bins)

            relative_dense = data["relative_l2"][..., 1:]
            relative_l2 = relative_dense.reshape(-1, 8) if valid_mask.all() else relative_dense[valid_mask]
            log_dense = data["log_norm_ratio"][..., 1:]
            log_norm_ratio = log_dense.reshape(-1, 8) if valid_mask.all() else log_dense[valid_mask]
            delta_dense = data["delta_mse"][..., 1:]
            delta_mse = delta_dense.reshape(-1, 8) if valid_mask.all() else delta_dense[valid_mask]
            if not (np.isfinite(relative_l2).all() and np.isfinite(log_norm_ratio).all() and np.isfinite(delta_mse).all()):
                raise RuntimeError(f"non-finite drift metric in {shard}")
            reference_rms, reference_valid = reference_rms_proxy(delta_mse, relative_l2)
            symmetric_relative = 2.0 * relative_l2 / (1.0 + np.exp(log_norm_ratio))
            transformed_metrics = (
                np.log10(relative_l2 + 1e-30),
                log_norm_ratio,
                symmetric_relative,
                np.log10(reference_rms + 1e-30),
            )
            for metric_index, values in enumerate(transformed_metrics):
                _accumulate_joint(acc, percentile_bins, metric_index, values)
            _accumulate_stable_diagnostics(acc, stable_counts, relative_l2, log_norm_ratio, reference_rms)

            if valid_mask.all():
                dense_indices = np.arange(valid_mask.size, dtype=np.int64)
            else:
                dense_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int64)
            sequence = valid_mask.shape[1]
            sample_for_token = sample_ids[dense_indices // sequence]
            positions = (dense_indices % sequence).astype(np.uint16)
            token_ids = input_token_ids.reshape(-1)[dense_indices]
            occurrence = sample_for_token.astype(np.uint64) * np.uint64(sequence) + positions.astype(np.uint64)
            for top_index, top_percent in enumerate(TOP_PERCENTS):
                qualify = np.flatnonzero(stable_counts[:, top_index] >= 6)
                if qualify.size == 0:
                    continue
                priority = _splitmix64(occurrence[qualify], int(1234 + int(top_percent) * 1009))
                take_count = min(CANDIDATES_PER_RANK_PER_TOP_P, qualify.size)
                if qualify.size > take_count:
                    local = np.argpartition(priority, take_count - 1)[:take_count]
                    qualify = qualify[local]
                    priority = priority[local]
                new = {
                    "priority": priority,
                    "sample_ids": sample_for_token[qualify].astype(np.int64),
                    "positions": positions[qualify],
                    "token_ids": token_ids[qualify].astype(np.int32),
                    "cosine": cosine[qualify].astype(np.float32),
                    "percentile": percentiles[qualify].astype(np.float32),
                    "stable_counts": stable_counts[qualify].astype(np.uint8),
                    "relative_l2": relative_l2[qualify].astype(np.float32),
                    "log_norm_ratio": log_norm_ratio[qualify].astype(np.float32),
                    "reference_rms": reference_rms[qualify].astype(np.float32),
                }
                candidates[top_index] = _candidate_merge(
                    candidates[top_index], new, CANDIDATES_PER_RANK_PER_TOP_P
                )
            acc["tokens"] += np.uint64(token_count)
            if int((~reference_valid).sum()) > 0:
                # Non-finite proxy rows are already represented in metric_nonfinite.
                pass

        completed_shards = shard_index + 1
        if completed_shards % CHECKPOINT_EVERY_SHARDS == 0 or completed_shards == len(shards):
            _atomic_npz(
                progress_path,
                _progress_payload(config_hash, completed_shards, acc, candidates),
            )
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"[{rank_dir.name}] shards={completed_shards}/{len(shards)} "
                f"tokens={int(acc['tokens']):,} rate={int(acc['tokens']) / elapsed:,.0f} token/s",
                flush=True,
            )

    os.replace(progress_path, final_path)
    return {
        "rank": rank_dir.name,
        "cached": False,
        "tokens": int(acc["tokens"]),
        "shards": len(shards),
        "seconds": time.monotonic() - started,
    }


def _load_partial(path: Path, config_hash: str) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], int]:
    restored = _restore_progress(path, config_hash)
    if restored is None:
        raise RuntimeError(f"missing or incompatible partial {path}")
    next_shard, acc, candidates = restored
    return acc, candidates, next_shard


def _merge_partials(
    ranks: list[tuple[Path, list[Path]]],
    output_dir: Path,
    config_hash: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    merged = _empty_accumulator()
    merged_candidates = [_candidate_empty() for _ in TOP_PERCENTS]
    worker_rows = []
    for rank_dir, shards in ranks:
        acc, candidates, next_shard = _load_partial(output_dir / "partials" / f"{rank_dir.name}.npz", config_hash)
        if next_shard != len(shards):
            raise RuntimeError(f"incomplete partial for {rank_dir.name}: {next_shard}/{len(shards)}")
        for name in merged:
            if name in ("aggregate_min",):
                merged[name] = np.minimum(merged[name], acc[name])
            elif name in ("aggregate_max",):
                merged[name] = np.maximum(merged[name], acc[name])
            else:
                merged[name] += acc[name]
        for top_index in range(len(TOP_PERCENTS)):
            merged_candidates[top_index] = _candidate_merge(
                merged_candidates[top_index], candidates[top_index], FINAL_CANDIDATES_PER_TOP_P
            )
        worker_rows.append(
            {
                "worker": rank_dir.name,
                "tokens": int(acc["tokens"]),
                "shards": len(shards),
                "percentile_mean": (acc["percentile_sum"] / max(int(acc["tokens"]), 1)).tolist(),
            }
        )
    return merged, merged_candidates, worker_rows


def _pattern_derived(pattern_counts: np.ndarray) -> dict[str, Any]:
    pattern_ids = np.arange(256, dtype=np.uint16)
    bits = ((pattern_ids[:, None] >> np.arange(8, dtype=np.uint16)) & 1).astype(bool)
    stable_count = bits.sum(axis=1)
    total = int(pattern_counts.sum(dtype=np.uint64))
    selected = np.asarray(
        [pattern_counts[bits[:, layer]].sum(dtype=np.uint64) for layer in range(8)], dtype=np.uint64
    )
    intersections = np.zeros((8, 8), dtype=np.uint64)
    for left in range(8):
        for right in range(8):
            intersections[left, right] = pattern_counts[bits[:, left] & bits[:, right]].sum(dtype=np.uint64)
    unions = selected[:, None] + selected[None, :] - intersections
    jaccard = np.divide(intersections, unions, out=np.zeros((8, 8), dtype=np.float64), where=unions > 0)
    overlap_coefficient = np.divide(
        intersections,
        np.minimum(selected[:, None], selected[None, :]),
        out=np.zeros((8, 8), dtype=np.float64),
        where=np.minimum(selected[:, None], selected[None, :]) > 0,
    )
    expected = selected[:, None].astype(np.float64) * selected[None, :].astype(np.float64) / max(total, 1)
    lift = np.divide(intersections, expected, out=np.zeros((8, 8), dtype=np.float64), where=expected > 0)
    count_hist = np.asarray(
        [pattern_counts[stable_count == count].sum(dtype=np.uint64) for count in range(9)], dtype=np.uint64
    )
    late_mask = np.uint16((1 << 5) | (1 << 6) | (1 << 7))
    late_all = int(pattern_counts[(pattern_ids & late_mask) == late_mask].sum(dtype=np.uint64))
    return {
        "total": total,
        "selected": selected,
        "intersections": intersections,
        "unions": unions,
        "jaccard": jaccard,
        "overlap_coefficient": overlap_coefficient,
        "lift": lift,
        "stable_count_hist": count_hist,
        "late_l7_l9_all": late_all,
    }


def _correlation_from_moments(count: int, sums: np.ndarray, cross: np.ndarray) -> np.ndarray:
    means = sums / count
    covariance = cross / count - means[:, None] * means[None, :]
    variance = np.maximum(np.diag(covariance), 0.0)
    denominator = np.sqrt(variance[:, None] * variance[None, :])
    result = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    np.fill_diagonal(result, 1.0)
    return result


def _hist_quantile(counts: np.ndarray, lower: float, upper: float, q: float) -> float:
    total = int(counts.sum(dtype=np.uint64))
    if total == 0:
        return float("nan")
    target = max(1, int(math.ceil(q * total)))
    index = int(np.searchsorted(np.cumsum(counts, dtype=np.uint64), target, side="left"))
    width = (upper - lower) / counts.size
    return lower + (index + 0.5) * width


def _hist_fraction_between(counts: np.ndarray, lower: float, upper: float, lo: float, hi: float) -> float:
    centers = lower + (np.arange(counts.size) + 0.5) * (upper - lower) / counts.size
    total = int(counts.sum(dtype=np.uint64))
    return float(counts[(centers >= lo) & (centers <= hi)].sum(dtype=np.uint64) / total) if total else float("nan")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _render_heatmap(matrix: np.ndarray, path: Path, title: str, value_format: str = ".3f") -> None:
    from PIL import Image, ImageDraw

    size, margin_left, margin_top, cell = 1120, 150, 150, 105
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, value_font = _font(30), _font(23), _font(18)
    draw.text((size // 2, 35), title, fill="#111111", font=title_font, anchor="ma")
    minimum, maximum = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    if maximum <= minimum:
        maximum = minimum + 1.0
    for row in range(8):
        draw.text((margin_left - 18, margin_top + row * cell + cell // 2), f"L{row+2}", fill="#222", font=label_font, anchor="rm")
        draw.text((margin_left + row * cell + cell // 2, margin_top - 18), f"L{row+2}", fill="#222", font=label_font, anchor="mb")
        for column in range(8):
            value = float(matrix[row, column])
            intensity = (value - minimum) / (maximum - minimum)
            color = (int(245 - 175 * intensity), int(250 - 125 * intensity), int(255 - 35 * intensity))
            box = (
                margin_left + column * cell,
                margin_top + row * cell,
                margin_left + (column + 1) * cell,
                margin_top + (row + 1) * cell,
            )
            draw.rectangle(box, fill=color, outline="#ffffff")
            draw.text(
                ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                format(value, value_format),
                fill="#111111",
                font=value_font,
                anchor="mm",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _render_stable_counts(rows_by_top: list[dict[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1500, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(30), _font(23), _font(18)
    draw.text((width // 2, 30), "Number of stable Layers 2-9 per token", fill="#111", font=title_font, anchor="ma")
    left, top, right, bottom = 110, 115, 1150, 820
    draw.rectangle((left, top, right, bottom), outline="#333", width=2)
    colors = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4")
    for exp in range(-10, 1, 2):
        y = bottom - (exp + 10) / 10 * (bottom - top)
        draw.line((left, y, right, y), fill="#e5e5e5")
        draw.text((left - 10, y), f"1e{exp}", fill="#333", font=tick_font, anchor="rm")
    for top_index, row in enumerate(rows_by_top):
        fractions = np.maximum(np.asarray(row["fractions"], dtype=np.float64), 1e-10)
        points = []
        for count, fraction in enumerate(fractions):
            x = left + count / 8 * (right - left)
            y = bottom - (math.log10(fraction) + 10) / 10 * (bottom - top)
            points.append((int(x), int(y)))
        draw.line(points, fill=colors[top_index], width=4)
        for x, y in points:
            draw.ellipse((x-4, y-4, x+4, y+4), fill=colors[top_index])
        legend_y = top + top_index * 55
        draw.line((1200, legend_y, 1260, legend_y), fill=colors[top_index], width=5)
        draw.text((1280, legend_y), f"top {int(TOP_PERCENTS[top_index])}%", fill="#222", font=label_font, anchor="lm")
    for count in range(9):
        x = left + count / 8 * (right - left)
        draw.text((x, bottom + 12), str(count), fill="#333", font=tick_font, anchor="ma")
    draw.text(((left+right)//2, bottom+58), "stable layer count", fill="#222", font=label_font, anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _render_layer_metric(
    rows: list[dict[str, Any]],
    key: str,
    path: Path,
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1450, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(30), _font(22), _font(18)
    draw.text((width // 2, 28), title, fill="#111", font=title_font, anchor="ma")
    left, top, right, bottom = 110, 110, 1120, 780
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    minimum, maximum = min(values), max(values)
    padding = max((maximum - minimum) * 0.1, 1e-6)
    minimum, maximum = minimum - padding, maximum + padding
    draw.rectangle((left, top, right, bottom), outline="#333", width=2)
    colors = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4")
    for top_index, top_percent in enumerate(TOP_PERCENTS):
        selected_rows = [row for row in rows if int(row["top_percent"]) == int(top_percent)]
        points = []
        for row in selected_rows:
            layer = int(row["layer"])
            value = float(row[key])
            x = left + (layer - 2) / 7 * (right - left)
            y = bottom - (value - minimum) / (maximum - minimum) * (bottom - top)
            points.append((int(x), int(y)))
        draw.line(points, fill=colors[top_index], width=4)
        for x, y in points:
            draw.ellipse((x-4, y-4, x+4, y+4), fill=colors[top_index])
        legend_y = top + top_index * 55
        draw.line((1170, legend_y, 1230, legend_y), fill=colors[top_index], width=5)
        draw.text((1250, legend_y), f"top {int(top_percent)}%", fill="#222", font=label_font, anchor="lm")
    for layer in LAYERS:
        x = left + (int(layer) - 2) / 7 * (right - left)
        draw.text((x, bottom+12), f"L{int(layer)}", fill="#333", font=tick_font, anchor="ma")
    for fraction in np.linspace(0, 1, 6):
        value = minimum + fraction * (maximum - minimum)
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill="#ededed")
        draw.text((left-10, y), f"{value:.3g}", fill="#333", font=tick_font, anchor="rm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _render_joint_panels(
    joint: np.ndarray,
    path: Path,
    title: str,
    metric_lower: float,
    metric_upper: float,
    metric_label: str,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1900, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(30), _font(22), _font(17)
    draw.text((width // 2, 25), title, fill="#111", font=title_font, anchor="ma")
    outer_left, outer_top, gap_x, gap_y = 70, 100, 30, 45
    cell_w, cell_h = 430, 465
    for layer_index, layer in enumerate(LAYERS):
        row, column = divmod(layer_index, 4)
        x0 = outer_left + column * (cell_w + gap_x)
        y0 = outer_top + row * (cell_h + gap_y)
        plot_left, plot_top = x0 + 65, y0 + 40
        plot_right, plot_bottom = x0 + cell_w - 15, y0 + cell_h - 55
        values = np.log10(joint[layer_index].T.astype(np.float64) + 1.0)
        maximum = float(values.max())
        normalized = values / maximum if maximum > 0 else values
        red = (245 - 205 * normalized).astype(np.uint8)
        green = (250 - 120 * normalized).astype(np.uint8)
        blue = (255 - 25 * normalized).astype(np.uint8)
        rgb = np.stack((red, green, blue), axis=-1)
        panel = Image.fromarray(rgb[::-1], mode="RGB").resize(
            (plot_right - plot_left, plot_bottom - plot_top), resample=Image.Resampling.BILINEAR
        )
        image.paste(panel, (plot_left, plot_top))
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#333", width=1)
        draw.text(((plot_left+plot_right)//2, y0+3), f"Layer {int(layer)}", fill="#111", font=label_font, anchor="ma")
        for percentile in (0.0, 0.5, 1.0):
            x = plot_left + percentile * (plot_right - plot_left)
            draw.text((x, plot_bottom+8), f"{percentile:g}", fill="#333", font=tick_font, anchor="ma")
        for fraction in (0.0, 0.5, 1.0):
            value = metric_lower + fraction * (metric_upper - metric_lower)
            y = plot_bottom - fraction * (plot_bottom - plot_top)
            draw.text((plot_left-7, y), f"{value:.2g}", fill="#333", font=tick_font, anchor="rm")
        draw.text(((plot_left+plot_right)//2, plot_bottom+34), "cosine percentile", fill="#333", font=tick_font, anchor="ma")
        draw.text((x0+5, (plot_top+plot_bottom)//2), metric_label, fill="#333", font=tick_font, anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _stable_diagnostic_rows(merged: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    ranges = (METRIC_RANGES[0], METRIC_RANGES[0], np.asarray([0.0, 4.0]), METRIC_RANGES[3])
    rows = []
    for top_index, top_percent in enumerate(TOP_PERCENTS):
        for stable_count in range(9):
            histograms = merged["stable_diagnostic_hist"][top_index, :, stable_count]
            row: dict[str, Any] = {
                "top_percent": int(top_percent),
                "stable_layer_count": stable_count,
                "count": int(histograms[0].sum(dtype=np.uint64)),
            }
            for diagnostic_index, (name, value_range) in enumerate(zip(STABLE_DIAGNOSTIC_NAMES, ranges)):
                histogram = histograms[diagnostic_index]
                q50 = _hist_quantile(histogram, float(value_range[0]), float(value_range[1]), 0.5)
                q95 = _hist_quantile(histogram, float(value_range[0]), float(value_range[1]), 0.95)
                if diagnostic_index in (0, 1, 3):
                    q50, q95 = 10.0**q50, 10.0**q95
                row[f"{name}_median"] = q50
                row[f"{name}_p95"] = q95
            rows.append(row)
    return rows


def _render_stable_diagnostic(
    rows: list[dict[str, Any]],
    key: str,
    path: Path,
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1450, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(30), _font(22), _font(18)
    draw.text((width//2, 28), title, fill="#111", font=title_font, anchor="ma")
    left, top, right, bottom = 110, 105, 1110, 780
    finite_values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    minimum, maximum = min(finite_values), max(finite_values)
    padding = max((maximum-minimum)*0.1, 1e-8)
    minimum, maximum = minimum-padding, maximum+padding
    draw.rectangle((left, top, right, bottom), outline="#333", width=2)
    colors = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4")
    for top_index, top_percent in enumerate(TOP_PERCENTS):
        selected = [row for row in rows if int(row["top_percent"]) == int(top_percent)]
        points = []
        for row in selected:
            stable_count = int(row["stable_layer_count"])
            value = float(row[key])
            x = left + stable_count/8*(right-left)
            y = bottom - (value-minimum)/(maximum-minimum)*(bottom-top)
            points.append((int(x),int(y)))
        draw.line(points, fill=colors[top_index], width=4)
        for x,y in points: draw.ellipse((x-4,y-4,x+4,y+4),fill=colors[top_index])
        legend_y=top+top_index*55
        draw.line((1160,legend_y,1220,legend_y),fill=colors[top_index],width=5)
        draw.text((1240,legend_y),f"top {int(top_percent)}%",fill="#222",font=label_font,anchor="lm")
    for count in range(9):
        x=left+count/8*(right-left); draw.text((x,bottom+12),str(count),fill="#333",font=tick_font,anchor="ma")
    for fraction in np.linspace(0,1,6):
        value=minimum+fraction*(maximum-minimum); y=bottom-fraction*(bottom-top)
        draw.line((left,y,right,y),fill="#ededed"); draw.text((left-10,y),f"{value:.3g}",fill="#333",font=tick_font,anchor="rm")
    draw.text(((left+right)//2,bottom+58),"stable layer count",fill="#222",font=label_font,anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path,format="PNG",optimize=True)


def _conditional_rows(
    merged: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    relative_rows, norm_rows, reference_rows = [], [], []
    percentile_start = [int(math.ceil(threshold * PERCENTILE_BINS)) for threshold in TOP_THRESHOLDS]
    # Global reference-RMS distributions and layer-specific bottom thresholds.
    global_reference = merged["joint_metric"][3].sum(axis=1, dtype=np.uint64)
    ref_lower, ref_upper = METRIC_RANGES[3]
    ref_bottom = {
        layer_index: {
            q: 10.0 ** _hist_quantile(global_reference[layer_index], ref_lower, ref_upper, q)
            for q in (0.01, 0.05, 0.10)
        }
        for layer_index in range(8)
    }
    for top_index, top_percent in enumerate(TOP_PERCENTS):
        start = percentile_start[top_index]
        for layer_index, layer in enumerate(LAYERS):
            rel_hist = merged["joint_metric"][0, layer_index, start:].sum(axis=0, dtype=np.uint64)
            log_hist = merged["joint_metric"][1, layer_index, start:].sum(axis=0, dtype=np.uint64)
            sym_hist = merged["joint_metric"][2, layer_index, start:].sum(axis=0, dtype=np.uint64)
            ref_hist = merged["joint_metric"][3, layer_index, start:].sum(axis=0, dtype=np.uint64)
            count = int(rel_hist.sum(dtype=np.uint64))
            rel_q = [10.0 ** _hist_quantile(rel_hist, *METRIC_RANGES[0], q) for q in (0.05, 0.5, 0.95)]
            sym_q = [_hist_quantile(sym_hist, *METRIC_RANGES[2], q) for q in (0.05, 0.5, 0.95)]
            relative_rows.append(
                {
                    "top_percent": int(top_percent), "layer": int(layer), "selected_count": count,
                    "relative_l2_p05": rel_q[0], "relative_l2_median": rel_q[1], "relative_l2_p95": rel_q[2],
                    "symmetric_relative_l2_p05": sym_q[0], "symmetric_relative_l2_median": sym_q[1],
                    "symmetric_relative_l2_p95": sym_q[2],
                    "fraction_relative_l2_le_0p1": _hist_fraction_between(rel_hist, *METRIC_RANGES[0], -30, math.log10(0.1)),
                    "fraction_relative_l2_le_0p2": _hist_fraction_between(rel_hist, *METRIC_RANGES[0], -30, math.log10(0.2)),
                    "fraction_relative_l2_le_0p5": _hist_fraction_between(rel_hist, *METRIC_RANGES[0], -30, math.log10(0.5)),
                }
            )
            log_q = [_hist_quantile(log_hist, *METRIC_RANGES[1], q) for q in (0.05, 0.5, 0.95)]
            norm_rows.append(
                {
                    "top_percent": int(top_percent), "layer": int(layer), "selected_count": count,
                    "log_norm_ratio_p05": log_q[0], "log_norm_ratio_median": log_q[1], "log_norm_ratio_p95": log_q[2],
                    "norm_ratio_p05": math.exp(log_q[0]), "norm_ratio_median": math.exp(log_q[1]),
                    "norm_ratio_p95": math.exp(log_q[2]),
                    "fraction_norm_ratio_0p9_to_1p1": _hist_fraction_between(log_hist, *METRIC_RANGES[1], math.log(0.9), math.log(1.1)),
                    "fraction_norm_ratio_0p8_to_1p2": _hist_fraction_between(log_hist, *METRIC_RANGES[1], math.log(0.8), math.log(1.2)),
                }
            )
            ref_q = [10.0 ** _hist_quantile(ref_hist, ref_lower, ref_upper, q) for q in (0.05, 0.5, 0.95)]
            centers = 10.0 ** (ref_lower + (np.arange(METRIC_BINS) + 0.5) * (ref_upper - ref_lower) / METRIC_BINS)
            total = max(int(ref_hist.sum(dtype=np.uint64)), 1)
            reference_rows.append(
                {
                    "top_percent": int(top_percent), "layer": int(layer), "selected_count": count,
                    "reference_rms_p05": ref_q[0], "reference_rms_median": ref_q[1], "reference_rms_p95": ref_q[2],
                    "global_reference_rms_bottom01_cutoff": ref_bottom[layer_index][0.01],
                    "global_reference_rms_bottom05_cutoff": ref_bottom[layer_index][0.05],
                    "global_reference_rms_bottom10_cutoff": ref_bottom[layer_index][0.10],
                    "fraction_in_global_bottom01": float(ref_hist[centers <= ref_bottom[layer_index][0.01]].sum(dtype=np.uint64) / total),
                    "fraction_in_global_bottom05": float(ref_hist[centers <= ref_bottom[layer_index][0.05]].sum(dtype=np.uint64) / total),
                    "fraction_in_global_bottom10": float(ref_hist[centers <= ref_bottom[layer_index][0.10]].sum(dtype=np.uint64) / total),
                }
            )
    return relative_rows, norm_rows, reference_rows


def _save_candidates(path: Path, candidates: list[dict[str, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {"layer_numbers": LAYERS, "top_percents": TOP_PERCENTS}
    for top_index, candidate in enumerate(candidates):
        for name, values in candidate.items():
            payload[f"top{int(TOP_PERCENTS[top_index]):02d}_{name}"] = values
    _atomic_npz(path, payload)


def _validate_merged(
    merged: dict[str, np.ndarray],
    expected_tokens: int,
    ranks: list[tuple[Path, list[Path]]],
) -> dict[str, Any]:
    errors = []
    tokens = int(merged["tokens"])
    if tokens != expected_tokens:
        errors.append(f"token count {tokens} != expected {expected_tokens}")
    for top_index, top_percent in enumerate(TOP_PERCENTS):
        total = int(merged["pattern_counts"][top_index].sum(dtype=np.uint64))
        if total != tokens:
            errors.append(f"top{top_percent} pattern total {total} != {tokens}")
    for metric_index, metric in enumerate(METRIC_NAMES):
        totals = merged["joint_metric"][metric_index].sum(axis=(1, 2), dtype=np.uint64)
        if not np.all(totals == tokens):
            errors.append(f"{metric} layer totals mismatch: {totals.tolist()}")
    aggregate_totals = merged["aggregate_hist"].sum(axis=1, dtype=np.uint64)
    if not np.all(aggregate_totals == tokens):
        errors.append(f"aggregate histogram totals mismatch: {aggregate_totals.tolist()}")
    nonfinite = int(merged["metric_nonfinite"].sum(dtype=np.uint64))
    if nonfinite:
        errors.append(f"metric nonfinite count={nonfinite}")
    return {
        "schema": "code_token_cross_layer_stability_validation_v1",
        "passed": not errors,
        "errors": errors,
        "tokens": tokens,
        "expected_tokens": expected_tokens,
        "workers": len(ranks),
        "shards": sum(len(shards) for _, shards in ranks),
        "layers": LAYERS.tolist(),
        "top_percents": TOP_PERCENTS.tolist(),
        "nonfinite_count": nonfinite,
        "gt_assigned": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cosine-counts", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-shards-per-rank", type=int)
    parser.add_argument("--expected-tokens", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    cosine_counts = (
        args.cosine_counts
        or root / "analysis" / "cosine_histogram" / "cosine_histogram_counts.npz"
    ).resolve()
    ranks = _discover(root, args.limit_shards_per_rank)
    lookup, percentile_metadata = build_percentile_lookup(cosine_counts)
    config_hash = _config_hash(root, ranks, cosine_counts)
    expected_tokens = args.expected_tokens
    if expected_tokens is None:
        expected_tokens = (
            EXPECTED_TOTAL_TOKENS
            if args.limit_shards_per_rank is None
            else sum(
                int(json.load((shard.with_suffix(".json")).open())["valid_tokens"])
                for _, shards in ranks for shard in shards
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": SCHEMA,
        "complete": False,
        "config_hash": config_hash,
        "source_root": str(root),
        "cosine_counts": str(cosine_counts),
        "layers": LAYERS.tolist(),
        "excluded_layers": [1],
        "top_percents": TOP_PERCENTS.tolist(),
        "expected_tokens": expected_tokens,
        "workers": len(ranks),
        "shards": sum(len(shards) for _, shards in ranks),
        "percentile_definition": "midrank empirical CDF from fixed 1e-4 cosine bins",
        "spearman_definition": "Pearson correlation of histogram-CDF percentile midranks (approximate Spearman)",
        "reference_rms_proxy": "sqrt(delta_mse)/relative_l2",
        "gt_assigned": False,
        **percentile_metadata,
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    started = time.monotonic()
    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(ranks))) as pool:
        for rank_dir, shards in ranks:
            futures.append(
                pool.submit(
                    _scan_rank,
                    str(rank_dir),
                    [str(path) for path in shards],
                    str(output_dir),
                    config_hash,
                    lookup,
                    args.force,
                )
            )
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            _atomic_json(
                output_dir / "progress.json",
                {
                    "schema": "code_token_cross_layer_progress_v1",
                    "completed_workers": len(results),
                    "workers": len(ranks),
                    "completed_tokens_visible": sum(int(item["tokens"]) for item in results),
                    "worker_results": sorted(results, key=lambda item: item["rank"]),
                    "gt_assigned": False,
                },
            )
            print(json.dumps(row, sort_keys=True), flush=True)

    merged, candidates, worker_rows = _merge_partials(ranks, output_dir, config_hash)
    validation = _validate_merged(merged, expected_tokens, ranks)
    _atomic_json(output_dir / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"cross-layer validation failed: {validation['errors']}")

    tokens = int(merged["tokens"])
    spearman = _correlation_from_moments(tokens, merged["percentile_sum"], merged["percentile_cross"])
    correlation_rows = [
        {"layer_i": int(LAYERS[i]), "layer_j": int(LAYERS[j]), "spearman_approx": float(spearman[i, j])}
        for i in range(8) for j in range(8)
    ]
    _write_csv(output_dir / "layer_percentile_correlation.csv", list(correlation_rows[0]), correlation_rows)
    _render_heatmap(spearman, output_dir / "layer_percentile_spearman.png", "Approximate Spearman: cosine percentile, Layers 2-9")

    stable_rows = []
    membership_rows = []
    stable_plot_rows = []
    overlap_summaries = []
    for top_index, top_percent in enumerate(TOP_PERCENTS):
        pattern = merged["pattern_counts"][top_index]
        derived = _pattern_derived(pattern)
        fractions = derived["stable_count_hist"].astype(np.float64) / tokens
        stable_plot_rows.append({"fractions": fractions.tolist()})
        for count in range(9):
            stable_rows.append(
                {"top_percent": int(top_percent), "stable_layer_count": count,
                 "count": int(derived["stable_count_hist"][count]), "fraction": float(fractions[count])}
            )
        for pattern_id, count in enumerate(pattern):
            membership_rows.append(
                {"top_percent": int(top_percent), "pattern_decimal": pattern_id,
                 "pattern_l2_to_l9": format(pattern_id, "08b")[::-1], "count": int(count),
                 "fraction": float(count / tokens)}
            )
        pair_rows = []
        for i in range(8):
            for j in range(8):
                pair_rows.append(
                    {"layer_i": int(LAYERS[i]), "layer_j": int(LAYERS[j]),
                     "selected_i": int(derived["selected"][i]), "selected_j": int(derived["selected"][j]),
                     "intersection": int(derived["intersections"][i,j]), "union": int(derived["unions"][i,j]),
                     "jaccard": float(derived["jaccard"][i,j]),
                     "overlap_coefficient": float(derived["overlap_coefficient"][i,j]),
                     "independence_lift": float(derived["lift"][i,j])}
                )
        prefix = f"top{int(top_percent):02d}"
        _write_csv(output_dir / f"{prefix}_jaccard.csv", list(pair_rows[0]), pair_rows)
        _render_heatmap(derived["jaccard"], output_dir / f"{prefix}_jaccard.png", f"Jaccard of layerwise top {int(top_percent)}% token sets")
        overlap_summaries.append(
            {"top_percent": int(top_percent),
             "selected_fraction_by_layer": (derived["selected"].astype(np.float64)/tokens).tolist(),
             "at_least_2": float(derived["stable_count_hist"][2:].sum(dtype=np.uint64)/tokens),
             "at_least_4": float(derived["stable_count_hist"][4:].sum(dtype=np.uint64)/tokens),
             "at_least_6": float(derived["stable_count_hist"][6:].sum(dtype=np.uint64)/tokens),
             "all_8": float(derived["stable_count_hist"][8]/tokens),
             "late_l7_l9_all": float(derived["late_l7_l9_all"]/tokens),
             "mean_off_diagonal_jaccard": float(derived["jaccard"][~np.eye(8,dtype=bool)].mean()),
             "mean_off_diagonal_lift": float(derived["lift"][~np.eye(8,dtype=bool)].mean())}
        )
    _write_csv(output_dir / "stable_layer_count.csv", list(stable_rows[0]), stable_rows)
    _write_csv(output_dir / "membership_pattern_counts.csv", list(membership_rows[0]), membership_rows)
    _render_stable_counts(stable_plot_rows, output_dir / "stable_layer_count.png")

    relative_rows, norm_rows, reference_rows = _conditional_rows(merged)
    _write_csv(output_dir / "high_cos_relative_l2.csv", list(relative_rows[0]), relative_rows)
    _write_csv(output_dir / "high_cos_norm_ratio.csv", list(norm_rows[0]), norm_rows)
    _write_csv(output_dir / "high_cos_reference_norm.csv", list(reference_rows[0]), reference_rows)
    _render_layer_metric(relative_rows, "relative_l2_median", output_dir / "high_cos_relative_l2.png", "Median relative L2 inside layerwise high-cos sets")
    _render_layer_metric(norm_rows, "norm_ratio_median", output_dir / "high_cos_norm_ratio.png", "Median norm ratio inside layerwise high-cos sets")
    _render_layer_metric(reference_rows, "fraction_in_global_bottom05", output_dir / "high_cos_reference_norm.png", "Fraction of high-cos tokens in global bottom-5% reference norm")
    _render_joint_panels(
        merged["joint_metric"][0], output_dir / "cosine_percentile_vs_relative_l2_joint.png",
        "Cosine percentile vs relative L2 (log10 counts)", *METRIC_RANGES[0], "log10 relative L2"
    )
    _render_joint_panels(
        merged["joint_metric"][1], output_dir / "cosine_percentile_vs_log_norm_ratio_joint.png",
        "Cosine percentile vs log norm ratio (log10 counts)", *METRIC_RANGES[1], "log norm ratio"
    )
    stable_diagnostic_rows = _stable_diagnostic_rows(merged)
    _write_csv(
        output_dir / "stable_layer_diagnostics.csv",
        list(stable_diagnostic_rows[0]),
        stable_diagnostic_rows,
    )
    _render_stable_diagnostic(
        stable_diagnostic_rows, "mean_relative_l2_median",
        output_dir / "stable_count_vs_relative_l2.png",
        "Stable-layer count vs median mean relative L2",
    )
    _render_stable_diagnostic(
        stable_diagnostic_rows, "late_l7_l9_relative_l2_median",
        output_dir / "late_stability_vs_relative_l2.png",
        "Stable-layer count vs median late-layer relative L2",
    )
    _save_candidates(output_dir / "candidate_reservoir.npz", candidates)

    mean = merged["percentile_sum"] / tokens
    covariance = merged["percentile_cross"] / tokens - mean[:,None]*mean[None,:]
    full_mean = float(merged["full_mean_sum"] / tokens)
    full_var = float(merged["full_mean_sum_sq"] / tokens - full_mean**2)
    loo_correlations = []
    for layer_index, layer in enumerate(LAYERS):
        loo_mean = float(merged["loo_sum"][layer_index] / tokens)
        loo_var = float(merged["loo_sum_sq"][layer_index] / tokens - loo_mean**2)
        cov = float(merged["full_loo_cross"][layer_index] / tokens - full_mean*loo_mean)
        loo_correlations.append(cov / math.sqrt(max(full_var*loo_var, 1e-30)))
    aggregate_rows = []
    for index, name in enumerate(AGGREGATE_NAMES):
        value_mean = float(merged["aggregate_sum"][index]/tokens)
        value_var = float(merged["aggregate_sum_sq"][index]/tokens-value_mean**2)
        aggregate_rows.append(
            {"score": name, "mean": value_mean, "std": math.sqrt(max(value_var,0.0)),
             "min": float(merged["aggregate_min"][index]), "max": float(merged["aggregate_max"][index])}
        )
    _write_csv(output_dir / "aggregate_score_summary.csv", list(aggregate_rows[0]), aggregate_rows)
    _write_csv(output_dir / "worker_subset_summary.csv", list(worker_rows[0]), worker_rows)

    metadata.update(
        {
            "complete": True,
            "tokens": tokens,
            "seconds": time.monotonic() - started,
            "overlap_summaries": overlap_summaries,
            "mean_off_diagonal_spearman": float(spearman[~np.eye(8,dtype=bool)].mean()),
            "adjacent_layer_spearman": [float(spearman[i,i+1]) for i in range(7)],
            "leave_one_layer_out_correlation_with_full_percentile_mean": {
                str(int(layer)): float(value) for layer, value in zip(LAYERS, loo_correlations)
            },
            "aggregate_score_summary": aggregate_rows,
            "worker_subset_summary": worker_rows,
            "metric_underflow": merged["metric_underflow"].tolist(),
            "metric_overflow": merged["metric_overflow"].tolist(),
            "metric_nonfinite": merged["metric_nonfinite"].tolist(),
            "artifacts": {
                "validation": str(output_dir / "validation.json"),
                "candidate_reservoir": str(output_dir / "candidate_reservoir.npz"),
                "spearman_png": str(output_dir / "layer_percentile_spearman.png"),
                "stable_count_png": str(output_dir / "stable_layer_count.png"),
            },
        }
    )
    _atomic_json(output_dir / "metadata.json", metadata)
    _atomic_npz(
        output_dir / "cross_layer_numeric_summary.npz",
        {
            "layer_numbers": LAYERS, "top_percents": TOP_PERCENTS,
            "spearman_approx": spearman, "pattern_counts": merged["pattern_counts"],
            "aggregate_hist": merged["aggregate_hist"], "joint_metric": merged["joint_metric"],
            "stable_diagnostic_hist": merged["stable_diagnostic_hist"],
        },
    )
    print(json.dumps({"complete": True, "tokens": tokens, "seconds": metadata["seconds"],
                      "mean_off_diagonal_spearman": metadata["mean_off_diagonal_spearman"],
                      "overlap_summaries": overlap_summaries}, indent=2))


if __name__ == "__main__":
    main()
