#!/usr/bin/env python3
"""Audit and review exact targeted CKA old-like token candidates.

This is the CPU-only review stage that follows ``cka_gt_targeted_gt.py``.
It treats the exact packed masks and occurrence arrays as untrusted inputs,
replays their complete identity chain, and produces the evidence needed to
choose among the frozen Wiki-calibrated 95/97/99 bundles.

The review deliberately does *not* open the sealed pilot test split and does
not materialize a final GT.  Bundle 95 is the strictest candidate (tightest
thresholds) and bundle 99 is the loosest.  The recommendation is therefore
the first, strictest bundle that passes exact count, layer-consensus, and
concentration gates; it remains a recommendation for a human threshold lock.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .cka_gt_pilot_windows import (
        MMapIndexedDatasetLite,
        WINDOW_DTYPE,
        file_sha256,
        load_document_window,
    )
    from .cka_gt_targeted_gt import (
        BUNDLE_LEVELS,
        CONDITION_COUNT_NAMES,
        OCCURRENCE_DTYPE,
        PACKED_BYTES_PER_WINDOW,
        PROGRESS_SCHEMA,
        SHARD_SCHEMA,
        SUMMARY_SCHEMA,
        unpack_mask,
    )
except ImportError:  # Direct script/test execution.
    from cka_gt_pilot_windows import (
        MMapIndexedDatasetLite,
        WINDOW_DTYPE,
        file_sha256,
        load_document_window,
    )
    from cka_gt_targeted_gt import (
        BUNDLE_LEVELS,
        CONDITION_COUNT_NAMES,
        OCCURRENCE_DTYPE,
        PACKED_BYTES_PER_WINDOW,
        PROGRESS_SCHEMA,
        SHARD_SCHEMA,
        SUMMARY_SCHEMA,
        unpack_mask,
    )


SCHEMA = "cka_gt_exact_targeted_review_v1"
VALIDATION_SCHEMA = "cka_gt_exact_targeted_review_validation_v1"
EXPECTED_THRESHOLD_REVIEW_SCHEMA = "cka_gt_full_census_threshold_review_v1"
DEFAULT_TOKENIZER = Path(
    "/data2/seonghyeonnoh/homecache/huggingface/hub/"
    "models--EleutherAI--pythia-12b/snapshots/"
    "bb1e3e710cdf6b524461d543cfb5ba773f0a81b6"
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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {key: _jsonable(row.get(key)) for key in columns}
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(
                _jsonable(row),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _identity(path: str | Path) -> dict[str, Any]:
    value = Path(path).resolve()
    return {
        "path": str(value),
        "size_bytes": int(value.stat().st_size),
        "sha256": file_sha256(value),
    }


def _verify_identity(
    expected: Mapping[str, Any],
    *,
    expected_path: str | Path | None = None,
    label: str,
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"{label}: missing file identity")
    raw_path = expected.get("path")
    if not raw_path:
        raise RuntimeError(f"{label}: identity has no path")
    path = Path(str(raw_path)).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise RuntimeError(f"{label}: identity binds a different path")
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    observed = _identity(path)
    for key in ("size_bytes", "sha256"):
        if key not in expected:
            raise RuntimeError(f"{label}: identity has no {key}")
        if observed[key] != expected[key]:
            raise RuntimeError(f"{label}: {key} mismatch")
    return observed


def _decode_metadata(value: np.ndarray) -> dict[str, Any]:
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))


def _gini(counts: np.ndarray) -> float | None:
    values = np.sort(np.asarray(counts, dtype=np.float64))
    values = values[values > 0]
    if not values.size:
        return None
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.dot(ranks, values) / values.sum() - values.size - 1)
        / values.size
    )


def concentration_metrics(ids: np.ndarray, *, top_k: int = 10) -> dict[str, Any]:
    """Exact selected-occurrence concentration over an identity axis."""

    value = np.asarray(ids)
    if not value.size:
        return {
            "selected_occurrences": 0,
            "active_units": 0,
            "max_share": None,
            "top5_share": None,
            "hhi": None,
            "effective_units": None,
            "gini_among_active": None,
            "top_units": [],
        }
    unique, counts = np.unique(value, return_counts=True)
    order = np.lexsort((unique, -counts))[:top_k]
    shares = counts.astype(np.float64) / value.size
    hhi = float(np.square(shares).sum())
    descending = np.sort(shares)[::-1]
    return {
        "selected_occurrences": int(value.size),
        "active_units": int(unique.size),
        "max_share": float(descending[0]),
        "top5_share": float(descending[:5].sum()),
        "hhi": hhi,
        "effective_units": float(1.0 / hhi),
        "gini_among_active": _gini(counts),
        "median_occurrences_per_active_unit": float(np.median(counts)),
        "p95_occurrences_per_active_unit": float(np.quantile(counts, 0.95)),
        "top_units": [
            {
                "id": int(unique[index]),
                "count": int(counts[index]),
                "share": float(counts[index] / value.size),
            }
            for index in order
        ],
    }


def token_repetition_metrics(
    token_ids: np.ndarray, window_ids: np.ndarray, *, top_k: int = 10
) -> dict[str, Any]:
    token_ids = np.asarray(token_ids, dtype=np.int64)
    window_ids = np.asarray(window_ids, dtype=np.int64)
    if token_ids.shape != window_ids.shape:
        raise ValueError("token/window IDs must align")
    if not token_ids.size:
        return {
            "selected_occurrences": 0,
            "unique_token_ids": 0,
            "entropy_bits": None,
            "normalized_entropy": None,
            "excess_repeat_fraction": None,
            "top_token_share": None,
            "top_token_ids": [],
        }
    unique, counts = np.unique(token_ids, return_counts=True)
    probabilities = counts.astype(np.float64) / token_ids.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    pair = np.empty(token_ids.size, dtype=[("window", "<i8"), ("token", "<i8")])
    pair["window"], pair["token"] = window_ids, token_ids
    order = np.lexsort((unique, -counts))[:top_k]
    return {
        "selected_occurrences": int(token_ids.size),
        "unique_token_ids": int(unique.size),
        "unique_token_ratio": float(unique.size / token_ids.size),
        "entropy_bits": entropy,
        "normalized_entropy": (
            float(entropy / math.log2(unique.size)) if unique.size > 1 else 0.0
        ),
        "effective_token_vocabulary": float(2.0**entropy),
        "token_id_hhi": float(np.square(probabilities).sum()),
        "top_token_share": float(probabilities.max()),
        "top10_token_share": float(np.sort(probabilities)[-10:].sum()),
        "excess_repeat_fraction": float(
            (token_ids.size - unique.size) / token_ids.size
        ),
        "within_window_excess_same_token_fraction": float(
            (token_ids.size - np.unique(pair).size) / token_ids.size
        ),
        "top_token_ids": [
            {
                "token_id": int(unique[index]),
                "count": int(counts[index]),
                "share": float(counts[index] / token_ids.size),
            }
            for index in order
        ],
    }


def _load_tokenizer(path: str | Path | None) -> tuple[Any | None, dict[str, Any]]:
    """Load a local decoder, falling back from transformers to tokenizers."""

    if path is None:
        return None, {
            "available": False,
            "reason": "tokenizer path disabled",
            "backend": None,
        }
    source = Path(path).resolve()
    failures: list[str] = []
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(source), local_files_only=True
        )

        def decode(ids: Sequence[int]) -> str:
            return tokenizer.decode(
                [int(item) for item in ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        return decode, {
            "available": True,
            "path": str(source),
            "backend": "transformers.AutoTokenizer",
            "fallback_used": False,
        }
    except Exception as error:  # Optional human-readable output.
        failures.append(f"transformers: {type(error).__name__}: {error}")

    tokenizer_json = source if source.name == "tokenizer.json" else source / "tokenizer.json"
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_json))

        def decode(ids: Sequence[int]) -> str:
            return tokenizer.decode(
                [int(item) for item in ids], skip_special_tokens=False
            )

        return decode, {
            "available": True,
            "path": str(source),
            "tokenizer_json": str(tokenizer_json),
            "backend": "tokenizers.Tokenizer",
            "fallback_used": True,
            "prior_failures": failures,
        }
    except Exception as error:  # Token IDs and all audits remain available.
        failures.append(f"tokenizers: {type(error).__name__}: {error}")
    return None, {
        "available": False,
        "path": str(source),
        "backend": None,
        "failures": failures,
    }


def _stable_priority(level: int, candidate: int, position: int, seed: int) -> int:
    payload = f"{seed}:{level}:{candidate}:{position}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _chosen_occurrence_indices(
    occurrences: np.ndarray, *, level: int, count: int, seed: int
) -> np.ndarray:
    if count <= 0 or not occurrences.size:
        return np.empty(0, dtype=np.int64)
    priorities = np.fromiter(
        (
            _stable_priority(
                level,
                int(row["candidate_index"]),
                int(row["position"]),
                seed,
            )
            for row in occurrences
        ),
        dtype=np.uint64,
        count=occurrences.size,
    )
    take = min(int(count), occurrences.size)
    chosen = np.argpartition(priorities, take - 1)[:take] if take < occurrences.size else np.arange(take)
    return chosen[np.argsort(priorities[chosen], kind="stable")]


def _context_record(
    occurrence: np.void,
    *,
    level: int,
    reason: str,
    candidate_rows: np.ndarray,
    dataset: Any,
    decoder: Any | None,
    radius: int,
) -> dict[str, Any]:
    candidate = int(occurrence["candidate_index"])
    row = candidate_rows[candidate]
    position = int(occurrence["position"])
    token_ids = np.asarray(
        load_document_window(
            dataset,
            int(row["document_id"]),
            int(row["window_offset"]),
            int(row["window_length"]),
        ),
        dtype=np.int64,
    )
    if int(token_ids[position]) != int(occurrence["token_id"]):
        raise RuntimeError("context enrichment found a token identity mismatch")
    start, stop = max(0, position - radius), min(len(token_ids), position + radius + 1)
    context = token_ids[start:stop].tolist()
    payload = {
        "bundle": int(level),
        "selection_reason": reason,
        "candidate_index": candidate,
        "sample_order": int(occurrence["sample_order"]),
        "source_window_index": int(occurrence["source_window_index"]),
        "document_id": int(occurrence["document_id"]),
        "window_offset": int(occurrence["window_offset"]),
        "window_length": int(row["window_length"]),
        "eligible_token_count": int(row["eligible_token_count"]),
        "position": position,
        "document_token_offset": int(row["window_offset"]) + position,
        "token_id": int(occurrence["token_id"]),
        "context_start_position": start,
        "target_offset_in_context": position - start,
        "context_token_ids": context,
        "token_identity_verified": True,
    }
    if decoder is not None:
        payload["context_text"] = decoder(context)
        payload["target_token_text"] = decoder([int(occurrence["token_id"])])
        payload["left_text"] = decoder(context[: position - start])
        payload["right_text"] = decoder(context[position - start + 1 :])
    return payload


def _top_context_indices(
    occurrences: np.ndarray,
    *,
    field: str,
    top_units: Sequence[Mapping[str, Any]],
    level: int,
    seed: int,
) -> list[int]:
    result: list[int] = []
    for unit in top_units:
        candidates = np.flatnonzero(occurrences[field] == int(unit["id"]))
        if not candidates.size:
            continue
        priorities = np.asarray(
            [
                _stable_priority(
                    level,
                    int(occurrences[index]["candidate_index"]),
                    int(occurrences[index]["position"]),
                    seed,
                )
                for index in candidates
            ],
            dtype=np.uint64,
        )
        result.append(int(candidates[int(np.argmin(priorities))]))
    return result


def _verify_manifest_content_hash(manifest: Mapping[str, Any]) -> bool:
    expected = manifest.get("manifest_content_sha256")
    if expected is None:
        return False
    payload = dict(manifest)
    payload.pop("manifest_content_sha256", None)
    encoded = (
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise RuntimeError("candidate manifest content hash differs")
    return True


def _verify_source_manifest_and_dataset(
    candidate_manifest: Mapping[str, Any], dataset_prefix: str | Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind the candidate axis to the exhaustive manifest and Code dataset."""

    binding = candidate_manifest.get("source_manifest")
    if not isinstance(binding, Mapping) or not binding.get("path"):
        raise RuntimeError("candidate manifest has no exhaustive source-manifest binding")
    path = Path(str(binding["path"])).resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("schema") != "cka_gt_full_train_windows_v1":
        raise RuntimeError("exhaustive source manifest schema differs")
    if not _verify_manifest_content_hash(source):
        raise RuntimeError("exhaustive source manifest has no content hash")
    if source.get("manifest_content_sha256") != binding.get("content_identity"):
        raise RuntimeError("candidate/source-manifest content identity differs")
    requested_prefix = Path(str(dataset_prefix)).resolve()
    source_prefix = Path(str(source.get("dataset_prefix", ""))).resolve()
    if requested_prefix != source_prefix:
        raise RuntimeError("review dataset prefix differs from exhaustive source manifest")
    light = source.get("dataset_identity_light", {})
    if Path(str(light.get("resolved_prefix", ""))).resolve() != requested_prefix:
        raise RuntimeError("source manifest lightweight dataset prefix differs")
    for suffix in ("idx", "bin"):
        expected = light.get(suffix, {})
        physical = Path(str(expected.get("path", ""))).resolve()
        required = Path(str(requested_prefix) + f".{suffix}")
        if physical != required or not physical.is_file():
            raise RuntimeError(f"source manifest {suffix} path differs/missing")
        if int(expected.get("size_bytes", -1)) != int(physical.stat().st_size):
            raise RuntimeError(f"source manifest {suffix} size differs")
    identities = [_identity(path)]
    if isinstance(source.get("windows"), Mapping):
        identities.append(
            _verify_identity(source["windows"], label="source_manifest.windows")
        )
    return {
        "manifest": _identity(path),
        "manifest_content_sha256": source["manifest_content_sha256"],
        "dataset_prefix": str(requested_prefix),
        "dataset_idx_size": int(Path(str(requested_prefix) + ".idx").stat().st_size),
        "dataset_bin_size": int(Path(str(requested_prefix) + ".bin").stat().st_size),
        "source_window_count": int(source.get("statistics", {}).get("window_count", -1)),
        "source_eligible_token_count": int(
            source.get("statistics", {}).get("eligible_token_count", -1)
        ),
    }, identities


def _verify_prior_identities(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verify every physical SHA identity intentionally exposed by the review.

    The threshold review contains only open calibration bindings; it does not
    include or dereference the sealed test artifact.  Recursive traversal is
    therefore safe and gives an auditable list of every checked file.
    """

    found: dict[tuple[str, str], Mapping[str, Any]] = {}

    def visit(value: Any, trail: str) -> None:
        if isinstance(value, Mapping):
            if {"path", "size_bytes", "sha256"}.issubset(value):
                found[(str(value["path"]), str(value["sha256"]))] = value
            for key, child in value.items():
                visit(child, f"{trail}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{trail}[{index}]")

    visit(payload, "threshold_review")
    verified = []
    for index, expected in enumerate(found.values()):
        verified.append(
            _verify_identity(expected, label=f"threshold_review_identity_{index}")
        )
    return verified


def _load_and_validate_shards(
    exact_dir: Path,
    summary: Mapping[str, Any],
    candidate_rows: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Reconstruct final/same/B masks and condition totals from journal shards."""

    progress_path = exact_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("schema") != PROGRESS_SCHEMA or not progress.get("finalized"):
        raise RuntimeError("exact targeted progress is not finalized")
    _verify_identity(progress.get("summary", {}), expected_path=exact_dir / "summary.json", label="progress.summary")
    if int(progress.get("processed_windows", -1)) != len(candidate_rows):
        raise RuntimeError("progress processed-window count differs from candidates")

    packed = {
        level: np.zeros((len(candidate_rows), PACKED_BYTES_PER_WINDOW), dtype=np.uint8)
        for level in BUNDLE_LEVELS
    }
    same = {level: np.zeros_like(packed[level]) for level in BUNDLE_LEVELS}
    b_only = {level: np.zeros_like(packed[level]) for level in BUNDLE_LEVELS}
    conditions = {
        level: np.zeros(len(CONDITION_COUNT_NAMES), dtype=np.int64)
        for level in BUNDLE_LEVELS
    }
    occurrence_parts: dict[int, list[np.ndarray]] = {
        level: [] for level in BUNDLE_LEVELS
    }
    seen = np.zeros(len(candidate_rows), dtype=np.bool_)
    next_batch = 0
    verified_shards = 0
    for shard_index, record in enumerate(progress.get("shards", [])):
        if int(record.get("index", -1)) != shard_index:
            raise RuntimeError("exact shard index is non-contiguous")
        if int(record.get("batch_start", -1)) != next_batch:
            raise RuntimeError("exact shard batch journal has a gap")
        shard_path = Path(record["path"]).resolve()
        _verify_identity(record, expected_path=shard_path, label=f"exact.shard.{shard_index}")
        with np.load(shard_path, allow_pickle=False) as shard:
            metadata = _decode_metadata(shard["metadata_json"])
            if metadata.get("schema") != SHARD_SCHEMA:
                raise RuntimeError("exact shard metadata schema differs")
            for key in ("index", "batch_start", "batch_stop", "window_count"):
                if int(metadata.get(key, -1)) != int(record.get(key, -2)):
                    raise RuntimeError(f"exact shard metadata differs for {key}")
            indices = np.asarray(shard["candidate_indices"], dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= len(candidate_rows)):
                raise RuntimeError("exact shard candidate index is out of range")
            if seen[indices].any() or np.unique(indices).size != indices.size:
                raise RuntimeError("exact shard candidate coverage is duplicated")
            if not np.array_equal(shard["rows"], candidate_rows[indices]):
                raise RuntimeError("exact shard row identity differs from candidates")
            seen[indices] = True
            for level in BUNDLE_LEVELS:
                primary_value = np.asarray(shard[f"bundle_{level}_packed"])
                same_value = np.asarray(shard[f"bundle_{level}_same_layer_packed"])
                b_value = np.asarray(shard[f"bundle_{level}_b_packed"])
                for name, value in (
                    ("primary", primary_value),
                    ("same", same_value),
                    ("B", b_value),
                ):
                    if value.dtype != np.uint8 or value.shape != (
                        len(indices),
                        PACKED_BYTES_PER_WINDOW,
                    ):
                        raise RuntimeError(f"exact shard {name} packed shape differs")
                packed[level][indices] = primary_value
                same[level][indices] = same_value
                b_only[level][indices] = b_value
                occurrence_parts[level].append(shard[f"bundle_{level}_occurrences"])
                shard_counts = np.asarray(
                    shard[f"bundle_{level}_condition_counts"], dtype=np.int64
                )
                if shard_counts.shape != (len(CONDITION_COUNT_NAMES),):
                    raise RuntimeError("exact shard condition count shape differs")
                conditions[level] += shard_counts
        next_batch = int(record["batch_stop"])
        verified_shards += 1
    if not seen.all():
        raise RuntimeError("exact shard candidate coverage is incomplete")
    if next_batch != int(progress.get("next_batch_index", -1)):
        raise RuntimeError("exact shard journal terminal batch differs")

    for level in BUNDLE_LEVELS:
        final_packed = np.load(
            exact_dir / f"bundle_{level}_packed.npy", mmap_mode="r", allow_pickle=False
        )
        if not np.array_equal(packed[level], final_packed):
            raise RuntimeError(f"bundle {level} final packed mask differs from shards")
        occurrence = (
            np.concatenate(occurrence_parts[level])
            if occurrence_parts[level]
            else np.empty(0, dtype=OCCURRENCE_DTYPE)
        )
        if occurrence.size:
            occurrence = occurrence[
                np.lexsort((occurrence["position"], occurrence["candidate_index"]))
            ]
        final_occurrence = np.load(
            exact_dir / f"bundle_{level}_occurrences.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if not np.array_equal(occurrence, final_occurrence):
            raise RuntimeError(f"bundle {level} final occurrences differ from shards")
        summary_counts = summary["bundles"][str(level)]["condition_counts"]
        for index, name in enumerate(CONDITION_COUNT_NAMES):
            if int(conditions[level][index]) != int(summary_counts[name]):
                raise RuntimeError(
                    f"bundle {level} condition count differs for {name}"
                )
    return same, b_only, conditions, verified_shards


def _verify_masks_occurrences_and_dataset(
    *,
    candidate_rows: np.ndarray,
    masks: Mapping[int, np.ndarray],
    occurrences: Mapping[int, np.ndarray],
    dataset: Any,
) -> dict[str, Any]:
    """Prove mask/occurrence bijection and every selected token's source ID."""

    if candidate_rows.dtype != WINDOW_DTYPE or candidate_rows.ndim != 1:
        raise RuntimeError("candidate windows have wrong dtype/shape")
    if np.any(candidate_rows["window_length"] > 512):
        raise RuntimeError("candidate window exceeds the 512-token axis")
    if np.any(candidate_rows["eligible_token_count"] > candidate_rows["window_length"]):
        raise RuntimeError("candidate eligible length exceeds window length")
    eligible = (
        np.arange(512, dtype=np.int32)[None, :]
        < candidate_rows["eligible_token_count"][:, None]
    )
    union_active = np.zeros(len(candidate_rows), dtype=np.bool_)
    for level in BUNDLE_LEVELS:
        mask, occurrence = masks[level], occurrences[level]
        if mask.shape != (len(candidate_rows), 512) or mask.dtype != np.bool_:
            raise RuntimeError(f"bundle {level} unpacked mask shape/dtype differs")
        if np.any(mask & ~eligible):
            raise RuntimeError(f"bundle {level} selects an ineligible suffix")
        if occurrence.dtype != OCCURRENCE_DTYPE or occurrence.ndim != 1:
            raise RuntimeError(f"bundle {level} occurrence dtype/shape differs")
        candidate_index, position = np.nonzero(mask)
        if occurrence.size != candidate_index.size:
            raise RuntimeError(f"bundle {level} mask/occurrence cardinality differs")
        if not np.array_equal(occurrence["candidate_index"], candidate_index):
            raise RuntimeError(f"bundle {level} occurrence candidate ordering differs")
        if not np.array_equal(occurrence["position"], position):
            raise RuntimeError(f"bundle {level} occurrence position ordering differs")
        selected_rows = candidate_rows[candidate_index]
        for name in (
            "sample_order",
            "source_window_index",
            "document_id",
            "window_offset",
        ):
            if not np.array_equal(occurrence[name], selected_rows[name]):
                raise RuntimeError(f"bundle {level} occurrence {name} differs")
        union_active |= mask.any(axis=1)

    token_checks = 0
    active_indices = np.flatnonzero(union_active)
    token_cache: dict[int, np.ndarray] = {}
    for candidate in active_indices.tolist():
        row = candidate_rows[candidate]
        token_ids = np.asarray(
            load_document_window(
                dataset,
                int(row["document_id"]),
                int(row["window_offset"]),
                int(row["window_length"]),
            ),
            dtype=np.int64,
        )
        if token_ids.shape != (int(row["window_length"]),):
            raise RuntimeError("IndexedDataset returned a padded/malformed window")
        token_cache[candidate] = token_ids
    for level in BUNDLE_LEVELS:
        occurrence = occurrences[level]
        unique_candidates, starts, counts = np.unique(
            occurrence["candidate_index"], return_index=True, return_counts=True
        )
        for candidate, start, count in zip(
            unique_candidates.tolist(), starts.tolist(), counts.tolist()
        ):
            selected = occurrence[start : start + count]
            observed = token_cache[int(candidate)][selected["position"].astype(np.int64)]
            if not np.array_equal(observed.astype(np.int32), selected["token_id"]):
                raise RuntimeError(
                    f"bundle {level} occurrence token IDs differ from IndexedDataset"
                )
            token_checks += int(selected.size)
    return {
        "mask_occurrence_bijection": True,
        "bundle_nesting_checked_separately": True,
        "ineligible_suffix_selected": 0,
        "source_windows_read_without_padding": int(active_indices.size),
        "source_token_id_comparisons": token_checks,
        "all_selected_occurrence_token_ids_verified": True,
    }


def recommendation(
    bundle_rows: Sequence[Mapping[str, Any]],
    *,
    min_selected: int,
    min_documents: int,
    min_same_layer_jaccard: float,
    max_document_share: float,
    max_window_share: float,
    max_token_share: float,
) -> dict[str, Any]:
    """Choose the strictest exact bundle that clears predeclared audit gates."""

    by_level = {int(row["bundle"]): row for row in bundle_rows}
    decisions: dict[str, Any] = {}
    chosen: int | None = None
    for level in BUNDLE_LEVELS:  # 95 strictest -> 99 loosest.
        row = by_level[level]
        document_share = row.get("max_document_share")
        window_share = row.get("max_window_share")
        token_share = row.get("top_token_share")
        gates = {
            "minimum_exact_occurrences": int(row["exact_selected_occurrences"])
            >= min_selected,
            "minimum_active_documents": int(row["active_documents"])
            >= min_documents,
            "same_layer_jaccard": float(row["same_layer_jaccard"])
            >= min_same_layer_jaccard,
            "maximum_document_share": document_share is not None
            and float(document_share) <= max_document_share,
            "maximum_window_share": window_share is not None
            and float(window_share) <= max_window_share,
            "maximum_token_id_share": token_share is not None
            and float(token_share) <= max_token_share,
        }
        decisions[str(level)] = {
            "strictness": "strictest" if level == 95 else "loosest" if level == 99 else "middle",
            "gates": gates,
            "passes_all": bool(all(gates.values())),
        }
        if chosen is None and all(gates.values()):
            chosen = level
    return {
        "policy": (
            "precision-first: examine 95 (strictest), then 97, then 99 "
            "(loosest); recommend the first bundle passing every exact audit gate"
        ),
        "bundle_order": [95, 97, 99],
        "bundle_95_is_strictest": True,
        "bundle_99_is_loosest": True,
        "gates": {
            "min_selected": min_selected,
            "min_documents": min_documents,
            "min_same_layer_jaccard": min_same_layer_jaccard,
            "max_document_share": max_document_share,
            "max_window_share": max_window_share,
            "max_token_share": max_token_share,
        },
        "per_bundle": decisions,
        "recommended_bundle": chosen,
        "status": (
            "READY_FOR_HUMAN_THRESHOLD_LOCK"
            if chosen is not None
            else "NO_BUNDLE_PASSES_EXACT_AUDIT"
        ),
        "human_lock_required": True,
        "sealed_test_opened": False,
        "note": (
            "Reservoir-CI agreement is diagnostic, not a gate: the exact pass "
            "supersedes the 5M-reservoir count estimate. Inspect contexts before lock."
        ),
    }


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Exact targeted CKA GT review",
        "",
        "This review validates exact Code-train occurrences. It does **not** open the",
        "sealed pilot test split and does **not** lock or materialize final GT.",
        "Bundle **95 is the strictest** (tightest Wiki-calibrated cuts), bundle 97",
        "is intermediate, and bundle **99 is the loosest**.",
        "",
        "## Exact result versus the 5M-reservoir estimate",
        "",
        "| Bundle | Exact count | Full eligible coverage | Reservoir estimate | CI95 | In CI | Same-layer Jaccard |",
        "|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in payload["bundle_summary"]:
        lines.append(
            f"| {row['bundle']} | {row['exact_selected_occurrences']:,} | "
            f"{row['full_eligible_coverage']:.6%} | "
            f"{row['reservoir_estimated_count']:,} | "
            f"{row['reservoir_ci95_low']:,}–{row['reservoir_ci95_high']:,} | "
            f"{'yes' if row['exact_count_inside_reservoir_ci95'] else 'no'} | "
            f"{row['same_layer_jaccard']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The reservoir interval is a sampling diagnostic. Exact counts supersede",
            "the estimate; an out-of-CI result is reported rather than silently rejected.",
            "",
            "## Exact concentration",
            "",
            "| Bundle | Windows | Documents | Max window | Top-5 windows | Window HHI | Max doc | Top-5 docs | Doc HHI | Token entropy | Top token |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["bundle_summary"]:
        lines.append(
            f"| {row['bundle']} | {row['active_windows']:,} | "
            f"{row['active_documents']:,} | {row['max_window_share']:.4%} | "
            f"{row['top5_window_share']:.4%} | {row['window_hhi']:.6g} | "
            f"{row['max_document_share']:.4%} | {row['top5_document_share']:.4%} | "
            f"{row['document_hhi']:.6g} | {row['token_entropy_bits']:.4f} | "
            f"{row['top_token_share']:.4%} |"
        )
    recommendation_payload = payload["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Status: `{recommendation_payload['status']}`",
            f"- Recommended candidate: `{recommendation_payload['recommended_bundle']}`",
            f"- Policy: {recommendation_payload['policy']}",
            "- This is not an automatic final label decision. Review deterministic",
            "  context samples, then record a separate human threshold lock.",
            "",
            "## Validation",
            "",
            f"- Exact shards with SHA-256 verified: {payload['validation']['verified_exact_shards']:,}",
            f"- Selected source windows read without padding: {payload['validation']['source_windows_read_without_padding']:,}",
            f"- Token-ID comparisons against Code IndexedDataset: {payload['validation']['source_token_id_comparisons']:,}",
            "- Packed-mask/occurrence bijections: passed",
            "- 95 subset 97 subset 99: passed",
            "- Sealed pilot test opened: **false**",
            "",
            "Condition marginals in the tables are exact *within the B99 candidate-window",
            "axis*. T/L2/R were intentionally not recomputed outside that targeted axis,",
            "so they must not be presented as unconditional full-corpus marginals.",
            "",
            "Semantic Wiki-likeness is not proven by CKA stability alone. The decoded",
            "context samples are the required human inspection surface before locking GT.",
            "",
        ]
    )
    return "\n".join(lines)


def run_review(
    *,
    exact_dir: str | Path,
    threshold_review_dir: str | Path,
    dataset_prefix: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path | None = DEFAULT_TOKENIZER,
    seed: int = 20260816,
    representative_contexts: int = 100,
    top_units: int = 10,
    context_radius: int = 32,
    min_selected: int = 100,
    min_documents: int = 10,
    min_same_layer_jaccard: float = 0.9,
    max_document_share: float = 0.25,
    max_window_share: float = 0.05,
    max_token_share: float = 0.25,
    dataset: Any | None = None,
    decoder: Any | None = None,
) -> dict[str, Any]:
    exact_dir = Path(exact_dir).resolve()
    threshold_review_dir = Path(threshold_review_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty review dir: {output_dir}")
    summary_path = exact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or not summary.get("complete")
        or not summary.get("not_human_locked_final_gt")
        or summary.get("sealed_test_opened")
    ):
        raise RuntimeError("exact targeted summary is incomplete/locked/test-opened")
    if tuple(int(item) for item in summary.get("bundle_levels", ())) != BUNDLE_LEVELS:
        raise RuntimeError("exact targeted summary bundle levels differ")

    hash_checks: list[dict[str, Any]] = [_identity(summary_path)]
    candidate_identity = summary.get("candidate_windows", {})
    candidate_path = Path(candidate_identity.get("path", "")).resolve()
    hash_checks.append(
        _verify_identity(
            candidate_identity,
            expected_path=candidate_path,
            label="exact.candidate_windows",
        )
    )
    candidate_rows = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    if candidate_rows.dtype != WINDOW_DTYPE or candidate_rows.ndim != 1:
        raise RuntimeError("candidate-window axis is malformed")
    if len(candidate_rows) != int(summary.get("candidate_window_count", -1)):
        raise RuntimeError("exact summary candidate count differs")

    manifest_identity = summary.get("candidate_manifest", {})
    hash_checks.append(
        _verify_identity(manifest_identity, label="exact.candidate_manifest")
    )
    manifest = json.loads(Path(manifest_identity["path"]).read_text(encoding="utf-8"))
    if manifest.get("schema") != "cka_gt_b_candidate_windows_v2":
        raise RuntimeError("candidate manifest schema differs")
    manifest_hash_verified = _verify_manifest_content_hash(manifest)
    worker_bindings = manifest.get("census_worker_model_bindings", {})
    binding_content = dict(worker_bindings) if isinstance(worker_bindings, Mapping) else {}
    recorded_binding_hash = binding_content.pop("binding_set_content_sha256", None)
    if (
        worker_bindings.get("schema") != "cka_gt_full_census_model_binding_set_v1"
        or worker_bindings.get("validated") is not True
        or worker_bindings.get("all_workers_agree") is not True
        or int(worker_bindings.get("worker_count", -1)) != 4
        or recorded_binding_hash != hashlib.sha256(_canonical_json(binding_content)).hexdigest()
        or worker_bindings.get("common_content_sha256")
        != hashlib.sha256(_canonical_json(worker_bindings.get("common_content", {}))).hexdigest()
        or worker_bindings != summary.get("census_worker_model_bindings")
    ):
        raise RuntimeError("candidate/exact census-worker model-binding lineage differs")
    source_binding, source_identities = _verify_source_manifest_and_dataset(
        manifest, dataset_prefix
    )
    if source_binding["source_eligible_token_count"] != int(
        summary["full_eligible_token_count"]
    ):
        raise RuntimeError("source manifest/exact eligible-token denominator differs")
    hash_checks.extend(source_identities)
    candidate_manifest_identities = _verify_prior_identities(manifest)
    hash_checks.extend(candidate_manifest_identities)
    hash_checks.append(
        _verify_identity(
            manifest.get("candidate_windows", {}),
            expected_path=candidate_path,
            label="candidate_manifest.candidate_windows",
        )
    )
    hash_checks.append(
        _verify_identity(
            summary.get("analysis_config", {}),
            label="exact.analysis_config",
        )
    )
    hash_checks.append(
        _verify_identity(
            summary.get("authoritative_b", {}),
            label="exact.authoritative_b",
        )
    )
    for level in BUNDLE_LEVELS:
        expected = summary.get("outputs", {}).get(str(level), {})
        for key, filename in (
            ("packed_mask", f"bundle_{level}_packed.npy"),
            ("occurrences", f"bundle_{level}_occurrences.npy"),
        ):
            hash_checks.append(
                _verify_identity(
                    expected.get(key, {}),
                    expected_path=exact_dir / filename,
                    label=f"exact.bundle{level}.{key}",
                )
            )

    threshold_analysis_path = threshold_review_dir / "analysis.json"
    threshold_report_path = threshold_review_dir / "REPORT.md"
    threshold_review = json.loads(threshold_analysis_path.read_text(encoding="utf-8"))
    if (
        threshold_review.get("schema") != EXPECTED_THRESHOLD_REVIEW_SCHEMA
        or not threshold_review.get("complete")
        or not threshold_review.get("threshold_review_only")
        or threshold_review.get("exact_gt_created")
        or threshold_review.get("sealed_test_opened")
    ):
        raise RuntimeError("prior threshold review is invalid or test-opened")
    if int(threshold_review.get("full_eligible_tokens", -1)) != int(
        summary["full_eligible_token_count"]
    ):
        raise RuntimeError("prior review/exact full eligible denominator differs")
    if not threshold_report_path.is_file():
        raise FileNotFoundError(threshold_report_path)
    hash_checks.extend([_identity(threshold_analysis_path), _identity(threshold_report_path)])
    prior_nested_identities = _verify_prior_identities(threshold_review)
    hash_checks.extend(prior_nested_identities)

    same_packed, b_packed, condition_counts, verified_shards = _load_and_validate_shards(
        exact_dir, summary, candidate_rows
    )
    b_count_artifact = summary.get("exact_b_count_validation", {}).get("artifact", {})
    hash_checks.append(
        _verify_identity(
            b_count_artifact,
            expected_path=exact_dir / "b_count_validation.json",
            label="exact.b_count_validation",
        )
    )
    b_count_validation = json.loads(
        (exact_dir / "b_count_validation.json").read_text(encoding="utf-8")
    )
    if not b_count_validation.get("passed"):
        raise RuntimeError("exact B-count validation did not pass")
    b_index = CONDITION_COUNT_NAMES.index("B")
    for level in BUNDLE_LEVELS:
        expected_b = int(
            manifest["statistics"]["per_bundle"][str(level)]["B_passing_tokens"]
        )
        observed_b = int(condition_counts[level][b_index])
        record = b_count_validation.get("bundles", {}).get(str(level), {})
        if (
                observed_b != expected_b
                or int(record.get("expected_from_candidate_manifest", -1)) != expected_b
                or int(record.get("observed_authoritative_full_census", -1))
                != observed_b
                or record.get("authoritative_matches") is not True
            ):
            raise RuntimeError(f"bundle {level} exact B count differs from candidate manifest")
    packed: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    same_masks: dict[int, np.ndarray] = {}
    occurrences: dict[int, np.ndarray] = {}
    for level in BUNDLE_LEVELS:
        packed[level] = np.load(
            exact_dir / f"bundle_{level}_packed.npy", allow_pickle=False
        )
        if packed[level].dtype != np.uint8 or packed[level].shape != (
            len(candidate_rows),
            PACKED_BYTES_PER_WINDOW,
        ):
            raise RuntimeError(f"bundle {level} packed mask shape/dtype differs")
        masks[level] = unpack_mask(packed[level])
        same_masks[level] = unpack_mask(same_packed[level])
        occurrences[level] = np.load(
            exact_dir / f"bundle_{level}_occurrences.npy", allow_pickle=False
        )
    nesting = {
        "95_subset_97": bool(np.all(~masks[95] | masks[97])),
        "97_subset_99": bool(np.all(~masks[97] | masks[99])),
    }
    if not all(nesting.values()):
        raise RuntimeError(f"exact bundle nesting failed: {nesting}")
    if not unpack_mask(b_packed[99]).any(axis=1).all():
        raise RuntimeError("candidate axis contains a window without recomputed B99")

    if dataset is None:
        dataset = MMapIndexedDatasetLite(str(dataset_prefix))
    dataset_validation = _verify_masks_occurrences_and_dataset(
        candidate_rows=candidate_rows,
        masks=masks,
        occurrences=occurrences,
        dataset=dataset,
    )
    if decoder is None:
        decoder, tokenizer_info = _load_tokenizer(tokenizer_path)
    else:
        tokenizer_info = {
            "available": True,
            "backend": "injected_decoder",
            "path": None,
            "fallback_used": False,
        }

    bundle_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    repetition_rows: list[dict[str, Any]] = []
    top_document_rows: list[dict[str, Any]] = []
    top_window_rows: list[dict[str, Any]] = []
    context_files: dict[str, Any] = {}
    concentration_by_level: dict[int, dict[str, Any]] = {}
    repetition_by_level: dict[int, dict[str, Any]] = {}
    full_eligible = int(summary["full_eligible_token_count"])
    candidate_eligible = int(candidate_rows["eligible_token_count"].sum(dtype=np.int64))

    for level in BUNDLE_LEVELS:
        occurrence = occurrences[level]
        window_concentration = concentration_metrics(
            occurrence["sample_order"], top_k=top_units
        )
        document_concentration = concentration_metrics(
            occurrence["document_id"], top_k=top_units
        )
        repetition = token_repetition_metrics(
            occurrence["token_id"], occurrence["sample_order"], top_k=top_units
        )
        concentration_by_level[level] = {
            "window": window_concentration,
            "document": document_concentration,
        }
        repetition_by_level[level] = repetition
        primary, same = masks[level], same_masks[level]
        intersection = int(np.count_nonzero(primary & same))
        union = int(np.count_nonzero(primary | same))
        exact_count = int(occurrence.size)
        boot = threshold_review["document_bootstrap"][str(level)]
        estimate = int(boot["estimated_full_count"])
        ci_low, ci_high = [int(item) for item in boot["estimated_full_count_ci95"]]
        reservoir_hits = int(
            threshold_review["candidates"][str(level)]["counts"]["selected"]
        )
        row = {
            "bundle": level,
            "strictness": "strictest" if level == 95 else "loosest" if level == 99 else "middle",
            "exact_selected_occurrences": exact_count,
            "full_eligible_tokens": full_eligible,
            "full_eligible_coverage": exact_count / full_eligible,
            "candidate_eligible_tokens": candidate_eligible,
            "candidate_axis_coverage": exact_count / candidate_eligible,
            "reservoir_rows": int(threshold_review["reservoir_rows"]),
            "reservoir_selected_hits": reservoir_hits,
            "reservoir_estimated_count": estimate,
            "reservoir_ci95_low": ci_low,
            "reservoir_ci95_high": ci_high,
            "exact_count_inside_reservoir_ci95": ci_low <= exact_count <= ci_high,
            "exact_minus_reservoir_estimate": exact_count - estimate,
            "exact_over_reservoir_estimate": exact_count / estimate if estimate else None,
            "same_layer_count": int(same.sum()),
            "same_layer_intersection": intersection,
            "same_layer_union": union,
            "same_layer_jaccard": intersection / union if union else 1.0,
            "active_windows": window_concentration["active_units"],
            "active_documents": document_concentration["active_units"],
            "max_window_share": window_concentration["max_share"],
            "top5_window_share": window_concentration["top5_share"],
            "window_hhi": window_concentration["hhi"],
            "window_gini": window_concentration["gini_among_active"],
            "max_document_share": document_concentration["max_share"],
            "top5_document_share": document_concentration["top5_share"],
            "document_hhi": document_concentration["hhi"],
            "document_gini": document_concentration["gini_among_active"],
            "unique_token_ids": repetition["unique_token_ids"],
            "token_entropy_bits": repetition["entropy_bits"],
            "normalized_token_entropy": repetition["normalized_entropy"],
            "top_token_share": repetition["top_token_share"],
            "excess_repeat_fraction": repetition["excess_repeat_fraction"],
            "within_window_excess_same_token_fraction": repetition[
                "within_window_excess_same_token_fraction"
            ],
        }
        bundle_rows.append(row)

        for index, name in enumerate(CONDITION_COUNT_NAMES):
            count = int(condition_counts[level][index])
            marginal_rows.append(
                {
                    "bundle": level,
                    "condition": name,
                    "exact_pass_count_on_candidate_axis": count,
                    "candidate_eligible_tokens": candidate_eligible,
                    "fraction_of_candidate_eligible": count / candidate_eligible,
                    "scope": "B99 candidate-window axis; not unconditional full corpus",
                }
            )
        for unit_name, result in (
            ("window", window_concentration),
            ("document", document_concentration),
        ):
            concentration_rows.append(
                {
                    "bundle": level,
                    "unit": unit_name,
                    **{key: value for key, value in result.items() if key != "top_units"},
                }
            )
        repetition_rows.append(
            {
                "bundle": level,
                **{key: value for key, value in repetition.items() if key != "top_token_ids"},
            }
        )
        for rank, item in enumerate(document_concentration["top_units"], start=1):
            top_document_rows.append(
                {"bundle": level, "rank": rank, "document_id": item["id"], **{key: value for key, value in item.items() if key != "id"}}
            )
        for rank, item in enumerate(window_concentration["top_units"], start=1):
            sample_order = int(item["id"])
            matches = np.flatnonzero(candidate_rows["sample_order"] == sample_order)
            if matches.size != 1:
                raise RuntimeError("top window sample_order does not map uniquely")
            candidate = int(matches[0])
            source = candidate_rows[candidate]
            top_window_rows.append(
                {
                    "bundle": level,
                    "rank": rank,
                    "candidate_index": candidate,
                    "sample_order": sample_order,
                    "document_id": int(source["document_id"]),
                    "window_offset": int(source["window_offset"]),
                    "count": item["count"],
                    "share": item["share"],
                }
            )

        chosen = _chosen_occurrence_indices(
            occurrence,
            level=level,
            count=representative_contexts,
            seed=seed,
        )
        records = [
            _context_record(
                occurrence[index],
                level=level,
                reason="deterministic_representative",
                candidate_rows=candidate_rows,
                dataset=dataset,
                decoder=decoder,
                radius=context_radius,
            )
            for index in chosen
        ]
        records.extend(
            _context_record(
                occurrence[index],
                level=level,
                reason="top_concentrated_document",
                candidate_rows=candidate_rows,
                dataset=dataset,
                decoder=decoder,
                radius=context_radius,
            )
            for index in _top_context_indices(
                occurrence,
                field="document_id",
                top_units=document_concentration["top_units"],
                level=level,
                seed=seed + 1,
            )
        )
        # occurrence candidate_index is the window unit, whereas the public
        # top-window table uses global sample_order.
        candidate_top = [
            {"id": row["candidate_index"]}
            for row in top_window_rows
            if int(row["bundle"]) == level
        ]
        records.extend(
            _context_record(
                occurrence[index],
                level=level,
                reason="top_concentrated_window",
                candidate_rows=candidate_rows,
                dataset=dataset,
                decoder=decoder,
                radius=context_radius,
            )
            for index in _top_context_indices(
                occurrence,
                field="candidate_index",
                top_units=candidate_top,
                level=level,
                seed=seed + 2,
            )
        )
        context_path = output_dir / "contexts" / f"bundle_{level}.jsonl"
        _atomic_jsonl(context_path, records)
        context_files[str(level)] = {
            **_identity(context_path),
            "records": len(records),
            "representative_records": len(chosen),
            "top_document_records": len(document_concentration["top_units"]),
            "top_window_records": len(window_concentration["top_units"]),
        }

    pairwise_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(BUNDLE_LEVELS):
        for right in BUNDLE_LEVELS[left_index + 1 :]:
            intersection = int(np.count_nonzero(masks[left] & masks[right]))
            union = int(np.count_nonzero(masks[left] | masks[right]))
            pairwise_rows.append(
                {
                    "left_bundle": left,
                    "right_bundle": right,
                    "left_count": int(masks[left].sum()),
                    "right_count": int(masks[right].sum()),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 1.0,
                    "left_subset_of_right": bool(np.all(~masks[left] | masks[right])),
                }
            )

    same_layer_rows = [
        {
            "bundle": row["bundle"],
            "condition_specific_count": row["exact_selected_occurrences"],
            "same_layer_count": row["same_layer_count"],
            "intersection": row["same_layer_intersection"],
            "union": row["same_layer_union"],
            "jaccard": row["same_layer_jaccard"],
            "manual_review_required": row["same_layer_jaccard"] < 0.9,
        }
        for row in bundle_rows
    ]

    recommendation_payload = recommendation(
        bundle_rows,
        min_selected=min_selected,
        min_documents=min_documents,
        min_same_layer_jaccard=min_same_layer_jaccard,
        max_document_share=max_document_share,
        max_window_share=max_window_share,
        max_token_share=max_token_share,
    )
    validation = {
        "schema": VALIDATION_SCHEMA,
        "passed": True,
        "sealed_test_opened": False,
        "manifest_content_hash_verified": manifest_hash_verified,
        "verified_sha256_identities": len(hash_checks),
        "verified_prior_review_nested_identities": len(prior_nested_identities),
        "verified_exact_shards": verified_shards,
        "exact_shard_candidate_coverage_once": True,
        "final_masks_equal_shard_reconstruction": True,
        "final_occurrences_equal_shard_reconstruction": True,
        "condition_counts_equal_shard_reconstruction": True,
        "exact_b_counts_equal_candidate_manifest": True,
        "census_worker_model_binding_lineage_verified": True,
        "bundle_nesting": nesting,
        **dataset_validation,
    }
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "review_only": True,
        "exact_gt_materialized": False,
        "human_threshold_locked": False,
        "sealed_test_opened": False,
        "bundle_strictness": {
            "95": "strictest/tightest",
            "97": "intermediate",
            "99": "loosest",
        },
        "source_exact_summary": _identity(summary_path),
        "source_threshold_review_analysis": _identity(threshold_analysis_path),
        "source_threshold_review_report": _identity(threshold_report_path),
        "source_candidate_windows": _identity(candidate_path),
        "source_manifest_binding": source_binding,
        "dataset_prefix": str(Path(dataset_prefix).resolve()),
        "tokenizer": tokenizer_info,
        "deterministic_context_seed": int(seed),
        "full_eligible_tokens": full_eligible,
        "candidate_eligible_tokens": candidate_eligible,
        "bundle_summary": bundle_rows,
        "condition_marginals_scope": (
            "exact within B99 candidate windows; T/M outside this axis were not computed"
        ),
        "condition_marginals": {
            str(level): {
                name: int(condition_counts[level][index])
                for index, name in enumerate(CONDITION_COUNT_NAMES)
            }
            for level in BUNDLE_LEVELS
        },
        "concentration": {
            str(level): concentration_by_level[level] for level in BUNDLE_LEVELS
        },
        "token_repetition": {
            str(level): repetition_by_level[level] for level in BUNDLE_LEVELS
        },
        "pairwise_overlap": pairwise_rows,
        "context_files": context_files,
        "recommendation": recommendation_payload,
        "validation": validation,
        "limitations": [
            "CKA stability is a replay-anchor criterion, not proof of semantic Wiki membership",
            "condition marginals are conditional on the targeted B99 candidate-window axis",
            "decoded contexts require human review before threshold lock",
            "pilot sealed test remains unopened and cannot influence this recommendation",
        ],
        "next_step": (
            "human reviews contexts and exact tables, explicitly locks one bundle, "
            "then materialize_locked_bundle creates immutable GT"
        ),
    }

    _write_csv(output_dir / "tables" / "bundle_summary.csv", bundle_rows)
    _write_csv(output_dir / "tables" / "exact_vs_reservoir.csv", bundle_rows)
    _write_csv(output_dir / "tables" / "condition_marginals.csv", marginal_rows)
    _write_csv(output_dir / "tables" / "concentration.csv", concentration_rows)
    _write_csv(output_dir / "tables" / "token_id_repetition.csv", repetition_rows)
    _write_csv(output_dir / "tables" / "same_layer_diagnostic.csv", same_layer_rows)
    _write_csv(output_dir / "tables" / "pairwise_overlap.csv", pairwise_rows)
    _write_csv(output_dir / "tables" / "top_documents.csv", top_document_rows)
    _write_csv(output_dir / "tables" / "top_windows.csv", top_window_rows)
    _atomic_json(output_dir / "validation.json", validation)
    _atomic_json(output_dir / "analysis.json", payload)
    _atomic_text(output_dir / "REPORT.md", _report(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--threshold-review-dir", required=True)
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--representative-contexts", type=int, default=100)
    parser.add_argument("--top-units", type=int, default=10)
    parser.add_argument("--context-radius", type=int, default=32)
    parser.add_argument("--min-selected", type=int, default=100)
    parser.add_argument("--min-documents", type=int, default=10)
    parser.add_argument("--min-same-layer-jaccard", type=float, default=0.9)
    parser.add_argument("--max-document-share", type=float, default=0.25)
    parser.add_argument("--max-window-share", type=float, default=0.05)
    parser.add_argument("--max-token-share", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_review(
        exact_dir=args.exact_dir,
        threshold_review_dir=args.threshold_review_dir,
        dataset_prefix=args.dataset_prefix,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
        representative_contexts=args.representative_contexts,
        top_units=args.top_units,
        context_radius=args.context_radius,
        min_selected=args.min_selected,
        min_documents=args.min_documents,
        min_same_layer_jaccard=args.min_same_layer_jaccard,
        max_document_share=args.max_document_share,
        max_window_share=args.max_window_share,
        max_token_share=args.max_token_share,
    )
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "concentration_metrics",
    "recommendation",
    "run_review",
    "token_repetition_metrics",
]
