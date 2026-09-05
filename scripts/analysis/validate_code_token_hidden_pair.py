#!/usr/bin/env python3
"""Validate a completed multi-worker Code token hidden-pair extraction.

The inexpensive checks validate worker/run metadata, exact partition coverage,
shard manifests, progress, and the final raw-hidden reservoir.  ``--deep`` also
streams through every token-metric shard and verifies array shapes/dtypes,
finite/range constraints, sample identity, and the logical token hash recorded
in each sidecar.  No histogram or GT is produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from code_token_hidden_pair_core import METRIC_NAMES, SCHEMA_VERSION


DEFAULT_REFERENCE = (
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
    "g2_olddata_kd_9run_20260808/01_common_kd_init/"
    "code_e8_to_e16_wiki_kd_step600"
)
DEFAULT_CURRENT = (
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
    "flame_code_bootstrap_20260810/"
    "g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt-v2"
)
DEFAULT_DATASET = (
    "/data2/seonghyeonnoh/LLM-continual-learning-data/"
    "flamedata2.data2-verified-backup/code/train/train_text_document"
)
WORKER_PATTERN = re.compile(r"^rank_(\d{3})$")


class ValidationError(RuntimeError):
    """Raised on the first integrity violation."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"missing JSON file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValidationError(f"expected a JSON object: {path}")
    return payload


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_path(actual: Any, expected: str) -> bool:
    if not isinstance(actual, str):
        return False
    return os.path.normpath(actual) == os.path.normpath(expected)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _validate_run_metadata(
    path: Path,
    worker_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    row = _load_json(path)
    context = str(path)
    expected_layers = args.expected_layers
    checks = {
        "schema_version": SCHEMA_VERSION,
        "metric_names": list(METRIC_NAMES),
        "metric_dtype": "float32",
        "layers": expected_layers,
        "hidden_size": args.expected_hidden_size,
        "sequence_length": args.expected_sequence_length,
        "total_samples": args.expected_total_samples,
        "seed": args.expected_seed,
        "worker_index": worker_index,
        "worker_count": args.expected_workers,
        "runtime_data_parallel_rank": 0,
        "runtime_data_parallel_world_size": 1,
        "reference_tracker_step": args.expected_reference_step,
        "current_tracker_step": args.expected_current_step,
        "representation": "residual_included_transformer_layer_output",
        "dataset_split": "100,0,0",
        "natural_routing": True,
        "eval_mode": True,
        "no_grad": True,
    }
    for key, expected in checks.items():
        _require(
            row.get(key) == expected,
            f"run metadata mismatch for {key} in {context}: "
            f"expected={expected!r}, got={row.get(key)!r}",
        )
    _require(
        _same_path(row.get("reference_load"), args.expected_reference_load),
        f"reference checkpoint mismatch in {context}: {row.get('reference_load')!r}",
    )
    _require(
        _same_path(row.get("current_load"), args.expected_current_load),
        f"current checkpoint mismatch in {context}: {row.get('current_load')!r}",
    )
    data_path = row.get("code_data_path")
    _require(
        _same_path(data_path, args.expected_dataset),
        f"Code dataset mismatch in {context}: {data_path!r}",
    )
    start = row.get("partition_start_sample")
    samples = row.get("partition_samples")
    shard_samples = row.get("shard_samples")
    _require(isinstance(start, int) and start >= 0, f"invalid partition start in {context}")
    _require(isinstance(samples, int) and samples > 0, f"invalid partition size in {context}")
    _require(
        start + samples <= args.expected_total_samples,
        f"partition exceeds total in {context}: [{start}, {start + samples})",
    )
    _require(
        isinstance(shard_samples, int) and shard_samples > 0,
        f"invalid shard_samples in {context}: {shard_samples!r}",
    )
    config_hash = _json_hash(row)
    return row, config_hash


def _validate_shard_payload(
    data_path: Path,
    sidecar: dict[str, Any],
    run: dict[str, Any],
) -> None:
    samples = int(sidecar["samples"])
    sequence = int(run["sequence_length"])
    layers = np.asarray(run["layers"], dtype=np.int16)
    dense_shape = (samples, sequence)
    metric_shape = dense_shape + (layers.size,)
    required = {
        "layer_numbers",
        "sample_ids",
        "input_token_ids",
        "label_token_ids",
        "valid_mask",
        *METRIC_NAMES,
    }
    try:
        archive = np.load(data_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValidationError(f"could not open shard {data_path}: {error}") from error
    with archive as data:
        _require(
            set(data.files) == required,
            f"array set mismatch in {data_path}: expected={sorted(required)}, "
            f"got={sorted(data.files)}",
        )
        layer_numbers = data["layer_numbers"]
        _require(
            layer_numbers.dtype == np.int16
            and layer_numbers.shape == layers.shape
            and np.array_equal(layer_numbers, layers),
            f"layer_numbers mismatch in {data_path}: "
            f"shape={layer_numbers.shape}, dtype={layer_numbers.dtype}",
        )
        sample_ids = data["sample_ids"]
        _require(
            sample_ids.dtype == np.int64 and sample_ids.shape == (samples,),
            f"sample_ids shape/dtype mismatch in {data_path}: "
            f"{sample_ids.shape}/{sample_ids.dtype}",
        )
        expected_ids = np.arange(
            int(sidecar["start_sample"]), int(sidecar["end_sample"]), dtype=np.int64
        )
        _require(
            np.array_equal(sample_ids, expected_ids),
            f"non-contiguous sample_ids in {data_path}",
        )

        input_ids = data["input_token_ids"]
        label_ids = data["label_token_ids"]
        valid_mask = data["valid_mask"]
        for name, values, dtype in (
            ("input_token_ids", input_ids, np.int32),
            ("label_token_ids", label_ids, np.int32),
            ("valid_mask", valid_mask, np.uint8),
        ):
            _require(
                values.shape == dense_shape and values.dtype == dtype,
                f"{name} shape/dtype mismatch in {data_path}: "
                f"{values.shape}/{values.dtype}",
            )
        _require(
            bool(np.logical_or(valid_mask == 0, valid_mask == 1).all()),
            f"valid_mask contains values other than 0/1 in {data_path}",
        )
        valid_tokens = int(valid_mask.sum(dtype=np.int64))
        _require(
            valid_tokens == int(sidecar["valid_tokens"]),
            f"valid-token count mismatch in {data_path}: "
            f"manifest={sidecar['valid_tokens']}, actual={valid_tokens}",
        )
        token_hash = hashlib.sha256()
        token_hash.update(input_ids.tobytes())
        token_hash.update(label_ids.tobytes())
        token_hash.update(valid_mask.tobytes())
        _require(
            token_hash.hexdigest() == sidecar.get("logical_token_sha256"),
            f"logical token hash mismatch in {data_path}",
        )

        invalid_mask = valid_mask == 0
        for name in METRIC_NAMES:
            values = data[name]
            _require(
                values.shape == metric_shape and values.dtype == np.float32,
                f"{name} shape/dtype mismatch in {data_path}: "
                f"{values.shape}/{values.dtype}",
            )
            _require(np.isfinite(values).all(), f"non-finite {name} in {data_path}")
            minimum = float(values.min(initial=0.0))
            maximum = float(values.max(initial=0.0))
            if name in ("cosine", "feature_centered_cosine"):
                _require(
                    minimum >= -1.00001 and maximum <= 1.00001,
                    f"invalid {name} range in {data_path}: [{minimum}, {maximum}]",
                )
            if name in ("relative_l2", "symmetric_relative_l2", "delta_mse"):
                _require(
                    minimum >= 0.0,
                    f"negative {name} in {data_path}: min={minimum}",
                )
            if name == "symmetric_relative_l2":
                _require(
                    maximum <= 2.00001,
                    f"invalid {name} range in {data_path}: max={maximum}",
                )
            if invalid_mask.any():
                _require(
                    bool((values[invalid_mask] == 0.0).all()),
                    f"nonzero {name} stored for invalid tokens in {data_path}",
                )


def _validate_shards(
    worker_dir: Path,
    run: dict[str, Any],
    config_hash: str,
    *,
    deep: bool,
    require_complete: bool,
) -> dict[str, Any]:
    shard_dir = worker_dir / "token_metrics"
    _require(shard_dir.is_dir(), f"missing shard directory: {shard_dir}")
    temporaries = sorted(shard_dir.glob("*.inprogress*"))
    if require_complete:
        _require(not temporaries, f"unfinished shard files in {shard_dir}: {temporaries[:3]}")
    else:
        # A live extraction normally has one atomically-written NPZ in flight.
        # It is not part of the committed manifest chain and must be ignored by
        # partial validation; completed validation still rejects all remnants.
        _require(
            len(temporaries) <= 1,
            f"multiple unfinished shard files in live worker {shard_dir}: {temporaries[:3]}",
        )
    sidecars = sorted(shard_dir.glob("shard_*.json"))
    data_files = sorted(shard_dir.glob("shard_*.npz"))
    sidecar_data_names: set[str] = set()
    expected_start = int(run["partition_start_sample"])
    partition_end = expected_start + int(run["partition_samples"])
    valid_tokens = 0
    total_bytes = 0

    for index, sidecar_path in enumerate(sidecars):
        expected_sidecar_name = f"shard_{index:06d}.json"
        _require(
            sidecar_path.name == expected_sidecar_name,
            f"non-contiguous shard manifests in {shard_dir}: "
            f"expected={expected_sidecar_name}, got={sidecar_path.name}",
        )
        row = _load_json(sidecar_path)
        expected_data_name = f"shard_{index:06d}.npz"
        _require(row.get("schema_version") == SCHEMA_VERSION, f"schema mismatch: {sidecar_path}")
        _require(row.get("shard") == index, f"shard index mismatch: {sidecar_path}")
        _require(row.get("file") == expected_data_name, f"shard filename mismatch: {sidecar_path}")
        _require(
            row.get("config_sha256") == config_hash,
            f"config hash mismatch: {sidecar_path}",
        )
        samples = row.get("samples")
        start = row.get("start_sample")
        end = row.get("end_sample")
        _require(isinstance(samples, int) and samples > 0, f"invalid sample count: {sidecar_path}")
        _require(start == expected_start, f"sample gap/overlap at {sidecar_path}: expected {expected_start}, got {start}")
        _require(end == start + samples, f"end/sample mismatch: {sidecar_path}")
        _require(end <= partition_end, f"shard exceeds worker partition: {sidecar_path}")
        if end < partition_end:
            _require(
                samples == int(run["shard_samples"]),
                f"non-final shard has {samples} samples, expected {run['shard_samples']}: {sidecar_path}",
            )
        else:
            _require(
                samples <= int(run["shard_samples"]),
                f"final shard exceeds shard_samples: {sidecar_path}",
            )
        data_path = shard_dir / expected_data_name
        _require(data_path.is_file(), f"missing shard data: {data_path}")
        _require(
            isinstance(row.get("bytes"), int) and row["bytes"] == data_path.stat().st_size,
            f"file-size mismatch for {data_path}: manifest={row.get('bytes')}, "
            f"actual={data_path.stat().st_size}",
        )
        max_valid = samples * int(run["sequence_length"])
        _require(
            isinstance(row.get("valid_tokens"), int)
            and 0 <= row["valid_tokens"] <= max_valid,
            f"invalid valid_tokens in {sidecar_path}: {row.get('valid_tokens')}",
        )
        digest = row.get("logical_token_sha256")
        _require(
            isinstance(digest, str) and len(digest) == 64,
            f"invalid logical token hash in {sidecar_path}",
        )
        if deep:
            _validate_shard_payload(data_path, row, run)
        sidecar_data_names.add(expected_data_name)
        expected_start = end
        valid_tokens += int(row["valid_tokens"])
        total_bytes += int(row["bytes"])

    actual_data_names = {path.name for path in data_files}
    _require(
        actual_data_names == sidecar_data_names,
        f"orphan or missing shard files in {shard_dir}: "
        f"manifest_only={sorted(sidecar_data_names - actual_data_names)[:3]}, "
        f"data_only={sorted(actual_data_names - sidecar_data_names)[:3]}",
    )
    if require_complete:
        _require(
            expected_start == partition_end,
            f"incomplete worker partition in {worker_dir}: "
            f"reached={expected_start}, expected={partition_end}",
        )
    return {
        "shards": len(sidecars),
        "completed_samples": expected_start - int(run["partition_start_sample"]),
        "valid_tokens": valid_tokens,
        "bytes": total_bytes,
        "end_sample": expected_start,
    }


def _validate_reservoir_archive(
    path: Path,
    run: dict[str, Any],
    *,
    expected_tokens: int | None,
    expect_progress: bool,
) -> dict[str, Any]:
    _require(path.is_file(), f"missing reservoir archive: {path}")
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValidationError(f"could not open reservoir {path}: {error}") from error
    required = {
        "layer_numbers",
        "priority",
        "sample_ids",
        "positions",
        "token_ids",
        "reference",
        "current",
        "delta",
    }
    if expect_progress:
        required.add("completed_samples")
    with archive as data:
        _require(
            set(data.files) == required,
            f"reservoir array set mismatch in {path}: "
            f"expected={sorted(required)}, got={sorted(data.files)}",
        )
        layers = np.asarray(run["layers"], dtype=np.int16)
        layer_numbers = data["layer_numbers"]
        _require(
            layer_numbers.dtype == np.int16
            and np.array_equal(layer_numbers, layers),
            f"reservoir layer mismatch in {path}",
        )
        logical_hash = hashlib.sha256()
        logical_hash.update(layer_numbers.tobytes())
        priority = data["priority"]
        sample_ids = data["sample_ids"]
        positions = data["positions"]
        token_ids = data["token_ids"]
        count = int(priority.size)
        if expected_tokens is not None:
            _require(count == expected_tokens, f"reservoir token count mismatch in {path}: {count} != {expected_tokens}")
        for name, values, dtype in (
            ("priority", priority, np.uint64),
            ("sample_ids", sample_ids, np.int64),
            ("positions", positions, np.uint16),
            ("token_ids", token_ids, np.int32),
        ):
            _require(
                values.dtype == dtype and values.shape == (count,),
                f"reservoir {name} shape/dtype mismatch in {path}: "
                f"{values.shape}/{values.dtype}",
            )
            logical_hash.update(values.tobytes())
        if count:
            _require(bool((priority[1:] >= priority[:-1]).all()), f"unsorted reservoir priorities in {path}")
            start = int(run["partition_start_sample"])
            end = start + int(run["partition_samples"])
            _require(
                bool(((sample_ids >= start) & (sample_ids < end)).all()),
                f"reservoir sample outside worker partition in {path}",
            )
            _require(
                bool((positions < int(run["sequence_length"])).all()),
                f"reservoir position outside sequence in {path}",
            )
            occurrence = sample_ids.astype(np.uint64) * np.uint64(run["sequence_length"]) + positions.astype(np.uint64)
            _require(np.unique(occurrence).size == count, f"duplicate reservoir occurrences in {path}")
        hidden_shape = (count, len(run["layers"]), int(run["hidden_size"]))
        for name in ("reference", "current", "delta"):
            values = data[name]
            _require(
                values.dtype == np.float16 and values.shape == hidden_shape,
                f"reservoir {name} shape/dtype mismatch in {path}: "
                f"{values.shape}/{values.dtype}, expected={hidden_shape}/float16",
            )
            _require(np.isfinite(values).all(), f"non-finite reservoir {name} in {path}")
            logical_hash.update(values.tobytes())
        if expect_progress:
            completed = data["completed_samples"]
            _require(
                completed.shape == () and completed.dtype == np.int64,
                f"completed_samples scalar mismatch in {path}: "
                f"{completed.shape}/{completed.dtype}",
            )
        return {
            "tokens": count,
            "bytes": path.stat().st_size,
            "logical_sha256": logical_hash.hexdigest(),
        }


def _validate_final_state(
    worker_dir: Path,
    run: dict[str, Any],
    config_hash: str,
    shard_summary: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    progress = _load_json(worker_dir / "progress.json")
    expected_completed = int(shard_summary["completed_samples"])
    progress_checks = {
        "config_sha256": config_hash,
        "completed_samples": expected_completed,
        "partition_samples": int(run["partition_samples"]),
        "completed_valid_tokens": int(shard_summary["valid_tokens"]),
        "next_global_sample": int(run["partition_start_sample"]) + expected_completed,
        "next_shard": int(shard_summary["shards"]),
        "nonfinite_count": 0,
    }
    for key, expected in progress_checks.items():
        _require(
            progress.get(key) == expected,
            f"progress mismatch for {key} in {worker_dir}: "
            f"expected={expected!r}, got={progress.get(key)!r}",
        )

    reservoir_dir = worker_dir / "reservoir"
    progress_reservoir = reservoir_dir / "reservoir_progress.npz"
    if not require_complete:
        reservoir = _validate_reservoir_archive(
            progress_reservoir, run, expected_tokens=None, expect_progress=True
        )
        with np.load(progress_reservoir, allow_pickle=False) as data:
            _require(
                int(data["completed_samples"]) == expected_completed,
                f"reservoir/token progress mismatch in {worker_dir}",
            )
        return {"complete": False, "reservoir": reservoir}

    final = _load_json(worker_dir / "metadata.json")
    for key, expected in run.items():
        _require(
            final.get(key) == expected,
            f"final/run metadata mismatch for {key} in {worker_dir}",
        )
    final_checks = {
        "config_sha256": config_hash,
        "completed_samples": int(run["partition_samples"]),
        "completed_valid_tokens": int(shard_summary["valid_tokens"]),
        "shards": int(shard_summary["shards"]),
        "completed": True,
    }
    for key, expected in final_checks.items():
        _require(
            final.get(key) == expected,
            f"final metadata mismatch for {key} in {worker_dir}: "
            f"expected={expected!r}, got={final.get(key)!r}",
        )
    reservoir_tokens = final.get("reservoir_tokens")
    _require(
        isinstance(reservoir_tokens, int) and reservoir_tokens >= 0,
        f"invalid reservoir_tokens in {worker_dir}: {reservoir_tokens!r}",
    )
    final_reservoir = _validate_reservoir_archive(
        reservoir_dir / "hidden_reservoir.npz",
        run,
        expected_tokens=reservoir_tokens,
        expect_progress=False,
    )
    progress_reservoir_summary = _validate_reservoir_archive(
        progress_reservoir,
        run,
        expected_tokens=reservoir_tokens,
        expect_progress=True,
    )
    with np.load(progress_reservoir, allow_pickle=False) as data:
        _require(
            int(data["completed_samples"]) == int(run["partition_samples"]),
            f"reservoir final progress mismatch in {worker_dir}",
        )
    _require(
        final_reservoir["logical_sha256"]
        == progress_reservoir_summary["logical_sha256"],
        f"final/progress reservoir content mismatch in {worker_dir}",
    )
    return {
        "complete": True,
        "reservoir": final_reservoir,
        "reservoir_progress": progress_reservoir_summary,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    _require(root.is_dir(), f"missing extraction root: {root}")
    worker_dirs: dict[int, Path] = {}
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        match = WORKER_PATTERN.match(candidate.name)
        if match:
            worker_index = int(match.group(1))
            _require(worker_index not in worker_dirs, f"duplicate worker index: {worker_index}")
            worker_dirs[worker_index] = candidate
    expected_indexes = set(range(args.expected_workers))
    _require(
        set(worker_dirs) == expected_indexes,
        f"worker directory mismatch under {root}: "
        f"missing={sorted(expected_indexes - set(worker_dirs))}, "
        f"extra={sorted(set(worker_dirs) - expected_indexes)}",
    )

    run_rows: dict[int, dict[str, Any]] = {}
    config_hashes: dict[int, str] = {}
    for index in sorted(worker_dirs):
        row, digest = _validate_run_metadata(
            worker_dirs[index] / "run_metadata.json", index, args
        )
        run_rows[index] = row
        config_hashes[index] = digest

    # Except for explicit partition/worker identity, every worker must describe
    # the same checkpoints, dataset, representation, and extraction geometry.
    variable_keys = {"partition_start_sample", "partition_samples", "worker_index"}
    canonical = {key: value for key, value in run_rows[0].items() if key not in variable_keys}
    for index in range(1, args.expected_workers):
        comparable = {
            key: value for key, value in run_rows[index].items() if key not in variable_keys
        }
        _require(
            comparable == canonical,
            f"cross-worker run metadata mismatch between rank_000 and rank_{index:03d}",
        )

    ordered_ranges = sorted(
        (
            int(row["partition_start_sample"]),
            int(row["partition_start_sample"]) + int(row["partition_samples"]),
            index,
        )
        for index, row in run_rows.items()
    )
    cursor = 0
    for start, end, index in ordered_ranges:
        _require(
            start == cursor,
            f"worker partition gap/overlap before rank_{index:03d}: expected start={cursor}, got={start}",
        )
        cursor = end
    _require(
        cursor == args.expected_total_samples,
        f"worker partitions do not cover total: covered [0,{cursor}), "
        f"expected [0,{args.expected_total_samples})",
    )

    worker_reports: dict[str, Any] = {}
    total_shards = 0
    total_valid_tokens = 0
    total_bytes = 0
    for index in sorted(worker_dirs):
        worker_dir = worker_dirs[index]
        shard_summary = _validate_shards(
            worker_dir,
            run_rows[index],
            config_hashes[index],
            deep=args.deep,
            require_complete=not args.allow_incomplete,
        )
        final_summary = _validate_final_state(
            worker_dir,
            run_rows[index],
            config_hashes[index],
            shard_summary,
            require_complete=not args.allow_incomplete,
        )
        key = f"rank_{index:03d}"
        worker_reports[key] = {
            "partition": [
                int(run_rows[index]["partition_start_sample"]),
                int(run_rows[index]["partition_start_sample"])
                + int(run_rows[index]["partition_samples"]),
            ],
            **shard_summary,
            **final_summary,
        }
        total_shards += int(shard_summary["shards"])
        total_valid_tokens += int(shard_summary["valid_tokens"])
        total_bytes += int(shard_summary["bytes"])

    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "passed": True,
        "deep": bool(args.deep),
        "complete_required": not args.allow_incomplete,
        "workers": args.expected_workers,
        "total_samples": args.expected_total_samples,
        "total_dense_tokens": args.expected_total_samples * args.expected_sequence_length,
        "total_valid_tokens": total_valid_tokens,
        "total_shards": total_shards,
        "metric_shard_bytes": total_bytes,
        "worker_reports": worker_reports,
    }


def _parse_layers(value: str) -> list[int]:
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid layer list: {value}") from error
    if not layers or layers != sorted(set(layers)):
        raise argparse.ArgumentTypeError("layers must be a sorted, unique comma-separated list")
    return layers


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extraction root containing rank_000 ...")
    parser.add_argument("--output", help="Optional atomic JSON validation report")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Read every shard and validate arrays, finite/ranges, and token hashes",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate a partial extraction using reservoir_progress instead of requiring final files",
    )
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--expected-total-samples", type=int, default=4_147_200)
    parser.add_argument("--expected-sequence-length", type=int, default=512)
    parser.add_argument("--expected-hidden-size", type=int, default=1024)
    parser.add_argument("--expected-layers", type=_parse_layers, default=list(range(1, 10)))
    parser.add_argument("--expected-seed", type=int, default=1234)
    parser.add_argument("--expected-reference-step", type=int, default=600)
    parser.add_argument("--expected-current-step", type=int, default=1800)
    parser.add_argument("--expected-reference-load", default=DEFAULT_REFERENCE)
    parser.add_argument("--expected-current-load", default=DEFAULT_CURRENT)
    parser.add_argument("--expected-dataset", default=DEFAULT_DATASET)
    args = parser.parse_args(argv)
    if args.expected_workers <= 0:
        parser.error("--expected-workers must be positive")
    if args.expected_total_samples <= 0:
        parser.error("--expected-total-samples must be positive")
    if args.expected_sequence_length <= 0 or args.expected_hidden_size <= 0:
        parser.error("expected sequence/hidden sizes must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args)
    except ValidationError as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        return 1
    if args.output:
        _write_report(Path(args.output), report)
    summary = {
        key: report[key]
        for key in (
            "passed",
            "deep",
            "workers",
            "total_samples",
            "total_dense_tokens",
            "total_valid_tokens",
            "total_shards",
            "metric_shard_bytes",
        )
    }
    if args.output:
        summary["report"] = str(Path(args.output).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
