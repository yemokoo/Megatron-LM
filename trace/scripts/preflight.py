#!/usr/bin/env python3
"""Offline, model-free preflight for the SLoRA reproduction scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_CONFIG = ROOT / "config" / "trace_experiments.json"
MODEL_CONFIG = ROOT / "config" / "models.json"
MODEL_CONFIG_FILES = ("config.json",)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_schema(value) -> dict:
    if isinstance(value, list):
        first = value[0] if value else None
        signatures = set()
        for item in value:
            if isinstance(item, dict):
                signatures.add(
                    tuple((key, type(item[key]).__name__) for key in sorted(item))
                )
            else:
                signatures.add((("<item>", type(item).__name__),))
        return {
            "container": "list",
            "item_keys": sorted(first) if isinstance(first, dict) else None,
            "item_types": (
                {key: type(first[key]).__name__ for key in sorted(first)}
                if isinstance(first, dict)
                else None
            ),
            "uniform_item_schema": len(signatures) <= 1,
            "schema_variants": len(signatures),
        }
    return {"container": type(value).__name__}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def audit_dataset(config: dict, required: bool) -> tuple[list[dict], list[str]]:
    tasks = config["paper"]["task_order"]
    root = project_path(
        os.environ.get("TRACE_DATA_ROOT", config["paths"]["paper_trace_data"])
    )
    rows: list[dict] = []
    errors: list[str] = []
    expected = config["paper"]["train_samples_per_trace_task"]
    for task in tasks:
        for split in ("train", "eval", "test"):
            path = root / task / f"{split}.json"
            if not path.is_file():
                if required:
                    errors.append(f"missing dataset file: {path}")
                continue
            try:
                value = load_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSON {path}: {exc}")
                continue
            count = len(value) if isinstance(value, list) else None
            schema = json_schema(value)
            if count is None:
                errors.append(f"dataset root is not a list: {path}")
            if not schema.get("uniform_item_schema", True):
                errors.append(f"non-uniform item schema: {path}")
            if split == "train" and count != expected:
                errors.append(f"{task}/train expected {expected}, got {count}")
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "path": str(path),
                    "records": count,
                    "schema": schema,
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    return rows, errors


def audit_models(registry: dict) -> list[dict]:
    rows = []
    for key, value in registry["models"].items():
        common_overrides = {
            "llama31_8b_instruct": "SLORA_LLAMA31_PATH",
            "llama31_8b": "SLORA_LLAMA31_PATH",
            "qwen25_7b_instruct": "SLORA_QWEN25_7B_PATH",
        }
        configured = os.environ.get(
            common_overrides.get(key, f"SLORA_MODEL_{key.upper()}"),
            value["planned_path"],
        )
        path = project_path(configured)
        configs = [str(path / name) for name in MODEL_CONFIG_FILES if (path / name).is_file()]
        tokenizers = [str(path / name) for name in TOKENIZER_FILES if (path / name).is_file()]
        weights = sorted(
            {str(item) for pattern in WEIGHT_PATTERNS for item in path.glob(pattern)}
        ) if path.is_dir() else []
        index_paths = sorted(path.glob("*.index.json")) if path.is_dir() else []
        indexes = [str(item) for item in index_paths]
        missing_index_shards = []
        invalid_indexes = []
        for index_path in index_paths:
            try:
                index = load_json(index_path)
                referenced = sorted(set(index.get("weight_map", {}).values()))
                missing_index_shards.extend(
                    str(path / shard)
                    for shard in referenced
                    if not (path / shard).is_file()
                )
                if not referenced:
                    invalid_indexes.append(f"{index_path}: no weight_map entries")
            except (OSError, json.JSONDecodeError) as exc:
                invalid_indexes.append(f"{index_path}: {exc}")
        usable_index = bool(index_paths and not missing_index_shards and not invalid_indexes)
        weight_payload_ready = usable_index if index_paths else bool(weights)
        ready = bool(configs and tokenizers and weight_payload_ready)
        rows.append(
            {
                "key": key,
                "hf_id": value["hf_id"],
                "paper_scope": value["paper_scope"],
                "resolved_path": str(path),
                "directory_exists": path.is_dir(),
                "config_files": configs,
                "tokenizer_files": tokenizers,
                "weight_files": weights,
                "weight_indexes": indexes,
                "missing_index_shards": missing_index_shards,
                "invalid_indexes": invalid_indexes,
                "ready": ready,
                "expected_during_scaffold": "missing",
            }
        )
    return rows


def git_metadata(path: Path) -> dict:
    # Vendored snapshots intentionally have no nested .git directory. Without
    # this guard Git walks upward and incorrectly reports the parent project
    # worktree as dirtiness of every pristine snapshot.
    if not (path / ".git").exists():
        return {
            "path": str(path),
            "snapshot": "vendored_without_git_metadata",
            "provenance": str(ROOT / "manifests" / "source_provenance.json"),
            "dirty": False,
        }

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "path": str(path),
            "remote": run("remote", "get-url", "origin"),
            "sha": run("rev-parse", "HEAD"),
            "commit_date": run("show", "-s", "--format=%cI", "HEAD"),
            "branch": run("branch", "--show-current") or "(detached)",
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"path": str(path), "error": "not a readable git repository"}


def batch_table(config: dict) -> list[dict]:
    samples = config["paper"]["train_samples_per_trace_task"]
    rows = []
    for name, profile in config["batch_profiles"].items():
        micro = profile["micro_batch_per_device"]
        world = profile["world_size"]
        accumulation = profile["gradient_accumulation"]
        effective = micro * world * accumulation
        batches = math.ceil(samples / (micro * world))
        steps = math.ceil(batches / accumulation)
        rows.append(
            {
                "profile": name,
                "status": profile["status"],
                "samples": samples,
                "micro_batch_per_device": micro,
                "world_size": world,
                "gradient_accumulation": accumulation,
                "effective_global_batch": effective,
                "dataloader_batches_per_epoch": batches,
                "optimizer_steps_per_epoch": steps,
                "logging_steps": profile.get("logging_steps"),
                "note": "logging_steps is not a batch-size field",
            }
        )
    return rows


def hardware() -> dict:
    gpu = {"available": False, "models": [], "driver": None}
    if shutil.which("nvidia-smi"):
        command = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
        try:
            lines = subprocess.check_output(command, text=True).strip().splitlines()
            gpu = {"available": True, "count": len(lines), "devices": lines}
        except subprocess.CalledProcessError as exc:
            gpu = {"available": False, "error": str(exc)}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("scaffold", "full"), default="scaffold",
        help="scaffold expects models to be absent; full requires every registry entry",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "manifests" / "preflight.json"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="registry keys required in full mode (default: every paper backbone)",
    )
    args = parser.parse_args()
    trace = load_json(TRACE_CONFIG)
    models = load_json(MODEL_CONFIG)
    dataset_rows, errors = audit_dataset(trace, required=args.mode == "full")
    model_rows = audit_models(models)
    if args.mode == "full":
        known_keys = {row["key"] for row in model_rows}
        required_keys = set(args.models or [
            key for key, value in models["models"].items()
            if "legacy_olora_only" not in value["paper_scope"]
        ])
        unknown = required_keys - known_keys
        if unknown:
            errors.append(f"unknown model registry keys: {sorted(unknown)}")
        errors.extend(
            f"model not ready: {row['key']} at {row['resolved_path']}"
            for row in model_rows
            if row["key"] in required_keys and not row["ready"]
        )
    repos = [
        git_metadata(ROOT / "upstream" / name)
        for name in ("SLoRA", "O-LoRA", "TRACE")
    ]
    errors.extend(
        f"pristine snapshot is dirty: {row['path']}" for row in repos if row.get("dirty")
    )
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "required_models": sorted(required_keys) if args.mode == "full" else [],
        "ok": not errors,
        "policy": models["download_policy"],
        "hardware": hardware(),
        "repositories": repos,
        "models": model_rows,
        "datasets": dataset_rows,
        "batch_calculations": batch_table(trace),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    train_rows = [row for row in dataset_rows if row["split"] == "train"]
    print(f"preflight mode={args.mode}: {'PASS' if report['ok'] else 'FAIL'}")
    print(f"verified dataset files={len(dataset_rows)}, train files={len(train_rows)}")
    counts = [row["records"] for row in train_rows]
    print(f"TRACE paper dataset: tasks={len(counts)}, counts={sorted(set(counts))}")
    ready = sum(row["ready"] for row in model_rows)
    expectation = (
        "missing models are expected" if args.mode == "scaffold"
        else "required models must be ready"
    )
    print(f"models ready={ready}/{len(model_rows)} ({expectation})")
    for row in report["batch_calculations"]:
        print(
            f"{row['profile']}: effective_batch={row['effective_global_batch']}, "
            f"steps/epoch={row['optimizer_steps_per_epoch']}, "
            f"logging_steps={row['logging_steps']}"
        )
    print(f"manifest={args.output}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
