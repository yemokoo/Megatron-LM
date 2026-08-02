#!/usr/bin/env python3
"""Validate the installed runtime and local model metadata before GPU work."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "llama31": ROOT / "models" / "Llama-3.1-8B-Instruct",
    "qwen25_7b": ROOT / "models" / "Qwen2.5-7B-Instruct",
}
EXPECTED = {
    "torch": "2.4.1",
    "transformers": "4.51.3",
    "peft": "0.12.0",
    "trl": "0.16.1",
    "datasets": "3.2.0",
    "accelerate": "1.2.1",
    "deepspeed": "0.16.9",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_PATHS))
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    errors: list[str] = []
    versions = {}
    for package, expected in EXPECTED.items():
        try:
            module = importlib.import_module(package)
        except Exception as exc:
            errors.append(f"cannot import {package}: {type(exc).__name__}: {exc}")
            continue
        actual = str(getattr(module, "__version__", "unknown"))
        versions[package] = actual
        if package == "torch":
            if not actual.startswith(expected):
                errors.append(f"{package} expected {expected}, got {actual}")
        elif actual != expected:
            errors.append(f"{package} expected {expected}, got {actual}")

    model_path = MODEL_PATHS[args.model]
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model_metadata = {
            "path": str(model_path),
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    except Exception as exc:
        model_metadata = {"path": str(model_path)}
        errors.append(f"cannot load local model config/tokenizer: {type(exc).__name__}: {exc}")

    gpu_rows = []
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in output.strip().splitlines():
            index, free, total = [part.strip() for part in line.split(",")]
            gpu_rows.append(
                {"index": int(index), "free_mib": int(free), "total_mib": int(total)}
            )
        if len(gpu_rows) < args.world_size:
            errors.append(
                f"world_size={args.world_size}, but only {len(gpu_rows)} GPUs are visible"
            )
    except Exception as exc:
        errors.append(f"cannot query GPUs: {type(exc).__name__}: {exc}")

    payload = {
        "ok": not errors,
        "versions": versions,
        "model": model_metadata,
        "gpus": gpu_rows,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
