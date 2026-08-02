#!/usr/bin/env python3
"""Validate the isolated TRACE/SLoRA runtime and local model metadata."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "llama31": ROOT / "models" / "Llama-3.1-8B-Instruct",
    "qwen25_7b": ROOT / "models" / "Qwen2.5-7B-Instruct",
}
EXPECTED = {
    "torch": "2.4.1",
    "transformers": "4.51.3",
    "tokenizers": "0.21.4",
    "peft": "0.12.0",
    "trl": "0.16.1",
    "datasets": "3.2.0",
    "accelerate": "1.2.1",
    "deepspeed": "0.16.9",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.1",
    "evaluate": "0.4.3",
    "huggingface-hub": "0.36.2",
    "wandb": "0.28.1",
}
IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "huggingface-hub": "huggingface_hub",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PATHS))
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    versions: dict[str, str] = {}
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python 3.10 required, got {sys.version.split()[0]}")
    if sys.prefix == sys.base_prefix:
        errors.append("TRACE must run in trace/.venv-runtime (no virtualenv detected)")

    for distribution, expected in EXPECTED.items():
        try:
            actual = metadata.version(distribution)
            importlib.import_module(IMPORT_NAMES.get(distribution, distribution.replace("-", "_")))
        except Exception as exc:
            errors.append(f"cannot load {distribution}: {type(exc).__name__}: {exc}")
            continue
        versions[distribution] = actual
        mismatch = (
            not actual.startswith(expected)
            if distribution == "torch"
            else actual != expected
        )
        if mismatch:
            errors.append(f"{distribution} expected {expected}, got {actual}")

    model_metadata: dict[str, object] | None = None
    if args.model:
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

    gpu_rows: list[dict[str, int]] = []
    if not args.skip_gpu:
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
    else:
        warnings.append("GPU check skipped")

    payload = {
        "ok": not errors,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": versions,
        "model": model_metadata,
        "gpus": gpu_rows,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
