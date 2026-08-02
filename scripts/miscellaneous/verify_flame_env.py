#!/usr/bin/env python3
"""Validate the isolated FLAME/Megatron runtime before launching training."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "torchaudio": "2.4.1",
    "transformers": "4.33.1",
    "tokenizers": "0.13.3",
    "datasets": "2.20.0",
    "accelerate": "0.33.0",
    "wandb": "0.28.1",
    "tensorboard": "2.9.0",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "sentencepiece": "0.1.96",
    "safetensors": "0.4.3",
    "grouped-gemm": "1.1.4",
    "flash-attn": "2.4.2",
    "apex": "0.1",
    "transformer-engine": "1.11.0",
}
IMPORT_NAMES = {
    "grouped-gemm": "grouped_gemm",
    "flash-attn": "flash_attn",
    "transformer-engine": "transformer_engine.pytorch",
}
EXPECTED_SOURCE_COMMITS = {
    "Megatron-LM": "22af3b7399ca3f7ac4e2b11ca0fae2a7eb4a4ddc",
    "apex": "c02c6c891eedfabf91f0de8127d7636d4292356d",
    "TransformerEngine": "fc034785f5e3a5bc5600a88766d9a1d75137ce77",
}
GROUPED_GEMM_COMMIT = "172fada89fa7364fe5d026b3a0dfab58b591ffdd"


def command_version(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--allow-system-python",
        action="store_true",
        help="Audit an old system/NGC runtime instead of enforcing Conda isolation.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    versions: dict[str, str] = {}
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python 3.10 required, got {sys.version.split()[0]}")
    if not os.environ.get("CONDA_PREFIX") and not args.allow_system_python:
        errors.append("CONDA_PREFIX is unset; activate the dedicated FLAME environment")
    if not os.environ.get("PYTHONNOUSERSITE"):
        warnings.append("PYTHONNOUSERSITE is unset; user-site packages can shadow the environment")

    for distribution, expected in EXPECTED.items():
        try:
            actual = metadata.version(distribution)
            importlib.import_module(IMPORT_NAMES.get(distribution, distribution.replace("-", "_")))
        except Exception as exc:
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")
            continue
        versions[distribution] = actual
        if not actual.startswith(expected):
            errors.append(f"{distribution} expected {expected}, got {actual}")

    source_commits: dict[str, str] = {}
    for directory, expected in EXPECTED_SOURCE_COMMITS.items():
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(ROOT / directory), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            source_commits[directory] = actual
            if actual != expected:
                errors.append(f"{directory} expected commit {expected}, got {actual}")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"cannot inspect {directory} source commit: {exc}")

    try:
        direct_url = json.loads(metadata.distribution("grouped-gemm").read_text("direct_url.json") or "{}")
        actual = str(direct_url.get("vcs_info", {}).get("commit_id", ""))
        source_commits["grouped-gemm"] = actual
        if actual != GROUPED_GEMM_COMMIT:
            errors.append(f"grouped-gemm expected commit {GROUPED_GEMM_COMMIT}, got {actual or 'unknown'}")
    except Exception as exc:
        errors.append(f"cannot inspect grouped-gemm source commit: {type(exc).__name__}: {exc}")

    sys.path.insert(0, str(ROOT / "Megatron-LM"))
    try:
        importlib.import_module("megatron.core")
        versions["megatron.core"] = "import-ok"
    except Exception as exc:
        errors.append(f"megatron.core: {type(exc).__name__}: {exc}")

    gpu: dict[str, object] = {}
    try:
        import torch

        gpu = {
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            gpu["device_0"] = torch.cuda.get_device_name(0)
            gpu["capability_0"] = list(torch.cuda.get_device_capability(0))
        elif args.require_gpu:
            errors.append("CUDA GPU is not available to PyTorch")
    except Exception as exc:
        errors.append(f"torch CUDA query failed: {type(exc).__name__}: {exc}")

    payload = {
        "ok": not errors,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "versions": versions,
        "source_commits": source_commits,
        "gpu": gpu,
        "tools": {
            "nvcc": command_version(["nvcc", "--version"]),
            "gcc": command_version(["gcc", "--version"]),
        },
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
