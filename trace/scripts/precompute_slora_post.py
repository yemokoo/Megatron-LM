#!/usr/bin/env python3
"""Precompute one SLoRA post-denoised adapter without evaluating a task."""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from src.model.builder import denoising


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.is_file():
        print(f"[SKIP] {output}", flush=True)
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    state = {
        key: value.to(model.device)
        for key, value in load_file(args.adapter).items()
    }
    result = denoising(model, state, mode="max")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    save_file(result, temporary)
    os.replace(temporary, output)
    print(f"[OK] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
