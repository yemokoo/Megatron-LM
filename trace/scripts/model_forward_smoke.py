#!/usr/bin/env python3
"""Load a local paper backbone on one GPU and run a one-token forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama31": ROOT / "models" / "Llama-3.1-8B-Instruct",
    "qwen25_7b": ROOT / "models" / "Qwen2.5-7B-Instruct",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    model_path = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()

    encoded = tokenizer("Hello", return_tensors="pt")
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits

    print(
        json.dumps(
            {
                "ok": True,
                "model": args.model,
                "path": str(model_path),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
                "input_shape": list(encoded["input_ids"].shape),
                "logits_shape": list(logits.shape),
                "finite": bool(torch.isfinite(logits).all().item()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
