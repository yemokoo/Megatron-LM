#!/usr/bin/env python3
"""Run tiny forward/backward CUDA-extension checks without launching training."""

import json

import torch
from flash_attn import flash_attn_func
from grouped_gemm import ops as grouped_gemm_ops
import transformer_engine.pytorch as te


def main() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (9, 0)
    device = torch.device("cuda:0")

    a = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
    matmul = a @ a

    qkv = torch.randn(
        2, 32, 3, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    flash = flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
    flash.float().sum().backward()

    te_input = torch.randn(
        16, 64, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    te_output = te.Linear(64, 32, device=device, params_dtype=torch.bfloat16)(te_input)
    te_output.float().sum().backward()

    grouped_a = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    grouped_b = torch.randn(2, 16, 12, device=device, dtype=torch.bfloat16)
    grouped = grouped_gemm_ops.gmm(
        grouped_a, grouped_b, torch.tensor([3, 5], device="cpu")
    )

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "ok": True,
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "bf16_matmul": list(matmul.shape),
                "flash_attn_forward_backward": list(flash.shape),
                "transformer_engine_forward_backward": list(te_output.shape),
                "grouped_gemm": list(grouped.shape),
                "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
