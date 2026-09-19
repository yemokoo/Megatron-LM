#!/usr/bin/env python3
"""Assert V3_EXPERT_DISPATCH=dense computes the same thing as the expert loop.

The loop path walks the batch-wide union of selected experts, gathering only
that expert's tokens; the dense path stacks every expert's LoRA pair and zeroes
unselected rank blocks with the routing weights.  With dropout 0 the two must
agree to floating-point summation order, including when the router emits
residual-expert ids (>= num_experts) and padded tokens, which both paths must
drop.

CPU only, no checkpoint needed.  usage: check_dispatch_equivalence.py [-v]
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "implementations/llmcl_benchmark"
sys.path.insert(0, str(ROOT))

from model import Ours_LoRA_MoE_V3 as V3  # noqa: E402


def make_context(tokens, num_experts, top_k, n_residual, mask_some, seed):
    generator = torch.Generator().manual_seed(seed)
    pool = num_experts + n_residual
    indices = torch.stack([
        torch.randperm(pool, generator=generator)[:top_k]
        for _ in range(tokens)])
    logits = torch.randn(tokens, top_k, generator=generator)
    weights = torch.softmax(logits, dim=-1)
    valid = None
    if mask_some:
        valid = torch.ones(tokens, dtype=torch.bool)
        valid[::7] = False
    return V3.RoutingContext(
        expert_indices=indices, expert_weights=weights,
        valid_token_mask=valid, num_experts=num_experts)


def both_modes(module, inputs, context_factory):
    outputs = {}
    for mode in ("loop", "dense"):
        V3.set_expert_dispatch_mode(mode)
        context = context_factory()
        module.set_routing_context(context)
        with torch.no_grad():
            outputs[mode] = module(inputs).clone()
        module.set_routing_context(None)
    V3.set_expert_dispatch_mode("loop")
    return outputs["loop"], outputs["dense"]


def check(label, loop, dense, tol):
    diff = (loop - dense).abs().max().item()
    scale = loop.abs().max().item() or 1.0
    ok = diff <= tol * scale
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: max|loop-dense| = {diff:.3e} "
          f"(scale {scale:.3e}, tol {tol:.1e} relative)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    tokens, hidden, inter = 96, 64, 128
    failures = 0

    for experts_per_task, rank, top_k in ((1, 64, 1), (2, 32, 2),
                                          (4, 16, 4), (8, 8, 8)):
        rounds = 3
        num_experts = experts_per_task * rounds
        alpha = 2 * rank
        print(f"E={experts_per_task} rank={rank} top_k={top_k} "
              f"num_experts={num_experts} (rank sum/task "
              f"{experts_per_task * rank})")
        for n_residual, mask_some in ((0, False), (experts_per_task, True)):
            tag = (f"residual={n_residual} padded_tokens={mask_some}")
            inputs = torch.randn(1, tokens, hidden)

            attention = V3.RoutedLoRALinear(
                torch.nn.Linear(hidden, hidden, bias=False), rank, alpha, 0.0)
            attention.add_experts(num_experts)
            for pair in attention.experts:          # B starts at zero
                torch.nn.init.normal_(pair.B, std=0.02)

            base_mlp = torch.nn.Module()
            base_mlp.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
            base_mlp.up_proj = torch.nn.Linear(hidden, inter, bias=False)
            base_mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False)
            mlp = V3.RoutedLoRAMLP(base_mlp, rank, alpha, 0.0)
            mlp.add_experts(num_experts)
            for expert in mlp.experts:
                for key in ("gate", "up", "down"):
                    torch.nn.init.normal_(expert[key].B, std=0.02)

            factory = lambda: make_context(   # noqa: E731
                tokens, num_experts, top_k, n_residual, mask_some, seed=7)
            loop, dense = both_modes(attention, inputs, factory)
            failures += not check(f"attention  {tag}", loop, dense, 2e-6)
            loop, dense = both_modes(mlp, inputs, factory)
            failures += not check(f"ffn        {tag}", loop, dense, 2e-6)

    print()
    if failures:
        print(f"{failures} FAILURE(S) -- dense dispatch is NOT equivalent")
        return 1
    print("ALL EQUIVALENT (dropout 0; dense differs only in dropout RNG "
          "when lora_moe_dropout > 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
