#!/usr/bin/env python
"""Short real-checkpoint smoke test for Track-1 growing LoRA-MoE.

This is not a training run. It loads a small Qwen checkpoint, performs one
LM-only backward, audits phase masks/growth, and round-trips a delta checkpoint
inside a temporary directory.
"""
import argparse
import os
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from model.Ours_LoRA_MoE import (
    add_experts_to_all_layers,
    attach_lora_moe,
    freeze_lora_moe_experts,
    freeze_lora_moe_routers,
    load_lora_moe_checkpoint,
    save_lora_moe_meta,
    set_router_token_mask,
)
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="/home/work/Agent_HJ/00_models/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def trainable_names(model):
    return {name for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def assert_phase1_mask(model, current_index):
    names = trainable_names(model)
    if not names:
        raise AssertionError("phase 1 has no trainable parameters")
    allowed_expert = f".mlp.experts.{current_index}."
    bad = [name for name in names
           if allowed_expert not in name and ".mlp.router.weight" not in name]
    if bad:
        raise AssertionError(f"phase-1 trainable leakage: {bad[:5]}")
    if not any(allowed_expert in name for name in names):
        raise AssertionError("current expert is not trainable")
    if not any(".mlp.router.weight" in name for name in names):
        raise AssertionError("router is not trainable")


def assert_phase2_mask(model):
    names = trainable_names(model)
    if not names or any(".mlp.router.weight" not in name for name in names):
        raise AssertionError(f"phase-2 mask is not router-only: {sorted(names)[:5]}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    model = create_hf_model(
        AutoModelForCausalLM, args.model_name_or_path, tokenizer,
        disable_dropout=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)
    attach_lora_moe(
        model, r=2, alpha=4, top_k=1, aux_loss_coeff=0.01,
        z_loss_coeff=0.001, routing_weight_mode="full_softmax")

    # Task 0: two experts make the full softmax non-degenerate for the top-1
    # LM-gradient audit. Only the second one represents the current task.
    add_experts_to_all_layers(model, 2)
    freeze_lora_moe_experts(model, trainable_expert_indices={1})
    freeze_lora_moe_routers(model, trainable=True)
    assert_phase1_mask(model, current_index=1)

    # Production LoRA B matrices start at zero, so the very first LM backward has
    # no router signal until B changes. Seed a tiny nonzero delta solely to audit
    # the top-1 differentiable routing path directly, without aux/z loss.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".mlp.experts." in name and name.endswith(".B"):
                parameter.normal_(mean=0.0, std=1e-3)

    encoded = tokenizer(
        ["short example", "a somewhat longer smoke-test example"],
        padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    model.train()
    set_router_token_mask(model, encoded["attention_mask"])
    try:
        loss = model(**encoded, labels=labels, use_cache=False).loss
        loss.backward()
    finally:
        set_router_token_mask(model, None)

    router_grad = 0.0
    forbidden_grad = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not parameter.requires_grad:
                forbidden_grad.append(name)
            if ".mlp.router.weight" in name:
                router_grad += parameter.grad.float().norm().item()
    if forbidden_grad:
        raise AssertionError(f"frozen parameters received gradients: {forbidden_grad[:5]}")
    if router_grad <= 0.0:
        raise AssertionError("top-1 LM-only router gradient is zero")

    # Task 1 growth and phase masks.
    add_experts_to_all_layers(model, 1)
    counts = {layer.mlp.num_experts for layer in model.model.layers}
    if counts != {3}:
        raise AssertionError(f"expert growth differs across layers: {counts}")
    freeze_lora_moe_experts(model, trainable_expert_indices={2})
    freeze_lora_moe_routers(model, trainable=True)
    assert_phase1_mask(model, current_index=2)
    freeze_lora_moe_experts(model, trainable_expert_indices=None)
    freeze_lora_moe_routers(model, trainable=True)
    assert_phase2_mask(model)

    delta = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".mlp.experts." in name or ".mlp.router." in name
    }
    with tempfile.TemporaryDirectory(prefix="lora_moe_smoke_") as checkpoint_dir:
        torch.save(delta, os.path.join(checkpoint_dir, "pytorch_model.bin"))
        save_lora_moe_meta(model, checkpoint_dir)
        del model
        torch.cuda.empty_cache()
        loaded, meta = load_lora_moe_checkpoint(
            checkpoint_dir, tokenizer,
            base_model_name_or_path=args.model_name_or_path,
            device=device, dtype=torch.bfloat16)
        if meta["routing_weight_mode"] != "full_softmax":
            raise AssertionError(f"routing metadata mismatch: {meta}")
        loaded_state = loaded.state_dict()
        for name, expected in delta.items():
            if not torch.equal(loaded_state[name].cpu(), expected):
                raise AssertionError(f"delta round-trip mismatch: {name}")

    print(
        "LORA_MOE_REAL_SMOKE=PASS "
        f"experts_per_layer=3 top1_lm_router_grad={router_grad:.6g} "
        "phase1_mask=PASS phase2_mask=PASS delta_roundtrip=PASS")


if __name__ == "__main__":
    main()
