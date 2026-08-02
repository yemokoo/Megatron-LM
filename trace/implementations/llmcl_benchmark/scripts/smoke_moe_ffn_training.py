#!/usr/bin/env python
"""Short real-OLMoE structural smoke test for Track 2 (not a training job)."""
import argparse
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from model.Ours_MoE_FFN import (
    MOE_FFN_DELTA_FORMAT,
    MOE_FFN_SAVE_KEY_SUBSTRINGS,
    _moe_layers,
    add_experts_to_all_layers,
    attach_growing_moe,
    collect_moe_losses,
    freeze_moe_experts,
    freeze_moe_routers,
    load_moe_ffn_checkpoint,
    save_moe_ffn_meta,
    set_phase1_forced_experts,
    set_router_token_mask,
)
from utils.model.model_utils import create_hf_model, resolve_attention_implementation
from utils.utils import get_optimizer_grouped_parameters, load_hf_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="auto")
    return parser.parse_args()


def optimizer_parameter_ids(optimizer):
    return {id(parameter) for group in optimizer.param_groups
            for parameter in group["params"]}


def assert_optimizer_exact(model, optimizer):
    expected = {id(parameter) for parameter in model.parameters()
                if parameter.requires_grad}
    actual = optimizer_parameter_ids(optimizer)
    if actual != expected:
        raise AssertionError(
            f"optimizer parameter mismatch: missing={len(expected-actual)}, "
            f"extra={len(actual-expected)}")


def assert_phase1(model, current_index):
    for name, parameter in model.named_parameters():
        expected = (
            f".mlp.new_experts.{current_index}." in name or
            f".mlp.new_gates.{current_index}." in name)
        if parameter.requires_grad != expected:
            raise AssertionError(
                f"phase-1 requires_grad mismatch: {name}="
                f"{parameter.requires_grad}, expected={expected}")


def assert_phase2(model, num_new):
    for name, parameter in model.named_parameters():
        expected = (
            ".mlp.base_block.gate." in name or
            any(f".mlp.new_gates.{index}." in name
                for index in range(num_new)))
        if parameter.requires_grad != expected:
            raise AssertionError(
                f"phase-2 requires_grad mismatch: {name}="
                f"{parameter.requires_grad}, expected={expected}")


def audit_forced_routing(layer, valid_mask):
    # Natural selections deliberately contain no appended expert.
    selected = torch.arange(layer.top_k, device=valid_mask.device).repeat(
        valid_mask.numel(), 1)
    forced = layer._force_new_experts_into_topk(selected, valid_mask)
    new_id = layer.num_base
    if not (forced[valid_mask] == new_id).any(dim=-1).all():
        raise AssertionError("forced routing did not reach the new expert")
    if not torch.equal(forced[~valid_mask], selected[~valid_mask]):
        raise AssertionError("forced routing modified padding positions")


def audit_padding_router_loss(layer, valid_mask):
    torch.manual_seed(1234)
    logits = torch.randn(
        valid_mask.numel(), layer.num_experts, device=valid_mask.device,
        dtype=torch.float32, requires_grad=True)
    probs = F.softmax(logits, dim=-1)
    selected = probs.topk(layer.top_k, dim=-1).indices
    masked = layer._router_loss(logits, probs, selected, valid_mask)
    expected = layer._router_loss(
        logits[valid_mask], probs[valid_mask], selected[valid_mask])
    if not torch.equal(masked, expected):
        raise AssertionError("aux/z padding mask changed the valid-token loss")


def main():
    args = parse_args()
    device = torch.device(args.device)
    resolved_attn = resolve_attention_implementation(args.attn_implementation)
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    model = create_hf_model(
        AutoModelForCausalLM, args.model_name_or_path, tokenizer,
        disable_dropout=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation=resolved_attn,
        forbid_vocab_growth=True).to(device)
    actual_attn = getattr(model.config, "_attn_implementation", None)
    if actual_attn != resolved_attn:
        raise AssertionError(
            f"attention backend mismatch: resolved={resolved_attn}, "
            f"model={actual_attn}")

    attach_growing_moe(model, aux_loss_coeff=0.01, z_loss_coeff=0.001)
    add_experts_to_all_layers(model, 1)
    layers = _moe_layers(model)
    if not layers or {(layer.num_base, layer.num_new) for layer in layers} != {(64, 1)}:
        raise AssertionError("task 0 did not grow every layer from 64 to 65 experts")

    freeze_moe_experts(model, trainable_new_indices={0})
    freeze_moe_routers(
        model, trainable_new_indices={0}, trainable_original=False)
    set_phase1_forced_experts(model, {0})
    assert_phase1(model, current_index=0)
    optimizer0 = torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, 0.0), lr=1e-4)
    assert_optimizer_exact(model, optimizer0)

    encoded = tokenizer(
        ["short", "a somewhat longer OLMoE smoke example"],
        padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    flat_mask = encoded["attention_mask"].reshape(-1).bool()
    audit_forced_routing(layers[0], flat_mask)
    audit_padding_router_loss(layers[0], flat_mask)

    model.train()
    set_router_token_mask(model, encoded["attention_mask"])
    try:
        output = model(**encoded, labels=labels, use_cache=False)
        moe_loss = collect_moe_losses(model)
        (output.loss + moe_loss).backward()
    finally:
        set_router_token_mask(model, None)
    forbidden = [name for name, parameter in model.named_parameters()
                 if parameter.grad is not None and not parameter.requires_grad]
    if forbidden:
        raise AssertionError(f"frozen parameters received gradients: {forbidden[:5]}")
    down_grad = sum(
        parameter.grad.float().norm().item()
        for name, parameter in model.named_parameters()
        if ".mlp.new_experts.0.down_proj.weight" in name and
        parameter.grad is not None)
    if down_grad <= 0.0:
        raise AssertionError("forced new expert received no down_proj gradient")

    # Task 1: append exactly one more expert/row, then rebuild optimizer from the
    # new requires_grad mask. The old optimizer must not be reused.
    del optimizer0
    add_experts_to_all_layers(model, 1)
    if {(layer.num_base, layer.num_new) for layer in layers} != {(64, 2)}:
        raise AssertionError("task 1 did not grow every layer from 65 to 66 experts")
    freeze_moe_experts(model, trainable_new_indices={1})
    freeze_moe_routers(
        model, trainable_new_indices={1}, trainable_original=False)
    set_phase1_forced_experts(model, None)
    assert_phase1(model, current_index=1)
    optimizer1 = torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, 0.0), lr=1e-4)
    assert_optimizer_exact(model, optimizer1)

    freeze_moe_experts(model, trainable_new_indices=None)
    freeze_moe_routers(
        model, trainable_new_indices={0, 1}, trainable_original=True)
    assert_phase2(model, num_new=2)
    optimizer2 = torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, 0.0), lr=1e-4)
    assert_optimizer_exact(model, optimizer2)
    del optimizer1, optimizer2

    delta = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if any(tag in name for tag in MOE_FFN_SAVE_KEY_SUBSTRINGS)
    }
    with tempfile.TemporaryDirectory(prefix="moe_ffn_smoke_") as checkpoint_dir:
        torch.save(delta, os.path.join(checkpoint_dir, "pytorch_model.bin"))
        save_moe_ffn_meta(
            model, checkpoint_dir,
            base_model_name_or_path=args.model_name_or_path,
            experts_per_task=1, checkpoint_format=MOE_FFN_DELTA_FORMAT,
            phase1_new_expert_routing="force")
        del model
        torch.cuda.empty_cache()
        loaded, meta = load_moe_ffn_checkpoint(
            checkpoint_dir, tokenizer,
            base_model_name_or_path=args.model_name_or_path,
            device=device, dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation)
        if meta.get("checkpoint_format") != MOE_FFN_DELTA_FORMAT:
            raise AssertionError(f"delta metadata mismatch: {meta}")
        loaded_state = loaded.state_dict()
        for name, expected in delta.items():
            if not torch.equal(loaded_state[name].cpu(), expected):
                raise AssertionError(f"delta round-trip mismatch: {name}")

    print(
        "MOE_FFN_REAL_SMOKE=PASS base=64 growth=65,66 "
        f"forced_down_grad={down_grad:.6g} padding=PASS masks=PASS "
        f"optimizer_rebuild=PASS delta_roundtrip=PASS attention={actual_attn}")


if __name__ == "__main__":
    main()
