#!/usr/bin/env python
"""Real Llama-3.1-8B CUDA smoke for the FFN-only V2-new-top4 profile."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM  # noqa: E402

from model.Ours_LoRA_MoE import (  # noqa: E402
    LoRAMoEMLP,
    Ours_LoRA_MoE_V2,
    add_experts_to_all_layers,
    attach_lora_moe,
    collect_moe_losses,
    freeze_lora_moe_experts,
    freeze_lora_moe_routers,
    limit_lora_moe_experts,
    set_router_token_mask,
)


def lora_layers(model):
    layers = [layer.mlp for layer in model.model.layers]
    if not layers or not all(isinstance(layer, LoRAMoEMLP) for layer in layers):
        raise RuntimeError("model does not have LoRAMoEMLP on every decoder layer")
    return layers


def expert_parameters(layers, indices):
    rows = []
    for layer_index, layer in enumerate(layers):
        for expert_index in indices:
            for name, parameter in layer.experts[expert_index].named_parameters():
                rows.append((
                    f"layer{layer_index}.expert{expert_index}.{name}",
                    parameter,
                ))
    return rows


def expert_b_parameters(layers, indices):
    return [
        (name, parameter)
        for name, parameter in expert_parameters(layers, indices)
        if name.endswith(".B")
    ]


def snapshots(named_parameters):
    return [
        (name, parameter, parameter.detach().clone())
        for name, parameter in named_parameters
    ]


def changed_count(records):
    return sum(
        not torch.equal(parameter, before)
        for _, parameter, before in records
    )


def bit_exact(records):
    return all(
        torch.equal(parameter, before)
        for _, parameter, before in records
    )


def nonzero_grad_count(named_parameters):
    return sum(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for _, parameter in named_parameters
    )


def router_nonzero_grad_layers(layers, row_slice=None):
    count = 0
    for layer in layers:
        grad = layer.router.weight.grad
        if grad is None:
            continue
        if row_slice is not None:
            grad = grad[row_slice]
        count += torch.count_nonzero(grad).item() > 0
    return count


def router_changed_layers(layers, before, row_slice=None):
    count = 0
    for layer, snapshot in zip(layers, before):
        current = layer.router.weight
        if row_slice is not None:
            current = current[row_slice]
        count += not torch.equal(current, snapshot)
    return count


def forward_logits(model, batch, active_prefix=None):
    model.eval()
    set_router_token_mask(model, batch["attention_mask"])
    try:
        context = (
            limit_lora_moe_experts(model, active_prefix)
            if active_prefix is not None else torch.no_grad()
        )
        if active_prefix is not None:
            with torch.no_grad(), context:
                return model(**batch, use_cache=False).logits.detach().clone()
        with context:
            return model(**batch, use_cache=False).logits.detach().clone()
    finally:
        set_router_token_mask(model, None)


def lm_backward(model, batch, include_moe_loss):
    model.train()
    labels = batch["input_ids"].clone()
    set_router_token_mask(model, batch["attention_mask"])
    try:
        output = model(**batch, labels=labels, use_cache=False)
        loss = output.loss
        if include_moe_loss:
            moe_loss = collect_moe_losses(model)
            if moe_loss is not None:
                loss = loss + moe_loss
        loss.backward()
        return loss.detach()
    finally:
        set_router_token_mask(model, None)


def allowed_trainable_parameter(name, new_indices):
    if ".mlp.router." in name:
        return True
    match = re.search(r"\.mlp\.experts\.(\d+)\.", name)
    return match is not None and int(match.group(1)) in new_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real 8B smoke")
    if args.sequence_length < 2:
        raise ValueError("sequence length must be at least two")

    torch.manual_seed(2025)
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.config.use_cache = False
    vocab_size = int(model.config.vocab_size)
    input_ids = (
        torch.arange(args.sequence_length, device=args.device)
        .remainder(vocab_size - 3).add(3).unsqueeze(0)
    )
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    replay_batch = {
        "input_ids": input_ids.roll(shifts=1, dims=1),
        "attention_mask": torch.ones_like(input_ids),
    }

    with torch.no_grad():
        model.eval()
        baseline = model(**batch, use_cache=False).logits.detach().clone()

    attach_lora_moe(
        model, r=16, alpha=128, top_k=4,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    add_experts_to_all_layers(model, 4)
    layers = lora_layers(model)
    first_indices = set(range(4))
    freeze_lora_moe_experts(model, trainable_expert_indices=first_indices)
    freeze_lora_moe_routers(model, trainable=True)

    zero_initialized = forward_logits(model, batch)
    zero_init_max_abs_diff = (
        baseline.float() - zero_initialized.float()).abs().max().item()
    with torch.no_grad():
        generated = model.generate(
            **batch, max_new_tokens=1, do_sample=False,
            pad_token_id=model.config.eos_token_id,
            use_cache=True)

    initial_b = expert_b_parameters(layers, first_indices)
    initial_b_before = snapshots(initial_b)
    initial_router_before = [
        layer.router.weight.detach().clone() for layer in layers]
    unexpected_first_trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not allowed_trainable_parameter(name, first_indices)
    ]
    initial_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=0.0)
    initial_optimizer.zero_grad(set_to_none=True)
    initial_loss = lm_backward(model, batch, include_moe_loss=True)
    initial_b_grad_count = nonzero_grad_count(initial_b)
    initial_router_grad_layers = router_nonzero_grad_layers(layers)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], 1.0)
    initial_optimizer.step()
    initial_b_changed_count = changed_count(initial_b_before)
    initial_router_changed_layers = router_changed_layers(
        layers, initial_router_before)
    initial_optimizer.zero_grad(set_to_none=True)
    del initial_optimizer

    old_pool_logits = forward_logits(model, batch)
    old_experts_before_growth = snapshots(
        expert_parameters(layers, first_indices))
    add_experts_to_all_layers(model, 4)
    layers = lora_layers(model)
    prefix_recovered = forward_logits(model, batch, active_prefix=4)
    old_prefix_max_abs_diff = (
        old_pool_logits.float() - prefix_recovered.float()).abs().max().item()

    new_indices = set(range(4, 8))
    freeze_lora_moe_experts(model, trainable_expert_indices=new_indices)
    freeze_lora_moe_routers(model, trainable=True)
    unexpected_kd_trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not allowed_trainable_parameter(name, new_indices)
    ]
    new_b = expert_b_parameters(layers, new_indices)
    new_b_before_kd = snapshots(new_b)
    old_experts_before_kd = snapshots(
        expert_parameters(layers, first_indices))
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.raw_model = model
    old_router_rows = trainer._snapshot_old_router_rows(model, 4)
    new_router_before_kd = [
        layer.router.weight[4:].detach().clone() for layer in layers]
    kd_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=0.0)
    kd_args = SimpleNamespace(
        v2_kd_token_scope="nonpad",
        v2_kd_temperature=1.0,
        v2_kd_chunk_tokens=256,
    )
    kd_losses = []
    kd_new_b_grad_names = set()
    kd_new_router_grad_layer_indices = set()
    # Two steps intentionally exercise zero-init dynamics. On step one a layer
    # that selects only zero-output new experts can update B but not its router;
    # after B opens, step two must propagate KD into that router too.
    for _ in range(2):
        kd_optimizer.zero_grad(set_to_none=True)
        model.train()
        set_router_token_mask(model, batch["attention_mask"])
        try:
            student_logits = model(**batch, use_cache=False).logits
            kd_loss = Ours_LoRA_MoE_V2._kd_kl_loss(
                student_logits, prefix_recovered, batch, kd_args)
            kd_loss.backward()
        finally:
            set_router_token_mask(model, None)
        kd_losses.append(kd_loss.detach())
        kd_new_b_grad_names.update(
            name for name, parameter in new_b
            if parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0)
        kd_new_router_grad_layer_indices.update(
            index for index, layer in enumerate(layers)
            if layer.router.weight.grad is not None
            and torch.count_nonzero(
                layer.router.weight.grad[4:]).item() > 0)
        trainer._freeze_old_router_row_update(model, old_router_rows, 4)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad], 1.0)
        kd_optimizer.step()
        trainer._freeze_old_router_row_update(model, old_router_rows, 4)
    kd_new_b_grad_count = len(kd_new_b_grad_names)
    kd_new_router_grad_layers = len(kd_new_router_grad_layer_indices)
    kd_optimizer.zero_grad(set_to_none=True)
    del kd_optimizer
    kd_old_experts_bit_exact = bit_exact(old_experts_before_kd)
    kd_old_router_rows_bit_exact = all(
        torch.equal(layer.router.weight[:4], before)
        for layer, before in zip(layers, old_router_rows)
    )
    kd_new_b_changed_count = changed_count(new_b_before_kd)
    kd_new_router_changed_layers = router_changed_layers(
        layers, new_router_before_kd, row_slice=slice(4, None))
    prefix_after_kd = forward_logits(model, batch, active_prefix=4)
    prefix_after_kd_max_abs_diff = (
        old_pool_logits.float() - prefix_after_kd.float()).abs().max().item()

    # One actual 1-phase-style combined update: primary CE+MoE first, then a
    # replay CE backward with every expert temporarily frozen.
    freeze_lora_moe_experts(model, trainable_expert_indices=new_indices)
    freeze_lora_moe_routers(model, trainable=True)
    old_experts_before_joint = snapshots(
        expert_parameters(layers, first_indices))
    new_parameters = expert_parameters(layers, new_indices)
    new_parameters_before_joint = snapshots(new_parameters)
    router_before_joint = [
        layer.router.weight.detach().clone() for layer in layers]
    joint_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=0.0)
    joint_optimizer.zero_grad(set_to_none=True)
    primary_loss = lm_backward(model, batch, include_moe_loss=True)
    expert_grads_before_replay = [
        (name, parameter, None if parameter.grad is None
         else parameter.grad.detach().clone())
        for name, parameter in new_parameters
    ]
    router_grads_before_replay = [
        None if layer.router.weight.grad is None
        else layer.router.weight.grad.detach().clone()
        for layer in layers
    ]
    with trainer._router_only_replay():
        replay_loss = lm_backward(
            model, replay_batch, include_moe_loss=False)
    replay_preserved_expert_grads = all(
        (before is None and parameter.grad is None)
        or (before is not None and parameter.grad is not None
            and torch.equal(parameter.grad, before))
        for _, parameter, before in expert_grads_before_replay
    )
    replay_changed_router_grad_layers = sum(
        before is not None and layer.router.weight.grad is not None
        and not torch.equal(layer.router.weight.grad, before)
        for layer, before in zip(layers, router_grads_before_replay)
    )
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], 1.0)
    joint_optimizer.step()
    joint_optimizer.zero_grad(set_to_none=True)
    del joint_optimizer
    joint_old_experts_bit_exact = bit_exact(old_experts_before_joint)
    joint_new_parameter_changed_count = changed_count(
        new_parameters_before_joint)
    joint_router_changed_layers = router_changed_layers(
        layers, router_before_joint)

    report = {
        "model_path": str(args.model.resolve()),
        "dtype": str(next(model.parameters()).dtype),
        "decoder_layers": len(layers),
        "sequence_length": args.sequence_length,
        "rank": layers[0].r,
        "alpha": layers[0].alpha,
        "top_k": layers[0].top_k,
        "routing_weight_mode": layers[0].routing_weight_mode,
        "experts_after_first_growth": 4,
        "experts_after_second_growth": layers[0].num_experts,
        "generated_shape": list(generated.shape),
        "zero_init_max_abs_diff": zero_init_max_abs_diff,
        "initial_loss": float(initial_loss),
        "initial_b_grad_count": initial_b_grad_count,
        "initial_b_tensor_count": len(initial_b),
        "initial_router_grad_layers": initial_router_grad_layers,
        "initial_b_changed_count": initial_b_changed_count,
        "initial_router_changed_layers": initial_router_changed_layers,
        "old_prefix_max_abs_diff": old_prefix_max_abs_diff,
        "kd_steps": len(kd_losses),
        "kd_losses": [float(loss) for loss in kd_losses],
        "kd_new_b_grad_count": kd_new_b_grad_count,
        "kd_new_b_tensor_count": len(new_b),
        "kd_new_router_grad_layers": kd_new_router_grad_layers,
        "kd_old_experts_bit_exact": kd_old_experts_bit_exact,
        "kd_old_router_rows_bit_exact": kd_old_router_rows_bit_exact,
        "kd_new_b_changed_count": kd_new_b_changed_count,
        "kd_new_router_changed_layers": kd_new_router_changed_layers,
        "prefix_after_kd_max_abs_diff": prefix_after_kd_max_abs_diff,
        "primary_loss_joint": float(primary_loss),
        "replay_loss_joint": float(replay_loss),
        "replay_preserved_expert_grads": replay_preserved_expert_grads,
        "replay_changed_router_grad_layers":
            replay_changed_router_grad_layers,
        "joint_old_experts_bit_exact": joint_old_experts_bit_exact,
        "joint_new_parameter_changed_count":
            joint_new_parameter_changed_count,
        "joint_new_parameter_count": len(new_parameters),
        "joint_router_changed_layers": joint_router_changed_layers,
        "old_experts_survived_growth_bit_exact":
            bit_exact(old_experts_before_growth),
        "unexpected_first_trainable": unexpected_first_trainable,
        "unexpected_kd_trainable": unexpected_kd_trainable,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(report, indent=2))

    assert len(layers) == model.config.num_hidden_layers == 32
    assert layers[0].num_experts == 8
    assert layers[0].r == 16 and layers[0].alpha == 128
    assert layers[0].top_k == 4
    assert layers[0].routing_weight_mode == "straight_through_topk"
    assert zero_init_max_abs_diff == 0.0
    assert old_prefix_max_abs_diff == 0.0
    assert prefix_after_kd_max_abs_diff == 0.0
    assert torch.isfinite(initial_loss)
    assert initial_b_grad_count == len(initial_b)
    assert initial_router_grad_layers == len(layers)
    assert initial_b_changed_count == len(initial_b)
    assert initial_router_changed_layers == len(layers)
    assert not unexpected_first_trainable
    assert not unexpected_kd_trainable
    assert all(torch.isfinite(loss) for loss in kd_losses)
    assert kd_new_b_grad_count == len(new_b)
    assert kd_new_router_grad_layers == len(layers)
    assert kd_old_experts_bit_exact
    assert kd_old_router_rows_bit_exact
    assert kd_new_b_changed_count == len(new_b)
    assert kd_new_router_changed_layers == len(layers)
    assert replay_preserved_expert_grads
    assert replay_changed_router_grad_layers == len(layers)
    assert joint_old_experts_bit_exact
    assert joint_new_parameter_changed_count > 0
    assert joint_router_changed_layers == len(layers)
    assert report["old_experts_survived_growth_bit_exact"]
    print("V2_NEW_TOP4_REAL_8B_CUDA_SMOKE=PASS")


if __name__ == "__main__":
    main()
