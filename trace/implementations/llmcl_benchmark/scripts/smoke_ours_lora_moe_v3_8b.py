#!/usr/bin/env python
"""CUDA smoke for V3 on a real Llama checkpoint, without an optimizer step."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Model-only validation must remain runnable while TRACE data is synchronized.
# Real training still imports the concrete collators and does not use this stub.
try:
    import utils.data.data_collator  # noqa: F401
except ModuleNotFoundError:
    data_package = types.ModuleType("utils.data")
    data_package.__path__ = []
    collator_module = types.ModuleType("utils.data.data_collator")
    for class_name in (
            "DataCollator", "SLoRATraceDataCollator",
            "PreTokenizedSLoRATraceDataCollator"):
        setattr(collator_module, class_name, type(class_name, (), {}))
    sys.modules["utils.data"] = data_package
    sys.modules["utils.data.data_collator"] = collator_module

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from model.Ours_LoRA_MoE_V3 import (  # noqa: E402
    Ours_LoRA_MoE_V3,
    add_v3_experts,
    attach_shared_qkvo_lora_moe,
    collect_v3_moe_losses,
    freeze_v3_experts,
    freeze_v3_routers,
    limit_v3_experts,
    set_v3_router_token_mask,
    shared_router_layers,
)


def _randomize_old_expert_outputs(model, std):
    with torch.no_grad():
        for layer in shared_router_layers(model):
            for projection in layer.attention_expert_projections:
                projection.experts[0].B.normal_(mean=0.0, std=std)
            for pair in layer.mlp.experts[0].values():
                pair.B.normal_(mean=0.0, std=std)


def _gradient_report(model):
    family_norms = {name: 0.0 for name in ("q", "k", "v", "o", "ffn")}
    router_norms = []
    old_grad_count = 0
    missing_new_b_grads = []
    for layer_index, layer in enumerate(shared_router_layers(model)):
        router_grad = layer.shared_expert_router.weight.grad
        router_norms.append(
            0.0 if router_grad is None else router_grad.float().norm().item())
        for target in ("q", "k", "v", "o"):
            projection = getattr(layer.self_attn, f"{target}_proj")
            old_grad_count += sum(
                parameter.grad is not None
                for parameter in projection.experts[0].parameters())
            grad = projection.experts[1].B.grad
            if grad is None:
                missing_new_b_grads.append(f"layer{layer_index}.{target}.B")
            else:
                family_norms[target] += grad.float().norm().item()
        for pair_name, pair in layer.mlp.experts[0].items():
            old_grad_count += sum(
                parameter.grad is not None for parameter in pair.parameters())
        for pair_name, pair in layer.mlp.experts[1].items():
            grad = pair.B.grad
            if grad is None:
                missing_new_b_grads.append(
                    f"layer{layer_index}.ffn.{pair_name}.B")
            else:
                family_norms["ffn"] += grad.float().norm().item()
    return {
        "new_b_grad_norm_sums": family_norms,
        "missing_new_b_grads": missing_new_b_grads,
        "old_expert_grad_tensor_count": old_grad_count,
        "router_grad_min": min(router_norms),
        "router_grad_max": max(router_norms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-sequence-length", type=int, default=0,
                        help="Use a synthetic fixed-length batch for the real backward.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--optimizer-step", action="store_true",
        help="Perform one primary-phase AdamW step and verify update bounds.")
    parser.add_argument(
        "--kd-optimizer-step", action="store_true",
        help="Perform a real old-prefix KD step before the primary step.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-checkpoint smoke")

    torch.manual_seed(2025)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    encoded = tokenizer(
        "A short CUDA smoke test for the shared expert router.",
        return_tensors="pt")
    batch = {key: value.to(args.device) for key, value in encoded.items()}

    with torch.no_grad():
        baseline = model(**batch, use_cache=False).logits.detach().clone()

    attach_shared_qkvo_lora_moe(
        model, r=args.rank, alpha=args.alpha, top_k=2,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="full_softmax", dropout=0.0)
    add_v3_experts(model, 1)
    layers = shared_router_layers(model)
    routing_records = [dict(router=[], consumers=[]) for _ in layers]
    handles = []
    for layer_index, layer in enumerate(layers):
        record = routing_records[layer_index]
        handles.append(layer.shared_expert_router.register_forward_hook(
            lambda module, inputs, output, record=record:
            record["router"].append(id(output))))

        def capture_context(module, inputs, record=record):
            record["consumers"].append(id(module._routing_context))

        handles.extend(module.register_forward_pre_hook(capture_context)
                       for module in (
                           *layer.attention_expert_projections, layer.mlp))
    set_v3_router_token_mask(model, batch.get("attention_mask"))
    try:
        with torch.no_grad():
            zero_initialized = model(
                **batch, use_cache=False).logits.detach().clone()
    finally:
        set_v3_router_token_mask(model, None)
        for handle in handles:
            handle.remove()
    zero_init_max_abs_diff = (
        baseline.float() - zero_initialized.float()).abs().max().item()
    shared_context_ok = all(
        len(record["router"]) == 1
        and len(record["consumers"]) == 5
        and set(record["consumers"]) == set(record["router"])
        for record in routing_records)

    with torch.no_grad():
        generated = model.generate(
            **batch, max_new_tokens=2, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)

    _randomize_old_expert_outputs(model, std=1e-4)
    model.eval()
    set_v3_router_token_mask(model, batch.get("attention_mask"))
    try:
        with torch.no_grad():
            old_pool_logits = model(
                **batch, use_cache=False).logits.detach().clone()
    finally:
        set_v3_router_token_mask(model, None)
    add_v3_experts(model, 1)
    set_v3_router_token_mask(model, batch.get("attention_mask"))
    try:
        with torch.no_grad(), limit_v3_experts(model, 1):
            recovered_old_pool = model(
                **batch, use_cache=False).logits.detach().clone()
    finally:
        set_v3_router_token_mask(model, None)
    old_prefix_max_abs_diff = (
        old_pool_logits.float() - recovered_old_pool.float()).abs().max().item()

    freeze_v3_experts(model, trainable_expert_indices={1})
    freeze_v3_routers(model, trainable=True)

    kd_report = {
        "kd_optimizer_step_performed": False,
        "kd_loss_finite": None,
        "kd_old_experts_bit_exact": None,
        "kd_old_router_rows_bit_exact": None,
        "kd_new_b_changed_count": 0,
        "kd_new_b_tensor_count": 0,
        "kd_new_router_changed_layers": 0,
    }
    if args.kd_optimizer_step:
        kd_trainer = object.__new__(Ours_LoRA_MoE_V3)
        kd_old_experts = []
        kd_new_b = []
        for layer in layers:
            pools = [layer.mlp.experts]
            pools.extend(
                projection.experts
                for projection in layer.attention_expert_projections)
            for pool in pools:
                kd_old_experts.extend(
                    (parameter, parameter.detach().clone())
                    for parameter in pool[0].parameters())
                kd_new_b.extend(
                    (parameter, parameter.detach().clone())
                    for name, parameter in pool[1].named_parameters()
                    if name.endswith("B"))
        kd_old_router_rows = kd_trainer._snapshot_old_router_rows(model, 1)
        kd_new_router_rows = [
            layer.shared_expert_router.weight[1:].detach().clone()
            for layer in layers]
        kd_optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            lr=1e-3, weight_decay=0.0)
        model.train()
        set_v3_router_token_mask(model, batch.get("attention_mask"))
        try:
            student_logits = model(
                **batch, use_cache=False).logits
            kd_args = types.SimpleNamespace(
                v2_kd_token_scope="nonpad", v2_kd_temperature=1.0,
                v2_kd_chunk_tokens=256)
            kd_loss = Ours_LoRA_MoE_V3._kd_kl_loss(
                student_logits, recovered_old_pool, batch, kd_args)
            kd_loss.backward()
        finally:
            set_v3_router_token_mask(model, None)
        kd_trainer._freeze_old_router_row_update(
            model, kd_old_router_rows, 1)
        kd_optimizer.step()
        kd_trainer._freeze_old_router_row_update(
            model, kd_old_router_rows, 1)
        kd_report = {
            "kd_optimizer_step_performed": True,
            "kd_loss_finite": bool(torch.isfinite(kd_loss).item()),
            "kd_old_experts_bit_exact": all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in kd_old_experts),
            "kd_old_router_rows_bit_exact": all(
                torch.equal(
                    layer.shared_expert_router.weight[:1], snapshot)
                for layer, snapshot in zip(layers, kd_old_router_rows)),
            "kd_new_b_changed_count": sum(
                not torch.equal(parameter, snapshot)
                for parameter, snapshot in kd_new_b),
            "kd_new_b_tensor_count": len(kd_new_b),
            "kd_new_router_changed_layers": sum(
                not torch.equal(
                    layer.shared_expert_router.weight[1:], snapshot)
                for layer, snapshot in zip(layers, kd_new_router_rows)),
        }
        kd_optimizer.zero_grad(set_to_none=True)
        del kd_optimizer

    old_expert_snapshots = []
    new_b_snapshots = []
    router_snapshots = []
    for layer in layers:
        router_snapshots.append(
            layer.shared_expert_router.weight.detach().clone())
        pools = [layer.mlp.experts]
        pools.extend(
            projection.experts
            for projection in layer.attention_expert_projections)
        for pool in pools:
            old_expert_snapshots.extend(
                (parameter, parameter.detach().clone())
                for parameter in pool[0].parameters())
            new_b_snapshots.extend(
                (parameter, parameter.detach().clone())
                for name, parameter in pool[1].named_parameters()
                if name.endswith("B"))
    unexpected_trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ".experts.1." not in name
        and ".shared_expert_router.router." not in name]
    train_batch = batch
    if args.train_batch_size > 1 or args.train_sequence_length > 0:
        sequence_length = args.train_sequence_length or batch["input_ids"].shape[1]
        pattern = torch.arange(sequence_length, device=args.device)
        pattern = pattern.remainder(model.config.vocab_size - 3).add(3)
        input_ids = pattern.unsqueeze(0).repeat(args.train_batch_size, 1)
        train_batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
    model.train()
    labels = train_batch["input_ids"].clone()
    set_v3_router_token_mask(model, train_batch.get("attention_mask"))
    try:
        outputs = model(**train_batch, labels=labels, use_cache=False)
        moe_loss = collect_v3_moe_losses(model)
        loss = outputs.loss + moe_loss
        loss.backward()
    finally:
        set_v3_router_token_mask(model, None)
    gradients = _gradient_report(model)

    optimizer_created = False
    optimizer_step_performed = False
    old_experts_bit_exact_after_step = None
    new_b_changed_count = 0
    router_changed_layers = 0
    if args.optimizer_step:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            lr=1e-3, weight_decay=0.0)
        optimizer_created = True
        optimizer.step()
        optimizer_step_performed = True
        old_experts_bit_exact_after_step = all(
            torch.equal(parameter, snapshot)
            for parameter, snapshot in old_expert_snapshots)
        new_b_changed_count = sum(
            not torch.equal(parameter, snapshot)
            for parameter, snapshot in new_b_snapshots)
        router_changed_layers = sum(
            not torch.equal(layer.shared_expert_router.weight, snapshot)
            for layer, snapshot in zip(layers, router_snapshots))

    first = layers[0]
    report = {
        "model_path": str(args.model.resolve()),
        "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": args.attn_implementation,
        "decoder_layers": len(layers),
        "hidden_size": first.hidden_size,
        "experts_per_layer": first.num_experts,
        "rank": first.mlp.r,
        "qkvo_base_weight_shapes": [
            list(projection.base_layer.weight.shape)
            for projection in first.attention_expert_projections],
        "ffn_base_weight_shapes": {
            name: list(getattr(first.mlp.base_mlp, f"{name}_proj").weight.shape)
            for name in ("gate", "up", "down")},
        "input_tokens": train_batch["input_ids"].shape[-1],
        "train_batch_size": train_batch["input_ids"].shape[0],
        "gradient_checkpointing": args.gradient_checkpointing,
        "generated_shape": list(generated.shape),
        "zero_init_max_abs_diff": zero_init_max_abs_diff,
        "shared_context_all_layers": shared_context_ok,
        "old_prefix_max_abs_diff": old_prefix_max_abs_diff,
        "loss_finite": bool(torch.isfinite(loss).item()),
        **gradients,
        **kd_report,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "elapsed_seconds": time.time() - started,
        "unexpected_trainable_parameters": unexpected_trainable,
        "optimizer_created": optimizer_created,
        "optimizer_step_performed": optimizer_step_performed,
        "old_experts_bit_exact_after_step":
            old_experts_bit_exact_after_step,
        "new_b_changed_count": new_b_changed_count,
        "new_b_tensor_count": len(new_b_snapshots),
        "router_changed_layers": router_changed_layers,
    }
    print(json.dumps(report, indent=2))

    assert len(layers) == model.config.num_hidden_layers == 32
    assert zero_init_max_abs_diff == 0.0
    assert shared_context_ok
    assert old_prefix_max_abs_diff == 0.0
    assert report["loss_finite"]
    assert not gradients["missing_new_b_grads"]
    assert gradients["old_expert_grad_tensor_count"] == 0
    assert all(value > 0 for value in gradients["new_b_grad_norm_sums"].values())
    assert gradients["router_grad_min"] > 0
    assert not unexpected_trainable
    if args.kd_optimizer_step:
        assert kd_report["kd_loss_finite"]
        assert kd_report["kd_old_experts_bit_exact"]
        assert kd_report["kd_old_router_rows_bit_exact"]
        assert (kd_report["kd_new_b_changed_count"]
                == kd_report["kd_new_b_tensor_count"])
        assert kd_report["kd_new_router_changed_layers"] == len(layers)
    if args.optimizer_step:
        assert old_experts_bit_exact_after_step
        assert new_b_changed_count == len(new_b_snapshots)
        assert router_changed_layers == len(layers)
    print("V3_REAL_8B_CUDA_SMOKE=PASS")


if __name__ == "__main__":
    main()
