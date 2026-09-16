#!/usr/bin/env python
"""Structural tests for the Table-1 baselines on a tiny real Llama.

A synthetic nn.Module is not enough here: the ffn_attn scope replaces whole
decoder layers, so the layouts have to be exercised against an actual
LlamaForCausalLM.  The config below is small enough to run on CPU in seconds.

  python scripts/test_tab1_baselines.py
"""
import copy
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tab1_lora import (EWCState, adapter_parameter_count,
                             attach_olora_targets, attach_seq_lora_targets,
                             estimate_diagonal_fisher, merge_olora_into_base,
                             olora_regularization, resolve_targets,
                             set_olora_task, snapshot_parameters)
from model.tab1_moe import (RouterLogitCapture, build_scope,
                            validate_shared_path, attach_shared_path)

TASKS = ["t0", "t1", "t2"]


def tiny_model():
    config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, torch_dtype=torch.float32)
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def tiny_batch(batch=2, length=6, vocab=128):
    input_ids = torch.randint(0, vocab, (batch, length))
    return {"input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone()}


def base_args(**overrides):
    args = SimpleNamespace(
        lora_targets="all7", lora_rank=4, lora_alpha=8.0, lora_dropout=0.0,
        moe_scope="ffn", lora_moe_rank=4, lora_moe_alpha=8.0,
        lora_moe_dropout=0.0, experts_per_task=1, top_k=1,
        routing_weight_mode="straight_through_topk",
        moe_aux_loss_coeff=0.01, moe_z_loss_coeff=0.001,
        lifelong_shared_targets="attn", lifelong_train_shared=1,
        lifelong_kd_coeff=1.0, lifelong_kd_temperature=1.0,
        lpr_gamma=0.1, lpr_review_epochs=1, ewc_lambda=400.0,
        ewc_mode="online", ewc_fisher_samples=0)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


# ---------------------------------------------------------------------------

def test_target_sets():
    print("target sets")
    model = tiny_model()
    wrapped = attach_seq_lora_targets(model, resolve_targets("all7"), 4, 8.0)
    check(wrapped == 2 * 7, f"all7 wraps 14 projections, got {wrapped}")
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    check(all(".lora." in name for name in trainable),
          "only LoRA tensors are trainable after attach")
    check(len(trainable) == 2 * 7 * 2, "each projection contributes A and B")
    out = model(**tiny_batch())
    check(torch.isfinite(out.loss), "seq-LoRA forward is finite")
    counts = adapter_parameter_count(model)
    check(counts["trainable"] == counts["adapter"],
          "trainable equals adapter parameter count")


def test_olora():
    print("olora")
    model = tiny_model()
    attach_olora_targets(model, resolve_targets("all7"), 4, 8.0, num_tasks=3)
    set_olora_task(model, 0)
    orthogonal, l2 = olora_regularization(model)
    check(float(orthogonal) == 0.0, "task 0 has no orthogonality term")
    check(float(l2) > 0.0, "task 0 still has an L2 term")

    # Give slot 0 a non-zero B so it actually contributes, then move to task 1.
    for name, parameter in model.named_parameters():
        if ".adapters.0.B" in name:
            with torch.no_grad():
                parameter.normal_(0, 0.02)
    set_olora_task(model, 1)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    check(all(".adapters.1." in name for name in trainable),
          "only the current slot trains")
    orthogonal, _ = olora_regularization(model)
    check(float(orthogonal) > 0.0, "task 1 penalises overlap with slot 0")

    batch = tiny_batch()
    before = model(**batch).logits.detach().clone()
    merged = merge_olora_into_base(model, upto_task=1)
    check(merged == 2 * 7, f"merge touched every projection, got {merged}")
    for layer in model.model.layers:
        layer.self_attn.q_proj.adapters = torch.nn.ModuleList()
        layer.self_attn.k_proj.adapters = torch.nn.ModuleList()
        layer.self_attn.v_proj.adapters = torch.nn.ModuleList()
        layer.self_attn.o_proj.adapters = torch.nn.ModuleList()
        layer.mlp.gate_proj.adapters = torch.nn.ModuleList()
        layer.mlp.up_proj.adapters = torch.nn.ModuleList()
        layer.mlp.down_proj.adapters = torch.nn.ModuleList()
    after = model(**batch).logits.detach()
    check(torch.allclose(before, after, atol=1e-4),
          f"merged weights reproduce the adapter path (max diff "
          f"{(before - after).abs().max():.2e})")


def test_ewc():
    print("ewc")
    model = tiny_model()
    attach_seq_lora_targets(model, resolve_targets("all7"), 4, 8.0)
    loader = [tiny_batch() for _ in range(3)]
    state = EWCState(mode="online")
    check(state.penalty(model.named_parameters()) is None,
          "no penalty before any task completes")
    fisher = estimate_diagonal_fisher(model, loader, torch.device("cpu"))
    check(all(torch.isfinite(v).all() for v in fisher.values()),
          "Fisher entries are finite")
    total = sum(float(v.sum()) for v in fisher.values())
    check(total > 0, f"Fisher is non-trivial (sum {total:.3e})")
    state.absorb(fisher, snapshot_parameters(model))
    penalty = state.penalty(model.named_parameters())
    check(float(penalty) == 0.0, "penalty is zero at the anchor itself")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.05)
    penalty = state.penalty(model.named_parameters())
    check(float(penalty) > 0, f"penalty grows away from the anchor ({penalty:.3e})")
    summary = state.state_summary()
    check(summary["tasks_seen"] == 1, "one task absorbed")

    # Real runs hold bf16 parameters, where a realistic drift (~1e-4) squared
    # and weighted by a Fisher entry (~1e-7) underflows the accumulation if the
    # penalty is computed in the parameter dtype.
    bf16 = tiny_model().to(torch.bfloat16)
    attach_seq_lora_targets(bf16, resolve_targets("all7"), 4, 8.0)
    for parameter in bf16.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    bf16_state = EWCState(mode="online")
    bf16_fisher = {name: torch.full_like(p, 1e-7, dtype=torch.float32)
                   for name, p in bf16.named_parameters() if p.requires_grad}
    bf16_state.absorb(bf16_fisher, snapshot_parameters(bf16))
    with torch.no_grad():
        for name, parameter in bf16.named_parameters():
            if parameter.requires_grad:
                parameter.add_(torch.full_like(parameter, 1e-4))
    bf16_penalty = bf16_state.penalty(bf16.named_parameters())
    check(float(bf16_penalty) > 0,
          f"bf16 parameters still give a non-zero penalty ({float(bf16_penalty):.3e})")


def test_moe_scope(scope_name):
    print(f"moe scope: {scope_name}")
    args = base_args(moe_scope=scope_name)
    model = tiny_model()
    scope = build_scope(scope_name)
    scope.attach(model, args)
    check(scope.num_experts(model) == 0, "no experts before the first task")

    scope.add_experts(model, 1)
    check(scope.num_experts(model) == 1, "one expert after task 0")
    scope.freeze_experts(model, trainable_expert_indices={0})
    scope.freeze_routers(model, trainable=True)
    batch = tiny_batch()
    scope.set_token_mask(model, batch["attention_mask"])
    out = model(**batch)
    loss = out.loss + (scope.collect_losses(model) or 0.0)
    loss.backward()
    scope.set_token_mask(model, None)
    check(torch.isfinite(loss), "task-0 forward/backward is finite")

    # Grow a second expert and confirm the old router rows can be pinned.
    model.zero_grad(set_to_none=True)
    rows = scope.snapshot_router_rows(model, 1)
    scope.add_experts(model, 1)
    check(scope.num_experts(model) == 2, "two experts after task 1")
    scope.freeze_experts(model, trainable_expert_indices={1})
    scope.freeze_routers(model, trainable=True)
    scope.set_token_mask(model, batch["attention_mask"])
    with RouterLogitCapture(scope, model) as capture:
        out = model(**batch)
        logits = capture.logits(2)
        check(len(logits) == len(list(scope.router_hosts(model))),
              "captured one router logit tensor per layer")
        check(logits[0].shape[-1] == 2, "captured logits have one column per expert")
        target = torch.full((logits[0].shape[0],), 1, dtype=torch.long)
        lpr = sum(F.cross_entropy(item.float(), target) for item in logits)
        (out.loss + 0.1 * lpr).backward()
    scope.set_token_mask(model, None)
    check(torch.isfinite(lpr), "LPR cross-entropy is finite")

    # A step must not move the pinned prefix.
    with torch.no_grad():
        for _, host in [(h, h) for h, _ in scope.router_hosts(model)]:
            pass
    scope.restore_router_rows(model, rows, 1)
    after = scope.snapshot_router_rows(model, 1)
    check(all(torch.equal(a, b) for a, b in zip(rows, after)),
          "old router rows are restored bit-identically")

    old_expert_grads = []
    for name, parameter in model.named_parameters():
        if ".experts.0." in name:
            old_expert_grads.append(parameter.requires_grad)
    check(old_expert_grads and not any(old_expert_grads),
          "every task-0 expert tensor is frozen during task 1")


def test_lifelong_shared_path():
    print("lifelong shared path")
    check(validate_shared_path("ffn", "attn") == resolve_targets("attn"),
          "attn is the valid shared path under the ffn scope")
    check(validate_shared_path("ffn", "none") == (),
          "none disables the shared path")
    for scope_name, targets in (("ffn", "all7"), ("ffn", "ffn"),
                                ("ffn_attn", "attn")):
        try:
            validate_shared_path(scope_name, targets)
        except ValueError:
            print(f"  ok  {scope_name}/{targets} is refused with an explanation")
        else:
            raise AssertionError(f"{scope_name}/{targets} should be refused")

    args = base_args()
    model = tiny_model()
    shared = attach_shared_path(model, "ffn", "attn", 4, 8.0, 0.0)
    check(shared == 2 * 4, f"shared adapter on 8 attention projections, got {shared}")
    build_scope("ffn").attach(model, args)
    build_scope("ffn").add_experts(model, 1)
    names = [n for n, _ in model.named_parameters() if ".lora." in n]
    check(len(names) == 2 * 4 * 2, "shared adapter survives the expert layout")
    for name, parameter in model.named_parameters():
        if ".lora." in name:
            parameter.requires_grad = True
    out = model(**tiny_batch())
    out.loss.backward()
    grads = [p.grad is not None for n, p in model.named_parameters()
             if ".lora." in n]
    check(all(grads), "the shared adapter receives gradient")


def test_lpr_semantics():
    """The LPR term must match pretrain_gpt.py::_masked_task_group_lpr.

    Two properties the wiki implementation has and a naive port loses:
    the newest task carries no label, and the reduction is a sum over
    supervised tokens divided by ALL valid tokens (Megatron adds it to the
    loss numerator), not a mean over supervised tokens.
    """
    print("lpr semantics")
    from model.tab1_moe import MoELPR

    trainer = MoELPR.__new__(MoELPR)
    trainer.args = base_args(lpr_gamma=1.0)
    torch.manual_seed(0)

    tokens, experts, layers = 4, 3, 2
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    batch = {"attention_mask": attention}
    task_labels = torch.tensor([0, 2])          # sample 0 old, sample 1 current
    current = 2
    logit_tensors = [torch.randn(2 * tokens, experts) for _ in range(layers)]

    class _Capture:
        def logits(self, active):
            assert active == experts
            return logit_tensors

    value = trainer._lpr_loss(batch, _Capture(), task_labels, current, experts)

    valid = attention.bool()
    supervised = (valid & (task_labels[:, None].expand_as(valid) != current)).reshape(-1)
    expected = 0.0
    for logits in logit_tensors:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        expected += -log_probs[supervised][:, 0].sum() / int(valid.sum())
    expected = float(expected / layers)
    check(abs(float(value) - expected) < 1e-6,
          f"sum/valid-token reduction matches Megatron ({float(value):.6f})")

    # A mean over supervised tokens would be larger by valid/supervised.
    naive = expected * int(valid.sum()) / int(supervised.sum())
    check(abs(naive - expected) > 1e-4,
          f"a supervised-mean would differ ({naive:.6f}), so the choice matters")

    # Batches holding only current-task samples contribute nothing.
    only_current = trainer._lpr_loss(
        batch, _Capture(), torch.tensor([current, current]), current, experts)
    check(only_current is None, "the newest task alone yields no LPR term")

    # Sample 1 is the current task, so masking it out must not change the value.
    flipped = trainer._lpr_loss(
        batch, _Capture(), torch.tensor([0, current]), current, experts)
    check(abs(float(flipped) - float(value)) < 1e-9,
          "current-task tokens never enter the numerator")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_target_sets()
    test_olora()
    test_ewc()
    test_moe_scope("ffn")
    test_moe_scope("ffn_attn")
    test_lifelong_shared_path()
    test_lpr_semantics()
    print("\nall tab1 structural tests passed")
