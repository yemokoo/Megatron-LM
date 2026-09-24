#!/usr/bin/env python
"""Checks for the LLaVA-DyMoE port (model/tab1_dymoe.py).

1. parity   -- DyMoELinear against upstream IncMoELinear.forward, TAG and RSR
               copied verbatim from zhaoc5/DyMoE (B, L, E layout): outputs,
               RSR terms and gradients must match.
2. ckpt     -- RSR collected through non-reentrant gradient checkpointing gives
               the same loss and gradients as the unchecked forward.
3. trainer  -- three tiny tasks through DyMoETab1: old banks stay bit-frozen,
               banks fold at every task end, RSR switches on in the second half
               of task >= 2 only, and the saved round reloads through
               load_tab1_checkpoint with identical eval logits.

  CUDA_VISIBLE_DEVICES=0 python scripts/test_tab1_dymoe.py
"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tab1_checkpoint import load_tab1_checkpoint
from model.tab1_dymoe import (DyMoELinear, DyMoETab1, attach_dymoe_targets,
                              dymoe_total_experts, iter_dymoe)
from model.tab1_lora import resolve_targets

VOCAB = 64
DEVICE = torch.device("cuda") if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# Upstream reference, verbatim from llava/peft/tuners/incmoelora.py
# ---------------------------------------------------------------------------

def upstream_rsr(z, attention_mask=None, expert_num=16, k=16, temp=1.0, eps=1e-8):
    E = z.size(-1)
    num_E_old = E - expert_num
    z_f32 = z.float()
    k_eff = min(k, E)
    _, topk_idx = torch.topk(z_f32, k=k_eff, dim=-1)
    sel_mask = torch.zeros_like(z_f32, dtype=torch.bool).scatter(-1, topk_idx, True)
    z_selected = z_f32.masked_fill(~sel_mask, -1e9)
    w = F.softmax(z_selected / max(temp, eps), dim=-1)
    g_new = w[..., num_E_old:].sum(dim=-1)
    g_old = w[..., :num_E_old].sum(dim=-1)
    L_exc_tok = g_old * g_new
    g_tilde_old = torch.max(w[..., :num_E_old], dim=-1).values.detach()
    y = (1.0 - g_tilde_old).clamp(0.0, 1.0)
    L_spe_tok = -(y * torch.log(g_new.clamp(min=eps)) + (1.0 - y) * torch.log((1.0 - g_new).clamp(min=eps)))
    if attention_mask is None:
        mask = torch.ones_like(g_new)
    else:
        mask = attention_mask.to(dtype=torch.float32, device=z.device)
        if mask.shape != g_new.shape:
            mask = mask.expand_as(g_new)
    denom = mask.sum().clamp_min(1.0)
    L_exc = (L_exc_tok * mask).sum() / denom
    L_spe = (L_spe_tok * mask).sum() / denom
    return L_exc, L_spe


def upstream_tag(router, expert_num, conflict_ratio):
    num_E_old = router.size(-1) - expert_num
    if num_E_old <= 0:
        return router
    z_old = router[..., :num_E_old]
    z_new = router[..., num_E_old:]
    old_max = z_old.max(dim=-1, keepdim=True).values
    new_max = z_new.max(dim=-1, keepdim=True).values
    denominator = torch.max(torch.abs(old_max), torch.abs(new_max))
    relative_difference = torch.where(
        denominator == 0,
        torch.zeros_like(old_max),
        torch.abs(old_max - new_max) / denominator,
    )
    use_new = (relative_difference > conflict_ratio) & (new_max > old_max)
    z_old_masked = z_old.masked_fill(use_new, float('-inf'))
    z_new_masked = z_new.masked_fill(~use_new, float('-inf'))
    return torch.cat([z_old_masked, z_new_masked], dim=-1)


def upstream_forward(layer, x, training, task_id, attention_mask, group_router,
                     conflict_ratio, rsr_temperature, exc=True):
    """IncMoELinear.forward with lora_A/B/router read off a DyMoELinear."""
    aux = {}
    result = F.linear(x, layer.base.weight, bias=layer.base.bias)
    B, L, _ = x.shape
    s = layer.cosine_similarity_scale

    def score(w):
        return s * torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(w, p=2, dim=-1).T)

    router = score(layer.dymoe_router_old)
    has_new = training
    if has_new:
        router = torch.cat([router, score(layer.dymoe_router_new)], dim=-1)
        if exc and task_id > 1:
            aux["loss_exc"], aux["loss_spe"] = upstream_rsr(
                router, attention_mask=attention_mask, expert_num=layer.expert_num,
                k=layer.top_k, temp=rsr_temperature)
        if group_router:
            router = upstream_tag(router, layer.expert_num, conflict_ratio)
    total_expert_num = layer.total_expert_num
    expert_num = layer.expert_num if has_new else 0
    if layer.top_k < 0 or layer.top_k >= (total_expert_num + expert_num):
        router = torch.softmax(router / layer.router_temperature, dim=-1)
    else:
        topk_values, topk_indices = torch.topk(router, k=layer.top_k, dim=-1)
        mask = torch.full_like(router, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_values)
        router = torch.softmax(mask / layer.router_temperature, dim=-1)
    router_expanded = router.unsqueeze(-1)
    x_a = F.linear(x, layer.dymoe_A_old)
    if has_new:
        x_a = torch.cat([x_a, F.linear(x, layer.dymoe_A_new)], dim=-1)
    num_experts = total_expert_num + expert_num
    x_a = x_a.reshape(B, L, num_experts, -1)
    x_a = x_a * router_expanded
    x_a = x_a.reshape(B, L, -1)
    total_r = layer.total_r
    lora_output = F.linear(x_a[..., :total_r], layer.dymoe_B_old)
    if has_new:
        lora_output = lora_output + F.linear(x_a[..., total_r:], layer.dymoe_B_new)
    return result + lora_output * layer.scaling, aux


def check_parity():
    torch.manual_seed(0)
    base = nn.Linear(24, 20, bias=False)
    layer = DyMoELinear(base, r=8, alpha=16.0, dropout=0.0, expert_num=4,
                        top_k=4, router_temperature=0.01,
                        cosine_similarity_scale=1.0)
    # Task 3 state: two folded tasks + a trained-looking new bank.
    for _ in range(2):
        layer.add_new_bank()
        with torch.no_grad():
            layer.dymoe_B_new.normal_(0, 0.1)
        layer.fold_new_into_old()
    layer.add_new_bank()
    with torch.no_grad():
        layer.dymoe_B_new.normal_(0, 0.1)
    x = torch.randn(3, 5, 24)
    mask = torch.ones(3, 5, dtype=torch.long)
    mask[1, 3:] = 0
    mask[2, 1:] = 0

    for tau, name in ((0.2, "tau=0.2"), (0.0, "tau=0"), (0.9, "tau=0.9")):
        layer.train()
        layer.token_mask = mask
        layer.rsr_temperature = 0.1
        layer.tag_conflict_ratio = tau
        ours = layer(x)
        exc, spe = layer.last_rsr
        ref, aux = upstream_forward(layer, x, True, 3, mask, True, tau, 0.1)
        torch.testing.assert_close(ours, ref, rtol=0, atol=1e-6)
        torch.testing.assert_close(exc, aux["loss_exc"], rtol=0, atol=1e-7)
        torch.testing.assert_close(spe, aux["loss_spe"], rtol=0, atol=1e-7)

        grads = []
        for out in ((ours.square().mean() + exc + spe),
                    (ref.square().mean() + aux["loss_exc"] + aux["loss_spe"])):
            params = layer.new_parameters()
            grads.append(torch.autograd.grad(out, params))
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        print(f"  ok  parity train {name}: output/RSR/grad match upstream")

    # Without TAG (IncMoELoRA) and in eval mode (old bank only, top-4 of 8).
    layer.tag_conflict_ratio = None
    ours = layer(x)
    ref, _ = upstream_forward(layer, x, True, 3, mask, False, 0.2, 0.1)
    torch.testing.assert_close(ours, ref, rtol=0, atol=1e-6)
    layer.eval()
    ours = layer(x)
    ref, _ = upstream_forward(layer, x, False, 3, mask, True, 0.2, 0.1)
    torch.testing.assert_close(ours, ref, rtol=0, atol=1e-6)
    print("  ok  parity no-TAG and eval paths")

    # TAG really partitions: a token's routing mass lives in exactly one group.
    layer.train()
    layer.tag_conflict_ratio = 0.2
    router = torch.cat([layer._score(x.reshape(-1, 24), layer.dymoe_router_old),
                        layer._score(x.reshape(-1, 24), layer.dymoe_router_new)], -1)
    from model.tab1_dymoe import token_assignment_guidance
    masked = token_assignment_guidance(router, 4, 0.2)
    finite_old = torch.isfinite(masked[:, :8]).any(-1)
    finite_new = torch.isfinite(masked[:, 8:]).any(-1)
    assert bool((finite_old ^ finite_new).all()), "TAG left a token in both groups"
    print(f"  ok  TAG exclusive: {int(finite_new.sum())}/{finite_new.numel()} "
          "tokens to the new group")


# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

def tiny_config():
    return LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, max_position_embeddings=64)


class StubTokenizer:
    eos_token_id = VOCAB - 1
    pad_token_id = VOCAB - 1

    def __len__(self):
        return VOCAB

    def save_pretrained(self, *args, **kwargs):
        pass


def dymoe_args(output_dir, **overrides):
    args = SimpleNamespace(
        output_dir=output_dir, global_rank=0, local_rank=-1,
        lora_targets="all7", lora_rank=8, lora_alpha=16.0, lora_dropout=0.0,
        dymoe_variant="dymoe", dymoe_experts_per_task=4, dymoe_top_k=4,
        dymoe_router_temperature=0.01, dymoe_cosine_scale=1.0, dymoe_tag=1,
        dymoe_conflict_ratio=0.2, dymoe_exc_coeff=1e-3, dymoe_spe_coeff=1e-3,
        dymoe_rsr_temperature=0.1, dymoe_rsr_start_fraction=0.5,
        learning_rate=1e-2, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999,
        adam_epsilon=1e-8, lr_scheduler_type="cosine", num_warmup_steps=0,
        warmup_ratio=0.0, loss_log_interval=1, max_train_steps_per_task=0,
        start_task=0)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def attach(model, args):
    return attach_dymoe_targets(
        model, resolve_targets(args.lora_targets), args.lora_rank,
        args.lora_alpha, args.lora_dropout, args.dymoe_experts_per_task,
        args.dymoe_top_k, args.dymoe_router_temperature,
        args.dymoe_cosine_scale)


def make_batch(generator, batch=4, length=12, pad_from=None):
    ids = torch.randint(0, VOCAB - 1, (batch, length), generator=generator)
    mask = torch.ones_like(ids)
    if pad_from is not None:
        mask[-1, pad_from:] = 0
        ids[-1, pad_from:] = VOCAB - 1
    labels = ids.masked_fill(mask == 0, -100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def check_checkpointing(base_dir):
    """RSR through non-reentrant checkpointing == RSR without it."""
    args = dymoe_args("")
    results = []
    for ckpt in (False, True):
        torch.manual_seed(1)
        model = LlamaForCausalLM.from_pretrained(base_dir).to(DEVICE)
        attach(model, args)
        trainer = DyMoETab1.__new__(DyMoETab1)
        trainer.args = args
        trainer.raw_model = model
        trainer._layers = list(iter_dymoe(model))
        trainer._task_id = 2
        trainer._task_updates = 10
        trainer.lr_scheduler = SimpleNamespace(last_epoch=8)
        for layer in trainer._layers:
            layer.add_new_bank()
            with torch.no_grad():
                layer.dymoe_B_new.normal_(0, 0.1)
            layer.fold_new_into_old()
            layer.add_new_bank()
            with torch.no_grad():
                layer.dymoe_B_new.normal_(0, 0.1)
        for p in model.parameters():
            p.requires_grad = False
        params = [p for layer in trainer._layers for p in layer.new_parameters()]
        for p in params:
            p.requires_grad = True
        hook = model.register_forward_pre_hook(trainer._before_forward,
                                               with_kwargs=True)
        if ckpt:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        batch = {k: v.to(DEVICE) for k, v in
                 make_batch(torch.Generator().manual_seed(3), pad_from=7).items()}
        outputs = model(**batch, use_cache=False)
        extra = trainer.extra_loss(batch, outputs)
        assert extra is not None and trainer._last_terms[2] > 0
        loss = outputs.loss + 100.0 * extra  # amplify so RSR grads register
        loss.backward()
        results.append((float(loss), float(extra),
                        [p.grad.detach().clone() for p in params]))
        hook.remove()
    (l0, e0, g0), (l1, e1, g1) = results
    assert abs(l0 - l1) < 1e-5 and abs(e0 - e1) < 1e-8, (l0, l1, e0, e1)
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    print(f"  ok  grad checkpointing: loss {l0:.6f}, rsr term {e0:.3e}, "
          f"{len(g0)} grads equal")


def check_trainer(base_dir):
    work = tempfile.mkdtemp(prefix="tab1_dymoe_")
    try:
        tasks = ["t0", "t1", "t2"]
        generator = torch.Generator().manual_seed(7)
        loaders = {}
        for task in tasks:
            records = [make_batch(generator, batch=1, pad_from=9)
                       for _ in range(8)]
            records = [{k: v[0] for k, v in r.items()} for r in records]

            def collate(items):
                out = {k: torch.stack([item[k] for item in items]) for k in items[0]}
                out["sources"] = ["x"] * len(items)
                return out
            loaders[task] = DataLoader(records, batch_size=2, collate_fn=collate)
        args = dymoe_args(work, num_train_epochs=[2, 2, 2],
                          batch_by_task={t: 2 for t in tasks},
                          effective_global_batch=4,
                          gradient_accumulation_steps=2, ckpt_tasks={"t1"})
        torch.manual_seed(5)
        model = LlamaForCausalLM.from_pretrained(base_dir).to(DEVICE)
        attach(model, args)
        trainer = DyMoETab1(model, StubTokenizer(), None, loaders,
                            {t: None for t in tasks}, {t: None for t in tasks},
                            args)

        snapshots = {}
        rsr_seen = {}
        original_extra = trainer.extra_loss

        def spy(batch, outputs):
            term = original_extra(batch, outputs)
            rsr_seen.setdefault(trainer._task_id, []).append(term is not None)
            return term
        trainer.extra_loss = spy

        original_after = trainer.after_task

        def after(task, i_task):
            original_after(task, i_task)
            layer = trainer._layers[0]
            if i_task > 0:
                prev = snapshots[i_task - 1]
                torch.testing.assert_close(
                    layer.dymoe_A_old[:prev.shape[0]], prev, rtol=0, atol=0)
            snapshots[i_task] = layer.dymoe_A_old.detach().clone()
            assert dymoe_total_experts(model) == 4 * (i_task + 1)
            assert not layer.has_new_bank
        trainer.after_task = after
        trainer.train_continual()

        # 8 records / micro 2 = 4 steps x 2 epochs, accum 2 -> 4 updates; RSR
        # off on task 1, on for the second half of tasks 2 and 3 only.
        assert not any(rsr_seen[1]), rsr_seen[1]
        for task_id in (2, 3):
            flags = rsr_seen[task_id]
            assert not any(flags[:len(flags) // 2]) and any(flags), flags
        print(f"  ok  trainer: 3 tasks, frozen prefixes intact, "
              f"{dymoe_total_experts(model)} experts, RSR schedule {rsr_seen}")

        batch = {k: v.to(DEVICE) for k, v in
                 make_batch(torch.Generator().manual_seed(11)).items()
                 if k != "labels"}
        model.eval()
        with torch.no_grad():
            before = model(**batch).logits.float()
        reloaded, meta = load_tab1_checkpoint(
            os.path.join(work, "2"), StubTokenizer(), base_dir, device=DEVICE,
            dtype=torch.float32)
        with torch.no_grad():
            after_logits = reloaded(**batch).logits.float()
        diff = (before - after_logits).abs().max().item()
        assert diff < 1e-5, diff
        assert meta["total_expert_num"] == 12 and meta["total_r"] == 24
        assert meta["parameters"]["adapter"] > 0
        print(f"  ok  reload round 2: max logit diff {diff:.1e}, meta "
              f"experts={meta['total_expert_num']} total_r={meta['total_r']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    print("parity")
    check_parity()
    if DEVICE is None:
        raise SystemExit("trainer checks need a GPU (the trainer hardcodes cuda)")
    base_dir = tempfile.mkdtemp(prefix="tab1_dymoe_base_")
    try:
        torch.manual_seed(0)
        LlamaForCausalLM(tiny_config()).save_pretrained(base_dir)
        print("checkpointing")
        check_checkpointing(base_dir)
        print("trainer")
        check_trainer(base_dir)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
    print("\nall DyMoE tests passed")


if __name__ == "__main__":
    main()
