#!/usr/bin/env python
"""Save/reload round-trip for every Table-1 layout.

The metadata is produced by the trainers' own ``write_meta`` and consumed by
``load_tab1_checkpoint``, so a field the writer forgets or the reader renames
fails here in seconds instead of after an eight-task training run.  Reloaded
logits must match the trained model's bit-for-bit within fp32 tolerance.

  python scripts/test_tab1_checkpoint_roundtrip.py
"""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tab1_baselines import EWCTab1, MTLTab1, OLoRATab1, SeqLoRATab1
from model.tab1_checkpoint import load_tab1_checkpoint
from model.tab1_lora import (TAB1_STATE_KEY_SUBSTRINGS, attach_olora_targets,
                             attach_seq_lora_targets, resolve_targets,
                             set_olora_task)
from model.tab1_moe import LifelongMoE, MoELPR, attach_shared_path, build_scope

TASKS = ["t0", "t1", "t2"]
VOCAB = 64


class StubTokenizer:
    """create_hf_model needs len() for vocab resizing and the two special ids."""

    eos_token_id = VOCAB - 1
    pad_token_id = VOCAB - 1

    def __len__(self):
        return VOCAB

    def save_pretrained(self, *args, **kwargs):
        pass


def tiny_config():
    return LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64)


def base_args(output_dir, **overrides):
    args = SimpleNamespace(
        output_dir=output_dir, global_rank=0, local_rank=-1,
        lora_targets="all7", lora_rank=4, lora_alpha=8.0, lora_dropout=0.0,
        moe_scope="ffn", lora_moe_rank=4, lora_moe_alpha=8.0,
        lora_moe_dropout=0.0, experts_per_task=1, top_k=1,
        routing_weight_mode="straight_through_topk",
        moe_aux_loss_coeff=0.01, moe_z_loss_coeff=0.001,
        lifelong_shared_targets="attn", lifelong_train_shared=1,
        lifelong_kd_coeff=1.0, lifelong_kd_temperature=1.0,
        lpr_gamma=0.1, lpr_review_epochs=1,
        router_replay_exposure_samples=1000,
        replay_manifest_path="", replay_subset_ratio=0.001,
        replay_subset_seed=-1, replay_selection_mode="random",
        replay_distribution="equal_task", replay_recency_power=1.0,
        ewc_lambda=400.0, ewc_mode="online", ewc_fisher_samples=0,
        olora_lambda_orthogonal="0.5", olora_lambda_l2="0.0",
        olora_merge_at_end=0, mtl_epochs=5)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def randomize_adapters(model):
    """B starts at zero, so an untouched adapter is a no-op and proves nothing."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(fragment in name for fragment in TAB1_STATE_KEY_SUBSTRINGS):
                parameter.normal_(0, 0.05)


def logits_of(model, batch):
    model.eval()
    with torch.no_grad():
        return model(**batch).logits.float().clone()


def save_adapters(model, directory, key_substrings):
    os.makedirs(directory, exist_ok=True)
    state = {key: value for key, value in model.state_dict().items()
             if any(fragment in key for fragment in key_substrings)}
    if not state:
        raise AssertionError(f"nothing to save for {directory}")
    torch.save(state, os.path.join(directory, "pytorch_model.bin"))
    return len(state)


def run_case(name, base_dir, build, trainer_class, args, key_substrings,
             batch):
    work = tempfile.mkdtemp(prefix=f"tab1_{name}_")
    try:
        args.output_dir = work
        model = LlamaForCausalLM(tiny_config())
        model.load_state_dict(
            LlamaForCausalLM.from_pretrained(base_dir).state_dict())
        build(model, args)
        randomize_adapters(model)
        before = logits_of(model, batch)

        trainer = trainer_class.__new__(trainer_class)
        trainer.args = args
        trainer.raw_model = model
        trainer.model = model
        trainer.train_task_list = {task: None for task in TASKS}
        if hasattr(trainer_class, "scope"):
            pass
        if trainer_class in (LifelongMoE, MoELPR):
            trainer.scope = build_scope(args.moe_scope)
        round_dir = os.path.join(work, "2")
        trainer.write_meta(2, **({"num_tasks": len(TASKS), "current_task": 2,
                                  "merged": False}
                                 if name.startswith("olora") else {}))
        if name.startswith("lifelong"):
            trainer.write_meta(
                2, shared_targets=args.lifelong_shared_targets,
                train_shared=args.lifelong_train_shared)
        tensors = save_adapters(model, round_dir, key_substrings)

        reloaded, meta = load_tab1_checkpoint(
            round_dir, StubTokenizer(), base_dir, device=torch.device("cpu"),
            dtype=torch.float32)
        after = logits_of(reloaded, batch)
        difference = (before - after).abs().max().item()
        if difference > 1e-5:
            raise AssertionError(f"{name}: logits differ by {difference:.2e}")
        print(f"  ok  {name:22s} {tensors:>4d} tensors, "
              f"max logit diff {difference:.1e}, meta={meta['method']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    torch.manual_seed(0)
    base_dir = tempfile.mkdtemp(prefix="tab1_base_")
    try:
        LlamaForCausalLM(tiny_config()).save_pretrained(base_dir)
        input_ids = torch.randint(0, VOCAB, (2, 6))
        batch = {"input_ids": input_ids,
                 "attention_mask": torch.ones_like(input_ids)}

        print("round-trip")
        lora_keys = [".lora."]
        run_case("seq_lora", base_dir,
                 lambda m, a: attach_seq_lora_targets(
                     m, resolve_targets(a.lora_targets), a.lora_rank,
                     a.lora_alpha, a.lora_dropout),
                 SeqLoRATab1, base_args(""), lora_keys, batch)
        run_case("ewc", base_dir,
                 lambda m, a: attach_seq_lora_targets(
                     m, resolve_targets(a.lora_targets), a.lora_rank,
                     a.lora_alpha, a.lora_dropout),
                 EWCTab1, base_args(""), lora_keys, batch)
        run_case("mtl", base_dir,
                 lambda m, a: attach_seq_lora_targets(
                     m, resolve_targets(a.lora_targets), a.lora_rank,
                     a.lora_alpha, a.lora_dropout),
                 MTLTab1, base_args(""), lora_keys, batch)

        def build_olora(m, a):
            attach_olora_targets(m, resolve_targets(a.lora_targets),
                                 a.lora_rank, a.lora_alpha,
                                 num_tasks=len(TASKS), dropout=a.lora_dropout)
            set_olora_task(m, 2)
        run_case("olora", base_dir, build_olora, OLoRATab1, base_args(""),
                 [".adapters."], batch)

        for scope_name in ("ffn", "ffn_attn"):
            scope = build_scope(scope_name)

            def build_moe(m, a, scope=scope):
                scope.attach(m, a)
                scope.add_experts(m, 3)
            args = base_args("", moe_scope=scope_name)
            run_case(f"moe_lpr[{scope_name}]", base_dir, build_moe, MoELPR,
                     args, scope.save_key_substrings, batch)

            shared = "attn" if scope_name == "ffn" else "none"

            def build_lifelong(m, a, scope=scope, shared=shared):
                attach_shared_path(m, a.moe_scope, shared, a.lora_rank,
                                   a.lora_alpha, a.lora_dropout)
                scope.attach(m, a)
                scope.add_experts(m, 3)
            args = base_args("", moe_scope=scope_name,
                             lifelong_shared_targets=shared)
            run_case(f"lifelong[{scope_name}]", base_dir, build_lifelong,
                     LifelongMoE, args,
                     list(scope.save_key_substrings) + [".lora."], batch)
        print("\nall tab1 round-trip tests passed")
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
