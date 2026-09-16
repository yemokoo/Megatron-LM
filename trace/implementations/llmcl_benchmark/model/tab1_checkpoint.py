"""Rebuild a Table-1 checkpoint over its frozen pretrained base.

Every Table-1 run saves adapters only -- experts, routers and LoRA pairs --
next to a ``tab1_meta.json`` describing the layout.  This module turns that
pair back into a runnable model.  The Ours loaders cannot be reused directly:
``load_lora_moe_checkpoint`` knows nothing about Lifelong-MoE's shared
adapter, and ``load_paper_baseline_checkpoint`` only ever attaches to the FFN
triple, so both would reject a Table-1 state dict as having unexpected keys.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch

from model.tab1_lora import (attach_olora_targets, attach_seq_lora_targets,
                             merge_olora_into_base, resolve_targets,
                             set_olora_task)
from model.tab1_moe import attach_shared_path, build_scope

TAB1_META_NAME = "tab1_meta.json"
LORA_METHODS = {"seq_lora", "ewc", "olora", "mtl"}
MOE_METHODS = {"lifelong_moe", "moe_lpr"}


def read_tab1_meta(checkpoint_dir: str) -> Dict[str, object]:
    with open(os.path.join(checkpoint_dir, TAB1_META_NAME),
              encoding="utf-8") as handle:
        return json.load(handle)


def load_tab1_checkpoint(checkpoint_dir: str, tokenizer,
                         base_model_name_or_path: str,
                         device="cuda", dtype=torch.bfloat16,
                         device_map=None,
                         merge_olora: Optional[bool] = None
                         ) -> Tuple[torch.nn.Module, Dict[str, object]]:
    from transformers import AutoModelForCausalLM
    from utils.model.model_utils import create_hf_model

    if base_model_name_or_path is None:
        raise ValueError(
            "Table-1 checkpoints hold adapters only; "
            "base_model_name_or_path is required")
    meta = read_tab1_meta(checkpoint_dir)
    method = meta["method"]

    model = create_hf_model(
        AutoModelForCausalLM, base_model_name_or_path, tokenizer,
        disable_dropout=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        forbid_vocab_growth=True, device_map=device_map)

    if method in LORA_METHODS:
        targets = resolve_targets(meta["targets"])
        if method == "olora":
            attach_olora_targets(
                model, targets, meta["r"], meta["alpha"],
                num_tasks=meta["num_tasks"], dropout=meta.get("dropout", 0.0))
        else:
            attach_seq_lora_targets(model, targets, meta["r"], meta["alpha"],
                                    meta.get("dropout", 0.0))
    elif method in MOE_METHODS:
        scope_name = meta["moe_scope"]
        if method == "lifelong_moe":
            attach_shared_path(
                model, scope_name, meta.get("shared_targets", "none"),
                meta["r"], meta["alpha"], meta.get("dropout", 0.0))
        scope = build_scope(scope_name)
        # ``attach`` reads the same argument names the trainer used, so hand it
        # a view of the metadata rather than re-listing every field here.
        scope.attach(model, _MetaArgs(meta))
        scope.add_experts(model, meta["num_experts"])
    else:
        raise ValueError(f"unknown Table-1 method in metadata: {method}")

    state = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"),
                       map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected Table-1 checkpoint keys: {unexpected[:5]}")
    adapter_missing = [key for key in missing
                       if any(fragment in key for fragment in
                              (".lora.", ".adapters.", ".experts.", ".router."))]
    if adapter_missing:
        raise RuntimeError(
            f"Table-1 checkpoint is missing adapter keys: {adapter_missing[:5]}")

    if method == "olora":
        current = int(meta.get("current_task", meta["num_tasks"] - 1))
        set_olora_task(model, current)
        should_merge = (meta.get("merged", False) if merge_olora is None
                        else merge_olora)
        if should_merge:
            merge_olora_into_base(model, upto_task=current)
            for module in model.modules():
                if hasattr(module, "adapters") and hasattr(module, "active_task"):
                    module.adapters = torch.nn.ModuleList()
                    module.active_task = -1

    if device_map is None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model, meta


class _MetaArgs:
    """Attribute view over tab1_meta.json for the scope ``attach`` signature."""

    def __init__(self, meta: Dict[str, object]):
        self.lora_moe_rank = meta["r"]
        self.lora_moe_alpha = meta["alpha"]
        self.lora_moe_dropout = meta.get("dropout", 0.0)
        self.top_k = meta["top_k"]
        self.moe_aux_loss_coeff = meta.get("aux_loss_coeff", 0.0)
        self.moe_z_loss_coeff = meta.get("z_loss_coeff", 0.0)
        self.routing_weight_mode = meta["routing_weight_mode"]
