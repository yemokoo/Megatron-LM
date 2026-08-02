"""
Growing FFN LoRA-MoE for continual learning.

Each transformer layer's FFN (gate/up/down proj) keeps its original dense
weights frozen forever, and gets a token-routed mixture of small LoRA
"experts" bolted on top. At the start of every new task, `experts_per_task`
brand-new experts are appended (to every layer, simultaneously) and the
router's output grows to cover them. Only the just-added experts + the
router train on the new task's data (phase 1); afterwards, ALL experts
(including the brand-new ones) are frozen and only the router is re-tuned
on a capped pool of all seen-task fixed subsets, including current data
(phase 2), so routing stays calibrated as the expert pool grows.

Attention is untouched -- no LoRA there, dense and frozen like the rest of
the backbone.
"""
import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import get_scheduler

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:  # pragma: no cover - older PyTorch fallback
    FlopCounterMode = None


def _gqa_sdpa_flop(query_shape, key_shape, value_shape, *args,
                   out_shape=None, **kwargs):
    """Count SDPA FLOPs for regular MHA and grouped-query attention."""
    batch, query_heads, query_tokens, query_dim = query_shape
    key_batch, key_heads, key_tokens, key_dim = key_shape
    value_batch, value_heads, value_tokens, value_dim = value_shape
    if not (batch == key_batch == value_batch
            and key_heads == value_heads
            and query_heads % key_heads == 0
            and query_dim == key_dim
            and key_tokens == value_tokens):
        raise ValueError(
            "unsupported SDPA shapes for FLOP counting: "
            f"q={query_shape} k={key_shape} v={value_shape}")
    return (
        2 * batch * query_heads * query_tokens * key_tokens * query_dim
        + 2 * batch * query_heads * query_tokens * key_tokens * value_dim)


def _flop_counter_custom_mapping():
    if FlopCounterMode is None:
        return {}
    aten = torch.ops.aten
    return {
        aten._scaled_dot_product_efficient_attention: _gqa_sdpa_flop,
        aten._scaled_dot_product_flash_attention: _gqa_sdpa_flop,
        aten._scaled_dot_product_cudnn_attention: _gqa_sdpa_flop,
    }

from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_optimizer_grouped_parameters
from utils.data.data_collator import (DataCollator, SLoRATraceDataCollator,
                                      PreTokenizedSLoRATraceDataCollator)


class RepeatedSubsetDataset(Dataset):
    """Deterministic exposure stream drawn only from one fixed task subset."""

    def __init__(self, subset, exposure_indices):
        self.subset = subset
        self.exposure_indices = list(exposure_indices)

    def __len__(self):
        return len(self.exposure_indices)

    def __getitem__(self, index):
        return self.subset[self.exposure_indices[index]]


class LoRAPair(nn.Module):
    """One rank-r LoRA delta: out = (x @ A^T) @ B^T * (alpha / r)."""

    def __init__(self, in_features, out_features, r, alpha, dropout=0.0):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)  # B stays zero -> new expert starts as a no-op

    def forward(self, x):
        return (self.dropout(x) @ self.A.T) @ self.B.T * self.scaling


def _make_expert(hidden_size, intermediate_size, r, alpha, dropout=0.0):
    return nn.ModuleDict({
        "gate": LoRAPair(hidden_size, intermediate_size, r, alpha, dropout),
        "up":   LoRAPair(hidden_size, intermediate_size, r, alpha, dropout),
        "down": LoRAPair(intermediate_size, hidden_size, r, alpha, dropout),
    })


class LoRAMoEMLP(nn.Module):
    """Wraps a frozen dense FFN (base_mlp) with a growing pool of routed LoRA experts."""

    def __init__(self, base_mlp, r, alpha, top_k, aux_loss_coeff, z_loss_coeff,
                 routing_weight_mode="topk_softmax", dropout=0.0):
        super().__init__()
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        self.hidden_size = base_mlp.gate_proj.in_features
        self.intermediate_size = base_mlp.gate_proj.out_features
        self.r, self.alpha = r, alpha
        self.dropout = dropout
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        if routing_weight_mode not in {"topk_softmax", "full_softmax"}:
            raise ValueError(f"unknown routing_weight_mode: {routing_weight_mode}")
        self.routing_weight_mode = routing_weight_mode

        self.experts = nn.ModuleList()
        self.router = None  # nn.Linear(hidden_size, num_experts, bias=False); grown via add_experts()
        self._last_moe_loss = None
        self._router_token_mask = None
        # v2 KD uses the expanded module as its own frozen pre-expansion teacher.
        # Limiting the active prefix exactly recovers the old expert/router shape
        # without keeping a second 8B backbone in memory.
        self._active_expert_count = None

    @property
    def num_experts(self):
        return len(self.experts)

    def add_experts(self, n):
        device = self.base_mlp.gate_proj.weight.device
        dtype = self.base_mlp.gate_proj.weight.dtype
        for _ in range(n):
            self.experts.append(
                _make_expert(
                    self.hidden_size, self.intermediate_size, self.r,
                    self.alpha, self.dropout).to(device=device, dtype=dtype)
            )

        old_n = self.router.out_features if self.router is not None else 0
        new_router = nn.Linear(self.hidden_size, old_n + n, bias=False).to(device=device, dtype=dtype)
        if self.router is not None:
            with torch.no_grad():
                new_router.weight[:old_n] = self.router.weight
        self.router = new_router

    def forward(self, x):
        active_experts = (
            self.num_experts
            if self._active_expert_count is None
            else int(self._active_expert_count)
        )
        if active_experts == 0:
            self._last_moe_loss = x.new_zeros(())
            return self.base_mlp(x)
        if not 0 < active_experts <= self.num_experts:
            raise ValueError(
                f"active expert prefix {active_experts} is invalid for "
                f"{self.num_experts} experts")

        gate = self.base_mlp.gate_proj(x)
        up = self.base_mlp.up_proj(x)

        # Slice before softmax/top-k: this is the exact pre-expansion router,
        # not an expanded distribution with the new rows merely masked later.
        flat_logits = self.router(x)[..., :active_experts].reshape(
            -1, active_experts)                                            # [N, E]
        k = min(self.top_k, active_experts)
        topk_val, topk_idx = flat_logits.topk(k, dim=-1)
        full_probs = None
        if self.routing_weight_mode == "full_softmax":
            # Gather probabilities from the full router distribution without
            # renormalizing over the selected experts. For top-k=1 the dispatch
            # weight therefore does not collapse to the constant 1, and the LM
            # objective can train the router.
            full_probs = F.softmax(flat_logits, dim=-1, dtype=torch.float)
            topk_weight = full_probs.gather(-1, topk_idx)
        else:
            # Legacy behavior retained for loading existing checkpoints.
            topk_weight = F.softmax(topk_val, dim=-1)

        valid_token_mask = self._router_token_mask
        if valid_token_mask is not None:
            valid_token_mask = valid_token_mask.reshape(-1).to(
                device=flat_logits.device, dtype=torch.bool)
            if valid_token_mask.numel() != flat_logits.shape[0]:
                raise ValueError(
                    f"router mask has {valid_token_mask.numel()} tokens but "
                    f"hidden states have {flat_logits.shape[0]}")

        # Dispatch only the tokens selected for each expert. The previous path
        # applied every active expert to the complete [batch, seq] tensor and
        # multiplied non-routed positions by zero afterwards. With a long prefill,
        # nearly every expert is active somewhere, so that turned sparse top-k
        # routing into E dense LoRA forwards. It also used torch.any() in Python,
        # introducing a GPU synchronization for every expert in every layer.
        flat_x = x.reshape(-1, x.shape[-1])
        flat_gate_delta = gate.new_zeros((flat_x.shape[0], gate.shape[-1]))
        flat_up_delta = up.new_zeros((flat_x.shape[0], up.shape[-1]))
        routes = [
            torch.where(topk_idx.transpose(0, 1) == expert_index)
            for expert_index in range(active_experts)
        ]
        for e, expert in enumerate(self.experts[:active_experts]):
            slot, token_idx = routes[e]
            if valid_token_mask is not None:
                keep = valid_token_mask[token_idx]
                slot, token_idx = slot[keep], token_idx[keep]
            if token_idx.numel() == 0:
                continue
            routed_x = flat_x[token_idx]
            weight = topk_weight[token_idx, slot, None].to(routed_x.dtype)
            flat_gate_delta.index_add_(
                0, token_idx, expert["gate"](routed_x) * weight)
            flat_up_delta.index_add_(
                0, token_idx, expert["up"](routed_x) * weight)

        gate_delta = flat_gate_delta.reshape_as(gate)
        up_delta = flat_up_delta.reshape_as(up)
        h = F.silu(gate + gate_delta) * (up + up_delta)
        down = self.base_mlp.down_proj(h)

        flat_h = h.reshape(-1, h.shape[-1])
        flat_down_delta = down.new_zeros((flat_h.shape[0], down.shape[-1]))
        for e, expert in enumerate(self.experts[:active_experts]):
            slot, token_idx = routes[e]
            if valid_token_mask is not None:
                keep = valid_token_mask[token_idx]
                slot, token_idx = slot[keep], token_idx[keep]
            if token_idx.numel() == 0:
                continue
            routed_h = flat_h[token_idx]
            weight = topk_weight[token_idx, slot, None].to(routed_h.dtype)
            flat_down_delta.index_add_(
                0, token_idx, expert["down"](routed_h) * weight)
        down = down + flat_down_delta.reshape_as(down)

        # Generation never consumes the auxiliary router loss. Avoid a full
        # softmax, dense hard-routing mask and z-loss on every decode token.
        self._last_moe_loss = (
            self._router_loss(flat_logits, topk_idx, full_probs=full_probs,
                              token_mask=valid_token_mask)
            if self.training else None)
        return down

    def _router_loss(self, flat_logits, topk_idx, full_probs=None, token_mask=None):
        # Switch-Transformer aux loss + router z-loss, same formulas/coefficients
        # as LLM-continual-learning's Megatron router (moe_utils.py).
        if token_mask is not None:
            flat_logits = flat_logits[token_mask]
            topk_idx = topk_idx[token_mask]
            if full_probs is not None:
                full_probs = full_probs[token_mask]
        num_tokens, num_experts = flat_logits.shape
        if num_tokens == 0:
            return flat_logits.sum() * 0.0
        if full_probs is None:
            full_probs = F.softmax(flat_logits, dim=-1, dtype=torch.float)
        tokens_per_expert = torch.bincount(
            topk_idx.reshape(-1), minlength=num_experts).to(full_probs.dtype)
        aggregated_probs_per_expert = full_probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * self.aux_loss_coeff /
            (num_tokens * num_tokens * topk_idx.shape[-1])
        )
        z_loss = torch.mean(torch.square(
            torch.logsumexp(flat_logits.float(), dim=-1))) * self.z_loss_coeff
        return aux_loss + z_loss


def attach_lora_moe(model, r, alpha, top_k, aux_loss_coeff, z_loss_coeff,
                    routing_weight_mode="topk_softmax", dropout=0.0):
    """Replace every decoder layer's .mlp with an (initially expert-less) LoRAMoEMLP.

    Then freeze the WHOLE backbone -- attention, embeddings, norms, lm_head and
    every FFN's dense base_mlp stay frozen forever. Only the routed LoRA experts
    and the routers ever train; both are created later with requires_grad=True and
    are (de)activated per phase by freeze_lora_moe_experts / freeze_lora_moe_routers.
    Without this, get_optimizer_grouped_parameters (which filters on requires_grad)
    would happily train the entire backbone every task -- the exact forgetting this
    method is designed to avoid, and it would make phase-2 not router-only.
    """
    layers = model.model.layers
    for layer in layers:
        layer.mlp = LoRAMoEMLP(
            layer.mlp, r, alpha, top_k, aux_loss_coeff, z_loss_coeff,
            routing_weight_mode=routing_weight_mode, dropout=dropout)
    for p in model.parameters():  # no experts/routers exist yet -> freezes pure backbone
        p.requires_grad = False
    return model


def _lora_moe_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRAMoEMLP)]


@contextmanager
def limit_lora_moe_experts(model, active_expert_count):
    """Temporarily expose only the pre-expansion expert/router prefix.

    v2 calls this under ``torch.no_grad()`` to obtain a fixed teacher target.
    Old experts and old router rows remain frozen during KD, so this teacher is
    identical across every KD optimizer step without duplicating the backbone.
    """
    layers = _lora_moe_layers(model)
    previous = [layer._active_expert_count for layer in layers]
    for layer in layers:
        if not 0 <= active_expert_count <= layer.num_experts:
            raise ValueError(
                f"active_expert_count={active_expert_count}, "
                f"layer experts={layer.num_experts}")
        layer._active_expert_count = active_expert_count
    try:
        yield
    finally:
        for layer, value in zip(layers, previous):
            layer._active_expert_count = value


def add_experts_to_all_layers(model, n):
    for layer in _lora_moe_layers(model):
        layer.add_experts(n)


def freeze_lora_moe_experts(model, trainable_expert_indices=None):
    """trainable_expert_indices=None freezes every expert; else only those indices train."""
    for layer in _lora_moe_layers(model):
        for i, expert in enumerate(layer.experts):
            requires_grad = trainable_expert_indices is not None and i in trainable_expert_indices
            for p in expert.parameters():
                p.requires_grad = requires_grad


def freeze_lora_moe_routers(model, trainable):
    for layer in _lora_moe_layers(model):
        if layer.router is not None:
            for p in layer.router.parameters():
                p.requires_grad = trainable


def collect_moe_losses(model):
    total = None
    for layer in _lora_moe_layers(model):
        if layer._last_moe_loss is not None:
            total = layer._last_moe_loss if total is None else total + layer._last_moe_loss
    return total


def set_router_token_mask(model, attention_mask):
    """Mask padding in LoRA dispatch and router aux/z objectives.

    Keep the mask installed through backward because gradient checkpointing
    re-runs layer forwards during backward; clear it afterwards with None.
    """
    for layer in _lora_moe_layers(model):
        layer._router_token_mask = attention_mask


class Ours_LoRA_MoE(CL_Base_Model):
    """Per task: (1) grow experts_per_task new experts + train them with the router
    on the new task, then (2) freeze every expert and retune only the router on
    a capped pool of all seen-task fixed subsets, including current data.

    self.raw_model is the plain nn.Module (survives across tasks; add_experts/
    freeze mutate it directly). self.model is a DistributedDataParallel wrapper
    around it (or raw_model itself if not distributed) -- rebuilt via
    _reinit_engine() at the start of EVERY phase, since (a) growing experts
    creates brand-new nn.Parameters a stale DDP wrapper/optimizer never
    registered, and (b) get_optimizer_grouped_parameters filters by
    p.requires_grad, so each phase needs its own optimizer built from the
    freeze pattern that's active *at that moment*. (Previously used
    deepspeed.initialize() here, but repeatedly re-initializing a DeepSpeed
    engine on the same model leaks GPU memory without bound -- confirmed via a
    synthetic repro that grew every re-init regardless of ZeRO stage,
    destroy()+gc.collect(), or bucket size, and eventually OOM'd mid-run. Our
    trainable set is tiny (LoRA experts + router) so ZeRO sharding buys nothing
    anyway; plain DDP + AdamW re-created the same way showed zero growth across
    repeated re-inits.)
    """

    # Save only the grown params (LoRA experts + router), not the frozen 16GB
    # backbone -- the eval loader rebuilds the base from base_model_name_or_path
    # and loads these on top. ".mlp.experts." / ".mlp.router." exclude the frozen
    # dense ".mlp.base_mlp." and everything else in the backbone.
    save_key_substrings = [".mlp.experts.", ".mlp.router."]

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.raw_model = model
        self._dist_initialized = False
        self._fixed_task_subsets = {}
        self._fixed_task_subset_indices = {}
        self._replay_manifest = None
        replay_manifest_path = getattr(args, "replay_manifest_path", "")
        if replay_manifest_path:
            with open(replay_manifest_path, encoding="utf-8") as handle:
                self._replay_manifest = json.load(handle)
            if not isinstance(self._replay_manifest.get("tasks"), dict):
                raise ValueError(
                    f"invalid replay manifest: {replay_manifest_path}")
            self._replay_manifest["_path"] = replay_manifest_path

        self._active_task_workload = None
        self._workload_records = self._load_workload_records()

    def _load_workload_records(self):
        path = os.path.join(self.args.output_dir, "training_workload.json")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as handle:
                return list(json.load(handle).get("tasks", []))
        except (OSError, ValueError, TypeError):
            return []

    @staticmethod
    def _empty_workload_role():
        return {
            "input_sample_exposures": 0,
            "input_nonpad_token_exposures": 0,
            "forward_sample_instances": 0,
            "forward_nonpad_token_instances": 0,
            "forward_microbatches": 0,
            "backward_microbatches": 0,
        }

    def _count_workload_batch(self, role, batch, forward_passes=1,
                              backward_passes=1):
        if self._active_task_workload is None:
            return
        mask = batch.get("attention_mask")
        if mask is None:
            samples = int(batch["input_ids"].shape[0])
            tokens = int(batch["input_ids"].numel())
        else:
            samples = int(mask.shape[0])
            tokens = int(mask.detach().sum().item())
        metrics = self._active_task_workload["roles"].setdefault(
            role, self._empty_workload_role())
        metrics["input_sample_exposures"] += samples
        metrics["input_nonpad_token_exposures"] += tokens
        metrics["forward_sample_instances"] += samples * forward_passes
        metrics["forward_nonpad_token_instances"] += tokens * forward_passes
        metrics["forward_microbatches"] += forward_passes
        metrics["backward_microbatches"] += backward_passes

    def _count_workload_update(self):
        if self._active_task_workload is not None:
            self._active_task_workload["optimizer_updates"] += 1

    def _reduce_scalar(self, value, operation="sum"):
        if not torch.distributed.is_initialized():
            return value
        tensor = torch.tensor(
            float(value), dtype=torch.float64,
            device=torch.device("cuda", self.args.local_rank))
        reduce_op = (torch.distributed.ReduceOp.MAX if operation == "max"
                     else torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(tensor, op=reduce_op)
        return tensor.item()

    def _finalize_task_workload(self, elapsed_seconds, local_flops,
                                flop_counter_status):
        workload = self._active_task_workload
        reduced_roles = {}
        for role in sorted(workload["roles"]):
            reduced_roles[role] = {
                key: int(self._reduce_scalar(value))
                for key, value in workload["roles"][role].items()
            }
        workload["roles"] = reduced_roles
        workload["optimizer_updates"] = int(self._reduce_scalar(
            workload["optimizer_updates"], operation="max"))
        workload["training_wall_time_seconds"] = self._reduce_scalar(
            elapsed_seconds, operation="max")
        workload["counted_operator_flops_global"] = (
            int(self._reduce_scalar(local_flops))
            if local_flops is not None else None)
        workload["flop_counter_status"] = flop_counter_status
        workload["flop_counter_scope"] = (
            "PyTorch-supported operator FLOPs summed across ranks; "
            "communication and unsupported operators excluded")
        return workload

    def _write_workload_records(self):
        if self.args.global_rank != 0:
            return
        by_round = {
            int(record["round"]): record for record in self._workload_records}
        tasks = [by_round[index] for index in sorted(by_round)]
        role_totals = {}
        for task in tasks:
            for role, metrics in task["roles"].items():
                total = role_totals.setdefault(role, self._empty_workload_role())
                for key, value in metrics.items():
                    total[key] += value
        flops = [task.get("counted_operator_flops_global") for task in tasks]
        payload = {
            "schema_version": 1,
            "method": f"ours_lora_moe_{self.args.training_version}",
            "world_size": (torch.distributed.get_world_size()
                           if torch.distributed.is_initialized() else 1),
            "tasks": tasks,
            "totals": {
                "roles": role_totals,
                "optimizer_updates": sum(
                    task["optimizer_updates"] for task in tasks),
                "training_wall_time_seconds": sum(
                    task["training_wall_time_seconds"] for task in tasks),
                "counted_operator_flops_global": (
                    sum(flops) if all(value is not None for value in flops)
                    else None),
            },
        }
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, "training_workload.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)

    def train_continual(self):
        """Train every task and persist task/global compute and time totals."""
        for i_task, task in enumerate(self.train_task_list):
            if i_task < getattr(self.args, "start_task", 0):
                print_rank_0(
                    f"Skipping completed task {i_task}: {task}",
                    self.args.global_rank)
                continue
            epochs = int(self.args.num_train_epochs[i_task])
            self._active_task_workload = {
                "round": i_task,
                "task": task,
                "epochs": epochs,
                "router_replay_exposure_budget_global": (
                    self.args.router_replay_exposure_samples),
                "roles": {},
                "optimizer_updates": 0,
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            track_flops = not self.args.disable_training_flop_counter
            flop_counter = (
                FlopCounterMode(
                    display=False,
                    custom_mapping=_flop_counter_custom_mapping())
                if track_flops and FlopCounterMode is not None else None)
            flop_status = ("enabled" if flop_counter is not None else
                           "unavailable" if track_flops else "disabled")
            with (flop_counter if flop_counter is not None else nullcontext()):
                self.train_one_task(task, i_task, epochs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            local_flops = (flop_counter.get_total_flops()
                           if flop_counter is not None else None)
            record = self._finalize_task_workload(
                elapsed, local_flops, flop_status)

            save_started = time.perf_counter()
            self.save_model(i_task)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record["checkpoint_save_wall_time_seconds"] = self._reduce_scalar(
                time.perf_counter() - save_started, operation="max")
            self._workload_records = [
                item for item in self._workload_records
                if int(item["round"]) != i_task]
            self._workload_records.append(record)
            self._write_workload_records()
            self._active_task_workload = None

    def _reinit_engine(self, num_training_steps, learning_rate=None):
        args = self.args
        if self._dist_initialized:
            del self.model
            torch.cuda.empty_cache()
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.raw_model, args.weight_decay)
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate if learning_rate is None else learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon)
        warmup_steps = args.num_warmup_steps
        if getattr(args, "warmup_ratio", 0.0) > 0:
            warmup_steps = math.ceil(
                num_training_steps * args.warmup_ratio)
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler_type, self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, num_training_steps))
        if args.local_rank != -1:
            self.model = DDP(self.raw_model, device_ids=[args.local_rank],
                             output_device=args.local_rank, find_unused_parameters=True,
                             broadcast_buffers=False, gradient_as_bucket_view=True)
        else:
            self.model = self.raw_model
        self._dist_initialized = True

    def _set_grad_ckpt(self, enable):
        """Toggle gradient checkpointing on raw_model for the upcoming phase. Pure
        memory/compute tradeoff -- weights are identical either way. Always clears any
        prior input-require-grads hook first so repeated ON transitions across tasks
        don't stack orphaned hooks on the input embeddings."""
        m = self.raw_model
        if getattr(m, "_require_grads_hook", None) is not None:
            m.disable_input_require_grads()
            m._require_grads_hook = None
        if enable:
            # Frozen backbone -> embedding output has requires_grad=False, so a
            # checkpointed segment has no grad_fn; force it on via the input hook.
            m.enable_input_require_grads()
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            m.gradient_checkpointing_disable()
        if self.args.global_rank == 0:
            print_rank_0(f"  [grad_ckpt] {'ON' if enable else 'off'}", self.args.global_rank)

    def _optimizer_update_count(self, dataloader, epochs):
        """Scheduler length in optimizer updates, matching epoch-local accumulation."""
        accum = max(1, self.args.gradient_accumulation_steps)
        return max(1, int(epochs) * math.ceil(len(dataloader) / accum))

    def _set_phase_gradient_accumulation(self, batch_size, phase):
        world_size = (torch.distributed.get_world_size()
                      if torch.distributed.is_initialized() else 1)
        denominator = int(batch_size) * world_size
        global_batch = int(self.args.effective_global_batch)
        if global_batch % denominator != 0:
            raise ValueError(
                f"{phase}: effective global batch {global_batch} is not "
                f"divisible by micro-batch {batch_size} * world size {world_size}")
        accumulation = global_batch // denominator
        self.args.gradient_accumulation_steps = accumulation
        print_rank_0(
            f"  [batch contract] {phase}: micro_batch={batch_size} "
            f"grad_accum={accumulation} effective_global_batch={global_batch}",
            self.args.global_rank)
        return accumulation

    @staticmethod
    def _allocate_memory_counts(lengths, total, mode):
        """Allocate one global exposure budget without growing with task count."""
        if not lengths or total <= 0:
            return [0] * len(lengths)
        total = int(total)
        if mode == "proportional":
            raw = [total * length / sum(lengths) for length in lengths]
        elif mode == "equal_task":
            raw = [total / len(lengths)] * len(lengths)
        else:
            raise ValueError(f"unknown replay distribution: {mode}")
        counts = [int(value) for value in raw]
        remainder = total - sum(counts)
        order = sorted(
            range(len(lengths)),
            key=lambda index: (raw[index] - int(raw[index]), lengths[index]),
            reverse=True)
        for index in order[:remainder]:
            counts[index] += 1
        return counts

    def _fixed_subset_seed(self):
        configured = self.args.replay_subset_seed
        return self.args.seed if configured < 0 else configured

    def _ensure_fixed_task_subset(self, task):
        """Create one persistent deterministic 1%/10% subset for a task."""
        if task in self._fixed_task_subsets:
            return self._fixed_task_subsets[task]
        task_names = list(self.train_task_list)
        task_index = task_names.index(task)
        dataset = self.train_task_list[task].dataset
        unique_samples = max(
            1, min(len(dataset), round(
                len(dataset) * self.args.replay_subset_ratio)))
        if getattr(self, "_replay_manifest", None) is not None:
            entry = self._replay_manifest["tasks"].get(task)
            if entry is None:
                raise KeyError(f"replay manifest has no task entry for {task}")
            indices = [int(index) for index in entry.get("indices", [])]
            if entry.get("source_samples") != len(dataset):
                raise ValueError(
                    f"replay source size mismatch for {task}: "
                    f"{entry.get('source_samples')} != {len(dataset)}")
            if len(indices) != unique_samples or len(set(indices)) != len(indices):
                raise ValueError(
                    f"replay manifest for {task} must contain "
                    f"{unique_samples} unique indices, got {len(indices)}")
            if any(index < 0 or index >= len(dataset) for index in indices):
                raise ValueError(f"replay manifest has out-of-range index for {task}")
            seed = int(entry["seed"])
        else:
            seed = self._fixed_subset_seed() + task_index * 1009
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(
                len(dataset), generator=generator)[:unique_samples].tolist()
        subset = Subset(dataset, indices)
        self._fixed_task_subsets[task] = subset
        self._fixed_task_subset_indices[task] = indices

        if self.args.global_rank == 0:
            digest = hashlib.sha256(
                ",".join(map(str, indices)).encode("utf-8")).hexdigest()
            output_dir = os.path.join(
                self.args.output_dir, "fixed_replay_memory")
            os.makedirs(output_dir, exist_ok=True)
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task)
            metadata = {
                "schema_version": 1,
                "task_index": task_index,
                "task": task,
                "source_samples": len(dataset),
                "subset_ratio": self.args.replay_subset_ratio,
                "unique_samples": unique_samples,
                "seed": seed,
                "indices_sha256": digest,
                "manifest_path": (
                    getattr(self, "_replay_manifest", None) or {}).get("_path"),
                "indices": indices,
                "available_from_next_task_for_v2_past_replay": True,
            }
            with open(os.path.join(
                    output_dir, f"task_{task_index}_{safe_task}.json"),
                    "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        return subset

    @staticmethod
    def _deterministic_exposure_indices(unique_samples, exposure_samples, seed):
        """Repeat/shuffle a fixed subset until exactly exposure_samples are drawn."""
        if unique_samples <= 0 or exposure_samples <= 0:
            return []
        generator = torch.Generator().manual_seed(seed)
        indices = []
        while len(indices) < exposure_samples:
            permutation = torch.randperm(
                unique_samples, generator=generator).tolist()
            indices.extend(permutation[:exposure_samples - len(indices)])
        return indices

    def _build_fixed_memory_loader(self, task_names, round_index, phase,
                                   exposure_samples=None, batch_size=None):
        """Build replay only from accumulated task-fixed subsets.

        ``exposure_samples=None`` exposes each unique memory item once; callers
        such as joint replay may cycle that loader. An integer builds an exact
        total exposure stream distributed across tasks, repeating small subsets
        as needed without increasing the total budget.
        """
        if not task_names:
            return None
        subsets = [self._ensure_fixed_task_subset(task) for task in task_names]
        unique_counts = [len(subset) for subset in subsets]
        if exposure_samples is None:
            exposure_counts = list(unique_counts)
        else:
            exposure_counts = self._allocate_memory_counts(
                unique_counts, int(exposure_samples),
                self.args.replay_distribution)

        phase_seed = sum(ord(character) for character in phase)
        stream_datasets = []
        plan_tasks = []
        base_seed = self._fixed_subset_seed() + round_index * 100003 + phase_seed
        for offset, (task, subset, unique_count, exposure_count) in enumerate(
                zip(task_names, subsets, unique_counts, exposure_counts)):
            if exposure_count > 0:
                exposure_indices = self._deterministic_exposure_indices(
                    unique_count, exposure_count, base_seed + offset * 1009)
                stream_datasets.append(RepeatedSubsetDataset(
                    subset, exposure_indices))
            plan_tasks.append({
                "task": task,
                "unique_samples": unique_count,
                "exposure_samples": exposure_count,
            })
        if not stream_datasets:
            return None
        combined = ConcatDataset(stream_datasets)
        if batch_size is None or batch_size <= 0:
            batch_size = min(self.args.batch_by_task[task] for task in task_names)
        max_train_len = getattr(self.args, "max_train_len", 0)
        if getattr(self.args, "train_format", "raw_answer") == "slora_chat_full":
            if getattr(self.args, "use_pretokenized_train_cache", False):
                collator = PreTokenizedSLoRATraceDataCollator(self.tokenizer)
            else:
                collator = SLoRATraceDataCollator(
                    self.tokenizer, max_length=(max_train_len or (
                        self.args.max_prompt_len + self.args.max_ans_len)))
        else:
            collator = DataCollator(
                self.tokenizer, padding="longest",
                max_prompt_len=(max_train_len or self.args.max_prompt_len),
                max_ans_len=(0 if max_train_len else self.args.max_ans_len),
                pad_to_multiple_of=8, inference=False)
        sampler = (
            RandomSampler(combined)
            if self.args.local_rank == -1
            else DistributedSampler(
                combined, shuffle=True, seed=base_seed))
        loader = DataLoader(
            combined, collate_fn=collator, sampler=sampler,
            batch_size=batch_size, num_workers=4, pin_memory=True)

        if self.args.global_rank == 0:
            plan_dir = os.path.join(self.args.output_dir, "replay_plans")
            os.makedirs(plan_dir, exist_ok=True)
            plan = {
                "schema_version": 1,
                "round": round_index,
                "phase": phase,
                "subset_ratio_per_task": self.args.replay_subset_ratio,
                "distribution": self.args.replay_distribution,
                "unique_pool_samples": sum(unique_counts),
                "planned_exposure_samples": sum(exposure_counts),
                "batch_size_per_rank": batch_size,
                "tasks": plan_tasks,
            }
            with open(os.path.join(
                    plan_dir, f"round_{round_index}_{phase}.json"),
                    "w", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2)
        return loader

    @staticmethod
    def _snapshot_old_router_rows(model, old_expert_count):
        return [
            layer.router.weight[:old_expert_count].detach().clone()
            for layer in _lora_moe_layers(model)
        ]

    @staticmethod
    def _freeze_old_router_row_update(model, snapshots, old_expert_count):
        """Keep a router prefix bit-identical across an optimizer update."""
        for layer, snapshot in zip(_lora_moe_layers(model), snapshots):
            if layer.router.weight.grad is not None:
                layer.router.weight.grad[:old_expert_count].zero_()
            with torch.no_grad():
                layer.router.weight[:old_expert_count].copy_(snapshot)

    def train_one_task(self, task, i_task, epochs):
        device = torch.device("cuda", self.args.local_rank) if self.args.local_rank != -1 else torch.device("cuda")
        args = self.args

        # Each task contributes one immutable 1%/10% memory subset. v1 may use
        # the current task immediately in its post-training router phase; v2
        # exposes it only to later tasks as past memory.
        self._ensure_fixed_task_subset(task)
        old_expert_count = len(_lora_moe_layers(self.raw_model)[0].experts)
        add_experts_to_all_layers(self.raw_model, args.experts_per_task)
        new_indices = set(range(
            old_expert_count, old_expert_count + args.experts_per_task))
        old_router_rows = (
            self._snapshot_old_router_rows(self.raw_model, old_expert_count)
            if old_expert_count > 0 else None)

        # Phase 1: checkpointing per the task's survey (short=off/fast, long=on).
        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} phase1")
        self._set_grad_ckpt(task in getattr(args, "ckpt_tasks", set()))
        freeze_lora_moe_experts(self.raw_model, trainable_expert_indices=new_indices)
        freeze_lora_moe_routers(self.raw_model, trainable=True)
        self._reinit_engine(self._optimizer_update_count(
            self.train_task_list[task], epochs))
        self._run_epochs(
            self.train_task_list[task], epochs, device,
            f"{task} [phase1 new-expert+new-router-row]",
            frozen_router_prefix=(old_expert_count, old_router_rows)
            if old_router_rows is not None else None)

        if args.router_retune_epochs > 0:
            replay_loader = self._build_replay_loader(i_task)
            if replay_loader is not None:
                # v1 is a two-phase method: after the current-task phase, its
                # router retune uses every seen fixed subset, including the
                # current task. A long sequence may land in any batch, so
                # checkpoint the whole router-retune phase.
                self._set_grad_ckpt(True)
                freeze_lora_moe_experts(
                    self.raw_model, trainable_expert_indices=None)
                freeze_lora_moe_routers(self.raw_model, trainable=True)
                self._set_phase_gradient_accumulation(
                    replay_loader.batch_size, f"{task} phase2 router retune")
                # The replay stream itself contains exactly the configured
                # global exposure budget, so it is consumed once. For a first
                # 5,000-sample task this is the 50-item 1% subset repeated 20
                # times, i.e. exactly 1,000 router-finetune exposures.
                self._reinit_engine(self._optimizer_update_count(
                    replay_loader, 1))
                self._run_epochs(
                    replay_loader, 1, device,
                    f"{task} [phase2 router retune exact seen replay]")

    def _run_epochs(self, dataloader, epochs, device, phase_name,
                    frozen_router_prefix=None):
        args = self.args
        total_steps = epochs * len(dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(f"{phase_name}: epoch {epoch+1}/{epochs}, {len(dataloader)} steps",
                        args.global_rank)
            self.model.train()
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(dataloader):
                workload_role = (
                    "router_replay" if "phase2 router retune" in phase_name
                    else "new_task")
                self._count_workload_batch(workload_role, batch)
                del batch['sources']
                batch = to_device(batch, device)
                accum_steps = args.gradient_accumulation_steps
                window_start = (step // accum_steps) * accum_steps
                window_size = min(accum_steps, len(dataloader) - window_start)
                is_window_end = (step - window_start + 1) == window_size
                sync_context = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) and not is_window_end
                    else nullcontext())
                set_router_token_mask(self.raw_model, batch.get("attention_mask"))
                try:
                    with sync_context:
                        outputs = self.model(**batch, use_cache=False)
                        moe_loss = collect_moe_losses(self.raw_model)
                        loss = outputs.loss if moe_loss is None else outputs.loss + moe_loss
                        (loss / window_size).backward()
                finally:
                    set_router_token_mask(self.raw_model, None)
                if args.global_rank == 0:
                    progress_bar.update(1)
                    if (step % args.loss_log_interval == 0 or
                            step + 1 == len(dataloader)):
                        progress_bar.set_description(
                            f"{phase_name} | epoch {epoch+1} step {step} "
                            f"loss {loss.detach().float().cpu().item():.4f}",
                            refresh=False)
                if is_window_end:
                    if frozen_router_prefix is not None:
                        old_count, snapshots = frozen_router_prefix
                        self._freeze_old_router_row_update(
                            self.raw_model, snapshots, old_count)
                    trainable_params = [
                        p for p in self.raw_model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    self.optimizer.step()
                    if frozen_router_prefix is not None:
                        self._freeze_old_router_row_update(
                            self.raw_model, snapshots, old_count)
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()

    def save_model(self, round):
        # Persist the HF-format checkpoint (base_mlp + experts + router are all in
        # the state_dict) plus a lora_moe_meta.json so load_lora_moe_checkpoint can
        # rebuild the exact expert-pool size / hyperparameters at eval time.
        super().save_model(round)
        if self.args.global_rank == 0:
            output_dir = os.path.join(self.args.output_dir, str(round))
            extra = {
                "training_version": getattr(self.args, "training_version", "v1"),
                "experts_per_task": self.args.experts_per_task,
                "training_profile": {
                    "format": self.args.train_format,
                    "max_length": self.args.max_train_len or (
                        self.args.max_prompt_len + self.args.max_ans_len),
                    "adam_beta1": self.args.adam_beta1,
                    "adam_beta2": self.args.adam_beta2,
                    "adam_epsilon": self.args.adam_epsilon,
                },
                "replay_memory": {
                    "subset_ratio_per_task": self.args.replay_subset_ratio,
                    "exposure_samples_per_round":
                        self.args.router_replay_exposure_samples,
                    "v1_router_retune_enabled":
                        self.args.router_retune_epochs > 0,
                    "distribution": self.args.replay_distribution,
                    "subset_seed": self.args.replay_subset_seed,
                },
            }
            if extra["training_version"] in ("v2", "v2_5"):
                extra["v2"] = {
                    "memory_batch_size": self.args.v2_memory_batch_size,
                    "kd_exposure_samples_per_round":
                        self.args.router_replay_exposure_samples,
                    "kd_loss_coeff": self.args.v2_kd_loss_coeff,
                    "kd_temperature": self.args.v2_kd_temperature,
                    "kd_learning_rate": self.args.v2_kd_learning_rate,
                    "kd_chunk_tokens": self.args.v2_kd_chunk_tokens,
                    "kd_token_scope": self.args.v2_kd_token_scope,
                    "joint_replay_loss_coeff": self.args.v2_joint_replay_loss_coeff,
                    "max_replay_batches_per_step":
                        self.args.v2_max_replay_batches_per_step,
                }
            save_lora_moe_meta(self.raw_model, output_dir, extra=extra)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_replay_loader(self, i_task):
        """v1: exactly one fixed global exposure budget over all seen tasks.

        Every 5,000-sample task contributes one immutable 1% (50-record)
        subset. The current task is included, counts are allocated equally
        across all seen tasks, and small subsets repeat deterministically until
        the per-round global exposure total is exactly the configured budget.
        """
        args = self.args
        replay_tasks = list(self.train_task_list)[:i_task + 1]
        replay_batch_size = min(
            args.batch_by_task[name] for name in replay_tasks)
        return self._build_fixed_memory_loader(
            replay_tasks, round_index=i_task,
            phase="v1_router_retune",
            exposure_samples=args.router_replay_exposure_samples,
            batch_size=replay_batch_size)



class Ours_LoRA_MoE_V2(Ours_LoRA_MoE):
    """KD-initialized growth plus budgeted joint router replay.

    For task t>0, KD and joint replay draw from the same deterministic 1%
    subsets of strictly past tasks. Both phases consume the same deterministic
    1,000-sample global exposure stream exactly once. Joint replay spreads its
    stream across all new-task epochs; those replay batches are distributed
    uniformly over the primary microsteps. A replay backward freezes every
    expert and adds only router gradients before the normal shared update.
    """

    def train_one_task(self, task, i_task, epochs):
        device = (
            torch.device("cuda", self.args.local_rank)
            if self.args.local_rank != -1 else torch.device("cuda")
        )
        args = self.args
        self._ensure_fixed_task_subset(task)
        old_expert_count = len(_lora_moe_layers(self.raw_model)[0].experts)
        add_experts_to_all_layers(self.raw_model, args.experts_per_task)
        new_indices = set(range(
            old_expert_count, old_expert_count + args.experts_per_task))

        # Separate DataLoader instances use the same phase seed, subset
        # indices, exposure allocation and batch size, so KD and router replay
        # consume the same deterministic 1,000-record stream.
        kd_loader = self._build_v2_memory_loader(i_task)
        replay_loader = self._build_v2_memory_loader(i_task)
        if (old_expert_count > 0 and kd_loader is not None
                and args.v2_kd_loss_coeff > 0):
            self._run_v2_kd_init(
                kd_loader, old_expert_count, new_indices, device, task)

        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} v2 primary")

        # One optimizer owns the new experts and every router row. On scheduled
        # replay microsteps the second backward freezes every expert, adding
        # only router gradients to the primary gradients before the shared step.
        self._set_grad_ckpt(
            replay_loader is not None
            or task in getattr(args, "ckpt_tasks", set()))
        freeze_lora_moe_experts(
            self.raw_model, trainable_expert_indices=new_indices)
        freeze_lora_moe_routers(self.raw_model, trainable=True)
        self._reinit_engine(self._optimizer_update_count(
            self.train_task_list[task], epochs))
        if replay_loader is None:
            self._run_epochs(
                self.train_task_list[task], epochs, device,
                f"{task} [v2 primary-only; no prior memory]")
        else:
            self._run_v2_joint_epochs(
                self.train_task_list[task], replay_loader, epochs, device,
                f"{task} [v2 joint exact-budget router replay]")

    def _memory_task_names(self, i_task):
        # Current data already supplies expert+router gradients. It enters the
        # strictly past replay/KD pool only from the next round.
        return list(self.train_task_list)[:i_task]

    def _build_v2_memory_loader(self, i_task):
        args = self.args
        task_names = self._memory_task_names(i_task)
        if not task_names:
            return None
        # Batch size one per rank spreads replay broadly. Both calls use an
        # identical seed/phase and therefore the exact same 1,000-record stream.
        memory_batch_size = args.v2_memory_batch_size or 1
        return self._build_fixed_memory_loader(
            task_names, round_index=i_task,
            phase="v2_shared_exact_memory",
            exposure_samples=args.router_replay_exposure_samples,
            batch_size=memory_batch_size)


    @staticmethod
    def _kd_kl_loss(student_logits, teacher_logits, batch, args):
        if args.v2_kd_token_scope == "labels":
            mask = batch["labels"].ne(-100)
        else:
            mask = batch["attention_mask"].bool()
        student = student_logits[mask]
        teacher = teacher_logits[mask]
        if student.numel() == 0:
            return student_logits.sum() * 0.0
        temperature = args.v2_kd_temperature
        total = student.new_zeros((), dtype=torch.float)
        chunk_tokens = max(1, args.v2_kd_chunk_tokens)
        for start in range(0, student.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, student.shape[0])
            student_log_probs = F.log_softmax(
                student[start:stop].float() / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(
                teacher[start:stop].float() / temperature, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            total = total + torch.sum(
                teacher_probs * (teacher_log_probs - student_log_probs))
        return total * (temperature ** 2) / student.shape[0]

    def _run_v2_kd_init(self, dataloader, old_expert_count, new_indices,
                        device, task):
        args = self.args
        self._set_phase_gradient_accumulation(
            dataloader.batch_size, f"{task} v2 KD init")
        self._set_grad_ckpt(True)
        freeze_lora_moe_experts(
            self.raw_model, trainable_expert_indices=new_indices)
        freeze_lora_moe_routers(self.raw_model, trainable=True)
        # The loader itself is the complete exact 1,000-exposure KD
        # budget, so KD consumes it once rather than multiplying by epochs.
        total_microsteps = len(dataloader)
        updates = self._optimizer_update_count(dataloader, 1)
        kd_lr = args.v2_kd_learning_rate or args.learning_rate
        self._reinit_engine(updates, learning_rate=kd_lr)
        old_router_rows = self._snapshot_old_router_rows(
            self.raw_model, old_expert_count)
        progress = tqdm(
            total=total_microsteps, leave=True,
            disable=args.global_rank != 0)
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(0)
        for step, source_batch in enumerate(dataloader):
            # The same KD record is evaluated once by the frozen teacher
            # and once by the expanded student, then backpropagated once.
            self._count_workload_batch(
                "kd_init", source_batch, forward_passes=2,
                backward_passes=1)
            batch = dict(source_batch)
            batch.pop("sources", None)
            batch = to_device(batch, device)
            accum = max(1, args.gradient_accumulation_steps)
            window_start = (step // accum) * accum
            window_size = min(accum, len(dataloader) - window_start)
            window_index = step - window_start
            should_step = window_index + 1 == window_size
            sync = (
                self.model.no_sync()
                if isinstance(self.model, DDP) and not should_step
                else nullcontext())
            set_router_token_mask(
                self.raw_model, batch.get("attention_mask"))
            try:
                self.raw_model.eval()
                with torch.no_grad(), limit_lora_moe_experts(
                        self.raw_model, old_expert_count):
                    teacher_logits = self.raw_model(
                        **batch, use_cache=False).logits.detach()
                self.model.train()
                with sync:
                    student_logits = self.model(
                        **batch, use_cache=False).logits
                    kd_loss = self._kd_kl_loss(
                        student_logits, teacher_logits, batch, args)
                    loss = args.v2_kd_loss_coeff * kd_loss
                    (loss / window_size).backward()
            finally:
                set_router_token_mask(self.raw_model, None)
            if args.global_rank == 0:
                progress.update(1)
                if (step % args.loss_log_interval == 0
                        or step + 1 == len(dataloader)):
                    progress.set_description(
                        f"{task} [v2 KD-init exact 1000] s{step} "
                        f"kl={kd_loss.detach().float().item():.5f}",
                        refresh=False)
            if should_step:
                # Old rows define the fixed pre-expansion teacher. Zero
                # their gradients and restore exact values after AdamW.
                self._freeze_old_router_row_update(
                    self.raw_model, old_router_rows, old_expert_count)
                trainable = [
                    p for p in self.raw_model.parameters()
                    if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                self.optimizer.step()
                self._freeze_old_router_row_update(
                    self.raw_model, old_router_rows, old_expert_count)
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._count_workload_update()

    @contextmanager
    def _router_only_replay(self):
        expert_parameters = [
            parameter
            for layer in _lora_moe_layers(self.raw_model)
            for expert in layer.experts
            for parameter in expert.parameters()
        ]
        previous = [parameter.requires_grad for parameter in expert_parameters]
        for parameter in expert_parameters:
            parameter.requires_grad = False
        try:
            yield
        finally:
            for parameter, requires_grad in zip(expert_parameters, previous):
                parameter.requires_grad = requires_grad

    @staticmethod
    def _manual_average_gradients(model):
        if not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(world_size)

    @staticmethod
    def _valid_token_count(batch):
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            raise ValueError(
                "v2 token-ratio replay requires an attention_mask")
        return int(attention_mask.sum().item())

    def _merge_replay_batches(self, batches):
        """Left-pad several loader batches and return one replay forward batch."""
        if not batches:
            raise ValueError("cannot merge an empty replay batch list")
        if len(batches) == 1:
            return dict(batches[0])
        max_length = max(batch["input_ids"].shape[1] for batch in batches)
        merged = {}
        tensor_keys = set.intersection(*[
            {key for key, value in batch.items() if torch.is_tensor(value)}
            for batch in batches
        ])
        for key in tensor_keys:
            values = []
            if key == "input_ids":
                pad_value = self.tokenizer.pad_token_id
            elif key == "labels":
                pad_value = -100
            else:
                pad_value = 0
            for batch in batches:
                value = batch[key]
                if value.ndim != 2:
                    raise ValueError(
                        f"unsupported replay tensor shape for {key}: "
                        f"{tuple(value.shape)}")
                pad_length = max_length - value.shape[1]
                values.append(F.pad(value, (pad_length, 0), value=pad_value))
            merged[key] = torch.cat(values, dim=0)
        merged["sources"] = [
            source for batch in batches
            for source in batch.get("sources", [])
        ]
        return merged

    def _run_v2_joint_epochs(self, primary_loader, memory_loader, epochs,
                             device, phase_name):
        """Spread one exact replay stream over all primary microsteps.

        Every primary backward trains the new expert and routers. At uniformly
        scheduled microsteps, one additional past-data backward freezes all
        experts and adds router-only gradients. The replay loader is consumed
        exactly once over the complete multi-epoch task, so its global dataset
        length is the complete router-replay exposure budget for the round.
        """
        args = self.args
        total_steps = epochs * len(primary_loader)
        total_memory_batches = len(memory_loader)
        if total_steps < 1 or total_memory_batches < 1:
            raise ValueError("v2 joint training requires non-empty loaders")
        progress = tqdm(
            total=total_steps, leave=True, disable=args.global_rank != 0)

        memory_sampler = getattr(memory_loader, "sampler", None)
        if hasattr(memory_sampler, "set_epoch"):
            memory_sampler.set_epoch(0)
        memory_iterator = iter(memory_loader)
        consumed_memory_batches = 0
        global_primary_step = 0
        total_primary_tokens = 0
        total_replay_tokens = 0
        total_replay_steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            primary_sampler = getattr(primary_loader, "sampler", None)
            if hasattr(primary_sampler, "set_epoch"):
                primary_sampler.set_epoch(epoch)
            epoch_primary_tokens = 0
            epoch_replay_tokens = 0
            epoch_replay_steps = 0

            for step, source_batch in enumerate(primary_loader):
                primary_tokens = self._valid_token_count(source_batch)
                epoch_primary_tokens += primary_tokens
                total_primary_tokens += primary_tokens
                self._count_workload_batch("new_task", source_batch)
                primary = dict(source_batch)
                primary.pop("sources", None)
                primary = to_device(primary, device)
                accum = max(1, args.gradient_accumulation_steps)
                window_start = (step // accum) * accum
                window_size = min(accum, len(primary_loader) - window_start)
                window_index = step - window_start
                should_step = window_index + 1 == window_size

                # Both backward calls remain local; the combined gradient is
                # explicitly averaged once at the accumulation boundary.
                no_sync = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) else nullcontext())
                set_router_token_mask(
                    self.raw_model, primary.get("attention_mask"))
                try:
                    with no_sync:
                        primary_outputs = self.model(
                            **primary, use_cache=False)
                        moe_loss = collect_moe_losses(self.raw_model)
                        primary_loss = primary_outputs.loss
                        if moe_loss is not None:
                            primary_loss = primary_loss + moe_loss
                        (primary_loss / window_size).backward()
                finally:
                    set_router_token_mask(self.raw_model, None)

                # Cumulative floor scheduling consumes exactly all replay
                # batches while distributing them as uniformly as possible.
                target_after_step = (
                    (global_primary_step + 1) * total_memory_batches
                    // total_steps)
                replay_batches_this_step = (
                    target_after_step - consumed_memory_batches)
                if (args.v2_max_replay_batches_per_step > 0
                        and replay_batches_this_step
                        > args.v2_max_replay_batches_per_step):
                    raise RuntimeError(
                        "exact replay budget needs "
                        f"{replay_batches_this_step} batches on one step, above "
                        "--v2_max_replay_batches_per_step="
                        f"{args.v2_max_replay_batches_per_step}")

                replay_loss = None
                replay_token_count = 0
                if replay_batches_this_step > 0:
                    replay_sources = []
                    for _ in range(replay_batches_this_step):
                        try:
                            replay_source = next(memory_iterator)
                        except StopIteration as error:
                            raise RuntimeError(
                                "v2 replay stream ended before its exact budget")                                 from error
                        replay_sources.append(replay_source)
                        replay_token_count += self._valid_token_count(
                            replay_source)
                    replay_source_batch = self._merge_replay_batches(
                        replay_sources)
                    self._count_workload_batch(
                        "router_replay", replay_source_batch)
                    replay = dict(replay_source_batch)
                    replay.pop("sources", None)
                    replay = to_device(replay, device)
                    with self._router_only_replay():
                        set_router_token_mask(
                            self.raw_model, replay.get("attention_mask"))
                        try:
                            replay_sync = (
                                self.model.no_sync()
                                if isinstance(self.model, DDP)
                                else nullcontext())
                            with replay_sync:
                                replay_output = self.model(
                                    **replay, use_cache=False)
                                replay_loss = replay_output.loss
                                (args.v2_joint_replay_loss_coeff
                                 * replay_loss / window_size).backward()
                        finally:
                            set_router_token_mask(self.raw_model, None)
                    consumed_memory_batches += replay_batches_this_step
                    epoch_replay_tokens += replay_token_count
                    total_replay_tokens += replay_token_count
                    epoch_replay_steps += 1
                    total_replay_steps += 1

                if args.global_rank == 0:
                    progress.update(1)
                    if (step % args.loss_log_interval == 0
                            or step + 1 == len(primary_loader)):
                        replay_text = (
                            f"{replay_loss.detach().float().item():.4f}"
                            if replay_loss is not None else "-")
                        progress.set_description(
                            f"{phase_name} e{epoch + 1} s{step} "
                            f"new={primary_loss.detach().float().item():.4f} "
                            f"replay={replay_text} "
                            f"budget={consumed_memory_batches}/"
                            f"{total_memory_batches}",
                            refresh=False)
                if should_step:
                    self._manual_average_gradients(self.raw_model)
                    trainable = [
                        p for p in self.raw_model.parameters()
                        if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()
                global_primary_step += 1

            if args.global_rank == 0:
                print_rank_0(
                    f"{phase_name} epoch {epoch + 1}: valid tokens "
                    f"new={epoch_primary_tokens}, replay={epoch_replay_tokens}, "
                    f"scheduled_replay_steps={epoch_replay_steps}, "
                    f"consumed_batches={consumed_memory_batches}/"
                    f"{total_memory_batches}",
                    args.global_rank)

        if consumed_memory_batches != total_memory_batches:
            raise RuntimeError(
                "v2 exact replay schedule consumed "
                f"{consumed_memory_batches}/{total_memory_batches} batches")
        if args.global_rank == 0:
            print_rank_0(
                f"{phase_name} complete: new_tokens={total_primary_tokens}, "
                f"replay_tokens={total_replay_tokens}, "
                f"token_ratio={total_primary_tokens / max(1, total_replay_tokens):.3f}:1, "
                f"replay_steps={total_replay_steps}",
                args.global_rank)


# ---------------------------------------------------------------------------
# Checkpoint round-trip: save_hf_format stores the full state_dict (base_mlp +
# experts + router) under the *base* Qwen config.json, so a plain
# from_pretrained / vLLM load can't reconstruct the custom LoRAMoEMLP. These
# helpers write a small self-describing meta at save time and rebuild the exact
# grown model at load time.
# ---------------------------------------------------------------------------
LORA_MOE_META_NAME = "lora_moe_meta.json"


def save_lora_moe_meta(model, output_dir, extra=None):
    """Dump the hyperparameters + current expert count next to the checkpoint."""
    layers = _lora_moe_layers(model)
    if not layers:
        return
    layer = layers[0]
    meta = {
        "r": layer.r,
        "alpha": layer.alpha,
        "dropout": layer.dropout,
        "top_k": layer.top_k,
        "aux_loss_coeff": layer.aux_loss_coeff,
        "z_loss_coeff": layer.z_loss_coeff,
        "routing_weight_mode": layer.routing_weight_mode,
        "num_experts": layer.num_experts,
    }
    if extra:
        meta.update(extra)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, LORA_MOE_META_NAME), "w") as f:
        json.dump(meta, f, indent=2)


def num_experts_from_state_dict(state_dict):
    """Infer the grown expert-pool size straight from checkpoint keys (fallback
    when no lora_moe_meta.json is present)."""
    pat = re.compile(r"\.mlp\.experts\.(\d+)\.")
    max_idx = -1
    for k in state_dict:
        m = pat.search(k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_lora_moe_checkpoint(checkpoint_dir, tokenizer, base_model_name_or_path=None,
                             device="cuda", dtype=torch.bfloat16,
                             device_map=None):
    """Rebuild a trained growing-LoRA-MoE model from a save_hf_format checkpoint.

    checkpoint_dir contains pytorch_model.bin (the FULL state_dict: base weights,
    every layer's mlp.base_mlp.* frozen dense FFN, mlp.experts.*, mlp.router.*),
    config.json (base architecture), the tokenizer, and lora_moe_meta.json.

    base_model_name_or_path (recommended): the original pretrained model. We build
    the skeleton from it (correct base weights + config incl. the training-time
    embedding resize), then strict-load the checkpoint on top -- so any key
    mismatch surfaces loudly instead of silently leaving a tensor random. If it is
    None we build an empty skeleton from the checkpoint's own config.json and rely
    on the checkpoint holding every weight.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from utils.model.model_utils import create_hf_model

    meta_path = os.path.join(checkpoint_dir, LORA_MOE_META_NAME)
    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        # No meta (e.g. checkpoints saved before save_lora_moe_meta existed):
        # recover the expert count from the weights and fall back to defaults
        # for the routing hyperparameters.
        meta = {"r": None, "alpha": None, "top_k": 2,
                "aux_loss_coeff": 0.01, "z_loss_coeff": 0.001,
                "dropout": 0.0,
                "routing_weight_mode": "topk_softmax",
                "num_experts": num_experts_from_state_dict(state_dict)}

    # Existing checkpoints were trained with softmax over selected top-k logits.
    # New runs explicitly record full_softmax when requested.
    meta.setdefault("routing_weight_mode", "topk_softmax")
    meta.setdefault("dropout", 0.0)

    n_experts = meta["num_experts"]
    ckpt_experts = num_experts_from_state_dict(state_dict)
    if ckpt_experts and ckpt_experts != n_experts:
        raise ValueError(
            f"lora_moe_meta.json says num_experts={n_experts} but the checkpoint "
            f"holds {ckpt_experts} experts ({checkpoint_dir}).")

    # r/alpha only affect LoRAPair.scaling (= alpha / r), which is baked into the
    # forward, not the weights. If meta lacks them, infer r from an A tensor and
    # keep scaling==1.0 by setting alpha==r (any factor is absorbed into B anyway
    # for a *loaded* expert, but we reproduce training exactly when meta is present).
    if meta["r"] is None:
        a_key = next(k for k in state_dict if re.search(r"\.experts\.0\.gate\.A$", k))
        meta["r"] = state_dict[a_key].shape[0]
        meta["alpha"] = meta["r"]

    # Partial checkpoint (trainable_only save): only mlp.experts.* / mlp.router.*
    # keys, no backbone -> the base weights MUST come from base_model_name_or_path,
    # and we load with strict=False (backbone keys are simply absent).
    is_partial = all((".mlp.experts." in k) or (".mlp.router." in k) for k in state_dict)

    if base_model_name_or_path is not None:
        model = create_hf_model(
            AutoModelForCausalLM,
            base_model_name_or_path,
            tokenizer,
            disable_dropout=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            forbid_vocab_growth=is_partial,
            device_map=device_map,
        )
    elif is_partial:
        raise ValueError(
            f"{checkpoint_dir} is a trainable-only checkpoint (experts+router, no "
            f"backbone); pass base_model_name_or_path so the base can be rebuilt.")
    else:
        if device_map is not None:
            raise ValueError(
                "device_map evaluation requires base_model_name_or_path so the "
                "checkpoint can be loaded and dispatched by from_pretrained().")
        config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.config.pad_token_id = model.config.eos_token_id

    attach_lora_moe(model, r=meta["r"], alpha=meta["alpha"], top_k=meta["top_k"],
                    aux_loss_coeff=meta["aux_loss_coeff"], z_loss_coeff=meta["z_loss_coeff"],
                    routing_weight_mode=meta["routing_weight_mode"],
                    dropout=meta["dropout"])
    add_experts_to_all_layers(model, n_experts)

    # Full checkpoint -> strict; partial -> non-strict (backbone came from the base,
    # only experts/router are overlaid). Guard against silently-random experts by
    # asserting the checkpoint's grown keys all matched.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys when loading {checkpoint_dir}: {unexpected[:5]}"
    if not is_partial:
        assert not missing, f"missing keys when loading full checkpoint {checkpoint_dir}: {missing[:5]}"
    else:
        # every grown key present; the only 'missing' should be backbone (from base)
        grown_missing = [m for m in missing if ".mlp.experts." in m or ".mlp.router." in m]
        assert not grown_missing, f"partial ckpt missing grown keys: {grown_missing[:5]}"
    # A dispatched model already has layers placed across its device map; calling
    # .to() would collapse it back onto one GPU and may OOM.
    if device_map is None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model, meta
