"""
Growing FFN experts on a NATIVE MoE backbone (OLMoE-1B-7B-0125) for continual learning.

Track 2 (cf. Ours_LoRA_MoE = track 1, dense FFN + added LoRA experts). OLMoE's FFN
is already a sparse MoE: each layer is an OlmoeSparseMoeBlock with a 64-way gate and
64 full OlmoeMLP experts, top-8 routed. Here we keep the ENTIRE pretrained backbone
-- attention, embeddings, norms, lm_head, the original 64 experts, and the original
gate -- frozen, and per task append `experts_per_task` brand-new FULL OlmoeMLP
experts, growing the router 64 -> 64+n. Routing is top-k over the whole 64+n pool
using OLMoE's own softmax / topk / norm_topk_prob.

New experts start as no-ops (down_proj initialised to 0). Phase 1 trains only the
current task's new experts + their matching appended router rows; every earlier
expert/router row and the original 64-way gate stay frozen. Phase 2 freezes ALL
experts and retunes the WHOLE extended router (original 64-way gate + every new
row) on replayed past+current-task data. During phase 1, the current new expert is
forced into one top-k slot so it cannot starve before its new router row is learned.
"""
import copy
from contextlib import nullcontext
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import get_constant_schedule_with_warmup

from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_optimizer_grouped_parameters


def _new_olmoe_expert(config, device, dtype):
    """A fresh OlmoeMLP with down_proj zeroed -> outputs 0 (no-op) until trained."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
    expert = OlmoeMLP(config)
    with torch.no_grad():
        expert.down_proj.weight.zero_()
    return expert.to(device=device, dtype=dtype)


class GrowingOlmoeMoE(nn.Module):
    """Wraps a frozen OlmoeSparseMoeBlock with a growing pool of extra full FFN experts
    and a growing router tail. Drop-in for the original block: returns (hidden, logits)."""

    def __init__(self, base_block, aux_loss_coeff, z_loss_coeff):
        super().__init__()
        self.base_block = base_block          # original gate(64) + experts(64), frozen
        for p in self.base_block.parameters():
            p.requires_grad = False

        self.num_base = base_block.num_experts
        self.top_k = base_block.top_k
        self.norm_topk_prob = base_block.norm_topk_prob
        self.hidden_size = base_block.gate.in_features
        self.expert_config = base_block.experts[0].config
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff

        self.new_experts = nn.ModuleList()
        # Keep each appended router row as its own Parameter/module.  A single
        # nn.Linear(hidden, num_new) cannot freeze individual rows: setting
        # requires_grad is tensor-wide, so task t's phase 1 would also update all
        # earlier task rows.  Per-row modules let phase 1 train only the row(s)
        # appended for the current task and phase 2 unfreeze the complete router.
        self.new_gates = nn.ModuleList()      # one nn.Linear(hidden, 1) per new expert
        self.phase1_forced_new_indices = ()
        self.router_token_mask = None
        self._last_moe_loss = None

    @property
    def num_new(self):
        return len(self.new_experts)

    @property
    def num_experts(self):
        return self.num_base + self.num_new

    def add_experts(self, n):
        device = self.base_block.gate.weight.device
        dtype = self.base_block.gate.weight.dtype
        for _ in range(n):
            self.new_experts.append(_new_olmoe_expert(self.expert_config, device, dtype))
            new_gate = nn.Linear(self.hidden_size, 1, bias=False).to(device=device, dtype=dtype)
            with torch.no_grad():
                new_gate.weight.zero_()       # new row starts at 0 -> modest initial routing
            self.new_gates.append(new_gate)

    def _expert(self, e):
        return self.base_block.experts[e] if e < self.num_base else self.new_experts[e - self.num_base]

    def set_phase1_forced_experts(self, new_indices=None):
        indices = tuple(sorted(new_indices or ()))
        if len(indices) > self.top_k:
            raise ValueError(
                f"Cannot force {len(indices)} new experts into top_k={self.top_k}.")
        if any(i < 0 or i >= self.num_new for i in indices):
            raise IndexError(
                f"Forced new-expert indices {indices} are outside [0, {self.num_new}).")
        self.phase1_forced_new_indices = indices

    def set_router_token_mask(self, token_mask=None):
        """Set the current batch's non-padding mask for dispatch/router losses."""
        self.router_token_mask = token_mask

    def _force_new_experts_into_topk(self, selected, token_mask=None):
        """Replace the lowest-ranked non-forced slots for tokens missing a new expert."""
        if not self.phase1_forced_new_indices:
            return selected
        forced_ids = torch.tensor(
            [self.num_base + i for i in self.phase1_forced_new_indices],
            device=selected.device, dtype=selected.dtype)
        dispatch_selected = selected.clone()
        slot_ids = torch.arange(selected.shape[1], device=selected.device)
        for expert_id in forced_ids.unbind():
            present = (dispatch_selected == expert_id).any(dim=-1)
            forced_slots = (
                dispatch_selected.unsqueeze(-1) == forced_ids.view(1, 1, -1)
            ).any(dim=-1)
            replacement = torch.where(
                ~forced_slots, slot_ids.view(1, -1), -1).amax(dim=-1)
            needs_replacement = ~present & (replacement >= 0)
            if token_mask is not None:
                needs_replacement &= token_mask
            rows = torch.where(needs_replacement)[0]
            dispatch_selected[rows, replacement[rows]] = expert_id
        return dispatch_selected

    def forward(self, hidden_states):
        base = self.base_block
        if self.num_new == 0:                 # nothing added yet -> exact original block
            return base(hidden_states)

        B, S, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)

        # One router GEMM over all 64+n rows. Keeping the appended rows as
        # independent Parameters is still important for phase-1 isolation; cat's
        # autograd only writes gradients to the leaf rows whose requires_grad=True.
        router_weight = torch.cat(
            [base.gate.weight] + [gate.weight for gate in self.new_gates], dim=0)
        logits = F.linear(hidden_states, router_weight)                    # [N, 64+num_new]
        num_experts = logits.shape[-1]
        k = min(self.top_k, num_experts)

        full_routing_probs = F.softmax(logits, dim=1, dtype=torch.float)
        routing_weights, selected = torch.topk(full_routing_probs, k, dim=-1)
        token_mask = self.router_token_mask
        if token_mask is not None:
            token_mask = token_mask.reshape(-1).to(device=selected.device, dtype=torch.bool)
            if token_mask.numel() != selected.shape[0]:
                raise ValueError(
                    f"Router token mask has {token_mask.numel()} positions, but "
                    f"the MoE layer received {selected.shape[0]} tokens.")
        natural_selected = selected
        if self.training and self.phase1_forced_new_indices:
            selected = self._force_new_experts_into_topk(selected, token_mask)
            routing_weights = torch.gather(full_routing_probs, 1, selected)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final = torch.zeros((B * S, H), dtype=hidden_states.dtype, device=hidden_states.device)
        for e in range(num_experts):
            # Avoid HF's dense int64 [experts, top_k, tokens] one-hot dispatch
            # tensor. The transposed comparison preserves its slot-major order while
            # reducing peak dispatch memory to a temporary [top_k, tokens] bool mask.
            idx, top_x = torch.where(selected.transpose(0, 1) == e)
            if token_mask is not None and top_x.numel() > 0:
                keep = token_mask[top_x]
                idx, top_x = idx[keep], top_x[keep]
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[None, top_x].reshape(-1, H)
            current = self._expert(e)(current_state) * routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, current.to(hidden_states.dtype))
        final = final.reshape(B, S, H)

        # Load balancing sees the router's natural choices, not our temporary
        # phase-1 anti-starvation assignment (otherwise it would punish the new
        # expert for being deliberately present on every current-task token).
        self._last_moe_loss = (
            self._router_loss(
                logits, full_routing_probs, natural_selected, token_mask)
            if self.training else None)
        return final, logits

    def _router_loss(self, logits, full_probs, selected, token_mask=None):
        # Same Switch aux loss + router z-loss as Ours_LoRA_MoE, over the 64+n pool.
        if token_mask is not None:
            logits = logits[token_mask]
            full_probs = full_probs[token_mask]
            selected = selected[token_mask]
        num_tokens, num_experts = logits.shape
        if num_tokens == 0:
            return logits.sum() * 0.0
        # bincount avoids allocating a dense [tokens, experts] hard-routing mask,
        # and reuse the FP32 softmax already computed by the router forward.
        tokens_per_expert = torch.bincount(
            selected.reshape(-1), minlength=num_experts).to(full_probs.dtype)
        aggregated_probs_per_expert = full_probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * self.aux_loss_coeff / (num_tokens * num_tokens * self.top_k)
        )
        z_loss = torch.mean(torch.square(
            torch.logsumexp(logits.float(), dim=-1))) * self.z_loss_coeff
        return aux_loss + z_loss


def attach_growing_moe(model, aux_loss_coeff, z_loss_coeff):
    """Replace every OLMoE layer's .mlp with a GrowingOlmoeMoE, then freeze the whole
    backbone (attention, embeddings, norms, lm_head, original gate + 64 experts). Only
    the added experts and the router (new tail, plus the original gate in phase 2) ever
    train -- toggled per phase by the freeze_* helpers."""
    for layer in model.model.layers:
        layer.mlp = GrowingOlmoeMoE(layer.mlp, aux_loss_coeff, z_loss_coeff)
    for p in model.parameters():   # no new experts/gate exist yet -> freezes pure backbone
        p.requires_grad = False
    return model


def _moe_layers(model):
    return [m for m in model.modules() if isinstance(m, GrowingOlmoeMoE)]


def add_experts_to_all_layers(model, n):
    for layer in _moe_layers(model):
        layer.add_experts(n)


def freeze_moe_experts(model, trainable_new_indices=None):
    """Original 64 experts stay frozen always. Among the ADDED experts, only those in
    trainable_new_indices train (None -> all added experts frozen)."""
    for layer in _moe_layers(model):
        for i, expert in enumerate(layer.new_experts):
            rg = trainable_new_indices is not None and i in trainable_new_indices
            for p in expert.parameters():
                p.requires_grad = rg


def freeze_moe_routers(model, trainable_new_indices=None, trainable_original=False):
    """Set router trainability at row granularity.

    ``trainable_new_indices`` contains indices into the appended-expert/router pool.
    Phase 1 passes only the indices added for the current task; phase 2 passes every
    appended index and also sets ``trainable_original=True`` to retune all 64+n rows.
    ``None`` freezes every appended row.
    """
    for layer in _moe_layers(model):
        for i, gate in enumerate(layer.new_gates):
            trainable = trainable_new_indices is not None and i in trainable_new_indices
            for p in gate.parameters():
                p.requires_grad = trainable
        for p in layer.base_block.gate.parameters():
            p.requires_grad = trainable_original


def set_phase1_forced_experts(model, new_indices=None):
    """Temporarily guarantee current-task experts receive tokens during phase 1."""
    for layer in _moe_layers(model):
        layer.set_phase1_forced_experts(new_indices)


def set_router_token_mask(model, token_mask=None):
    """Exclude padding positions from expert dispatch and custom router losses."""
    for layer in _moe_layers(model):
        layer.set_router_token_mask(token_mask)


def collect_moe_losses(model):
    total = None
    for layer in _moe_layers(model):
        if layer._last_moe_loss is not None:
            total = layer._last_moe_loss if total is None else total + layer._last_moe_loss
    return total


class Ours_MoE_FFN(CL_Base_Model):
    """Per task: (1) grow experts_per_task new full FFN experts and router rows, then
    train only those newly appended parameters on the new task; (2) freeze every
    expert and retune the WHOLE extended router (original gate + all appended rows)
    on replayed past+current-task data.

    Mirrors Ours_LoRA_MoE's engine handling: self.raw_model is the plain nn.Module that
    survives across tasks; self.model is a DistributedDataParallel wrapper around it
    (or raw_model itself if not distributed), rebuilt via _reinit_engine at the start
    of every phase (new nn.Parameters from add_experts + a fresh optimizer filtered by
    the freeze pattern active at that moment). See Ours_LoRA_MoE for why this no longer
    uses deepspeed.initialize() -- repeated re-init on the same model leaked GPU memory
    without bound regardless of ZeRO stage/cleanup; plain DDP+AdamW does not."""

    # The pretrained backbone is reproducibly rebuilt from --base_model_name_or_path.
    # Persist only learned additions and the COMPLETE router: appended rows plus the
    # original rows that phase 2 retunes.
    save_key_substrings = [
        ".mlp.new_experts.",
        ".mlp.new_gates.",
        ".mlp.base_block.gate.",
    ]

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                 test_task_list, args, router_pretrain_dataset=None):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.raw_model = model
        if router_pretrain_dataset is None:
            raise ValueError("Track 2 requires the OLMoE pretraining replay dataset for phase 2.")
        self.router_pretrain_dataset = router_pretrain_dataset
        self._dist_initialized = False

    def _reinit_engine(self):
        args = self.args
        if self._dist_initialized:
            del self.model
            torch.cuda.empty_cache()
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.raw_model, args.weight_decay)
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
        self.lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=args.num_warmup_steps)
        if args.local_rank != -1:
            self.model = DDP(self.raw_model, device_ids=[args.local_rank],
                             output_device=args.local_rank,
                             find_unused_parameters=True,
                             broadcast_buffers=False,
                             gradient_as_bucket_view=True)
        else:
            self.model = self.raw_model
        self._dist_initialized = True

    def _set_grad_ckpt(self, enable):
        """Set activation checkpointing for the next phase without stacking hooks."""
        model = self.raw_model
        if getattr(model, "_require_grads_hook", None) is not None:
            model.disable_input_require_grads()
            model._require_grads_hook = None
        if enable:
            # The backbone and embeddings are frozen. Checkpointed decoder blocks
            # still need a grad-carrying input so gradients reach the trainable
            # expert/router parameters inside them.
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        print_rank_0(
            f"  [grad_ckpt] {'ON' if enable else 'off'}",
            self.args.global_rank)

    def train_one_task(self, task, i_task, epochs):
        device = torch.device("cuda", self.args.local_rank) if self.args.local_rank != -1 else torch.device("cuda")
        args = self.args

        new_start = i_task * args.experts_per_task
        new_indices = set(range(new_start, new_start + args.experts_per_task))
        resume_phase2 = bool(getattr(args, "resume_phase2_checkpoint", None))

        # Phase 1: only this task's new experts + matching router rows.  Earlier
        # appended experts/rows and the original 64-way gate stay frozen.
        if resume_phase2:
            print_rank_0(f"Skipping completed phase 1 for task {i_task}: {task}", args.global_rank)
        else:
            add_experts_to_all_layers(self.raw_model, args.experts_per_task)
            self._set_grad_ckpt(task in getattr(args, "ckpt_tasks", set()))
            freeze_moe_experts(self.raw_model, trainable_new_indices=new_indices)
            freeze_moe_routers(self.raw_model, trainable_new_indices=new_indices,
                               trainable_original=False)
            set_phase1_forced_experts(
                self.raw_model,
                new_indices if getattr(args, "phase1_new_expert_routing", "force") == "force" else None)
            self._reinit_engine()
            self._run_epochs(self.train_task_list[task], epochs, device, f"{task} [phase1 expert+router]")
            self.save_model(os.path.join(".recovery", f"task_{i_task}_phase1"))

        # Phase 2 runs after every task, including task 0: all experts are frozen
        # and every original/appended router row is retuned on globally shuffled
        # samples from every task seen so far.
        self._set_grad_ckpt(True)
        set_phase1_forced_experts(self.raw_model, None)
        freeze_moe_experts(self.raw_model, trainable_new_indices=None)
        all_new_indices = set(range((i_task + 1) * args.experts_per_task))
        freeze_moe_routers(self.raw_model, trainable_new_indices=all_new_indices,
                           trainable_original=True)
        self._reinit_engine()
        replay_loader = self._build_replay_loader(i_task)
        self._run_epochs(replay_loader, args.router_retune_epochs, device, f"{task} [phase2 router retune]")

    def _run_epochs(self, dataloader, epochs, device, phase_name):
        args = self.args
        total_steps = epochs * len(dataloader)
        accumulation_steps = max(1, args.gradient_accumulation_steps)
        loss_log_interval = max(1, getattr(args, "loss_log_interval", 10))
        progress_bar = tqdm(total=total_steps, leave=True, disable=(args.global_rank != 0))
        for epoch in range(epochs):
            # DistributedSampler otherwise repeats the exact same ordering every
            # epoch. This is especially important for phase-2 cross-task mixing.
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
            print_rank_0(f"{phase_name}: epoch {epoch+1}/{epochs}, {len(dataloader)} steps", args.global_rank)
            self.model.train()
            for step, batch in enumerate(dataloader):
                # Scale the last, possibly shorter accumulation window by its real
                # size so no gradients are silently dropped at epoch end.
                window_start = (step // accumulation_steps) * accumulation_steps
                window_size = min(accumulation_steps, len(dataloader) - window_start)
                accumulation_index = step - window_start
                should_step = accumulation_index + 1 == window_size
                sync_context = (
                    self.model.no_sync()
                    if hasattr(self.model, "no_sync") and not should_step
                    else nullcontext()
                )
                with sync_context:
                    del batch['sources']
                    batch = to_device(batch, device)
                    set_router_token_mask(self.raw_model, batch.get("attention_mask"))
                    try:
                        outputs = self.model(**batch, use_cache=False)
                        moe_loss = collect_moe_losses(self.model)
                        loss = outputs.loss if moe_loss is None else outputs.loss + moe_loss
                        self._optimizer_step(
                            loss, accum_step=accumulation_index,
                            accum_steps=window_size)
                    finally:
                        set_router_token_mask(self.raw_model, None)
                if args.global_rank == 0:
                    progress_bar.update(1)
                    if step % loss_log_interval == 0 or step + 1 == len(dataloader):
                        progress_bar.set_description(
                            f"{phase_name} | epoch {epoch+1} step {step} "
                            f"loss {loss.detach().float().item():.4f}", refresh=False)

    def save_model(self, round):
        super().save_model(round)
        if self.args.global_rank == 0:
            output_dir = os.path.join(self.args.output_dir, str(round))
            save_moe_ffn_meta(
                self.raw_model, output_dir,
                base_model_name_or_path=self.args.model_name_or_path,
                experts_per_task=self.args.experts_per_task,
                phase1_new_expert_routing=getattr(
                    self.args, "phase1_new_expert_routing", "force"),
                checkpoint_format=MOE_FFN_DELTA_FORMAT)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_replay_loader(self, i_task):
        # Keep collator dependencies out of the module import path so the lightweight
        # expert/router checkpoint loader also works in eval-only environments.
        from utils.data.data_collator import DataCollator

        args = self.args
        # miniset = past + current task (accumulated), so phase-2 keeps the just-learned
        # task's routing calibrated too, not only the past ones.
        replay_tasks = list(self.train_task_list.keys())[:i_task + 1]
        # The original OLMoE pretraining distribution is always one replay source.
        # It is constructed once in main and reused here without re-tokenization.
        replay_datasets = [self.router_pretrain_dataset]
        for task in replay_tasks:
            # Reuse the already-built phase-1 dataset. Re-tokenizing every seen
            # task at every phase 2 caused quadratic preprocessing and disk writes.
            train_dataset = self.train_task_list[task].dataset
            if args.past_task_ratio < 1.0:
                keep = int(len(train_dataset) * args.past_task_ratio)
                train_dataset = Subset(train_dataset, range(keep))
            replay_datasets.append(train_dataset)
        replay_source_sizes = [("OLMoE-pretrain", len(self.router_pretrain_dataset))]
        replay_source_sizes.extend(
            (task, len(dataset)) for task, dataset in zip(replay_tasks, replay_datasets[1:]))
        print_rank_0(
            "Phase-2 replay sources: "
            + ", ".join(f"{name}={size}" for name, size in replay_source_sizes),
            args.global_rank)
        combined = ConcatDataset(replay_datasets)

        collator = DataCollator(self.tokenizer, padding="longest", max_prompt_len=args.max_prompt_len,
                                max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=False)
        # ConcatDataset + a global random sampler mixes examples across task
        # boundaries. Do not replace this with task-homogeneous batch sampling.
        sampler = (RandomSampler(combined) if args.local_rank == -1 else
                   DistributedSampler(combined, shuffle=True, seed=args.seed))
        return DataLoader(combined, collate_fn=collator, sampler=sampler,
                          batch_size=args.router_retune_batch_size,
                          num_workers=4, pin_memory=True)


# ---------------------------------------------------------------------------
# Checkpoint round-trip (mirrors Ours_LoRA_MoE): the wrapped block puts the
# original module under .base_block and adds .new_experts / .new_gates, so a plain
# from_pretrained can't rebuild it. Save a self-describing meta; rebuild at load.
# ---------------------------------------------------------------------------
MOE_FFN_META_NAME = "moe_ffn_meta.json"
MOE_FFN_DELTA_FORMAT = "moe_ffn_delta_v1"
MOE_FFN_SAVE_KEY_SUBSTRINGS = tuple(Ours_MoE_FFN.save_key_substrings)


def save_moe_ffn_meta(model, output_dir, base_model_name_or_path=None,
                      experts_per_task=None, checkpoint_format=None,
                      phase1_new_expert_routing=None):
    layers = _moe_layers(model)
    if not layers:
        return
    layer = layers[0]
    meta = {
        "num_base_experts": layer.num_base,
        "num_new_experts": layer.num_new,
        "aux_loss_coeff": layer.aux_loss_coeff,
        "z_loss_coeff": layer.z_loss_coeff,
        "base_model_name_or_path": base_model_name_or_path,
        "experts_per_task": experts_per_task,
        "phase1_new_expert_routing": phase1_new_expert_routing,
    }
    if checkpoint_format is not None:
        meta["checkpoint_format"] = checkpoint_format
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, MOE_FFN_META_NAME), "w") as f:
        json.dump(meta, f, indent=2)


def num_new_experts_from_state_dict(state_dict):
    pat = re.compile(r"\.mlp\.new_experts\.(\d+)\.")
    max_idx = -1
    for k in state_dict:
        m = pat.search(k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _upgrade_legacy_new_gate_state_dict(state_dict):
    """Convert old checkpoints' combined ``new_gate.weight`` into per-row gates.

    Track-2 checkpoints written before router-row isolation stored all appended
    rows in one ``nn.Linear``.  The model now stores one ``nn.Linear(hidden, 1)``
    per row so phase 1 can freeze earlier rows.  Split legacy matrices on load;
    current-format checkpoints pass through unchanged.
    """
    suffix = ".mlp.new_gate.weight"
    legacy_keys = [k for k in state_dict if k.endswith(suffix)]
    if not legacy_keys:
        return state_dict
    upgraded = copy.copy(state_dict)
    for key in legacy_keys:
        weight = upgraded.pop(key)
        prefix = key[:-len(suffix)]
        for i, row in enumerate(weight):
            upgraded[f"{prefix}.mlp.new_gates.{i}.weight"] = row.unsqueeze(0)
    return upgraded


def load_moe_ffn_checkpoint(checkpoint_dir, tokenizer, base_model_name_or_path=None,
                            device="cuda", dtype=torch.bfloat16,
                            attn_implementation="auto", device_map=None):
    """Rebuild a trained growing-MoE model from a full or delta checkpoint.

    New checkpoints contain only learned experts and the complete retuned router, so
    the original pretrained OLMoE is rebuilt first. ``base_model_name_or_path`` takes
    precedence over the path recorded in metadata and is recommended after moving a
    checkpoint to another machine.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from utils.model.model_utils import create_hf_model, resolve_attention_implementation

    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")
    state_dict = _upgrade_legacy_new_gate_state_dict(state_dict)

    meta_path = os.path.join(checkpoint_dir, MOE_FFN_META_NAME)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"num_new_experts": num_new_experts_from_state_dict(state_dict),
                "aux_loss_coeff": 0.01, "z_loss_coeff": 0.001}

    all_delta_keys = bool(state_dict) and all(
        any(tag in key for tag in MOE_FFN_SAVE_KEY_SUBSTRINGS)
        for key in state_dict)
    is_delta = meta.get("checkpoint_format") == MOE_FFN_DELTA_FORMAT or all_delta_keys
    if is_delta and not all_delta_keys:
        raise ValueError(
            f"{checkpoint_dir} declares {MOE_FFN_DELTA_FORMAT}, but contains "
            "non-delta state-dict keys.")

    n_new = meta["num_new_experts"]
    ckpt_new = num_new_experts_from_state_dict(state_dict)
    if ckpt_new and ckpt_new != n_new:
        raise ValueError(f"meta says num_new_experts={n_new} but checkpoint holds {ckpt_new}.")

    base_source = base_model_name_or_path or meta.get("base_model_name_or_path")
    if is_delta and not base_source:
        raise ValueError(
            "This is an expert/router-only checkpoint. Pass "
            "base_model_name_or_path for the original pretrained OLMoE.")

    if base_source is not None:
        resolved_attn = resolve_attention_implementation(attn_implementation)
        model = create_hf_model(
            AutoModelForCausalLM, base_source, tokenizer, disable_dropout=True,
            torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=resolved_attn,
            forbid_vocab_growth=is_delta,
            device_map=device_map)
    else:
        if device_map is not None:
            raise ValueError(
                "device_map evaluation requires base_model_name_or_path so the "
                "checkpoint can be loaded and dispatched by from_pretrained().")
        config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.config.pad_token_id = model.config.eos_token_id

    attach_growing_moe(model, aux_loss_coeff=meta["aux_loss_coeff"], z_loss_coeff=meta["z_loss_coeff"])
    layers = _moe_layers(model)
    expected_base = meta.get("num_base_experts")
    if expected_base is not None and any(layer.num_base != expected_base for layer in layers):
        actual = layers[0].num_base if layers else None
        raise ValueError(
            f"Checkpoint expects {expected_base} base experts per layer, but "
            f"{base_source!r} has {actual}.")
    add_experts_to_all_layers(model, n_new)

    if is_delta:
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_learned = [
            key for key in incompatible.missing_keys
            if any(tag in key for tag in MOE_FFN_SAVE_KEY_SUBSTRINGS)
        ]
        if unexpected or missing_learned:
            raise RuntimeError(
                "Invalid MoE-FFN delta checkpoint: "
                f"unexpected keys={unexpected}, missing learned keys={missing_learned}")
    else:
        model.load_state_dict(state_dict, strict=True)
    if device_map is None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model, meta
