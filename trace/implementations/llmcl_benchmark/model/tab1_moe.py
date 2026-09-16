"""Table-1 expansion baselines: Lifelong-MoE and MoE-LPR.

Both grow one LoRA expert per task on a frozen backbone, exactly like Ours, so
that a Table-1 comparison isolates the continual-learning mechanism rather than
the capacity schedule.  Each is available in two scopes:

``ffn``       one expert pool per FFN, routed per token (the Ours-FFN layout)
``ffn_attn``  one shared per-layer router driving equal-rank QKVO *and* FFN
              experts (the Ours-V3 layout)

The second scope exists so a row can be reported at the same trainable-
parameter count as Ours rather than at 3/7 of it; with only the FFN scope the
two methods differ in degrees of freedom as well as in mechanism, and the
comparison no longer says what it appears to say.

The two mechanisms:

Lifelong-MoE (Chen et al., ICML 2023) freezes previously-trained experts and
the router rows that address them, adds new experts for the new distribution,
and keeps every *other* parameter trainable, with an output-level distillation
term against the pre-task model to hold the old behaviour in place.  On a
frozen backbone "every other parameter" has no direct analogue, so the shared
path is emulated by a LoRA adapter carried across all tasks -- see
``--lifelong_shared_targets``.

MoE-LPR (Zhou et al., 2024) is two-phase: post-training the new experts on new
data, then a review phase that unfreezes *only* the router and trains it on a
small replay pool with a language-priority-routing term.  With one expert per
task that term is a plain cross-entropy from each replay token onto the expert
that owns its task.
"""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.Ours_LoRA_MoE import (
    Ours_LoRA_MoE, _lora_moe_layers, add_experts_to_all_layers,
    attach_lora_moe, collect_moe_losses, freeze_lora_moe_experts,
    freeze_lora_moe_routers, limit_lora_moe_experts, set_router_token_mask)
from model.Ours_LoRA_MoE_V3 import (
    add_v3_experts, attach_shared_qkvo_lora_moe, collect_v3_moe_losses,
    freeze_v3_experts, freeze_v3_routers, limit_v3_experts,
    set_v3_router_token_mask, shared_router_layers)
from model.tab1_lora import attach_seq_lora_targets, iter_seq_lora, resolve_targets
from utils.data.data_collator import (DataCollator,
                                      PreTokenizedSLoRATraceDataCollator,
                                      SLoRATraceDataCollator)
from utils.utils import print_rank_0, to_device


# ---------------------------------------------------------------------------
# Scope adapters
# ---------------------------------------------------------------------------

class MoEScope:
    """Uniform handle on the two expansion layouts.

    Both layouts already expose module-level add/freeze/mask helpers; this only
    puts one name on each pair so a trainer can be written once.  Router logits
    are recovered with a forward pre-hook rather than by editing the layout
    modules: the FFN layout slices ``router.weight`` and calls ``F.linear``
    directly (never the ``nn.Linear``), so an output hook would never fire, and
    both layouts are shared with Ours and must stay untouched.
    """

    name = "scope"
    save_key_substrings: List[str] = []

    def attach(self, model, args):
        raise NotImplementedError

    def add_experts(self, model, count):
        raise NotImplementedError

    def freeze_experts(self, model, trainable_expert_indices=None):
        raise NotImplementedError

    def freeze_routers(self, model, trainable):
        raise NotImplementedError

    def set_token_mask(self, model, mask):
        raise NotImplementedError

    def collect_losses(self, model):
        raise NotImplementedError

    def limit_experts(self, model, active_expert_count):
        raise NotImplementedError

    def num_experts(self, model):
        raise NotImplementedError

    def router_hosts(self, model):
        """Modules whose forward input is the router input, paired with the router."""
        raise NotImplementedError

    def snapshot_router_rows(self, model, count):
        return [self._router_of(host).weight[:count].detach().clone()
                for host, _ in self.router_hosts(model)]

    def restore_router_rows(self, model, snapshots, count):
        for (host, _), snapshot in zip(self.router_hosts(model), snapshots):
            router = self._router_of(host)
            if router.weight.grad is not None:
                router.weight.grad[:count].zero_()
            with torch.no_grad():
                router.weight[:count].copy_(snapshot)

    def _router_of(self, host):
        raise NotImplementedError


class FFNScope(MoEScope):
    name = "ffn"
    save_key_substrings = [".mlp.experts.", ".mlp.router."]

    def attach(self, model, args):
        return attach_lora_moe(
            model, r=args.lora_moe_rank, alpha=args.lora_moe_alpha,
            top_k=args.top_k, aux_loss_coeff=args.moe_aux_loss_coeff,
            z_loss_coeff=args.moe_z_loss_coeff,
            routing_weight_mode=args.routing_weight_mode,
            dropout=args.lora_moe_dropout)

    def add_experts(self, model, count):
        add_experts_to_all_layers(model, count)

    def freeze_experts(self, model, trainable_expert_indices=None):
        freeze_lora_moe_experts(model, trainable_expert_indices)

    def freeze_routers(self, model, trainable):
        freeze_lora_moe_routers(model, trainable)

    def set_token_mask(self, model, mask):
        set_router_token_mask(model, mask)

    def collect_losses(self, model):
        return collect_moe_losses(model)

    def limit_experts(self, model, active_expert_count):
        return limit_lora_moe_experts(model, active_expert_count)

    def num_experts(self, model):
        layers = _lora_moe_layers(model)
        return len(layers[0].experts) if layers else 0

    def router_hosts(self, model):
        return [(layer, layer) for layer in _lora_moe_layers(model)]

    def _router_of(self, host):
        return host.router


class SharedAttnFFNScope(MoEScope):
    name = "ffn_attn"
    save_key_substrings = [
        ".shared_expert_router.router.",
        ".self_attn.q_proj.experts.", ".self_attn.k_proj.experts.",
        ".self_attn.v_proj.experts.", ".self_attn.o_proj.experts.",
        ".mlp.experts.",
    ]

    def attach(self, model, args):
        return attach_shared_qkvo_lora_moe(
            model, r=args.lora_moe_rank, alpha=args.lora_moe_alpha,
            top_k=args.top_k, aux_loss_coeff=args.moe_aux_loss_coeff,
            z_loss_coeff=args.moe_z_loss_coeff,
            routing_weight_mode=args.routing_weight_mode,
            dropout=args.lora_moe_dropout)

    def add_experts(self, model, count):
        add_v3_experts(model, count)

    def freeze_experts(self, model, trainable_expert_indices=None):
        freeze_v3_experts(model, trainable_expert_indices)

    def freeze_routers(self, model, trainable):
        freeze_v3_routers(model, trainable)

    def set_token_mask(self, model, mask):
        set_v3_router_token_mask(model, mask)

    def collect_losses(self, model):
        return collect_v3_moe_losses(model)

    def limit_experts(self, model, active_expert_count):
        return limit_v3_experts(model, active_expert_count)

    def num_experts(self, model):
        layers = shared_router_layers(model)
        return layers[0].shared_expert_router.num_experts if layers else 0

    def router_hosts(self, model):
        return [(layer.shared_expert_router, layer)
                for layer in shared_router_layers(model)]

    def _router_of(self, host):
        return host.router


SCOPES = {"ffn": FFNScope, "ffn_attn": SharedAttnFFNScope}


def validate_shared_path(moe_scope: str, shared_targets: str) -> Sequence[str]:
    """Decide where Lifelong-MoE's emulated shared path may live, or refuse.

    A projection cannot carry both a routed expert pool and a shared adapter.
    Under the ``ffn`` scope ``LoRAMoEMLP`` reads ``base_mlp.gate_proj.weight``
    when it grows an expert, so wrapping those three projections in a
    ``SeqLoRALinear`` first makes expert growth fail at the first task; under
    ``ffn_attn`` every one of the seven projections is already routed.  Rather
    than silently attach nothing -- which would report a Lifelong-MoE row that
    never trained a shared path at all -- the invalid combinations raise here.
    """
    if shared_targets in {"", "none"}:
        return ()
    targets = resolve_targets(shared_targets)
    if moe_scope == "ffn":
        collide = [t for t in targets if t in {"gate_proj", "up_proj", "down_proj"}]
        if collide:
            raise ValueError(
                "Lifelong-MoE under --moe_scope ffn cannot put a shared "
                f"adapter on {collide}: those projections already host the "
                "routed expert pool, and LoRAMoEMLP.add_experts reads "
                "base_mlp.gate_proj.weight directly. Use "
                "--lifelong_shared_targets attn, or move to the ffn_attn "
                "scope and decide the shared path separately.")
    elif moe_scope == "ffn_attn":
        raise ValueError(
            "Lifelong-MoE under --moe_scope ffn_attn has no free projection "
            "for a shared adapter: the shared router already drives QKVO and "
            "FFN experts. Run it with --lifelong_shared_targets none, or "
            "settle the 'unfreeze the rest of the backbone' variant first.")
    return targets


def attach_shared_path(model, moe_scope: str, shared_targets: str, r: int,
                       alpha: float, dropout: float) -> int:
    """Attach Lifelong-MoE's shared adapter BEFORE the expert layout wraps it."""
    targets = validate_shared_path(moe_scope, shared_targets)
    if not targets:
        return 0
    return attach_seq_lora_targets(model, targets, r, alpha, dropout)


def build_scope(name: str) -> MoEScope:
    try:
        return SCOPES[name]()
    except KeyError:
        raise ValueError(
            f"unknown MoE scope {name!r}; choose {sorted(SCOPES)}") from None


class RouterLogitCapture:
    """Recompute per-token router logits alongside the ordinary forward.

    A pre-hook stores each router's input; the logits are then reproduced with
    the same bias-free GEMM the layout performs internally.  This costs one
    ``[tokens, hidden] x [hidden, experts]`` product per layer -- with at most
    eight experts that is far below noise next to an 8B forward -- and keeps
    the shared layout modules read-only.
    """

    def __init__(self, scope: MoEScope, model):
        self.scope = scope
        self.model = model
        self._inputs: Dict[int, torch.Tensor] = {}
        self._handles = []
        self._hosts = []

    def __enter__(self):
        for index, (host, _) in enumerate(self.scope.router_hosts(self.model)):
            self._hosts.append(host)

            def hook(module, inputs, index=index):
                self._inputs[index] = inputs[0]

            self._handles.append(host.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._inputs.clear()
        self._hosts.clear()
        return False

    def logits(self, active_experts: int) -> List[torch.Tensor]:
        out = []
        for index, host in enumerate(self._hosts):
            hidden = self._inputs.get(index)
            if hidden is None:
                continue
            weight = self.scope._router_of(host).weight[:active_experts]
            flat = hidden.reshape(-1, hidden.shape[-1])
            out.append(F.linear(flat, weight, bias=None))
        if not out:
            raise RuntimeError("no router inputs were captured this forward")
        return out


# ---------------------------------------------------------------------------
# Shared trainer
# ---------------------------------------------------------------------------

class MoEBaselineTrainer(Ours_LoRA_MoE):
    """Engine shared by the two expansion baselines.

    Inherits Ours' deterministic fixed-subset bookkeeping, replay loaders and
    per-phase engine rebuild, and replaces only the pieces that assume the FFN
    layout (token mask, router-loss collection, router-row freezing) with scope
    calls.
    """

    method_name = "moe_baseline"

    def __init__(self, model, tokenizer, optimizer, train_task_list,
                 eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list,
                         eval_task_list, test_task_list, args)
        self.scope = build_scope(args.moe_scope)
        type(self).save_key_substrings = list(self.scope.save_key_substrings)
        # Ours_LoRA_MoE.train_continual/_write_workload_records read this off
        # args without a default. Set it to the Table-1 identity so the
        # inherited accounting works; _write_workload_records then rewrites the
        # method field, since the inherited string is "ours_lora_moe_<this>".
        args.training_version = f"tab1_{self.method_name}_{self.scope.name}"

    # -- loop ---------------------------------------------------------------
    def _moe_run_epochs(self, dataloader, epochs, device, phase_name,
                        frozen_router_prefix=None, include_moe_loss=True,
                        step_loss_fn=None, before_forward_fn=None):
        """Ours' epoch loop with scope-dispatched router plumbing.

        ``step_loss_fn(batch, outputs, capture, task_label)`` returns an
        additive term or ``None``; MoE-LPR uses it for the routing
        cross-entropy and Lifelong-MoE for its distillation term.

        ``before_forward_fn(batch)`` runs BEFORE the student forward. Anything
        that pushes a second forward through the model has to go there: with
        non-reentrant gradient checkpointing, a forward inserted between the
        student forward and its backward leaves the checkpoint bookkeeping
        inconsistent and backward dies with "A different number of tensors was
        saved during the original forward and recomputation".
        """
        args = self.args
        total = epochs * len(dataloader)
        bar = tqdm(total=total, leave=True, disable=(args.global_rank != 0))
        needs_capture = step_loss_fn is not None
        for epoch in range(epochs):
            print_rank_0(f"{phase_name}: epoch {epoch+1}/{epochs}, "
                         f"{len(dataloader)} steps", args.global_rank)
            self.model.train()
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            step_cap = int(getattr(args, "max_train_steps_per_task", 0))
            for step, batch in enumerate(dataloader):
                if step_cap and step >= step_cap:
                    break
                batch.pop("sources", None)
                task_label = batch.pop("_task_label", None)
                batch = to_device(batch, device)
                accum = args.gradient_accumulation_steps
                window_start = (step // accum) * accum
                window_size = min(accum, len(dataloader) - window_start)
                is_window_end = (step - window_start + 1) == window_size
                sync = (self.model.no_sync()
                        if isinstance(self.model, DDP) and not is_window_end
                        else nullcontext())
                if before_forward_fn is not None:
                    before_forward_fn(batch)
                self.scope.set_token_mask(
                    self.raw_model, batch.get("attention_mask"))
                capture_ctx = (RouterLogitCapture(self.scope, self.raw_model)
                               if needs_capture else nullcontext())
                try:
                    with sync, capture_ctx as capture:
                        outputs = self.model(**batch, use_cache=False)
                        moe_loss = (self.scope.collect_losses(self.raw_model)
                                    if include_moe_loss else None)
                        loss = (outputs.loss if moe_loss is None
                                else outputs.loss + moe_loss)
                        if step_loss_fn is not None:
                            extra = step_loss_fn(batch, outputs, capture,
                                                 task_label)
                            if extra is not None:
                                loss = loss + extra
                        (loss / window_size).backward()
                finally:
                    self.scope.set_token_mask(self.raw_model, None)
                if args.global_rank == 0:
                    bar.update(1)
                    if (step % args.loss_log_interval == 0
                            or step + 1 == len(dataloader)):
                        bar.set_description(
                            f"{phase_name} | epoch {epoch+1} step {step} "
                            f"loss {loss.detach().float().cpu().item():.4f}",
                            refresh=False)
                if is_window_end:
                    if frozen_router_prefix is not None:
                        old_count, snapshots = frozen_router_prefix
                        self.scope.restore_router_rows(
                            self.raw_model, snapshots, old_count)
                    trainable = [p for p in self.raw_model.parameters()
                                 if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    self.optimizer.step()
                    if frozen_router_prefix is not None:
                        self.scope.restore_router_rows(
                            self.raw_model, snapshots, old_count)
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

    def _write_workload_records(self):
        """Relabel the inherited workload file for this method.

        ``Ours_LoRA_MoE._write_workload_records`` hardcodes
        ``ours_lora_moe_<training_version>`` as the method name. Left alone it
        would stamp a Table-1 baseline's per-task FLOP and wall-time record as
        an Ours run, which is exactly the provenance mix-up the separate
        directories exist to prevent.
        """
        super()._write_workload_records()
        if self.args.global_rank != 0:
            return
        path = os.path.join(self.args.output_dir, "training_workload.json")
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["method"] = f"tab1_{self.method_name}"
        payload["moe_scope"] = self.scope.name
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)

    def write_meta(self, round_index, **extra):
        if self.args.global_rank != 0:
            return
        directory = os.path.join(self.args.output_dir, str(round_index))
        os.makedirs(directory, exist_ok=True)
        meta = {
            "method": self.method_name,
            "moe_scope": self.scope.name,
            "r": self.args.lora_moe_rank,
            "alpha": self.args.lora_moe_alpha,
            "dropout": self.args.lora_moe_dropout,
            "top_k": self.args.top_k,
            "routing_weight_mode": self.args.routing_weight_mode,
            "experts_per_task": self.args.experts_per_task,
            "num_experts": self.scope.num_experts(self.raw_model),
            "aux_loss_coeff": self.args.moe_aux_loss_coeff,
            "z_loss_coeff": self.args.moe_z_loss_coeff,
            "round": round_index,
        }
        if self.scope.name == "ffn_attn":
            # Same field names load_v3_checkpoint validates, so a pure-MoE
            # Table-1 checkpoint also opens with the existing Ours evaluator.
            from model.Ours_LoRA_MoE_V3 import ATTENTION_TARGETS, V3_ARCHITECTURE
            meta["architecture"] = V3_ARCHITECTURE
            meta["attention_targets"] = list(ATTENTION_TARGETS)
            meta["attention_rank"] = self.args.lora_moe_rank
        meta.update(extra)
        filenames = ["tab1_meta.json"]
        # A Lifelong-MoE run with a shared adapter carries .lora. keys the Ours
        # loaders do not build, so publishing a lora_moe_meta.json beside it
        # would advertise a checkpoint the Ours evaluator then rejects. Only
        # the pure-expert layouts get the compatibility file.
        if not self._has_shared_adapter():
            filenames.append("lora_moe_meta.json")
        for filename in filenames:
            with open(os.path.join(directory, filename), "w",
                      encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)

    def _has_shared_adapter(self):
        return any(".lora." in name
                   for name, _ in self.raw_model.named_parameters())

    def save_model(self, round):
        # Ours' save_model writes a lora_moe_meta.json describing the Ours
        # recipe; go straight to the base implementation and write our own.
        from model.base_model import CL_Base_Model
        CL_Base_Model.save_model(self, round)
        self.write_meta(round)


# ---------------------------------------------------------------------------
# Lifelong-MoE
# ---------------------------------------------------------------------------

class LifelongMoE(MoEBaselineTrainer):
    """Frozen old experts and router rows; everything else keeps training.

    Per task: grow ``experts_per_task`` experts, hold the old router rows
    bit-identical across every optimizer update, and train the new experts, the
    new router rows and the shared adapter.  The distillation term is the
    paper's output-level regularizer: the pre-task model scores the same batch
    and the current model is pulled towards its distribution.

    The teacher is this model with (a) the expert prefix limited to the old
    count and (b) the shared adapter rolled back to its pre-task values, so no
    second 8B backbone is held.  Rolling the adapter back and forward costs two
    copies of the adapter per step, which is negligible next to the forward.
    """

    method_name = "lifelong_moe"

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        self._shared_snapshot: Optional[Dict[str, torch.Tensor]] = None
        self._l2_anchor: Optional[Dict[str, torch.Tensor]] = None
        self._teacher_logits: Optional[torch.Tensor] = None
        expected = validate_shared_path(
            self.args.moe_scope, self.args.lifelong_shared_targets)
        if not expected and not getattr(
                self.args, "lifelong_allow_no_shared_path", 0):
            # Lifelong-MoE's forgetting comes from the shared path that keeps
            # being retrained on every task -- in the wiki study that channel
            # is what drives FM from .1665 at 0.25x to .2139 at 4x while every
            # other expansion method stays flat.  Strip it and what remains is
            # "grow, freeze, distil", which forgets about as little as Ours and
            # is no longer the published method.  Reporting that number as
            # Lifelong-MoE would be wrong, so it takes an explicit opt-in.
            raise ValueError(
                "Lifelong-MoE with no shared adapter is not Lifelong-MoE: the "
                "continually retrained shared path is the mechanism the paper's "
                f"forgetting comes from. --moe_scope {self.args.moe_scope} "
                f"with --lifelong_shared_targets "
                f"{self.args.lifelong_shared_targets} leaves nothing shared. "
                "Run Lifelong-MoE at --moe_scope ffn, or pass "
                "--lifelong_allow_no_shared_path 1 and report the row as an "
                "ablation rather than as Lifelong-MoE.")
        attached = len(self._shared_parameters())
        if bool(expected) != bool(attached):
            raise RuntimeError(
                f"--lifelong_shared_targets {self.args.lifelong_shared_targets} "
                f"expects {'a' if expected else 'no'} shared adapter but the "
                f"model carries {attached} shared LoRA tensors; "
                "attach_shared_path must run before the expert layout")

    # -- shared adapter ----------------------------------------------------
    def _shared_parameters(self):
        return [(name, parameter)
                for name, parameter in self.raw_model.named_parameters()
                if ".lora." in name]

    def _snapshot_shared(self):
        return {name: parameter.detach().clone()
                for name, parameter in self._shared_parameters()}

    @contextmanager
    def _pre_task_model(self, old_expert_count):
        """Temporarily restore the model as it was when the last task ended."""
        saved = None
        if self._shared_snapshot:
            saved = self._snapshot_shared()
            with torch.no_grad():
                for name, parameter in self._shared_parameters():
                    parameter.copy_(self._shared_snapshot[name])
        try:
            if old_expert_count > 0:
                with self.scope.limit_experts(self.raw_model, old_expert_count):
                    yield
            else:
                # Nothing was learned yet; the teacher is the bare backbone.
                with self.scope.limit_experts(self.raw_model, 0):
                    yield
        finally:
            if saved is not None:
                with torch.no_grad():
                    for name, parameter in self._shared_parameters():
                        parameter.copy_(saved[name])

    # -- distillation ------------------------------------------------------
    def _l2_penalty(self):
        """lambda * ||theta_shared - theta_shared_prev||^2 (paper Eq. 5).

        The wiki fork implements this alongside the KL term and runs with it at
        zero; it is exposed here so the two studies can be configured the same
        way.  Only parameters that existed in the previous task are anchored --
        pulling a freshly grown expert toward its random init would be wrong,
        and the shared adapter is the only thing that carries over anyway.
        """
        if self.args.lifelong_l2_coeff <= 0 or not self._l2_anchor:
            return None
        total = None
        for name, parameter in self._shared_parameters():
            reference = self._l2_anchor.get(name)
            if reference is None:
                continue
            term = ((parameter - reference.to(parameter.device)) ** 2).sum()
            total = term if total is None else total + term
        if total is None:
            return None
        return self.args.lifelong_l2_coeff * total

    def _score_teacher(self, batch, old_expert_count):
        """Cache the pre-task model's logits for this batch.

        Runs before the student forward, never between it and backward -- see
        ``_moe_run_epochs``. Only the logits survive; they are detached, so the
        teacher contributes no graph.
        """
        self._teacher_logits = None
        if self.args.lifelong_kd_coeff <= 0 or old_expert_count == 0:
            return
        with torch.no_grad(), self._pre_task_model(old_expert_count):
            self.scope.set_token_mask(
                self.raw_model, batch.get("attention_mask"))
            try:
                self._teacher_logits = self.raw_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False).logits.detach()
            finally:
                self.scope.set_token_mask(self.raw_model, None)

    def _kd_loss(self, batch, student_logits):
        args = self.args
        teacher_logits = self._teacher_logits
        if teacher_logits is None:
            return None
        temperature = args.lifelong_kd_temperature
        mask = batch["attention_mask"].reshape(-1).bool()
        student = student_logits.reshape(-1, student_logits.shape[-1])[mask]
        teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])[mask]
        if student.numel() == 0:
            return None
        # Accumulate over token chunks instead of one F.kl_div over the whole
        # batch. A micro-batch of 16 x 1024 tokens against Llama's 128k vocab
        # is 8.4 GB per fp32 [tokens, vocab] tensor, and the single-call form
        # holds five of them at once (two casts, two log_softmax results, the
        # kl_div output) -- ~40 GB, which OOMs an 80 GB card the moment KD
        # first switches on at task 2. Ours solves this the same way; see
        # Ours_LoRA_MoE._kd_kl_loss and --v2_kd_chunk_tokens.
        chunk = max(1, int(getattr(args, "lifelong_kd_chunk_tokens", 256)))
        total = student.new_zeros((), dtype=torch.float)
        for start in range(0, student.shape[0], chunk):
            stop = min(start + chunk, student.shape[0])
            student_log_probs = F.log_softmax(
                student[start:stop].float() / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(
                teacher[start:stop].float() / temperature, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            total = total + torch.sum(
                teacher_probs * (teacher_log_probs - student_log_probs))
        loss = total * (temperature ** 2) / student.shape[0]
        return args.lifelong_kd_coeff * loss

    def train_one_task(self, task, i_task, epochs):
        args = self.args
        device = (torch.device("cuda", args.local_rank)
                  if args.local_rank != -1 else torch.device("cuda"))
        old_expert_count = self.scope.num_experts(self.raw_model)
        self.scope.add_experts(self.raw_model, args.experts_per_task)
        new_indices = set(range(old_expert_count,
                                old_expert_count + args.experts_per_task))
        old_rows = (self.scope.snapshot_router_rows(
            self.raw_model, old_expert_count) if old_expert_count else None)

        # Old experts and the router rows that address them are frozen; the new
        # experts, the new router rows and the shared adapter are not.
        self.scope.freeze_experts(self.raw_model,
                                  trainable_expert_indices=new_indices)
        self.scope.freeze_routers(self.raw_model, trainable=True)
        for _, parameter in self._shared_parameters():
            parameter.requires_grad = bool(args.lifelong_train_shared)

        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} lifelong")
        self._set_grad_ckpt(task in getattr(args, "ckpt_tasks", set()))
        loader = self.train_task_list[task]
        self._reinit_engine(self._optimizer_update_count(loader, epochs))

        def before_forward(batch):
            self._score_teacher(batch, old_expert_count)

        def step_loss(batch, outputs, capture, task_label):
            kd = self._kd_loss(batch, outputs.logits)
            l2 = self._l2_penalty()
            if kd is None:
                return l2
            return kd if l2 is None else kd + l2

        distils = args.lifelong_kd_coeff > 0 and old_expert_count > 0
        regularized = distils or args.lifelong_l2_coeff > 0
        self._moe_run_epochs(
            loader, epochs, device, f"{task} [lifelong-moe]",
            frozen_router_prefix=((old_expert_count, old_rows)
                                  if old_rows is not None else None),
            step_loss_fn=step_loss if regularized else None,
            before_forward_fn=before_forward if distils else None)
        self._shared_snapshot = self._snapshot_shared()
        self._l2_anchor = {name: value.detach().float().clone()
                           for name, value in self._shared_snapshot.items()}

    def save_model(self, round):
        # The shared adapter is part of the method, so it has to ride along in
        # the checkpoint next to the experts and routers.
        type(self).save_key_substrings = (
            list(self.scope.save_key_substrings) + [".lora."])
        super().save_model(round)
        if self.args.global_rank == 0:
            self.write_meta(
                round, kd_coeff=self.args.lifelong_kd_coeff,
                kd_temperature=self.args.lifelong_kd_temperature,
                shared_targets=self.args.lifelong_shared_targets,
                train_shared=self.args.lifelong_train_shared,
                l2_coeff=self.args.lifelong_l2_coeff)


# ---------------------------------------------------------------------------
# MoE-LPR
# ---------------------------------------------------------------------------

class _TaskTaggedDataset(Dataset):
    """Concatenated review stream that remembers which task each record is from.

    The shared SLoRA/TRACE collators take plain records, so the task id cannot
    ride inside the record without risking a key collision. It is carried
    beside the record instead and reattached by ``_TaskTaggedCollator``.
    """

    def __init__(self, parts):
        # parts: [(dataset, indices, task_index), ...]
        self._items = [(dataset, index, task_index)
                       for dataset, indices, task_index in parts
                       for index in indices]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, position):
        dataset, index, task_index = self._items[position]
        return dataset[index], task_index


class _TaskTaggedCollator:
    """Wrap a stock collator so the batch carries a per-sample task id."""

    def __init__(self, base):
        self.base = base

    def __call__(self, items):
        records = [record for record, _ in items]
        task_ids = torch.tensor([task for _, task in items], dtype=torch.long)
        batch = self.base(records)
        batch["_task_label"] = task_ids
        return batch


class MoELPR(MoEBaselineTrainer):
    """Two-phase upcycling: expert post-training, then router-only review.

    Phase 1 is identical to Ours' phase 1 -- new experts plus new router rows,
    old rows held fixed -- so the two methods share a plasticity mechanism and
    Table 1 reads as a difference in how the old distribution is protected.

    Phase 2 unfreezes only the router and trains it on a review stream with
    ``L = L_LM + gamma * L_LPR``. Three details follow the wiki implementation
    (``pretrain_gpt.py::_masked_task_group_lpr``) exactly, because gamma only
    transfers between the two studies if it means the same thing in both:

    1. **The newest task carries no LPR label.** The wiki config marks it "-"
       in ``--moe-lpr-task-expert-ranges`` and the loss skips it; its tokens
       still contribute the LM term. LPR exists to hold OLD data on its own
       experts; supervising the current task toward its own expert is an extra
       objective the paper does not have.
    2. **The review stream mixes every seen task with equal dataset weight**,
       newest included (``MIXED_DATA_WEIGHT_MODE=equal`` over wiki+code+conv).
       With a fixed review budget that means each task's share shrinks as the
       sequence grows -- the intended behaviour, not an accident.
    3. **The LPR term is a token SUM divided by the batch's valid-token
       count.** Megatron adds it straight to the loss NUMERATOR while
       converting other token-mean terms with
       ``_token_mean_to_loss_numerator``; reproducing that here means
       ``sum(nll over supervised tokens) / all valid tokens``, not a mean over
       supervised tokens. The difference is the supervised-token fraction, so a
       plain mean would silently rescale gamma.

    With one expert per task the paper's group log-sum-exp over an expert range
    reduces to the log-probability of that single expert, which is what the
    per-token gather below computes.
    """

    method_name = "moe_lpr"

    def _review_updates(self, task, epochs):
        """Review length as a fraction of this task's own training updates."""
        accumulation = self._gradient_accumulation_for_batch_size(
            self.args.batch_by_task[task], f"{task} review sizing")
        loader = self.train_task_list[task]
        task_updates = max(1, int(epochs) * math.ceil(len(loader) / accumulation))
        return max(1, round(self.args.lpr_review_fraction * task_updates))

    def _build_review_loader(self, task, i_task, updates):
        """Equal-weight stream over every seen task, tagged with its task id.

        Old tasks draw from their immutable fixed subsets; the current task
        draws from its full training split, mirroring the wiki blend where the
        two old languages came from replay memories and the new one did not.
        """
        args = self.args
        seen = list(self.train_task_list)[:i_task + 1]
        batch_size = min(args.batch_by_task[name] for name in seen)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_samples = updates * int(args.effective_global_batch)
        per_task = total_samples // len(seen)
        per_task -= per_task % (batch_size * world_size)
        if per_task <= 0:
            return None, batch_size

        parts = []
        for task_index, name in enumerate(seen):
            if name == task:
                source = self.train_task_list[name].dataset
            else:
                source = self._ensure_fixed_task_subset(name)
            indices = self._deterministic_exposure_indices(
                len(source), per_task, self._fixed_subset_seed() + 7919 * task_index)
            parts.append((source, indices, task_index))
            print_rank_0(
                f"  [lpr review] {name}: {per_task} exposures from "
                f"{len(source)} unique ({'current task' if name == task else 'fixed subset'})",
                args.global_rank)

        combined = _TaskTaggedDataset(parts)
        max_length = getattr(args, "max_train_len", 0) or (
            args.max_prompt_len + args.max_ans_len)
        if getattr(args, "train_format", "") == "slora_chat_full":
            if getattr(args, "use_pretokenized_train_cache", False):
                base = PreTokenizedSLoRATraceDataCollator(self.tokenizer)
            else:
                base = SLoRATraceDataCollator(self.tokenizer, max_length=max_length)
        else:
            base = DataCollator(
                self.tokenizer, padding="longest", max_prompt_len=max_length,
                max_ans_len=0, pad_to_multiple_of=8, inference=False)
        sampler = (RandomSampler(combined) if args.local_rank == -1
                   else DistributedSampler(combined, shuffle=True,
                                           seed=self._fixed_subset_seed()))
        loader = DataLoader(
            combined, collate_fn=_TaskTaggedCollator(base), sampler=sampler,
            batch_size=batch_size, num_workers=4, pin_memory=True)
        return loader, batch_size

    def _lpr_loss(self, batch, capture, task_labels, current_task_index,
                  active_experts):
        """gamma * mean_over_layers( sum_supervised(-log p[expert]) / valid tokens )."""
        gamma = self.args.lpr_gamma
        if gamma <= 0 or task_labels is None:
            return None
        valid = batch["attention_mask"].bool()
        token_task = task_labels.to(valid.device)[:, None].expand_as(valid)
        supervised = (valid & (token_task != current_task_index)).reshape(-1)
        valid_tokens = int(valid.sum())
        if not bool(supervised.any()) or valid_tokens == 0:
            return None
        target = token_task.reshape(-1)[supervised]
        old_group = current_task_index * self.args.experts_per_task
        label_mode = getattr(self.args, "lpr_label_mode", "old")

        total = None
        layer_logits = capture.logits(active_experts)
        for logits in layer_logits:
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            if label_mode == "old":
                # All older tasks share one label: the whole old-expert prefix.
                chosen = torch.logsumexp(log_probs[supervised, :old_group], dim=-1)
            else:
                chosen = log_probs[supervised].gather(-1, target[:, None]).squeeze(-1)
            term = -chosen.sum() / valid_tokens
            total = term if total is None else total + term
        return gamma * (total / len(layer_logits))

    def train_one_task(self, task, i_task, epochs):
        args = self.args
        device = (torch.device("cuda", args.local_rank)
                  if args.local_rank != -1 else torch.device("cuda"))
        self._ensure_fixed_task_subset(task)
        old_expert_count = self.scope.num_experts(self.raw_model)
        self.scope.add_experts(self.raw_model, args.experts_per_task)
        new_indices = set(range(old_expert_count,
                                old_expert_count + args.experts_per_task))
        old_rows = (self.scope.snapshot_router_rows(
            self.raw_model, old_expert_count) if old_expert_count else None)

        # -- phase 1: post-train the new experts on new-task data ------------
        self.scope.freeze_experts(self.raw_model,
                                  trainable_expert_indices=new_indices)
        self.scope.freeze_routers(self.raw_model, trainable=True)
        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} lpr phase1")
        self._set_grad_ckpt(task in getattr(args, "ckpt_tasks", set()))
        loader = self.train_task_list[task]
        self._reinit_engine(self._optimizer_update_count(loader, epochs))
        self._moe_run_epochs(
            loader, epochs, device, f"{task} [moe-lpr phase1]",
            frozen_router_prefix=((old_expert_count, old_rows)
                                  if old_rows is not None else None))

        # -- phase 2: review, router only ------------------------------------
        if args.lpr_review_fraction <= 0:
            return
        # Task 1 has no older task, so nothing carries an LPR label and the
        # review would be a router-only LM pass the paper does not describe.
        if i_task == 0:
            print_rank_0("  [lpr review] task 1 has no old task; skipped",
                         args.global_rank)
            return
        updates = self._review_updates(task, epochs)
        review_loader, batch_size = self._build_review_loader(task, i_task, updates)
        if review_loader is None:
            print_rank_0("  [lpr review] budget too small for one global batch",
                         args.global_rank)
            return
        active_experts = self.scope.num_experts(self.raw_model)
        self._set_grad_ckpt(True)
        self.scope.freeze_experts(self.raw_model, trainable_expert_indices=None)
        self.scope.freeze_routers(self.raw_model, trainable=True)
        self._set_phase_gradient_accumulation(
            batch_size, f"{task} lpr phase2 review")
        self._reinit_engine(self._optimizer_update_count(review_loader, 1))
        print_rank_0(
            f"  [lpr review] {updates} updates "
            f"({args.lpr_review_fraction:.0%} of the task's training), "
            f"gamma={args.lpr_gamma}, current task {i_task} unlabelled",
            args.global_rank)

        def step_loss(batch, outputs, capture, task_label):
            return self._lpr_loss(batch, capture, task_label, i_task,
                                  active_experts)

        self._moe_run_epochs(
            review_loader, 1, device, f"{task} [moe-lpr phase2 review]",
            include_moe_loss=False, step_loss_fn=step_loss)

    def _expert_index_for_task(self, task_index):
        """Routing label for a replay sample: the expert that task grew."""
        per_task = int(self.args.experts_per_task)
        if per_task != 1:
            raise ValueError(
                "the single-expert LPR log-probability is defined for "
                f"experts_per_task=1, got {per_task}; a multi-expert task "
                "needs the paper's group log-sum-exp over its expert range")
        return task_index

    def save_model(self, round):
        super().save_model(round)
        if self.args.global_rank == 0:
            self.write_meta(round, lpr_gamma=self.args.lpr_gamma,
                            lpr_label_mode=getattr(self.args, "lpr_label_mode", "old"),
                            lpr_review_fraction=self.args.lpr_review_fraction)


TAB1_MOE_TRAINERS = {
    "lifelong_moe": LifelongMoE,
    "moe_lpr": MoELPR,
}
