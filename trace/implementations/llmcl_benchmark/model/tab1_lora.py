"""Target-set-aware LoRA plumbing for the Table-1 continual-learning baselines.

``continual_lora`` fixes every adapter to the FFN triple because the five
baselines it was written for are all FFN-only.  Table 1 needs the same LoRA
parameterization on attention as well: EWC has to sit where Seq-LoRA sits to be
a fair control, O-LoRA's official Llama recipe touches ``q_proj``/``v_proj``,
and Lifelong-MoE's "everything that is not a frozen expert stays trainable"
becomes, on a frozen backbone, "LoRA on every projection the paper trains".

Nothing here re-implements a LoRA: ``LoRAPair``, ``SeqLoRALinear`` and
``OLoRALinear`` are imported from ``continual_lora`` unchanged, so the two
module families stay bit-comparable.  What this file adds is (a) attachment
driven by a named target set instead of a hardcoded one, (b) the merge step
O-LoRA performs after its last task, and (c) the Fisher estimator EWC needs.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from model.continual_lora import (
    LoRAPair, OLoRALinear, SeqLoRALinear, _decoder_layers, _freeze_backbone)


ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_TARGETS = ("gate_proj", "up_proj", "down_proj")

#: Named target sets.  ``all7`` is the SLoRA/Seq-LoRA contract every Table-1
#: row is normalised against; ``qv`` reproduces PEFT's Llama default, which is
#: what official O-LoRA silently uses because it passes no ``target_modules``.
TARGET_SETS: Dict[str, Tuple[str, ...]] = {
    "ffn": FFN_TARGETS,
    "attn": ATTN_TARGETS,
    "all7": ATTN_TARGETS + FFN_TARGETS,
    "qv": ("q_proj", "v_proj"),
}

TAB1_STATE_KEY_SUBSTRINGS = (".lora.", ".adapters.", ".experts.", ".router.",
                             ".shared_expert_router.")


def resolve_targets(name: str) -> Tuple[str, ...]:
    try:
        return TARGET_SETS[name]
    except KeyError:
        raise ValueError(
            f"unknown target set {name!r}; choose one of "
            f"{sorted(TARGET_SETS)}") from None


def _parent_of(layer: nn.Module, target: str) -> Optional[nn.Module]:
    """Return the submodule that owns ``target`` on one decoder layer."""
    if target in ATTN_TARGETS:
        return getattr(layer, "self_attn", None)
    if target in FFN_TARGETS:
        return getattr(layer, "mlp", None)
    raise ValueError(f"cannot place {target!r}: not an attention or FFN name")


def wrap_targets(model: nn.Module, targets: Sequence[str], factory,
                 freeze_backbone: bool = True) -> int:
    """Replace every named projection with ``factory(linear, target)``.

    Returns the number of wrapped projections so callers can assert that the
    layout they think they configured is the one that got built -- a silently
    empty target set would otherwise train nothing and only show up hours later
    as a flat loss curve.
    """
    if freeze_backbone:
        _freeze_backbone(model)
    wrapped = 0
    for layer in _decoder_layers(model):
        for target in targets:
            parent = _parent_of(layer, target)
            if parent is None:
                raise TypeError(f"decoder layer has no owner for {target}")
            linear = getattr(parent, target, None)
            if not isinstance(linear, nn.Linear):
                raise TypeError(
                    f"{target} is {type(linear).__name__}, expected nn.Linear")
            setattr(parent, target, factory(linear, target))
            wrapped += 1
    if wrapped == 0:
        raise ValueError(f"target set {tuple(targets)} matched no projection")
    return wrapped


def attach_seq_lora_targets(model: nn.Module, targets: Sequence[str], r: int,
                            alpha: float, dropout: float = 0.0) -> int:
    return wrap_targets(
        model, targets,
        lambda linear, _: SeqLoRALinear(linear, r, alpha, dropout))


def attach_olora_targets(model: nn.Module, targets: Sequence[str], r: int,
                         alpha: float, num_tasks: int,
                         dropout: float = 0.0) -> int:
    return wrap_targets(
        model, targets,
        lambda linear, _: OLoRALinear(linear, r, alpha, dropout, num_tasks))


def iter_wrapped(model: nn.Module, kind) -> Iterable[nn.Module]:
    return (module for module in model.modules() if isinstance(module, kind))


def iter_seq_lora(model: nn.Module) -> Iterable[SeqLoRALinear]:
    return iter_wrapped(model, SeqLoRALinear)


def iter_olora(model: nn.Module) -> Iterable[OLoRALinear]:
    return iter_wrapped(model, OLoRALinear)


def set_olora_task(model: nn.Module, task_index: int) -> None:
    for layer in iter_olora(model):
        layer.set_task(task_index)


def olora_regularization(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum the official O-LoRA penalties over every adapted projection.

    Official ``uie_trainer_lora.py`` accumulates ``|A_prev A_new^T|.sum()`` over
    the lora_A matrices and ``||theta||_2`` (a plain Frobenius norm, not its
    square) over every new adapter tensor, then adds them as
    ``loss + l1*orthogonal + l2*l2_loss``.  Both shapes are preserved here.
    """
    orthogonal: Optional[torch.Tensor] = None
    l2: Optional[torch.Tensor] = None
    for layer in iter_olora(model):
        layer_orthogonal = layer.orthogonal_loss()
        layer_l2 = layer.current_l2_loss()
        orthogonal = (layer_orthogonal if orthogonal is None
                      else orthogonal + layer_orthogonal)
        l2 = layer_l2 if l2 is None else l2 + layer_l2
    if orthogonal is None or l2 is None:
        raise RuntimeError("model has no O-LoRA layers")
    return orthogonal, l2


@torch.no_grad()
def merge_olora_into_base(model: nn.Module, upto_task: Optional[int] = None
                          ) -> int:
    """Fold every trained O-LoRA adapter back into its frozen base weight.

    O-LoRA grows one rank-r block per task, so after eight tasks the residual
    path is eight blocks deep.  The paper merges them at the end, which is what
    makes O-LoRA's inference cost identical to the untouched backbone; without
    this the method would be reported with 8x the adapter parameters of every
    other row.  Returns the number of merged projections.
    """
    merged = 0
    for layer in iter_olora(model):
        last = layer.active_task if upto_task is None else int(upto_task)
        weight = layer.base.weight
        delta = torch.zeros_like(weight, dtype=torch.float32)
        for adapter in layer.adapters[:last + 1]:
            delta += (adapter.B.float() @ adapter.A.float()) * adapter.scaling
        weight.add_(delta.to(dtype=weight.dtype))
        merged += 1
    return merged


def adapter_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Total / trainable / adapter-only parameter counts for the run manifest."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = 0
    for module in model.modules():
        if isinstance(module, LoRAPair) or getattr(module, "_tab1_adapter", False):
            adapter += module.parameter_count
        elif isinstance(module, nn.Linear) and getattr(
                module, "_tab1_router", False):
            adapter += sum(p.numel() for p in module.parameters())
    return {"total": total, "trainable": trainable, "adapter": adapter}


# --------------------------------------------------------------------------
# EWC
# --------------------------------------------------------------------------

class EWCState:
    """Diagonal Fisher and parameter anchors for the quadratic EWC penalty.

    Two things separate this from ``TRACE/EWC.py`` (and from the port in
    ``paper_baselines.EWCLoRA``, which replicates it on purpose):

    1. The Fisher is estimated AFTER a task converges, in a dedicated pass with
       the penalty switched off.  TRACE accumulates ``grad**2`` inside the
       training loop, so from task 2 on its "Fisher" is the curvature of the
       regularised objective, not of the task likelihood.
    2. It is normalised by the number of contributing samples.  TRACE divides
       by ``len(dataloader)`` -- a batch count -- and its per-task reset is
       commented out, so the running sum grows without bound across tasks and
       lambda no longer means what the paper says it means.

    ``mode="online"`` keeps one accumulated Fisher and anchors at the most
    recent task (Schwarz et al.); ``mode="per_task"`` keeps the textbook sum of
    one penalty per completed task, at 8x the state.

    ``device`` is where the Fisher and anchors live between steps. Keeping them
    on the accelerator costs 2 x the trainable size (~1.3 GB for r64 LoRA on
    all seven projections of an 8B model, x tasks in per_task mode) and keeps
    the penalty a pure device computation; parking them on the host instead
    moves that much over PCIe on EVERY training step, which measured 11 s/step
    against 0.35 s/step for the same run.
    """

    def __init__(self, mode: str = "online", device: Optional[str] = None):
        if mode not in {"online", "per_task"}:
            raise ValueError(f"unknown EWC mode: {mode}")
        self.mode = mode
        self.device = device
        self.fisher: Dict[str, torch.Tensor] = {}
        self.anchor: Dict[str, torch.Tensor] = {}
        self.terms: List[Tuple[Dict[str, torch.Tensor],
                               Dict[str, torch.Tensor]]] = []
        self.tasks_seen = 0

    def __bool__(self) -> bool:
        return self.tasks_seen > 0

    def absorb(self, fisher: Dict[str, torch.Tensor],
               anchor: Dict[str, torch.Tensor]) -> None:
        if self.device is not None:
            fisher = {key: value.to(self.device) for key, value in fisher.items()}
            anchor = {key: value.to(self.device) for key, value in anchor.items()}
        if self.mode == "per_task":
            self.terms.append((fisher, anchor))
        else:
            if not self.fisher:
                self.fisher = {k: v.clone() for k, v in fisher.items()}
            else:
                for key, value in fisher.items():
                    self.fisher[key] += value
            self.anchor = {k: v.clone() for k, v in anchor.items()}
        self.tasks_seen += 1

    def penalty(self, named_parameters) -> Optional[torch.Tensor]:
        """0.5-free quadratic term; the caller applies ``0.5 * lambda``."""
        if not self.tasks_seen:
            return None
        pairs = (self.terms if self.mode == "per_task"
                 else [(self.fisher, self.anchor)])
        total: Optional[torch.Tensor] = None
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            for fisher, anchor in pairs:
                weight = fisher.get(name)
                if weight is None:
                    continue
                # Compute in fp32, never in the parameter's dtype. Casting the
                # Fisher DOWN to bf16 destroys the term: one task's parameter
                # drift is ~1e-4, its square ~1e-8, and multiplying by a Fisher
                # entry of ~1e-7 lands at ~1e-15 -- summing 168M of those in
                # bf16's 8-bit mantissa rounds the whole penalty to exactly
                # zero, which is what the first smoke run printed.
                reference = anchor[name].to(device=parameter.device,
                                            dtype=torch.float32)
                weight = weight.to(device=parameter.device,
                                   dtype=torch.float32)
                term = (weight * (parameter.float() - reference) ** 2).sum()
                total = term if total is None else total + term
        return total

    def state_summary(self) -> Dict[str, object]:
        if self.mode == "per_task":
            tensors = [t for fisher, _ in self.terms for t in fisher.values()]
        else:
            tensors = list(self.fisher.values())
        if not tensors:
            return {"mode": self.mode, "tasks_seen": 0}
        flat = torch.cat([t.reshape(-1).float().cpu() for t in tensors])
        return {
            "mode": self.mode,
            "tasks_seen": self.tasks_seen,
            "fisher_mean": float(flat.mean()),
            "fisher_max": float(flat.max()),
            "fisher_nonzero_fraction": float((flat > 0).float().mean()),
        }


def estimate_diagonal_fisher(model: nn.Module, dataloader, device,
                             max_samples: int = 0,
                             store_device: Optional[str] = None,
                             log_fn=None) -> Dict[str, torch.Tensor]:
    """Empirical diagonal Fisher of the trainable parameters, post-convergence.

    Uses the model's own causal-LM loss on the task's training records, which
    is the empirical Fisher every EWC-for-LLM implementation uses; sampling
    labels from the model would be the true Fisher but costs a generation pass
    per record for no reported benefit at this scale.

    ``max_samples=0`` walks the whole loader.  Gradients are squared per
    micro-batch and averaged over contributing samples, and the result is
    all-reduced so every rank ends a task with the same Fisher.

    The running sum stays on the accelerator and only moves to ``store_device``
    once at the end.  Accumulating on the host instead costs one device-to-host
    copy of every trainable tensor per micro-batch -- for r64 LoRA on all seven
    projections of an 8B model that is ~0.7 GB over PCIe per batch, which left
    the GPU idle at 0% utilisation and made the pass slower than the training
    it follows.
    """
    was_training = model.training
    model.eval()
    # ``None`` keeps the result where it was computed, which is what every
    # caller wants; an explicit store_device only exists for the memory-starved
    # case, and it is the slow path.
    store_device = device if store_device is None else store_device
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    accumulator = {n: torch.zeros_like(p, dtype=torch.float32, device=device)
                   for n, p in named}
    seen = 0
    for batch in dataloader:
        batch = dict(batch)
        batch.pop("sources", None)
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        outputs = model(**batch, use_cache=False)
        outputs.loss.backward()
        count = int(batch["input_ids"].shape[0])
        for name, parameter in named:
            if parameter.grad is not None:
                accumulator[name].addcmul_(
                    parameter.grad.detach().float(),
                    parameter.grad.detach().float(), value=float(count))
        seen += count
        if max_samples and seen >= max_samples:
            break
    model.zero_grad(set_to_none=True)

    totals = torch.tensor([float(seen)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        for name in accumulator:
            dist.all_reduce(accumulator[name], op=dist.ReduceOp.SUM)
    global_samples = max(1.0, float(totals.item()))
    for name in accumulator:
        accumulator[name] = (accumulator[name] / global_samples).to(store_device)
    if log_fn is not None:
        log_fn(f"  [ewc] Fisher over {int(global_samples)} samples "
               f"({len(accumulator)} tensors)")
    if was_training:
        model.train()
    return accumulator


def snapshot_parameters(model: nn.Module,
                        store_device: Optional[str] = None
                        ) -> Dict[str, torch.Tensor]:
    """``None`` leaves each anchor on the device its parameter already lives on."""
    return {name: (parameter.detach().float().clone() if store_device is None
                   else parameter.detach().float().to(store_device).clone())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad}
