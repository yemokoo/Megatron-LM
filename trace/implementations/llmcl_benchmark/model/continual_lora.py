"""Paper-aligned FFN LoRA modules for continual-learning baselines.

All baselines in arXiv:2602.12587 use a frozen backbone and LoRA on the
FFN gate/up/down projections. This module keeps that parameterization fixed
and changes only the continual-learning mechanism.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_BASELINE_META = "paper_baseline_meta.json"
FFN_TARGETS = ("gate_proj", "up_proj", "down_proj")
PAPER_STATE_KEY_SUBSTRINGS = (".lora.", ".experts.", ".router.", ".adapters.")


class LoRAPair(nn.Module):
    """A standard rank-r residual: B(A(dropout(x))) * alpha/r."""

    def __init__(self, in_features: int, out_features: int, r: int,
                 alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Parameter(torch.empty(self.r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.dropout(x) @ self.A.t()) @ self.B.t()) * self.scaling

    @property
    def parameter_count(self) -> int:
        return self.A.numel() + self.B.numel()


class _FrozenLinearWrapper(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(base).__name__}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


class SeqLoRALinear(_FrozenLinearWrapper):
    """One shared LoRA adapter, updated sequentially over every task."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__(base)
        self.lora = LoRAPair(self.in_features, self.out_features, r, alpha, dropout)
        self.lora.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora(x)


class LoRAMoELinear(_FrozenLinearWrapper):
    """Sparse probability-weighted LoRA experts around one common linear map.

    This follows Ablustrund/LoRAMoE's equation
    ``base(x) + sum_e p(e|x) B_e A_e x``. The target paper's sparse variant is
    obtained with top_k smaller than num_experts.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float,
                 num_experts: int, top_k: int, routing_weight_mode: str,
                 aux_loss_coeff: float, z_loss_coeff: float):
        super().__init__(base)
        if num_experts < 1:
            raise ValueError("LoRAMoE needs at least one expert")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if routing_weight_mode not in {"full_softmax", "topk_softmax"}:
            raise ValueError(f"unknown routing mode: {routing_weight_mode}")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.routing_weight_mode = routing_weight_mode
        self.aux_loss_coeff = float(aux_loss_coeff)
        self.z_loss_coeff = float(z_loss_coeff)
        self.experts = nn.ModuleList([
            LoRAPair(self.in_features, self.out_features, r, alpha, dropout)
            for _ in range(self.num_experts)
        ])
        self.router = nn.Linear(self.in_features, self.num_experts, bias=False)
        self.experts.to(device=base.weight.device, dtype=base.weight.dtype)
        self.router.to(device=base.weight.device, dtype=base.weight.dtype)
        self._router_token_mask: Optional[torch.Tensor] = None
        self._last_router_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.base(x)
        flat_x = x.reshape(-1, x.shape[-1])
        logits = self.router(flat_x)
        selected_logits, selected = logits.topk(self.top_k, dim=-1)
        full_probs = F.softmax(logits, dim=-1, dtype=torch.float)
        if self.routing_weight_mode == "full_softmax":
            weights = full_probs.gather(-1, selected)
        else:
            weights = F.softmax(selected_logits, dim=-1, dtype=torch.float)

        flat_delta = base_result.new_zeros((flat_x.shape[0], self.out_features))
        # DDP must see a stable parameter graph although sparse routing can
        # select a different expert set on every rank/step. Scalar zero anchors
        # register every expert without evaluating inactive LoRA matmuls or
        # changing the forward value.
        ddp_anchor = logits.reshape(-1)[0] * 0.0
        for expert in self.experts:
            ddp_anchor = (ddp_anchor + expert.A.reshape(-1)[0] * 0.0 +
                          expert.B.reshape(-1)[0] * 0.0)
        flat_delta = flat_delta + ddp_anchor.to(flat_delta.dtype)
        selected_t = selected.t()
        valid = self._router_token_mask
        if valid is not None:
            valid = valid.reshape(-1).to(device=x.device, dtype=torch.bool)
            if valid.numel() != flat_x.shape[0]:
                raise ValueError("router token mask does not match hidden states")
        for expert_index, expert in enumerate(self.experts):
            slot, token_index = torch.where(selected_t == expert_index)
            if valid is not None and token_index.numel():
                keep = valid[token_index]
                slot, token_index = slot[keep], token_index[keep]
            if token_index.numel() == 0:
                continue
            expert_delta = expert(flat_x[token_index])
            weight = weights[token_index, slot, None].to(expert_delta.dtype)
            flat_delta.index_add_(0, token_index, expert_delta * weight)

        self._last_router_loss = (
            self._router_loss(logits, selected, full_probs, valid)
            if (self.training and
                (self.aux_loss_coeff != 0 or self.z_loss_coeff != 0))
            else None)
        return base_result + flat_delta.reshape_as(base_result)

    def _router_loss(self, logits: torch.Tensor, selected: torch.Tensor,
                     full_probs: torch.Tensor,
                     valid: Optional[torch.Tensor]) -> torch.Tensor:
        if valid is not None:
            logits, selected, full_probs = (
                logits[valid], selected[valid], full_probs[valid])
        n_tokens = logits.shape[0]
        if n_tokens == 0:
            return logits.sum() * 0.0
        counts = torch.bincount(
            selected.reshape(-1), minlength=self.num_experts).to(full_probs.dtype)
        probability_mass = full_probs.sum(dim=0)
        aux = torch.sum(counts * probability_mass) * (
            self.num_experts * self.aux_loss_coeff /
            (n_tokens * n_tokens * self.top_k))
        z = torch.mean(torch.square(torch.logsumexp(logits.float(), dim=-1)))
        return aux + self.z_loss_coeff * z

    @property
    def activated_adapter_parameters(self) -> int:
        return self.top_k * self.experts[0].parameter_count


class OLoRALinear(_FrozenLinearWrapper):
    """Original O-LoRA task-wise rank growth on an FFN linear map."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float,
                 num_tasks: int):
        super().__init__(base)
        if num_tasks < 1:
            raise ValueError("O-LoRA needs at least one task adapter")
        self.adapters = nn.ModuleList([
            LoRAPair(self.in_features, self.out_features, r, alpha, dropout)
            for _ in range(num_tasks)
        ])
        self.adapters.to(device=base.weight.device, dtype=base.weight.dtype)
        self.active_task = 0
        self.set_task(0)

    def set_task(self, task_index: int) -> None:
        if not 0 <= task_index < len(self.adapters):
            raise IndexError(task_index)
        self.active_task = int(task_index)
        for index, adapter in enumerate(self.adapters):
            trainable = index == self.active_task
            for parameter in adapter.parameters():
                parameter.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        for adapter in self.adapters[:self.active_task + 1]:
            result = result + adapter(x)
        return result

    def orthogonal_loss(self) -> torch.Tensor:
        current = self.adapters[self.active_task]
        if self.active_task == 0:
            return current.A.sum() * 0.0
        previous_a = torch.cat(
            [adapter.A.detach() for adapter in self.adapters[:self.active_task]],
            dim=0)
        return torch.abs(previous_a @ current.A.t()).sum()

    def current_l2_loss(self) -> torch.Tensor:
        current = self.adapters[self.active_task]
        return torch.norm(current.A, p=2) + torch.norm(current.B, p=2)


def _decoder_layers(model: nn.Module) -> Iterable[nn.Module]:
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen/Llama-style model.model.layers backbone")
    return layers


def _freeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _wrap_ffn(model: nn.Module, factory) -> nn.Module:
    _freeze_backbone(model)
    for layer in _decoder_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise TypeError("decoder layer has no .mlp")
        for target in FFN_TARGETS:
            linear = getattr(mlp, target, None)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"{target} is not nn.Linear")
            setattr(mlp, target, factory(linear, target))
    return model


def attach_seq_lora(model: nn.Module, r: int, alpha: float,
                    dropout: float = 0.0) -> nn.Module:
    return _wrap_ffn(
        model, lambda linear, _: SeqLoRALinear(linear, r, alpha, dropout))


def attach_loramoe(model: nn.Module, r: int, alpha: float,
                   num_experts: int, top_k: int = 1, dropout: float = 0.0,
                   routing_weight_mode: str = "full_softmax",
                   aux_loss_coeff: float = 0.0,
                   z_loss_coeff: float = 0.0) -> nn.Module:
    return _wrap_ffn(model, lambda linear, _: LoRAMoELinear(
        linear, r, alpha, dropout, num_experts, top_k,
        routing_weight_mode, aux_loss_coeff, z_loss_coeff))


def attach_olora(model: nn.Module, r: int, alpha: float, num_tasks: int,
                 dropout: float = 0.0) -> nn.Module:
    return _wrap_ffn(model, lambda linear, _: OLoRALinear(
        linear, r, alpha, dropout, num_tasks))


def iter_seq_lora_layers(model: nn.Module) -> Iterable[SeqLoRALinear]:
    return (module for module in model.modules()
            if isinstance(module, SeqLoRALinear))


def iter_loramoe_layers(model: nn.Module) -> Iterable[LoRAMoELinear]:
    return (module for module in model.modules()
            if isinstance(module, LoRAMoELinear))


def iter_olora_layers(model: nn.Module) -> Iterable[OLoRALinear]:
    return (module for module in model.modules()
            if isinstance(module, OLoRALinear))


def set_router_token_mask(model: nn.Module,
                          attention_mask: Optional[torch.Tensor]) -> None:
    for layer in iter_loramoe_layers(model):
        layer._router_token_mask = attention_mask


def collect_loramoe_loss(model: nn.Module) -> Optional[torch.Tensor]:
    total = None
    for layer in iter_loramoe_layers(model):
        if layer._last_router_loss is not None:
            total = (layer._last_router_loss if total is None
                     else total + layer._last_router_loss)
    return total


def set_olora_task(model: nn.Module, task_index: int) -> None:
    for layer in iter_olora_layers(model):
        layer.set_task(task_index)


def collect_olora_regularization(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    orthogonal = None
    l2 = None
    for layer in iter_olora_layers(model):
        layer_orthogonal = layer.orthogonal_loss()
        layer_l2 = layer.current_l2_loss()
        orthogonal = layer_orthogonal if orthogonal is None else orthogonal + layer_orthogonal
        l2 = layer_l2 if l2 is None else l2 + layer_l2
    if orthogonal is None or l2 is None:
        raise RuntimeError("model has no O-LoRA layers")
    return orthogonal, l2


def trainable_named_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def parameter_report(model: nn.Module, method: str) -> Dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    activated = 0
    adapters = 0
    routers = 0
    if method in {"seqlora", "ewc", "gem"}:
        layers = list(iter_seq_lora_layers(model))
        activated = sum(layer.lora.parameter_count for layer in layers)
        adapters = activated
    elif method == "loramoe":
        layers = list(iter_loramoe_layers(model))
        activated = sum(layer.activated_adapter_parameters for layer in layers)
        adapters = sum(expert.parameter_count
                       for layer in layers for expert in layer.experts)
        routers = sum(parameter.numel() for layer in layers
                      for parameter in layer.router.parameters())
    elif method == "olora":
        layers = list(iter_olora_layers(model))
        activated = sum(
            sum(adapter.parameter_count
                for adapter in layer.adapters[:layer.active_task + 1])
            for layer in layers)
        adapters = sum(adapter.parameter_count
                       for layer in layers for adapter in layer.adapters)
    if hasattr(model, "num_parameters"):
        non_embedding = int(model.num_parameters(exclude_embeddings=True))
    else:
        non_embedding = total
    activated_non_embedding = non_embedding - adapters + activated
    return {"total": total, "trainable": trainable,
            "non_embedding": non_embedding,
            "adapter_parameters": adapters,
            "router_parameters": routers,
            "activated_adapter": activated,
            "activated_non_embedding": activated_non_embedding}


def save_paper_baseline_meta(model: nn.Module, output_dir: str, method: str,
                             r: int, alpha: float, dropout: float,
                             **extra) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "method": method, "r": int(r), "alpha": float(alpha),
        "dropout": float(dropout),
    }
    meta.update(extra)
    meta["parameter_report"] = parameter_report(model, method)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, PAPER_BASELINE_META), "w") as handle:
        json.dump(meta, handle, indent=2)
    return meta


def load_paper_baseline_checkpoint(checkpoint_dir: str, tokenizer,
                                   base_model_name_or_path: str,
                                   device: torch.device | str = "cuda",
                                   dtype: torch.dtype = torch.bfloat16,
                                   device_map=None):
    """Rebuild a paper baseline from its partial FFN-LoRA checkpoint."""
    from transformers import AutoModelForCausalLM
    from utils.model.model_utils import create_hf_model

    meta_path = os.path.join(checkpoint_dir, PAPER_BASELINE_META)
    with open(meta_path) as handle:
        meta = json.load(handle)
    model = create_hf_model(
        AutoModelForCausalLM, base_model_name_or_path, tokenizer,
        disable_dropout=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        forbid_vocab_growth=True, device_map=device_map)
    method = meta["method"]
    if method in {"seqlora", "ewc", "gem"}:
        attach_seq_lora(model, meta["r"], meta["alpha"], meta["dropout"])
    elif method == "loramoe":
        attach_loramoe(
            model, meta["r"], meta["alpha"], meta["num_experts"],
            meta["top_k"], meta["dropout"], meta["routing_weight_mode"],
            meta["aux_loss_coeff"], meta["z_loss_coeff"])
    elif method == "olora":
        attach_olora(model, meta["r"], meta["alpha"], meta["num_tasks"],
                     meta["dropout"])
        set_olora_task(model, meta["current_task"])
    else:
        raise ValueError(f"unknown paper baseline method: {method}")
    state = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"),
                       map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys[:8]}")
    if device_map is None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model, meta
