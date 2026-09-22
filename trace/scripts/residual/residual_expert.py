"""Residual (no-op) expert for V3 shared-router LoRA-MoE checkpoints.

Each decoder layer's SharedExpertRouter gains extra output rows that own no
LoRA pair.  A token routed to one of them gets *no* expert delta in that layer
(Q/K/V/O and FFN all fall back to the frozen base projection), i.e. the layer
behaves like the pretrained backbone for that token.

Mechanics (stock V3 code is not edited; the router forward is wrapped):
* the router scores ``num_experts + n_residual`` rows; top-k runs over all rows;
* the returned RoutingContext reports ``num_experts = <real experts>``, so
  ``active_expert_ids`` drops residual ids and no projection touches them;
* gradient path into the residual row: the straight-through weight of a real
  expert is its full-softmax prob, whose denominator contains the residual
  logit, so tokens whose expert delta hurts push the residual logit up;
* ``second_choice`` (train only): a token that picked residual is dispatched to
  its best *real* expert with weight ``p - p.detach()`` (exactly 0 forward, so
  the output is still the residual/base path) -- that expert's delta supplies
  the gradient that lets the token leave the residual row again.
"""
import json
import os

import torch
import torch.nn.functional as F

RESIDUAL_META = "residual_expert_meta.json"


def _v3():
    from model import Ours_LoRA_MoE_V3 as V3
    return V3


def add_residual_experts(model, n_residual=1, init="zeros", second_choice=True):
    V3 = _v3()
    layers = V3.shared_router_layers(model)
    if not layers:
        raise ValueError("model has no V3 shared-router layers")
    for layer in layers:
        router = layer.shared_expert_router
        if getattr(router, "_n_residual", 0):
            raise RuntimeError("residual experts already attached")
        real = router.num_experts
        old = router.router
        new = torch.nn.Linear(old.in_features, real + n_residual, bias=False,
                              device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:real].copy_(old.weight)
            if init == "zeros":
                new.weight[real:].zero_()
            elif init == "mean":
                new.weight[real:].copy_(old.weight.mean(0, keepdim=True))
            else:
                raise ValueError(init)
        router.router = new
        router._n_residual = n_residual
        router._n_real = real
        router._second_choice = second_choice
        router.forward = _residual_forward.__get__(router, type(router))
    return model


def _residual_forward(self, hidden_states):
    V3 = _v3()
    real = self._n_real
    total = real + self._n_residual
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = self.router(flat_hidden)[..., :total]
    k = min(self.top_k, total)
    topk_logits, topk_indices = logits.topk(k, dim=-1)
    full_probs = F.softmax(logits, dim=-1, dtype=torch.float)
    selected = full_probs.gather(-1, topk_indices)
    if self.routing_weight_mode == "straight_through_topk":
        normalized = F.softmax(topk_logits, dim=-1)
        weights = normalized.detach() + selected - selected.detach()
    elif self.routing_weight_mode == "full_softmax":
        weights = selected
    else:
        weights = F.softmax(topk_logits, dim=-1)

    is_residual = topk_indices >= real
    if self._second_choice and self.training and torch.is_grad_enabled() \
            and bool(is_residual.any()):
        # best real expert per token, used only for its gradient
        real_best = logits[..., :real].argmax(-1, keepdim=True).expand_as(topk_indices)
        real_prob = full_probs.gather(-1, real_best)
        topk_indices = torch.where(is_residual, real_best, topk_indices)
        weights = torch.where(is_residual, real_prob - real_prob.detach(), weights)

    valid_mask = self._router_token_mask
    if valid_mask is not None:
        valid_mask = valid_mask.reshape(-1).to(device=logits.device, dtype=torch.bool)
    self._last_moe_loss = (
        self._router_loss(logits, topk_indices.clamp(max=total - 1), full_probs, valid_mask)
        if self.training and not self._suppress_router_loss else None)
    if getattr(self, "_record_residual", False):
        rec = is_residual if valid_mask is None else is_residual[valid_mask]
        self._residual_hits = getattr(self, "_residual_hits", 0) + int(rec.sum())
        self._residual_total = getattr(self, "_residual_total", 0) + int(rec.numel())
    self._last_probe_indices = None
    return V3.RoutingContext(
        expert_indices=topk_indices,
        expert_weights=weights,
        valid_token_mask=valid_mask,
        num_experts=real,
    )


def router_parameters(model):
    V3 = _v3()
    return [layer.shared_expert_router.router.weight
            for layer in V3.shared_router_layers(model)]


def router_state(model):
    V3 = _v3()
    return {f"layer{i}.router.weight": layer.shared_expert_router.router.weight.detach().cpu()
            for i, layer in enumerate(V3.shared_router_layers(model))}


def save_router_checkpoint(model, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(router_state(model), os.path.join(out_dir, "router_state.pt"))
    with open(os.path.join(out_dir, RESIDUAL_META), "w") as f:
        json.dump(meta, f, indent=2)


def load_router_tuned(out_dir, tokenizer, base, device="cuda", dtype=torch.bfloat16):
    """V3 source checkpoint + (optional residual rows) + tuned router weights."""
    import evaluate_Ours_LoRA_MoE as E
    with open(os.path.join(out_dir, RESIDUAL_META)) as f:
        meta = json.load(f)
    model, v3meta = E.load_v3_checkpoint(meta["source_checkpoint"], tokenizer,
                                         base_model_name_or_path=base,
                                         device=device, dtype=dtype, device_map=None)
    if meta.get("n_residual", 0):
        add_residual_experts(model, meta["n_residual"], second_choice=False)
    state = torch.load(os.path.join(out_dir, "router_state.pt"), map_location="cpu")
    V3 = _v3()
    for i, layer in enumerate(V3.shared_router_layers(model)):
        w = layer.shared_expert_router.router.weight
        with torch.no_grad():
            w.copy_(state[f"layer{i}.router.weight"].to(w.device, w.dtype))
    model.to(device=device, dtype=dtype).eval()
    return model, {**v3meta, **meta}


def load_v3_any_checkpoint(checkpoint_dir, tokenizer, base, device="cuda",
                           dtype=torch.bfloat16, device_map=None):
    """Stock V3 loader, or the residual-aware one when the checkpoint was
    trained with a residual row from task 0 (its meta carries residual_expert)."""
    V3 = _v3()
    with open(os.path.join(checkpoint_dir, V3.V3_META_NAME)) as f:
        meta = json.load(f)
    if "residual_expert" in meta:
        return load_v3_residual_checkpoint(checkpoint_dir, tokenizer, base,
                                           device=device, dtype=dtype)
    return V3.load_v3_checkpoint(checkpoint_dir, tokenizer, base, device=device,
                                 dtype=dtype, device_map=device_map)


def load_v3_residual_checkpoint(checkpoint_dir, tokenizer, base, device="cuda",
                                dtype=torch.bfloat16):
    """Load a V3 checkpoint trained WITH a residual expert from task 0
    (train_residual_v3.py).  Mirrors load_v3_checkpoint, attaching the residual
    router before the state is loaded; builds on GPU directly."""
    import train_residual_v3 as TR  # installs the router/save patches
    from transformers import AutoModelForCausalLM
    from utils.model.model_utils import create_hf_model
    V3 = _v3()
    with open(os.path.join(checkpoint_dir, V3.V3_META_NAME)) as f:
        meta = json.load(f)
    if "residual_expert" not in meta:
        raise ValueError(f"{checkpoint_dir} has no residual_expert metadata")
    model = create_hf_model(AutoModelForCausalLM, base, tokenizer, disable_dropout=True,
                            torch_dtype=dtype, low_cpu_mem_usage=True,
                            forbid_vocab_growth=True, device_map={"": torch.cuda.current_device()})
    V3.attach_shared_qkvo_lora_moe(
        model, r=meta["r"], alpha=meta["alpha"], top_k=meta["top_k"],
        aux_loss_coeff=meta["aux_loss_coeff"], z_loss_coeff=meta["z_loss_coeff"],
        routing_weight_mode=meta["routing_weight_mode"], dropout=meta.get("dropout", 0.0))
    V3.add_v3_experts(model, meta["num_experts"])
    TR.attach_residual(model)
    state = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"),
                       map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:5]}")
    need = [k for k in missing if any(s in k for s in V3.Ours_LoRA_MoE_V3.save_key_substrings)]
    if need:
        raise RuntimeError(f"missing grown/residual keys: {need[:5]}")
    model.to(device=device, dtype=dtype).eval()
    return model, meta
