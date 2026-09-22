"""BOS-guarded routing: at every shared-router layer EXCEPT the last, force
the residual (skip-connection, no expert delta) whenever the current
position's token is the BOS token.

Rationale: the backbone (self-attn + MLP) is frozen throughout training, so a
position that never picks a real expert produces a hidden state that is
mathematically identical to the plain backbone's own forward pass --
independent of which round/checkpoint we are at, since it depends on nothing
that training ever changes.  Only the LAST layer (which carries a trained
conditioning bias, see scripts/bos_token/train_bos_token.py) ever applies a
real expert delta at a BOS position, so the round-to-round divergence a
frozen bias fights against is reduced to one layer's decision instead of the
whole depth's accumulated routing history.

Composes on top of whatever SharedExpertRouter.forward is already installed
(train_residual_v3 / train_residual_v3_split) -- this module changes nothing
else: install it last, after those patches are in place.

  install_bos_guard(model, bos_token_id, header_ids)   # call once per model instance
    - tags each shared-router layer with whether it is the last one
    - wraps SharedExpertRouter.forward to override BOS-position routing
    - registers a forward pre-hook on `model` that computes the guarded
      position masks from `input_ids` automatically, so no caller (training
      loop or generation script) needs to thread anything through by hand

Header guard (header_ids given): wherever a row contains the fixed chat
template header (BOS BOS <system turn> <user header> "\n\n"), every header
position EXCEPT the last one is routed to the residual at ALL layers (their
hidden states / KV entries become exactly the plain backbone's, at every
round), and the last header token -- the position that predicts the first
user-turn content token -- is guarded at layers 1..L-1 only, so the last
layer (with its conditioning bias) is the single decision point.  Positions
outside a header keep the plain BOS rule above.
"""
from __future__ import annotations

import os
import torch

from model import Ours_LoRA_MoE_V3 as V3

BOS_TOKEN_ID = int(os.environ.get("BOS_GUARD_TOKEN_ID", "128000"))  # Llama-3.1 <|begin_of_text|>

_bos_mask = {"value": None, "full": None, "decision": None}
_prev_forward = None   # captured on first install; wrapped exactly once
HEADER_IDS = None      # default header for install_bos_guard(header_ids=None); set by callers


def set_bos_position_mask(mask, full_mask=None):
    """mask: (batch, seq) bool, positions guarded at layers 1..L-1 (BOS tokens + header decision
    token); full_mask: (batch, seq) bool, positions guarded at EVERY layer (header body), or None."""
    _bos_mask["value"] = mask
    _bos_mask["full"] = full_mask


def header_masks(input_ids, header_ids, all_full=False):
    """Locate every occurrence of `header_ids` in each row (position independent, so left padding
    and decode steps are fine).  Returns (full, part): full marks header positions except the last,
    part marks the last header position -- or, with all_full, every header position is in `full`
    and `part` is empty (the whole header is residual at every layer; the first token after it is
    the free decision position).  Both None when no row contains the header."""
    H = len(header_ids)
    if input_ids.shape[1] < H:
        return None, None
    hdr = torch.tensor(header_ids, device=input_ids.device)
    hit = (input_ids.unfold(1, H, 1) == hdr).all(-1)          # (batch, T-H+1): header starts here
    if not bool(hit.any()):
        return None, None
    full = torch.zeros_like(input_ids, dtype=torch.bool)
    part = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, start in hit.nonzero().tolist():
        if all_full:
            full[row, start:start + H] = True
        else:
            full[row, start:start + H - 1] = True
            part[row, start + H - 1] = True
    return full, part


def _tag_layers(model):
    layers = V3.shared_router_layers(model)
    for i, layer in enumerate(layers):
        layer.shared_expert_router._is_last_layer = (i == len(layers) - 1)


def decision_mask(input_ids, header_ids):
    """(batch, seq) bool marking the first position AFTER each header occurrence (the free decision
    token, e.g. "\n\n" for decision=none), or None when no row has such a position."""
    H = len(header_ids)
    if input_ids.shape[1] <= H:
        return None
    hdr = torch.tensor(header_ids, device=input_ids.device)
    hit = (input_ids.unfold(1, H, 1) == hdr).all(-1)
    dec = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, start in hit.nonzero().tolist():
        if start + H < input_ids.shape[1]:
            dec[row, start + H] = True
    return dec if bool(dec.any()) else None


def decision_position_mask():
    """Flat (batch*seq) bool mask of decision positions for the current forward, or None.
    Assign as `router._logit_bias_positions` to confine a logit bias to the decision token."""
    d = _bos_mask["decision"]
    return None if d is None else d.reshape(-1)


def apply_input_ids(model, input_ids):
    """Compute and install the guard masks for `input_ids` on a guarded model.  The model
    pre-hook does this automatically when the model is called with input_ids; callers that
    forward with inputs_embeds only (e.g. train_bos_token) must call this themselves first."""
    part = input_ids == model._bos_guard_token_id
    full = None
    _bos_mask["decision"] = None
    if model._bos_guard_header_ids:
        full, hpart = header_masks(input_ids, model._bos_guard_header_ids,
                                   getattr(model, "_bos_guard_header_all_full", False))
        if hpart is not None:
            part = part | hpart
        _bos_mask["decision"] = decision_mask(input_ids, model._bos_guard_header_ids)
    if FORCE_LAST_PROMPT_POS and input_ids.shape[1] > 1:      # prompt pass only; decode steps (T=1) stay free
        d = _bos_mask["decision"]
        d = torch.zeros_like(input_ids, dtype=torch.bool) if d is None else d.clone()
        d[:, -1] = True
        _bos_mask["decision"] = d
    set_bos_position_mask(part, full)


def _install_model_hooks(model, bos_token_id, header_ids):
    model._bos_guard_token_id = bos_token_id
    if getattr(model, "_bos_guard_hooks_installed", False):
        return

    def pre(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None and torch.is_tensor(input_ids):
            apply_input_ids(model, input_ids)
        return args, kwargs

    # No post-hook: gradient checkpointing recomputes each decoder layer's
    # forward during backward by calling the LAYER's forward again, not the
    # top-level model's -- a post-hook here would clear the mask before that
    # recomputation runs, making the original forward and the recompute take
    # different code paths (checkpoint requires them identical) and raising
    # torch.utils.checkpoint's tensor-count mismatch.  The mask is instead
    # left in place until the next top-level forward's pre-hook overwrites
    # it, which is always correct: every top-level call sets its own mask
    # before any layer runs, and no layer forward happens outside one.
    model.register_forward_pre_hook(pre, with_kwargs=True)
    model._bos_guard_hooks_installed = True


def _bos_guarded_forward(self, hidden_states):
    out = _prev_forward(self, hidden_states)
    full = _bos_mask["full"]
    if getattr(self, "_is_last_layer", False):
        mask = full                                 # last layer: only the header body; bias decides the rest
    else:
        mask = _bos_mask["value"] if full is None else (_bos_mask["value"] | full)
    if mask is None:
        return out
    flat_mask = mask.reshape(-1).to(device=out.expert_indices.device)
    if flat_mask.numel() != out.expert_indices.shape[0]:
        raise ValueError(
            f"bos_guard: mask has {flat_mask.numel()} positions, "
            f"routing has {out.expert_indices.shape[0]}")
    if not bool(flat_mask.any()):
        return out
    expert_indices = out.expert_indices.clone()
    expert_weights = out.expert_weights.clone()
    expert_indices[flat_mask] = out.num_experts     # residual sits right after the real experts
    expert_weights[flat_mask] = 1.0                 # single slot selected -> full weight on it
    return V3.RoutingContext(expert_indices=expert_indices, expert_weights=expert_weights,
                             valid_token_mask=out.valid_token_mask, num_experts=out.num_experts)


HEADER_ALL_FULL = False   # default for install_bos_guard(header_all_full=None)
FORCE_LAST_PROMPT_POS = False   # when True, the last position of a multi-token forward (the prompt's final
                                # token, e.g. the assistant-turn "\n\n" in stage B) is also a decision position


def install_bos_guard(model, bos_token_id=None, header_ids=None, header_all_full=None):
    """header_ids: token ids of the fixed chat header (see train_bos_token.guard_header_spec); None
    falls back to the module default HEADER_IDS (None = plain BOS guard only).  header_all_full:
    guard every header position at every layer (no last-layer-free decision token inside the
    header); None falls back to HEADER_ALL_FULL."""
    global _prev_forward
    if _prev_forward is None:
        _prev_forward = V3.SharedExpertRouter.forward
        V3.SharedExpertRouter.forward = _bos_guarded_forward
    _tag_layers(model)
    header_ids = HEADER_IDS if header_ids is None else header_ids
    model._bos_guard_header_ids = list(header_ids) if header_ids else None
    model._bos_guard_header_all_full = bool(HEADER_ALL_FULL if header_all_full is None else header_all_full)
    _install_model_hooks(model, BOS_TOKEN_ID if bos_token_id is None else bos_token_id,
                         model._bos_guard_header_ids)
