"""Paper-faithful GEM: episodic data memory + one global gradient projection.

The existing ``trace_gem`` module reproduces the TRACE benchmark's GEM, which
deviates from Lopez-Paz & Ranzato (2017) in two ways: it stores one *terminal
gradient* per past task instead of past *examples*, and it solves the dual QP
independently for every parameter tensor.  This module implements the paper:

* the constraint gradients ``g_k`` are recomputed every step from an episodic
  memory of stored examples (the caller supplies them), and
* the inequality-constrained projection is solved once on the concatenated
  gradient vector, so a violated constraint is repaired using the full
  parameter space rather than tensor-by-tensor.

Paper form (Eq. 11): minimise ``0.5 * ||g_tilde - g||^2`` subject to
``<g_tilde, g_k> >= 0`` for every past task ``k``.  Its dual is

    min_v  0.5 v^T G G^T v + g^T G^T v      s.t.  v >= 0,

with ``G`` the stacked past-task gradients, after which ``g_tilde = g + G^T v``.
``margin`` shifts the lower bound on ``v`` exactly as the reference qpth code
does (``G = -I, h = -margin``), and ``eps`` regularises the Gram matrix.

With at most two past tasks (Wiki -> Code -> Conversation) the active set is
enumerable in closed form, so no QP solver dependency is required; the
enumeration is identical in structure to ``trace_gem.project_tensor_gradient``
but runs once on the global vector.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def flatten_gradients(gradients: Dict[str, torch.Tensor], order: Sequence[str]) -> torch.Tensor:
    """Concatenate per-tensor gradients into one float32 vector."""
    if not order:
        raise RuntimeError("GEM global projection needs at least one parameter tensor")
    return torch.cat([gradients[name].detach().float().reshape(-1) for name in order])


def unflatten_into(vector: torch.Tensor, gradients: Dict[str, torch.Tensor], order: Sequence[str]) -> None:
    """Write a concatenated vector back into the per-tensor gradient buffers."""
    offset = 0
    for name in order:
        target = gradients[name]
        count = target.numel()
        chunk = vector[offset : offset + count].reshape(target.shape)
        target.copy_(chunk.to(target.dtype))
        offset += count
    if offset != vector.numel():
        raise RuntimeError(
            f"GEM global projection size mismatch: wrote {offset} of {vector.numel()} elements"
        )


def project_global_gradient(
    gradient: torch.Tensor,
    memories: List[torch.Tensor],
    margin: float = 0.5,
    eps: float = 1.0e-3,
) -> torch.Tensor:
    """Project one global gradient vector onto the GEM feasible cone.

    ``gradient`` and every entry of ``memories`` are 1-D float32 vectors laid
    out with the same parameter order.  Returns the projected vector; when no
    constraint is violated the input is returned unchanged.
    """
    if not memories:
        return gradient
    matrix = torch.stack(memories)              # [k, d]
    q = matrix @ gradient                       # <g_k, g>
    if torch.all(q >= 0):
        return gradient

    count = len(memories)
    gram = matrix @ matrix.T
    gram = 0.5 * (gram + gram.T) + torch.eye(count, device=gradient.device, dtype=gradient.dtype) * float(eps)
    lower = gradient.new_full((count,), float(margin))

    best_alpha = None
    best_objective = None
    # A set bit fixes that coefficient at its lower bound; k <= 2 here, so the
    # full active-set enumeration is four solves at most.
    for fixed_mask in range(1 << count):
        fixed = [i for i in range(count) if fixed_mask & (1 << i)]
        free = [i for i in range(count) if i not in fixed]
        alpha = lower.clone()
        if free:
            p_ff = gram[free][:, free]
            rhs = -q[free]
            if fixed:
                rhs = rhs - gram[free][:, fixed] @ lower[fixed]
            try:
                alpha[free] = torch.linalg.solve(p_ff, rhs)
            except Exception:
                continue
            if torch.any(alpha[free] < lower[free] - 1.0e-5):
                continue
        stationarity = gram @ alpha + q
        if free and torch.any(stationarity[free].abs() > 2.0e-3):
            continue
        if fixed and torch.any(stationarity[fixed] < -2.0e-3):
            continue
        objective = 0.5 * alpha @ gram @ alpha + q @ alpha
        if best_alpha is None or objective < best_objective:
            best_alpha = alpha
            best_objective = objective
    if best_alpha is None:
        raise RuntimeError("GEM global QP has no numerically feasible solution")
    return gradient + matrix.T @ best_alpha


def violation_stats(gradient: torch.Tensor, memories: List[torch.Tensor]) -> dict:
    """Audit helper: cosine and dot product against every past-task gradient."""
    stats = []
    g_norm = float(torch.linalg.vector_norm(gradient).cpu().item())
    for index, memory in enumerate(memories):
        dot = float((memory @ gradient).cpu().item())
        m_norm = float(torch.linalg.vector_norm(memory).cpu().item())
        cosine = dot / (g_norm * m_norm) if g_norm > 0 and m_norm > 0 else 0.0
        stats.append({"task": index, "dot": dot, "cosine": cosine, "memory_norm": m_norm})
    return {"gradient_norm": g_norm, "constraints": stats}
