"""TRACE-compatible per-parameter GEM with the corrected QP signs."""

from __future__ import annotations

from typing import Iterable, List

import torch


def project_tensor_gradient(
    gradient: torch.Tensor,
    memories: Iterable[torch.Tensor],
    margin: float = 0.5,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Apply TRACE's per-parameter GEM dual with the corrected qpth signs.

    TRACE invokes the dual only when ``g·m < 0`` and constrains its dual
    coefficients as ``v >= margin``.  The qpth form is therefore
    ``q=M g, G=-I, h=-margin``; margin is *not* a lower bound on the projected
    primal dot products.  With Wiki/Code/Conversation there are at most two
    dual variables, so lower-bound active-set enumeration is exact and avoids
    an optional qpth dependency.
    """
    original_dtype = gradient.dtype
    original_shape = gradient.shape
    g = gradient.detach().float().reshape(-1)
    memory = [tensor.detach().to(g.device, torch.float32).reshape(-1) for tensor in memories]
    if not memory:
        return gradient
    matrix = torch.stack(memory)
    q = matrix @ g
    if torch.all(q >= 0):
        return gradient

    count = len(memory)
    gram = matrix @ matrix.T
    gram = 0.5 * (gram + gram.T) + torch.eye(
        count, device=g.device, dtype=g.dtype
    ) * float(eps)
    lower = g.new_full((count,), float(margin))
    best_alpha = None
    best_objective = None
    # A set bit denotes a coefficient fixed at its lower bound.  Every
    # possible active set is cheap to enumerate because count <= 2.
    for fixed_mask in range(1 << count):
        fixed = [index for index in range(count) if fixed_mask & (1 << index)]
        free = [index for index in range(count) if index not in fixed]
        alpha = lower.clone()
        if free:
            p_ff = gram[free][:, free]
            rhs = -q[free]
            if fixed:
                rhs = rhs - gram[free][:, fixed] @ lower[fixed]
            alpha[free] = torch.linalg.solve(p_ff, rhs)
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
        raise RuntimeError("TRACE-GEM per-parameter QP has no numerically feasible solution")
    projected = g + matrix.T @ best_alpha
    return projected.reshape(original_shape).to(original_dtype)
