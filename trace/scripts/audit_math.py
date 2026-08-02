"""Small, dependency-light reference math used by correctness audits."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


def gem_project_exact(
    gradient: np.ndarray,
    memories: np.ndarray,
    margin: float = 0.0,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Project a gradient onto M g' >= margin for small audit problems.

    This enumerates active constraint sets and is intended for deterministic
    tests, not large training jobs. `memories` has one prior-task gradient per
    row.
    """

    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    m = np.asarray(memories, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != g.size:
        raise ValueError("memories must have shape [tasks, parameters]")
    if m.shape[0] == 0 or np.all(m @ g >= margin - tolerance):
        return g.copy()

    best = None
    best_distance = np.inf
    indices = range(m.shape[0])
    for active_size in range(1, m.shape[0] + 1):
        for active in itertools.combinations(indices, active_size):
            a = m[list(active)]
            rhs = np.full(active_size, margin) - a @ g
            gram = a @ a.T
            multipliers = np.linalg.pinv(gram) @ rhs
            if np.any(multipliers < -tolerance):
                continue
            candidate = g + a.T @ multipliers
            if np.all(m @ candidate >= margin - tolerance):
                distance = np.linalg.norm(candidate - g)
                if distance < best_distance:
                    best = candidate
                    best_distance = distance
    if best is None:
        raise RuntimeError("failed to find a feasible projection")
    return best


def gem_upstream_qpth_1d(
    gradient: float, memory: float, margin: float = 0.5
) -> float:
    """Closed-form result of TRACE's current one-constraint qpth translation.

    TRACE passes q=-M g and the constraint v<=margin to qpth, then returns
    g'=g+vM. This function makes that behavior testable without qpth/CUDA.
    """

    p = memory * memory
    q = -(memory * gradient)
    unconstrained_v = -q / p
    v = min(unconstrained_v, margin)
    return gradient + v * memory


def grassmann_similarity(update_basis: np.ndarray, base_basis: np.ndarray) -> float:
    """Squared Frobenius overlap used by Equation (3) of the SLoRA paper."""

    u = np.asarray(update_basis, dtype=np.float64)
    b = np.asarray(base_basis, dtype=np.float64)
    if u.ndim != 2 or b.ndim != 2 or u.shape != b.shape:
        raise ValueError("basis matrices must have the same [dimension, rank] shape")
    return float(np.linalg.norm(b.T @ u, ord="fro") ** 2)


def olora_paper_penalty(
    old_up_projections: Iterable[np.ndarray], new_up_projection: np.ndarray
) -> float:
    """Equation (8), mapped to PEFT's B matrices (out_features x rank)."""

    new_b = np.asarray(new_up_projection, dtype=np.float64)
    total = 0.0
    for old in old_up_projections:
        old_b = np.asarray(old, dtype=np.float64)
        overlap = old_b.T @ new_b
        total += float(np.sum(overlap**2))
    return total


def olora_upstream_penalty(
    old_down_projections: Iterable[np.ndarray], new_down_projection: np.ndarray
) -> float:
    """Current repository behavior: L1 overlap of PEFT A matrices."""

    new_a = np.asarray(new_down_projection, dtype=np.float64)
    total = 0.0
    for old in old_down_projections:
        old_a = np.asarray(old, dtype=np.float64)
        total += float(np.abs(old_a @ new_a.T).sum())
    return total
