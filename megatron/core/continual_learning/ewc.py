"""Canonical diagonal empirical-Fisher EWC utilities."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch


TensorMap = Mapping[str, torch.Tensor]


def ewc_penalty(
    parameters: Mapping[str, torch.nn.Parameter],
    fisher_sum: TensorMap,
    mean: TensorMap,
    coefficient: float,
) -> torch.Tensor:
    terms = []
    for name, parameter in parameters.items():
        if name not in fisher_sum or name not in mean:
            raise KeyError(f"EWC state is missing scoped parameter {name}")
        fisher = fisher_sum[name].to(device=parameter.device, dtype=torch.float32)
        anchor = mean[name].to(device=parameter.device, dtype=torch.float32)
        terms.append((fisher * (parameter.float() - anchor).square()).sum())
    if not terms:
        raise RuntimeError("EWC has no scoped parameters")
    return 0.5 * float(coefficient) * torch.stack(terms).sum()


def consolidate_equal_lambda(
    old_fisher: TensorMap,
    old_mean: TensorMap,
    new_fisher: TensorMap,
    new_mean: TensorMap,
    eps: float = 1.0e-20,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Exactly consolidate a sum of equal-lambda quadratic penalties."""
    keys = set(old_fisher) | set(new_fisher)
    fisher_sum, mean = {}, {}
    for name in keys:
        old_f = old_fisher.get(name)
        new_f = new_fisher.get(name)
        if old_f is None:
            fisher_sum[name] = new_f.float().cpu()
            mean[name] = new_mean[name].float().cpu()
            continue
        if new_f is None:
            fisher_sum[name] = old_f.float().cpu()
            mean[name] = old_mean[name].float().cpu()
            continue
        combined = old_f.float().cpu() + new_f.float().cpu()
        numerator = (
            old_f.float().cpu() * old_mean[name].float().cpu()
            + new_f.float().cpu() * new_mean[name].float().cpu()
        )
        fallback = new_mean[name].float().cpu()
        fisher_sum[name] = combined
        mean[name] = torch.where(combined > eps, numerator / combined.clamp_min(eps), fallback)
    return fisher_sum, mean
