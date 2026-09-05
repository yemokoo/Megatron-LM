"""Numerical core for the CKA old-like-token GT pilot.

This module deliberately contains no model, dataset, storage, threshold-fitting,
or training code.  It implements the fixed pilot equations on aligned before /
after hidden states and the two layer-consensus interpretations that the report
must compare.

All metric arithmetic is performed in float32, even when the supplied hidden
states come from a lower-precision forward.  The token axis is the penultimate
axis throughout, so a single chunk has shape ``[tokens, hidden]`` and a batch of
equal-length chunks has shape ``[..., tokens, hidden]``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch


DEFAULT_NUMERIC_FLOOR = 1.0e-12
DEFAULT_OFFDIAG_WARNING_FRACTION = 0.01

# ``invalid_reason`` is a bit mask.  Keeping X/Y and non-finite causes separate
# makes aggregate reports useful without turning an invalid measurement into a
# numeric score.
INVALID_NONE = 0
INVALID_X_RSM_NORM = 1 << 0
INVALID_Y_RSM_NORM = 1 << 1
INVALID_NONFINITE_INPUT = 1 << 2

# ``t_invalid_reason`` is separate because a standard diagonal-including CKA
# can be valid while the off-diagonal share denominator is not.
T_INVALID_NONE = 0
T_INVALID_CHUNK = 1 << 0
T_INVALID_CKA_OFF = 1 << 1

INVALID_REASON_LABELS = {
    INVALID_NONE: "valid",
    INVALID_X_RSM_NORM: "x_rsm_norm_below_floor_or_nonfinite",
    INVALID_Y_RSM_NORM: "y_rsm_norm_below_floor_or_nonfinite",
    INVALID_X_RSM_NORM | INVALID_Y_RSM_NORM: "both_rsm_norms_below_floor_or_nonfinite",
}


def _validate_aligned_hidden(before: torch.Tensor, after: torch.Tensor) -> None:
    if before.shape != after.shape:
        raise ValueError(
            f"before/after shapes must match, got {tuple(before.shape)} and "
            f"{tuple(after.shape)}"
        )
    if before.ndim < 2:
        raise ValueError(
            f"hidden tensors must have shape [..., tokens, hidden], got {tuple(before.shape)}"
        )
    if before.shape[-2] < 2:
        raise ValueError("CKA requires at least two tokens per chunk")
    if before.shape[-1] < 1:
        raise ValueError("hidden dimension must be positive")


def _nan_like(value: torch.Tensor) -> torch.Tensor:
    return torch.full_like(value, float("nan"), dtype=torch.float32)


def centered_rsms(
    before: torch.Tensor,
    after: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fp32 chunk-centered hidden states and their linear RSMs.

    This low-level helper exists for focused diagnostics.  The pilot runner
    should normally call :func:`centered_linear_cka_metrics`, whose default
    return avoids retaining these O(n^2) matrices after scalar extraction.
    """
    _validate_aligned_hidden(before, after)
    x = before.float()
    y = after.float()
    x_centered = x - x.mean(dim=-2, keepdim=True)
    y_centered = y - y.mean(dim=-2, keepdim=True)
    k = torch.matmul(x_centered, x_centered.transpose(-1, -2))
    l = torch.matmul(y_centered, y_centered.transpose(-1, -2))
    return x_centered, y_centered, k, l


def centered_linear_cka_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    numeric_floor: float = DEFAULT_NUMERIC_FLOOR,
    offdiag_warning_fraction: float = DEFAULT_OFFDIAG_WARNING_FRACTION,
    return_rsms: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the fixed centered-linear-CKA and token contribution metrics.

    Args:
        before: Aligned reference hidden states, ``[..., n, p]``.
        after: Aligned current hidden states, same shape as ``before``.
        numeric_floor: The pilot's absolute fp32 validity floor.
        offdiag_warning_fraction: Diagnostic-only warning threshold.  A valid
            chunk is warned when ``CKA_off < fraction * max(CKA, floor)``.  The
            warning never changes CKA validity or selector membership.

    Returns:
        A dictionary whose chunk-level tensors have shape ``[...]`` and whose
        token-level tensors have shape ``[..., n]``.  If ``return_rsms=True``,
        the ephemeral fp32 ``k`` and ``l`` RSMs are also returned; they must not
        be written as raw pilot output.

    Notes:
        ``sum(c_i) == CKA`` up to fp32 reduction error.  ``s_i`` is NaN when
        ``CKA_off <= numeric_floor`` as required by the pilot.  Negative valid
        shares remain numeric and are also exposed through ``neg_contrib``;
        the downstream T consensus must reject them rather than clipping them.
    """
    _validate_aligned_hidden(before, after)
    if numeric_floor <= 0.0 or not math.isfinite(numeric_floor):
        raise ValueError("numeric_floor must be finite and positive")
    if offdiag_warning_fraction < 0.0 or not math.isfinite(offdiag_warning_fraction):
        raise ValueError("offdiag_warning_fraction must be finite and non-negative")

    # Explicit casts prevent autocast or the forward dtype from leaking into
    # any RSM, norm, or reduction.
    x = before.float()
    y = after.float()
    x_centered, y_centered, k, l = centered_rsms(x, y)
    k_norm = torch.linalg.vector_norm(k, dim=(-2, -1))
    l_norm = torch.linalg.vector_norm(l, dim=(-2, -1))

    x_nonfinite = ~torch.isfinite(x).all(dim=(-2, -1))
    y_nonfinite = ~torch.isfinite(y).all(dim=(-2, -1))
    x_bad = (~torch.isfinite(k_norm)) | (k_norm < numeric_floor)
    y_bad = (~torch.isfinite(l_norm)) | (l_norm < numeric_floor)

    invalid_reason = torch.zeros_like(k_norm, dtype=torch.uint8)
    invalid_reason = invalid_reason | x_bad.to(torch.uint8) * INVALID_X_RSM_NORM
    invalid_reason = invalid_reason | y_bad.to(torch.uint8) * INVALID_Y_RSM_NORM
    invalid_reason = invalid_reason | (x_nonfinite | y_nonfinite).to(torch.uint8) * INVALID_NONFINITE_INPUT
    chunk_valid = invalid_reason == INVALID_NONE

    denominator = k_norm * l_norm
    safe_denominator = torch.where(chunk_valid, denominator, torch.ones_like(denominator))
    row_cross = (k * l).sum(dim=-1)
    numerator = row_cross.sum(dim=-1)
    c_i_raw = row_cross / safe_denominator.unsqueeze(-1)
    cka_raw = numerator / safe_denominator

    k_diag = torch.diagonal(k, dim1=-2, dim2=-1)
    l_diag = torch.diagonal(l, dim1=-2, dim2=-1)
    diag_cross_by_token = k_diag * l_diag
    c_i_off_raw = (row_cross - diag_cross_by_token) / safe_denominator.unsqueeze(-1)
    cka_off_raw = c_i_off_raw.sum(dim=-1)

    cka = torch.where(chunk_valid, cka_raw, _nan_like(cka_raw))
    c_i = torch.where(chunk_valid.unsqueeze(-1), c_i_raw, _nan_like(c_i_raw))
    c_i_off = torch.where(
        chunk_valid.unsqueeze(-1), c_i_off_raw, _nan_like(c_i_off_raw)
    )
    cka_off = torch.where(chunk_valid, cka_off_raw, _nan_like(cka_off_raw))

    t_valid = chunk_valid & torch.isfinite(cka_off_raw) & (cka_off_raw > numeric_floor)
    t_invalid_reason = torch.zeros_like(invalid_reason)
    t_invalid_reason = t_invalid_reason | (~chunk_valid).to(torch.uint8) * T_INVALID_CHUNK
    t_invalid_reason = t_invalid_reason | (
        chunk_valid & ((~torch.isfinite(cka_off_raw)) | (cka_off_raw <= numeric_floor))
    ).to(torch.uint8) * T_INVALID_CKA_OFF

    n_tokens = int(x.shape[-2])
    safe_off_mean = torch.where(
        t_valid,
        cka_off_raw / float(n_tokens),
        torch.ones_like(cka_off_raw),
    )
    s_raw = c_i_off_raw / safe_off_mean.unsqueeze(-1)
    s_i = torch.where(t_valid.unsqueeze(-1), s_raw, _nan_like(s_raw))
    neg_contrib = t_valid.unsqueeze(-1) & (s_raw < 0.0)

    # Diagonal ratio is undefined only when its own denominator is numerically
    # zero; this does not retroactively invalidate the standard CKA metric.
    diag_numerator = diag_cross_by_token.sum(dim=-1)
    diag_valid = chunk_valid & torch.isfinite(numerator) & (numerator.abs() >= numeric_floor)
    diag_ratio_raw = diag_numerator / torch.where(
        diag_valid, numerator, torch.ones_like(numerator)
    )
    diag_ratio = torch.where(diag_valid, diag_ratio_raw, _nan_like(diag_ratio_raw))

    # Off-diagonal Pearson correlation per RSM row, without materializing an
    # O(n^3) diagonal-exclusion mask.  The centered row moments below are the
    # exact Pearson moments over j != i.
    off_count = n_tokens - 1
    k_off_sum = k.sum(dim=-1) - k_diag
    l_off_sum = l.sum(dim=-1) - l_diag
    kl_off_sum = row_cross - diag_cross_by_token
    k2_off_sum = k.square().sum(dim=-1) - k_diag.square()
    l2_off_sum = l.square().sum(dim=-1) - l_diag.square()
    pearson_numerator = kl_off_sum - k_off_sum * l_off_sum / float(off_count)
    k_centered_ss = (k2_off_sum - k_off_sum.square() / float(off_count)).clamp_min(0.0)
    l_centered_ss = (l2_off_sum - l_off_sum.square() / float(off_count)).clamp_min(0.0)
    pearson_denominator = torch.sqrt(k_centered_ss * l_centered_ss)
    pearson_valid = (
        chunk_valid.unsqueeze(-1)
        & torch.isfinite(pearson_denominator)
        & (pearson_denominator >= numeric_floor)
    )
    pearson_raw = pearson_numerator / torch.where(
        pearson_valid, pearson_denominator, torch.ones_like(pearson_denominator)
    )
    r_i = torch.where(
        pearson_valid, pearson_raw.clamp(-1.0, 1.0), _nan_like(pearson_raw)
    )

    cka_warning_reference = torch.maximum(
        torch.nan_to_num(cka, nan=0.0), torch.full_like(cka_raw, numeric_floor)
    )
    offdiag_warning = chunk_valid & (
        cka_off_raw < offdiag_warning_fraction * cka_warning_reference
    )

    element_count = float(n_tokens * x.shape[-1])
    centered_var_x_raw = x_centered.square().sum(dim=(-2, -1)) / element_count
    centered_var_y_raw = y_centered.square().sum(dim=(-2, -1)) / element_count
    centered_var_x = torch.where(
        torch.isfinite(centered_var_x_raw), centered_var_x_raw, _nan_like(centered_var_x_raw)
    )
    centered_var_y = torch.where(
        torch.isfinite(centered_var_y_raw), centered_var_y_raw, _nan_like(centered_var_y_raw)
    )

    result = {
        "cka": cka.float(),
        "c_i": c_i.float(),
        "c_i_off": c_i_off.float(),
        "cka_off": cka_off.float(),
        "s_i": s_i.float(),
        "diag_ratio": diag_ratio.float(),
        "r_i": r_i.float(),
        "centered_var_x": centered_var_x.float(),
        "centered_var_y": centered_var_y.float(),
        "k_norm": k_norm.float(),
        "l_norm": l_norm.float(),
        "invalid_reason": invalid_reason,
        "t_invalid_reason": t_invalid_reason,
        "chunk_valid": chunk_valid,
        "t_valid": t_valid,
        "offdiag_warning": offdiag_warning,
        "neg_contrib": neg_contrib,
    }
    if return_rsms:
        # Ephemeral values useful to exact validation and custom diagnostics.
        result["k"] = k.float()
        result["l"] = l.float()
    return result


def centered_linear_cka_b_t_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    numeric_floor: float = DEFAULT_NUMERIC_FLOOR,
) -> dict[str, torch.Tensor]:
    """Compute only the production census' CKA (B) and ``s_i`` (T).

    This is algebraically identical to the corresponding outputs of
    :func:`centered_linear_cka_metrics`, but intentionally skips RSM-row
    Pearson correlation, diagonal diagnostics, variances, and warning flags.
    A full-corpus census invokes this kernel tens of billions of token-layer
    times, so avoiding diagnostics that are neither selected nor stored is
    material to H100 utilization.
    """

    _validate_aligned_hidden(before, after)
    if numeric_floor <= 0.0 or not math.isfinite(numeric_floor):
        raise ValueError("numeric_floor must be finite and positive")
    x = before.float()
    y = after.float()
    x_centered = x - x.mean(dim=-2, keepdim=True)
    y_centered = y - y.mean(dim=-2, keepdim=True)
    k = torch.matmul(x_centered, x_centered.transpose(-1, -2))
    l = torch.matmul(y_centered, y_centered.transpose(-1, -2))
    k_norm = torch.linalg.vector_norm(k, dim=(-2, -1))
    l_norm = torch.linalg.vector_norm(l, dim=(-2, -1))
    input_finite = torch.isfinite(x).all(dim=(-2, -1)) & torch.isfinite(y).all(
        dim=(-2, -1)
    )
    chunk_valid = (
        input_finite
        & torch.isfinite(k_norm)
        & torch.isfinite(l_norm)
        & (k_norm >= numeric_floor)
        & (l_norm >= numeric_floor)
    )
    denominator = k_norm * l_norm
    safe_denominator = torch.where(chunk_valid, denominator, torch.ones_like(denominator))
    row_cross = (k * l).sum(dim=-1)
    cka_raw = row_cross.sum(dim=-1) / safe_denominator
    cka = torch.where(chunk_valid, cka_raw, _nan_like(cka_raw))

    diagonal_cross = torch.diagonal(k, dim1=-2, dim2=-1) * torch.diagonal(
        l, dim1=-2, dim2=-1
    )
    c_i_off_raw = (row_cross - diagonal_cross) / safe_denominator.unsqueeze(-1)
    cka_off_raw = c_i_off_raw.sum(dim=-1)
    t_valid = chunk_valid & torch.isfinite(cka_off_raw) & (cka_off_raw > numeric_floor)
    safe_off_mean = torch.where(
        t_valid,
        cka_off_raw / float(x.shape[-2]),
        torch.ones_like(cka_off_raw),
    )
    s_raw = c_i_off_raw / safe_off_mean.unsqueeze(-1)
    s_i = torch.where(t_valid.unsqueeze(-1), s_raw, _nan_like(s_raw))
    return {
        "cka": cka.float(),
        "s_i": s_i.float(),
        "chunk_valid": chunk_valid,
        "t_valid": t_valid,
    }


def uncentered_token_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    numeric_floor: float = DEFAULT_NUMERIC_FLOOR,
) -> dict[str, torch.Tensor]:
    """Compute the pilot's uncentered token x layer metrics in float32."""
    if before.shape != after.shape or before.ndim < 1:
        raise ValueError(
            f"expected aligned [..., hidden] tensors, got {tuple(before.shape)} "
            f"and {tuple(after.shape)}"
        )
    if before.shape[-1] < 1:
        raise ValueError("hidden dimension must be positive")
    if numeric_floor <= 0.0 or not math.isfinite(numeric_floor):
        raise ValueError("numeric_floor must be finite and positive")

    x = before.float()
    y = after.float()
    delta = y - x
    x_norm = torch.linalg.vector_norm(x, dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    cosine_denominator = (x_norm * y_norm).clamp_min(numeric_floor)
    cosine = (x * y).sum(dim=-1) / cosine_denominator
    relative_l2 = delta_norm / x_norm.clamp_min(numeric_floor)
    symmetric_relative_l2 = 2.0 * delta_norm / (x_norm + y_norm).clamp_min(numeric_floor)
    log_r = torch.log(y_norm.clamp_min(numeric_floor) / x_norm.clamp_min(numeric_floor))
    ref_rms = x_norm / math.sqrt(float(x.shape[-1]))

    return {
        "cosine": cosine.clamp(-1.0, 1.0).float(),
        "rel_l2": relative_l2.float(),
        "sym_rel_l2": symmetric_relative_l2.float(),
        "log_r": log_r.float(),
        "ref_rms": ref_rms.float(),
    }


def _deterministic_randperm(length: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(length, generator=generator, device="cpu").to(device=device)


def permutation_null_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    seed: int = 1234,
    permutation: torch.Tensor | None = None,
    numeric_floor: float = DEFAULT_NUMERIC_FLOOR,
) -> dict[str, Any]:
    """Compute a within-chunk After-token permutation null.

    This helper intentionally accepts exactly one chunk.  A runner should use a
    deterministic seed derived from window/chunk identity for independent null
    draws rather than silently reusing one permutation for a batch.
    """
    _validate_aligned_hidden(before, after)
    if before.ndim != 2:
        raise ValueError("permutation_null_metrics expects one [tokens, hidden] chunk")
    n_tokens = int(before.shape[-2])
    if permutation is None:
        permutation = _deterministic_randperm(n_tokens, seed, after.device)
    else:
        permutation = torch.as_tensor(permutation, dtype=torch.long, device=after.device)
        if permutation.shape != (n_tokens,):
            raise ValueError(f"permutation must have shape {(n_tokens,)}, got {tuple(permutation.shape)}")
        if not torch.equal(
            torch.sort(permutation.detach().cpu()).values, torch.arange(n_tokens)
        ):
            raise ValueError("permutation must contain every token index exactly once")
    shuffled_after = after.index_select(-2, permutation)
    result: dict[str, Any] = centered_linear_cka_metrics(
        before, shuffled_after, numeric_floor=numeric_floor
    )
    result["permutation"] = permutation
    return result


def random_pair_indices(num_windows: int, *, seed: int = 1234) -> torch.Tensor:
    """Return a deterministic no-fixed-point cyclic pairing of window IDs."""
    if num_windows < 2:
        raise ValueError("random-pair null requires at least two windows")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    # A random non-zero cyclic offset is a permutation and guarantees that no
    # Before window is paired with its aligned After window.
    offset = int(torch.randint(1, num_windows, (1,), generator=generator).item())
    return (torch.arange(num_windows, dtype=torch.long) + offset) % num_windows


def random_pair_null_metrics(
    before_windows: torch.Tensor,
    after_windows: torch.Tensor,
    *,
    seed: int = 1234,
    pair_indices: torch.Tensor | None = None,
    numeric_floor: float = DEFAULT_NUMERIC_FLOOR,
) -> dict[str, Any]:
    """Pair each Before window with a different After window and compute CKA.

    Equal chunk lengths are required for vectorization.  Variable tail lengths
    should be grouped by length by the caller before using this helper.
    """
    _validate_aligned_hidden(before_windows, after_windows)
    if before_windows.ndim != 3:
        raise ValueError(
            "random_pair_null_metrics expects [windows, tokens, hidden] tensors"
        )
    count = int(before_windows.shape[0])
    if pair_indices is None:
        pair_indices = random_pair_indices(count, seed=seed)
    pair_indices = torch.as_tensor(
        pair_indices, dtype=torch.long, device=after_windows.device
    )
    if pair_indices.shape != (count,):
        raise ValueError(f"pair_indices must have shape {(count,)}, got {tuple(pair_indices.shape)}")
    pair_cpu = pair_indices.detach().cpu()
    if not torch.equal(torch.sort(pair_cpu).values, torch.arange(count)):
        raise ValueError("pair_indices must be a permutation of window indices")
    if torch.any(pair_cpu == torch.arange(count)):
        raise ValueError("random-pair null must not contain fixed points")
    paired_after = after_windows.index_select(0, pair_indices)
    result: dict[str, Any] = centered_linear_cka_metrics(
        before_windows, paired_after, numeric_floor=numeric_floor
    )
    result["pair_indices"] = pair_indices
    return result


def _compare_condition(
    values: torch.Tensor,
    thresholds: torch.Tensor | float,
    comparison: str,
) -> torch.Tensor:
    threshold = torch.as_tensor(thresholds, dtype=values.dtype, device=values.device)
    if comparison == "ge":
        return values >= threshold
    if comparison == "le":
        return values <= threshold
    raise ValueError(f"comparison must be 'ge' or 'le', got {comparison!r}")


def condition_layer_consensus(
    values: torch.Tensor,
    thresholds: torch.Tensor | float,
    *,
    comparison: str,
    min_valid_layers: int = 6,
    required_pass_layers: int = 7,
    reject_negative: bool = False,
) -> dict[str, torch.Tensor]:
    """Apply one condition's independent layer consensus.

    The final axis is the layer axis.  With the pilot defaults, six measured
    layers require 6/6 passes, seven require 7/7, and eight require at least
    7/8.  Missing (NaN/Inf) values are measurement absence, never zero scores.
    """
    if values.ndim < 1:
        raise ValueError("values must include a final layer axis")
    layer_count = int(values.shape[-1])
    if not 1 <= min_valid_layers <= layer_count:
        raise ValueError("min_valid_layers must be within the layer axis")
    if required_pass_layers < 1:
        raise ValueError("required_pass_layers must be positive")

    values = values.float()
    valid_by_layer = torch.isfinite(values)
    pass_by_layer = valid_by_layer & _compare_condition(values, thresholds, comparison)
    if reject_negative:
        pass_by_layer = pass_by_layer & (values >= 0.0)
    n_valid = valid_by_layer.sum(dim=-1)
    pass_count = pass_by_layer.sum(dim=-1)
    required = torch.minimum(
        n_valid, torch.full_like(n_valid, int(required_pass_layers))
    )
    eligible = n_valid >= int(min_valid_layers)
    passed = eligible & (pass_count >= required)
    return {
        "eligible": eligible,
        "passed": passed,
        "n_valid": n_valid,
        "pass_count": pass_count,
        "required_pass_count": required,
        "valid_by_layer": valid_by_layer,
        "pass_by_layer": pass_by_layer,
    }


def condition_specific_consensus(
    conditions: Mapping[str, torch.Tensor],
    thresholds: Mapping[str, torch.Tensor | float],
    comparisons: Mapping[str, str],
    *,
    reject_negative: Iterable[str] = (),
    min_valid_layers: int = 6,
    required_pass_layers: int = 7,
) -> dict[str, Any]:
    """Evaluate every B/T/M condition independently, then AND their results."""
    if not conditions:
        raise ValueError("at least one condition is required")
    names = tuple(conditions)
    if set(names) != set(thresholds) or set(names) != set(comparisons):
        raise ValueError("conditions, thresholds, and comparisons must have identical keys")
    shapes = {tuple(value.shape) for value in conditions.values()}
    if len(shapes) != 1:
        raise ValueError(f"all condition tensors must have the same shape, got {shapes}")
    reject = set(reject_negative)
    unknown_reject = reject - set(names)
    if unknown_reject:
        raise ValueError(f"reject_negative contains unknown conditions: {sorted(unknown_reject)}")

    by_condition: dict[str, dict[str, torch.Tensor]] = {}
    for name in names:
        by_condition[name] = condition_layer_consensus(
            conditions[name],
            thresholds[name],
            comparison=comparisons[name],
            min_valid_layers=min_valid_layers,
            required_pass_layers=required_pass_layers,
            reject_negative=name in reject,
        )
    eligible = torch.stack(
        [by_condition[name]["eligible"] for name in names], dim=0
    ).all(dim=0)
    passed = torch.stack(
        [by_condition[name]["passed"] for name in names], dim=0
    ).all(dim=0)
    return {"eligible": eligible, "passed": passed, "by_condition": by_condition}


def same_layer_consensus(
    conditions: Mapping[str, torch.Tensor],
    thresholds: Mapping[str, torch.Tensor | float],
    comparisons: Mapping[str, str],
    *,
    reject_negative: Iterable[str] = (),
    min_valid_layers: int = 6,
    required_pass_layers: int = 7,
) -> dict[str, torch.Tensor]:
    """Require every B/T/M condition on the same layer before layer consensus."""
    if not conditions:
        raise ValueError("at least one condition is required")
    names = tuple(conditions)
    if set(names) != set(thresholds) or set(names) != set(comparisons):
        raise ValueError("conditions, thresholds, and comparisons must have identical keys")
    shapes = {tuple(value.shape) for value in conditions.values()}
    if len(shapes) != 1:
        raise ValueError(f"all condition tensors must have the same shape, got {shapes}")
    reject = set(reject_negative)
    if reject - set(names):
        raise ValueError("reject_negative contains unknown condition names")

    valid_parts = []
    pass_parts = []
    for name in names:
        values = conditions[name].float()
        valid = torch.isfinite(values)
        passed = valid & _compare_condition(values, thresholds[name], comparisons[name])
        if name in reject:
            passed = passed & (values >= 0.0)
        valid_parts.append(valid)
        pass_parts.append(passed)
    valid_by_layer = torch.stack(valid_parts, dim=0).all(dim=0)
    pass_by_layer = torch.stack(pass_parts, dim=0).all(dim=0) & valid_by_layer
    n_valid = valid_by_layer.sum(dim=-1)
    pass_count = pass_by_layer.sum(dim=-1)
    required = torch.minimum(
        n_valid, torch.full_like(n_valid, int(required_pass_layers))
    )
    eligible = n_valid >= int(min_valid_layers)
    passed = eligible & (pass_count >= required)
    return {
        "eligible": eligible,
        "passed": passed,
        "n_valid": n_valid,
        "pass_count": pass_count,
        "required_pass_count": required,
        "valid_by_layer": valid_by_layer,
        "pass_by_layer": pass_by_layer,
    }


def consensus_strategy_diagnostic(
    condition_specific_pass: torch.Tensor,
    same_layer_pass: torch.Tensor,
) -> dict[str, float | int]:
    """Return set sizes and Jaccard for the mandated one-line diagnostic."""
    if condition_specific_pass.shape != same_layer_pass.shape:
        raise ValueError("the two selector masks must have identical shapes")
    a = condition_specific_pass.bool().reshape(-1)
    b = same_layer_pass.bool().reshape(-1)
    intersection = int((a & b).sum().item())
    union = int((a | b).sum().item())
    return {
        "condition_specific_count": int(a.sum().item()),
        "same_layer_count": int(b.sum().item()),
        "intersection_count": intersection,
        "union_count": union,
        "jaccard": float(intersection / union) if union else 1.0,
    }


__all__ = [
    "DEFAULT_NUMERIC_FLOOR",
    "DEFAULT_OFFDIAG_WARNING_FRACTION",
    "INVALID_NONE",
    "INVALID_X_RSM_NORM",
    "INVALID_Y_RSM_NORM",
    "INVALID_NONFINITE_INPUT",
    "T_INVALID_NONE",
    "T_INVALID_CHUNK",
    "T_INVALID_CKA_OFF",
    "INVALID_REASON_LABELS",
    "centered_rsms",
    "centered_linear_cka_metrics",
    "centered_linear_cka_b_t_metrics",
    "uncentered_token_metrics",
    "permutation_null_metrics",
    "random_pair_indices",
    "random_pair_null_metrics",
    "condition_layer_consensus",
    "condition_specific_consensus",
    "same_layer_consensus",
    "consensus_strategy_diagnostic",
]
