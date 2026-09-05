"""Corrected SLoRA-Pre denoising and merge helpers."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterator, Tuple

import torch

from .lora_adapter import ContinualLowRankAdapter
from .parameter_scope import iter_layers


def _target_records(model, layer_start: int, layer_end: int):
    """Yield ``(key, base_weight, adapter, optional packed slice)`` records."""
    for layer_number, layer in iter_layers(model):
        if not layer_start <= layer_number <= layer_end:
            continue
        attention = layer.self_attention
        groups = int(attention.num_query_groups_per_partition)
        query_per_group = (
            attention.num_attention_heads_per_partition // attention.num_query_groups_per_partition
        ) * attention.hidden_size_per_attention_head
        kv_per_group = attention.hidden_size_per_attention_head
        qkv_weight = attention.linear_qkv.weight
        qkv_view = qkv_weight.view(groups, query_per_group + 2 * kv_per_group, -1)
        if getattr(attention, "continual_q_adapters", None):
            yield (
                f"layer{layer_number}.attention.q",
                qkv_view[:, :query_per_group, :].reshape(-1, qkv_weight.shape[1]),
                attention.continual_q_adapters[0],
                (qkv_view, slice(0, query_per_group)),
            )
            yield (
                f"layer{layer_number}.attention.k",
                qkv_view[:, query_per_group : query_per_group + kv_per_group, :].reshape(
                    -1, qkv_weight.shape[1]
                ),
                attention.continual_k_adapters[0],
                (qkv_view, slice(query_per_group, query_per_group + kv_per_group)),
            )
            yield (
                f"layer{layer_number}.attention.v",
                qkv_view[:, query_per_group + kv_per_group :, :].reshape(-1, qkv_weight.shape[1]),
                attention.continual_v_adapters[0],
                (qkv_view, slice(query_per_group + kv_per_group, query_per_group + 2 * kv_per_group)),
            )
            yield (
                f"layer{layer_number}.attention.o",
                attention.linear_proj.weight,
                attention.continual_o_adapters[0],
                None,
            )
        mlp = layer.mlp
        if getattr(mlp, "continual_gate_adapter", None) is not None:
            fc1 = mlp.linear_fc1.weight
            width = mlp.ffn_hidden_size
            yield (
                f"layer{layer_number}.mlp.gate",
                fc1[:width],
                mlp.continual_gate_adapter,
                (fc1, slice(0, width)),
            )
            yield (
                f"layer{layer_number}.mlp.up",
                fc1[width : 2 * width],
                mlp.continual_up_adapter,
                (fc1, slice(width, 2 * width)),
            )
            yield (
                f"layer{layer_number}.mlp.down",
                mlp.linear_fc2.weight,
                mlp.continual_down_adapter,
                None,
            )


def snapshot_reference(model, layer_start: int, layer_end: int) -> Dict[str, torch.Tensor]:
    return {
        key: weight.detach().float().cpu().clone()
        for key, weight, _adapter, _packed in _target_records(model, layer_start, layer_end)
    }


def reference_checksum(reference: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(reference):
        digest.update(key.encode("utf-8"))
        digest.update(reference[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _corrected_denoised_delta(
    delta_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    trained_rank: int,
    mode: str = "max",
) -> Tuple[torch.Tensor, int]:
    """SLoRA's 10%--100% candidate search without null-space padding."""
    if mode not in {"max", "min", "minor"}:
        raise ValueError(f"unsupported SLoRA denoising mode: {mode}")
    delta = delta_weight.float()
    reference = reference_weight.to(delta.device, torch.float32)
    u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    reference_u, _, _ = torch.linalg.svd(reference, full_matrices=False)
    best_score = None
    best = None
    seen = set()
    for ratio_index in range(1, 11):
        candidate = max(1, int(trained_rank * ratio_index / 10))
        candidate = min(candidate, singular.numel())
        if candidate in seen:
            continue
        seen.add(candidate)
        if mode == "minor":
            candidate_u = u[:, -candidate:]
            candidate_s = singular[-candidate:]
            candidate_vh = vh[-candidate:, :]
        else:
            candidate_u = u[:, :candidate]
            candidate_s = singular[:candidate]
            candidate_vh = vh[:candidate, :]
        score = torch.linalg.matrix_norm(
            candidate_u.T @ reference_u[:, :candidate], ord="fro"
        ).item()
        is_better = best_score is None or (score < best_score if mode == "min" else score > best_score)
        if is_better:
            best_score = score
            best = (
                (candidate_u * candidate_s.unsqueeze(0)) @ candidate_vh,
                candidate,
            )
    if best is None:
        raise RuntimeError("SLoRA candidate-rank search produced no candidate")
    return best


@torch.no_grad()
def initialize_fresh_adapter(model, rank: int, layer_start: int, layer_end: int, init_std: float) -> int:
    count = 0
    for _key, _weight, adapter, _packed in _target_records(model, layer_start, layer_end):
        adapter.reset_active(rank, init_std)
        count += 1
    if count == 0:
        raise RuntimeError("SLoRA found no q/k/v/o/gate/up/down targets in Layer 2--9")
    return count


@torch.no_grad()
def denoise_merge_and_clear(
    model,
    reference: Dict[str, torch.Tensor],
    rank: int,
    layer_start: int,
    layer_end: int,
    mode: str,
) -> dict:
    retained = {}
    for key, base_weight, adapter, packed in _target_records(model, layer_start, layer_end):
        if key not in reference:
            raise KeyError(f"SLoRA immutable reference is missing {key}")
        if adapter.active_rank != rank:
            raise RuntimeError(f"SLoRA active rank mismatch for {key}: {adapter.active_rank} != {rank}")
        delta, candidate_rank = _corrected_denoised_delta(
            adapter.local_delta_weight(rank=rank), reference[key], rank, mode
        )
        delta = delta.to(device=base_weight.device, dtype=base_weight.dtype)
        if packed is None:
            base_weight.add_(delta)
        else:
            container, target_slice = packed
            if container.ndim == 3:
                container[:, target_slice, :].add_(delta.view_as(container[:, target_slice, :]))
            else:
                container[target_slice].add_(delta)
        retained[key] = candidate_rank
        adapter.clear()
    return retained
