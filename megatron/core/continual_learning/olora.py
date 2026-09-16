"""Paper-correct O-LoRA squared-Frobenius orthogonality.

Wang et al. (2023) Eq. (2) fixes the convention: ``h = W_init x + A B x`` with
``W_init`` in ``R^{d x k}``, so ``k`` is the input dimension and ``d`` the
output one.  ``B`` in ``R^{r x k}`` compresses and ``A`` in ``R^{d x r}``
expands.  The task subspace is the span of the *columns of A* (Eq. 3--4), the
overlap is ``O_{i,t} = A_i^T A_t`` (Eq. 6), and the penalty sums its squared
entries (Eq. 8) -- so the regularized factor is the output-side expansion
matrix, which is PEFT's ``lora_B`` and this module's ``output_factor()``.

The public implementation instead applies an L1 overlap to PEFT's ``lora_A``,
i.e. to the paper's compression factor ``B``.  This module deliberately
follows the paper on both counts.
"""

from __future__ import annotations

import torch
import torch.distributed.nn.functional as dist_nn

from megatron.core import parallel_state

from .parameter_scope import iter_layers



def _olora_slot_index(task: str) -> int:
    """현재 태스크가 쓰는 LoRA 슬롯.

    기본(기존 동작): wiki 가 슬롯0 을 쓰므로 code=1, conv=2.
    OLORA_WIKI_BACKBONE_ONLY=1 이면 wiki 는 백본만 학습하고 슬롯을 안 쓰므로
    code=0, conv=1 로 한 칸씩 당긴다 (논문의 "고정 백본 + 태스크별 LoRA" 전제).
    """
    import os
    shift = 0 if os.environ.get("OLORA_WIKI_BACKBONE_ONLY", "0") == "1" else 1
    return (0 if task == "code" else 1) + shift

def _global_gram(previous, current):
    """The paper's overlap ``O_{i,t} = A_i^T A_t``.

    ``A`` spans the output dimension, which column-parallel Q/V shard across
    tensor-parallel ranks, so the Gram must be summed over them.
    """
    gram = previous.output_factor().detach().float().T @ current.output_factor().float()
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        gram = dist_nn.all_reduce(gram, group=parallel_state.get_tensor_model_parallel_group())
    return gram


def orthogonality_penalty(model, task: str, coefficient: float, layer_start: int, layer_end: int):
    if task == "wiki":
        reference = next(iter(model)).parameters() if isinstance(model, (list, tuple)) else model.parameters()
        parameter = next(reference)
        return parameter.new_zeros((), dtype=torch.float32)
    current_index = _olora_slot_index(task)
    previous_indices = range(current_index)
    losses = []
    for layer_number, layer in iter_layers(model):
        if not layer_start <= layer_number <= layer_end:
            continue
        for module in layer.modules():
            for collection_name in ("continual_q_adapters", "continual_v_adapters"):
                adapters = getattr(module, collection_name, None)
                if adapters is None or len(adapters) != 3:
                    continue
                current = adapters[current_index]
                for previous_index in previous_indices:
                    gram = _global_gram(adapters[previous_index], current)
                    losses.append(gram.square().sum())
    if not losses:
        if current_index == 0:      # 첫 LoRA 태스크 — 직교할 이전 슬롯이 없다
            reference = next(iter(model)).parameters() if isinstance(model, (list, tuple)) else model.parameters()
            return next(reference).new_zeros((), dtype=torch.float32)
        raise RuntimeError("O-LoRA found no Q/V adapter pairs in Layer 2--9")
    return float(coefficient) * torch.stack(losses).sum()


def adapter_l2_penalty(model, task: str, coefficient: float, layer_start: int, layer_end: int):
    """O-LoRA's lambda_2 term: L2 shrinkage on the *current* task's adapter.

    The public implementation (``O-LoRA/src/uie_trainer_lora.py``) computes

        l2_loss = sum over parameters named ``loranew_*`` of ||param||_2
        loss    = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2

    i.e. the plain Frobenius norm (not its square) of the newly added task's
    A and B factors only -- previously learned slots are left alone.  This
    function mirrors that definition on our slot layout: ``loranew_`` is the
    slot indexed by the current task, and Q/V are the only adapted modules.
    """
    if coefficient == 0.0 or task == "wiki":
        return torch.zeros((), dtype=torch.float32)
    current_index = _olora_slot_index(task)
    terms = []
    for layer_number, layer in iter_layers(model):
        if not layer_start <= layer_number <= layer_end:
            continue
        for parent in layer.modules():
            for collection_name in ("continual_q_adapters", "continual_v_adapters"):
                adapters = getattr(parent, collection_name, None)
                if adapters is None or len(adapters) != 3:
                    continue
                current = adapters[current_index]
                rank = max(int(current.active_rank), 0)
                if rank == 0:
                    continue
                terms.append(torch.linalg.vector_norm(current.lora_a[:, :rank].float()))
                terms.append(torch.linalg.vector_norm(current.lora_b[:rank, :].float()))
    if not terms:
        raise RuntimeError("O-LoRA lambda_2 found no active Q/V adapter in Layer 2--9")
    return float(coefficient) * torch.stack(terms).sum()


@torch.no_grad()
def adapter_slot_norms(model, layer_start: int, layer_end: int):
    """Return aggregate Q/V A/B norms for the three persistent task slots."""
    a_squares = [None, None, None]
    b_squares = [None, None, None]
    for layer_number, layer in iter_layers(model):
        if not layer_start <= layer_number <= layer_end:
            continue
        for module in layer.modules():
            for collection_name in ("continual_q_adapters", "continual_v_adapters"):
                adapters = getattr(module, collection_name, None)
                if adapters is None or len(adapters) != 3:
                    continue
                for index, adapter in enumerate(adapters):
                    a_term = adapter.lora_a.detach().float().square().sum()
                    b_term = adapter.lora_b.detach().float().square().sum()
                    a_squares[index] = a_term if a_squares[index] is None else a_squares[index] + a_term
                    b_squares[index] = b_term if b_squares[index] is None else b_squares[index] + b_term
    if any(value is None for value in a_squares + b_squares):
        raise RuntimeError("O-LoRA norm audit found no complete three-slot Q/V adapters")
    return [
        {
            "slot": index,
            "a_l2": float(a_squares[index].sqrt().cpu().item()),
            "b_l2": float(b_squares[index].sqrt().cpu().item()),
        }
        for index in range(3)
    ]
