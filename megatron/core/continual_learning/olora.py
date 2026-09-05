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
    current_index = 1 if task == "code" else 2
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
        raise RuntimeError("O-LoRA found no Q/V adapter pairs in Layer 2--9")
    return float(coefficient) * torch.stack(losses).sum()


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
