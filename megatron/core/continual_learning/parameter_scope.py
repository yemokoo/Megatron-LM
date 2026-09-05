"""Exact Layer-2--9 trainability masks shared by all six baselines."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Iterator, List, Tuple

import torch

from megatron.core.transformer.transformer_layer import TransformerLayer

from .lora_adapter import ContinualLowRankAdapter


def iter_shards(model) -> Iterator[torch.nn.Module]:
    if isinstance(model, (list, tuple)):
        yield from model
    else:
        yield model


def iter_layers(model) -> Iterator[Tuple[int, TransformerLayer]]:
    for shard in iter_shards(model):
        for module in shard.modules():
            if isinstance(module, TransformerLayer):
                yield int(module.layer_number), module


def named_parameters(model) -> Iterator[Tuple[str, torch.nn.Parameter]]:
    for shard_index, shard in enumerate(iter_shards(model)):
        prefix = f"shard{shard_index}."
        for name, parameter in shard.named_parameters():
            yield prefix + name, parameter


def _set_all(model, value: bool) -> None:
    for _, parameter in named_parameters(model):
        parameter.requires_grad_(value)


def _adapter_slot(module: ContinualLowRankAdapter, slot: int, parent) -> int:
    for collection_name in (
        "continual_q_adapters",
        "continual_k_adapters",
        "continual_v_adapters",
        "continual_o_adapters",
    ):
        collection = getattr(parent, collection_name, ())
        for index, candidate in enumerate(collection):
            if candidate is module:
                return index
    return 0


def apply_parameter_scope(model, method: str, task: str, layer_start: int, layer_end: int) -> dict:
    """Apply the agreed Wiki and post-Wiki trainability rules."""
    if task == "wiki":
        _set_all(model, True)
        if method == "slora_pre":
            for shard in iter_shards(model):
                for module in shard.modules():
                    if isinstance(module, ContinualLowRankAdapter):
                        for parameter in module.parameters():
                            parameter.requires_grad_(False)
        elif method == "olora":
            for _, layer in iter_layers(model):
                for parent in layer.modules():
                    for collection_name in ("continual_q_adapters", "continual_v_adapters"):
                        for index, adapter in enumerate(getattr(parent, collection_name, ())):
                            for parameter in adapter.parameters():
                                parameter.requires_grad_(index == 0)
    else:
        _set_all(model, False)
        if method in {"ewc", "trace_gem", "sequential_dense", "fixed_moe"}:
            for layer_number, layer in iter_layers(model):
                if layer_start <= layer_number <= layer_end:
                    for parameter in layer.parameters():
                        parameter.requires_grad_(True)
        elif method == "slora_pre":
            for layer_number, layer in iter_layers(model):
                if not layer_start <= layer_number <= layer_end:
                    continue
                for module in layer.modules():
                    if isinstance(module, ContinualLowRankAdapter):
                        for parameter in module.parameters():
                            parameter.requires_grad_(True)
        elif method == "olora":
            current_slot = 1 if task == "code" else 2
            for layer_number, layer in iter_layers(model):
                if not layer_start <= layer_number <= layer_end:
                    continue
                for parent in layer.modules():
                    for collection_name in ("continual_q_adapters", "continual_v_adapters"):
                        for index, adapter in enumerate(getattr(parent, collection_name, ())):
                            for parameter in adapter.parameters():
                                parameter.requires_grad_(index == current_slot)
        else:
            raise ValueError(f"unsupported continual method: {method}")

    trainable = [(name, p) for name, p in named_parameters(model) if p.requires_grad]
    frozen = [(name, p) for name, p in named_parameters(model) if not p.requires_grad]
    return {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "frozen_tensors": len(frozen),
        "frozen_parameters": sum(p.numel() for _, p in frozen),
        "trainable_names": [name for name, _ in trainable],
    }


def scoped_trainable_parameters(model, layer_start: int, layer_end: int):
    """Return trainable parameters in Layer 2--9, keyed by stable shard name."""
    allowed_ids = set()
    for layer_number, layer in iter_layers(model):
        if layer_start <= layer_number <= layer_end:
            allowed_ids.update(id(parameter) for parameter in layer.parameters())
    return {
        name: parameter
        for name, parameter in named_parameters(model)
        if parameter.requires_grad and id(parameter) in allowed_ids
    }


def frozen_checksum(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters(model):
        if parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def outside_layer_scope_checksum(model, layer_start: int, layer_end: int) -> str:
    """Checksum embeddings/head/Layer 1 without including mergeable target layers."""
    scoped_ids = set()
    for layer_number, layer in iter_layers(model):
        if layer_start <= layer_number <= layer_end:
            scoped_ids.update(id(parameter) for parameter in layer.parameters())
    digest = hashlib.sha256()
    for name, parameter in named_parameters(model):
        if id(parameter) in scoped_ids:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
