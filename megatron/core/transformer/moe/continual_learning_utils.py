import re
from typing import Any, Dict, List

import torch

from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.qv_lora_attention import QVLoraExpertRouter
from megatron.core.transformer.shared_router_hybrid import SharedQVLoraExperts


_EXPERT_SUFFIX_RE = re.compile(r"(?:weight|bias)(\d+)$")


def _load_matching_state(dst_module, src_module):
    dst_state = dst_module.state_dict()
    src_state = src_module.state_dict()
    matched_state = {
        name: tensor
        for name, tensor in src_state.items()
        if (
            name in dst_state
            and dst_state[name] is not None
            and tensor is not None
            and dst_state[name].shape == tensor.shape
        )
    }
    dst_module.load_state_dict(matched_state, strict=False)


def _copy_router(dst_router, src_router, num_existing_experts):
    with torch.no_grad():
        dst_router.weight[:num_existing_experts].copy_(src_router.weight[:num_existing_experts])
        if getattr(dst_router, "expert_bias", None) is not None and getattr(
            src_router, "expert_bias", None
        ) is not None:
            dst_router.expert_bias[:num_existing_experts].copy_(
                src_router.expert_bias[:num_existing_experts]
            )


def _copy_grouped_mlp(dst_mlp, src_mlp, num_existing_experts):
    with torch.no_grad():
        dst_weight1 = dst_mlp.weight1.view(dst_mlp.config.hidden_size, dst_mlp.num_local_experts, -1)
        src_weight1 = src_mlp.weight1.view(src_mlp.config.hidden_size, src_mlp.num_local_experts, -1)
        dst_weight1[:, :num_existing_experts, :].copy_(src_weight1[:, :num_existing_experts, :])

        dst_weight2 = dst_mlp.weight2.view(dst_mlp.num_local_experts, -1, dst_mlp.config.hidden_size)
        src_weight2 = src_mlp.weight2.view(src_mlp.num_local_experts, -1, src_mlp.config.hidden_size)
        dst_weight2[:num_existing_experts, :, :].copy_(src_weight2[:num_existing_experts, :, :])


def _copy_qv_lora_router(dst_router, src_router, num_existing_experts):
    with torch.no_grad():
        dst_router.router_weight[:num_existing_experts].copy_(
            src_router.router_weight[:num_existing_experts]
        )
        dst_router.q_lora_a[:num_existing_experts].copy_(src_router.q_lora_a[:num_existing_experts])
        dst_router.q_lora_b[:num_existing_experts].copy_(src_router.q_lora_b[:num_existing_experts])
        dst_router.v_lora_a[:num_existing_experts].copy_(src_router.v_lora_a[:num_existing_experts])
        dst_router.v_lora_b[:num_existing_experts].copy_(src_router.v_lora_b[:num_existing_experts])
        if (
            getattr(dst_router, "o_lora_a", None) is not None
            and getattr(src_router, "o_lora_a", None) is not None
        ):
            dst_router.o_lora_a[:num_existing_experts].copy_(
                src_router.o_lora_a[:num_existing_experts]
            )
            dst_router.o_lora_b[:num_existing_experts].copy_(
                src_router.o_lora_b[:num_existing_experts]
            )


def _copy_shared_qv_lora_experts(dst_experts, src_experts, num_existing_experts):
    with torch.no_grad():
        dst_experts.q_lora_a[:num_existing_experts].copy_(
            src_experts.q_lora_a[:num_existing_experts]
        )
        dst_experts.q_lora_b[:num_existing_experts].copy_(
            src_experts.q_lora_b[:num_existing_experts]
        )
        dst_experts.v_lora_a[:num_existing_experts].copy_(
            src_experts.v_lora_a[:num_existing_experts]
        )
        dst_experts.v_lora_b[:num_existing_experts].copy_(
            src_experts.v_lora_b[:num_existing_experts]
        )
        if (
            getattr(dst_experts, "o_lora_a", None) is not None
            and getattr(src_experts, "o_lora_a", None) is not None
        ):
            dst_experts.o_lora_a[:num_existing_experts].copy_(
                src_experts.o_lora_a[:num_existing_experts]
            )
            dst_experts.o_lora_b[:num_existing_experts].copy_(
                src_experts.o_lora_b[:num_existing_experts]
            )


def expand_moe_model(target_model, source_model, num_existing_experts):
    _load_matching_state(target_model, source_model)

    source_modules = dict(source_model.named_modules())
    for module_name, target_module in target_model.named_modules():
        source_module = source_modules.get(module_name)
        if source_module is None:
            continue

        if isinstance(target_module, Router) and isinstance(source_module, Router):
            _copy_router(target_module, source_module, num_existing_experts)
        elif isinstance(target_module, GroupedMLP) and isinstance(source_module, GroupedMLP):
            _copy_grouped_mlp(target_module, source_module, num_existing_experts)
        elif isinstance(target_module, QVLoraExpertRouter) and isinstance(
            source_module, QVLoraExpertRouter
        ):
            _copy_qv_lora_router(target_module, source_module, num_existing_experts)
        elif isinstance(target_module, SharedQVLoraExperts) and isinstance(
            source_module, SharedQVLoraExperts
        ):
            _copy_shared_qv_lora_experts(target_module, source_module, num_existing_experts)


def _freeze_router(module, num_existing_experts):
    def _zero_existing_router_grads(grad):
        grad = grad.clone()
        grad[:num_existing_experts].zero_()
        return grad

    module.weight.register_hook(_zero_existing_router_grads)


def _freeze_grouped_experts(module, num_existing_experts):
    weight1_per_expert = module.weight1.shape[1] // module.num_local_experts
    weight2_per_expert = module.weight2.shape[0] // module.num_local_experts

    def _zero_existing_weight1_grads(grad):
        grad = grad.clone()
        grad[:, : num_existing_experts * weight1_per_expert].zero_()
        return grad

    def _zero_existing_weight2_grads(grad):
        grad = grad.clone()
        grad[: num_existing_experts * weight2_per_expert, :].zero_()
        return grad

    module.weight1.register_hook(_zero_existing_weight1_grads)
    module.weight2.register_hook(_zero_existing_weight2_grads)


def _freeze_qv_lora_experts(module, num_existing_experts):
    def _zero_existing_expert_grads(grad):
        grad = grad.clone()
        grad[:num_existing_experts].zero_()
        return grad

    module.q_lora_a.register_hook(_zero_existing_expert_grads)
    module.q_lora_b.register_hook(_zero_existing_expert_grads)
    module.v_lora_a.register_hook(_zero_existing_expert_grads)
    module.v_lora_b.register_hook(_zero_existing_expert_grads)
    if getattr(module, "o_lora_a", None) is not None:
        module.o_lora_a.register_hook(_zero_existing_expert_grads)
        module.o_lora_b.register_hook(_zero_existing_expert_grads)


def _freeze_qv_lora_router(module, num_existing_experts):
    def _zero_existing_router_grads(grad):
        grad = grad.clone()
        grad[:num_existing_experts].zero_()
        return grad

    module.router_weight.register_hook(_zero_existing_router_grads)


def _freeze_te_grouped_experts(module, num_existing_experts):
    for name, param in module.named_parameters():
        match = _EXPERT_SUFFIX_RE.search(name)
        if match is not None and int(match.group(1)) < num_existing_experts:
            param.requires_grad = False


def freeze_preexisting_moe_params(
    model, num_existing_experts, freeze_existing_experts, freeze_existing_router
):
    for module in model.modules():
        if freeze_existing_router and isinstance(module, Router):
            _freeze_router(module, num_existing_experts)

        if not freeze_existing_experts:
            continue

        if isinstance(module, SequentialMLP):
            for expert_idx, expert in enumerate(module.local_experts):
                if expert_idx >= num_existing_experts:
                    break
                for param in expert.parameters():
                    param.requires_grad = False
        elif isinstance(module, GroupedMLP):
            _freeze_grouped_experts(module, num_existing_experts)
        elif isinstance(module, TEGroupedMLP):
            _freeze_te_grouped_experts(module, num_existing_experts)
        elif isinstance(module, QVLoraExpertRouter):
            if freeze_existing_router:
                _freeze_qv_lora_router(module, num_existing_experts)
            if freeze_existing_experts:
                _freeze_qv_lora_experts(module, num_existing_experts)
        elif isinstance(module, SharedQVLoraExperts):
            if freeze_existing_experts:
                _freeze_qv_lora_experts(module, num_existing_experts)


def freeze_all_but_new_moe_params(
    model, num_existing_experts, freeze_existing_experts=True, freeze_existing_router=True
):
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, Router):
            module.weight.requires_grad = True
            if getattr(module, "expert_bias", None) is not None:
                module.expert_bias.requires_grad = True
            if freeze_existing_router:
                _freeze_router(module, num_existing_experts)
            continue

        if isinstance(module, SequentialMLP):
            for expert_idx, expert in enumerate(module.local_experts):
                trainable = expert_idx >= num_existing_experts if freeze_existing_experts else True
                for param in expert.parameters():
                    param.requires_grad = trainable
            continue

        if isinstance(module, GroupedMLP):
            module.weight1.requires_grad = True
            module.weight2.requires_grad = True
            if freeze_existing_experts:
                _freeze_grouped_experts(module, num_existing_experts)
            continue

        if isinstance(module, TEGroupedMLP):
            for name, param in module.named_parameters():
                match = _EXPERT_SUFFIX_RE.search(name)
                if match is None:
                    continue
                expert_idx = int(match.group(1))
                param.requires_grad = expert_idx >= num_existing_experts if freeze_existing_experts else True
            continue

        if isinstance(module, QVLoraExpertRouter):
            module.router_weight.requires_grad = True
            module.q_lora_a.requires_grad = True
            module.q_lora_b.requires_grad = True
            module.v_lora_a.requires_grad = True
            module.v_lora_b.requires_grad = True
            if getattr(module, "o_lora_a", None) is not None:
                module.o_lora_a.requires_grad = True
                module.o_lora_b.requires_grad = True
            if freeze_existing_router:
                _freeze_qv_lora_router(module, num_existing_experts)
            if freeze_existing_experts:
                _freeze_qv_lora_experts(module, num_existing_experts)
            continue

        if isinstance(module, SharedQVLoraExperts):
            module.q_lora_a.requires_grad = True
            module.q_lora_b.requires_grad = True
            module.v_lora_a.requires_grad = True
            module.v_lora_b.requires_grad = True
            if getattr(module, "o_lora_a", None) is not None:
                module.o_lora_a.requires_grad = True
                module.o_lora_b.requires_grad = True
            if freeze_existing_experts:
                _freeze_qv_lora_experts(module, num_existing_experts)
            continue

        # Keep dense attention full-rank LoRA trainable even when shared weights are frozen.
        qkv_full_rank_lora = getattr(module, "qkv_full_rank_lora", None)
        if qkv_full_rank_lora is not None:
            for param in qkv_full_rank_lora.parameters():
                param.requires_grad = True

        for attr_name in ("q_full_rank_lora", "k_full_rank_lora", "v_full_rank_lora"):
            adapter = getattr(module, attr_name, None)
            if adapter is not None:
                for param in adapter.parameters():
                    param.requires_grad = True

        proj_full_rank_lora = getattr(module, "proj_full_rank_lora", None)
        if proj_full_rank_lora is not None:
            for param in proj_full_rank_lora.parameters():
                param.requires_grad = True




def freeze_all_but_attn_lora_router_params(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, QVLoraExpertRouter):
            for param in module.parameters():
                param.requires_grad = True

def _tensor_summary(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _max_abs_diff(lhs, rhs):
    with torch.no_grad():
        return float((lhs.detach().float() - rhs.detach().float()).abs().max().item())


def _audit_router(module_name, target_router, source_router, num_existing_experts):
    record = {
        "module": module_name,
        "target_weight": _tensor_summary(target_router.weight),
        "source_weight": _tensor_summary(source_router.weight),
        "copied_rows": num_existing_experts,
        "copied_router_rows_match": bool(
            torch.equal(
                target_router.weight[:num_existing_experts].detach().cpu(),
                source_router.weight[:num_existing_experts].detach().cpu(),
            )
        ),
        "copied_router_rows_max_abs_diff": _max_abs_diff(
            target_router.weight[:num_existing_experts],
            source_router.weight[:num_existing_experts],
        ),
        "new_router_rows": max(target_router.weight.shape[0] - num_existing_experts, 0),
    }
    if getattr(target_router, "expert_bias", None) is not None:
        record["target_expert_bias"] = _tensor_summary(target_router.expert_bias)
    if getattr(source_router, "expert_bias", None) is not None:
        record["source_expert_bias"] = _tensor_summary(source_router.expert_bias)
    return record


def _audit_sequential_mlp(module_name, target_mlp, source_mlp, num_existing_experts):
    experts: List[Dict[str, Any]] = []
    for expert_idx, target_expert in enumerate(target_mlp.local_experts):
        expert_record = {
            "expert_index": expert_idx,
            "trainable_after_freeze_expected": expert_idx >= num_existing_experts,
            "fc1_weight": _tensor_summary(target_expert.linear_fc1.weight),
            "fc2_weight": _tensor_summary(target_expert.linear_fc2.weight),
        }
        if expert_idx < len(source_mlp.local_experts):
            source_expert = source_mlp.local_experts[expert_idx]
            copied_params_match = True
            copied_params_max_abs_diff = 0.0
            for target_param, source_param in zip(target_expert.parameters(), source_expert.parameters()):
                copied_params_match = copied_params_match and torch.equal(
                    target_param.detach().cpu(), source_param.detach().cpu()
                )
                copied_params_max_abs_diff = max(
                    copied_params_max_abs_diff,
                    _max_abs_diff(target_param, source_param),
                )
            expert_record["copied_from_source"] = True
            expert_record["copied_params_match"] = copied_params_match
            expert_record["copied_params_max_abs_diff"] = copied_params_max_abs_diff
        else:
            expert_record["copied_from_source"] = False
        experts.append(expert_record)

    fc1_shapes = [expert["fc1_weight"]["shape"] for expert in experts]
    fc2_shapes = [expert["fc2_weight"]["shape"] for expert in experts]
    return {
        "module": module_name,
        "type": "SequentialMLP",
        "num_target_experts": len(target_mlp.local_experts),
        "num_source_experts": len(source_mlp.local_experts),
        "all_fc1_shapes_identical": len({tuple(shape) for shape in fc1_shapes}) == 1,
        "all_fc2_shapes_identical": len({tuple(shape) for shape in fc2_shapes}) == 1,
        "experts": experts,
    }


def _audit_qv_lora_router(module_name, target_router, source_router, num_existing_experts):
    record = {
        "module": module_name,
        "type": "QVLoraExpertRouter",
        "copied_rows": num_existing_experts,
        "num_target_experts": int(target_router.router_weight.shape[0]),
        "num_source_experts": int(source_router.router_weight.shape[0]),
        "router_weight": _tensor_summary(target_router.router_weight),
        "q_lora_a": _tensor_summary(target_router.q_lora_a),
        "q_lora_b": _tensor_summary(target_router.q_lora_b),
        "v_lora_a": _tensor_summary(target_router.v_lora_a),
        "v_lora_b": _tensor_summary(target_router.v_lora_b),
        "copied_router_rows_match": bool(
            torch.equal(
                target_router.router_weight[:num_existing_experts].detach().cpu(),
                source_router.router_weight[:num_existing_experts].detach().cpu(),
            )
        ),
        "copied_q_lora_a_max_abs_diff": _max_abs_diff(
            target_router.q_lora_a[:num_existing_experts],
            source_router.q_lora_a[:num_existing_experts],
        ),
        "copied_q_lora_b_max_abs_diff": _max_abs_diff(
            target_router.q_lora_b[:num_existing_experts],
            source_router.q_lora_b[:num_existing_experts],
        ),
        "copied_v_lora_a_max_abs_diff": _max_abs_diff(
            target_router.v_lora_a[:num_existing_experts],
            source_router.v_lora_a[:num_existing_experts],
        ),
        "copied_v_lora_b_max_abs_diff": _max_abs_diff(
            target_router.v_lora_b[:num_existing_experts],
            source_router.v_lora_b[:num_existing_experts],
        ),
    }
    if (
        getattr(target_router, "o_lora_a", None) is not None
        and getattr(source_router, "o_lora_a", None) is not None
    ):
        record["o_lora_a"] = _tensor_summary(target_router.o_lora_a)
        record["o_lora_b"] = _tensor_summary(target_router.o_lora_b)
        record["copied_o_lora_a_max_abs_diff"] = _max_abs_diff(
            target_router.o_lora_a[:num_existing_experts],
            source_router.o_lora_a[:num_existing_experts],
        )
        record["copied_o_lora_b_max_abs_diff"] = _max_abs_diff(
            target_router.o_lora_b[:num_existing_experts],
            source_router.o_lora_b[:num_existing_experts],
        )
    return record


def _audit_shared_qv_lora_experts(module_name, target_experts, source_experts, num_existing_experts):
    record = {
        "module": module_name,
        "type": "SharedQVLoraExperts",
        "copied_rows": num_existing_experts,
        "num_target_experts": int(target_experts.q_lora_a.shape[0]),
        "num_source_experts": int(source_experts.q_lora_a.shape[0]),
        "q_lora_a": _tensor_summary(target_experts.q_lora_a),
        "q_lora_b": _tensor_summary(target_experts.q_lora_b),
        "v_lora_a": _tensor_summary(target_experts.v_lora_a),
        "v_lora_b": _tensor_summary(target_experts.v_lora_b),
        "copied_q_lora_a_max_abs_diff": _max_abs_diff(
            target_experts.q_lora_a[:num_existing_experts],
            source_experts.q_lora_a[:num_existing_experts],
        ),
        "copied_q_lora_b_max_abs_diff": _max_abs_diff(
            target_experts.q_lora_b[:num_existing_experts],
            source_experts.q_lora_b[:num_existing_experts],
        ),
        "copied_v_lora_a_max_abs_diff": _max_abs_diff(
            target_experts.v_lora_a[:num_existing_experts],
            source_experts.v_lora_a[:num_existing_experts],
        ),
        "copied_v_lora_b_max_abs_diff": _max_abs_diff(
            target_experts.v_lora_b[:num_existing_experts],
            source_experts.v_lora_b[:num_existing_experts],
        ),
    }
    if (
        getattr(target_experts, "o_lora_a", None) is not None
        and getattr(source_experts, "o_lora_a", None) is not None
    ):
        record["o_lora_a"] = _tensor_summary(target_experts.o_lora_a)
        record["o_lora_b"] = _tensor_summary(target_experts.o_lora_b)
        record["copied_o_lora_a_max_abs_diff"] = _max_abs_diff(
            target_experts.o_lora_a[:num_existing_experts],
            source_experts.o_lora_a[:num_existing_experts],
        )
        record["copied_o_lora_b_max_abs_diff"] = _max_abs_diff(
            target_experts.o_lora_b[:num_existing_experts],
            source_experts.o_lora_b[:num_existing_experts],
        )
    return record


def inspect_moe_expansion(target_model, source_model, num_existing_experts):
    audit: Dict[str, Any] = {
        "num_existing_experts": num_existing_experts,
        "routers": [],
        "expert_modules": [],
    }

    source_modules = dict(source_model.named_modules())
    for module_name, target_module in target_model.named_modules():
        source_module = source_modules.get(module_name)
        if source_module is None:
            continue

        if isinstance(target_module, Router) and isinstance(source_module, Router):
            audit["routers"].append(
                _audit_router(module_name, target_module, source_module, num_existing_experts)
            )
        elif isinstance(target_module, SequentialMLP) and isinstance(source_module, SequentialMLP):
            audit["expert_modules"].append(
                _audit_sequential_mlp(module_name, target_module, source_module, num_existing_experts)
            )
        elif isinstance(target_module, QVLoraExpertRouter) and isinstance(
            source_module, QVLoraExpertRouter
        ):
            audit["expert_modules"].append(
                _audit_qv_lora_router(
                    module_name, target_module, source_module, num_existing_experts
                )
            )
        elif isinstance(target_module, SharedQVLoraExperts) and isinstance(
            source_module, SharedQVLoraExperts
        ):
            audit["expert_modules"].append(
                _audit_shared_qv_lora_experts(
                    module_name, target_module, source_module, num_existing_experts
                )
            )

    return audit
