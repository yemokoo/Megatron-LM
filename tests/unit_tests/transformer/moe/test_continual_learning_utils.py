# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch
from types import MethodType

from megatron.core.transformer.moe.continual_learning_utils import (
    allow_existing_router_grads,
    expand_moe_model,
    freeze_all_but_new_moe_params,
    freeze_all_but_new_shared_router_params,
    freeze_all_but_router_params,
    freeze_all_but_shared_router_params,
    teacher_student_router_kl,
)
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.shared_router_hybrid import (
    SharedFullRankLoraExperts,
    SharedRouterHybridTransformerLayer,
    TwoRouterHybridTransformerLayer,
    topk_with_all_new_experts_routing,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class RouterOnlyModel(torch.nn.Module):
    def __init__(self, num_experts=4, hidden_size=3):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_experts,
            moe_router_topk=2,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        self.router = TopKRouter(config)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)


class MinimalGroupedMLP(GroupedMLP):
    def __init__(self, num_experts=4, hidden_size=3, expert_width=2):
        torch.nn.Module.__init__(self)
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=expert_width,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        self.num_local_experts = num_experts
        self.weight1 = torch.nn.Parameter(torch.ones(hidden_size, num_experts * expert_width))
        self.weight2 = torch.nn.Parameter(torch.ones(num_experts * expert_width, hidden_size))


class RouterLoraAndGroupedExpertsModel(torch.nn.Module):
    def __init__(self, num_experts=4, hidden_size=3):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_experts,
            moe_router_topk=2,
            attn_lora_num_experts=num_experts,
            attn_full_rank_lora_rank=2,
            attn_full_rank_lora_alpha=2,
            attn_full_rank_lora_targets="qkvo",
            attn_full_rank_lora_active_targets="",
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        self.router = TopKRouter(config)
        self.attn_lora_experts = SharedFullRankLoraExperts(
            config,
            input_size=hidden_size,
            query_output_size=hidden_size,
            value_output_size=hidden_size,
        )
        self.ffn_experts = MinimalGroupedMLP(num_experts=num_experts, hidden_size=hidden_size)
        self.q_full_rank_lora = torch.nn.Linear(hidden_size, hidden_size)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)


class ConstantOutput(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", value.clone())

    def forward(self, hidden_states):
        return self.value.to(hidden_states.device, hidden_states.dtype)


class RecordingAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.routing_context = None

    def forward(self, hidden_states, **kwargs):
        self.routing_context = kwargs.get("routing_context")
        return torch.zeros_like(hidden_states)


class ZeroCrossAttention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states)


class RecordingMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.routing_context = None

    def forward(self, hidden_states, routing_context=None):
        self.routing_context = routing_context
        return torch.zeros_like(hidden_states)


class TwoRouterOnlyModel(torch.nn.Module):
    def __init__(self, num_experts=8, hidden_size=8):
        super().__init__()
        config = _two_router_test_config(num_experts=num_experts, hidden_size=hidden_size)
        self.attn_expert_router = TopKRouter(config)
        self.ffn_expert_router = TopKRouter(config)


def _two_router_test_config(num_experts=8, hidden_size=8, topk=4):
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=1,
        num_moe_experts=num_experts,
        moe_router_topk=topk,
        moe_router_load_balancing_type="none",
        moe_router_pre_softmax=True,
        moe_router_score_function="softmax",
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=None,
        attn_lora_num_experts=num_experts,
        attn_lora_topk=topk,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )


def _bda_no_dropout(training, bias_dropout_fusion):
    def inner(output_with_bias, residual, hidden_dropout):
        return residual + output_with_bias

    return inner


def _force_cpu_router_forward(router):
    def cpu_forward(self, input):
        logits = torch.nn.functional.linear(input.float(), self.weight.float())
        return self.routing(logits)

    router.forward = MethodType(cpu_forward, router)


def _build_two_router_forward_harness():
    config = _two_router_test_config()
    layer = TwoRouterHybridTransformerLayer.__new__(TwoRouterHybridTransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 1
    layer.hidden_dropout = 0.0
    layer.is_moe_layer = True
    layer.training = False
    layer.bias_dropout_add_exec_handler = torch.enable_grad
    layer.attn_expert_router = TopKRouter(config)
    layer.ffn_expert_router = TopKRouter(config)
    layer.shared_expert_router = None
    _force_cpu_router_forward(layer.attn_expert_router)
    _force_cpu_router_forward(layer.ffn_expert_router)

    attn_input = torch.tensor([[9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0]])
    ffn_input = torch.tensor([[0.0, 0.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0]])
    layer.input_layernorm = ConstantOutput(attn_input)
    layer.pre_cross_attn_layernorm = torch.nn.Identity()
    layer.pre_mlp_layernorm = ConstantOutput(ffn_input)
    layer.self_attention = RecordingAttention()
    layer.cross_attention = ZeroCrossAttention()
    layer.mlp = RecordingMlp()
    layer.self_attn_bda = _bda_no_dropout
    layer.cross_attn_bda = _bda_no_dropout
    layer.mlp_bda = _bda_no_dropout

    with torch.no_grad():
        identity_router_weight = torch.eye(8)
        layer.attn_expert_router.weight.copy_(identity_router_weight)
        layer.ffn_expert_router.weight.copy_(identity_router_weight)

    records = []

    def record_compute_routing(self, hidden_states, router, *, capture=True):
        records.append(
            {
                "router": "attn" if router is self.attn_expert_router else "ffn",
                "input": hidden_states.detach().clone(),
            }
        )
        return SharedRouterHybridTransformerLayer._compute_routing(
            self,
            hidden_states,
            router,
            capture=capture,
        )

    layer._compute_routing = MethodType(record_compute_routing, layer)
    return layer, attn_input, ffn_input, records


def _backward_router_weight_sum(model):
    model.zero_grad(set_to_none=True)
    model.router.weight.sum().backward()
    return model.router.weight.grad.detach()


def _trainable_expert_and_router_loss(model):
    loss = model.router.weight.sum()
    loss = loss + model.ffn_experts.weight1.sum() + model.ffn_experts.weight2.sum()
    for param in model.attn_lora_experts.parameters():
        loss = loss + param.sum()
    return loss


def test_freeze_all_but_new_moe_params_masks_existing_router_rows_by_default():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=True,
    )

    grad = _backward_router_weight_sum(model)

    assert torch.count_nonzero(grad[:2]) == 0
    assert torch.all(grad[2:] == 1)


def test_freeze_all_but_new_moe_params_can_train_all_router_rows():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=False,
    )

    grad = _backward_router_weight_sum(model)

    assert torch.all(grad == 1)
    assert not model.dense.weight.requires_grad


def test_freeze_all_but_new_moe_params_can_train_all_experts_and_router_rows():
    model = RouterLoraAndGroupedExpertsModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=False,
        freeze_existing_router=False,
        train_dense_attention_lora=False,
    )

    model.zero_grad(set_to_none=True)
    _trainable_expert_and_router_loss(model).backward()

    assert torch.all(model.router.weight.grad == 1)
    assert torch.all(model.ffn_experts.weight1.grad == 1)
    assert torch.all(model.ffn_experts.weight2.grad == 1)
    for param in model.attn_lora_experts.parameters():
        assert param.requires_grad
        assert torch.all(param.grad == 1)
    assert not model.q_full_rank_lora.weight.requires_grad
    assert model.q_full_rank_lora.weight.grad is None
    assert not model.dense.weight.requires_grad
    assert model.dense.weight.grad is None


def test_freeze_all_but_shared_router_params_trains_only_router_weights():
    model = RouterLoraAndGroupedExpertsModel()

    freeze_all_but_shared_router_params(model)

    model.zero_grad(set_to_none=True)
    _trainable_expert_and_router_loss(model).backward()

    assert torch.all(model.router.weight.grad == 1)
    assert model.ffn_experts.weight1.grad is None
    assert model.ffn_experts.weight2.grad is None
    for param in model.attn_lora_experts.parameters():
        assert not param.requires_grad
        assert param.grad is None
    assert not model.q_full_rank_lora.weight.requires_grad
    assert model.q_full_rank_lora.weight.grad is None
    assert not model.dense.weight.requires_grad
    assert model.dense.weight.grad is None


def test_freeze_all_but_router_params_trains_only_generic_moe_router_weights():
    model = RouterLoraAndGroupedExpertsModel()

    freeze_all_but_router_params(model)

    model.zero_grad(set_to_none=True)
    _trainable_expert_and_router_loss(model).backward()

    assert torch.all(model.router.weight.grad == 1)
    assert model.ffn_experts.weight1.grad is None
    assert model.ffn_experts.weight2.grad is None
    for param in model.attn_lora_experts.parameters():
        assert not param.requires_grad
        assert param.grad is None
    assert not model.q_full_rank_lora.weight.requires_grad
    assert model.q_full_rank_lora.weight.grad is None
    assert not model.dense.weight.requires_grad
    assert model.dense.weight.grad is None


def test_freeze_all_but_new_shared_router_params_trains_only_new_router_rows():
    model = RouterLoraAndGroupedExpertsModel()

    freeze_all_but_new_shared_router_params(model, num_existing_experts=2)

    model.zero_grad(set_to_none=True)
    _trainable_expert_and_router_loss(model).backward()

    assert torch.count_nonzero(model.router.weight.grad[:2]) == 0
    assert torch.all(model.router.weight.grad[2:] == 1)
    assert model.ffn_experts.weight1.grad is None
    assert model.ffn_experts.weight2.grad is None
    for param in model.attn_lora_experts.parameters():
        assert not param.requires_grad
        assert param.grad is None
    assert not model.q_full_rank_lora.weight.requires_grad
    assert model.q_full_rank_lora.weight.grad is None
    assert not model.dense.weight.requires_grad
    assert model.dense.weight.grad is None


def test_allow_existing_router_grads_temporarily_bypasses_existing_row_mask():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=True,
    )

    model.zero_grad(set_to_none=True)
    with allow_existing_router_grads():
        model.router.weight.sum().backward()
    grad = model.router.weight.grad.detach()

    assert torch.all(grad == 1)


def test_zero_weight_router_kd_does_not_change_accumulated_lm_router_grads():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=True,
    )

    model.zero_grad(set_to_none=True)
    model.router.weight.sum().backward()
    lm_grad = model.router.weight.grad.detach().clone()

    with allow_existing_router_grads():
        (model.router.weight.sum() * 0.0).backward()
    grad_after_zero_weight_kd = model.router.weight.grad.detach()

    assert torch.equal(grad_after_zero_weight_kd, lm_grad)
    assert torch.count_nonzero(grad_after_zero_weight_kd[:2]) == 0
    assert torch.all(grad_after_zero_weight_kd[2:] == 1)


def test_teacher_student_router_kl_zero_padding_updates_new_student_logits():
    student_logits = torch.tensor([[0.3, -0.2, 1.1, -0.7]], requires_grad=True)
    teacher_logits = torch.tensor([[1.0, -0.5]])

    loss = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=False,
    )
    loss.backward()

    assert torch.count_nonzero(student_logits.grad[:, 2:]) == 2


def test_teacher_student_router_kl_existing_only_matches_manual_wiki_row_kl():
    student_logits = torch.tensor(
        [
            [0.3, -0.2, 1.1, -0.7],
            [-0.4, 0.8, 0.5, 2.0],
        ]
    )
    teacher_logits = torch.tensor(
        [
            [1.0, -0.5],
            [-0.3, 0.2],
        ]
    )

    actual = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=True,
    )

    teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
    student_wiki_log_probs = torch.log_softmax(student_logits.float()[:, :2], dim=-1)
    expected = torch.nn.functional.kl_div(
        student_wiki_log_probs,
        teacher_probs,
        reduction="batchmean",
    )

    assert torch.allclose(actual, expected)


def test_teacher_student_router_kl_existing_only_is_invariant_to_new_logits():
    teacher_logits = torch.tensor([[1.0, -0.5]])
    student_logits = torch.tensor([[0.3, -0.2, 1.1, -0.7]])
    changed_new_logits = torch.tensor([[0.3, -0.2, 100.0, -100.0]])

    original_loss = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=True,
    )
    changed_loss = teacher_student_router_kl(
        changed_new_logits,
        teacher_logits,
        existing_experts_only=True,
    )

    assert torch.equal(original_loss, changed_loss)


def test_teacher_student_router_kl_existing_only_ignores_new_student_logits():
    student_logits = torch.tensor([[0.3, -0.2, 1.1, -0.7]], requires_grad=True)
    teacher_logits = torch.tensor([[1.0, -0.5]])

    loss = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=True,
    )
    loss.backward()

    assert torch.count_nonzero(student_logits.grad[:, :2]) == 2
    assert torch.count_nonzero(student_logits.grad[:, 2:]) == 0


def test_topk_with_all_new_experts_uses_union_and_original_softmax_weights():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.5, -0.5]])

    scores, routing_map = topk_with_all_new_experts_routing(
        logits,
        topk=2,
        num_existing_experts=4,
    )

    full_scores = torch.softmax(logits.float(), dim=-1)
    expected_map = torch.tensor([[True, True, False, False, True, True]])
    expected_scores = full_scores * expected_map.to(full_scores.dtype)

    assert torch.equal(routing_map, expected_map)
    assert torch.allclose(scores, expected_scores)
    assert scores.sum() < 1.0


def test_topk_with_all_new_experts_does_not_duplicate_new_topk_experts():
    logits = torch.tensor([[0.0, 0.1, 0.2, 0.3, 9.0, 8.0]])

    scores, routing_map = topk_with_all_new_experts_routing(
        logits,
        topk=2,
        num_existing_experts=4,
    )

    assert torch.equal(routing_map, torch.tensor([[False, False, False, False, True, True]]))
    assert torch.allclose(scores, torch.softmax(logits.float(), dim=-1) * routing_map.float())


def test_g2_2router_forward_uses_independent_attn_and_ffn_router_inputs():
    layer, attn_input, ffn_input, records = _build_two_router_forward_harness()

    output, context = TwoRouterHybridTransformerLayer.forward(layer, torch.zeros_like(attn_input))

    attn_record, ffn_record = records
    attn_map = layer.self_attention.routing_context.routing_map
    ffn_map = layer.mlp.routing_context.routing_map

    assert context is None
    assert torch.equal(output, torch.zeros_like(attn_input))
    assert attn_record["router"] == "attn"
    assert ffn_record["router"] == "ffn"
    assert torch.equal(attn_record["input"], attn_input)
    assert torch.equal(ffn_record["input"], ffn_input)
    assert layer.attn_expert_router is not layer.ffn_expert_router
    assert torch.equal(attn_map, torch.tensor([[True, True, True, True, False, False, False, False]]))
    assert torch.equal(ffn_map, torch.tensor([[False, False, False, False, True, True, True, True]]))
    assert torch.all(attn_map.sum(dim=-1) == 4)
    assert torch.all(ffn_map.sum(dim=-1) == 4)
    assert not torch.equal(attn_map, ffn_map)
    print("[g2-2router] OK: attn router uses LN_attn(x), FFN router uses LN_ffn(h), top4 differs")


def test_g2_2router_code_expansion_copies_both_router_existing_rows():
    source = TwoRouterOnlyModel(num_experts=8, hidden_size=8)
    target = TwoRouterOnlyModel(num_experts=16, hidden_size=8)

    with torch.no_grad():
        source.attn_expert_router.weight.copy_(torch.arange(64, dtype=torch.float32).view(8, 8))
        source.ffn_expert_router.weight.copy_(
            torch.arange(1000, 1064, dtype=torch.float32).view(8, 8)
        )
        target.attn_expert_router.weight.fill_(-1.0)
        target.ffn_expert_router.weight.fill_(-2.0)

    expand_moe_model(target, source, num_existing_experts=8)

    assert torch.equal(
        target.attn_expert_router.weight[:8],
        source.attn_expert_router.weight,
    )
    assert torch.equal(
        target.ffn_expert_router.weight[:8],
        source.ffn_expert_router.weight,
    )
    assert torch.all(target.attn_expert_router.weight[8:] == -1.0)
    assert torch.all(target.ffn_expert_router.weight[8:] == -2.0)
    print("[g2-2router] OK: code expansion copies existing rows for both added routers")
