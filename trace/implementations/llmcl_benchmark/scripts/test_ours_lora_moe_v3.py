#!/usr/bin/env python
"""CPU invariants for Ours V3 shared-router QKVO+FFN experts."""

import copy
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The repository can be synchronized while this structural test is running.
# Keep the model-only invariants independent of TRACE's dataset upload; real
# training still imports and requires the concrete collators.
try:
    import utils.data.data_collator  # noqa: F401
except ModuleNotFoundError:
    data_package = types.ModuleType("utils.data")
    data_package.__path__ = []
    collator_module = types.ModuleType("utils.data.data_collator")
    for class_name in (
            "DataCollator", "SLoRATraceDataCollator",
            "PreTokenizedSLoRATraceDataCollator"):
        setattr(collator_module, class_name, type(class_name, (), {}))
    sys.modules["utils.data"] = data_package
    sys.modules["utils.data.data_collator"] = collator_module

from model.Ours_LoRA_MoE_V3 import (  # noqa: E402
    ATTENTION_TARGETS,
    Ours_LoRA_MoE_V3,
    Ours_LoRA_MoE_V3_New,
    RoutingContext,
    SharedExpertRouter,
    V3_NEW_TRAINING_VERSIONS,
    add_v3_experts,
    attach_shared_qkvo_lora_moe,
    collect_v3_moe_losses,
    freeze_v3_experts,
    freeze_v3_routers,
    limit_v3_experts,
    load_v3_checkpoint,
    save_v3_meta,
    set_v3_router_token_mask,
    shared_router_layers,
)
from training.main_Ours_LoRA_MoE import (  # noqa: E402
    V2_NEW_MEMORY_TRAINING_VERSIONS,
    V2_NEW_TRAINING_VERSIONS,
    V3_TRAINING_VERSIONS,
    parse_args,
    resolve_training_version_defaults,
    validate_v2_new_args,
)
from utils.chat_templates import (  # noqa: E402
    LLAMA31_STANDARD_CHAT_TEMPLATE,
    LLAMA31_STANDARD_TEMPLATE_VERSION,
    ensure_llama31_chat_template,
)
from training.main_Ours_LoRA_MoE import (  # noqa: E402
    validate_v2_new_resume_persisted_identities,
)


class TinyAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        mixed = (
            self.q_proj(hidden_states)
            + self.k_proj(hidden_states)
            + self.v_proj(hidden_states)) / 3.0
        return self.o_proj(torch.tanh(mixed)), None


class TinyMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, inputs):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(inputs))
            * self.up_proj(inputs))


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size=6, intermediate_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = TinyAttention(hidden_size)
        self.mlp = TinyMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, output_attentions=False, **kwargs):
        residual = hidden_states
        hidden_states, weights = self.self_attn(
            self.input_layernorm(hidden_states), **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (weights,)
        return outputs


class TinyBackbone(nn.Module):
    def __init__(self, layers=2, hidden_size=6):
        super().__init__()
        self.layers = nn.ModuleList([
            TinyDecoderLayer(hidden_size=hidden_size) for _ in range(layers)])


class TinyModel(nn.Module):
    def __init__(self, layers=2, hidden_size=6):
        super().__init__()
        self.model = TinyBackbone(layers=layers, hidden_size=hidden_size)
        self.config = SimpleNamespace()

    def forward(self, hidden_states, **kwargs):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, **kwargs)[0]
        return hidden_states


class TinyCausalModel(TinyModel):
    def __init__(self, layers=1, hidden_size=6, vocab_size=24):
        super().__init__(layers=layers, hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, labels=None,
                use_cache=False, **kwargs):
        self.forward_calls += 1
        hidden_states = self.embedding(input_ids)
        hidden_states = super().forward(hidden_states, **kwargs)
        logits = self.lm_head(hidden_states)
        loss = logits.float().square().mean() if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class TinyBatchLoader:
    """Minimal deterministic loader exposing the attributes V2/V3 use."""

    def __init__(self, batches, batch_size):
        self.batches = list(batches)
        self.batch_size = batch_size
        self.dataset = range(len(self.batches) * batch_size)
        self.sampler = SimpleNamespace()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr=lr, weight_decay=0.01)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def make_causal_v3(experts, top_k=2,
                   routing_weight_mode="full_softmax"):
    model = TinyCausalModel()
    attach_shared_qkvo_lora_moe(
        model, r=2, alpha=4, top_k=top_k,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode=routing_weight_mode, dropout=0.0)
    if experts:
        add_v3_experts(model, experts)
    return model


def make_batch(values, source):
    ids = torch.tensor(values)
    return {
        "input_ids": ids,
        "attention_mask": ids.ne(0).long(),
        "labels": ids.clone(),
        "sources": [source] * ids.shape[0],
    }


def install_counting_engine(trainer):
    calls = []

    def reinit_engine(num_training_steps, learning_rate=None):
        lr = trainer.args.learning_rate if learning_rate is None \
            else learning_rate
        calls.append((num_training_steps, lr))
        trainable = [
            parameter for parameter in trainer.raw_model.parameters()
            if parameter.requires_grad]
        trainer.optimizer = CountingAdamW(trainable, lr=lr)
        trainer.lr_scheduler = CountingScheduler()
        trainer.model = trainer.raw_model

    trainer._reinit_engine = reinit_engine
    return calls


def expert_state(model, expert_index):
    fragment = f".experts.{expert_index}."
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if fragment in name
    }


def make_v3(layers=2, experts=1, top_k=1):
    model = TinyModel(layers=layers)
    attach_shared_qkvo_lora_moe(
        model, r=2, alpha=4, top_k=top_k,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="full_softmax", dropout=0.0)
    if experts:
        add_v3_experts(model, experts)
    return model


def make_v3_new_top4_contract_args(**overrides):
    values = {
        "training_version": "v3_new_top4",
        "replay_subset_ratio": None,
        "v2_joint_new_to_replay_ratio": None,
        "routing_weight_mode": None,
        "replay_subset_seed": -1,
        "seed": 2025,
        "v2_new_persistent_samples_per_task": 500,
        "v2_new_active_memory_cap": 1000,
        "replay_selection_mode": "random",
        "replay_distribution": "equal_task",
        "lora_moe_rank": 16,
        "lora_moe_alpha": 128,
        "experts_per_task": 4,
        "top_k": 4,
        "v2_memory_batch_size": 0,
        "v2_replay_forward_batch_size": 8,
        "v2_kd_memory_batch_size": 0,
        "v2_kd_loss_coeff": 1.0,
        "v2_kd_pass_multiplier": 2,
        "v2_kd_temperature": 1.0,
        "v2_kd_learning_rate": 0.0,
        "v2_kd_chunk_tokens": 256,
        "v2_kd_token_scope": "nonpad",
        "v2_joint_replay_loss_coeff": 1.0,
        "v2_max_replay_batches_per_step": 0,
        "v2_new_expert_aux_mix": 0.0,
        "v2_new_expert_aux_loss_coeff": 1.0,
        "v2_new_expert_quota_schedule": [],
    }
    values.update(overrides)
    return resolve_training_version_defaults(SimpleNamespace(**values))


def test_v3_new_top4_cli_and_exact_profile_contract():
    argv = [
        "main_Ours_LoRA_MoE.py",
        "--data_path", "/tmp/data",
        "--model_name_or_path", "/tmp/model",
        "--num_train_epochs", "1",
        "--output_dir", "/tmp/output",
        "--training_version", "v3_new_top4",
    ]
    with patch.object(sys, "argv", argv):
        parsed = parse_args()
    assert parsed.training_version == "v3_new_top4"
    assert "v3_new_top4" in V3_TRAINING_VERSIONS
    assert "v3_new_top4" in V2_NEW_TRAINING_VERSIONS
    assert "v3_new_top4" in V2_NEW_MEMORY_TRAINING_VERSIONS

    args = make_v3_new_top4_contract_args()
    validate_v2_new_args(args)
    assert args.replay_subset_ratio == 0.1
    assert args.v2_joint_new_to_replay_ratio == 5
    assert args.replay_subset_seed == 2025
    assert args.routing_weight_mode == "straight_through_topk"

    for field, invalid in {
            "experts_per_task": 1,
            "lora_moe_rank": 64,
            "lora_moe_alpha": 32,
            "top_k": 1,
            "routing_weight_mode": "full_softmax",
            "v2_kd_pass_multiplier": 1,
    }.items():
        candidate = make_v3_new_top4_contract_args()
        setattr(candidate, field, invalid)
        try:
            validate_v2_new_args(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {field} was accepted")


def test_v3_new_top4_growth_kd_prefix_and_new_only_gradients():
    torch.manual_seed(2025)
    model = TinyModel(layers=1)
    attach_shared_qkvo_lora_moe(
        model, r=16, alpha=128, top_k=4,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    add_v3_experts(model, 4)
    randomize_expert_outputs(model)
    inputs = torch.tensor(
        [0.25, -1.0, 0.75, 1.5, -0.5, 0.125]).repeat(2, 3, 1)
    layer = shared_router_layers(model)[0]
    routed_hidden = layer.input_layernorm(inputs)[0, 0]
    with torch.no_grad():
        layer.shared_expert_router.weight.copy_(
            -routed_hidden.repeat(4, 1))
    with torch.no_grad():
        before_growth = model(inputs).clone()

    old_experts = expert_state(model, 0)
    for index in range(1, 4):
        old_experts.update(expert_state(model, index))
    old_router = layer.shared_expert_router.weight.detach().clone()
    add_v3_experts(model, 4)
    assert layer.num_experts == 8
    assert layer.shared_expert_router.top_k == 4
    assert layer.mlp.r == 16
    assert all(
        projection.r == 16 and len(projection.experts) == 8
        for projection in layer.attention_expert_projections)
    assert len(layer.mlp.experts) == 8

    # KD's old-prefix teacher remains exactly the pre-growth four-expert model.
    with torch.no_grad(), limit_v3_experts(model, 4):
        prefix_after_growth = model(inputs).clone()
    torch.testing.assert_close(
        prefix_after_growth, before_growth, rtol=0, atol=0)

    new_indices = set(range(4, 8))
    freeze_v3_experts(model, trainable_expert_indices=new_indices)
    freeze_v3_routers(model, trainable=True)
    with torch.no_grad():
        # Make every token select all four new rows, exercising every new pool.
        layer.shared_expert_router.weight[4:].copy_(
            routed_hidden.repeat(4, 1))
    trainer = object.__new__(Ours_LoRA_MoE_V3)
    trainer.raw_model = model
    old_router_snapshot = trainer._snapshot_old_router_rows(model, 4)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], lr=0.1)
    output = model(inputs)
    loss = torch.nn.functional.mse_loss(output, prefix_after_growth)
    loss = loss + collect_v3_moe_losses(model)
    loss.backward()

    pools = [layer.mlp.experts]
    pools.extend(
        projection.experts for projection in layer.attention_expert_projections)
    assert all(
        parameter.grad is None
        for pool in pools for expert in pool[:4]
        for parameter in expert.parameters())
    for pool in pools:
        for expert in pool[4:]:
            b_gradients = [
                parameter.grad for name, parameter in expert.named_parameters()
                if name.endswith("B")]
            assert b_gradients and all(gradient is not None
                                       for gradient in b_gradients)
            assert any(torch.count_nonzero(gradient).item() > 0
                       for gradient in b_gradients)

    trainer._freeze_old_router_row_update(model, old_router_snapshot, 4)
    optimizer.step()
    trainer._freeze_old_router_row_update(model, old_router_snapshot, 4)
    final_state = model.state_dict()
    for name, expected in old_experts.items():
        torch.testing.assert_close(final_state[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        layer.shared_expert_router.weight[:4], old_router[:4], rtol=0, atol=0)
    with torch.no_grad(), limit_v3_experts(model, 4):
        prefix_after_update = model(inputs).clone()
    torch.testing.assert_close(
        prefix_after_update, before_growth, rtol=0, atol=0)


def test_straight_through_top1_has_unit_forward_weight_and_router_gradient():
    torch.manual_seed(123)
    router = SharedExpertRouter(
        hidden_size=6, top_k=1, aux_loss_coeff=0.0,
        z_loss_coeff=0.0,
        routing_weight_mode="straight_through_topk")
    router.add_experts(4)
    inputs = torch.randn(2, 5, 6, requires_grad=True)
    context = router(inputs)
    torch.testing.assert_close(
        context.expert_weights,
        torch.ones_like(context.expert_weights), rtol=0, atol=0)
    context.expert_weights.sum().backward()
    assert router.weight.grad is not None
    assert router.weight.grad.norm().item() > 0


def test_base_llama31_gets_versioned_chat_template_fallback():
    class BaseTokenizer:
        chat_template = None

        @staticmethod
        def convert_tokens_to_ids(token):
            return {
                "<|begin_of_text|>": 1,
                "<|start_header_id|>": 2,
                "<|end_header_id|>": 3,
                "<|eot_id|>": 4,
            }.get(token)

    tokenizer = BaseTokenizer()
    source = ensure_llama31_chat_template(
        tokenizer, "/models/Llama-3.1-8B")
    assert source == LLAMA31_STANDARD_TEMPLATE_VERSION
    assert tokenizer.chat_template == LLAMA31_STANDARD_CHAT_TEMPLATE


def randomize_expert_outputs(model, indices=None):
    with torch.no_grad():
        for layer in shared_router_layers(model):
            for projection in layer.attention_expert_projections:
                experts = projection.experts
                selected = range(len(experts)) if indices is None else indices
                for index in selected:
                    expert = experts[index]
                    expert.B.normal_(mean=0.0, std=0.1)
            experts = layer.mlp.experts
            selected = range(len(experts)) if indices is None else indices
            for index in selected:
                expert = experts[index]
                for pair in expert.values():
                    pair.B.normal_(mean=0.0, std=0.1)


def test_new_experts_are_exact_noop():
    torch.manual_seed(3)
    original = TinyModel(layers=2)
    reference = copy.deepcopy(original)
    inputs = torch.randn(2, 4, 6)
    expected = reference(inputs).detach()
    attach_shared_qkvo_lora_moe(
        original, r=2, alpha=4, top_k=1,
        aux_loss_coeff=0.0, z_loss_coeff=0.0)
    add_v3_experts(original, 1)
    actual = original(inputs).detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v3_new_top4_grows_qkvo_ffn_and_records_exact_contract():
    model = TinyModel(layers=1)
    attach_shared_qkvo_lora_moe(
        model, r=16, alpha=128, top_k=4,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.05)
    add_v3_experts(model, 4)
    layer = shared_router_layers(model)[0]
    assert layer.num_experts == 4
    assert layer.shared_expert_router.top_k == 4
    assert layer.mlp.r == 16
    assert layer.mlp.alpha == 128
    assert all(len(projection.experts) == 4
               for projection in layer.attention_expert_projections)
    assert "v3_new_top4" in V3_NEW_TRAINING_VERSIONS
    assert issubclass(Ours_LoRA_MoE_V3_New, Ours_LoRA_MoE_V3)

    args = SimpleNamespace(
        training_version="v3_new_top4", experts_per_task=4,
        train_format="slora_chat_full", chat_template_source="test",
        max_train_len=1024, max_prompt_len=1024, max_ans_len=512,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        replay_subset_ratio=0.1, router_replay_exposure_samples=1000,
        replay_distribution="equal_task", replay_subset_seed=2025,
        replay_selection_mode="random",
        v2_memory_batch_size=0, v2_kd_memory_batch_size=4,
        v2_replay_forward_batch_size=8, v2_kd_loss_coeff=1.0,
        v2_kd_temperature=1.0, v2_kd_learning_rate=0.0,
        v2_kd_chunk_tokens=256, v2_kd_token_scope="nonpad",
        v2_joint_replay_loss_coeff=1.0,
        v2_joint_new_to_replay_ratio=5,
        v2_max_replay_batches_per_step=0,
        v2_new_active_memory_cap=1000,
        v2_new_persistent_samples_per_task=500,
        v2_kd_pass_multiplier=2,
    )
    persisted = {
        "C-STANCE": {
            "resolved_seed": 2025,
            "indices_sha256": "1" * 64,
        },
    }
    trainer = SimpleNamespace(
        _v2_new_persistent_memory_records=persisted,
        _v2_new_sampler_pass_digests={},
        _fixed_subset_seed=lambda: 2025,
    )
    with tempfile.TemporaryDirectory() as output_dir:
        save_v3_meta(model, output_dir, args, trainer=trainer)
        metadata = json.loads((
            Path(output_dir) / "lora_moe_meta.json").read_text())
    assert metadata["training_version"] == "v3_new_top4"
    assert metadata["architecture"] == "shared_router_qkvo_ffn"
    assert metadata["experts_per_task"] == 4
    assert metadata["r"] == metadata["attention_rank"] == 16
    assert metadata["alpha"] == 128
    assert metadata["top_k"] == 4
    assert metadata["num_experts"] == 4
    assert metadata["v2"]["kd_pass_multiplier"] == 2
    assert metadata["v2_new"]["kd_stream_pass_multiplier"] == 2
    assert metadata["replay_memory"]["configured_subset_seed"] == 2025
    assert metadata["replay_memory"]["resolved_base_subset_seed"] == 2025
    expected_identities = {
        "C-STANCE": {
            "resolved_seed": 2025,
            "indices_sha256": "1" * 64,
        },
    }
    assert metadata["replay_memory"]["persisted_identities"] == (
        expected_identities)
    assert metadata["v2_new"]["persisted_identities"] == expected_identities
    assert validate_v2_new_resume_persisted_identities(
        metadata, completed_round=0,
        task_names=["C-STANCE", "FOMC"]) == expected_identities


def test_router_runs_once_and_context_is_shared_by_all_projections():
    torch.manual_seed(5)
    model = make_v3(layers=1, experts=2, top_k=2)
    layer = shared_router_layers(model)[0]
    router_calls = []
    context_ids = []

    router_handle = layer.shared_expert_router.register_forward_hook(
        lambda module, inputs, output: router_calls.append(id(output)))

    def capture_context(module, inputs):
        assert module._routing_context is not None
        context_ids.append(id(module._routing_context))

    handles = [
        module.register_forward_pre_hook(capture_context)
        for module in (*layer.attention_expert_projections, layer.mlp)
    ]
    try:
        model(torch.randn(2, 3, 6))
    finally:
        router_handle.remove()
        for handle in handles:
            handle.remove()
    assert len(router_calls) == 1, router_calls
    assert len(context_ids) == 5, context_ids
    assert set(context_ids) == set(router_calls), (context_ids, router_calls)


def test_routing_context_caches_each_expert_route_and_respects_valid_mask():
    indices = torch.tensor([
        [0, 2], [1, 0], [2, 1], [0, 1], [1, 2], [2, 0]])
    valid_mask = torch.tensor([True, False, True, True, False, True])
    context = RoutingContext(
        expert_indices=indices,
        expert_weights=torch.ones_like(indices, dtype=torch.float32),
        valid_token_mask=valid_mask,
        num_experts=3)

    original_where = torch.where
    with patch.object(torch, "where", wraps=original_where) as counted_where:
        first_routes = [context.routes_for(index) for index in range(3)]
        for _ in range(4):
            repeated_routes = [
                context.routes_for(index) for index in range(3)]

    # One lookup per expert is enough for all Q/K/V/O/FFN consumers.
    assert counted_where.call_count == context.num_experts
    for first, repeated in zip(first_routes, repeated_routes):
        assert repeated[0] is first[0]
        assert repeated[1] is first[1]
        assert valid_mask.index_select(0, repeated[1]).all()

    transposed = indices.transpose(0, 1)
    for expert_index, actual in enumerate(first_routes):
        slot, token_index = original_where(transposed == expert_index)
        keep = valid_mask[token_index]
        expected = (slot[keep], token_index[keep])
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_cached_routes_preserve_v3_output_and_gradients_bit_exact():
    torch.manual_seed(51)
    cached_model = make_v3(layers=1, experts=3, top_k=2)
    randomize_expert_outputs(cached_model)
    reference_model = copy.deepcopy(cached_model)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    set_v3_router_token_mask(cached_model, mask)
    set_v3_router_token_mask(reference_model, mask)
    cached_input = torch.randn(2, 4, 6, requires_grad=True)
    reference_input = cached_input.detach().clone().requires_grad_(True)

    def uncached_routes_for(context, expert_index):
        slot, token_index = torch.where(
            context.expert_indices.transpose(0, 1) == expert_index)
        if (context.valid_token_mask is not None
                and token_index.numel() > 0):
            keep = context.valid_token_mask[token_index]
            slot, token_index = slot[keep], token_index[keep]
        return slot, token_index

    with patch.object(RoutingContext, "routes_for", uncached_routes_for):
        reference_output = reference_model(reference_input)
        reference_loss = reference_output.square().sum()
        reference_loss.backward()

    cached_output = cached_model(cached_input)
    cached_loss = cached_output.square().sum()
    cached_loss.backward()

    torch.testing.assert_close(
        cached_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(
        cached_input.grad, reference_input.grad, rtol=0, atol=0)
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in cached_model.named_parameters():
        reference_gradient = reference_parameters[name].grad
        assert (parameter.grad is None) == (reference_gradient is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, reference_gradient, rtol=0, atol=0)


def test_qkvo_and_ffn_have_equal_rank_and_expert_count():
    model = make_v3(layers=2, experts=3, top_k=2)
    for layer in shared_router_layers(model):
        assert layer.num_experts == 3
        assert layer.mlp.r == 2
        for target in ATTENTION_TARGETS:
            projection = getattr(layer.self_attn, f"{target}_proj")
            assert projection.r == layer.mlp.r
            assert len(projection.experts) == len(layer.mlp.experts) == 3


def test_task_growth_keeps_old_qkvo_ffn_and_router_prefix_bit_exact():
    torch.manual_seed(6)
    model = make_v3(layers=2, experts=1, top_k=1)
    randomize_expert_outputs(model)
    old_experts = expert_state(model, 0)
    old_router_rows = [
        layer.shared_expert_router.weight.detach().clone()
        for layer in shared_router_layers(model)]

    add_v3_experts(model, 2)

    final_state = model.state_dict()
    for name, expected in old_experts.items():
        torch.testing.assert_close(final_state[name], expected, rtol=0, atol=0)
    for layer, expected in zip(shared_router_layers(model), old_router_rows):
        assert layer.num_experts == 3
        torch.testing.assert_close(
            layer.shared_expert_router.weight[:1], expected, rtol=0, atol=0)
        pools = [layer.mlp.experts]
        pools.extend(
            projection.experts
            for projection in layer.attention_expert_projections)
        assert all(len(pool) == 3 for pool in pools)
        for pool in pools:
            for expert in pool[1:]:
                b_parameters = [
                    parameter for name, parameter in expert.named_parameters()
                    if name.endswith("B")]
                assert b_parameters
                assert all(torch.count_nonzero(value) == 0
                           for value in b_parameters)


def test_sparse_qkvo_and_ffn_match_dense_shared_routing_reference():
    torch.manual_seed(6)
    model = make_v3(layers=1, experts=3, top_k=2)
    randomize_expert_outputs(model)
    layer = shared_router_layers(model)[0]
    inputs = torch.randn(2, 4, 6)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    layer.shared_expert_router._router_token_mask = mask
    context = layer.shared_expert_router(inputs)
    flat_mask = mask.reshape(-1)
    combined = inputs.new_zeros(
        (inputs.numel() // inputs.shape[-1], context.num_experts))
    combined.scatter_(
        1, context.expert_indices,
        context.expert_weights.to(inputs.dtype))
    combined[~flat_mask] = 0
    combined = combined.reshape(*inputs.shape[:-1], context.num_experts)

    for projection in layer.attention_expert_projections:
        projection.set_routing_context(context)
        actual = projection(inputs)
        projection.set_routing_context(None)
        expected = projection.base_layer(inputs)
        for index, expert in enumerate(projection.experts):
            expected = expected + combined[..., index, None] * expert(inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    layer.mlp.set_routing_context(context)
    actual = layer.mlp(inputs)
    layer.mlp.set_routing_context(None)
    gate = layer.mlp.base_mlp.gate_proj(inputs)
    up = layer.mlp.base_mlp.up_proj(inputs)
    for index, expert in enumerate(layer.mlp.experts):
        weight = combined[..., index, None]
        gate = gate + weight * expert["gate"](inputs)
        up = up + weight * expert["up"](inputs)
    intermediate = torch.nn.functional.silu(gate) * up
    expected = layer.mlp.base_mlp.down_proj(intermediate)
    for index, expert in enumerate(layer.mlp.experts):
        expected = expected + combined[..., index, None] * expert["down"](
            intermediate)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    layer.shared_expert_router._router_token_mask = None


def test_pre_expansion_teacher_prefix_is_exact():
    torch.manual_seed(7)
    model = make_v3(layers=1, experts=1, top_k=1)
    randomize_expert_outputs(model)
    inputs = torch.randn(2, 3, 6)
    before = model(inputs).detach()
    add_v3_experts(model, 1)
    randomize_expert_outputs(model, indices=[1])
    with limit_v3_experts(model, 1):
        recovered = model(inputs).detach()
    torch.testing.assert_close(recovered, before, rtol=0, atol=0)


def test_only_new_attention_and_ffn_experts_receive_gradients():
    torch.manual_seed(11)
    model = make_v3(layers=1, experts=2, top_k=2)
    randomize_expert_outputs(model)
    freeze_v3_experts(model, trainable_expert_indices={1})
    freeze_v3_routers(model, trainable=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    set_v3_router_token_mask(model, mask)
    try:
        output = model(torch.randn(2, 4, 6))
        router_loss = collect_v3_moe_losses(model)
        (output.square().mean() + router_loss).backward()
    finally:
        set_v3_router_token_mask(model, None)

    layer = shared_router_layers(model)[0]
    old_experts = [layer.mlp.experts[0]] + [
        projection.experts[0]
        for projection in layer.attention_expert_projections]
    new_experts = [layer.mlp.experts[1]] + [
        projection.experts[1]
        for projection in layer.attention_expert_projections]
    assert all(
        parameter.grad is None
        for expert in old_experts for parameter in expert.parameters())
    assert all(
        parameter.grad is not None
        for expert in new_experts for parameter in expert.parameters())
    assert layer.shared_expert_router.weight.grad is not None


def test_v3_kd_init_runs_old_prefix_teacher_and_updates_only_new_side():
    torch.manual_seed(12)
    model = make_causal_v3(experts=1, top_k=2)
    randomize_expert_outputs(model)
    with torch.no_grad():
        shared_router_layers(model)[0].shared_expert_router.weight.normal_(
            mean=0.0, std=0.2)
    add_v3_experts(model, 1)
    layer = shared_router_layers(model)[0]
    old_before = expert_state(model, 0)
    new_before = expert_state(model, 1)
    old_router_before = layer.shared_expert_router.weight[:1].detach().clone()
    new_router_before = layer.shared_expert_router.weight[1:].detach().clone()

    loader = TinyBatchLoader([
        make_batch([[1, 2, 3], [4, 5, 0]], "memory-a"),
        make_batch([[6, 7, 8], [9, 10, 11]], "memory-b"),
        make_batch([[12, 13, 0], [14, 15, 16]], "memory-c"),
    ], batch_size=2)
    trainer = object.__new__(Ours_LoRA_MoE_V3)
    trainer.raw_model = model
    trainer.model = model
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._active_task_workload = None
    trainer.args = SimpleNamespace(
        global_rank=1, effective_global_batch=4,
        gradient_accumulation_steps=1, learning_rate=0.02,
        v2_kd_learning_rate=0.05, v2_kd_loss_coeff=1.0,
        v2_kd_temperature=2.0, v2_kd_chunk_tokens=4,
        v2_kd_token_scope="nonpad", loss_log_interval=1)
    grad_ckpt_calls = []
    trainer._set_grad_ckpt = grad_ckpt_calls.append
    engine_calls = install_counting_engine(trainer)
    active_prefixes = []
    handle = layer.shared_expert_router.register_forward_hook(
        lambda module, inputs, output: active_prefixes.append(
            output.num_experts))
    try:
        trainer._run_v2_kd_init(
            loader, old_expert_count=1, new_indices={1},
            device=torch.device("cpu"), task="task-1")
    finally:
        handle.remove()

    assert active_prefixes == [1, 2, 1, 2, 1, 2], active_prefixes
    assert model.forward_calls == 2 * len(loader)
    assert grad_ckpt_calls == [True]
    assert trainer.args.gradient_accumulation_steps == 2
    assert engine_calls == [(2, 0.05)]
    assert trainer.optimizer.step_calls == 2
    assert trainer.lr_scheduler.step_calls == 2
    final_state = model.state_dict()
    for name, expected in old_before.items():
        torch.testing.assert_close(final_state[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        layer.shared_expert_router.weight[:1], old_router_before,
        rtol=0, atol=0)
    assert any(
        name.endswith("B") and not torch.equal(final_state[name], expected)
        for name, expected in new_before.items())
    assert not torch.equal(
        layer.shared_expert_router.weight[1:], new_router_before)


def test_v3_st_top1_kd_learns_from_zero_initialized_new_expert():
    """The corrected top-1 path must not make KD a permanent zero-loss no-op."""
    torch.manual_seed(120)
    model = make_causal_v3(
        experts=1, top_k=1,
        routing_weight_mode="straight_through_topk")
    randomize_expert_outputs(model)
    add_v3_experts(model, 1)
    layer = shared_router_layers(model)[0]

    # Route a repeated token to the newly added expert. Its B matrices still
    # start at exactly zero, so the first KD update has to bootstrap B before
    # the straight-through router surrogate can receive a useful signal.
    ids = torch.tensor([[1, 1, 1], [1, 1, 1]])
    with torch.no_grad():
        routed_hidden = layer.input_layernorm(model.embedding(ids))[0, 0]
        layer.shared_expert_router.weight[0].zero_()
        layer.shared_expert_router.weight[1].copy_(routed_hidden)
    new_before = expert_state(model, 1)
    new_router_before = (
        layer.shared_expert_router.weight[1:].detach().clone())

    loader = TinyBatchLoader([
        make_batch([[1, 1, 1], [1, 1, 1]], "memory-a"),
        make_batch([[1, 1, 1], [1, 1, 1]], "memory-b"),
        make_batch([[1, 1, 1], [1, 1, 1]], "memory-c"),
    ], batch_size=2)
    trainer = object.__new__(Ours_LoRA_MoE_V3)
    trainer.raw_model = model
    trainer.model = model
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._active_task_workload = None
    trainer.args = SimpleNamespace(
        global_rank=1, effective_global_batch=2,
        gradient_accumulation_steps=1, learning_rate=0.02,
        v2_kd_learning_rate=0.05, v2_kd_loss_coeff=1.0,
        v2_kd_temperature=1.0, v2_kd_chunk_tokens=4,
        v2_kd_token_scope="nonpad", loss_log_interval=1)
    trainer._set_grad_ckpt = lambda enabled: None
    install_counting_engine(trainer)
    trainer._run_v2_kd_init(
        loader, old_expert_count=1, new_indices={1},
        device=torch.device("cpu"), task="task-st-top1")

    final_state = model.state_dict()
    assert any(
        name.endswith("B") and not torch.equal(final_state[name], expected)
        for name, expected in new_before.items())
    assert not torch.equal(
        layer.shared_expert_router.weight[1:], new_router_before)


def test_v3_primary_phase_updates_new_qkvo_ffn_and_router_only():
    torch.manual_seed(13)
    model = make_causal_v3(experts=1, top_k=2)
    randomize_expert_outputs(model)
    add_v3_experts(model, 1)
    old_before = expert_state(model, 0)
    new_before = expert_state(model, 1)
    layer = shared_router_layers(model)[0]
    router_before = layer.shared_expert_router.weight.detach().clone()
    freeze_v3_experts(model, trainable_expert_indices={1})
    freeze_v3_routers(model, trainable=True)

    loader = TinyBatchLoader([
        make_batch([[1, 2, 3], [4, 5, 0]], "primary-a"),
        make_batch([[6, 7, 8], [9, 10, 11]], "primary-b"),
        make_batch([[12, 13, 0], [14, 15, 16]], "primary-c"),
    ], batch_size=2)
    trainer = object.__new__(Ours_LoRA_MoE_V3)
    trainer.raw_model = model
    trainer.model = model
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._active_task_workload = None
    trainer.args = SimpleNamespace(
        global_rank=1, effective_global_batch=4,
        gradient_accumulation_steps=1, learning_rate=0.03,
        loss_log_interval=1)
    trainer._set_phase_gradient_accumulation(
        loader.batch_size, "test v3 primary")
    engine_calls = install_counting_engine(trainer)
    trainer._reinit_engine(trainer._optimizer_update_count(loader, 1))
    router_calls = []
    handle = layer.shared_expert_router.register_forward_hook(
        lambda module, inputs, output: router_calls.append(id(output)))
    try:
        trainer._run_v3_primary_epochs(
            loader, epochs=1, device=torch.device("cpu"),
            phase_name="test-v3-primary")
    finally:
        handle.remove()

    assert len(router_calls) == len(loader)
    assert model.forward_calls == len(loader)
    assert trainer.args.gradient_accumulation_steps == 2
    assert engine_calls == [(2, 0.03)]
    assert trainer.optimizer.step_calls == 2
    assert trainer.lr_scheduler.step_calls == 2
    final_state = model.state_dict()
    for name, expected in old_before.items():
        torch.testing.assert_close(final_state[name], expected, rtol=0, atol=0)
    assert any(
        name.endswith("B") and not torch.equal(final_state[name], expected)
        for name, expected in new_before.items())
    assert not torch.equal(layer.shared_expert_router.weight, router_before)


def test_v3_epoch_probe_is_one_fixed_mixed_forward_and_writes_jsonl():
    torch.manual_seed(13)
    model = make_causal_v3(experts=2, top_k=1)
    old_batch = make_batch([[1, 2, 3], [4, 5, 6]], "old")
    new_batch = make_batch([[7, 8, 9], [10, 11, 12]], "new")
    old_batch["labels"][:, :2] = -100
    new_batch["labels"][:, :2] = -100
    with tempfile.TemporaryDirectory() as output_dir:
        trainer = object.__new__(Ours_LoRA_MoE_V3)
        trainer.raw_model = model
        trainer.model = model
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        trainer.train_task_list = {"old": None, "new": None}
        trainer.eval_task_list = {
            "old": TinyBatchLoader([old_batch], batch_size=2),
            "new": TinyBatchLoader([new_batch], batch_size=2),
        }
        trainer.args = SimpleNamespace(
            global_rank=0, v3_epoch_probe_samples=4,
            experts_per_task=1, output_dir=output_dir)
        before = model.forward_calls
        trainer._run_epoch_probe(
            "new", i_task=1, epoch=0, device=torch.device("cpu"))
        assert model.forward_calls == before + 1
        lines = (Path(output_dir) / "epoch_probe.jsonl").read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["global_samples"] == 4
        assert record["current_samples"] == 2
        assert record["past_samples"] == 2
        assert record["current_answer_ce"] >= 0
        assert record["past_answer_ce"] >= 0
        assert 0 <= record["current_new_expert_route_fraction"] <= 1
        assert 0 <= record["past_new_expert_route_fraction"] <= 1


def test_v3_joint_phase_keeps_replay_router_only():
    class CountingSGD(torch.optim.SGD):
        def __init__(self, parameters):
            super().__init__(parameters, lr=0.02)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    class Scheduler:
        def __init__(self):
            self.step_calls = 0

        def step(self):
            self.step_calls += 1

    torch.manual_seed(12)
    model = TinyCausalModel()
    attach_shared_qkvo_lora_moe(
        model, r=2, alpha=4, top_k=2,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="full_softmax", dropout=0.0)
    add_v3_experts(model, 2)
    randomize_expert_outputs(model)
    freeze_v3_experts(model, trainable_expert_indices={1})
    freeze_v3_routers(model, trainable=True)
    layer = shared_router_layers(model)[0]
    old_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if ".experts.0." in name}
    new_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if ".experts.1." in name}
    router_before = layer.shared_expert_router.weight.detach().clone()

    optimizer = CountingSGD([
        parameter for parameter in model.parameters()
        if parameter.requires_grad])
    scheduler = Scheduler()
    trainer = object.__new__(Ours_LoRA_MoE_V3)
    trainer.raw_model = model
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._active_task_workload = None
    trainer.args = SimpleNamespace(
        global_rank=1, gradient_accumulation_steps=1,
        v2_max_replay_batches_per_step=0,
        v2_replay_forward_batch_size=8,
        v2_joint_new_to_replay_ratio=2,
        v2_joint_replay_loss_coeff=1.0, loss_log_interval=1)

    def batch(values, source):
        ids = torch.tensor(values)
        return {
            "input_ids": ids,
            "attention_mask": ids.ne(0).long(),
            "labels": ids.clone(),
            "sources": [source] * ids.shape[0],
        }

    primary_loader = TinyBatchLoader([
        batch([
            [1, 2, 3], [4, 5, 6], [2, 4, 6], [3, 5, 7]],
            "primary-a"),
        batch([
            [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
            "primary-b"),
    ], batch_size=4)
    memory_loader = TinyBatchLoader([
        batch([[13, 14, 15]], "replay-a"),
        batch([[16, 17, 18, 19, 20]], "replay-b"),
    ], batch_size=1)
    trainer._run_v2_joint_epochs(
        primary_loader, memory_loader, epochs=1,
        device=torch.device("cpu"), phase_name="test-v3")

    # Both optimizer updates pack two variable-length replay records into one
    # router-only forward/backward while preserving four total exposures.
    assert model.forward_calls == 4
    assert optimizer.step_calls == 2
    assert scheduler.step_calls == 2
    final_state = model.state_dict()
    for name, expected in old_before.items():
        torch.testing.assert_close(final_state[name], expected, rtol=0, atol=0)
    assert any(not torch.equal(final_state[name], expected)
               for name, expected in new_before.items())
    assert not torch.equal(layer.shared_expert_router.weight, router_before)


def test_kd_router_prefix_restore_keeps_old_rows_bit_exact():
    torch.manual_seed(13)
    model = make_v3(layers=1, experts=2, top_k=2)
    layer = shared_router_layers(model)[0]
    trainer = object.__new__(Ours_LoRA_MoE_V3)
    snapshot = trainer._snapshot_old_router_rows(model, 1)
    optimizer = torch.optim.AdamW(
        layer.shared_expert_router.router.parameters(), lr=0.1,
        weight_decay=0.1)
    loss = layer.shared_expert_router.weight.sum()
    loss.backward()
    trainer._freeze_old_router_row_update(model, snapshot, 1)
    optimizer.step()
    trainer._freeze_old_router_row_update(model, snapshot, 1)
    torch.testing.assert_close(
        layer.shared_expert_router.weight[:1], snapshot[0], rtol=0, atol=0)
    assert not torch.equal(
        layer.shared_expert_router.weight[1:], snapshot[0])


def test_partial_state_dict_round_trip():
    torch.manual_seed(13)
    source = make_v3(layers=1, experts=2, top_k=2)
    randomize_expert_outputs(source)
    target = make_v3(layers=1, experts=2, top_k=2)
    fragments = (
        ".shared_expert_router.router.",
        ".self_attn.q_proj.experts.",
        ".self_attn.k_proj.experts.",
        ".self_attn.v_proj.experts.",
        ".self_attn.o_proj.experts.",
        ".mlp.experts.",
    )
    state = {
        key: value for key, value in source.state_dict().items()
        if any(fragment in key for fragment in fragments)}
    missing, unexpected = target.load_state_dict(state, strict=False)
    assert not unexpected
    assert not [key for key in missing
                if any(fragment in key for fragment in fragments)]
    for key, expected in state.items():
        torch.testing.assert_close(target.state_dict()[key], expected)


def test_v3_checkpoint_loader_round_trip_with_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    class TinyTokenizer:
        eos_token_id = 2

        def __len__(self):
            return 32

    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=2, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0)
    with tempfile.TemporaryDirectory() as temporary:
        base_dir = Path(temporary) / "base"
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        source = LlamaForCausalLM(config).eval()
        source.save_pretrained(base_dir)
        attach_shared_qkvo_lora_moe(
            source, r=2, alpha=4, top_k=2,
            aux_loss_coeff=0.01, z_loss_coeff=0.001,
            routing_weight_mode="full_softmax", dropout=0.0)
        add_v3_experts(source, 2)
        randomize_expert_outputs(source)
        fragments = Ours_LoRA_MoE_V3.save_key_substrings
        partial = {
            key: value for key, value in source.state_dict().items()
            if any(fragment in key for fragment in fragments)}
        torch.save(partial, checkpoint_dir / "pytorch_model.bin")
        args = SimpleNamespace(
            experts_per_task=1, train_format="raw_answer",
            max_train_len=16, max_prompt_len=12, max_ans_len=4,
            adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
            replay_subset_ratio=0.01,
            router_replay_exposure_samples=10,
            replay_distribution="equal_task", replay_subset_seed=17,
            v2_memory_batch_size=1, v2_kd_loss_coeff=1.0,
            v2_replay_forward_batch_size=8,
            v2_kd_temperature=1.0, v2_kd_learning_rate=0.0,
            v2_kd_chunk_tokens=8, v2_kd_token_scope="nonpad",
            v2_joint_replay_loss_coeff=1.0,
            v2_max_replay_batches_per_step=0)
        save_v3_meta(source, checkpoint_dir, args)
        loaded, meta = load_v3_checkpoint(
            checkpoint_dir, TinyTokenizer(), str(base_dir),
            device="cpu", dtype=torch.float32)
        assert meta["architecture"] == "shared_router_qkvo_ffn"
        assert meta["attention_rank"] == meta["r"] == 2
        loaded_state = loaded.state_dict()
        for key, expected in partial.items():
            torch.testing.assert_close(loaded_state[key], expected)

        ids = torch.tensor([[1, 3, 5, 7]])
        mask = torch.ones_like(ids)
        source.eval()
        set_v3_router_token_mask(source, mask)
        set_v3_router_token_mask(loaded, mask)
        try:
            with torch.no_grad():
                expected = source(
                    input_ids=ids, attention_mask=mask,
                    use_cache=False).logits
                actual = loaded(
                    input_ids=ids, attention_mask=mask,
                    use_cache=False).logits
        finally:
            set_v3_router_token_mask(source, None)
            set_v3_router_token_mask(loaded, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v3_new_top4_checkpoint_round_trip_records_exact_profile():
    from transformers import LlamaConfig, LlamaForCausalLM

    class TinyTokenizer:
        eos_token_id = 2

        def __len__(self):
            return 32

    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=2, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0)
    with tempfile.TemporaryDirectory() as temporary:
        base_dir = Path(temporary) / "base"
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        source = LlamaForCausalLM(config).eval()
        source.save_pretrained(base_dir)
        attach_shared_qkvo_lora_moe(
            source, r=16, alpha=128, top_k=4,
            aux_loss_coeff=0.01, z_loss_coeff=0.001,
            routing_weight_mode="straight_through_topk", dropout=0.0)
        add_v3_experts(source, 8)
        randomize_expert_outputs(source)
        partial = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
            if any(fragment in key
                   for fragment in Ours_LoRA_MoE_V3.save_key_substrings)
        }
        torch.save(partial, checkpoint_dir / "pytorch_model.bin")
        args = SimpleNamespace(
            training_version="v3_new_top4", experts_per_task=4,
            train_format="slora_chat_full",
            chat_template_source="llama31_standard_v1",
            max_train_len=16, max_prompt_len=12, max_ans_len=4,
            adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
            replay_subset_ratio=0.1, replay_selection_mode="random",
            router_replay_exposure_samples=1000,
            replay_distribution="equal_task", replay_subset_seed=2025,
            v2_memory_batch_size=0, v2_kd_memory_batch_size=8,
            v2_replay_forward_batch_size=8,
            v2_kd_loss_coeff=1.0, v2_kd_temperature=1.0,
            v2_kd_learning_rate=0.0, v2_kd_chunk_tokens=256,
            v2_kd_token_scope="nonpad",
            v2_joint_replay_loss_coeff=1.0,
            v2_joint_new_to_replay_ratio=5,
            v2_max_replay_batches_per_step=0,
            v2_new_active_memory_cap=1000,
            v2_new_persistent_samples_per_task=500,
            v2_kd_pass_multiplier=2,
        )
        identity = {
            "resolved_seed": 2025,
            "indices_sha256": "a" * 64,
        }
        trainer = SimpleNamespace(
            _v2_new_persistent_memory_records={"task0": identity},
            _v2_new_sampler_pass_digests={"task0": {"kd:0": "b" * 64}},
            _fixed_subset_seed=lambda: 2025,
        )
        save_v3_meta(source, checkpoint_dir, args, trainer=trainer)

        loaded, meta = load_v3_checkpoint(
            checkpoint_dir, TinyTokenizer(), str(base_dir),
            device="cpu", dtype=torch.float32)
        assert meta["training_version"] == "v3_new_top4"
        assert meta["architecture"] == "shared_router_qkvo_ffn"
        assert meta["attention_targets"] == ["q", "k", "v", "o"]
        assert meta["r"] == meta["attention_rank"] == 16
        assert meta["alpha"] == 128
        assert meta["experts_per_task"] == 4
        assert meta["top_k"] == 4
        assert meta["routing_weight_mode"] == "straight_through_topk"
        assert meta["num_experts"] == 8
        assert meta["v2"]["kd_pass_multiplier"] == 2
        assert meta["v2_new"]["persisted_identities"] == {"task0": identity}
        for key, expected in partial.items():
            torch.testing.assert_close(
                loaded.state_dict()[key], expected, rtol=0, atol=0)

        ids = torch.tensor([[1, 3, 5, 7]])
        mask = torch.ones_like(ids)
        source.eval()
        set_v3_router_token_mask(source, mask)
        set_v3_router_token_mask(loaded, mask)
        try:
            with torch.no_grad():
                expected = source(
                    input_ids=ids, attention_mask=mask,
                    use_cache=False).logits
                actual = loaded(
                    input_ids=ids, attention_mask=mask,
                    use_cache=False).logits
        finally:
            set_v3_router_token_mask(source, None)
            set_v3_router_token_mask(loaded, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_transformers_backbone(model_class, config):
    torch.manual_seed(17)
    model = model_class(config).eval()
    input_ids = torch.tensor([[1, 5, 7, 9], [1, 4, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    with torch.no_grad():
        expected = model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False).logits
    attach_shared_qkvo_lora_moe(
        model, r=2, alpha=4, top_k=1,
        aux_loss_coeff=0.0, z_loss_coeff=0.0,
        routing_weight_mode="full_softmax", dropout=0.0)
    add_v3_experts(model, 1)
    model.eval()
    set_v3_router_token_mask(model, attention_mask)
    try:
        with torch.no_grad():
            actual = model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False).logits
    finally:
        set_v3_router_token_mask(model, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Generation exercises single-token decode with a growing KV cache. The
    # same pre-attention routing decision must remain valid for QKVO and FFN.
    generated = model.generate(
        input_ids=input_ids[:1], attention_mask=attention_mask[:1],
        max_new_tokens=2, do_sample=False,
        pad_token_id=config.pad_token_id)
    assert generated.shape == (1, input_ids.shape[1] + 2)


def test_tiny_llama_gqa_forward_and_kv_cache():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0)
    config._attn_implementation = "eager"
    _assert_transformers_backbone(LlamaForCausalLM, config)


def test_tiny_qwen2_gqa_forward_and_kv_cache():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    _assert_transformers_backbone(Qwen2ForCausalLM, config)


def test_per_sample_router_gradient_hook_matches_individual_backward():
    torch.manual_seed(41)
    router = SharedExpertRouter(
        hidden_size=4, top_k=1, aux_loss_coeff=0.0,
        z_loss_coeff=0.0, routing_weight_mode="straight_through_topk")
    router.add_experts(3)
    inputs = torch.randn(2, 5, 4)
    router._capture_sample_grad = True
    context = router(inputs)
    context.expert_weights.sum().backward()
    captured_squared = torch.stack(router._sample_grad_scores).sum(dim=0)

    expected = []
    weight = router.weight.detach().clone().requires_grad_(True)
    for sample in inputs:
        logits = torch.nn.functional.linear(sample, weight)
        indices = logits.topk(1, dim=-1).indices
        selected = torch.softmax(logits, dim=-1).gather(-1, indices)
        gradient = torch.autograd.grad(selected.sum(), weight)[0]
        expected.append(gradient.square().sum())
    torch.testing.assert_close(
        captured_squared, torch.stack(expected), rtol=1e-5, atol=1e-6)


def test_router_gradient_memory_keeps_highest_mean_scores():
    with tempfile.TemporaryDirectory() as output_dir:
        trainer = Ours_LoRA_MoE_V3.__new__(Ours_LoRA_MoE_V3)
        trainer.args = SimpleNamespace(
            replay_selection_mode="router_gradient",
            replay_subset_ratio=0.2,
            global_rank=0,
            output_dir=output_dir,
        )
        dataset = TensorDataset(torch.arange(10))
        trainer.train_task_list = {"task": DataLoader(dataset)}
        trainer._fixed_task_subsets = {}
        trainer._fixed_task_subset_indices = {}
        trainer._router_gradient_memory_stats = {"task": {
            index: {"sum": float(index * 2), "count": 2, "max": float(index)}
            for index in range(10)
        }}
        trainer._finalize_gradient_replay_memory("task")
        assert trainer._fixed_task_subset_indices["task"] == [9, 8]
        metadata = json.loads((
            Path(output_dir) / "fixed_replay_memory" /
            "task_0_task.json").read_text())
        assert metadata["selection_mode"] == "router_gradient"
        assert metadata["indices"] == [9, 8]


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
