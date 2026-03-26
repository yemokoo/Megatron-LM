"""Verify that the selected-expert QVLoraExpertRouter produces identical results
to the original sequential loop implementation, for both top-1 and top-k routing,
and that backward passes compute correct gradients including freeze hooks."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Reference: original sequential loop implementation (copied verbatim)
# ---------------------------------------------------------------------------

def _ref_top1(hidden_states, expert_idx, expert_scores, q_lora_a, q_lora_b, v_lora_a, v_lora_b, scale, query_out, value_out, num_experts):
    num_tokens = hidden_states.shape[0]
    q_delta = hidden_states.new_zeros((num_tokens, query_out))
    v_delta = hidden_states.new_zeros((num_tokens, value_out))
    for expert_id in range(num_experts):
        token_indices = torch.nonzero(expert_idx == expert_id, as_tuple=False).flatten()
        if token_indices.numel() == 0:
            continue
        expert_hidden = hidden_states.index_select(0, token_indices)
        expert_scale = (
            expert_scores.index_select(0, token_indices).to(hidden_states.dtype).unsqueeze(-1) * scale
        )
        q_low_rank = expert_hidden @ q_lora_a[expert_id]
        v_low_rank = expert_hidden @ v_lora_a[expert_id]
        q_delta.index_copy_(0, token_indices, (q_low_rank @ q_lora_b[expert_id]) * expert_scale)
        v_delta.index_copy_(0, token_indices, (v_low_rank @ v_lora_b[expert_id]) * expert_scale)
    return q_delta, v_delta


def _ref_topk(hidden_states, expert_idx, expert_scores, q_lora_a, q_lora_b, v_lora_a, v_lora_b, scale, query_out, value_out, num_experts):
    num_tokens = hidden_states.shape[0]
    q_delta = hidden_states.new_zeros((num_tokens, query_out))
    v_delta = hidden_states.new_zeros((num_tokens, value_out))
    for expert_id in range(num_experts):
        matched = torch.nonzero(expert_idx == expert_id, as_tuple=False)
        if matched.numel() == 0:
            continue
        token_indices = matched[:, 0]
        expert_hidden = hidden_states.index_select(0, token_indices)
        expert_scale = expert_scores[matched[:, 0], matched[:, 1]].to(hidden_states.dtype).unsqueeze(-1) * scale
        q_low_rank = expert_hidden @ q_lora_a[expert_id]
        v_low_rank = expert_hidden @ v_lora_a[expert_id]
        q_delta.index_add_(0, token_indices, (q_low_rank @ q_lora_b[expert_id]) * expert_scale)
        v_delta.index_add_(0, token_indices, (v_low_rank @ v_lora_b[expert_id]) * expert_scale)
    return q_delta, v_delta


# ---------------------------------------------------------------------------
# New: selected-expert implementation (matches qv_lora_attention.py)
# ---------------------------------------------------------------------------

def _selected_impl(hidden_states, expert_idx, expert_scores, q_lora_a, q_lora_b, v_lora_a, v_lora_b, scale):
    if expert_idx.dim() == 1:
        expert_idx = expert_idx.unsqueeze(1)
        expert_scores = expert_scores.unsqueeze(1)

    num_tokens, k = expert_idx.shape
    scores = (expert_scores.to(hidden_states.dtype) * scale).unsqueeze(-1)
    flat_idx = expert_idx.reshape(-1)

    selected_qa = q_lora_a.index_select(0, flat_idx).reshape(num_tokens, k, q_lora_a.shape[1], q_lora_a.shape[2])
    selected_qb = q_lora_b.index_select(0, flat_idx).reshape(num_tokens, k, q_lora_b.shape[1], q_lora_b.shape[2])
    selected_va = v_lora_a.index_select(0, flat_idx).reshape(num_tokens, k, v_lora_a.shape[1], v_lora_a.shape[2])
    selected_vb = v_lora_b.index_select(0, flat_idx).reshape(num_tokens, k, v_lora_b.shape[1], v_lora_b.shape[2])

    q_low = torch.einsum('nd,nkdr->nkr', hidden_states, selected_qa)
    v_low = torch.einsum('nd,nkdr->nkr', hidden_states, selected_va)

    q_out = torch.einsum('nkr,nkrq->nkq', q_low, selected_qb)
    v_out = torch.einsum('nkr,nkrv->nkv', v_low, selected_vb)

    q_delta = (q_out * scores).sum(dim=1)
    v_delta = (v_out * scores).sum(dim=1)
    return q_delta, v_delta


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_test_data(num_tokens, input_size, rank, query_out, value_out, num_experts, topk, dtype=torch.float32):
    torch.manual_seed(42)
    hidden = torch.randn(num_tokens, input_size, dtype=dtype)
    q_lora_a = torch.randn(num_experts, input_size, rank, dtype=dtype)
    q_lora_b = torch.randn(num_experts, rank, query_out, dtype=dtype) * 0.01
    v_lora_a = torch.randn(num_experts, input_size, rank, dtype=dtype)
    v_lora_b = torch.randn(num_experts, rank, value_out, dtype=dtype) * 0.01

    router_weight = torch.randn(num_experts, input_size, dtype=dtype)
    logits = hidden @ router_weight.T
    probs = torch.softmax(logits, dim=-1)
    if topk == 1:
        scores, idx = torch.max(probs, dim=-1)
    else:
        scores, idx = torch.topk(probs, k=topk, dim=-1)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)

    scale = 1.0
    return hidden, idx, scores, q_lora_a, q_lora_b, v_lora_a, v_lora_b, scale


def test_top1_equivalence():
    N, D, R, Q, V, E = 128, 64, 8, 64, 64, 4
    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=1)

    q_ref, v_ref = _ref_top1(hidden, idx, scores, qa, qb, va, vb, scale, Q, V, E)
    q_new, v_new = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)

    q_err = (q_ref - q_new).abs().max().item()
    v_err = (v_ref - v_new).abs().max().item()
    print(f"[top-1] q max_abs_diff={q_err:.2e}, v max_abs_diff={v_err:.2e}", end=" ")
    assert q_err < 1e-5, f"top-1 Q mismatch: {q_err}"
    assert v_err < 1e-5, f"top-1 V mismatch: {v_err}"
    print("PASS")


def test_topk_equivalence():
    N, D, R, Q, V, E, K = 128, 64, 8, 64, 64, 7, 2
    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=K)

    q_ref, v_ref = _ref_topk(hidden, idx, scores, qa, qb, va, vb, scale, Q, V, E)
    q_new, v_new = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)

    q_err = (q_ref - q_new).abs().max().item()
    v_err = (v_ref - v_new).abs().max().item()
    print(f"[top-k] q max_abs_diff={q_err:.2e}, v max_abs_diff={v_err:.2e}", end=" ")
    assert q_err < 1e-5, f"top-k Q mismatch: {q_err}"
    assert v_err < 1e-5, f"top-k V mismatch: {v_err}"
    print("PASS")


def test_bf16_equivalence():
    N, D, R, Q, V, E = 256, 128, 16, 128, 128, 7
    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=1, dtype=torch.bfloat16)

    q_ref, v_ref = _ref_top1(hidden, idx, scores, qa, qb, va, vb, scale, Q, V, E)
    q_new, v_new = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)

    q_err = (q_ref.float() - q_new.float()).abs().max().item()
    v_err = (v_ref.float() - v_new.float()).abs().max().item()
    print(f"[bf16 top-1] q max_abs_diff={q_err:.2e}, v max_abs_diff={v_err:.2e}", end=" ")
    # bf16 has lower precision, allow slightly larger tolerance
    assert q_err < 5e-2, f"bf16 top-1 Q mismatch: {q_err}"
    assert v_err < 5e-2, f"bf16 top-1 V mismatch: {v_err}"
    print("PASS")


def test_bf16_topk_equivalence():
    N, D, R, Q, V, E, K = 256, 128, 16, 128, 128, 7, 2
    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=K, dtype=torch.bfloat16)

    q_ref, v_ref = _ref_topk(hidden, idx, scores, qa, qb, va, vb, scale, Q, V, E)
    q_new, v_new = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)

    q_err = (q_ref.float() - q_new.float()).abs().max().item()
    v_err = (v_ref.float() - v_new.float()).abs().max().item()
    print(f"[bf16 top-k] q max_abs_diff={q_err:.2e}, v max_abs_diff={v_err:.2e}", end=" ")
    assert q_err < 5e-2, f"bf16 top-k Q mismatch: {q_err}"
    assert v_err < 5e-2, f"bf16 top-k V mismatch: {v_err}"
    print("PASS")


def test_backward_gradients():
    """Ensure all LoRA parameters receive gradients via the einsum path."""
    N, D, R, Q, V, E = 64, 32, 8, 32, 32, 4
    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=1)

    # Make params require grad
    qa = qa.clone().requires_grad_(True)
    qb = qb.clone().requires_grad_(True)
    va = va.clone().requires_grad_(True)
    vb = vb.clone().requires_grad_(True)
    hidden = hidden.clone().requires_grad_(True)

    q_delta, v_delta = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)
    loss = q_delta.sum() + v_delta.sum()
    loss.backward()

    for name, param in [("q_lora_a", qa), ("q_lora_b", qb), ("v_lora_a", va), ("v_lora_b", vb), ("hidden", hidden)]:
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zeros"
    print("[backward] all params have non-zero gradients PASS")


def test_backward_topk_gradients():
    """Ensure gradients work for top-k routing."""
    N, D, R, Q, V, E, K = 64, 32, 8, 32, 32, 7, 2
    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=K)

    qa = qa.clone().requires_grad_(True)
    qb = qb.clone().requires_grad_(True)
    va = va.clone().requires_grad_(True)
    vb = vb.clone().requires_grad_(True)
    hidden = hidden.clone().requires_grad_(True)

    q_delta, v_delta = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)
    loss = q_delta.sum() + v_delta.sum()
    loss.backward()

    for name, param in [("q_lora_a", qa), ("q_lora_b", qb), ("v_lora_a", va), ("v_lora_b", vb), ("hidden", hidden)]:
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zeros"
    print("[backward top-k] all params have non-zero gradients PASS")


def test_freeze_hook_compatibility():
    """Verify that gradient hooks for freezing existing experts still work
    correctly with the einsum implementation."""
    N, D, R, Q, V = 64, 32, 8, 32, 32
    num_existing = 3
    num_total = 5

    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, num_total, topk=1)

    qa = qa.clone().requires_grad_(True)
    qb = qb.clone().requires_grad_(True)
    va = va.clone().requires_grad_(True)
    vb = vb.clone().requires_grad_(True)

    # Register freeze hooks (same as continual_learning_utils._freeze_qv_lora_experts)
    def _zero_existing_expert_grads(grad):
        grad = grad.clone()
        grad[:num_existing].zero_()
        return grad

    qa.register_hook(_zero_existing_expert_grads)
    qb.register_hook(_zero_existing_expert_grads)
    va.register_hook(_zero_existing_expert_grads)
    vb.register_hook(_zero_existing_expert_grads)

    q_delta, v_delta = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)
    loss = q_delta.sum() + v_delta.sum()
    loss.backward()

    for name, param in [("q_lora_a", qa), ("q_lora_b", qb), ("v_lora_a", va), ("v_lora_b", vb)]:
        grad = param.grad
        frozen_grad_norm = grad[:num_existing].abs().sum().item()
        new_grad_norm = grad[num_existing:].abs().sum().item()
        assert frozen_grad_norm == 0.0, f"{name} frozen experts have non-zero gradients: {frozen_grad_norm}"
        assert new_grad_norm > 0.0, f"{name} new experts have zero gradients"
    print("[freeze hooks] frozen experts have zero grad, new experts have non-zero grad PASS")


def test_empty_expert():
    """Test that the selected-expert approach handles the case where some experts
    receive zero tokens (which the original loop skipped)."""
    N, D, R, Q, V, E = 8, 16, 4, 16, 16, 10  # many experts, few tokens
    hidden, _, _, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=1)

    # Force all tokens to expert 0
    idx = torch.zeros(N, dtype=torch.long)
    scores = torch.ones(N)

    q_ref, v_ref = _ref_top1(hidden, idx, scores, qa, qb, va, vb, scale, Q, V, E)
    q_new, v_new = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)

    q_err = (q_ref - q_new).abs().max().item()
    v_err = (v_ref - v_new).abs().max().item()
    assert q_err < 1e-5, f"empty expert Q mismatch: {q_err}"
    assert v_err < 1e-5, f"empty expert V mismatch: {v_err}"
    print("[empty experts] correctly handles unused experts PASS")


def test_production_scale():
    """Test with production-like dimensions: seq=512*8=4096, hidden=1024, rank=16, E=7."""
    N, D, R, Q, V, E = 4096, 1024, 16, 1024, 1024, 7
    hidden, idx, scores, qa, qb, va, vb, scale = make_test_data(N, D, R, Q, V, E, topk=1)

    q_ref, v_ref = _ref_top1(hidden, idx, scores, qa, qb, va, vb, scale, Q, V, E)
    q_new, v_new = _selected_impl(hidden, idx, scores, qa, qb, va, vb, scale)

    q_err = (q_ref - q_new).abs().max().item()
    v_err = (v_ref - v_new).abs().max().item()
    print(f"[production scale] q max_abs_diff={q_err:.2e}, v max_abs_diff={v_err:.2e}", end=" ")
    assert q_err < 1e-4, f"production scale Q mismatch: {q_err}"
    assert v_err < 1e-4, f"production scale V mismatch: {v_err}"
    print("PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("QV LoRA einsum equivalence tests")
    print("=" * 60)
    test_top1_equivalence()
    test_topk_equivalence()
    test_bf16_equivalence()
    test_bf16_topk_equivalence()
    test_backward_gradients()
    test_backward_topk_gradients()
    test_freeze_hook_compatibility()
    test_empty_expert()
    test_production_scale()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
