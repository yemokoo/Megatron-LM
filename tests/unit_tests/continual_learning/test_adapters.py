from types import SimpleNamespace

import torch

from megatron.core.continual_learning.lora_adapter import ContinualLowRankAdapter
from megatron.core.continual_learning import olora
from megatron.core.continual_learning.slora import _corrected_denoised_delta


def _config():
    return SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.float32,
    )


def test_active_rank_prefix_and_scale():
    adapter = ContinualLowRankAdapter(_config(), 4, 3, max_rank=4, alpha=8.0)
    adapter.reset_active(2, init_std=0.01)
    with torch.no_grad():
        adapter.lora_a[:, :2].copy_(torch.arange(8).reshape(4, 2) / 10)
        adapter.lora_b[:2].copy_(torch.arange(6).reshape(2, 3) / 10)
    inputs = torch.randn(5, 4)
    expected = inputs @ adapter.lora_a[:, :2] @ adapter.lora_b[:2] * 4.0
    assert torch.allclose(adapter(inputs), expected)
    assert torch.allclose(inputs @ adapter.local_delta_weight().T, expected)


def test_inactive_allocation_has_no_forward_effect():
    adapter = ContinualLowRankAdapter(_config(), 3, 2, max_rank=4, alpha=4.0)
    adapter.reset_active(1)
    with torch.no_grad():
        adapter.lora_a[:, 1:].fill_(1000)
        adapter.lora_b[1:, :].fill_(1000)
    inputs = torch.randn(7, 3)
    output = adapter(inputs)
    expected = inputs @ adapter.lora_a[:, :1] @ adapter.lora_b[:1, :] * 4.0
    assert torch.allclose(output, expected)


def test_olora_gram_overlaps_the_paper_expansion_factor(monkeypatch):
    """O-LoRA Eq. (6) overlaps A_i^T A_t, and Eq. (2) makes A the expander.

    The paper multiplies on the left (``h = W_init x + A B x``), so its ``A``
    is the output-side factor -- this module's ``lora_b`` transposed -- not the
    compressor ``lora_a``.  Both adapters below share an identical ``lora_a``,
    so an overlap taken on the compressor would be maximal, while the paper's
    overlap on the orthogonal expansion columns is exactly zero.
    """
    # No tensor-parallel group exists here, so take the single-rank path.
    monkeypatch.setattr(
        olora.parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    previous = ContinualLowRankAdapter(_config(), 3, 4, max_rank=2, alpha=2.0)
    current = ContinualLowRankAdapter(_config(), 3, 4, max_rank=2, alpha=2.0)
    previous.reset_active(2)
    current.reset_active(2)
    with torch.no_grad():
        previous.lora_b.copy_(torch.eye(4)[:, :2].T)
        current.lora_b.copy_(torch.eye(4)[:, 2:].T)
        previous.lora_a.fill_(1.0)
        current.lora_a.fill_(1.0)

    gram = olora._global_gram(previous, current)
    assert gram.shape == (2, 2)
    assert torch.allclose(gram, torch.zeros(2, 2))

    # Overlapping expansion subspaces must be penalized.
    with torch.no_grad():
        current.lora_b.copy_(previous.lora_b)
    assert olora._global_gram(previous, current).square().sum() > 0

    # The penalty must be blind to the compression factor.
    with torch.no_grad():
        current.lora_b.copy_(torch.eye(4)[:, 2:].T)
        current.lora_a.mul_(-7.0)
    assert torch.allclose(olora._global_gram(previous, current), torch.zeros(2, 2))


def test_corrected_slora_candidate_shapes_and_determinism():
    torch.manual_seed(7)
    delta = torch.randn(12, 9)
    reference = torch.randn(12, 9)
    first, rank_first = _corrected_denoised_delta(delta, reference, trained_rank=8)
    second, rank_second = _corrected_denoised_delta(delta, reference, trained_rank=8)
    assert first.shape == delta.shape
    assert 1 <= rank_first <= 8
    assert rank_first == rank_second
    assert torch.allclose(first, second)
