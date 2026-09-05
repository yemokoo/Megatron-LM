from types import SimpleNamespace

import torch

from megatron.core.optimizer import _get_param_groups
from megatron.core.optimizer.optimizer import ChainedOptimizer


class _RouterAndExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ddp_config = SimpleNamespace(use_custom_fsdp=False)
        self.mlp = torch.nn.Module()
        self.mlp.router = torch.nn.Linear(4, 2, bias=False)
        self.mlp.experts = torch.nn.Linear(4, 4, bias=False)


def test_router_lr_multiplier_only_scales_router_group():
    model = _RouterAndExpertModel()
    groups = _get_param_groups(
        [model],
        no_weight_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0,
        lr=3.0e-4,
        min_lr=3.0e-5,
        decoupled_lr=None,
        decoupled_min_lr=None,
        moe_router_lr_multiplier=0.1,
    )
    group_by_param = {
        id(param): group for group in groups for param in group['params']
    }

    router_group = group_by_param[id(model.mlp.router.weight)]
    expert_group = group_by_param[id(model.mlp.experts.weight)]

    assert router_group['is_moe_router']
    assert router_group['lr_mult'] == 0.1
    assert not router_group['is_new_expert_lr_ramp']
    assert not expert_group['is_moe_router']
    assert expert_group['lr_mult'] == 1.0


class _FakeInnerOptimizer:
    def __init__(self, config, grad_norm):
        self.config = config
        self._grad_norm = grad_norm
        self._parameter = torch.nn.Parameter(torch.ones(1))
        self.step_count = 0

    def prepare_grads(self): return False
    def get_grad_norm(self): return self._grad_norm
    def get_parameters(self): return [self._parameter]
    def count_zeros(self): return 0
    def step_with_ready_grads(self):
        self.step_count += 1
        return True


def test_separate_clip_uses_partition_norms_without_replacing_optimizers(monkeypatch):
    config = SimpleNamespace(
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        use_precision_aware_optimizer=False,
        overlap_param_gather_with_optimizer_step=False,
        moe_separate_router_expert_grad_clip=True,
    )
    router = _FakeInnerOptimizer(config, 3.0)
    expert = _FakeInnerOptimizer(config, 4.0)
    chained = ChainedOptimizer([router, expert])
    original_ids = [id(inner) for inner in chained.chained_optimizers]
    clip_norms = []

    monkeypatch.setattr(
        'megatron.core.optimizer.optimizer.clip_grad_by_total_norm_fp32',
        lambda _params, max_norm, total_norm, use_decoupled_grad: clip_norms.append(total_norm),
    )
    success, reported_norm, _ = chained.step()

    assert success
    assert reported_norm == 5.0
    assert clip_norms == [3.0, 4.0]
    assert [id(inner) for inner in chained.chained_optimizers] == original_ids
    assert router.step_count == expert.step_count == 1
