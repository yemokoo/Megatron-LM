import torch
from megatron.training.training import (
    _restore_joint_replay_non_router_grads,
    _snapshot_joint_replay_non_router_grads,
)

def test_primary_expert_grad_is_kept_and_router_grad_accumulates():
    expert=torch.nn.Parameter(torch.zeros(2));router=torch.nn.Parameter(torch.zeros(2))
    expert.main_grad=torch.tensor([1.,2.]);router.main_grad=torch.tensor([3.,4.])
    snapshot=_snapshot_joint_replay_non_router_grads([expert,router],{id(router)})
    expert.main_grad.add_(torch.tensor([10.,20.]));router.main_grad.add_(torch.tensor([30.,40.]))
    _restore_joint_replay_non_router_grads(snapshot)
    assert torch.equal(expert.main_grad,torch.tensor([1.,2.]))
    assert torch.equal(router.main_grad,torch.tensor([33.,44.]))

def test_replay_only_expert_grad_is_zeroed():
    expert=torch.nn.Parameter(torch.zeros(2));expert.main_grad=None
    snapshot=_snapshot_joint_replay_non_router_grads([expert],set())
    expert.main_grad=torch.tensor([7.,8.]);_restore_joint_replay_non_router_grads(snapshot)
    assert torch.equal(expert.main_grad,torch.zeros(2))

def test_frozen_parameter_is_ignored():
    frozen=torch.nn.Parameter(torch.zeros(1),requires_grad=False);frozen.main_grad=torch.ones(1)
    assert _snapshot_joint_replay_non_router_grads([frozen],set())=={}
