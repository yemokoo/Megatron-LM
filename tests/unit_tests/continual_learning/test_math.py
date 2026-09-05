import torch

from megatron.core.continual_learning.audit import expected_active_counts
from megatron.core.continual_learning.ewc import consolidate_equal_lambda, ewc_penalty
from megatron.core.continual_learning.trace_gem import project_tensor_gradient


def test_active_parameter_equalities():
    counts = expected_active_counts()
    assert counts["dense_ffn_projection_parameters_per_layer"] == 4_325_376
    assert counts["fixed_moe_top4_projection_parameters_per_layer"] == 4_325_376
    assert counts["olora_final_three_qv_projection_parameters_per_layer"] == 4_325_376


def test_ewc_anchor_zero_and_finite_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    fisher = {"p": torch.tensor([3.0, 4.0])}
    mean = {"p": parameter.detach().clone()}
    penalty = ewc_penalty({"p": parameter}, fisher, mean, coefficient=400.0)
    assert penalty.item() == 0.0
    penalty.backward()
    assert torch.isfinite(parameter.grad).all()
    assert torch.equal(parameter.grad, torch.zeros_like(parameter))


def test_ewc_consolidation_has_identical_gradient():
    old_f = {"p": torch.tensor([1.0, 3.0])}
    new_f = {"p": torch.tensor([5.0, 7.0])}
    old_m = {"p": torch.tensor([-1.0, 2.0])}
    new_m = {"p": torch.tensor([3.0, -4.0])}
    f_sum, mean = consolidate_equal_lambda(old_f, old_m, new_f, new_m)
    x = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    separate = 0.5 * (
        (old_f["p"] * (x - old_m["p"]).square()).sum()
        + (new_f["p"] * (x - new_m["p"]).square()).sum()
    )
    consolidated = 0.5 * (f_sum["p"] * (x - mean["p"]).square()).sum()
    grad_separate = torch.autograd.grad(separate, x, retain_graph=True)[0]
    grad_consolidated = torch.autograd.grad(consolidated, x)[0]
    assert torch.allclose(grad_separate, grad_consolidated, atol=1e-6)


def test_trace_gem_conflict_projection_satisfies_constraints():
    gradient = torch.tensor([-2.0, -1.0])
    memories = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    projected = project_tensor_gradient(gradient, memories, margin=0.5)
    for memory in memories:
        assert torch.dot(projected, memory) >= -2e-4
    assert torch.allclose(projected, torch.zeros(2), atol=2e-4)


def test_trace_gem_margin_is_dual_lower_bound():
    gradient = torch.tensor([-0.1, 2.0])
    memory = torch.tensor([1.0, 0.0])
    projected = project_tensor_gradient(gradient, [memory], margin=0.5)
    assert torch.allclose(projected, torch.tensor([0.4, 2.0]), atol=2e-4)


def test_trace_gem_nonconflict_is_identity():
    gradient = torch.tensor([2.0, 1.0])
    memory = torch.tensor([1.0, 0.0])
    assert torch.equal(project_tensor_gradient(gradient, [memory], margin=0.5), gradient)
