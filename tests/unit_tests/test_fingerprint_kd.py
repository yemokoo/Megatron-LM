import torch
from types import SimpleNamespace

from pretrain_gpt import _accumulate_probe_router_usage, _masked_fingerprint_projected_mse


def _hidden(rows):
    # [batch, seq, hidden] -> Transformer [seq, batch, hidden]
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0).permute(1, 0, 2).contiguous()


def test_hard_gate_only_selected_token_gets_kd_gradient():
    labels = torch.zeros((1, 2), dtype=torch.long)
    loss_mask = torch.ones((1, 2), dtype=torch.float32)
    teacher_layer = _hidden([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
    student_layer = teacher_layer.clone()
    student_layer[:, :, 1] += 1.0
    student_layer.requires_grad_(True)
    score_basis = torch.zeros((1, 4, 1), dtype=torch.float32)
    score_basis[0, 0, 0] = 1.0
    loss_basis = torch.zeros((1, 4, 1), dtype=torch.float32)
    loss_basis[0, 1, 0] = 1.0
    bundle = {
        "layers": [9],
        "means": torch.zeros((1, 4), dtype=torch.float32),
        "score_bases": score_basis,
        "loss_bases": loss_basis,
    }
    result = _masked_fingerprint_projected_mse(
        {9: student_layer},
        {9: teacher_layer},
        labels,
        loss_mask,
        bundle,
        gate_mode="hard",
        threshold=0.5,
        soft_temperature=0.1,
    )
    result["loss"].backward()
    flat_grad = student_layer.grad.permute(1, 0, 2).reshape(2, 4)
    assert flat_grad[0, 1] != 0
    assert torch.count_nonzero(flat_grad[1]) == 0
    assert result["hard_coverage"].item() == 0.5
    assert not bundle["score_bases"].requires_grad
    assert not bundle["loss_bases"].requires_grad


def test_soft_gate_is_teacher_derived_and_layer9_is_included():
    labels = torch.zeros((1, 2), dtype=torch.long)
    loss_mask = torch.ones((1, 2), dtype=torch.float32)
    teacher = {
        2: _hidden([[1.0, 0.0], [0.0, 1.0]]),
        9: _hidden([[1.0, 0.0], [0.0, 1.0]]),
    }
    student = {layer: value.clone().requires_grad_(True) for layer, value in teacher.items()}
    student[2].data[:, :, 0] += 0.25
    student[9].data[:, :, 0] += 0.50
    basis = torch.zeros((2, 2, 1), dtype=torch.float32)
    basis[:, 0, 0] = 1.0
    bundle = {
        "layers": [2, 9],
        "means": torch.zeros((2, 2), dtype=torch.float32),
        "score_bases": basis,
        "loss_bases": basis,
    }
    result = _masked_fingerprint_projected_mse(
        student,
        teacher,
        labels,
        loss_mask,
        bundle,
        gate_mode="soft",
        threshold=0.5,
        soft_temperature=0.1,
    )
    result["loss"].backward()
    assert result["layer_loss_means"].shape == (2,)
    assert result["layer_loss_means"][1] > result["layer_loss_means"][0]
    assert student[9].grad is not None
    assert teacher[9].grad is None


def test_permuted_assignment_preserves_valid_weight_multiset_but_breaks_token_pairing():
    labels = torch.zeros((1, 4), dtype=torch.long)
    # The masked token must not receive or donate an applied KD weight.
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    teacher_layer = _hidden(
        [[2.0, 0.0], [1.0, 1.0], [0.0, 2.0], [0.2, 2.0]]
    )
    student_layer = teacher_layer.clone()
    # Make projected MSE distinct per token so assignment changes the loss.
    student_layer[:, :, 0] += torch.tensor([0.25, 0.5, 1.0, 2.0]).view(-1, 1)
    basis = torch.zeros((1, 2, 1), dtype=torch.float32)
    basis[0, 0, 0] = 1.0
    bundle = {
        "layers": [9],
        "means": torch.zeros((1, 2), dtype=torch.float32),
        "score_bases": basis,
        "loss_bases": basis,
    }
    kwargs = dict(
        student_hidden={9: student_layer},
        teacher_hidden={9: teacher_layer},
        labels=labels,
        loss_mask=loss_mask,
        bundle=bundle,
        gate_mode="soft",
        threshold=0.5,
        soft_temperature=0.1,
    )
    stable = _masked_fingerprint_projected_mse(**kwargs, weight_assignment="stable")
    permuted = _masked_fingerprint_projected_mse(**kwargs, weight_assignment="permuted")
    assert torch.equal(stable["mean_weight"], permuted["mean_weight"])
    assert torch.equal(stable["hard_coverage"], permuted["hard_coverage"])
    assert not torch.isclose(stable["loss"], permuted["loss"])
    assert permuted["score_weight_covariance"] < stable["score_weight_covariance"]


def test_full_hidden_loss_preserves_dimensions_outside_score_basis_and_layer9():
    labels = torch.zeros((1, 2), dtype=torch.long)
    loss_mask = torch.ones((1, 2), dtype=torch.float32)
    teacher = {
        2: _hidden([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        9: _hidden([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    }
    student = {layer: value.clone().requires_grad_(True) for layer, value in teacher.items()}
    # The selector only observes dim 0.  Full-hidden KD must nevertheless
    # produce gradients for drift in dims 1 and 2, including at layer 9.
    student[2].data[:, :, 1] += 0.25
    student[9].data[:, :, 2] += 0.50
    score_basis = torch.zeros((2, 3, 1), dtype=torch.float32)
    score_basis[:, 0, 0] = 1.0
    bundle = {
        "layers": [2, 9],
        "means": torch.zeros((2, 3), dtype=torch.float32),
        "score_bases": score_basis,
        "loss_bases": None,
    }
    result = _masked_fingerprint_projected_mse(
        student,
        teacher,
        labels,
        loss_mask,
        bundle,
        gate_mode="hard",
        threshold=0.5,
        soft_temperature=0.1,
    )
    result["loss"].backward()
    assert result["hard_coverage"].item() == 1.0
    assert student[2].grad[:, :, 1].abs().sum() > 0
    assert student[9].grad[:, :, 2].abs().sum() > 0
    assert student[2].grad[:, :, 0].abs().sum() == 0
    assert student[9].grad[:, :, 0].abs().sum() == 0
    assert teacher[9].grad is None


def test_probe_router_usage_accepts_standard_moe_dict_capture():
    router = SimpleNamespace(
        config=SimpleNamespace(moe_router_dtype="fp32"),
        weight=torch.eye(4, dtype=torch.float32),
    )
    captured = {
        9: torch.tensor(
            [[[8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]]],
            dtype=torch.float32,
        )
    }
    totals = {}
    hist_totals = {}
    found = _accumulate_probe_router_usage(
        captured,
        {9: router},
        num_existing_experts=2,
        args=SimpleNamespace(moe_router_topk=1),
        totals=totals,
        hist_totals=hist_totals,
    )
    assert found
    assert torch.isclose(totals["old_expert_fraction"], torch.tensor(0.5))
    assert torch.isclose(totals["new_expert_fraction"], torch.tensor(0.5))
    assert torch.equal(hist_totals[9], torch.tensor([0.5, 0.0, 0.0, 0.5]))
