import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Megatron-LM"))
spec = importlib.util.spec_from_file_location(
    "kd_reporting_pretrain_gpt", ROOT / "Megatron-LM" / "pretrain_gpt.py"
)
pretrain_gpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretrain_gpt)


def test_kd_reporting_uses_kl_sum_with_token_denominator(monkeypatch):
    temperature = 2.0
    args = SimpleNamespace(
        moe_expansion_distill_lm_loss_coeff=0.0,
        moe_old_model_kl_temperature=temperature,
        moe_expansion_distill_mode="logits",
        moe_old_model_kl_coeff=1.0,
        context_parallel_size=1,
        check_for_nan_in_loss_and_grad=False,
        check_for_spiky_loss=False,
    )
    monkeypatch.setattr(pretrain_gpt, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_gpt, "get_rerun_state_machine", object)
    monkeypatch.setattr(pretrain_gpt.mpu, "get_data_parallel_group", lambda: "dp")

    reductions = []

    def record_all_reduce(tensor, group=None):
        reductions.append((tuple(tensor.shape), group))

    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce)

    student_logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.5, -0.5, 1.5], [1.0, 2.0, 3.0]]
    )
    teacher_logits = torch.tensor(
        [[0.5, 1.0, -0.5], [1.5, 0.0, -1.0], [-2.0, 1.0, 0.0]]
    )
    loss_mask = torch.tensor([1.0, 1.0, 0.0])
    output_tensor = {
        "losses": torch.zeros(3),
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
    }

    objective_sum, num_tokens, reporting = pretrain_gpt.loss_func(loss_mask, output_tensor)

    per_token_kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected_sum = (per_token_kl * loss_mask).sum() * temperature**2
    expected_mean = expected_sum / loss_mask.sum()

    torch.testing.assert_close(objective_sum, expected_sum)
    assert num_tokens.item() == 2
    kd_numerator, kd_denominator = reporting["kd loss"]
    torch.testing.assert_close(kd_numerator, expected_sum)
    torch.testing.assert_close(kd_denominator, loss_mask.sum())
    torch.testing.assert_close(kd_numerator / kd_denominator, expected_mean)
    lm_numerator, lm_denominator = reporting["lm loss"]
    torch.testing.assert_close(lm_numerator / lm_denominator, expected_mean)
    assert reductions == [((2,), "dp"), ((1,), "dp")]
