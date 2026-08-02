import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lpr_pretrain_gpt", ROOT / "Megatron-LM" / "pretrain_gpt.py")
pretrain_gpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretrain_gpt)


class FixedRouter:
    def __init__(self, logits):
        self.logits = logits

    def gating(self, hidden):
        return self.logits[: hidden.shape[0]]


def run_lpr(logits, dataset_ids, ranges, prefix_counts=None, layers=1):
    batch = dataset_ids.numel()
    seq = logits.shape[0] // batch
    labels = torch.zeros((batch, seq), dtype=torch.long)
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    hidden = torch.zeros((seq, batch, 4))
    inputs = {i + 1: hidden for i in range(layers)}
    routers = {i + 1: FixedRouter(logits) for i in range(layers)}
    args = SimpleNamespace(
        moe_lpr_dataset_prefix_counts=prefix_counts or ",".join("1" for _ in ranges),
        moe_lpr_task_expert_ranges=",".join(ranges),
    )
    return pretrain_gpt._masked_task_group_lpr(
        inputs, routers, labels, loss_mask, dataset_ids, args
    )


def test_group_nll_matches_log_probability_mass():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]] * 4, requires_grad=True)
    loss = run_lpr(logits, torch.tensor([0, 1]), ["0:2", "-"])
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = -torch.logsumexp(log_probs[:2, :2], dim=-1).sum()
    torch.testing.assert_close(loss, expected)


def test_new_task_has_no_lpr_gradient():
    logits = torch.randn(6, 6, requires_grad=True)
    loss = run_lpr(logits, torch.tensor([0, 1, 2]), ["0:2", "2:4", "-"])
    loss.backward()
    assert logits.grad[:4].abs().sum() > 0
    assert logits.grad[4:].abs().sum() == 0


def test_group_members_are_not_individually_labeled():
    logits = torch.tensor([[3.0, 3.0, -2.0, -2.0]], requires_grad=True)
    loss = run_lpr(logits, torch.tensor([0]), ["0:2"])
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 0], logits.grad[0, 1])


def test_42_conversation_prefixes_map_to_new_task():
    logits = torch.randn(6, 24, requires_grad=True)
    # dataset ids: wiki prefix 0, code prefix 1, conversation prefixes 2..43
    loss = run_lpr(
        logits, torch.tensor([0, 1, 43]), ["0:8", "8:16", "-"], prefix_counts="1,1,42"
    )
    loss.backward()
    assert logits.grad[:4].abs().sum() > 0
    assert logits.grad[4:].abs().sum() == 0


def test_layer_losses_are_averaged_not_summed():
    logits = torch.randn(2, 4, requires_grad=True)
    one = run_lpr(logits, torch.tensor([0]), ["0:2"], layers=1)
    two = run_lpr(logits, torch.tensor([0]), ["0:2"], layers=2)
    torch.testing.assert_close(one, two)


def test_loss_mask_excludes_tokens():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], requires_grad=True)
    labels = torch.zeros((1, 2), dtype=torch.long)
    mask = torch.tensor([[1.0, 0.0]])
    hidden = torch.zeros((2, 1, 4))
    args = SimpleNamespace(moe_lpr_dataset_prefix_counts="1", moe_lpr_task_expert_ranges="0:2")
    loss = pretrain_gpt._masked_task_group_lpr(
        {1: hidden}, {1: FixedRouter(logits)}, labels, mask, torch.tensor([0]), args
    )
    loss.backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].abs().sum() == 0
