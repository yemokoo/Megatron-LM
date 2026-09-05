#!/usr/bin/env python
"""Eight-rank CUDA smoke for V3's sparse every-update replay schedule."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.Ours_LoRA_MoE_V3 import (  # noqa: E402
    Ours_LoRA_MoE_V3,
    add_v3_experts,
    attach_shared_qkvo_lora_moe,
    freeze_v3_experts,
    freeze_v3_routers,
    shared_router_layers,
)
from test_ours_lora_moe_v3 import (  # noqa: E402
    TinyBatchLoader,
    TinyCausalModel,
    randomize_expert_outputs,
)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def make_batch(values, source, repeats=1):
    input_ids = torch.tensor([values], dtype=torch.long).repeat(repeats, 1)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "sources": [source] * repeats,
    }


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise RuntimeError(f"this smoke expects 8 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(2025)

    model = TinyCausalModel().to(device)
    attach_shared_qkvo_lora_moe(
        model, r=2, alpha=4, top_k=2,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="full_softmax", dropout=0.0)
    add_v3_experts(model, 2)
    randomize_expert_outputs(model)
    freeze_v3_experts(model, trainable_expert_indices={1})
    freeze_v3_routers(model, trainable=True)
    old_experts = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if ".experts.0." in name
    }
    router_before = shared_router_layers(model)[
        0].shared_expert_router.weight.detach().clone()

    trainer = object.__new__(Ours_LoRA_MoE_V3)
    trainer.raw_model = model
    trainer.model = DDP(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True, broadcast_buffers=False)
    trainer.optimizer = CountingSGD([
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ])
    trainer.lr_scheduler = CountingScheduler()
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._active_task_workload = None
    trainer.args = SimpleNamespace(
        global_rank=rank, gradient_accumulation_steps=1,
        v2_max_replay_batches_per_step=0,
        v2_replay_forward_batch_size=8,
        v2_joint_new_to_replay_ratio=5,
        v2_joint_replay_loss_coeff=1.0, loss_log_interval=1)

    # Five updates expose 320 new samples globally (8 samples/rank/update).
    # At 5:1 this needs 64 replay exposures, assigned as 12/13 per update. The
    # stored pool is only 8 global samples, so every rank cycles its shard 8x.
    primary_loader = TinyBatchLoader([
        make_batch([1, 2, 3], f"primary-{step}", repeats=8)
        for step in range(5)
    ], batch_size=8)
    primary_loader.sampler.total_size = 320
    memory_loader = TinyBatchLoader([
        make_batch([13 + rank, 14 + rank, 15 + rank], f"replay-r{rank}")
    ], batch_size=1)
    memory_loader.dataset = range(8)

    trainer._run_v2_joint_epochs(
        primary_loader, memory_loader, epochs=1,
        device=device, phase_name="v3-ddp-smoke")

    if model.forward_calls != 10:
        raise AssertionError(
            f"rank {rank}: expected 5 primary + 5 packed replay forwards, "
            f"got {model.forward_calls}")
    if trainer.optimizer.step_calls != 5:
        raise AssertionError(
            f"rank {rank}: optimizer steps={trainer.optimizer.step_calls}")
    final_state = model.state_dict()
    for name, expected in old_experts.items():
        torch.testing.assert_close(
            final_state[name], expected, rtol=0, atol=0)
    router = shared_router_layers(model)[0].shared_expert_router.weight
    if torch.equal(router, router_before):
        raise AssertionError(f"rank {rank}: router did not update")
    gathered = [torch.empty_like(router) for _ in range(world_size)]
    dist.all_gather(gathered, router)
    for other in gathered[1:]:
        torch.testing.assert_close(other, gathered[0], rtol=0, atol=0)

    dist.barrier()
    if rank == 0:
        print(
            "V3_DDP_EVERY_UPDATE_REPLAY_SMOKE=PASS "
            "updates=5 stored_pool=8 global_replay_exposures=64 ratio=5:1 "
            "global_replay_per_update=12_or_13 local_replay_forwards=5")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
