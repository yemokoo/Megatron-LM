"""Trainers for the five non-MH-MoE baselines in arXiv:2602.12587.

Method-specific behavior is adapted from the original papers/repositories:
LoRAMoE (Ablustrund/LoRAMoE), GEM
(facebookresearch/GradientEpisodicMemory), and O-LoRA
(cmnfriend/O-LoRA). EWC and GEM operate only on the shared SeqLoRA
parameters, as required by the target paper.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import get_scheduler

from model.base_model import CL_Base_Model
from model.continual_lora import (
    PAPER_STATE_KEY_SUBSTRINGS,
    collect_loramoe_loss,
    collect_olora_regularization,
    parameter_report,
    save_paper_baseline_meta,
    set_olora_task,
    set_router_token_mask,
    trainable_named_parameters,
)
from utils.utils import get_optimizer_grouped_parameters, print_rank_0, to_device


def _device(args) -> torch.device:
    return (torch.device("cuda", args.local_rank)
            if args.local_rank != -1 else torch.device("cuda"))


def _flatten_gradients(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for parameter in parameters:
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            pieces.append(parameter.grad.detach().float().reshape(-1))
    return torch.cat(pieces)


def _flatten_autograd(grads: Sequence[Optional[torch.Tensor]],
                      parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(grads, parameters):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            pieces.append(gradient.detach().float().reshape(-1))
    return torch.cat(pieces)


def _write_flat_gradient(parameters: Sequence[torch.nn.Parameter],
                         flat: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        value = flat[offset:offset + count].view_as(parameter).to(parameter.dtype)
        if parameter.grad is None:
            parameter.grad = value.clone()
        else:
            parameter.grad.copy_(value)
        offset += count
    if offset != flat.numel():
        raise ValueError("flat gradient has the wrong length")


def _solve_lower_bound_qp(P: torch.Tensor, q: torch.Tensor,
                          lower: float) -> torch.Tensor:
    """Solve min .5 v'Pv + q'v, v>=lower by enumerating active sets.

    GEM has at most seven past-task constraints on TRACE, so exact active-set
    enumeration is tiny (2^7) and removes the unavailable quadprog dependency.
    """
    n = q.numel()
    if n == 0:
        return q
    # P and q are only at most 7x7/7. Solving the 2^n active sets with torch
    # on CUDA launches thousands of tiny kernels per model step. Move this tiny
    # QP to CPU once and use NumPy/LAPACK; the mathematical enumeration and KKT
    # acceptance criteria remain identical.
    output_device = q.device
    output_dtype = q.dtype
    p_np = P.detach().float().cpu().numpy()
    q_np = q.detach().float().cpu().numpy()
    best_value = None
    best = None
    indices = range(n)
    for active_bits in itertools.product((False, True), repeat=n):
        active = [i for i in indices if active_bits[i]]
        free = [i for i in indices if not active_bits[i]]
        v = np.full(n, float(lower), dtype=np.float32)
        if free:
            rhs = -q_np[free]
            if active:
                rhs = rhs - p_np[np.ix_(free, active)] @ v[active]
            matrix = p_np[np.ix_(free, free)]
            try:
                v[free] = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                v[free] = np.linalg.pinv(matrix) @ rhs
            if np.any(v[free] < lower - 1e-5):
                continue
        gradient = p_np @ v + q_np
        if active and np.any(gradient[active] < -1e-4):
            continue
        if free and np.any(np.abs(gradient[free]) > 2e-3):
            continue
        value = float(0.5 * v @ p_np @ v + q_np @ v)
        if best_value is None or value < best_value:
            best_value, best = value, v.copy()
    if best is None:
        raise RuntimeError("GEM dual QP active-set solver found no feasible point")
    return torch.from_numpy(best).to(device=output_device, dtype=output_dtype)


def project_gem_gradient(current: torch.Tensor,
                         memories: Sequence[torch.Tensor],
                         margin: float = 0.0,
                         eps: float = 1e-3) -> torch.Tensor:
    """Exact dual projection used by the official GEM implementation."""
    if not memories:
        return current
    n = len(memories)
    P = current.new_empty((n, n))
    for row in range(n):
        for col in range(n):
            P[row, col] = torch.dot(memories[row], memories[col])
    P = 0.5 * (P + P.t()) + torch.eye(n, device=P.device) * eps
    # quadprog (used by the official GEM repo) minimizes .5 v'Pv - a'v
    # with a=-M'g. Our solver uses the conventional .5 v'Pv + q'v, hence
    # q=M'g here.
    q = torch.stack([torch.dot(memory, current) for memory in memories])
    multipliers = _solve_lower_bound_qp(P, q, margin)
    projected = current.clone()
    for multiplier, memory in zip(multipliers, memories):
        projected.add_(memory, alpha=float(multiplier))
    return projected


class PaperBaseline(CL_Base_Model):
    """Shared DDP/AdamW loop; subclasses change only the CL constraint."""

    method = "seqlora"
    save_key_substrings = PAPER_STATE_KEY_SUBSTRINGS

    def __init__(self, model, tokenizer, optimizer, train_task_list,
                 eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list,
                         eval_task_list, test_task_list, args)
        self.raw_model = model
        self._wrap_distributed_model()
        self._build_optimizer()
        self.current_task = 0
        self.task_results: List[Dict[str, object]] = []
        self._process_started = time.monotonic()
        self._task_stats: Dict[str, int] = {}

    def _wrap_distributed_model(self) -> None:
        """Build DDP from the current requires_grad mask."""
        self.model = self.raw_model
        if self.args.local_rank != -1:
            self.model = DDP(
                self.raw_model, device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False, gradient_as_bucket_view=True)

    def _build_optimizer(self, task_index: Optional[int] = None) -> None:
        grouped = get_optimizer_grouped_parameters(
            self.raw_model, self.args.weight_decay)
        grouped = [group for group in grouped if group["params"]]
        self.optimizer = torch.optim.AdamW(
            grouped, lr=self.args.learning_rate, betas=(0.9, 0.95),
            eps=self.args.adam_epsilon)
        if task_index is None:
            accum = max(1, self.args.gradient_accumulation_steps)
            total_steps = sum(
                ((min(len(loader), self.args.max_train_steps_per_task)
                  if self.args.max_train_steps_per_task > 0 else len(loader))
                 + accum - 1) // accum
                * int(self.args.num_train_epochs[index])
                for index, loader in enumerate(self.train_task_list.values()))
        else:
            loader = list(self.train_task_list.values())[task_index]
            steps = (min(len(loader), self.args.max_train_steps_per_task)
                     if self.args.max_train_steps_per_task > 0 else len(loader))
            accum = max(1, self.args.gradient_accumulation_steps)
            total_steps = ((steps + accum - 1) // accum
                           * int(self.args.num_train_epochs[task_index]))
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type, self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=max(1, total_steps))
        self.optimizer.zero_grad(set_to_none=True)

    def _set_gradient_checkpointing(self, task: str) -> None:
        enabled = task in getattr(self.args, "ckpt_tasks", set())
        if enabled:
            self.raw_model.enable_input_require_grads()
            self.raw_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.raw_model.gradient_checkpointing_disable()

    def before_task(self, task: str, task_index: int) -> None:
        self.current_task = task_index
        self._set_gradient_checkpointing(task)

    def after_task(self, task: str, task_index: int) -> None:
        pass

    def additional_loss(self, task_index: int) -> torch.Tensor:
        parameter = next(parameter for parameter in self.raw_model.parameters()
                         if parameter.requires_grad)
        return parameter.sum() * 0.0

    def _forward_loss(self, batch: Dict[str, torch.Tensor],
                      task_index: int) -> torch.Tensor:
        outputs = self.model(**batch, use_cache=False)
        loss = outputs.loss + self.additional_loss(task_index)
        if self.method == "loramoe":
            router_loss = collect_loramoe_loss(self.raw_model)
            if router_loss is not None:
                loss = loss + router_loss
        return loss

    def _reset_task_stats(self) -> None:
        self._task_stats = {
            "train_tokens": 0,
            "train_sequences": 0,
            "train_micro_batches": 0,
            "optimizer_steps": 0,
        }

    def _record_compute_batch(self, batch: Dict[str, torch.Tensor],
                              phase: str) -> None:
        if phase != "train":
            raise ValueError(f"unknown compute phase: {phase}")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            tokens = int(attention_mask.detach().sum().item())
            sequences = int(attention_mask.shape[0])
        elif input_ids is not None:
            tokens = int(input_ids.numel())
            sequences = int(input_ids.shape[0])
        else:
            raise KeyError("batch has neither attention_mask nor input_ids")
        self._task_stats[f"{phase}_tokens"] += tokens
        self._task_stats[f"{phase}_sequences"] += sequences
        self._task_stats[f"{phase}_micro_batches"] += 1

    def _distributed_sum(self, value: int) -> int:
        tensor = torch.tensor(float(value), device=_device(self.args),
                              dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def _distributed_max(self, value: float) -> float:
        tensor = torch.tensor(value, device=_device(self.args),
                              dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def _task_result(self, task: str, task_index: int, epochs: int,
                     started_at: str, elapsed_local: float,
                     report: Dict[str, int]) -> Dict[str, object]:
        elapsed = self._distributed_max(elapsed_local)
        phase_stats = {}
        total_tokens = 0
        total_sequences = 0
        for phase in ("train",):
            tokens = self._distributed_sum(
                self._task_stats[f"{phase}_tokens"])
            sequences = self._distributed_sum(
                self._task_stats[f"{phase}_sequences"])
            micro_batches = self._distributed_sum(
                self._task_stats[f"{phase}_micro_batches"])
            phase_stats[phase] = {
                "tokens": tokens,
                "sequences": sequences,
                "global_micro_batches": micro_batches,
            }
            total_tokens += tokens
            total_sequences += sequences
        effective_parameters = int(report["activated_non_embedding"])
        checkpointed = task in getattr(self.args, "ckpt_tasks", set())
        estimated_flops = 0
        for phase, values in phase_stats.items():
            # Standard transformer training is ~6ND. Activation checkpointing
            # adds one ~2ND forward recomputation during backward.
            factor = 8 if checkpointed else 6
            phase_flops = factor * values["tokens"] * effective_parameters
            values["flop_factor"] = factor
            values["estimated_flops"] = phase_flops
            estimated_flops += phase_flops
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return {
            "task_index": task_index,
            "task": task,
            "epochs": epochs,
            "per_device_batch_size": self.args.batch_by_task[task],
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_seconds": elapsed,
            "gpu_seconds": elapsed * world_size,
            "optimizer_steps_per_rank": self._task_stats["optimizer_steps"],
            "tokens": total_tokens,
            "sequences": total_sequences,
            "phases": phase_stats,
            "parameter_report": report,
            "gradient_checkpointing": checkpointed,
            "estimated_flops": estimated_flops,
            "estimated_tflops": estimated_flops / 1.0e12,
            "estimated_tflops_per_second_per_gpu": (
                estimated_flops / max(elapsed * world_size, 1.0e-12) / 1.0e12),
            "checkpoint": os.path.join(self.args.output_dir, str(task_index)),
        }

    def _write_result(self) -> None:
        if self.args.global_rank != 0:
            return
        training_seconds = sum(float(item["training_seconds"])
                               for item in self.task_results)
        total_flops = sum(int(item["estimated_flops"])
                          for item in self.task_results)
        total_tokens = sum(int(item["tokens"]) for item in self.task_results)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        complete = len(self.task_results) == len(self.train_task_list)
        payload = {
            "schema_version": 1,
            "status": "completed" if complete else "running",
            "method": self.method,
            "model_name_or_path": self.args.model_name_or_path,
            "world_size": world_size,
            "dtype": "bfloat16",
            "tasks_requested": list(self.train_task_list.keys()),
            "flop_estimator": {
                "name": "transformer_6ND_or_checkpointed_8ND_active_non_embedding",
                "formula": "6ND normally; 8ND for activation-checkpointed passes (extra 2ND forward recompute)",
                "scope": "forward+backward token compute; EWC reuses training gradients; canonical GEM adds episodic-memory forward/backward passes that are reported separately; excludes optimizer, collectives, checkpoint I/O, and regularizer/QP elementwise ops",
                "is_estimate": True,
            },
            "config": {
                "epochs": list(self.args.num_train_epochs),
                "per_device_batch_size": self.args.batch_by_task,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "gradient_checkpointing_tasks": sorted(self.args.ckpt_tasks),
                "learning_rate": self.args.learning_rate,
                "lora_rank": self.args.lora_rank,
                "lora_alpha": self.args.lora_alpha,
                "max_train_steps_per_task": self.args.max_train_steps_per_task,
            },
            "tasks": self.task_results,
            "totals": {
                "training_seconds": training_seconds,
                "gpu_seconds": training_seconds * world_size,
                "process_wall_seconds": time.monotonic() - self._process_started,
                "tokens": total_tokens,
                "estimated_flops": total_flops,
                "estimated_tflops": total_flops / 1.0e12,
                "estimated_tflops_per_second_per_gpu": (
                    total_flops /
                    max(training_seconds * world_size, 1.0e-12) / 1.0e12),
            },
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, "result.json")
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)

    def train_one_task(self, task, task_index, epochs):
        dataloader = self.train_task_list[task]
        steps_per_epoch = len(dataloader)
        if self.args.max_train_steps_per_task > 0:
            steps_per_epoch = min(
                steps_per_epoch, self.args.max_train_steps_per_task)
        progress = tqdm(total=epochs * steps_per_epoch, leave=True,
                        disable=self.args.global_rank != 0)
        device = _device(self.args)
        for epoch in range(epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.model.train()
            for step, source_batch in enumerate(dataloader):
                if (self.args.max_train_steps_per_task > 0 and
                        step >= self.args.max_train_steps_per_task):
                    break
                batch = dict(source_batch)
                batch.pop("sources", None)
                batch = to_device(batch, device)
                self._record_compute_batch(batch, "train")
                accum = max(1, self.args.gradient_accumulation_steps)
                window_start = (step // accum) * accum
                window_size = min(accum, steps_per_epoch - window_start)
                accum_index = step - window_start
                should_step = accum_index + 1 == window_size
                sync = (self.model.no_sync()
                        if isinstance(self.model, DDP) and not should_step
                        else nullcontext())
                if self.method == "loramoe":
                    set_router_token_mask(
                        self.raw_model, batch.get("attention_mask"))
                try:
                    with sync:
                        loss = self._forward_loss(batch, task_index)
                        self._optimizer_step(loss, accum_index, window_size)
                finally:
                    # Keep this installed through backward: checkpointing reruns
                    # the wrapped FFN during backward.
                    if self.method == "loramoe":
                        set_router_token_mask(self.raw_model, None)
                if should_step:
                    self._task_stats["optimizer_steps"] += 1
                progress.update(1)
                if (self.args.global_rank == 0 and
                        (step % self.args.loss_log_interval == 0 or
                         step + 1 == len(dataloader))):
                    progress.set_description(
                        f"{self.method}:{task} e{epoch + 1} s{step} "
                        f"loss={loss.detach().float().item():.4f}", refresh=False)

    def train_continual(self):
        for task_index, task in enumerate(self.train_task_list):
            if task_index < getattr(self.args, "start_task", 0):
                continue
            self._reset_task_stats()
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.synchronize(_device(self.args))
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            self.before_task(task, task_index)
            report = parameter_report(self.raw_model, self.method)
            print_rank_0(f"{self.method} task={task} parameters={report}",
                         self.args.global_rank)
            epochs = int(self.args.num_train_epochs[task_index])
            self.train_one_task(task, task_index, epochs)
            self.after_task(task, task_index)
            torch.cuda.synchronize(_device(self.args))
            elapsed = time.monotonic() - started
            self.save_model(task_index)
            result = self._task_result(
                task, task_index, epochs, started_at, elapsed, report)
            self.task_results.append(result)
            self._write_result()
            print_rank_0(
                f"{self.method} task={task} result: "
                f"time={result['training_seconds']:.2f}s "
                f"tokens={result['tokens']} "
                f"TFLOPs={result['estimated_tflops']:.3f}",
                self.args.global_rank)

    def _meta_extra(self) -> Dict[str, object]:
        return {"current_task": self.current_task}

    def _trainer_state(self) -> Optional[Dict[str, object]]:
        return None

    def _load_trainer_state(self, state: Dict[str, object]) -> None:
        if state:
            raise ValueError(f"unexpected trainer state for {self.method}")

    def load_resume_state(self, checkpoint_dir: str) -> None:
        """Restore method state and prior task metrics after model weights load."""
        checkpoint_dir = os.path.normpath(checkpoint_dir)
        checkpoint_task = int(os.path.basename(checkpoint_dir))
        result_path = os.path.join(os.path.dirname(checkpoint_dir), "result.json")
        with open(result_path) as handle:
            result = json.load(handle)
        if result.get("method") != self.method:
            raise ValueError(
                f"resume result method={result.get('method')} != {self.method}")
        prior = result.get("tasks", [])
        indices = [item.get("task_index") for item in prior]
        expected = list(range(checkpoint_task + 1))
        if indices != expected:
            raise ValueError(
                f"resume tasks are not contiguous through {checkpoint_task}: {indices}")
        requested = list(self.train_task_list.keys())
        if result.get("tasks_requested") != requested:
            raise ValueError("resume task order differs from the requested task order")
        if checkpoint_task >= len(requested):
            raise ValueError(f"resume task index out of range: {checkpoint_task}")
        meta_path = os.path.join(checkpoint_dir, "paper_baseline_meta.json")
        with open(meta_path) as handle:
            meta = json.load(handle)
        if meta.get("method") != self.method:
            raise ValueError("resume checkpoint method mismatch")
        state_path = os.path.join(checkpoint_dir, "paper_baseline_state.pt")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            if state.get("schema_version") == 2:
                self._load_trainer_state(state.get("method_state", {}))
                self.optimizer.load_state_dict(state["optimizer"])
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
                for optimizer_state in self.optimizer.state.values():
                    for key, value in optimizer_state.items():
                        if torch.is_tensor(value):
                            optimizer_state[key] = value.to(_device(self.args))
            else:
                # Backward compatibility with the original EWC/GEM-only state.
                self._load_trainer_state(state)
        elif self.method in {"ewc", "gem"}:
            raise FileNotFoundError(
                f"{self.method} resume requires trainer state: {state_path}")
        else:
            print_rank_0(
                "Resume checkpoint predates optimizer-state saving; continuing "
                "with restored model weights and a fresh optimizer.",
                self.args.global_rank)
        self.current_task = checkpoint_task
        self.task_results = prior
        self.args.start_task = checkpoint_task + 1
        print_rank_0(
            f"Loaded {len(prior)} completed task results; next task index "
            f"is {self.args.start_task}", self.args.global_rank)

    def save_model(self, round):
        super().save_model(round)
        if self.args.global_rank != 0:
            return
        output_dir = os.path.join(self.args.output_dir, str(round))
        save_paper_baseline_meta(
            self.raw_model, output_dir, self.method,
            self.args.lora_rank, self.args.lora_alpha,
            self.args.lora_dropout, **self._meta_extra())
        trainer_state = {
            "schema_version": 2,
            "method_state": self._trainer_state() or {},
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(trainer_state,
                   os.path.join(output_dir, "paper_baseline_state.pt"))


class SeqLoRA(PaperBaseline):
    method = "seqlora"


class LoRAMoE(PaperBaseline):
    method = "loramoe"

    def _meta_extra(self):
        return {
            "current_task": self.current_task,
            "num_experts": self.args.loramoe_num_experts,
            "top_k": self.args.top_k,
            "routing_weight_mode": self.args.routing_weight_mode,
            "aux_loss_coeff": self.args.moe_aux_loss_coeff,
            "z_loss_coeff": self.args.moe_z_loss_coeff,
        }


class EWCLoRA(PaperBaseline):
    """TRACE EWC: online squared-gradient Fisher over SeqLoRA parameters."""

    method = "ewc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fisher = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in trainable_named_parameters(self.raw_model)}
        self.previous_params = {
            name: parameter.detach().float().clone()
            for name, parameter in trainable_named_parameters(self.raw_model)}

    def additional_loss(self, task_index: int) -> torch.Tensor:
        if task_index == 0:
            return super().additional_loss(task_index)
        named = dict(trainable_named_parameters(self.raw_model))
        penalty = None
        for name, parameter in named.items():
            term = self.fisher[name] * (
                parameter.float() - self.previous_params[name]).square()
            penalty = term.sum() if penalty is None else penalty + term.sum()
        return 0.5 * self.args.ewc_lambda * penalty

    def train_one_task(self, task, task_index, epochs):
        dataloader = self.train_task_list[task]
        steps_per_epoch = len(dataloader)
        if self.args.max_train_steps_per_task > 0:
            steps_per_epoch = min(steps_per_epoch,
                                  self.args.max_train_steps_per_task)
        progress = tqdm(total=epochs * steps_per_epoch, leave=True,
                        disable=self.args.global_rank != 0)
        named = trainable_named_parameters(self.raw_model)
        fisher_denominator = float(max(steps_per_epoch, 1))
        for epoch in range(epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.model.train()
            for step, source_batch in enumerate(dataloader):
                if (self.args.max_train_steps_per_task > 0 and
                        step >= self.args.max_train_steps_per_task):
                    break
                batch = dict(source_batch)
                batch.pop("sources", None)
                batch = to_device(batch, _device(self.args))
                self._record_compute_batch(batch, "train")
                loss = self._forward_loss(batch, task_index)
                loss.backward()
                # TRACE/EWC.py updates the running Fisher after every train
                # backward as grad^2 / len(task_loader), without resetting it
                # between tasks.
                for name, parameter in named:
                    if parameter.grad is not None:
                        stable_gradient = torch.nan_to_num(
                            parameter.grad.detach().float(), nan=0.0)
                        self.fisher[name].add_(
                            stable_gradient.square(),
                            alpha=1.0 / fisher_denominator)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in named], 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._task_stats["optimizer_steps"] += 1
                progress.update(1)
                if (self.args.global_rank == 0 and
                        (step % self.args.loss_log_interval == 0 or
                         step + 1 == len(dataloader))):
                    progress.set_description(
                        f"ewc:{task} e{epoch + 1} s{step} "
                        f"loss={loss.detach().float().item():.4f}", refresh=False)

    def after_task(self, task: str, task_index: int) -> None:
        self.previous_params = {
            name: parameter.detach().float().clone()
            for name, parameter in trainable_named_parameters(self.raw_model)}

    def _meta_extra(self):
        return {"current_task": self.current_task,
                "ewc_lambda": self.args.ewc_lambda,
                "fisher_update": "online_train_gradient_trace"}

    def _trainer_state(self):
        return {
            "fisher": {name: value.cpu() for name, value in self.fisher.items()},
            "previous_params": {
                name: value.cpu() for name, value in self.previous_params.items()},
        }

    def _load_trainer_state(self, state):
        device = _device(self.args)
        self.fisher = {name: value.to(device)
                       for name, value in state["fisher"].items()}
        self.previous_params = {
            name: value.to(device)
            for name, value in state["previous_params"].items()}


class GEMLoRA(PaperBaseline):
    """Canonical GEM restricted to the shared SeqLoRA parameters.

    Each completed task retains a small episodic sample set. At every optimizer
    boundary, gradients of those examples are recomputed at the current model
    parameters, concatenated across all LoRA tensors, and used as the constraints
    of one global GEM projection.
    """

    method = "gem"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.gem_memory_size < 1:
            raise ValueError("gem_memory_size must be positive")
        if self.args.gem_memory_batch_size < 1:
            raise ValueError("gem_memory_batch_size must be positive")
        self.episodic_memory: Dict[int, List[Dict[str, object]]] = {}
        self.gem_update_step = 0

    @staticmethod
    def _batch_size(batch: Dict[str, torch.Tensor]) -> int:
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim:
                return int(value.shape[0])
        raise ValueError("GEM memory batch has no batched tensor")

    def _build_task_memory(self, task: str, task_index: int) -> None:
        """Select deterministic raw examples for the task's episodic buffer."""
        dataset = self.train_task_list[task].dataset
        count = min(int(self.args.gem_memory_size), len(dataset))
        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed) + task_index)
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        self.episodic_memory[task_index] = [dataset[index] for index in indices]

    def _sample_memory_batches(
            self, task_index: int) -> List[Dict[str, torch.Tensor]]:
        """Sample one global memory minibatch and shard it across DDP ranks."""
        memory = self.episodic_memory[task_index]
        sample_count = min(int(self.args.gem_memory_batch_size), len(memory))
        generator = torch.Generator()
        # Every rank constructs the same permutation, then takes a disjoint shard.
        generator.manual_seed(
            int(self.args.seed) + 1_000_003 * self.gem_update_step
            + 10_007 * task_index)
        selected = torch.randperm(
            len(memory), generator=generator)[:sample_count].tolist()
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_indices = selected[rank::world]
        if not local_indices:
            return []
        examples = [memory[index] for index in local_indices]
        loader = list(self.train_task_list.values())[task_index]
        batch = dict(loader.collate_fn(examples))
        batch.pop("sources", None)
        return [batch]

    def _memory_gradient(
            self, batches: Sequence[Dict[str, torch.Tensor]],
            parameters: Sequence[torch.nn.Parameter],
            device: torch.device) -> torch.Tensor:
        """Mean episodic-loss gradient at the current parameter values."""
        total_gradient = torch.zeros(
            sum(parameter.numel() for parameter in parameters),
            device=device, dtype=torch.float32)
        total_examples = torch.zeros((), device=device, dtype=torch.float64)
        for cpu_batch in batches:
            batch = to_device(cpu_batch, device)
            size = self._batch_size(batch)
            loss = self.raw_model(**batch, use_cache=False).loss
            gradients = torch.autograd.grad(
                loss, parameters, allow_unused=True, retain_graph=False)
            total_gradient.add_(
                _flatten_autograd(gradients, parameters), alpha=float(size))
            total_examples.add_(size)
        # Each rank owns a disjoint shard of the sampled global memory batch.
        # Sum gradients/counts so every rank receives the same global mean.
        if dist.is_initialized():
            dist.all_reduce(total_gradient, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)
        if total_examples.item() == 0:
            raise RuntimeError("empty GEM episodic memory")
        return total_gradient / total_examples.float()

    def train_one_task(self, task, task_index, epochs):
        dataloader = self.train_task_list[task]
        steps_per_epoch = len(dataloader)
        if self.args.max_train_steps_per_task > 0:
            steps_per_epoch = min(
                steps_per_epoch, self.args.max_train_steps_per_task)
        progress = tqdm(total=epochs * steps_per_epoch, leave=True,
                        disable=self.args.global_rank != 0)
        named = trainable_named_parameters(self.raw_model)
        parameters = [parameter for _, parameter in named]
        device = _device(self.args)
        for epoch in range(epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.model.train()
            for step, source_batch in enumerate(dataloader):
                if (self.args.max_train_steps_per_task > 0 and
                        step >= self.args.max_train_steps_per_task):
                    break
                batch = dict(source_batch)
                batch.pop("sources", None)
                batch = to_device(batch, device)
                self._record_compute_batch(batch, "train")
                accum = max(1, self.args.gradient_accumulation_steps)
                window_start = (step // accum) * accum
                window_size = min(accum, steps_per_epoch - window_start)
                accum_index = step - window_start
                should_step = accum_index + 1 == window_size
                sync = (self.model.no_sync()
                        if isinstance(self.model, DDP) and not should_step
                        else nullcontext())
                with sync:
                    loss = self.model(**batch, use_cache=False).loss
                    (loss / window_size).backward()
                if should_step:
                    current = _flatten_gradients(parameters)
                    memories = [
                        self._memory_gradient(
                            self._sample_memory_batches(past), parameters, device)
                        for past in sorted(self.episodic_memory)
                    ]
                    if memories and any(
                            torch.dot(current, memory) < 0
                            for memory in memories):
                        projected = project_gem_gradient(
                            current, memories, self.args.gem_margin,
                            self.args.gem_qp_eps)
                        _write_flat_gradient(parameters, projected)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.gem_update_step += 1
                    self._task_stats["optimizer_steps"] += 1
                progress.update(1)
                if (self.args.global_rank == 0 and
                        (step % self.args.loss_log_interval == 0 or
                         step + 1 == steps_per_epoch)):
                    progress.set_description(
                        f"gem:{task} e{epoch + 1} s{step} "
                        f"loss={loss.detach().float().item():.4f}", refresh=False)
        progress.close()

    def after_task(self, task: str, task_index: int) -> None:
        self._build_task_memory(task, task_index)

    def _meta_extra(self):
        return {"current_task": self.current_task,
                "episodic_tasks": len(self.episodic_memory),
                "memory_examples_per_task": self.args.gem_memory_size,
                "memory_batch_size_global": self.args.gem_memory_batch_size,
                "memory_sampling": "deterministic_without_replacement_ddp_sharded",
                "projection_scope": "global_trainable_lora_vector",
                "margin": self.args.gem_margin}

    def _trainer_state(self):
        return {"episodic_memory": self.episodic_memory,
                "gem_update_step": self.gem_update_step}

    def _load_trainer_state(self, state):
        if "episodic_memory" not in state:
            raise ValueError(
                "legacy stored-gradient GEM checkpoints cannot resume canonical GEM")
        self.episodic_memory = {
            int(task): list(examples)
            for task, examples in state["episodic_memory"].items()}
        self.gem_update_step = int(state.get("gem_update_step", 0))


class OLoRA(PaperBaseline):
    method = "olora"

    def before_task(self, task: str, task_index: int) -> None:
        set_olora_task(self.raw_model, task_index)
        self.current_task = task_index
        self._set_gradient_checkpointing(task)
        # DDP captured the previous task's requires_grad mask. Rebuild both the
        # reducer and optimizer so the newly active adapter is synchronized.
        self._wrap_distributed_model()
        self._build_optimizer(task_index=task_index)

    def additional_loss(self, task_index: int) -> torch.Tensor:
        orthogonal, l2 = collect_olora_regularization(self.raw_model)
        return (self.args.olora_lambda_orthogonal * orthogonal +
                self.args.olora_lambda_l2 * l2)

    def _meta_extra(self):
        return {"current_task": self.current_task,
                "num_tasks": len(self.train_task_list),
                "lambda_orthogonal": self.args.olora_lambda_orthogonal,
                "lambda_l2": self.args.olora_lambda_l2}
