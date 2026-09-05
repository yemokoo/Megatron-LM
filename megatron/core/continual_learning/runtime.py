"""Minimal training-loop hooks for the six continual-learning baselines."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from megatron.core import parallel_state

from .audit import expected_active_counts, write_audit
from .ewc import consolidate_equal_lambda, ewc_penalty
from .lora_adapter import ContinualLowRankAdapter
from .olora import adapter_slot_norms, orthogonality_penalty
from .parameter_scope import (
    apply_parameter_scope,
    frozen_checksum,
    iter_layers,
    iter_shards,
    named_parameters,
    outside_layer_scope_checksum,
    scoped_trainable_parameters,
)
from .slora import (
    denoise_merge_and_clear,
    initialize_fresh_adapter,
    reference_checksum,
    snapshot_reference,
)
from .state import load_sidecar, save_sidecar
from .trace_gem import project_tensor_gradient


_CONTROLLER = None


def _gradient_tensor(parameter):
    if hasattr(parameter, "main_grad") and parameter.main_grad is not None:
        return parameter.main_grad
    return parameter.grad


class ContinualController:
    def __init__(self, model, args, training_model=None):
        self.model = model
        # Keep the DDP/Float16 wrappers for pipeline forward/backward and grad
        # buffer lifecycle, while `self.model` remains the unwrapped model used
        # for stable parameter names and method state.
        self.training_model = training_model if training_model is not None else model
        self.args = args
        self.method = args.continual_method
        self.task = args.continual_task_name
        self.layer_start = int(args.continual_trainable_layer_start)
        self.layer_end = int(args.continual_trainable_layer_end)
        self.state = None
        self.fisher_sum = {}
        self.ewc_mean = {}
        self.gem_memories = []
        self.slora_reference = None
        self.slora_retained_ranks = None
        self._estimating_fisher = False
        self._slora_finalized = False
        self._terminal_gradient_saved = False
        self._initial_frozen_checksum = None
        self._penalty_calls = 0
        self._last_penalty = None

        if args.continual_state_load:
            expected = self.method
            if self.method in {"sequential_dense"}:
                expected = None
            self.state = load_sidecar(args.continual_state_load, expected_method=expected)
            payload = self.state["payload"]
            if self.method == "ewc":
                self.fisher_sum = payload["fisher_sum"]
                self.ewc_mean = payload["mean"]
            elif self.method == "trace_gem":
                self.gem_memories = payload["memories"]
            elif self.method == "slora_pre":
                self.slora_reference = payload["reference"]

        self._configure_adapters()
        scope = apply_parameter_scope(
            model, self.method, self.task, self.layer_start, self.layer_end
        )
        if self.method == "ewc" and self.task != "wiki":
            # Cache Fisher diagonals and anchors once on the parameter device.
            # Leaving them on CPU would copy the full regularizer state on every
            # microbatch (18 times per optimizer step in the reference setup).
            scoped = scoped_trainable_parameters(model, self.layer_start, self.layer_end)
            self.fisher_sum = {
                name: self.fisher_sum[name].to(device=parameter.device, dtype=torch.float32)
                for name, parameter in scoped.items()
            }
            self.ewc_mean = {
                name: self.ewc_mean[name].to(device=parameter.device, dtype=torch.float32)
                for name, parameter in scoped.items()
            }
        if self.method == "trace_gem" and self.task != "wiki":
            scoped = scoped_trainable_parameters(model, self.layer_start, self.layer_end)
            self.gem_memories = [
                {
                    name: memory[name].to(device=parameter.device, dtype=torch.float32)
                    for name, parameter in scoped.items()
                    if name in memory
                }
                for memory in self.gem_memories
            ]
        if self.task != "wiki":
            self._initial_frozen_checksum = self._protected_checksum()
        self.audit = {
            "method": self.method,
            "task": self.task,
            "layer_scope": [self.layer_start, self.layer_end],
            "parameter_scope": scope,
            "active_counts": expected_active_counts(
                hidden_size=args.hidden_size,
                dense_ffn=args.continual_dense_ffn_hidden_size,
                expert_ffn=352,
                top_k=4,
                olora_rank=args.continual_olora_rank,
            ),
            "old_model_kd_coefficient": float(getattr(args, "moe_old_model_kl_coeff", 0.0)),
        }
        if self.audit["old_model_kd_coefficient"] != 0.0:
            raise RuntimeError("Six-baseline runs forbid old-model KD")
        self._write_audit("setup.json")

    def _state_dir(self, method: Optional[str] = None) -> str:
        return os.path.join(self.args.save, f"continual_state_{method or self.method}")

    def _protected_checksum(self) -> str:
        if self.method == "slora_pre":
            return outside_layer_scope_checksum(self.model, self.layer_start, self.layer_end)
        return frozen_checksum(self.model)

    def _configure_adapters(self):
        fresh_stage = int(getattr(self.args, "iteration", 0)) == 0
        init_std = float(getattr(self.args, "init_method_std", 0.02))
        if self.method == "slora_pre":
            if self.task == "wiki":
                return
            if self.slora_reference is None:
                if self.task != "code":
                    raise RuntimeError("SLoRA Conversation requires the immutable Wiki reference state")
                self.slora_reference = snapshot_reference(
                    self.model, self.layer_start, self.layer_end
                )
            if fresh_stage:
                rank = (
                    int(self.args.continual_slora_rank)
                    if self.task == "code"
                    else int(self.args.continual_slora_conversation_rank)
                )
                initialize_fresh_adapter(
                    self.model, rank, self.layer_start, self.layer_end, init_std
                )
            else:
                rank = (
                    int(self.args.continual_slora_rank)
                    if self.task == "code"
                    else int(self.args.continual_slora_conversation_rank)
                )
                for shard in iter_shards(self.model):
                    for module in shard.modules():
                        if isinstance(module, ContinualLowRankAdapter):
                            module.set_active(rank)
        elif self.method == "olora":
            active_count = {"wiki": 1, "code": 2, "conversation": 3}[self.task]
            current = active_count - 1
            for layer_number, layer in iter_layers(self.model):
                if not self.layer_start <= layer_number <= self.layer_end:
                    continue
                for parent in layer.modules():
                    for collection_name in ("continual_q_adapters", "continual_v_adapters"):
                        adapters = getattr(parent, collection_name, ())
                        if len(adapters) != 3:
                            continue
                        for index, adapter in enumerate(adapters):
                            if index < active_count:
                                if fresh_stage and index == current:
                                    adapter.reset_active(self.args.continual_olora_rank, init_std)
                                else:
                                    adapter.set_active(self.args.continual_olora_rank)
                            else:
                                adapter.set_active(0)

    def _write_audit(self, filename: str):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        write_audit(os.path.join(self.args.save, "continual_audit", filename), self.audit)

    def loss_penalty(self):
        if self._estimating_fisher:
            return None
        if self.method == "ewc" and self.task != "wiki":
            parameters = scoped_trainable_parameters(
                self.model, self.layer_start, self.layer_end
            )
            penalty = ewc_penalty(
                parameters,
                self.fisher_sum,
                self.ewc_mean,
                self.args.continual_ewc_lambda,
            )
        elif self.method == "olora":
            penalty = orthogonality_penalty(
                self.model,
                self.task,
                self.args.continual_olora_orth_lambda,
                self.layer_start,
                self.layer_end,
            )
        else:
            return None
        self._penalty_calls += 1
        # Own the scalar storage: pipeline schedules are free to recycle
        # forward tensors after backward, while this audit value is read only
        # after the complete train/probe cycle.
        self._last_penalty = penalty.detach().clone()
        return penalty

    def before_optimizer_step(self, current_step: int):
        trace_enabled = self.method == "trace_gem" or bool(
            getattr(self.args, "continual_capture_trace_terminal", False)
        )
        if not trace_enabled:
            return
        parameters = scoped_trainable_parameters(self.model, self.layer_start, self.layer_end)
        if self.method == "trace_gem" and self.task != "wiki":
            for name, parameter in parameters.items():
                gradient = _gradient_tensor(parameter)
                if gradient is None:
                    continue
                memories = [memory[name] for memory in self.gem_memories if name in memory]
                if memories:
                    gradient.copy_(
                        project_tensor_gradient(
                            gradient,
                            memories,
                            margin=self.args.continual_trace_gem_margin,
                            eps=self.args.continual_trace_gem_eps,
                        )
                    )
        if current_step == int(self.args.train_iters):
            terminal = {}
            for name, parameter in parameters.items():
                gradient = _gradient_tensor(parameter)
                if gradient is not None:
                    terminal[name] = gradient.detach().to("cpu", torch.bfloat16).clone()
            memories = list(self.gem_memories)
            memories.append(terminal)
            self.gem_memories = memories
            self._save_trace_state()
            self._terminal_gradient_saved = True

    def after_optimizer_step(self, current_step: int, update_successful: bool):
        if not update_successful or current_step != int(self.args.train_iters):
            return
        if self.method == "slora_pre" and self.task != "wiki" and not self._slora_finalized:
            rank = (
                int(self.args.continual_slora_rank)
                if self.task == "code"
                else int(self.args.continual_slora_conversation_rank)
            )
            self.slora_retained_ranks = denoise_merge_and_clear(
                self.model,
                self.slora_reference,
                rank,
                self.layer_start,
                self.layer_end,
                self.args.continual_slora_denoise_mode,
            )
            self._slora_finalized = True
            self._save_slora_state()

    def _save_trace_state(self):
        serializable_memories = [
            {
                name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).clone()
                for name, tensor in memory.items()
            }
            for memory in self.gem_memories
        ]
        save_sidecar(
            self._state_dir("trace_gem"),
            "trace_gem",
            self.task,
            {
                "memories": serializable_memories,
                "semantics": "trace_terminal_gradient_per_parameter_corrected_qp_sign",
                "margin": float(self.args.continual_trace_gem_margin),
            },
        )

    def _save_slora_state(self):
        save_sidecar(
            self._state_dir("slora_pre"),
            "slora_pre",
            self.task,
            {
                "reference": self.slora_reference,
                "reference_checksum": reference_checksum(self.slora_reference),
                "retained_ranks": self.slora_retained_ranks,
                "code_rank": int(self.args.continual_slora_rank),
                "conversation_rank": int(self.args.continual_slora_conversation_rank),
                "alpha": float(self.args.continual_slora_alpha),
            },
        )

    def _estimate_fisher(self, optimizer, train_data_iterator, forward_step_func, config):
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        parameters = scoped_trainable_parameters(self.model, self.layer_start, self.layer_end)
        fisher = {name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32) for name, parameter in parameters.items()}
        batches = min(100, int(self.args.continual_ewc_fisher_batches))
        if batches <= 0:
            raise ValueError("EWC Fisher batch count must be positive")
        training_modes = [shard.training for shard in iter_shards(self.model)]
        for shard in iter_shards(self.model):
            shard.eval()
        self._estimating_fisher = True
        started = time.monotonic()
        try:
            forward_backward_func = get_forward_backward_func()
            for _ in range(batches):
                for shard in iter_shards(self.training_model):
                    shard.zero_grad_buffer()
                optimizer.zero_grad()
                forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=train_data_iterator,
                    model=self.training_model,
                    num_microbatches=1,
                    seq_length=self.args.seq_length,
                    micro_batch_size=self.args.micro_batch_size,
                    decoder_seq_length=self.args.decoder_seq_length,
                    forward_only=False,
                )
                for name, parameter in parameters.items():
                    gradient = _gradient_tensor(parameter)
                    if gradient is not None:
                        fisher[name].add_(gradient.detach().float().cpu().square(), alpha=1.0 / batches)
        finally:
            self._estimating_fisher = False
            for shard, was_training in zip(iter_shards(self.model), training_modes):
                shard.train(was_training)
            for shard in iter_shards(self.training_model):
                shard.zero_grad_buffer()
            optimizer.zero_grad()
        mean = {name: parameter.detach().float().cpu().clone() for name, parameter in parameters.items()}
        elapsed = time.monotonic() - started
        if self.fisher_sum:
            fisher, mean = consolidate_equal_lambda(
                self.fisher_sum, self.ewc_mean, fisher, mean
            )
        self.fisher_sum, self.ewc_mean = fisher, mean
        save_sidecar(
            self._state_dir("ewc"),
            "ewc",
            self.task,
            {
                "fisher_sum": fisher,
                "mean": mean,
                "fisher_batches": batches,
                "fisher_elapsed_seconds": elapsed,
                "fisher_definition": "task_boundary_diagonal_empirical_fisher_pure_nll",
            },
        )
        self.audit["ewc_fisher"] = {"batches": batches, "elapsed_seconds": elapsed}

    def on_train_end(self, optimizer, train_data_iterator, forward_step_func, config):
        if self.method == "ewc" and self.task != "conversation":
            self._estimate_fisher(optimizer, train_data_iterator, forward_step_func, config)
        if (
            self.method == "trace_gem"
            or bool(getattr(self.args, "continual_capture_trace_terminal", False))
        ) and not self._terminal_gradient_saved:
            raise RuntimeError("TRACE-GEM did not capture the deterministic terminal gradient")
        if self.method == "slora_pre" and self.task != "wiki" and not self._slora_finalized:
            raise RuntimeError("SLoRA stage ended without denoise-and-merge finalization")
        if self.task != "wiki":
            final_checksum = self._protected_checksum()
            self.audit["frozen_checksum_before"] = self._initial_frozen_checksum
            self.audit["frozen_checksum_after"] = final_checksum
            self.audit["frozen_bit_identical"] = final_checksum == self._initial_frozen_checksum
            if final_checksum != self._initial_frozen_checksum:
                raise RuntimeError("Frozen parameters changed during the continual stage")
        if (self.method == "ewc" and self.task != "wiki") or self.method == "olora":
            if self._penalty_calls == 0 or self._last_penalty is None:
                raise RuntimeError(f"{self.method} regularizer was never added to the training loss")
            self.audit["regularizer"] = {
                "calls": self._penalty_calls,
                "last_value": float(self._last_penalty.float().cpu().item()),
            }
        if self.method == "olora":
            self.audit["olora_adapter_norms"] = adapter_slot_norms(
                self.model, self.layer_start, self.layer_end
            )
            save_sidecar(
                self._state_dir("olora"),
                "olora",
                self.task,
                {
                    "active_adapter_count": {"wiki": 1, "code": 2, "conversation": 3}[self.task],
                    "rank": int(self.args.continual_olora_rank),
                    "alpha": float(self.args.continual_olora_alpha),
                    "orth_lambda": float(self.args.continual_olora_orth_lambda),
                },
            )
        self._write_audit("final.json")


def configure_continual_learning(model, args, training_model=None) -> bool:
    """Configure state and trainability; return whether optimizer rebuilding is required."""
    global _CONTROLLER
    if getattr(args, "continual_method", "none") == "none":
        _CONTROLLER = None
        return False
    _CONTROLLER = ContinualController(model, args, training_model=training_model)
    return True


def continual_loss_penalty():
    return None if _CONTROLLER is None else _CONTROLLER.loss_penalty()


def continual_before_optimizer_step(current_step: int):
    if _CONTROLLER is not None:
        _CONTROLLER.before_optimizer_step(current_step)


def continual_after_optimizer_step(current_step: int, update_successful: bool):
    if _CONTROLLER is not None:
        _CONTROLLER.after_optimizer_step(current_step, update_successful)


def continual_on_train_end(optimizer, train_data_iterator, forward_step_func, config):
    if _CONTROLLER is not None:
        _CONTROLLER.on_train_end(optimizer, train_data_iterator, forward_step_func, config)
