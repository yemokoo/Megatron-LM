"""Table-1 non-MoE baselines: Seq-LoRA, EWC, O-LoRA and MTL.

All four share one frozen backbone plus LoRA on a named target set, one
optimizer/DDP rebuild per phase, and the SLoRA training contract supplied by
``training/main_tab1.py``.  Only the continual-learning mechanism differs:

======  =====================================================================
seq     nothing; the shared adapter is simply carried across tasks
ewc     quadratic penalty against a post-convergence diagonal Fisher
olora   per-task rank block, orthogonality penalty against every frozen block,
        merged into the backbone after the final task
mtl     no continual learning at all -- one adapter, all eight tasks shuffled
        together, used as the joint-training ceiling
======  =====================================================================

The engine mirrors ``Ours_LoRA_MoE`` (plain autograd + DDP rebuilt per phase,
grad clip 1.0, epoch-local accumulation windows) so a Table-1 row differs from
Ours only in method, never in optimizer plumbing.
"""
from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from model.base_model import CL_Base_Model
from model.tab1_lora import (
    EWCState, TAB1_STATE_KEY_SUBSTRINGS, adapter_parameter_count,
    estimate_diagonal_fisher, merge_olora_into_base, olora_regularization,
    set_olora_task, snapshot_parameters)
from utils.utils import (get_optimizer_grouped_parameters, print_rank_0,
                         to_device)


class Tab1BaseTrainer(CL_Base_Model):
    """Shared engine for the LoRA-only Table-1 rows."""

    save_key_substrings = list(TAB1_STATE_KEY_SUBSTRINGS)
    method_name = "tab1"

    def __init__(self, model, tokenizer, optimizer, train_task_list,
                 eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list,
                         eval_task_list, test_task_list, args)
        self.raw_model = model
        self._dist_initialized = False

    # -- engine ------------------------------------------------------------
    def _reinit_engine(self, num_training_steps, learning_rate=None):
        args = self.args
        if self._dist_initialized:
            del self.model
            torch.cuda.empty_cache()
        grouped = get_optimizer_grouped_parameters(
            self.raw_model, args.weight_decay)
        self.optimizer = torch.optim.AdamW(
            grouped,
            lr=args.learning_rate if learning_rate is None else learning_rate,
            betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)
        warmup_steps = args.num_warmup_steps
        if getattr(args, "warmup_ratio", 0.0) > 0:
            warmup_steps = math.ceil(num_training_steps * args.warmup_ratio)
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler_type, self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, num_training_steps))
        if args.local_rank != -1:
            self.model = DDP(
                self.raw_model, device_ids=[args.local_rank],
                output_device=args.local_rank, find_unused_parameters=True,
                broadcast_buffers=False, gradient_as_bucket_view=True)
        else:
            self.model = self.raw_model
        self._dist_initialized = True

    def _set_grad_ckpt(self, enable):
        model = self.raw_model
        if getattr(model, "_require_grads_hook", None) is not None:
            model.disable_input_require_grads()
            model._require_grads_hook = None
        if enable:
            # Frozen backbone: the embedding output carries no grad_fn, so a
            # checkpointed segment would have nothing to recompute through.
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        print_rank_0(f"  [grad_ckpt] {'ON' if enable else 'off'}",
                     self.args.global_rank)

    def _optimizer_update_count(self, dataloader, epochs):
        accum = max(1, self.args.gradient_accumulation_steps)
        return max(1, int(epochs) * math.ceil(len(dataloader) / accum))

    def _set_phase_gradient_accumulation(self, batch_size, phase):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        denominator = int(batch_size) * world_size
        global_batch = int(self.args.effective_global_batch)
        if global_batch % denominator != 0:
            raise ValueError(
                f"{phase}: effective global batch {global_batch} is not "
                f"divisible by micro-batch {batch_size} x world {world_size}")
        self.args.gradient_accumulation_steps = global_batch // denominator
        print_rank_0(
            f"  [batch contract] {phase}: micro_batch={batch_size} "
            f"grad_accum={self.args.gradient_accumulation_steps} "
            f"effective_global_batch={global_batch}", self.args.global_rank)
        return self.args.gradient_accumulation_steps

    def extra_loss(self, batch, outputs):
        """Method-specific additive term; ``None`` means plain LM loss."""
        return None

    def _extra_loss_log(self):
        """Suffix appended to the progress line; methods override to add terms."""
        return ""

    def _run_epochs(self, dataloader, epochs, device, phase_name):
        args = self.args
        total = epochs * len(dataloader)
        bar = tqdm(total=total, leave=True, disable=(args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"{phase_name}: epoch {epoch+1}/{epochs}, "
                f"{len(dataloader)} steps", args.global_rank)
            self.model.train()
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            step_cap = int(getattr(args, "max_train_steps_per_task", 0))
            for step, batch in enumerate(dataloader):
                if step_cap and step >= step_cap:
                    break
                batch.pop("sources", None)
                batch = to_device(batch, device)
                accum = args.gradient_accumulation_steps
                window_start = (step // accum) * accum
                window_size = min(accum, len(dataloader) - window_start)
                is_window_end = (step - window_start + 1) == window_size
                sync = (self.model.no_sync()
                        if isinstance(self.model, DDP) and not is_window_end
                        else nullcontext())
                with sync:
                    outputs = self.model(**batch, use_cache=False)
                    extra = self.extra_loss(batch, outputs)
                    loss = (outputs.loss if extra is None
                            else outputs.loss + extra)
                    (loss / window_size).backward()
                if args.global_rank == 0:
                    bar.update(1)
                    if (step % args.loss_log_interval == 0
                            or step + 1 == len(dataloader)):
                        bar.set_description(
                            f"{phase_name} | epoch {epoch+1} step {step} "
                            f"loss {loss.detach().float().cpu().item():.4f}"
                            f"{self._extra_loss_log()}", refresh=False)
                if is_window_end:
                    trainable = [p for p in self.raw_model.parameters()
                                 if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

    # -- default per-task loop --------------------------------------------
    def before_task(self, task, i_task):
        pass

    def after_task(self, task, i_task):
        pass

    def train_one_task(self, task, i_task, epochs):
        args = self.args
        device = (torch.device("cuda", args.local_rank)
                  if args.local_rank != -1 else torch.device("cuda"))
        self.before_task(task, i_task)
        loader = self.train_task_list[task]
        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} train")
        self._set_grad_ckpt(task in getattr(args, "ckpt_tasks", set()))
        self._reinit_engine(self._optimizer_update_count(loader, epochs))
        self._run_epochs(loader, epochs, device, f"{task} [{self.method_name}]")
        self.after_task(task, i_task)

    def write_meta(self, round_index, **extra):
        if self.args.global_rank != 0:
            return
        directory = os.path.join(self.args.output_dir, str(round_index))
        os.makedirs(directory, exist_ok=True)
        meta = {
            "method": self.method_name,
            "targets": self.args.lora_targets,
            "r": self.args.lora_rank,
            "alpha": self.args.lora_alpha,
            "dropout": self.args.lora_dropout,
            "round": round_index,
            "parameters": adapter_parameter_count(self.raw_model),
        }
        meta.update(extra)
        with open(os.path.join(directory, "tab1_meta.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    def save_model(self, round):
        super().save_model(round)
        self.write_meta(round)


class SeqLoRATab1(Tab1BaseTrainer):
    """No continual-learning mechanism; the control every other row is read against."""
    method_name = "seq_lora"


class EWCTab1(Tab1BaseTrainer):
    """Elastic Weight Consolidation over one shared LoRA adapter.

    Deliberately NOT the TRACE implementation.  TRACE folds the Fisher update
    into the training loop, never resets or reweights it, and divides by a
    batch count; from task 2 on it therefore measures the curvature of its own
    regularised loss.  Here each task ends with a separate no-penalty pass that
    estimates the diagonal Fisher of the task likelihood at the converged
    parameters, which is what Kirkpatrick et al. define.
    """
    method_name = "ewc"

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        state_device = getattr(self.args, "ewc_state_device", "auto")
        self.state = EWCState(mode=self.args.ewc_mode,
                              device=None if state_device == "auto"
                              else state_device)
        self._last_penalty = None

    def extra_loss(self, batch, outputs):
        penalty = self.state.penalty(self.raw_model.named_parameters())
        if penalty is None:
            return None
        term = 0.5 * self.args.ewc_lambda * penalty
        # TRACE hardcodes lambda=400, but that number is calibrated against its
        # own Fisher, which accumulates grad^2 across every step of every task
        # without a reset and divides by a BATCH count. A per-sample Fisher
        # taken after convergence is smaller by roughly 5x per task, so the
        # same lambda is a much weaker constraint here. Log the term against
        # the LM loss so "EWC and Seq-LoRA scored the same" can be traced to a
        # penalty that never mattered rather than to a property of EWC.
        self._last_penalty = (float(penalty.detach()), float(term.detach()),
                              float(outputs.loss.detach()))
        return term

    def _extra_loss_log(self):
        if not self._last_penalty:
            return ""
        raw, term, lm = self._last_penalty
        share = term / lm if lm else float("nan")
        return (f" | ewc raw {raw:.3e} term {term:.3e} "
                f"({share:.1%} of LM)")

    def after_task(self, task, i_task):
        args = self.args
        device = (torch.device("cuda", args.local_rank)
                  if args.local_rank != -1 else torch.device("cuda"))
        # The Fisher pass must not see the penalty, so it runs on raw_model
        # with the extra term bypassed entirely rather than on self.model.
        print_rank_0(f"  [ewc] estimating Fisher for {task}", args.global_rank)
        # estimate_diagonal_fisher() calls model.eval(), and HF's Llama gates
        # gradient checkpointing on `self.gradient_checkpointing and
        # self.training` -- so this pass runs UNCHECKPOINTED regardless of
        # the training micro-batch, and OOMs at a batch size training itself
        # handles fine. Give it its own much smaller batch on the same
        # dataset/collate_fn/sampler rather than reusing the training loader.
        train_loader = self.train_task_list[task]
        fisher_loader = DataLoader(
            train_loader.dataset, collate_fn=train_loader.collate_fn,
            sampler=train_loader.sampler,
            batch_size=args.ewc_fisher_batch_size)
        fisher = estimate_diagonal_fisher(
            self.raw_model, fisher_loader, device,
            max_samples=args.ewc_fisher_samples,
            log_fn=lambda message: print_rank_0(message, args.global_rank))
        anchor = snapshot_parameters(self.raw_model)
        self.state.absorb(fisher, anchor)
        print_rank_0(f"  [ewc] {self.state.state_summary()}", args.global_rank)

    def save_model(self, round):
        Tab1BaseTrainer.save_model(self, round)
        if self.args.global_rank == 0:
            self.write_meta(round, ewc=self.state.state_summary(),
                            ewc_lambda=self.args.ewc_lambda)


class OLoRATab1(Tab1BaseTrainer):
    """Official O-LoRA: one new rank block per task, orthogonal to the frozen ones.

    Loss shape follows ``upstream/O-LoRA/src/uie_trainer_lora.py``:
    ``loss + l1 * sum|A_prev A_new^T| + l2 * sum||theta_new||_2``.  Both
    penalties are the official ones (L1 on the A-side product, un-squared
    Frobenius on the new adapter); the corrected B/Frobenius-squared variant
    lives in TRACE-repro and is a different method, not a bug fix to apply here.

    Lambda schedule: official ``scripts/long.sh`` hand-tunes per position, but
    every position in the first eight uses ``l1=0.5`` (task 1 gets 0/0 because
    there is nothing to be orthogonal to) and ``l2`` is 0 except at three
    positions.  ``--olora_lambda_schedule`` takes a comma list if that exact
    schedule is wanted; the scalar defaults reproduce the eight-task region.
    """
    method_name = "olora"

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        self._l1_schedule = self._expand(self.args.olora_lambda_orthogonal,
                                         "olora_lambda_orthogonal")
        self._l2_schedule = self._expand(self.args.olora_lambda_l2,
                                         "olora_lambda_l2")
        self._task_index = 0
        self._last_reg = (0.0, 0.0)

    def _expand(self, value, name):
        tasks = len(self.train_task_list)
        values = [float(item) for item in str(value).split(",")]
        if len(values) == 1:
            values = values * tasks
        if len(values) != tasks:
            raise ValueError(
                f"--{name} needs 1 or {tasks} values, got {len(values)}")
        # Task 1 has no frozen block, so any orthogonality weight there is a
        # no-op; forcing it to zero keeps the logged schedule honest.
        values[0] = 0.0 if name == "olora_lambda_orthogonal" else values[0]
        return values

    def before_task(self, task, i_task):
        self._task_index = i_task
        set_olora_task(self.raw_model, i_task)
        print_rank_0(
            f"  [olora] task slot {i_task} active, "
            f"l1={self._l1_schedule[i_task]} l2={self._l2_schedule[i_task]}",
            self.args.global_rank)

    def extra_loss(self, batch, outputs):
        l1 = self._l1_schedule[self._task_index]
        l2 = self._l2_schedule[self._task_index]
        if l1 == 0.0 and l2 == 0.0:
            return None
        orthogonal, adapter_l2 = olora_regularization(self.raw_model)
        self._last_reg = (float(orthogonal.detach()),
                          float(adapter_l2.detach()))
        return l1 * orthogonal + l2 * adapter_l2

    def save_model(self, round):
        Tab1BaseTrainer.save_model(self, round)
        if self.args.global_rank == 0:
            self.write_meta(
                round, current_task=self._task_index,
                num_tasks=len(self.train_task_list),
                lambda_orthogonal=self._l1_schedule[self._task_index],
                lambda_l2=self._l2_schedule[self._task_index],
                last_regularization={"orthogonal": self._last_reg[0],
                                     "adapter_l2": self._last_reg[1]},
                merged=False)

    def train_continual(self):
        super().train_continual()
        if not self.args.olora_merge_at_end:
            return
        # After the last task the eight rank blocks fold into the frozen base
        # weights, which is what gives O-LoRA the same inference cost as the
        # untouched backbone.  Written beside the final round as a separate
        # directory so the unmerged checkpoint survives for inspection.
        merged = merge_olora_into_base(self.raw_model)
        print_rank_0(f"  [olora] merged {merged} projections into the backbone",
                     self.args.global_rank)
        final = len(self.train_task_list) - 1
        if self.args.global_rank == 0:
            directory = os.path.join(self.args.output_dir, f"{final}_merged")
            os.makedirs(directory, exist_ok=True)
            state = {k: v for k, v in self.raw_model.state_dict().items()
                     if ".adapters." not in k}
            torch.save(state, os.path.join(directory, "pytorch_model.bin"))
            self.raw_model.config.to_json_file(
                os.path.join(directory, "config.json"))
            self.tokenizer.save_pretrained(directory)
            with open(os.path.join(directory, "tab1_meta.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"method": "olora", "round": final, "merged": True,
                           "targets": self.args.lora_targets,
                           "r": self.args.lora_rank,
                           "alpha": self.args.lora_alpha,
                           "num_tasks": len(self.train_task_list)},
                          handle, indent=2)


class MTLTab1(Tab1BaseTrainer):
    """Joint multi-task training: the ceiling, not a continual-learning method.

    One adapter is trained on all eight task datasets shuffled together, so
    there is no task order and nothing to forget.  The default epoch count is
    chosen so the joint run consumes the same number of training samples as
    the sequential schedule (sum of 5,000 x per-task epochs = 200,000 = 5
    epochs over the 40,000-record union), which keeps it comparable on compute
    rather than only on final score.
    """
    method_name = "mtl"

    def train_continual(self):
        args = self.args
        device = (torch.device("cuda", args.local_rank)
                  if args.local_rank != -1 else torch.device("cuda"))
        loader = self.joint_loader
        epochs = args.mtl_epochs
        micro_batch = min(args.batch_by_task.values())
        self._set_phase_gradient_accumulation(micro_batch, "mtl joint")
        self._set_grad_ckpt(bool(getattr(args, "ckpt_tasks", set())))
        self._reinit_engine(self._optimizer_update_count(loader, epochs))
        self._run_epochs(loader, epochs, device, "joint [mtl]")
        # Saved under the last round index so the evaluator, which asks for the
        # checkpoint after task 8, finds an MTL model at the same path.
        self.save_model(len(self.train_task_list) - 1)

    def save_model(self, round):
        Tab1BaseTrainer.save_model(self, round)
        if self.args.global_rank == 0:
            self.write_meta(round, mtl_epochs=self.args.mtl_epochs,
                            joint_records=len(self.joint_loader.dataset))


TAB1_TRAINERS = {
    "seq_lora": SeqLoRATab1,
    "ewc": EWCTab1,
    "olora": OLoRATab1,
    "mtl": MTLTab1,
}
