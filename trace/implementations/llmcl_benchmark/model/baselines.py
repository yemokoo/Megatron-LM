"""Baseline trainers for the TRACE continual-learning comparison.

All baselines share one shape: a FIXED model (no per-task growth, no phase-2 router
retune) trained sequentially over the 8 tasks with a SINGLE model wrapper built up
front. What differs is only how the model is prepared (done in the entry point):

  * plain finetune  : the whole model is trainable (no experts attached).
  * static_lora_moe : attach_lora_moe -> add_experts(N) once -> freeze backbone,
                      train the N experts + router.  (Qwen LoRAMoE / SeqLoRA baseline)
  * static_moe_ffn  : attach_growing_moe -> add_experts(N) once -> freeze backbone
                      + original gate, train the N added experts + appended rows. (OLMoE
                      "pre-add 8" baseline)

Contrast with Ours_* (task-wise growth + phase-2): here experts are all added ONCE,
there is no add_experts during training, no _reinit_engine, and no replay retune.
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from transformers import get_constant_schedule_with_warmup

from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_optimizer_grouped_parameters


class StaticBaseline(CL_Base_Model):
    """Fixed-model sequential CL baseline. One wrapper, no growth, no phase-2.

    model must already be in its final trainable state (full-finetune, or attached +
    experts added + backbone frozen). moe_loss_fn (optional) adds the router aux/z
    loss each step so static MoE baselines train under the same objective as Ours,
    minus the growth/phase-2. meta_saver (optional) writes the checkpoint's *_meta.json
    so the custom loader can rebuild it at eval time.

    Uses plain DistributedDataParallel + AdamW rather than DeepSpeed -- see
    Ours_LoRA_MoE for why (repeated deepspeed.initialize() on the same model leaks
    GPU memory without bound). This class only builds its wrapper once (no per-phase
    growth), so it never hit that leak, but is kept consistent with the other
    trainers / base_model.py's _optimizer_step.
    """

    def __init__(self, model, tokenizer, train_task_list, eval_task_list, test_task_list,
                 args, moe_loss_fn=None, meta_saver=None, router_mask_fn=None):
        super().__init__(model, tokenizer, None, train_task_list, eval_task_list, test_task_list, args)
        self.raw_model = model
        self.moe_loss_fn = moe_loss_fn
        self.meta_saver = meta_saver
        self.router_mask_fn = router_mask_fn
        self._init_engine()

    def _init_engine(self):
        args = self.args
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.raw_model, args.weight_decay)
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
        self.lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=args.num_warmup_steps)
        if args.local_rank != -1:
            self.model = DDP(self.raw_model, device_ids=[args.local_rank],
                             output_device=args.local_rank, find_unused_parameters=True,
                             broadcast_buffers=False, gradient_as_bucket_view=True)
        else:
            self.model = self.raw_model

    def train_one_task(self, task, i_task, epochs):
        device = torch.device("cuda", self.args.local_rank) if self.args.local_rank != -1 else torch.device("cuda")
        dataloader = self.train_task_list[task]
        total_steps = epochs * len(dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            print_rank_0(f"{task}: epoch {epoch+1}/{epochs}, {len(dataloader)} steps", self.args.global_rank)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                accum_steps = max(1, self.args.gradient_accumulation_steps)
                window_start = (step // accum_steps) * accum_steps
                window_size = min(accum_steps, len(dataloader) - window_start)
                accumulation_index = step - window_start
                should_step = accumulation_index + 1 == window_size
                sync_context = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) and not should_step
                    else nullcontext())
                if self.router_mask_fn is not None:
                    self.router_mask_fn(
                        self.raw_model, batch.get("attention_mask"))
                try:
                    with sync_context:
                        outputs = self.model(**batch, use_cache=False)
                        loss = outputs.loss
                        if self.moe_loss_fn is not None:
                            moe_loss = self.moe_loss_fn(self.raw_model)
                            if moe_loss is not None:
                                loss = loss + moe_loss
                        self._optimizer_step(
                            loss, accum_step=accumulation_index,
                            accum_steps=window_size)
                finally:
                    if self.router_mask_fn is not None:
                        self.router_mask_fn(self.raw_model, None)
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    if (step % self.args.loss_log_interval == 0 or
                            step + 1 == len(dataloader)):
                        progress_bar.set_description(
                            f"{task} | epoch {epoch+1} step {step} "
                            f"loss {loss.detach().float().cpu().item():.4f}",
                            refresh=False)

    def save_model(self, round):
        super().save_model(round)
        if self.meta_saver is not None and self.args.global_rank == 0:
            self.meta_saver(self.raw_model, os.path.join(self.args.output_dir, str(round)))
