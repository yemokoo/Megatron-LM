"""SFTTrainer that adds replay to every optimizer update, in one graph.

Mirrors the v3 joint-replay contract so the only remaining difference is
where the replay gradient lands:

  v3    loss = LM(primary) + 1.0 * LM(replay),  replay grad -> routers only
  here  loss = LM(primary) + 1.0 * LM(replay),  replay grad -> the whole LoRA

One replay micro-batch per primary micro-batch keeps the step count at the
released 313/epoch and the replay exposure at 5,000/epoch, so neither
optimizer steps nor replay volume is a confound.

Why one forward instead of two: DeepSpeed ZeRO-2 attaches a reduction hook
per parameter and asserts each one receives a single gradient per backward.
Two separate forwards reach the *same* LoRA weights -- v3 gets away with two
because its replay gradient is confined to routers -- and the second reduction
trips "parameter N has already been reduced".  Concatenating the two groups
into one batch gives one graph and one reduction, while splitting the logits
afterwards keeps the two token-mean losses exactly as they would have been.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trl import SFTTrainer


def _token_mean_lm_loss(logits, labels):
    """Standard causal-LM loss over one group's supervised tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class JointReplaySFTTrainer(SFTTrainer):
    def __init__(self, *args, replay_dataset=None, replay_coeff=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._replay_coeff = float(replay_coeff)
        self._replay_raw = replay_dataset
        self._replay_loader = None
        self._replay_iterator = None
        self._replay_loss_sum = 0.0
        self._replay_loss_count = 0
        pad = getattr(self.processing_class, "pad_token_id", None)
        if pad is None:
            raise ValueError("joint replay needs a tokenizer pad_token_id")
        self._pad_token_id = int(pad)

    # ---------------- replay stream ----------------

    def _ensure_replay_loader(self):
        if self._replay_loader is not None:
            return
        # Same tokenization, truncation and packing decisions the primary
        # dataset went through; a hand-rolled collator here would silently
        # change the replay sequences.
        prepared = self._prepare_dataset(
            self._replay_raw, self.processing_class, self.args,
            self.args.packing, None, "replay")
        loader = DataLoader(
            prepared,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        # accelerate shards this the way it shards the train loader, so the
        # global replay batch is world_size x micro, matching the primary.
        self._replay_loader = self.accelerator.prepare(loader)

    def _next_replay_batch(self):
        self._ensure_replay_loader()
        if self._replay_iterator is None:
            self._replay_iterator = iter(self._replay_loader)
        try:
            return next(self._replay_iterator)
        except StopIteration:
            self._replay_iterator = iter(self._replay_loader)
            return next(self._replay_iterator)

    # ---------------- one-graph joint batch ----------------

    def _pad_to(self, tensor, width, value):
        if tensor.size(1) == width:
            return tensor
        pad = tensor.new_full((tensor.size(0), width - tensor.size(1)), value)
        return torch.cat((tensor, pad), dim=1)

    def _concat_groups(self, primary, replay):
        keys = ("input_ids", "attention_mask", "labels")
        missing = [k for k in keys if k not in primary or k not in replay]
        if missing:
            raise KeyError(f"joint replay batch is missing {missing}")
        width = max(primary["input_ids"].size(1), replay["input_ids"].size(1))
        fill = {"input_ids": self._pad_token_id, "attention_mask": 0,
                "labels": -100}
        merged = {}
        for key in keys:
            left = self._pad_to(primary[key], width, fill[key])
            right = self._pad_to(
                replay[key].to(left.device), width, fill[key])
            merged[key] = torch.cat((left, right), dim=0)
        return merged, primary["input_ids"].size(0)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        if self._replay_coeff <= 0 or not model.training:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch)

        replay = self._next_replay_batch()
        merged, split = self._concat_groups(inputs, replay)
        outputs = model(
            input_ids=merged["input_ids"],
            attention_mask=merged["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits
        primary_loss = _token_mean_lm_loss(
            logits[:split], merged["labels"][:split])
        replay_loss = _token_mean_lm_loss(
            logits[split:], merged["labels"][split:])

        self._replay_loss_sum += float(replay_loss.detach())
        self._replay_loss_count += 1

        loss = primary_loss + self._replay_coeff * replay_loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self._replay_loss_count:
            logs["replay_loss"] = round(
                self._replay_loss_sum / self._replay_loss_count, 4)
            self._replay_loss_sum = 0.0
            self._replay_loss_count = 0
        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)
