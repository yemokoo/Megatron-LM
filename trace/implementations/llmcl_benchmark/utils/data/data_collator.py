"""TRACE collators, including the released-SLoRA training format."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollator:
    """Legacy TRACE decoder-only collator.

    Training uses answer-only labels and left padding.  The optional ICL
    prompt dependency is imported lazily to avoid TRACE's original circular
    import between ``inference.ICL`` and this module.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False
    demonstrations: Optional[Any] = None
    task: Optional[str] = None

    def __call__(self, batch, return_tensors=None):
        return self.decoder_call(batch, return_tensors or self.return_tensors)

    def tokenize(self, sentence, cutoff_len, add_bos_token=True,
                 add_eos_token=True):
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )
        if (len(result["input_ids"]) < cutoff_len and add_eos_token
                and self.tokenizer.eos_token_id is not None):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        if (len(result["input_ids"]) < cutoff_len and add_bos_token
                and self.tokenizer.bos_token_id is not None):
            result["input_ids"] = (
                [self.tokenizer.bos_token_id] + result["input_ids"])
            result["attention_mask"] = [1] + result["attention_mask"]
        result["labels"] = result["input_ids"].copy()
        return result

    def decoder_call(self, batch, return_tensors):
        del return_tensors
        sources = []
        ground_truths = []
        tokenized_sources = []
        label_lengths = []
        actual_max_len = 0
        limit_len = (
            self.max_prompt_len + self.max_ans_len
            if not self.inference else self.max_prompt_len)

        for instance in batch:
            instruction = instance["prompt"]
            label = instance["answer"]
            sources.append(instruction)
            ground_truths.append(label)
            if not self.inference:
                tokenized_label = self.tokenize(
                    label, limit_len, add_bos_token=False,
                    add_eos_token=True)
                tokenized_source = self.tokenize(
                    instruction + label, limit_len,
                    add_bos_token=True, add_eos_token=True)
                label_lengths.append(len(tokenized_label["input_ids"]))
            else:
                if self.demonstrations is not None:
                    from inference.ICL import Constrained_PROMPT, TASK_PROMT

                    task_prompt = TASK_PROMT[self.task]
                    if self.task != "MeetingBank":
                        task_prompt += Constrained_PROMPT
                    for demonstration in self.demonstrations:
                        if self.task == "Py150":
                            task_prompt += "Code:\n"
                        task_prompt += demonstration["prompt"]
                        task_prompt += demonstration["answer"] + "\n\n"
                    if self.task == "Py150":
                        task_prompt += "Code:\n"
                    if self.task != "Py150":
                        instruction = instruction[len(TASK_PROMT[self.task]):]
                    instruction = task_prompt + instruction
                tokenized_source = self.tokenize(
                    instruction, limit_len,
                    add_bos_token=True, add_eos_token=False)
            tokenized_sources.append(tokenized_source)
            actual_max_len = max(
                actual_max_len, len(tokenized_source["input_ids"]))

        multiple = self.pad_to_multiple_of or 1
        actual_pad_len = (
            (actual_max_len + multiple - 1) // multiple * multiple)
        for index, tokenized in enumerate(tokenized_sources):
            pad_len = actual_pad_len - len(tokenized["input_ids"])
            if sum(tokenized["attention_mask"]) != len(
                    tokenized["input_ids"]):
                raise ValueError("TRACE collator received a pre-masked sample")
            tokenized["input_ids"] = (
                [self.tokenizer.pad_token_id] * pad_len
                + tokenized["input_ids"])
            tokenized["attention_mask"] = (
                [0] * pad_len + tokenized["attention_mask"])
            if not self.inference:
                label_len = label_lengths[index]
                label_mask_len = actual_pad_len - label_len
                tokenized["labels"] = (
                    [self.label_pad_token_id] * label_mask_len
                    + tokenized["labels"][-label_len:])

        model_inputs = {
            "input_ids": torch.tensor([
                source["input_ids"] for source in tokenized_sources]),
            "attention_mask": torch.tensor([
                source["attention_mask"] for source in tokenized_sources]),
            "sources": sources,
        }
        if not self.inference:
            model_inputs["labels"] = torch.tensor([
                source["labels"] for source in tokenized_sources])
        else:
            model_inputs["gts"] = ground_truths
        return model_inputs


def _right_pad_full_labels(tokenizer, samples, sources, label_starts=None):
    """Right-pad token IDs and label either the full sequence or a suffix."""
    if tokenizer.pad_token_id is None:
        raise ValueError("SLoRA collator requires tokenizer.pad_token_id")
    if not samples:
        raise ValueError("cannot collate an empty batch")
    maximum = max(len(input_ids) for input_ids in samples)
    input_ids = torch.full(
        (len(samples), maximum), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    if label_starts is None:
        label_starts = [0] * len(samples)
    if len(label_starts) != len(samples):
        raise ValueError("label_starts must match the sample count")
    for row, values in enumerate(samples):
        length = len(values)
        if length:
            values = torch.as_tensor(values, dtype=torch.long)
            input_ids[row, :length] = values
            attention_mask[row, :length] = 1
            label_start = min(max(0, int(label_starts[row])), length)
            labels[row, label_start:length] = values[label_start:length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sources": sources,
    }


@dataclass
class SLoRATraceDataCollator:
    """Released-SLoRA-style chat formatting with full-sequence labels."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int
    system_prompt: str = "You are a helpful assistant."
    label_scope: str = "full"

    def _encode(self, instance):
        if self.label_scope not in {"full", "answer"}:
            raise ValueError(f"unknown SLoRA label scope: {self.label_scope}")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instance["prompt"]},
            {"role": "assistant", "content": instance["answer"]},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False)
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None)
        input_ids = list(encoded["input_ids"])
        eos = self.tokenizer.eos_token_id
        if (eos is not None and len(input_ids) < self.max_length
                and (not input_ids or input_ids[-1] != eos)):
            input_ids.append(eos)
        label_start = 0
        if self.label_scope == "answer":
            prefix_messages = messages[:-1]
            prefix_text = self.tokenizer.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=True)
            prefix = self.tokenizer(
                prefix_text, truncation=True, max_length=self.max_length,
                padding=False, return_tensors=None)
            label_start = min(len(prefix["input_ids"]), len(input_ids))
        return input_ids, label_start

    def __call__(self, batch, return_tensors=None):
        del return_tensors
        encoded = [self._encode(instance) for instance in batch]
        samples = [values for values, _ in encoded]
        label_starts = [start for _, start in encoded]
        return _right_pad_full_labels(
            self.tokenizer, samples,
            [instance["prompt"] for instance in batch], label_starts)


@dataclass
class PreTokenizedSLoRATraceDataCollator:
    """Dynamic right padding for cached SLoRA token IDs."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch, return_tensors=None):
        del return_tensors
        samples = [list(instance["input_ids"]) for instance in batch]
        sources = [
            instance.get("prompt", str(instance.get("source_index", "")))
            for instance in batch]
        return _right_pad_full_labels(self.tokenizer, samples, sources)
