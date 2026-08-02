#!/usr/bin/env python
"""CPU test for length-bucketed evaluation generation and order restoration."""

from types import SimpleNamespace
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from utils.eval_generation import generate_predictions


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __call__(self, prompts, padding, truncation, max_length):
        input_ids = []
        attention_mask = []
        for index, prompt in enumerate(prompts):
            length = min(len(prompt), max_length)
            token_id = 10 + index
            input_ids.append([token_id] * length)
            attention_mask.append([1] * length)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def pad(self, features, padding, return_tensors):
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask = [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append([self.pad_token_id] * pad + feature["input_ids"])
            attention_mask.append([0] * pad + feature["attention_mask"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def batch_decode(self, token_ids, skip_special_tokens):
        return [str(int(row[0])) for row in token_ids]

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id


class FakeModel:
    def __init__(self):
        self.generation_config = SimpleNamespace(eos_token_id=[1])
        self.padded_tokens = 0

    def generate(self, input_ids, attention_mask, **kwargs):
        self.padded_tokens += input_ids.numel()
        first_real_position = attention_mask.to(torch.int64).argmax(dim=1)
        rows = torch.arange(input_ids.shape[0])
        generated = input_ids[rows, first_real_position].unsqueeze(1)
        return torch.cat([input_ids, generated], dim=1)


def run(length_bucketing):
    prompts = ["a", "bbbbbbbb", "cc", "ddddddd"]
    model = FakeModel()
    predictions = generate_predictions(
        model,
        FakeTokenizer(),
        prompts,
        device=torch.device("cpu"),
        batch_size=2,
        max_prompt_len=16,
        max_new_tokens=1,
        length_bucketing=length_bucketing,
        description="cpu-order-test",
    )
    if predictions != ["10", "11", "12", "13"]:
        raise AssertionError(f"prediction order changed: {predictions}")
    return model.padded_tokens


def main():
    bucketed_tokens = run(length_bucketing=True)
    original_tokens = run(length_bucketing=False)
    if bucketed_tokens >= original_tokens:
        raise AssertionError(
            f"bucketing did not reduce padding: {bucketed_tokens} >= {original_tokens}")
    print(
        "EVAL_LENGTH_BUCKETING=PASS "
        f"padded_tokens={original_tokens}->{bucketed_tokens}")


if __name__ == "__main__":
    main()
