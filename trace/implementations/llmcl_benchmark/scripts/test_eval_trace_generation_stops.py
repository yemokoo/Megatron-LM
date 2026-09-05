#!/usr/bin/env python
"""CPU/mock checks for score-preserving TRACE generation marker stops."""

from types import SimpleNamespace
import os
import string
import sys

import torch
from transformers import StopStringCriteria

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from utils.eval_generation import (  # noqa: E402
    GeneratedOnlyStopStringCriteria,
    generate_predictions,
)


class CharacterTokenizer:
    """Small tokenizer implementing the API used by StopStringCriteria."""

    def __init__(self):
        symbols = list(string.ascii_letters + string.digits + string.punctuation + " \n\t")
        self._vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        for symbol in symbols:
            if symbol not in self._vocab:
                self._vocab[symbol] = len(self._vocab)
        self._tokens = {index: token for token, index in self._vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def get_vocab(self):
        return dict(self._vocab)

    def _convert_id_to_token(self, token_id):
        return self._tokens[int(token_id)]

    def convert_tokens_to_string(self, tokens):
        special = {"<pad>", "<eos>", "<unk>"}
        return "".join(token for token in tokens if token not in special)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def encode(self, text):
        return [self._vocab.get(character, self.unk_token_id) for character in text]

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=False,
    ):
        del padding, add_special_tokens
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        input_ids = [self.encode(text) for text in texts]
        if truncation and max_length is not None:
            input_ids = [ids[:max_length] for ids in input_ids]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if single:
            return {key: values[0] for key, values in result.items()}
        return result

    def pad(self, features, padding, return_tensors):
        del padding, return_tensors
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask = [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append([self.pad_token_id] * pad + feature["input_ids"])
            attention_mask.append([0] * pad + feature["attention_mask"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_decode(self, rows, skip_special_tokens):
        decoded = []
        for row in rows.tolist():
            tokens = []
            for token_id in row:
                if skip_special_tokens and token_id in (
                    self.pad_token_id, self.eos_token_id,
                ):
                    continue
                tokens.append(self._tokens.get(token_id, "?"))
            decoded.append(self.convert_tokens_to_string(tokens))
        return decoded


class MockAutoregressiveModel:
    """Minimal generate loop honoring row-wise Hugging Face stopping criteria."""

    def __init__(self, tokenizer, responses):
        self.tokenizer = tokenizer
        self.responses = {
            tokenizer.convert_tokens_to_ids(prompt_key): tokenizer.encode(response)
            for prompt_key, response in responses.items()
        }
        self.generation_config = SimpleNamespace(eos_token_id=[tokenizer.eos_token_id])
        self.decode_steps = 0
        self.received_stopping_criteria = False

    def generate(self, input_ids, attention_mask, **kwargs):
        stopping_criteria = kwargs.get("stopping_criteria")
        self.received_stopping_criteria = stopping_criteria is not None
        max_new_tokens = kwargs["max_new_tokens"]
        pad_token_id = kwargs["pad_token_id"]
        eos_token_ids = kwargs.get("eos_token_id", [self.tokenizer.eos_token_id])
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        rows = torch.arange(input_ids.shape[0])
        first_real = attention_mask.to(torch.int64).argmax(dim=1)
        prompt_keys = input_ids[rows, first_real].tolist()
        responses = [self.responses[key] for key in prompt_keys]
        sequences = input_ids.clone()
        unfinished = torch.ones(input_ids.shape[0], dtype=torch.bool)
        vocab_size = len(self.tokenizer.get_vocab())

        for step in range(max_new_tokens):
            self.decode_steps += 1
            next_tokens = []
            for response in responses:
                token = response[step] if step < len(response) else self.tokenizer.eos_token_id
                next_tokens.append(token)
            next_tokens = torch.tensor(next_tokens, dtype=torch.long)
            next_tokens = torch.where(
                unfinished, next_tokens, torch.full_like(next_tokens, pad_token_id))
            sequences = torch.cat([sequences, next_tokens[:, None]], dim=1)

            eos_finished = torch.zeros_like(unfinished)
            for token_id in eos_token_ids:
                eos_finished |= next_tokens == token_id
            finished = eos_finished
            if stopping_criteria is not None:
                scores = torch.empty((input_ids.shape[0], vocab_size))
                finished |= stopping_criteria(sequences, scores)
            unfinished &= ~finished
            if not torch.any(unfinished):
                break
        return sequences


def check_integration_and_score_equivalence():
    tokenizer = CharacterTokenizer()
    responses = {
        "A": "alpha<EOL>discarded code and explanation",
        "B": "beta<EOL>another discarded continuation",
    }
    common = dict(
        tokenizer=tokenizer,
        prompts=["A", "B"],
        device=torch.device("cpu"),
        batch_size=2,
        max_prompt_len=64,
        max_new_tokens=64,
        length_bucketing=True,
        description="cpu-trace-stop-test",
    )

    baseline_model = MockAutoregressiveModel(tokenizer, responses)
    baseline = generate_predictions(model=baseline_model, **common)
    if baseline_model.received_stopping_criteria:
        raise AssertionError("default generation unexpectedly enabled marker stops")

    optimized_model = MockAutoregressiveModel(tokenizer, responses)
    optimized = generate_predictions(
        model=optimized_model, stop_strings=["<EOL>"], **common)
    if not optimized_model.received_stopping_criteria:
        raise AssertionError("opt-in generation did not receive stopping criteria")
    if optimized_model.decode_steps >= baseline_model.decode_steps:
        raise AssertionError(
            "marker stop did not reduce decode steps: "
            f"{optimized_model.decode_steps} >= {baseline_model.decode_steps}")

    baseline_scored = [text.split("<EOL>", 1)[0] for text in baseline]
    optimized_scored = [text.split("<EOL>", 1)[0] for text in optimized]
    if optimized_scored != baseline_scored:
        raise AssertionError(
            f"post-hoc normalized predictions changed: {baseline_scored} -> "
            f"{optimized_scored}")
    if not all(text.endswith("<EOL>") for text in optimized):
        raise AssertionError(f"optimized output did not retain marker: {optimized}")

    return baseline_model.decode_steps, optimized_model.decode_steps


def check_generated_only_boundary_and_rows():
    tokenizer = CharacterTokenizer()
    inner = StopStringCriteria(tokenizer, ["Question:"])

    # Prompt + continuation spells the marker, but the continuation alone does
    # not. TRACE post-hoc normalization must not cut this case.
    prompt = torch.tensor([tokenizer.encode("Quest")], dtype=torch.long)
    continuation = torch.tensor([tokenizer.encode("ion:")], dtype=torch.long)
    combined = torch.cat([prompt, continuation], dim=1)
    generated_only = GeneratedOnlyStopStringCriteria(inner, prompt.shape[1])
    boundary_match = generated_only(combined, scores=None)
    if boundary_match.tolist() != [False]:
        raise AssertionError("marker matched across the prompt/answer boundary")

    prompt_rows = torch.tensor([
        tokenizer.encode("AAAAAAAAAA"),
        tokenizer.encode("BBBBBBBBBB"),
    ], dtype=torch.long)
    generated_rows = torch.tensor([
        tokenizer.encode("Question:"),
        tokenizer.encode("not-a-hit"),
    ], dtype=torch.long)
    row_matches = GeneratedOnlyStopStringCriteria(
        inner, prompt_rows.shape[1])(
            torch.cat([prompt_rows, generated_rows], dim=1), scores=None)
    if row_matches.tolist() != [True, False]:
        raise AssertionError(f"row-wise marker mask is wrong: {row_matches.tolist()}")


def main():
    before, after = check_integration_and_score_equivalence()
    check_generated_only_boundary_and_rows()
    print(
        "EVAL_TRACE_GENERATION_STOPS=PASS "
        f"mock_decode_steps={before}->{after} score_equivalent=true")


if __name__ == "__main__":
    main()
