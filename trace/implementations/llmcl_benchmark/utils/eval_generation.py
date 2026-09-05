"""Shared low-overhead Hugging Face generation for custom MoE checkpoints."""

import torch
from tqdm import tqdm
from transformers import StopStringCriteria, StoppingCriteria, StoppingCriteriaList


class GeneratedOnlyStopStringCriteria(StoppingCriteria):
    """Apply ``StopStringCriteria`` only to tokens generated after the prompt.

    Hugging Face's string criterion normally sees the prompt and continuation as
    one sequence.  TRACE's post-hoc normalizer, however, searches only the raw
    continuation.  Slicing at the padded prompt width prevents a marker split
    across the prompt/answer boundary from stopping generation prematurely.
    """

    def __init__(self, criterion, prompt_length):
        self.criterion = criterion
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[:, self.prompt_length:]
        if generated_ids.shape[1] == 0:
            return torch.zeros(
                input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return self.criterion(generated_ids, scores, **kwargs)


def all_eos_token_ids(model, tokenizer):
    """Return every configured stop token, including Qwen chat/end tokens."""
    token_ids = set()
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and generation_config.eos_token_id is not None:
        eos = generation_config.eos_token_id
        token_ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    if tokenizer.eos_token_id is not None:
        token_ids.add(tokenizer.eos_token_id)
    for token in ("<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != tokenizer.unk_token_id
        ):
            token_ids.add(token_id)
    return sorted(token_ids)


@torch.inference_mode()
def generate_predictions(
    model,
    tokenizer,
    prompts,
    *,
    device,
    batch_size,
    max_prompt_len,
    max_new_tokens,
    temperature=0.0,
    eos_token_ids=None,
    stop_strings=None,
    length_bucketing=True,
    description="generate",
):
    """Generate predictions while preserving the input order.

    Prompts are tokenized once up front. Sorting by their truncated token length
    prevents a single long example from padding an otherwise short batch to the
    maximum context size. Predictions are written back to their original indices,
    so scoring and result JSON order remain unchanged.
    """
    if not prompts:
        return []

    tokenize_kwargs = {"padding": False}
    if max_prompt_len and max_prompt_len > 0:
        tokenize_kwargs.update(
            truncation=True, max_length=max_prompt_len)
    encoded = tokenizer(prompts, **tokenize_kwargs)
    lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
    order = (
        sorted(range(len(prompts)), key=lengths.__getitem__)
        if length_bucketing
        else list(range(len(prompts)))
    )
    predictions = [None] * len(prompts)
    stop_token_ids = (
        all_eos_token_ids(model, tokenizer)
        if eos_token_ids is None else list(dict.fromkeys(eos_token_ids))
    )
    do_sample = temperature > 0.0
    if isinstance(stop_strings, str):
        stop_strings = [stop_strings]
    else:
        stop_strings = list(stop_strings or [])
    # Vocabulary preprocessing is relatively expensive.  Build the matcher once
    # per task call and reuse it across every generation batch.
    stop_string_criterion = (
        StopStringCriteria(tokenizer, stop_strings)
        if stop_strings else None
    )

    progress = tqdm(
        range(0, len(order), batch_size),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
    )
    for start in progress:
        indices = order[start:start + batch_size]
        features = []
        for index in indices:
            feature = {}
            for key, values in encoded.items():
                if isinstance(values, list) and len(values) == len(prompts):
                    feature[key] = values[index]
            features.append(feature)
        model_inputs = tokenizer.pad(
            features, padding=True, return_tensors="pt")
        model_inputs = {
            key: value.to(device) for key, value in model_inputs.items()}

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": 1,
            "num_beams": 1,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_args["temperature"] = temperature
        if stop_token_ids:
            generation_args["eos_token_id"] = stop_token_ids
        if stop_string_criterion is not None:
            generation_args["stopping_criteria"] = StoppingCriteriaList([
                GeneratedOnlyStopStringCriteria(
                    stop_string_criterion,
                    prompt_length=model_inputs["input_ids"].shape[1],
                )
            ])

        output_ids = model.generate(**model_inputs, **generation_args)
        generated_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        for index, prediction in zip(indices, decoded):
            predictions[index] = prediction

    return predictions
