"""Chat framing for self-generated replay (Stage A user turn, Stage B answer).

Llama keeps the exact hand-written strings every existing run used, so those
runs stay reproducible byte for byte.  Any other model (Qwen3, ...) derives the
framing from its own chat template -- the same template SLoRATraceDataCollator
applies at training time -- so generated records match training input.  For
Qwen3 that includes the empty think block the template puts in front of every
assistant turn (think-off).
"""
from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = "You are a helpful assistant."
_MARK = "\u0000USER\u0000"

LLAMA3_HEADER = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n")
LLAMA3_ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


@dataclass(frozen=True)
class ChatProfile:
    name: str
    header: str          # everything before the user text
    assistant: str       # end of user turn + assistant header
    doc_start: str       # "--mode bos" prompt: what a document starts with
    stop_markers: tuple  # strings that end a generated turn
    stop_token_ids: tuple


def _ids(tokenizer, markers):
    out = []
    for m in markers:
        i = tokenizer.convert_tokens_to_ids(m)
        if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id:
            out.append(i)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in out:
        out.insert(0, tokenizer.eos_token_id)
    return tuple(out)


def chat_profile(tokenizer, model_type: str) -> ChatProfile:
    if model_type == "llama":
        # generate() always stopped on eos + <|eot_id|> only; <|end_of_text|> was
        # a text-cleanup marker, never a stop token -- keep both exactly.
        return ChatProfile("llama3", LLAMA3_HEADER, LLAMA3_ASSISTANT,
                           "<|begin_of_text|>", ("<|eot_id|>", "<|end_of_text|>"),
                           _ids(tokenizer, ("<|eot_id|>",)))
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _MARK}]
    kwargs = {"enable_thinking": False} if model_type == "qwen3" else {}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs)
    header, assistant = text.split(_MARK)
    if model_type.startswith("qwen"):
        markers = ("<|im_end|>", "<|endoftext|>")
        doc_start = "<|endoftext|>"    # Qwen has no BOS; documents are <|endoftext|>-separated
    else:
        raise ValueError(f"no chat profile for model_type={model_type!r}")
    return ChatProfile(model_type, header, assistant, doc_start, markers,
                       _ids(tokenizer, markers))


def clean(text: str, profile: ChatProfile) -> str:
    for marker in profile.stop_markers:
        text = text.split(marker)[0]
    return text
