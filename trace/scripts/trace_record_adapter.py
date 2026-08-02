"""Schema normalization shared by the model-free reproduction checks."""

from __future__ import annotations


def prompt_answer(record: dict) -> tuple[str, str]:
    """Accept official TRACE or SLoRA's undocumented conversation schema."""
    if "prompt" in record and "answer" in record:
        prompt, answer = record["prompt"], record["answer"]
    elif "conversations" in record:
        conversations = record["conversations"]
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError("conversations must contain user and assistant turns")
        prompt = conversations[0].get("value")
        answer = conversations[1].get("value")
    else:
        raise ValueError("record must contain prompt/answer or conversations")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise ValueError("prompt and answer must be strings")
    return prompt, answer


def sft_messages(record: dict) -> dict:
    prompt, answer = prompt_answer(record)
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }
