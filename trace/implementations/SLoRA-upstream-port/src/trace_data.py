"""Compatibility helpers for the official TRACE JSON schema."""


def prompt_answer(record):
    if "prompt" in record and "answer" in record:
        prompt, answer = record["prompt"], record["answer"]
    elif "conversations" in record:
        turns = record["conversations"]
        if len(turns) < 2:
            raise ValueError("TRACE conversation record needs two turns")
        prompt, answer = turns[0].get("value"), turns[1].get("value")
    else:
        raise ValueError(
            "Expected TRACE prompt/answer or SLoRA conversations schema"
        )
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise ValueError("TRACE prompt and answer must be strings")
    return prompt, answer


def to_sft_messages(record):
    prompt, answer = prompt_answer(record)
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }
