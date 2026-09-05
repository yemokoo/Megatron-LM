#!/usr/bin/env python3
"""Stage B: give every generated prompt its own answer, greedily.

Stage A (bos_sample_v3.py) emits the USER TURN only -- it stops at <|eot_id|>,
because that is where the training sequence's prompt ends.  The replay/KD loss
is computed on the answer span, so a prompt with no answer cannot be replayed.
This pass rebuilds each user turn (anchor + generated text), appends the
assistant header, and decodes the answer greedily: the label should be the
model's single most likely answer, not a sampled one.

Writes records.jsonl of {"prompt", "answer"} -- the same shape as TRACE
train.json, so the existing SLoRATraceDataCollator consumes it unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "implementations" / "llmcl_benchmark"))
from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint          # noqa: E402
from transformers import AutoTokenizer                          # noqa: E402

CHAT_HEADER = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n")
ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--stage-a", required=True, help="directory written by bos_sample_v3.py")
    p.add_argument("--out", required=True, help="records.jsonl path")
    p.add_argument("--max-answer-tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def clean(text, tokenizer):
    for marker in ("<|eot_id|>", "<|end_of_text|>"):
        text = text.split(marker)[0]
    return text


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    stage_a = Path(args.stage_a)
    rows = [json.loads(l) for l in (stage_a / "text.jsonl").open()]
    if args.limit:
        rows = rows[:args.limit]
    prompts = []
    for r in rows:
        user = (r.get("anchor", "") + r["text"])
        user = clean(user, tokenizer).strip()
        if user:
            prompts.append(user)
    print(f"[ans] {len(prompts)} prompts from {stage_a}", flush=True)

    model, _ = load_v3_checkpoint(
        args.checkpoint, tokenizer, args.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval()

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.append(eot)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0, written = time.time(), 0
    with out_path.open("w") as fh:
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            texts = [CHAT_HEADER + p + ASSISTANT for p in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, do_sample=False,                       # greedy: this is a label
                    max_new_tokens=args.max_answer_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
                    use_cache=True)
            gen = out[:, enc["input_ids"].shape[1]:]
            for prompt, ids in zip(chunk, gen):
                answer = clean(tokenizer.decode(ids, skip_special_tokens=False), tokenizer).strip()
                if not answer:
                    continue
                fh.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
                written += 1
            print(f"[ans] {min(start + args.batch, len(prompts))}/{len(prompts)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[ans] DONE {written} records -> {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
