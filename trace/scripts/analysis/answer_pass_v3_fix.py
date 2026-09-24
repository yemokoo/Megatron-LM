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
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "implementations" / "llmcl_benchmark"))
from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint          # noqa: E402
from transformers import AutoConfig, AutoTokenizer              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_profile import chat_profile, clean                    # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default=os.environ.get(
        "SLORA_LLAMA31_PATH", "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"))
    p.add_argument("--stage-a", required=True, help="directory written by bos_sample_v3.py")
    p.add_argument("--out", required=True, help="records.jsonl path")
    p.add_argument("--max-answer-tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bos-guard", action="store_true", help="install bos_guard (must match how the checkpoint was trained)")
    p.add_argument("--guard-header", action="store_true", help="with --bos-guard: also guard the fixed chat header")
    p.add_argument("--guard-decision", choices=("nn", "end_header", "none"), default="nn", help="header guard decision token (see train_bos_token)")
    p.add_argument("--template-header", action="store_true",
                   help="frame the prompt exactly like training (tokenizer chat template incl. date lines, BOS x2) instead of the hand-written header")
    p.add_argument("--force-expert", type=int, default=-1,
                   help="force this expert at every layer on the decision positions (user-turn \\n\\n and the assistant-turn \\n\\n); needs --bos-guard --guard-header")
    p.add_argument("--prompt-cue", default="",
                   help="the task's answer cue; every real prompt ends with it "
                        "(MeetingBank '\\nSummary:', ScienceQA '\\nAnswer:', ...). "
                        "Stage A only reproduces it ~30%% of the time, and a prompt "
                        "without it makes the model emit the cue AS the answer.")
    p.add_argument("--answer-temperature", type=float, default=0.0,
                   help="0 = greedy.  Greedy takes the argmax per example, which "
                        "turns a soft label prior into a collapse (C-STANCE 29%% -> 76%% 'C'); "
                        "sampling reproduces the model's own marginal instead.")
    p.add_argument("--answer-top-p", type=float, default=0.95)
    return p.parse_args()


def apply_cue(user, cue):
    """End the user turn exactly at the task's answer cue.

    Stage A stops wherever the sampler ran out, so the cue may be missing, or the
    model may have run past it and written its own answer -- that text belongs to
    the answer, not the prompt.  Cut at the first cue and re-append it.
    """
    if not cue:
        return user
    i = user.find(cue)
    if i >= 0:
        user = user[:i]
    return user.rstrip() + cue


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    profile = chat_profile(tokenizer, AutoConfig.from_pretrained(
        args.base_model, local_files_only=True).model_type)
    print(f"[ans] chat profile {profile.name}", flush=True)

    stage_a = Path(args.stage_a)
    rows = [json.loads(l) for l in (stage_a / "text.jsonl").open()]
    if args.limit:
        rows = rows[:args.limit]
    prompts = []
    for r in rows:
        user = (r.get("anchor", "") + r["text"])
        user = clean(user, profile).strip()
        user = apply_cue(user, args.prompt_cue)
        if user:
            prompts.append(user)
    print(f"[ans] {len(prompts)} prompts from {stage_a}", flush=True)

    sys.path.insert(0, str(REPO / "scripts" / "residual"))
    from residual_expert import load_v3_any_checkpoint   # residual-from-task-0 aware
    model, _ = load_v3_any_checkpoint(
        args.checkpoint, tokenizer, args.base_model, device="cuda", dtype=torch.bfloat16)
    if args.bos_guard:
        import bos_guard
        sys.path.insert(0, str(REPO / "scripts" / "bos_token"))
        from train_bos_token import install_header_guard
        if args.guard_header:
            install_header_guard(model, tokenizer, args.guard_decision)
        else:
            bos_guard.install_bos_guard(model)
        print(f"[ans] bos_guard ON header={args.guard_header} decision={args.guard_decision}", flush=True)
    if args.force_expert >= 0:
        if not (args.bos_guard and args.guard_header): raise SystemExit("--force-expert needs --bos-guard --guard-header")
        import bos_guard
        from train_bos_token import install_all_layer_bias
        n_slots = model.model.layers[0].shared_expert_router.router.weight.shape[0] + 1
        bias = torch.full((len(model.model.layers), n_slots), float("-inf")); bias[:, args.force_expert] = 0.0
        install_all_layer_bias(model, bias.cuda())
        bos_guard.FORCE_LAST_PROMPT_POS = True
        for layer in model.model.layers:
            layer.shared_expert_router._logit_bias_positions = bos_guard.decision_position_mask
        print(f"[ans] forcing E{args.force_expert} at the user-turn and assistant-turn decision positions", flush=True)
    model.eval()

    eos_ids = list(profile.stop_token_ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0, written = time.time(), 0
    with out_path.open("w") as fh:
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            if args.template_header:
                texts = [tokenizer.apply_chat_template(
                    [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True) for p in chunk]
                # training (SLoRATraceDataCollator) tokenizes the template text with add_special_tokens=True,
                # which prepends a second BOS -- reproduce that exactly so the header guard matches
                enc = tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=True).to(model.device)
            else:
                texts = [profile.header + p + profile.assistant for p in chunk]
                enc = tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
            with torch.no_grad():
                sample = args.answer_temperature > 0
                out = model.generate(
                    **enc, do_sample=sample,
                    **({"temperature": args.answer_temperature,
                        "top_p": args.answer_top_p} if sample else {}),
                    max_new_tokens=args.max_answer_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
                    use_cache=True)
            gen = out[:, enc["input_ids"].shape[1]:]
            for prompt, ids in zip(chunk, gen):
                answer = clean(tokenizer.decode(ids, skip_special_tokens=False), profile).strip()
                if not answer:
                    continue
                fh.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
                written += 1
            print(f"[ans] {min(start + args.batch, len(prompts))}/{len(prompts)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[ans] DONE {written} records -> {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
