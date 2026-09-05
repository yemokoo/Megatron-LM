#!/usr/bin/env python3
"""Stage B v2: answers with distribution-faithfulness guards.

Extends answer_pass_v3_fix.py.  The fix run (selfgen_cl_fix_20260901) showed
that restoring full-length generation without guarding its distribution makes
replay WORSE than near-empty replay: the two "fixed" tasks lost the most
(MeetingBank BWT -9.59 -> -13.59, Py150 -1.55 -> -5.73).  Root causes measured
against the lm baseline's actual 500-record replay:

  FOMC       label collapse: C 50.8% -> 89.1%, A 27.4% -> 2.2%
  ScienceQA  45% of answers drop the required letter prefix entirely
  MeetingBank answers drift from summary register to transcript register
             (0.2% -> 16.5% transcript openers), median length halves
  Py150      mode collapse onto import boilerplate (top-3 5.0% -> 26.2%)

Guards (all storage-free rules; the label priors are 3-5 floats per task):

  --label-prior A:p,B:p,...  constrained first-token choice under a quota drawn
                             from the real label prior.  Per sample the model's
                             own ranking is respected; when its top label's
                             quota is exhausted it takes its next-best label.
  --label-only               the answer is the letter alone (C-STANCE, FOMC).
                             Without it the chosen letter seeds a continuation
                             (ScienceQA: letter + explanation).
  --reject-regex / --min-answer-chars
                             resample (temperature) then drop answers in the
                             wrong register (MeetingBank transcript openers).
  --max-dup-share F          cap identical answers at F * n_prompts (Py150).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
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
    p.add_argument("--stage-a", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-answer-tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prompt-cue", default="")
    p.add_argument("--label-prior", default="",
                   help="'A:0.367,B:0.328,C:0.305' -- constrained first-token "
                        "labels under a real-prior quota")
    p.add_argument("--label-only", action="store_true",
                   help="answer is the chosen letter alone")
    p.add_argument("--reject-regex", default="",
                   help="resample-then-drop answers matching this")
    p.add_argument("--min-answer-chars", type=int, default=0)
    p.add_argument("--max-dup-share", type=float, default=0.0,
                   help=">0: cap identical answers at share * n_prompts")
    p.add_argument("--resample-tries", type=int, default=2)
    p.add_argument("--resample-temperature", type=float, default=0.8)
    p.add_argument("--resample-top-p", type=float, default=0.95)
    return p.parse_args()


def apply_cue(user, cue):
    if not cue:
        return user
    i = user.find(cue)
    if i >= 0:
        user = user[:i]
    return user.rstrip() + cue


def clean(text):
    for marker in ("<|eot_id|>", "<|end_of_text|>"):
        text = text.split(marker)[0]
    return text


def parse_prior(spec, tokenizer):
    """-> (letters, probs, first_token_ids)"""
    letters, probs, ids = [], [], []
    for part in spec.split(","):
        letter, prob = part.split(":")
        letter = letter.strip()
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"label {letter!r} is not a single token: {toks}")
        letters.append(letter)
        probs.append(float(prob))
        ids.append(toks[0])
    total = sum(probs)
    probs = [p / total for p in probs]
    return letters, probs, ids


def quota_counts(probs, n):
    """Largest-remainder apportionment of n samples over the prior."""
    raw = [p * n for p in probs]
    counts = [int(x) for x in raw]
    rest = n - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:rest]:
        counts[i] += 1
    return counts


def gen_batch(model, tokenizer, texts, max_new, eos_ids, sample=False,
              temperature=0.8, top_p=0.95):
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=sample,
            **({"temperature": temperature, "top_p": top_p} if sample else {}),
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
            use_cache=True)
    return out[:, enc["input_ids"].shape[1]:]


def first_token_logits(model, tokenizer, texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model(**enc)
    return out.logits[:, -1, :]          # [B, vocab] next-token logits


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
        user = clean(r.get("anchor", "") + r["text"]).strip()
        user = apply_cue(user, args.prompt_cue)
        if user:
            prompts.append(user)
    print(f"[ans2] {len(prompts)} prompts from {stage_a}", flush=True)

    model, _ = load_v3_checkpoint(
        args.checkpoint, tokenizer, args.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval()

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.append(eot)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records = []

    if args.label_prior:
        letters, probs, letter_ids = parse_prior(args.label_prior, tokenizer)
        quota = quota_counts(probs, len(prompts))
        print(f"[ans2] label quota: {dict(zip(letters, quota))}", flush=True)
        chosen = []
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            texts = [CHAT_HEADER + p + ASSISTANT for p in chunk]
            logits = first_token_logits(model, tokenizer, texts)
            lab_logits = logits[:, letter_ids]                     # [B, L]
            ranks = torch.argsort(lab_logits, dim=-1, descending=True)
            for row in range(len(chunk)):
                pick = None
                for j in ranks[row].tolist():
                    if quota[j] > 0:
                        pick = j
                        break
                if pick is None:                                   # quotas spent (rounding)
                    pick = int(ranks[row][0])
                else:
                    quota[pick] -= 1
                chosen.append(pick)
            print(f"[ans2] labels {min(start + args.batch, len(prompts))}/{len(prompts)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        if args.label_only:
            records = [{"prompt": p, "answer": letters[c]}
                       for p, c in zip(prompts, chosen)]
        else:
            # letter seeds a continuation (ScienceQA: letter + explanation)
            for start in range(0, len(prompts), args.batch):
                chunk = prompts[start:start + args.batch]
                picks = chosen[start:start + args.batch]
                texts = [CHAT_HEADER + p + ASSISTANT + letters[c]
                         for p, c in zip(chunk, picks)]
                gen = gen_batch(model, tokenizer, texts,
                                args.max_answer_tokens - 1, eos_ids)
                for p, c, ids in zip(chunk, picks, gen):
                    tail = clean(tokenizer.decode(ids, skip_special_tokens=False))
                    answer = (letters[c] + tail).strip()
                    if answer:
                        records.append({"prompt": p, "answer": answer})
                print(f"[ans2] cont {min(start + args.batch, len(prompts))}/{len(prompts)}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    else:
        reject_re = re.compile(args.reject_regex, re.I) if args.reject_regex else None
        dup_cap = (max(1, int(args.max_dup_share * len(prompts)))
                   if args.max_dup_share > 0 else 0)
        dup_counter = collections.Counter()

        def acceptable(answer):
            if not answer:
                return False
            if args.min_answer_chars and len(answer) < args.min_answer_chars:
                return False
            if reject_re is not None and reject_re.match(answer):
                return False
            if dup_cap and dup_counter[answer] >= dup_cap:
                return False
            return True

        pending = list(range(len(prompts)))
        answers = {}
        for attempt in range(args.resample_tries + 1):
            if not pending:
                break
            sample = attempt > 0
            nxt = []
            for start in range(0, len(pending), args.batch):
                idx = pending[start:start + args.batch]
                texts = [CHAT_HEADER + prompts[i] + ASSISTANT for i in idx]
                gen = gen_batch(model, tokenizer, texts, args.max_answer_tokens,
                                eos_ids, sample=sample,
                                temperature=args.resample_temperature,
                                top_p=args.resample_top_p)
                for i, ids in zip(idx, gen):
                    answer = clean(tokenizer.decode(ids, skip_special_tokens=False)).strip()
                    if acceptable(answer):
                        answers[i] = answer
                        if dup_cap:
                            dup_counter[answer] += 1
                    else:
                        nxt.append(i)
                print(f"[ans2] try{attempt} {min(start + args.batch, len(pending))}/"
                      f"{len(pending)}, {time.time() - t0:.0f}s", flush=True)
            print(f"[ans2] attempt {attempt}: {len(pending) - len(nxt)} accepted, "
                  f"{len(nxt)} rejected", flush=True)
            pending = nxt
        if pending:
            print(f"[ans2] dropping {len(pending)} unrecoverable records", flush=True)
        records = [{"prompt": prompts[i], "answer": answers[i]}
                   for i in sorted(answers)]

    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ans2] DONE {len(records)} records -> {out_path} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
