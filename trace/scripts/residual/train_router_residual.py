#!/usr/bin/env python
"""Router-only tuning of a finished V3 TRACE checkpoint, with or without a
residual (no-op) expert row.

Data mix (one epoch, shuffled):
  * TRACE replay: real samples (fixed_replay_memory indices -> train.json) or
    self-generated records (jsonl with prompt/answer), all 8 tasks incl.
    20Minuten -- SLoRA chat format, full-sequence LM labels (the V3 replay
    objective);
  * backbone BoS generations: raw text after <|begin_of_text|>, full labels.
Only router weights train (all LoRA experts + base frozen).  Output is a small
router-state checkpoint that ``residual_expert.load_router_tuned`` rebuilds.
"""
import argparse
import json
import math
import os
import random
import sys
import time

TRACE = "/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
LLMCL = f"{TRACE}/implementations/llmcl_benchmark"
sys.path.insert(0, LLMCL)
sys.path.insert(0, f"{TRACE}/scripts/residual")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import residual_expert as RE  # noqa: E402

BASE = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
DATA = "/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace"
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def load_replay(spec, per_task):
    """spec: 'real:<fixed_replay_memory dir>' or 'gen:<task>=<jsonl>,...'"""
    kind, rest = spec.split(":", 1)
    out = []
    if kind == "real":
        for i, task in enumerate(TASKS):
            mem = json.load(open(os.path.join(rest, f"task_{i}_{task}.json")))
            rows = json.load(open(os.path.join(DATA, task, "train.json")))
            for j in mem["indices"][:per_task]:
                out.append({"kind": "trace", "task": task,
                            "prompt": rows[j]["prompt"], "answer": rows[j]["answer"]})
    elif kind == "gen":
        for item in rest.split(","):
            task, path = item.split("=", 1)
            n = 0
            for line in open(path):
                r = json.loads(line)
                if not r.get("prompt") or r.get("answer") is None:
                    continue
                out.append({"kind": "trace", "task": task, "prompt": r["prompt"], "answer": r["answer"]})
                n += 1
                if n >= per_task:
                    break
            print(f"gen {task}: {n}", flush=True)
    else:
        raise ValueError(spec)
    return out


def encode(tok, rec, max_len):
    if rec["kind"] == "trace":
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["answer"]}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(text, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        if len(ids) < max_len and ids[-1] != tok.eos_token_id:
            ids.append(tok.eos_token_id)
    else:
        ids = [tok.bos_token_id] + tok(rec["text"], add_special_tokens=False,
                                       truncation=True, max_length=max_len - 2)["input_ids"]
        ids.append(tok.eos_token_id)
    return ids[:max_len]


def collate(batch, pad_id):
    L = max(len(x) for x in batch)
    inp = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, x in enumerate(batch):
        inp[i, :len(x)] = torch.tensor(x)
        lab[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    return inp, lab, att


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="V3 checkpoint dir (round 7)")
    p.add_argument("--replay", required=True)
    p.add_argument("--backbone", required=True, help="backbone BoS records.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--n_residual", type=int, default=1)
    p.add_argument("--residual_init", default="zeros")
    p.add_argument("--no_second_choice", action="store_true")
    p.add_argument("--residual_only", action="store_true",
                   help="train only the residual row(s); the existing expert rows stay "
                        "bit-identical, so top-1 expert choice and outputs of non-residual "
                        "tokens equal the source checkpoint")
    p.add_argument("--replay_per_task", type=int, default=500)
    p.add_argument("--backbone_n", type=int, default=4000)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2025)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(BASE)
    pad_id = 128004  # <|finetune_right_pad_id|>, as in the V3 trainer
    recs = load_replay(a.replay, a.replay_per_task)
    n_trace = len(recs)
    bb = [json.loads(l) for l in open(a.backbone)][:a.backbone_n]
    recs += [{"kind": "backbone", "text": r["text"]} for r in bb]
    print(f"data: trace {n_trace} + backbone {len(bb)} = {len(recs)}", flush=True)
    samples = [encode(tok, r, a.max_len) for r in recs]
    random.shuffle(samples)

    import evaluate_Ours_LoRA_MoE as E
    model, meta = E.load_v3_checkpoint(a.source, tok, base_model_name_or_path=BASE,
                                       device="cuda", dtype=torch.bfloat16, device_map=None)
    model.to("cuda")
    if a.n_residual:
        RE.add_residual_experts(model, a.n_residual, init=a.residual_init,
                                second_choice=not a.no_second_choice)
    for prm in model.parameters():
        prm.requires_grad = False
    params = RE.router_parameters(model)
    V3 = RE._v3()
    for layer in V3.shared_router_layers(model):
        r = layer.shared_expert_router
        r.router.float()                      # fp32 master weights; autocast casts per call
        r.router.weight.requires_grad = True
        r._suppress_router_loss = True        # LM loss only (the V3 replay objective)
    params = RE.router_parameters(model)
    if a.residual_only:
        if not a.n_residual:
            raise SystemExit("--residual_only needs --n_residual > 0")
        for layer in V3.shared_router_layers(model):
            real = layer.shared_expert_router._n_real
            w = layer.shared_expert_router.router.weight
            mask = torch.zeros_like(w)
            mask[real:] = 1.0
            w.register_hook(lambda g, m=mask: g * m)
        frozen_ref = [w[:l.shared_expert_router._n_real].detach().clone()
                      for w, l in zip(params, V3.shared_router_layers(model))]
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    print(f"trainable router params: {sum(x.numel() for x in params):,} "
          f"(n_residual={a.n_residual})", flush=True)

    steps_per_epoch = math.ceil(len(samples) / (a.batch * a.accum))
    total = int(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    step, micro, t0, run_loss = 0, 0, time.time(), 0.0
    order = list(range(len(samples)))
    ep = 0
    while step < total:
        if ep > 0:
            random.shuffle(order)
        for b in range(0, len(order), a.batch):
            inp, lab, att = collate([samples[i] for i in order[b:b + a.batch]], pad_id)
            inp, lab, att = inp.cuda(), lab.cuda(), att.cuda()
            V3.set_v3_router_token_mask(model, att)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=inp, attention_mask=att, labels=lab, use_cache=False)
            (out.loss / a.accum).backward()
            run_loss += out.loss.item() / a.accum
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                rec = {"step": step, "total": total, "loss": round(run_loss, 4),
                       "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            run_loss = 0.0
            if step >= total:
                break
        ep += 1
    V3.set_v3_router_token_mask(model, None)
    if a.residual_only:
        drift = max(float((w[:ref.shape[0]].detach() - ref).abs().max())
                    for w, ref in zip(params, frozen_ref))
        print(f"expert-row max drift after training: {drift:.3e}", flush=True)
        if drift != 0.0:
            raise RuntimeError("expert router rows changed under --residual_only")

    RE.save_router_checkpoint(model, a.out, {
        "source_checkpoint": a.source, "n_residual": a.n_residual,
        "residual_init": a.residual_init, "second_choice": not a.no_second_choice,
        "residual_only": a.residual_only,
        "replay": a.replay, "backbone": a.backbone, "replay_per_task": a.replay_per_task,
        "backbone_n": len(bb), "n_trace": n_trace, "steps": total, "lr": a.lr,
        "batch": a.batch, "accum": a.accum, "max_len": a.max_len})
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
