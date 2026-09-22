#!/usr/bin/env python3
"""Single-load replay regeneration: load the checkpoint ONCE, then for every assigned task run
stage A (sample whole user turns from the fixed chat header with "\\n\\n" forced to E_j) and
stage B (greedy answers under the training header, guard on, E_j forced at both "\\n\\n").

Outputs per task are byte-compatible with gen_doc.py + answer_pass_v3_fix.py:
  <dest>/<task>/stageA.s<k>/docs.jsonl, text.jsonl, stats.json
  <dest>/<task>/records.part<k>.jsonl              {"prompt","answer"}
Jobs come from a JSON list: [{"task_index","task","num_seqs","batch_a","batch_b","max_new_tokens","cue","seed"}, ...]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoConfig, StoppingCriteriaList

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "implementations" / "llmcl_benchmark", REPO / "scripts" / "bos_token", REPO / "scripts" / "residual", REPO / "scripts" / "analysis"):
    sys.path.insert(0, str(p))
from residual_expert import load_v3_any_checkpoint          # noqa: E402
import bos_guard                                            # noqa: E402
from train_bos_token import chat_header_ids, install_header_guard, install_all_layer_bias   # noqa: E402
from gen_doc import EotCount, parse_doc, SYSTEM_TEXT, USER_HDR   # noqa: E402
from answer_pass_v3_fix import apply_cue                    # noqa: E402
from chat_profile import chat_profile, clean                # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--jobs", required=True, help="JSON file: list of task jobs")
    p.add_argument("--dest", required=True, help="round dir; task subdirs are created")
    p.add_argument("--shard", type=int, required=True, help="k: names stageA.s<k> / records.part<k>")
    p.add_argument("--anchor-json", default=str(REPO / "scripts/selfgen/assets/anchors.json"))
    p.add_argument("--guard-decision", default="none")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


@torch.no_grad()
def main():
    a = parse_args()
    jobs = json.load(open(a.jobs))
    # optional hot override (round dir's parent / regen_overrides.json): batch sizes for long tasks
    ov_path = Path(a.dest).parent / "regen_overrides.json"
    if ov_path.exists():
        ov = json.load(open(ov_path))
        for job in jobs:
            if int(job["max_new_tokens"]) >= 1024:
                job["batch_a"] = int(ov.get("long_batch_a", job.get("batch_a", 64)))
                job["batch_b"] = int(ov.get("long_batch_b", job.get("batch_b", 32)))
            else:
                job["batch_a"] = int(ov.get("short_batch_a", job.get("batch_a", 125)))
                job["batch_b"] = int(ov.get("short_batch_b", job.get("batch_b", 64)))
        print(f"[regen] overrides applied from {ov_path}: {ov}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    profile = chat_profile(tok, AutoConfig.from_pretrained(a.base_model, local_files_only=True).model_type)
    anchors = json.load(open(a.anchor_json))
    t0 = time.time()
    model, meta = load_v3_any_checkpoint(a.checkpoint, tok, a.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval()
    install_header_guard(model, tok, a.guard_decision)
    n_layers = len(model.model.layers)
    n_slots = model.model.layers[0].shared_expert_router.router.weight.shape[0] + 1
    for layer in model.model.layers:
        layer.shared_expert_router._logit_bias_positions = bos_guard.decision_position_mask
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    header_ids = chat_header_ids(tok)
    given_text = SYSTEM_TEXT + USER_HDR
    eot_stop = 3 - given_text.count("<|eot_id|>")
    eos_ids = list(profile.stop_token_ids)
    print(f"[regen] loaded {a.checkpoint} experts={meta['num_experts']} in {time.time()-t0:.0f}s; jobs={[j['task'] for j in jobs]}", flush=True)

    for job in jobs:
        j = int(job["task_index"]); task = job["task"]; n = int(job["num_seqs"])
        bA = int(job.get("batch_a", n)); bB = int(job.get("batch_b", 32)); cap = int(job["max_new_tokens"]); cue = job.get("cue", "")
        seed = int(job.get("seed", 0)); anchor = anchors[str(j)]["anchor"]
        sa = Path(a.dest) / task / f"stageA.s{a.shard}"; sa.mkdir(parents=True, exist_ok=True)
        out_path = Path(a.dest) / task / f"records.part{a.shard}.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[regen] {task}: records exist, skip", flush=True); continue
        torch.manual_seed(seed)
        # force E_j at every layer on the decision positions only
        bias = torch.full((n_layers, n_slots), float("-inf")); bias[:, j] = 0.0
        handle = install_all_layer_bias(model, bias.cuda())
        # ---- stage A: whole user turns from the header prompt ("\n\n" is the only decision position)
        bos_guard.FORCE_LAST_PROMPT_POS = False
        rows, tA = [], time.time()
        for b0 in range(0, n, bA):
            b = min(bA, n - b0)
            ids = torch.tensor([header_ids] * b, device="cuda")
            outp = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                  temperature=a.temperature, top_p=a.top_p, max_new_tokens=cap,
                                  pad_token_id=tok.pad_token_id, eos_token_id=[128001],
                                  stopping_criteria=StoppingCriteriaList([EotCount(eot, eot_stop, len(header_ids))]), use_cache=True)
            for r in range(b):
                g = outp[r, len(header_ids):].tolist()
                cnt, cut = 0, len(g)
                for i, t in enumerate(g):
                    if t == eot:
                        cnt += 1
                        if cnt >= eot_stop:
                            cut = i + 1; break
                g = g[:cut]
                text = given_text + tok.decode(g, skip_special_tokens=False)
                rec = {"i": b0 + r, "n_tokens": len(g), "text": text, **parse_doc(text)}
                rec["starts_with_anchor"] = bool(rec["user"] and rec["user"].startswith(anchor))
                rows.append(rec)
        with (sa / "docs.jsonl").open("w") as fh:
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (sa / "text.jsonl").open("w") as fh:
            for r in rows:
                if r["user"]: fh.write(json.dumps({"i": r["i"], "anchor": "", "text": r["user"]}, ensure_ascii=False) + "\n")
        stats = {"cond": f"force_E{j}@decision+guard({a.guard_decision})+chat_header", "num": len(rows),
                 "has_user": float(np.mean([r["user"] is not None for r in rows])),
                 "starts_with_anchor": float(np.mean([r["starts_with_anchor"] for r in rows])),
                 "complete": float(np.mean([r["complete"] for r in rows])), "elapsed": time.time() - tA}
        (sa / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
        # ---- stage B: greedy answers, training header (guard matches), E_j forced at user "\n\n" + assistant "\n\n"
        prompts = []
        for r in rows:
            if not r["user"]: continue
            user = apply_cue(clean(r["user"], profile).strip(), cue)
            if user: prompts.append(user)
        bos_guard.FORCE_LAST_PROMPT_POS = True
        tB, written = time.time(), 0
        with out_path.open("w") as fh:
            for s0 in range(0, len(prompts), bB):
                chunk = prompts[s0:s0 + bB]
                texts = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}],
                                                 tokenize=False, add_generation_prompt=True) for p in chunk]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
                out = model.generate(**enc, do_sample=False, max_new_tokens=cap, pad_token_id=tok.pad_token_id,
                                     eos_token_id=eos_ids, use_cache=True)
                gen = out[:, enc["input_ids"].shape[1]:]
                for prompt, ids in zip(chunk, gen):
                    answer = clean(tok.decode(ids, skip_special_tokens=False), profile).strip()
                    if not answer: continue
                    fh.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n"); written += 1
        bos_guard.FORCE_LAST_PROMPT_POS = False
        handle.remove()
        for layer in model.model.layers: layer.shared_expert_router._logit_bias = None
        print(f"[regen] {task}: stageA {len(rows)} docs ({stats['elapsed']:.0f}s, anchor {stats['starts_with_anchor']:.2f}) -> stageB {written} records ({time.time()-tB:.0f}s)", flush=True)
    print(f"[regen] DONE shard {a.shard} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
