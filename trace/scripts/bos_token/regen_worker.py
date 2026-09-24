#!/usr/bin/env python3
"""One GPU worker for regen_pool.py: load the checkpoint once, then pull (task, shard) jobs from the
queue directory until it is empty.  Claiming a job is an atomic mkdir, so no locking service is needed."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoConfig, StoppingCriteriaList

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "implementations" / "llmcl_benchmark", REPO / "scripts" / "bos_token",
          REPO / "scripts" / "residual", REPO / "scripts" / "analysis"):
    sys.path.insert(0, str(p))
from residual_expert import load_v3_any_checkpoint          # noqa: E402
import bos_guard                                            # noqa: E402
from train_bos_token import chat_header_ids, install_header_guard, install_all_layer_bias   # noqa: E402
from gen_doc import EotCount, parse_doc, SYSTEM_TEXT, USER_HDR   # noqa: E402
from answer_pass_v3_fix import apply_cue                    # noqa: E402
from chat_profile import chat_profile, clean                # noqa: E402


def claim(queue: Path, index: int):
    try:
        (queue / f"claim.{index}").mkdir()
        return True
    except FileExistsError:
        return False


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--dest", required=True)
    ap.add_argument("--queue", required=True); ap.add_argument("--worker", required=True)
    ap.add_argument("--base-model", default=os.environ.get(
        "SLORA_LLAMA31_PATH", "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"))
    ap.add_argument("--anchor-json", default=str(REPO / "scripts/selfgen/assets/anchors.json"))
    ap.add_argument("--guard-decision", default="none")
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()
    queue = Path(a.queue); jobs = json.load(open(queue / "jobs.json"))
    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    profile = chat_profile(tok, AutoConfig.from_pretrained(a.base_model, local_files_only=True).model_type)
    anchors = json.load(open(a.anchor_json))
    t0 = time.time()
    model, meta = load_v3_any_checkpoint(a.checkpoint, tok, a.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval(); install_header_guard(model, tok, a.guard_decision)
    for layer in model.model.layers:
        layer.shared_expert_router._logit_bias_positions = bos_guard.decision_position_mask
    n_layers = len(model.model.layers)
    n_slots = model.model.layers[0].shared_expert_router.router.weight.shape[0] + 1
    eot = tok.convert_tokens_to_ids("<|eot_id|>"); header_ids = chat_header_ids(tok)
    given_text = SYSTEM_TEXT + USER_HDR; eot_stop = 3 - given_text.count("<|eot_id|>")
    eos_ids = list(profile.stop_token_ids)
    print(f"[worker {a.worker}] loaded in {time.time()-t0:.0f}s, experts={meta['num_experts']}", flush=True)
    done = 0
    for index, job in enumerate(jobs):
        if not claim(queue, index):
            continue
        j, task, k = int(job["task_index"]), job["task"], int(job["shard"])
        n, bA, bB = int(job["num_seqs"]), int(job["batch_a"]), int(job["batch_b"])
        cap, cue, seed = int(job["max_new_tokens"]), job.get("cue", ""), int(job["seed"])
        anchor = anchors[str(j)]["anchor"]
        sa = Path(a.dest) / task / f"stageA.s{k}"; sa.mkdir(parents=True, exist_ok=True)
        out_path = Path(a.dest) / task / f"records.part{k}.jsonl"
        tJ = time.time(); torch.manual_seed(seed)
        bias = torch.full((n_layers, n_slots), float("-inf")); bias[:, j] = 0.0
        handle = install_all_layer_bias(model, bias.cuda())
        bos_guard.FORCE_LAST_PROMPT_POS = False
        rows = []
        for b0 in range(0, n, bA):
            b = min(bA, n - b0)
            ids = torch.tensor([header_ids] * b, device="cuda")
            outp = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                  temperature=a.temperature, top_p=a.top_p, max_new_tokens=cap,
                                  pad_token_id=tok.pad_token_id, eos_token_id=[128001],
                                  stopping_criteria=StoppingCriteriaList([EotCount(eot, eot_stop, len(header_ids))]), use_cache=True)
            for r in range(b):
                g = outp[r, len(header_ids):].tolist(); cnt, cut = 0, len(g)
                for i, t in enumerate(g):
                    if t == eot:
                        cnt += 1
                        if cnt >= eot_stop: cut = i + 1; break
                g = g[:cut]; text = given_text + tok.decode(g, skip_special_tokens=False)
                rec = {"i": b0 + r, "n_tokens": len(g), "text": text, **parse_doc(text)}
                rec["starts_with_anchor"] = bool(rec["user"] and rec["user"].startswith(anchor)); rows.append(rec)
        with (sa / "docs.jsonl").open("w") as fh:
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (sa / "text.jsonl").open("w") as fh:
            for r in rows:
                if r["user"]: fh.write(json.dumps({"i": r["i"], "anchor": "", "text": r["user"]}, ensure_ascii=False) + "\n")
        (sa / "stats.json").write_text(json.dumps({"num": len(rows),
            "starts_with_anchor": float(np.mean([r["starts_with_anchor"] for r in rows])),
            "complete": float(np.mean([r["complete"] for r in rows]))}, indent=1, ensure_ascii=False))
        prompts = []
        for r in rows:
            if not r["user"]: continue
            user = apply_cue(clean(r["user"], profile).strip(), cue)
            if user: prompts.append(user)
        bos_guard.FORCE_LAST_PROMPT_POS = True; written = 0
        with out_path.open("w") as fh:
            for s0 in range(0, len(prompts), bB):
                chunk = prompts[s0:s0 + bB]
                texts = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."},
                                                  {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in chunk]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
                out = model.generate(**enc, do_sample=False, max_new_tokens=cap, pad_token_id=tok.pad_token_id,
                                     eos_token_id=eos_ids, use_cache=True)
                for prompt, ids_ in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
                    answer = clean(tok.decode(ids_, skip_special_tokens=False), profile).strip()
                    if not answer: continue
                    fh.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n"); written += 1
        bos_guard.FORCE_LAST_PROMPT_POS = False; handle.remove()
        for layer in model.model.layers: layer.shared_expert_router._logit_bias = None
        done += 1
        print(f"[worker {a.worker}] {task} shard{k}: {len(rows)} docs -> {written} records in {time.time()-tJ:.0f}s", flush=True)
    print(f"[worker {a.worker}] finished {done} jobs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
