#!/usr/bin/env python3
"""Generate WHOLE chat documents (system + user + assistant turns) from a bare
document-start prompt, optionally conditioned by a trained BoS token (v1) or a
last-layer bias (v2).  Stops at the N-th <|eot_id|> (3 = end of the assistant
turn) instead of the first.

Outputs: docs.jsonl (raw text + parsed turns + flags), text.jsonl (user turn
only, {"anchor": "", "text": ...} so answer_pass_v3_fix.py can run stage B on it),
stats.json.
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import StoppingCriteria, StoppingCriteriaList

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "implementations" / "llmcl_benchmark"))
sys.path.insert(0, str(REPO / "scripts" / "bos_token"))
from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint
sys.path.insert(0, str(REPO / "scripts" / "residual"))
from residual_expert import load_v3_any_checkpoint   # residual-from-task-0 aware    # noqa: E402
sys.path.insert(0, str(REPO / "scripts" / "residual"))
import bos_guard                                       # noqa: E402
from transformers import AutoTokenizer                  # noqa: E402
from train_bos_token import install_last_bias, install_all_layer_bias, chat_header_ids, GUARD_DECISIONS, install_header_guard   # noqa: E402

SYSTEM_TEXT = ("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
               "Today Date: 26 Jul 2024\n\nYou are a helpful assistant.<|eot_id|>")
USER_HDR = "<|start_header_id|>user<|end_header_id|>\n\n"
ASST_HDR = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--bos-token-file", default="", help="v1: bos_token.pt; prompt becomes that single token")
    p.add_argument("--last-bias-file", default="", help="v2: last_bias.pt; prompt stays the BOS run")
    p.add_argument("--all-layer-bias-file", default="", help="v3: all_layer_bias.pt; prompt stays the BOS run")
    p.add_argument("--bos-guard", action="store_true", help="force residual at layers 1..L-1 for BOS-token positions")
    p.add_argument("--guard-header", action="store_true", help="with --bos-guard: also guard the fixed chat header (see bos_guard.header_masks)")
    p.add_argument("--guard-decision", choices=GUARD_DECISIONS, default="nn", help="header guard decision token (see train_bos_token)")
    p.add_argument("--bias-positions", choices=("all", "decision"), default="all",
                   help="decision: apply the logit bias/mask only at the first token after the header (needs --bos-guard --guard-header)")
    p.add_argument("--bos-repeat", type=int, default=2, help="training documents start with 2 BOS")
    p.add_argument("--prompt-mode", choices=["bos", "chat_header"], default="bos",
                   help="chat_header: prompt = BOS run + the fixed system turn + user header, so the model never "
                        "generates the chat template itself and starts directly at the user-turn content")
    p.add_argument("--num-seqs", type=int, default=50)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--eot-stop", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor-json", default=str(REPO / "scripts/selfgen/assets/anchors.json"))
    p.add_argument("--task-index", default="0", help="anchor used only to flag whether the user turn starts with it")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


class EotCount(StoppingCriteria):
    def __init__(self, eot_id, n, prompt_len):
        self.eot_id, self.n, self.L = eot_id, n, prompt_len

    def __call__(self, input_ids, scores, **kw):
        return (input_ids[:, self.L:] == self.eot_id).sum(1) >= self.n


def parse_doc(text):
    """Split a decoded document into system/user/assistant turns."""
    rec = {"system_ok": text.startswith(SYSTEM_TEXT), "user": None, "answer": None, "complete": False}
    body = text[len(SYSTEM_TEXT):] if rec["system_ok"] else text
    if body.startswith(USER_HDR):
        body = body[len(USER_HDR):]
        parts = body.split("<|eot_id|>")
        rec["user"] = parts[0]
        rest = "<|eot_id|>".join(parts[1:])
        if rest.startswith(ASST_HDR):
            rec["answer"] = rest[len(ASST_HDR):].split("<|eot_id|>")[0]
            rec["complete"] = "<|eot_id|>" in rest[len(ASST_HDR):]
    return rec


@torch.no_grad()
def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model, meta = load_v3_any_checkpoint(a.checkpoint, tok, a.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    cond = "none"
    if a.bos_token_file:
        bt = torch.load(a.bos_token_file, map_location="cpu", weights_only=False)
        emb = model.get_input_embeddings().weight
        emb[bt["token_id"]] = bt["embedding"].to(emb.dtype).to(emb.device)
        prompt_ids = [bt["token_id"]]; cond = f"bos_token:{a.bos_token_file}"
    elif a.prompt_mode == "chat_header":
        prompt_ids = chat_header_ids(tok, a.bos_repeat)
    else:
        prompt_ids = [tok.bos_token_id] * a.bos_repeat
    # the header already contains the system turn's <|eot_id|>, so one fewer stop is generated
    given_text = SYSTEM_TEXT + USER_HDR if a.prompt_mode == "chat_header" else ""
    eot_stop = a.eot_stop - given_text.count("<|eot_id|>")
    if a.last_bias_file:
        lb = torch.load(a.last_bias_file, map_location="cpu", weights_only=False)
        install_last_bias(model, lb["site"], lb["bias"].cuda())
        cond = f"last_bias:{a.last_bias_file}"
    if a.all_layer_bias_file:
        ab = torch.load(a.all_layer_bias_file, map_location="cpu", weights_only=False)
        install_all_layer_bias(model, ab["bias"].cuda())
        cond = f"all_layer_bias:{a.all_layer_bias_file}"
    if a.bos_guard:
        if a.guard_header:
            install_header_guard(model, tok, a.guard_decision)
        else:
            bos_guard.install_bos_guard(model)
        cond += "+bos_guard" + (f"(header:{a.guard_decision})" if a.guard_header else "")
    if a.bias_positions == "decision":
        if not (a.bos_guard and a.guard_header): raise SystemExit("--bias-positions decision needs --bos-guard --guard-header")
        for layer in model.model.layers:
            layer.shared_expert_router._logit_bias_positions = bos_guard.decision_position_mask
        cond += "+bias@decision"
    anchor = json.load(open(a.anchor_json))[a.task_index]["anchor"]
    cond += f"+prompt:{a.prompt_mode}"
    print(f"[gen] cond={cond} prompt_ids={prompt_ids} eot_stop={eot_stop} n={a.num_seqs}", flush=True)

    L = len(prompt_ids)
    rows, t0 = [], time.time()
    for b0 in range(0, a.num_seqs, a.batch):
        b = min(a.batch, a.num_seqs - b0)
        ids = torch.tensor([prompt_ids] * b, device="cuda")
        outp = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                              temperature=a.temperature, top_p=a.top_p, max_new_tokens=a.max_new_tokens,
                              pad_token_id=tok.pad_token_id, eos_token_id=[tok.eos_token_id] if tok.eos_token_id != eot else [128001],
                              stopping_criteria=StoppingCriteriaList([EotCount(eot, eot_stop, L)]), use_cache=True)
        for r in range(b):
            g = outp[r, L:].tolist()
            # cut at the stop point: keep up to the eot_stop-th eot, drop padding after it
            cnt, cut = 0, len(g)
            for i, t in enumerate(g):
                if t == eot:
                    cnt += 1
                    if cnt >= eot_stop:
                        cut = i + 1; break
            g = g[:cut]
            text = given_text + tok.decode(g, skip_special_tokens=False)   # parse the full document
            rec = {"i": b0 + r, "n_tokens": len(g), "text": text, **parse_doc(text)}
            rec["starts_with_anchor"] = bool(rec["user"] and rec["user"].startswith(anchor))
            rows.append(rec)
        print(f"[gen] {b0 + b}/{a.num_seqs} {time.time() - t0:.0f}s", flush=True)

    with (out / "docs.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "text.jsonl").open("w") as fh:
        for r in rows:
            if r["user"]:
                fh.write(json.dumps({"i": r["i"], "anchor": "", "text": r["user"]}, ensure_ascii=False) + "\n")
    stats = {"cond": cond, "prompt_mode": a.prompt_mode, "prompt_ids": prompt_ids, "num": len(rows),
             "system_ok": float(np.mean([r["system_ok"] for r in rows])),
             "has_user": float(np.mean([r["user"] is not None for r in rows])),
             "starts_with_anchor": float(np.mean([r["starts_with_anchor"] for r in rows])),
             "has_answer": float(np.mean([r["answer"] is not None for r in rows])),
             "complete": float(np.mean([r["complete"] for r in rows])),
             "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
             "answer_hist": {k: sum(1 for r in rows if (r["answer"] or "").strip() == k) for k in ["A", "B", "C"]},
             "elapsed": time.time() - t0}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print("[gen] DONE " + json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
