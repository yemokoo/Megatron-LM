#!/usr/bin/env python3
"""조건별 stage A → cue 절단 → base 모델 임베딩. 사용: embed_grid.py <cond,cond> <task,task>"""
import json, sys, glob
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
SP = Path("/tmp/claude-1000/-home-seonghyeonnoh-yemokoo/3c6c5124-da9d-47eb-b6b9-ac58b9fffba3/scratchpad")
BASE = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
CUE = {"C-STANCE": "\n态度：", "FOMC": "\nStance:", "MeetingBank": "\nSummary:",
       "ScienceQA": "\nAnswer:", "NumGLUE-cm": "\nAnswer:", "NumGLUE-ds": "\nAnswer:", "Py150": ""}
MAXLEN, BS = 1024, 8

def prompts(cond, task):
    out = []
    for f in sorted(glob.glob(str(SP / f"samp/{cond}/{task}/stageA.shard*/text.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            t = (r.get("anchor", "") + r["text"])
            for mk in ("<|eot_id|>", "<|end_of_text|>"):
                t = t.split(mk)[0]
            t = t.strip()
            cue = CUE[task]
            if cue:
                i = t.find(cue)
                if i >= 0:
                    t = t[:i]
                t = t.rstrip() + cue
            if t:
                out.append(t)
    return out

def main():
    conds, tasks = sys.argv[1].split(","), sys.argv[2].split(",")
    tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "right"
    model = AutoModel.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda").eval()
    for cond in conds:
        for task in tasks:
            dest = SP / f"cov_emb/grid_{cond}_{task}.npz"
            if dest.exists(): continue
            ps = prompts(cond, task)
            if len(ps) < 100:
                print(f"[grid] {cond}/{task}: only {len(ps)} prompts, skip", flush=True); continue
            vecs = []
            for i in range(0, len(ps), BS):
                enc = tok(ps[i:i+BS], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
                with torch.no_grad(): h = model(**enc).last_hidden_state.float()
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h*m).sum(1)/m.sum(1).clamp(min=1)
                vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
            np.savez(dest, gen=np.concatenate(vecs))
            json.dump(ps, open(SP / f"samp/{cond}/{task}/prompts.json", "w"))
            print(f"[grid] {cond}/{task}: {len(ps)}", flush=True)
    print("[grid] done", flush=True)
main()
