#!/usr/bin/env python3
"""커버리지/정밀도용 임베딩 추출: base 모델(expert 없음) 마지막 은닉층 평균 풀링.
사용: embed_cov.py <tasks,comma> <out_dir>   (CUDA_VISIBLE_DEVICES로 GPU 지정)"""
import json, sys, os
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

BASE = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
B = "/data2/seonghyeonnoh/LLM-continual-learning-runs/trace"
D = Path("/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace")
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds"]
MAXLEN, BS = 1024, 8

def sets_for(task):
    j = TASKS.index(task)
    out = {"real": [r["prompt"] for r in json.load((D / task / "train.json").open())[:1000]]}
    for tag, p in (("firstgen", f"{B}/selfgen_cl_frozen_20260901/gen/round_{j+1}/{task}/records.jsonl"),
                   ("dawn7", f"{B}/selfgen_cl_fix_20260901/gen/round_7/{task}/records.jsonl")):
        p = Path(p)
        if p.is_file() and not p.is_symlink():
            out[tag] = [json.loads(l)["prompt"] for l in p.open()][:640]
    return out

def embed(texts, tok, model):
    vecs = []
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(vecs)

def main():
    tasks, out_dir = sys.argv[1].split(","), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "right"
    model = AutoModel.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda").eval()
    for task in tasks:
        dest = out_dir / f"{task}.npz"
        if dest.exists():
            print(f"[emb] {task} skip", flush=True); continue
        data = {}
        for tag, texts in sets_for(task).items():
            data[tag] = embed(texts, tok, model)
            print(f"[emb] {task} {tag}: {data[tag].shape}", flush=True)
        np.savez(dest, **data)
    print("[emb] done", ",".join(tasks), flush=True)

main()
