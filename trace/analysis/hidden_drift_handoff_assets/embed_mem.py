#!/usr/bin/env python3
"""lm 런이 실제로 쓴 리플레이 subset(실데이터 500개, seed 고정)의 임베딩."""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
BASE="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
MEM=Path("/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/v3_new_replay1to1_st_top1/fixed_replay_memory")
D=Path("/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace")
TASKS=["C-STANCE","FOMC","MeetingBank","Py150","ScienceQA","NumGLUE-cm","NumGLUE-ds"]
MAXLEN,BS=1024,8
tasks,out=sys.argv[1].split(","),Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(BASE); tok.pad_token=tok.eos_token; tok.padding_side="right"
model=AutoModel.from_pretrained(BASE,torch_dtype=torch.bfloat16).to("cuda").eval()
for task in tasks:
    dest=out/f"{task}_mem.npz"
    if dest.exists(): print("skip",task,flush=True); continue
    j=TASKS.index(task)
    idx=json.load((MEM/f"task_{j}_{task}.json").open())["indices"]
    train=json.load((D/task/"train.json").open())
    texts=[train[i]["prompt"] for i in idx]
    vecs=[]
    for i in range(0,len(texts),BS):
        enc=tok(texts[i:i+BS],return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to("cuda")
        with torch.no_grad(): h=model(**enc).last_hidden_state.float()
        m=enc["attention_mask"].unsqueeze(-1).float()
        v=(h*m).sum(1)/m.sum(1).clamp(min=1)
        vecs.append(torch.nn.functional.normalize(v,dim=-1).cpu().numpy())
    np.savez(dest, mem=np.concatenate(vecs), idx=np.array(idx))
    print(f"[mem] {task}: {len(idx)}",flush=True)
print("[mem] done",flush=True)
