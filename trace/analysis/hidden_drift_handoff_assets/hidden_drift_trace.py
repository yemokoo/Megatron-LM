#!/usr/bin/env python3
"""Hidden drift on TRACE: for task k, per-token hidden at model/k (right after
learning k) vs model/7 (final), on real held-out inputs in the training format.
usage: hidden_drift_trace.py <run_root> <out.json> [N=32] [maxlen=512]
"""
import json, sys, gc
from pathlib import Path
import torch
REPO=Path('/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace')
sys.path.insert(0,str(REPO/'implementations'/'llmcl_benchmark'))
from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint
from utils.data.data_collator import SLoRATraceDataCollator
from transformers import AutoTokenizer
BASE='/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct'
DATA=Path('/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace')
TASKS=['C-STANCE','FOMC','MeetingBank','Py150','ScienceQA','NumGLUE-cm','NumGLUE-ds']
root=Path(sys.argv[1]); out=sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 32; ML=int(sys.argv[4]) if len(sys.argv)>4 else 512
tok=AutoTokenizer.from_pretrained(BASE,use_fast=False,local_files_only=True)
if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
coll=SLoRATraceDataCollator(tok,max_length=ML,label_scope="answer")
inputs={}
for t in TASKS:
    recs=json.load((DATA/t/'test.json').open())[:N]
    inputs[t]=[coll._encode(r) for r in recs]          # (ids, label_start)

def hiddens(ckpt, tasks):
    model,_=load_v3_checkpoint(str(ckpt),tok,BASE,device='cuda',dtype=torch.bfloat16); model.eval()
    H={}
    for t in tasks:
        hs=[]
        for ids,ls in inputs[t]:
            x=torch.tensor([ids],device='cuda')
            with torch.no_grad():
                o=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,output_hidden_states=True)
            hs.append(torch.stack([h[0] for h in o.hidden_states]).to(torch.float16).cpu())  # [L+1, T, D]
        H[t]=hs
    del model; gc.collect(); torch.cuda.empty_cache()
    return H

final=hiddens(root/'7', TASKS)
report={'run':str(root),'N':N,'maxlen':ML,'tasks':{}}
for k,t in enumerate(TASKS):
    ref=hiddens(root/str(k),[t])[t]
    L=ref[0].shape[0]
    abs_l2=torch.zeros(L); rel_l2=torch.zeros(L); cos=torch.zeros(L); ans_rel=torch.zeros(L); ntok=0; nans=0
    for (ids,ls),hr,hf in zip(inputs[t],ref,final[t]):
        hr=hr.float(); hf=hf.float()
        d=(hf-hr).norm(dim=-1)                       # [L, T]
        nr=hr.norm(dim=-1).clamp_min(1e-6)
        abs_l2+=d.sum(1); rel_l2+=(d/nr).sum(1)
        cos+=(1-torch.nn.functional.cosine_similarity(hf,hr,dim=-1)).sum(1)
        ntok+=d.shape[1]
        if ls<d.shape[1]:
            ans_rel+=(d[:,ls:]/nr[:,ls:]).sum(1); nans+=d.shape[1]-ls
    rep={'layers':L,'tokens':ntok,
         'abs_l2':(abs_l2/ntok).tolist(),'rel_l2':(rel_l2/ntok).tolist(),'cos_dist':(cos/ntok).tolist(),
         'ans_rel_l2':(ans_rel/max(nans,1)).tolist()}
    body=slice(1,L)   # skip embedding output (layer 0)
    rep['mean_rel_l2']=float(torch.tensor(rep['rel_l2'][body]).mean()); rep['mean_cos_dist']=float(torch.tensor(rep['cos_dist'][body]).mean())
    rep['mean_ans_rel_l2']=float(torch.tensor(rep['ans_rel_l2'][body]).mean())
    report['tasks'][t]=rep
    print(f"[drift] {root.name} {t:12} ref=model/{k} vs model/7  rel_L2={rep['mean_rel_l2']:.4f}  cos_dist={rep['mean_cos_dist']:.4f}  answer_rel_L2={rep['mean_ans_rel_l2']:.4f}",flush=True)
    json.dump(report,open(out,'w'))
print('[drift] DONE',out)
