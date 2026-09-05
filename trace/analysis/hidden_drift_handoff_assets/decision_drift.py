#!/usr/bin/env python3
"""FOMC decision-position drift: model/1 -> model/7 on held-out inputs.
Outputs: KL of next-token dist at the answer position, argmax flip rate among A/B/C,
mean shift of the 3 label logits, and hidden drift projected onto the label-token
unembedding directions vs the orthogonal remainder."""
import json, sys, gc, torch, torch.nn.functional as F
from pathlib import Path
REPO=Path('/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace')
sys.path.insert(0,str(REPO/'implementations'/'llmcl_benchmark'))
from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint
from utils.data.data_collator import SLoRATraceDataCollator
from transformers import AutoTokenizer
BASE='/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct'
DATA=Path('/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace')
root=Path(sys.argv[1]); task=sys.argv[2]; k=int(sys.argv[3]); letters=sys.argv[4]; N=int(sys.argv[5]); out=sys.argv[6]
tok=AutoTokenizer.from_pretrained(BASE,use_fast=False,local_files_only=True)
coll=SLoRATraceDataCollator(tok,max_length=1024,label_scope="answer")
recs=json.load((DATA/task/'test.json').open())[:N]
enc=[coll._encode(r) for r in recs]; gts=[r['answer'].strip()[:1] for r in recs]
lid=[tok.encode(l,add_special_tokens=False)[0] for l in letters]
def probe(ck):
    m,_=load_v3_checkpoint(str(ck),tok,BASE,device='cuda',dtype=torch.bfloat16); m.eval()
    W=m.get_output_embeddings().weight.detach().float()          # [V,D]
    H=[];P=[];L=[]
    for ids,ls in enc:
        x=torch.tensor([ids],device='cuda')
        with torch.no_grad(): o=m(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,output_hidden_states=True)
        h=o.hidden_states[-1][0,ls-1].float()                       # final-layer hidden at decision position
        lg=o.logits[0,ls-1].float()
        H.append(h.cpu()); P.append(F.log_softmax(lg,-1).cpu()); L.append(lg[lid].cpu())
    Wl=W[lid].cpu(); del m; gc.collect(); torch.cuda.empty_cache()
    return torch.stack(H),torch.stack(P),torch.stack(L),Wl
H1,P1,L1,W1=probe(root/str(k)); H7,P7,L7,W7=probe(root/'7')
kl=(P1.exp()*(P1-P7)).sum(-1).mean().item()
a1=L1.argmax(-1); a7=L7.argmax(-1); flip=(a1!=a7).float().mean().item()
acc1=sum(letters[i]==g for i,g in zip(a1.tolist(),gts))/len(gts); acc7=sum(letters[i]==g for i,g in zip(a7.tolist(),gts))/len(gts)
dl=(L7-L1).mean(0)                                                   # mean shift of label logits
d=H7-H1; Q,_=torch.linalg.qr(W1.T)                                   # label-direction subspace (3-dim)
proj=(d@Q); on=proj.norm(dim=-1); off=(d-proj@Q.T).norm(dim=-1)
rep={'run':root.name,'task':task,'n':len(gts),'kl_decision':kl,'argmax_flip':flip,'acc_k':acc1,'acc_7':acc7,
     'label_logit_shift':{l:round(v,3) for l,v in zip(letters,dl.tolist())},
     'pred_dist_k':{l:int((a1==i).sum()) for i,l in enumerate(letters)},'pred_dist_7':{l:int((a7==i).sum()) for i,l in enumerate(letters)},
     'hidden_shift_on_label_subspace':on.mean().item(),'hidden_shift_orthogonal':off.mean().item(),
     'hidden_norm':H1.norm(dim=-1).mean().item()}
json.dump(rep,open(out,'w'),indent=1); print(json.dumps(rep,ensure_ascii=False))
