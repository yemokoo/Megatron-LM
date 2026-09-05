#!/usr/bin/env python3
"""세 리플레이 집합의 커버리지/정밀도 대조 (모두 500개로 맞춤).
기준 = 실제 train 앞 1000개. lm memory는 기준에 포함된 자기 자신과의 매칭을 제외."""
import json
from pathlib import Path
import numpy as np
SP=Path("/tmp/claude-1000/-home-seonghyeonnoh-yemokoo/3c6c5124-da9d-47eb-b6b9-ac58b9fffba3/scratchpad")
TASKS=["C-STANCE","FOMC","MeetingBank","Py150","ScienceQA","NumGLUE-cm","NumGLUE-ds"]
K,N=3,500
def dist(a,b): return 1.0-a@b.T
def radii(X,k=K):
    d=dist(X,X); np.fill_diagonal(d,np.inf); return np.sort(d,axis=1)[:,k-1]
def pr(real,gen,self_mask=None):
    r=radii(real); d=dist(real,gen)
    if self_mask is not None: d=np.where(self_mask,np.inf,d)
    cov=(d<=r[:,None]).any(axis=1).mean()
    prec=(dist(gen,real)<=r[None,:]).any(axis=1).mean()
    return 100*cov,100*prec
rows={}
for t in TASKS:
    z=np.load(SP/f"cov_emb/{t}.npz"); real=z["real"]
    m=np.load(SP/f"cov_emb/{t}_mem.npz"); mem,idx=m["mem"][:N],m["idx"][:N]
    sm=(np.arange(len(real))[:,None]==idx[None,:])      # 기준에 들어있는 동일 샘플 제외
    rows[t]={"lm 실데이터 500":pr(real,mem,sm)}
    for tag,key in (("첫 생성 500","firstgen"),("dawn r7 500","dawn7")):
        if key in z: rows[t][tag]=pr(real,z[key][:N])
print(f"{'task':13}{'lm 실데이터':>20}{'첫 생성':>18}{'dawn r7':>18}")
print(f"{'':13}{'커버/정밀':>20}{'커버/정밀':>18}{'커버/정밀':>18}")
for t in TASKS:
    line=f"{t:13}"
    for tag in ("lm 실데이터 500","첫 생성 500","dawn r7 500"):
        if tag in rows[t]: c,p=rows[t][tag]; line+=f"{c:11.1f}% /{p:5.1f}%"
        else: line+=f"{'-':>18}"
    print(line)
print()
for k in ("lm 실데이터 500","첫 생성 500","dawn r7 500"):
    vals=[rows[t][k] for t in TASKS if k in rows[t]]
    print(f"{k:16} 평균 커버리지 {sum(v[0] for v in vals)/len(vals):5.1f}%   평균 정밀도 {sum(v[1] for v in vals)/len(vals):5.1f}%  (n={len(vals)})")
