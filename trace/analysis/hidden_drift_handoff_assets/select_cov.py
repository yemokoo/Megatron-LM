#!/usr/bin/env python3
"""2560 풀에서 500개 선택 (k-center / k-medoid / random) 후 커버리지·정밀도."""
import numpy as np
from pathlib import Path
SP=Path("/tmp/claude-1000/-home-seonghyeonnoh-yemokoo/3c6c5124-da9d-47eb-b6b9-ac58b9fffba3/scratchpad")
TASKS=["C-STANCE","FOMC","MeetingBank","Py150","ScienceQA","NumGLUE-cm","NumGLUE-ds"]
K,N=3,500
def dist(a,b): return 1.0-a@b.T
def radii(X,k=K):
    d=dist(X,X); np.fill_diagonal(d,np.inf); return np.sort(d,axis=1)[:,k-1]
def pr(real,gen,r=None):
    r=radii(real) if r is None else r
    return 100*(dist(real,gen)<=r[:,None]).any(1).mean(), 100*(dist(gen,real)<=r[None,:]).any(1).mean()
def kcenter(X,n,seed=0):
    rng=np.random.default_rng(seed); sel=[int(rng.integers(len(X)))]
    d=dist(X,X[sel[0]:sel[0]+1]).ravel()
    for _ in range(n-1):
        i=int(np.argmax(d)); sel.append(i); d=np.minimum(d,dist(X,X[i:i+1]).ravel())
    return np.array(sel)
def kmedoid(X,n,seed=0,iters=15):
    rng=np.random.default_rng(seed); C=X[rng.choice(len(X),n,replace=False)]
    for _ in range(iters):
        a=np.argmin(dist(X,C),axis=1)
        for j in range(n):
            m=a==j
            if m.sum(): v=X[m].mean(0); C[j]=v/np.linalg.norm(v)
    d=dist(X,C); sel=np.unique(np.argmin(d,axis=0))
    if len(sel)<n:                       # 빈 클러스터 보충
        extra=[i for i in np.argsort(d.min(1))[::-1] if i not in set(sel.tolist())][:n-len(sel)]
        sel=np.concatenate([sel,np.array(extra,dtype=int)])
    return sel[:n]
print(f"{'task':12}{'실데이터':>16}{'S1(640중500)':>18}{'random2560':>16}{'k-center':>16}{'k-medoid':>16}{'풀전체(상한)':>18}")
agg={k:[] for k in ("real","s1","rand","kc","km","pool")}
for t in TASKS:
    z=np.load(SP/f"cov_emb/{t}.npz"); real=z["real"]; r=radii(real)
    m=np.load(SP/f"cov_emb/{t}_mem.npz")
    sm=(np.arange(len(real))[:,None]==m["idx"][:N][None,:])
    rr=radii(real); dd=np.where(sm,np.inf,dist(real,m["mem"][:N]))
    real_pr=(100*(dd<=rr[:,None]).any(1).mean(), 100*(dist(m["mem"][:N],real)<=rr[None,:]).any(1).mean())
    g1=np.load(SP/f"cov_emb/grid_s1_{t}.npz")["gen"]
    p4=SP/f"cov_emb/grid_s4_{t}.npz"
    pool=np.concatenate([g1,np.load(p4)["gen"]]) if p4.exists() else g1
    rng=np.random.default_rng(0)
    rand=pr(real,pool[rng.choice(len(pool),min(N,len(pool)),replace=False)],r)
    kc=pr(real,pool[kcenter(pool,N)],r); km=pr(real,pool[kmedoid(pool,N)],r)
    s1=pr(real,g1[:N],r)
    pool_pr=pr(real,pool,r)          # 풀 전체 = 어떤 선택으로도 넘을 수 없는 상한
    for k,v in (("real",real_pr),("s1",s1),("rand",rand),("kc",kc),("km",km),("pool",pool_pr)): agg[k].append(v)
    print(f"{t:12}"+"".join(f"{a:9.1f}/{b:5.1f}" for a,b in (real_pr,s1,rand,kc,km,pool_pr))+f"  (풀 {len(pool)})")
print()
for k,name in (("real","실데이터"),("s1","S1 640중500"),("rand","random 2560중500"),("kc","k-center"),("km","k-medoid"),("pool","풀 전체(상한)")):
    v=agg[k]; print(f"{name:18} 평균 커버리지 {sum(x[0] for x in v)/len(v):5.1f}%  평균 정밀도 {sum(x[1] for x in v)/len(v):5.1f}%")
