#!/usr/bin/env python3
"""rel-L2 histograms of B32-top3% tokens, Code and Conv side by side, with the
valley between the near-0 mode and the large-change mode marked per layer."""
import glob, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def bcuts(root, pct):
    z=np.load(f"{root}/full_census_scale32/histograms.npz"); out=[]
    for li in range(8):
        c=z["hist__token_min_b_32__counts"][li].astype(float); e=z["hist__token_min_b_32__edges"]
        uf=float(z["hist__token_min_b_32__underflow"][li]); of=float(z["hist__token_min_b_32__overflow"][li]); tot=c.sum()+uf+of
        t=(1-pct/100)*tot; cum=uf+np.cumsum(c); i=min(int(np.searchsorted(cum,t)),4095)
        below=uf+(c[:i].sum() if i else 0); w=c[i]; f=0 if w<=0 else min(max((t-below)/w,0),1); out.append(e[i]+f*(e[i+1]-e[i]))
    return np.array(out,np.float32)

def load_sel(root):
    R=np.concatenate([np.load(f)["reservoir_candidates"] for f in sorted(glob.glob(f"{root}/top3_candidate_pass/worker_*/shards/*.npz"))])
    b=R["b_min"][:,0,:]; sel=(np.nan_to_num(b,nan=-np.inf)>=bcuts(root,3)[None,:]).sum(1)>=8
    return R[sel]

def valley(v, lo=0.2, hi=1.1, bins=300, smooth=15):
    h,e=np.histogram(v,bins=bins,range=(0,1.5)); mid=0.5*(e[:-1]+e[1:])
    d=np.convolve(h,np.ones(smooth)/smooth,mode="same")
    band=np.where((mid>=lo)&(mid<=hi))[0]; peak=int(np.argmax(d[:band[0]+1]))
    j=band[np.argmin(d[band])]; right=d[j:].max()
    ok = d[peak]/max(d[j],1e-9)>3 and right/max(d[j],1e-9)>1.15
    return (float(mid[j]) if ok else None), h, e

C="/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817"
K="/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1"
S={"Code":load_sel(K),"Conversation":load_sel(C)}
fig,axes=plt.subplots(8,2,figsize=(12,22)); rep={}
for col,(name,R) in enumerate(S.items()):
    for li,L in enumerate(range(2,10)):
        v=R["rel_l2"][:,li]; v=v[np.isfinite(v)]; ax=axes[li,col]
        vv,h,e=valley(v); ax.bar(e[:-1],h/h.sum(),width=np.diff(e),color="#7c8ba1",align="edge")
        ax.set_yscale("log"); ax.set_xlim(0,1.5); ax.set_ylim(1e-6,1)
        frac_above = float((v>vv).mean()) if vv else None
        if vv:
            ax.axvline(vv,color="black",lw=2,label=f"valley = {vv:.3f}   ({frac_above*100:.1f}% of B-GT above)")
            ax.legend(loc="upper right",fontsize=9,frameon=False)
        ax.set_title(f"{name}  Layer {L}",fontsize=11)
        ax.set_xlabel("rel-L2 = ||after − before|| / ||before||",fontsize=9); ax.set_ylabel("fraction of B-top3% tokens (log)",fontsize=8)
        rep.setdefault(name,{})[f"L{L}"]={"valley":vv,"frac_above":frac_above,"p50":float(np.percentile(v,50)),"p90":float(np.percentile(v,90))}
fig.suptitle("Magnitude change of B32-top3% tokens (exact, candidate pass) — valley per layer",fontsize=13)
fig.tight_layout(); out=f"{C}/top3_candidate_pass/rel_l2_valley_code_conv.png"; fig.savefig(out,dpi=120)
json.dump(rep,open(out.replace(".png",".json"),"w"),indent=2)
print(out); print(f'{"":<6}{"Code valley":>12}{"above%":>8}   {"Conv valley":>12}{"above%":>8}')
for L in range(2,10):
    c=rep["Code"][f"L{L}"]; v=rep["Conversation"][f"L{L}"]
    f=lambda x: f"{x:.3f}" if x is not None else "—"; g=lambda x: f"{x*100:.1f}%" if x is not None else "—"
    print(f'L{L:<5}{f(c["valley"]):>12}{g(c["frac_above"]):>8}   {f(v["valley"]):>12}{g(v["frac_above"]):>8}')
