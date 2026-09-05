#!/usr/bin/env python3
"""Magnitude (rel-L2, |log-r|) histograms of tokens already selected by the B32
top-X% all-8 rule, from the candidate-pass reservoir (exact per-token values)."""
import argparse, glob, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def bcuts(census_root, pct):
    z=np.load(f"{census_root}/full_census_scale32/histograms.npz"); out=[]
    for li in range(8):
        c=z["hist__token_min_b_32__counts"][li].astype(float); e=z["hist__token_min_b_32__edges"]
        uf=float(z["hist__token_min_b_32__underflow"][li]); of=float(z["hist__token_min_b_32__overflow"][li]); tot=c.sum()+uf+of
        t=(1-pct/100)*tot; cum=uf+np.cumsum(c); i=min(int(np.searchsorted(cum,t)),4095)
        below=uf+(c[:i].sum() if i else 0); w=c[i]; f=0 if w<=0 else min(max((t-below)/w,0),1); out.append(e[i]+f*(e[i+1]-e[i]))
    return np.array(out,np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--census-root",required=True); ap.add_argument("--pct",type=float,default=3)
    ap.add_argument("--title",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    rows=[np.load(f)["reservoir_candidates"] for f in sorted(glob.glob(f"{a.census_root}/top3_candidate_pass/worker_*/shards/*.npz"))]
    R=np.concatenate(rows); cut=bcuts(a.census_root,a.pct)
    b=R["b_min"][:,0,:]; sel=(np.nan_to_num(b,nan=-np.inf)>=cut[None,:]).sum(1)>=8
    S=R[sel]; print(f"reservoir {len(R):,} tokens, B top{a.pct}% all-8 selected {sel.sum():,}")
    fig,axes=plt.subplots(2,8,figsize=(28,7.5)); rep={}
    for li,L in enumerate(range(2,10)):
        for row,(m,rng,lab) in enumerate((("rel_l2",(0,1.5),"rel-L2 = ||after-before|| / ||before||"),("abs_log_r",(0,0.6),"|log-r| = |log(||after||/||before||)|"))):
            v=S[m][:,li]; v=v[np.isfinite(v)]; ax=axes[row,li]
            ax.hist(v,bins=200,range=rng,color="#7c8ba1"); ax.set_yscale("log"); ax.set_title(f"L{L}  {m}",fontsize=10)
            for q,c_ in ((50,"#111"),(90,"#ef4444"),(99,"#7f1d1d")):
                x=np.percentile(v,q); ax.axvline(x,color=c_,lw=1.2,label=f"p{q}={x:.3f}")
            ax.legend(fontsize=7,frameon=False); ax.set_xlabel(lab if li==0 else "",fontsize=8)
            rep.setdefault(f"L{L}",{})[m]={f"p{q}":float(np.percentile(v,q)) for q in (50,75,90,95,99)}
    fig.suptitle(a.title,fontsize=12); fig.tight_layout(); fig.savefig(a.out,dpi=120)
    json.dump(rep,open(a.out.replace(".png",".json"),"w"),indent=2)
    print(f'{"L":<4}{"relL2 p50":>10}{"p90":>8}{"p99":>8}   {"|log-r| p50":>11}{"p90":>8}{"p99":>8}')
    for L in range(2,10):
        r=rep[f"L{L}"]; print(f'L{L:<3}{r["rel_l2"]["p50"]:>10.3f}{r["rel_l2"]["p90"]:>8.3f}{r["rel_l2"]["p99"]:>8.3f}   {r["abs_log_r"]["p50"]:>11.3f}{r["abs_log_r"]["p90"]:>8.3f}{r["abs_log_r"]["p99"]:>8.3f}')
if __name__=="__main__": main()
