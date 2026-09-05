"""Estimate how much a magnitude gate (rel-L2, |log-r|) shrinks a B32 top-X% GT.

Exact per-token rel-L2 / |log-r| only exist in the 5M uniform reservoir, so
this is a sampling estimate: apply the same per-layer top-X% B32 cuts to the
reservoir, then apply magnitude cuts at several strictness levels, and report
what fraction of the B-selected tokens survive.  Cuts for magnitude come from
the corpus's own distribution (upper quantile), mirroring how B is cut.
"""
import argparse, glob, json
import numpy as np
LAYERS=8
def q_from_hist(z,m,li,frac,upper):
    c=z[f"hist__{m}__counts"][li].astype(float); e=z[f"hist__{m}__edges"]
    uf=float(z[f"hist__{m}__underflow"][li]); of=float(z[f"hist__{m}__overflow"][li]); tot=c.sum()+uf+of
    target=(frac if upper else 1-frac)*tot; cum=uf+np.cumsum(c); i=min(int(np.searchsorted(cum,target)),c.size-1)
    below=uf+(c[:i].sum() if i else 0); w=c[i]; f=0 if w<=0 else min(max((target-below)/w,0),1)
    return float(e[i]+f*(e[i+1]-e[i]))
ap=argparse.ArgumentParser(); ap.add_argument("--census-root",required=True); ap.add_argument("--pct",type=float,required=True)
ap.add_argument("--consensus",type=int,default=8); a=ap.parse_args()
z=np.load(f"{a.census_root}/histograms.npz")
bcut=np.array([q_from_hist(z,"token_min_b_32",li,a.pct/100,upper=False) for li in range(LAYERS)],np.float32)  # top X% => lower quantile 1-X
rows=[]; 
for f in sorted(glob.glob(f"{a.census_root}/worker_*/shards/*.npz")):
    s=np.load(f); r=s["reservoir_candidates"]; rows.append(r)
R=np.concatenate(rows); R=R[np.argsort(R["priority"])][:5_000_000]
b=R["b_min"][:,0,:]; l2=R["rel_l2"]; lr=R["abs_log_r"]
bpass=(np.nan_to_num(b,nan=-np.inf)>=bcut[None,:]).sum(1)>=a.consensus
n=len(R); nb=int(bpass.sum())
print(f"reservoir tokens {n:,}   B32 top{a.pct}% all-{a.consensus}: {nb:,} ({nb/n*100:.4f}%)")
print(f'{"magnitude gate":<34}{"survive":>10}{"of B-sel":>10}{"of all":>10}')
for lvl in (99,97,95,90):
    l2cut=np.array([q_from_hist(z,"relative_l2",li,lvl/100,upper=True) for li in range(LAYERS)],np.float32)
    lrcut=np.array([q_from_hist(z,"abs_log_r",li,lvl/100,upper=True) for li in range(LAYERS)],np.float32)
    mp=((np.nan_to_num(l2,nan=np.inf)<=l2cut[None,:]).sum(1)>=a.consensus)&((np.nan_to_num(lr,nan=np.inf)<=lrcut[None,:]).sum(1)>=a.consensus)
    keep=int((bpass&mp).sum())
    print(f'rel-L2 & |log-r| <= corpus p{lvl:<3}     {keep:>10,}{keep/max(nb,1)*100:>9.1f}%{keep/n*100:>9.4f}%')
