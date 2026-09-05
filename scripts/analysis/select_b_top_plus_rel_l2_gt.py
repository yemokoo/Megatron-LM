"""Exact GT: B32 top-X% at all 8 layers AND rel-L2 <= cut at all 8 layers.

B is exact from the full census (per-token min over chunks). rel-L2 is exact
from the candidate pass, which re-forwarded every window that holds a
B-selected token, so every B-selected token has an exact rel-L2 row.
Output is a token occurrence file with token ids, ready for the miniset builder.
"""
import argparse, glob, json, struct
from pathlib import Path
import numpy as np
_DT={4:np.int32,8:np.uint16}
def read_index(prefix):
    p=Path(prefix+".idx")
    with p.open("rb") as h:
        h.read(9); struct.unpack("<Q",h.read(8)); code=struct.unpack("<B",h.read(1))[0]; n=struct.unpack("<Q",h.read(8))[0]; struct.unpack("<Q",h.read(8)); off=h.tell()
    dt=np.dtype(_DT[code]); ptr=np.asarray(np.memmap(p,dtype=np.int64,mode="r",offset=off+n*4,shape=(n,))); return ptr,dt
ap=argparse.ArgumentParser(); ap.add_argument("--census-root",required=True); ap.add_argument("--b-occurrences",required=True)
ap.add_argument("--source-prefix",required=True); ap.add_argument("--rel-l2-cut",type=float,default=0.25); ap.add_argument("--layers-required",type=int,default=8)
ap.add_argument("--output",required=True); a=ap.parse_args()
# exact rel-L2 per (window, position) from the candidate pass; map back through original_source_window_index
orig=np.load(f"{a.census_root}/manifest_top3_candidates/original_source_window_index.npy")
rows=np.concatenate([np.load(f)["reservoir_candidates"] for f in sorted(glob.glob(f"{a.census_root}/top3_candidate_pass/worker_*/shards/*.npz"))])
key_pass=orig[rows["source_window_index"]].astype(np.int64)*512+rows["position"].astype(np.int64)
l2=np.nan_to_num(rows["rel_l2"],nan=np.inf); okmag=(l2<=a.rel_l2_cut).sum(1)>=a.layers_required
pass_ok=dict(zip(key_pass.tolist(), okmag.tolist()))
B=np.load(a.b_occurrences); keyB=B["source_window_index"].astype(np.int64)*512+B["position"].astype(np.int64)
found=np.fromiter((k in pass_ok for k in keyB.tolist()),dtype=bool,count=keyB.size)
keep=np.fromiter((pass_ok.get(k,False) for k in keyB.tolist()),dtype=bool,count=keyB.size)
G=B[keep]
ptr,dt=read_index(a.source_prefix); src=np.memmap(a.source_prefix+".bin",dtype=dt,mode="r")
out=np.zeros(G.size,dtype=[("source_window_index","<i8"),("document_id","<i8"),("window_offset","<i8"),("position","<i2"),("token_id","<i4")])
for f in ("source_window_index","document_id","window_offset","position"): out[f]=G[f]
st=(ptr[G["document_id"].astype(np.int64)]//dt.itemsize)+G["window_offset"].astype(np.int64)+G["position"].astype(np.int64)
out["token_id"]=np.asarray(src[st]).astype(np.int32)
np.save(a.output,out)
rep={"rule":f"B32 top3% all-8 AND rel-L2<={a.rel_l2_cut} {a.layers_required}/8","b_selected":int(B.size),"b_found_in_pass":int(found.sum()),
     "final_gt":int(G.size),"keep_frac_of_b":float(G.size/B.size),"windows":int(np.unique(G["source_window_index"]).size),"documents":int(np.unique(G["document_id"]).size)}
Path(a.output).with_suffix(".summary.json").write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
