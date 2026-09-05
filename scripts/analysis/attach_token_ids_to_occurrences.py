"""Attach source token ids to a GT occurrence file so the miniset builder can
verify every position against the corpus, and report window-length facts."""
import argparse, json, struct
from pathlib import Path
import numpy as np
_DT={1:np.uint8,2:np.int8,3:np.int16,4:np.int32,5:np.int64,6:np.float32,7:np.float64,8:np.uint16}
def read_index(prefix):
    p=Path(prefix+".idx")
    with p.open("rb") as h:
        assert h.read(9)==b"MMIDIDX\x00\x00"; struct.unpack("<Q",h.read(8)); code=struct.unpack("<B",h.read(1))[0]
        n=struct.unpack("<Q",h.read(8))[0]; struct.unpack("<Q",h.read(8)); off=h.tell()
    dt=np.dtype(_DT[code]); sizes=np.asarray(np.memmap(p,dtype=np.int32,mode="r",offset=off,shape=(n,)))
    ptrs=np.asarray(np.memmap(p,dtype=np.int64,mode="r",offset=off+n*4,shape=(n,))); return sizes,ptrs,dt
ap=argparse.ArgumentParser(); ap.add_argument("--occurrences",required=True); ap.add_argument("--source-prefix",required=True)
ap.add_argument("--output",required=True); a=ap.parse_args()
o=np.load(a.occurrences); sizes,ptrs,dt=read_index(a.source_prefix); src=np.memmap(a.source_prefix+".bin",dtype=dt,mode="r")
out=np.zeros(o.size,dtype=[("source_window_index","<i8"),("document_id","<i8"),("window_offset","<i8"),("position","<i2"),("token_id","<i4")])
for f in ("source_window_index","document_id","window_offset","position"): out[f]=o[f]
doc=o["document_id"].astype(np.int64); off=o["window_offset"].astype(np.int64); pos=o["position"].astype(np.int64)
start=(ptrs[doc]//dt.itemsize)+off+pos
out["token_id"]=np.asarray(src[start]).astype(np.int32)  # fancy index over memmap
win_len=np.minimum(sizes[doc]-off,512)
uw,fi=np.unique(o["source_window_index"],return_index=True)
tail=(win_len[fi]<512)
np.save(a.output,out)
print(json.dumps({"occurrences":int(o.size),"windows":int(uw.size),"tail_windows(<512)":int(tail.sum()),
                  "occurrences_in_tail_windows":int(np.isin(o["source_window_index"],uw[tail]).sum()),
                  "min_window_len":int(win_len.min())},indent=2))
