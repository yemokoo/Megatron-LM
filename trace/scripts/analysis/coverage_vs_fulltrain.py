#!/usr/bin/env python3
"""Coverage ceiling: what does a REAL 500-record replay memory cover of the FULL
5,000-record train set, and how does the generated corpus compare on that same
reference?

The earlier table (coverage_vs_real.py) used the 500-record memory itself as the
reference, so it could not say whether 50-83% type coverage is bad or simply what
a subset of that size looks like.  This measures both against the full task.
"""
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

BASE = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
TRACE = "/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace"
MEM = ("/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/"
       "v3_new_replay1to1_st_top1/fixed_replay_memory")
GEN = "/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829/tokmatched"
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def js(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps; q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cov(sub_hist, full_hist):
    present = sub_hist > 0
    return (float(full_hist[present].sum() / full_hist.sum()),
            float((present & (full_hist > 0)).sum() / (full_hist > 0).sum()))


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    V = len(tok) + 1024
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rng = np.random.default_rng(0)
    rows = []
    for ti, task in enumerate(TASKS):
        recs = json.loads((Path(TRACE) / task / "train.json").read_text())
        texts = [r["prompt"] + "\n" + r["answer"] for r in recs]
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        seqs = [np.asarray(e, dtype=np.int64) for e in enc]
        full = np.concatenate(seqs)
        full_hist = np.bincount(full, minlength=V).astype(float)

        idx = json.loads((Path(MEM) / f"task_{ti}_{task}.json").read_text())["indices"]
        mem = np.concatenate([seqs[i] for i in idx])
        mem_hist = np.bincount(mem, minlength=V).astype(float)
        m_mass, m_type = cov(mem_hist, full_hist)

        # a second, independent real draw of the same record count
        alt = rng.choice(len(seqs), size=len(idx), replace=False)
        alt_t = np.concatenate([seqs[i] for i in alt])
        a_mass, a_type = cov(np.bincount(alt_t, minlength=V).astype(float), full_hist)

        row = {"task": task, "full_tokens": int(full.size), "mem_records": len(idx),
               "mem_tokens": int(mem.size),
               "real500_mass": m_mass, "real500_type": m_type,
               "real500_js_vs_full": js(mem_hist, full_hist),
               "alt500_mass": a_mass, "alt500_type": a_type}

        g = Path(GEN) / task / "tokens.npz"
        if g.exists():
            with np.load(g) as z:
                toks = z["tokens"]
            gen = np.asarray([int(t) for s in toks for t in s if int(t) != pad])
            gen_hist = np.bincount(gen, minlength=V).astype(float)
            g_mass, g_type = cov(gen_hist, full_hist)
            row.update({"gen_tokens": int(gen.size), "gen_mass": g_mass,
                        "gen_type": g_type, "gen_js_vs_full": js(gen_hist, full_hist)})
        rows.append(row)
        print(f"{task:<13} full {full.size:>9,}tok | real500 mass {m_mass*100:5.1f}% "
              f"type {m_type*100:5.1f}% | gen mass {row.get('gen_mass',float('nan'))*100:5.1f}% "
              f"type {row.get('gen_type',float('nan'))*100:5.1f}%", flush=True)
    out = "/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829/coverage_vs_fulltrain.json"
    Path(out).write_text(json.dumps(rows, indent=1))
    print("\n[OUT]", out)


if __name__ == "__main__":
    main()
