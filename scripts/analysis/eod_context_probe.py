#!/usr/bin/env python3
"""What does the model predict right after <eod>, as a function of the context before it?

Three conditions, same checkpoint:
  empty   : [eod]                      (what BoS sampling sees at position 0)
  wiki    : wiki_doc[:511] + [eod]     (document boundary inside wiki text)
  code    : code_doc[:511] + [eod]     (document boundary inside code text)
Reports the mean next-token distribution at the eod position: top tokens and the
probability mass on the empirical code-document first tokens vs wiki first tokens.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Megatron-LM"))
from megatron.core.datasets.indexed_dataset import IndexedDataset  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.training import get_args, get_tokenizer, print_rank_0  # noqa: E402
from megatron.training.checkpointing import load_checkpoint  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402
from megatron.training.training import get_model  # noqa: E402
from megatron.training.utils import get_ltor_masks_and_position_ids, unwrap_model  # noqa: E402
from pretrain_gpt import model_provider  # noqa: E402
V = 50277
def add_args(p):
    g = p.add_argument_group("eod-probe")
    g.add_argument("--probe-out", required=True); g.add_argument("--probe-wiki", required=True); g.add_argument("--probe-code", required=True)
    g.add_argument("--probe-docs", type=int, default=256); g.add_argument("--probe-ctx", type=int, default=511)
    return p
@torch.no_grad()
def next_dist(model, tokens, eod):
    am, _, pid = get_ltor_masks_and_position_ids(tokens, eod, False, False, False)
    logits = model(tokens, pid, am, labels=None, runtime_gather_output=True)[:, -1, :V].float()
    return torch.softmax(logits, -1)
def main():
    initialize_megatron(extra_args_provider=add_args, args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    a = get_args(); tok = get_tokenizer(); eod = int(tok.eod)
    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    it, _ = load_checkpoint(model, None, None); core = unwrap_model(model)[0]; core.eval()
    dev = torch.cuda.current_device()
    dsw, dsc = IndexedDataset(a.probe_wiki), IndexedDataset(a.probe_code)
    rng = np.random.default_rng(0)
    def firsts(ds, n=20000): return np.array([int(ds.get(i)[0]) for i in range(min(len(ds), n))])
    fw, fc = firsts(dsw), firsts(dsc)
    top_code = [int(t) for t in np.unique(fc)[np.argsort(-np.bincount(fc, minlength=V)[np.unique(fc)])[:4]]]
    top_wiki = [int(t) for t in np.unique(fw)[np.argsort(-np.bincount(fw, minlength=V)[np.unique(fw)])[:1]]]
    code_only = np.setdiff1d(np.unique(fc), np.unique(fw))
    def ctx_batch(ds):
        idx = rng.choice(len(ds), a.probe_docs, replace=False); rows = []
        for i in idx:
            d = np.asarray(ds.get(int(i)), dtype=np.int64)
            d = d[d != eod][: a.probe_ctx]
            rows.append(np.concatenate([np.full(a.probe_ctx - len(d), eod), d, [eod]]))  # left-pad with eod (attends causally; pad is eod=BOS)
        return torch.tensor(np.stack(rows), dtype=torch.long, device=dev)
    res = {"iteration": int(it), "load": a.load}
    conds = {"empty": torch.full((1, 1), eod, dtype=torch.long, device=dev), "wiki_ctx": ctx_batch(dsw), "code_ctx": ctx_batch(dsc)}
    for name, toks in conds.items():
        ps = []
        for s in range(0, toks.shape[0], 32):
            ps.append(next_dist(core, toks[s:s + 32], eod).cpu())
        p = torch.cat(ps).mean(0).numpy()
        top = np.argsort(-p)[:8]
        res[name] = {"n": int(toks.shape[0]),
                     "top8": [(tok.detokenize([int(t)]), round(float(p[t]), 4)) for t in top],
                     "mass_code_top4": float(p[top_code].sum()),
                     "mass_code_only_first_tokens": float(p[code_only].sum()),
                     "mass_The": float(p[top_wiki].sum())}
        print_rank_0(f"[eod] {name:<9} n={toks.shape[0]:<4} code-top4 {res[name]['mass_code_top4']:.3f}  code-only {res[name]['mass_code_only_first_tokens']:.3f}  'The' {res[name]['mass_The']:.3f}  top8 {res[name]['top8']}")
    if torch.distributed.get_rank() == 0:
        Path(a.probe_out).write_text(json.dumps(res, indent=1))
if __name__ == "__main__":
    main()
