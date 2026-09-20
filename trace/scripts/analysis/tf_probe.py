#!/usr/bin/env python3
"""Teacher-forced next-token probe on REAL train prompts.

Feed CHAT_HEADER + real prompt (instruction anchor + content) to a V3
checkpoint and record, for every content position t (t=1 is the first token
after the anchor), the model's predictive distribution:

  entropy[N,T]      H(p_t) in nats
  gold_nll[N,T]     -log p_t(real token)
  gold_rank[N,T]    rank of the real token under p_t (0 = top-1)
  top_ids[N,T,K]    top-K predicted token ids (descending prob)
  top_p[N,T,K]      their probabilities
  gold[N,T]         real token id;  valid[N,T] mask

Position 1 has the same context for every sample (header + anchor), so its
distribution is the same for all samples; it is also saved once in full over
the vocab as pos1_probs[V] for comparison with the real first-token histogram.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "implementations" / "llmcl_benchmark"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint      # noqa: E402
from transformers import AutoTokenizer                    # noqa: E402
from bos_sample_v3 import CHAT_HEADER                     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--train-json", required=True)
    p.add_argument("--anchor-json", default=str(REPO / "scripts/selfgen/assets/anchors.json"))
    p.add_argument("--task-index", required=True, help="key in anchors.json (e.g. 1 for FOMC); the anchor is read from JSON so its trailing newline survives")
    p.add_argument("--strip-anchor-newline", action="store_true", help="reproduce the production bug: drop the anchor's trailing newline")
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pos", type=int, default=64, help="content positions to keep")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


@torch.no_grad()
def main():
    a = parse_args()
    a.anchor = json.load(open(a.anchor_json))[a.task_index]["anchor"]
    anchor_for_context = a.anchor.rstrip("\n") if a.strip_anchor_newline else a.anchor
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model, meta = load_v3_checkpoint(a.checkpoint, tok, a.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval()

    recs = json.load(open(a.train_json))
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(recs))
    prompts = []
    for i in idx:
        p = recs[int(i)]["prompt"]
        if p.startswith(a.anchor):
            prompts.append(p)
        if len(prompts) >= a.num:
            break
    header_anchor = tok(CHAT_HEADER + anchor_for_context, add_special_tokens=False)["input_ids"]
    La = len(header_anchor)
    N, T, K = len(prompts), a.max_pos, a.topk
    entropy = np.full((N, T), np.nan, np.float32); gold_nll = np.full((N, T), np.nan, np.float32)
    gold_rank = np.full((N, T), -1, np.int32); top_ids = np.full((N, T, K), -1, np.int32)
    top_p = np.full((N, T, K), np.nan, np.float32); gold = np.full((N, T), -1, np.int32)
    valid = np.zeros((N, T), bool); pos1_probs = None; skipped = 0

    for b0 in range(0, N, a.batch):
        chunk = prompts[b0:b0 + a.batch]
        # Same conditioning as the generation probe: the fixed header+anchor id
        # sequence, followed by the content tokenized on its own.
        enc = [header_anchor + tok(p[len(a.anchor):], add_special_tokens=False)["input_ids"] for p in chunk]
        keep = [j for j, ids in enumerate(enc) if len(ids) > La]
        skipped += len(enc) - len(keep)
        if not keep:
            continue
        seqs = [enc[j][:La + T] for j in keep]          # need logits at La-1 .. La+T-2
        L = max(len(s) for s in seqs)
        ids_t = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids_t[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
        logits = model(input_ids=ids_t.cuda(), attention_mask=att.cuda(), use_cache=False).logits.float()
        logp = torch.log_softmax(logits, -1)            # [b, L, V]
        for r, s in enumerate(seqs):
            n_content = len(s) - La
            if n_content <= 0:
                continue
            lp = logp[r, La - 1: La - 1 + n_content]      # predicts content tokens 1..n_content
            g = torch.tensor(s[La:], device=lp.device)
            H = -(lp.exp() * lp).sum(-1)
            gnll = -lp.gather(1, g[:, None])[:, 0]
            rank = (lp > lp.gather(1, g[:, None])).sum(-1)
            tp, ti = lp.exp().topk(K, dim=-1)
            i = b0 + keep[r]
            entropy[i, :n_content] = H.cpu().numpy(); gold_nll[i, :n_content] = gnll.cpu().numpy()
            gold_rank[i, :n_content] = rank.cpu().numpy(); top_ids[i, :n_content] = ti.cpu().numpy()
            top_p[i, :n_content] = tp.cpu().numpy(); gold[i, :n_content] = g.cpu().numpy()
            valid[i, :n_content] = True
            if pos1_probs is None:
                pos1_probs = lp[0].exp().cpu().numpy().astype(np.float32)
        print(f"[tf] {min(b0 + a.batch, N)}/{N}", flush=True)

    np.savez_compressed(out / "tf.npz", entropy=entropy, gold_nll=gold_nll, gold_rank=gold_rank,
                        top_ids=top_ids, top_p=top_p, gold=gold, valid=valid, pos1_probs=pos1_probs)
    (out / "meta.json").write_text(json.dumps({"checkpoint": a.checkpoint, "train_json": a.train_json,
        "anchor": anchor_for_context, "strip_anchor_newline": a.strip_anchor_newline, "num": N, "skipped_tokenization_mismatch": skipped, "max_pos": T, "topk": K,
        "anchor_tokens": La}, indent=1))
    v = valid.sum(0)
    print(f"[tf] DONE {out} N={N} skipped={skipped} "
          f"H@1={np.nanmean(entropy[valid[:,0],0]):.3f} H@5={np.nanmean(entropy[valid[:,4],4]):.3f} "
          f"goldNLL@1={np.nanmean(gold_nll[valid[:,0],0]):.3f} top1acc@1={(gold_rank[valid[:,0],0]==0).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
