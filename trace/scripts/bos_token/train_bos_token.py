#!/usr/bin/env python3
"""Train a tiny task-conditioning parameter on a frozen V3 checkpoint.

--method bos_token (v1): ONE task-specific document-start token.
--method last_bias (v2): a zero-initialised bias vector added at the LAST decoder
  layer (--site layer_input: the hidden state entering the layer, i.e. what its
  router reads; --site router_logits: the router logits themselves).  Documents
  keep their original BOS run; only the last layer + lm_head are backpropagated.

The token reuses an existing reserved Llama-3.1 special token (no vocab growth,
no tokenizer change).  Its input embedding is a separate fp32 parameter that is
substituted into inputs_embeds wherever the token id appears; embed_tokens,
lm_head, experts and routers are never touched.  Training documents are the
exact SLoRA chat documents used for TRACE training, with the leading BOS run
replaced by the new token, and a full-document LM loss.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "implementations" / "llmcl_benchmark"))
from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint          # noqa: E402
from utils.data.data_collator import SLoRATraceDataCollator   # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--train-json", required=True)
    p.add_argument("--eval-json", required=True)
    p.add_argument("--eval-num", type=int, default=200)
    p.add_argument("--method", choices=["bos_token", "last_bias"], default="bos_token")
    p.add_argument("--site", choices=["layer_input", "router_logits"], default="layer_input")
    p.add_argument("--token-name", default="<|reserved_special_token_0|>")
    p.add_argument("--alias", default="<BoS_cstance>")
    p.add_argument("--init-from", default="<|begin_of_text|>")
    p.add_argument("--plain-base", action="store_true", help="train on the bare base model (no V3 experts): the only round-0 model where BOS does not already yield the task")
    p.add_argument("--init-file", default="", help="bos_token.pt whose trained embedding is the init instead of --init-from")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def build_docs(tok, records, max_length, bos_id, new_id):
    coll = SLoRATraceDataCollator(tokenizer=tok, max_length=max_length)
    docs = []
    for rec in records:
        ids, _ = coll._encode(rec)
        if new_id < 0:
            docs.append(ids); continue
        k = 0
        while k < len(ids) and ids[k] == bos_id:
            k += 1
        docs.append([new_id] + ids[k:])
    return docs


def install_last_bias(model, site, bias):
    """Add ``bias`` at the last decoder layer; returns the hook handle."""
    layer = model.model.layers[-1]
    if site == "layer_input":
        def pre(module, args, kwargs):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            h = h + bias.to(h.dtype)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = h
                return args, kwargs
            return (h,) + tuple(args[1:]), kwargs
        return layer.register_forward_pre_hook(pre, with_kwargs=True)
    router = layer.shared_expert_router.router
    if router.out_features != bias.numel():
        raise ValueError(f"router has {router.out_features} experts, bias has {bias.numel()}")
    return router.register_forward_hook(lambda m, i, o: o + bias.to(o.dtype))


def pad_batch(docs, pad_id):
    L = max(len(d) for d in docs)
    ids = torch.full((len(docs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(docs), L), dtype=torch.long)
    lab = torch.full((len(docs), L), -100, dtype=torch.long)
    for r, d in enumerate(docs):
        ids[r, :len(d)] = torch.tensor(d); att[r, :len(d)] = 1; lab[r, :len(d)] = torch.tensor(d)
    return ids, att, lab


def forward_loss(model, embed, vec, new_id, ids, att, lab):
    ids, att, lab = ids.cuda(), att.cuda(), lab.cuda()
    with torch.no_grad():
        emb = embed(ids)
    if vec is not None:
        emb = torch.where((ids == new_id)[..., None], vec.to(emb.dtype)[None, None, :], emb)
    out = model(inputs_embeds=emb, attention_mask=att, labels=lab, use_cache=False)
    return out.loss, int((lab[:, 1:] != -100).sum())


@torch.no_grad()
def evaluate(model, embed, vec, new_id, docs, pad_id, bsz=8):
    model.eval()
    tot, n = 0.0, 0
    for b in range(0, len(docs), bsz):
        ids, att, lab = pad_batch(docs[b:b + bsz], pad_id)
        loss, ntok = forward_loss(model, embed, vec, new_id, ids, att, lab)
        tot += float(loss) * ntok; n += ntok
    model.train()
    return tot / n


def main():
    a = parse_args()
    import os
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if a.grad_accum % world: raise SystemExit("grad-accum must be divisible by WORLD_SIZE")
        a.grad_accum //= world           # global batch = micro_batch * grad_accum stays fixed
    main_rank = rank == 0
    out = Path(a.out_dir)
    if main_rank: out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    new_id = tok.convert_tokens_to_ids(a.token_name)
    init_id = tok.convert_tokens_to_ids(a.init_from)
    bos_id = tok.bos_token_id
    assert isinstance(new_id, int) and new_id >= 0 and new_id != tok.unk_token_id, a.token_name
    assert tok(a.token_name, add_special_tokens=False)["input_ids"] == [new_id]

    if a.plain_base:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16, local_files_only=True).cuda()
        meta = {"num_experts": 0}
    else:
        model, meta = load_v3_checkpoint(a.checkpoint, tok, a.base_model, device="cuda", dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    if not a.no_grad_ckpt and a.method == "bos_token":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    embed = model.get_input_embeddings()
    if a.method == "bos_token":
        if a.init_file:
            vec = nn.Parameter(torch.load(a.init_file, map_location="cpu", weights_only=False)["embedding"].float().cuda())
        else:
            vec = nn.Parameter(embed.weight[init_id].detach().float().clone())
        param = vec
    else:
        n = (model.config.hidden_size if a.site == "layer_input"
             else model.model.layers[-1].shared_expert_router.router.out_features)
        param = nn.Parameter(torch.zeros(n, device="cuda", dtype=torch.float32))
        install_last_bias(model, a.site, param)
        vec = None
        new_id = -1                       # documents keep their original BOS run
    init_vec = param.detach().cpu().clone()

    train_recs = json.load(open(a.train_json)); eval_recs = json.load(open(a.eval_json))
    eval_recs = [eval_recs[int(i)] for i in rng.permutation(len(eval_recs))[:a.eval_num]]
    train_docs = build_docs(tok, train_recs, a.max_length, bos_id, new_id)
    eval_docs = build_docs(tok, eval_recs, a.max_length, bos_id, new_id)
    if main_rank: print(f"[bos] new_id={new_id} ({a.token_name} as {a.alias}) init_from={init_id} bos_id={bos_id}", flush=True)
    if main_rank: print(f"[bos] train docs={len(train_docs)} mean_len={np.mean([len(d) for d in train_docs]):.1f} "
          f"eval docs={len(eval_docs)}; doc[0][:8]={train_docs[0][:8]}", flush=True)
    if main_rank: print(f"[bos] doc[0] text: {tok.decode(train_docs[0][:60])!r}", flush=True)

    steps_per_epoch = math.ceil(len(train_docs) / (a.micro_batch * a.grad_accum * world))
    total_steps = math.ceil(steps_per_epoch * a.epochs)
    warm = max(1, int(round(total_steps * a.warmup_frac)))
    opt = torch.optim.Adam([param], lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    log = (out / "loss.jsonl").open("w") if main_rank else None
    if main_rank: (out / "meta.json").write_text(json.dumps({**vars(a), "new_id": new_id, "init_id": init_id, "bos_id": bos_id,
        "total_steps": total_steps, "warmup_steps": warm, "num_experts": meta["num_experts"],
        "train_docs": len(train_docs), "eval_docs": len(eval_docs)}, indent=1))

    def save(tag):
        if not main_rank: return
        if a.method == "bos_token":
            torch.save({"token_id": new_id, "token_name": a.token_name, "alias": a.alias, "init_from": a.init_from,
                        "embedding": param.detach().cpu().float(), "init_embedding": init_vec,
                        "checkpoint": a.checkpoint, "step": step}, out / f"bos_token{tag}.pt")
        else:
            torch.save({"site": a.site, "bias": param.detach().cpu().float(), "layer_index": len(model.model.layers) - 1,
                        "checkpoint": a.checkpoint, "step": step}, out / f"last_bias{tag}.pt")

    step = 0
    ev = evaluate(model, embed, vec, new_id, eval_docs, tok.pad_token_id) if main_rank else 0.0
    if main_rank:
        print(f"[bos] step 0 eval_nll={ev:.4f} (init)", flush=True)
        log.write(json.dumps({"step": 0, "eval_nll": ev}) + "\n"); log.flush()
    best = (ev, 0); t0 = time.time()
    order = []
    while len(order) < total_steps * a.micro_batch * a.grad_accum * world:
        order.extend(rng.permutation(len(train_docs)).tolist())
    ptr = rank * a.micro_batch           # rank-strided slices of the shared permutation
    for step in range(1, total_steps + 1):
        opt.zero_grad(set_to_none=True)
        acc_loss, acc_tok = 0.0, 0
        for _ in range(a.grad_accum):
            idx = order[ptr:ptr + a.micro_batch]; ptr += a.micro_batch * world
            ids, att, lab = pad_batch([train_docs[i] for i in idx], tok.pad_token_id)
            loss, ntok = forward_loss(model, embed, vec, new_id, ids, att, lab)
            (loss / a.grad_accum).backward()
            acc_loss += float(loss) * ntok; acc_tok += ntok
        if world > 1:
            dist.all_reduce(param.grad); param.grad /= world
            t_ = torch.tensor([acc_loss, float(acc_tok)], device="cuda"); dist.all_reduce(t_); acc_loss, acc_tok = t_[0].item(), int(t_[1].item())
        gnorm = float(param.grad.norm())
        opt.step(); sched.step()
        drift = float((param.detach().cpu() - init_vec).norm())
        rec = {"step": step, "train_loss": acc_loss / acc_tok, "lr": sched.get_last_lr()[0],
               "grad_norm": gnorm, "drift_from_init": drift, "elapsed": time.time() - t0}
        if (step % a.eval_every == 0 or step == total_steps) and main_rank:
            rec["eval_nll"] = evaluate(model, embed, vec, new_id, eval_docs, tok.pad_token_id)
            if rec["eval_nll"] < best[0]:
                best = (rec["eval_nll"], step); save("_best")
        if not main_rank: continue
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("[bos] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
    save("")
    if main_rank: print(f"[bos] DONE best eval_nll={best[0]:.4f}@{best[1]} final saved in {out}", flush=True)


if __name__ == "__main__":
    main()
