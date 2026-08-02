#!/usr/bin/env python
"""Measure real per-step wall time at the surveyed per-task batch sizes, so we can
estimate total Track-1 training time. Reuses batch_survey's model builder (same
frozen backbone + 1 expert + router + AdamW + optional checkpointing as training).
Times a TYPICAL random batch (not the worst-case longest one) over several steps."""
import argparse, os, sys, time, statistics, json
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import load_hf_tokenizer
from model.Ours_LoRA_MoE import collect_moe_losses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_survey import build_model  # same construction as training

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--data_output_path", default="/tmp/data_files_tput/")
    # each item: task:batch:ckpt(on/off)
    p.add_argument("--probes", required=True,
                   help="comma list of task:batch:ckpt e.g. C-STANCE:26:off,MeetingBank:12:on")
    p.add_argument("--max_prompt_len", type=int, default=1536)
    p.add_argument("--max_ans_len", type=int, default=512)
    p.add_argument("--lora_moe_rank", type=int, default=8)
    p.add_argument("--lora_moe_alpha", type=int, default=32)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None)
    return p.parse_args()

def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    tok = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    collator = DataCollator(tok, padding="longest", max_prompt_len=args.max_prompt_len,
                            max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=False)
    results = {}
    for spec in args.probes.split(","):
        task, batch, ckpt = spec.split(":")
        batch = int(batch); grad_ckpt = (ckpt == "on")
        ds, _, _ = create_prompt_dataset(0, os.path.join(args.data_path, task),
                                         args.data_output_path, args.seed, distributed=False)
        model, _, opt = build_model(args, device, grad_ckpt=grad_ckpt)
        model.train()
        n = len(ds)
        # typical batch: consecutive real samples (varied lengths)
        def make(i):
            idx = [(i * batch + j) % n for j in range(batch)]
            return [ds[k] for k in idx]
        times = []
        for it in range(args.warmup + args.iters):
            mi = collator(make(it)); mi.pop("sources", None)
            mi = {k: v.to(device) for k, v in mi.items() if isinstance(v, torch.Tensor)}
            torch.cuda.synchronize(); t0 = time.time()
            opt.zero_grad(set_to_none=True)
            out = model(**mi, use_cache=False)
            ml = collect_moe_losses(model)
            loss = out.loss if ml is None else out.loss + ml
            loss.backward(); opt.step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            if it >= args.warmup:
                times.append(dt)
        med = statistics.median(times)
        results[task] = {"batch": batch, "ckpt": ckpt, "s_per_step": med,
                         "samples_per_s_per_gpu": batch / med}
        print(f"{task:12s} b={batch:<3d} ckpt={ckpt:3s}  {med:.3f} s/step  "
              f"{batch/med:6.1f} samp/s/gpu", flush=True)
        del model, opt; torch.cuda.empty_cache()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
