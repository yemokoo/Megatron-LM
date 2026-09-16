#!/usr/bin/env python
"""lm-eval (MMLU/GSM8K/PIQA) on TRACE Table-1 final checkpoints.

The model is built with the exact loaders the TRACE evaluators use
(evaluate_tab1.load -> load_tab1_checkpoint, evaluate_Ours_LoRA_MoE.load ->
load_v3_checkpoint / load_lora_moe_checkpoint), then handed to lm-eval's HFLM.

  python run_lmeval_trace.py --ckpt DIR --out OUT [--tasks mmlu] [--batch_size 8] [--limit N]
  --ckpt base  -> plain Llama-3.1-8B-Instruct (zero-shot row)
"""
import argparse
import json
import os
import sys

LLMCL = "/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/implementations/llmcl_benchmark"
BASE = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
SLORA_PORT = "/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/implementations/SLoRA-upstream-port"
sys.path.insert(0, LLMCL)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def build(ckpt, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = torch.device("cuda")
    if ckpt == "base":
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype).to(dev)
        return model.eval(), tok, {"method": "base"}
    lm_meta = os.path.join(ckpt, "lora_moe_meta.json")
    if os.path.exists(lm_meta) and "residual_expert" in json.load(open(lm_meta)):
        # V3 trained with a residual expert from task 0
        sys.path.insert(0, "/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/scripts/residual")
        import residual_expert as RE
        model, meta = RE.load_v3_residual_checkpoint(ckpt, tok, BASE, dtype=dtype)
        from model import Ours_LoRA_MoE_V3 as V3
        for layer in V3.shared_router_layers(model):
            layer.shared_expert_router._residual_stats = [0, 0]
        return model.eval(), tok, meta
    if os.path.exists(os.path.join(ckpt, "residual_expert_meta.json")):
        # router-tuned V3 (optionally + residual no-op expert rows)
        sys.path.insert(0, "/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/scripts/residual")
        import residual_expert as RE
        model, meta = RE.load_router_tuned(ckpt, tok, BASE, device=dev, dtype=dtype)
        if meta.get("n_residual", 0):
            from model import Ours_LoRA_MoE_V3 as V3
            for layer in V3.shared_router_layers(model):
                layer.shared_expert_router._record_residual = True
        return model.eval(), tok, meta
    if os.path.isdir(os.path.join(ckpt, "order1")) and os.path.exists(
            os.path.join(ckpt, "order1", "max.safetensors")):
        # SLoRA-Pre, loaded exactly as Table 1 was scored (2026-09-13): every
        # denoised adapter merged at scaling 1 (upstream lora_alpha = rank).
        # The port now keeps alpha/r; handing it configs with alpha == r makes
        # its original_scaling 1, i.e. the pre-fix merge, without editing it.
        # SLORA_MERGE=table1 (default): alpha=rank override -> scaling 1 (the
        #   pre-fix merge the original slora_pre_released_gb64_20260912 was BOTH
        #   trained and scored with; see cl_train_slora.py:284-290).
        # SLORA_MERGE=fixed: no override -> port's alpha/r-preserving merge; use
        #   ONLY for checkpoints trained after the 2026-09-14 fix
        #   (tab1_slora_r64_scalefix), never for the 09-12 Table-1 checkpoint.
        sys.path.insert(0, SLORA_PORT)
        from peft import LoraConfig
        from src.model import builder
        merge_mode = os.environ.get("SLORA_MERGE", "table1")
        if merge_mode == "table1":
            orig = LoraConfig.from_pretrained.__func__

            def alpha_eq_r(cls, *args, **kwargs):
                cfg = orig(cls, *args, **kwargs)
                cfg.lora_alpha = cfg.r
                return cfg
            builder.LoraConfig.from_pretrained = classmethod(alpha_eq_r)
        elif merge_mode != "fixed":
            raise SystemExit(f"SLORA_MERGE must be table1|fixed, got {merge_mode}")
        print(f"SLoRA merge mode: {merge_mode}", flush=True)
        _, model, _, _ = builder.load_continual_pretrained_model(
            ckpt, BASE, "llama31", "slora_pre", device_map={"": 0}, test_order=8)
        model.to(torch.float16)
        return model.eval(), tok, {"method": "slora_pre", "merge": merge_mode}
    if os.path.exists(os.path.join(ckpt, "tab1_meta.json")):
        from model.tab1_checkpoint import load_tab1_checkpoint
        model, meta = load_tab1_checkpoint(ckpt, tok, BASE, device=dev, dtype=dtype)
    elif os.path.exists(os.path.join(ckpt, "lora_moe_meta.json")):
        import evaluate_Ours_LoRA_MoE as E
        with open(os.path.join(ckpt, "lora_moe_meta.json")) as f:
            m = json.load(f)
        loader = (E.load_v3_checkpoint if m.get("architecture") == E.V3_ARCHITECTURE
                  else E.load_lora_moe_checkpoint)
        model, meta = loader(ckpt, tok, base_model_name_or_path=BASE,
                             device=dev, dtype=dtype, device_map=None)
    else:
        raise SystemExit(f"unknown checkpoint layout: {ckpt}")
    if next(model.parameters()).device.type != "cuda":
        model.to(device=dev, dtype=dtype)
    print(f"model device={next(model.parameters()).device}", flush=True)
    model.eval()
    return model, tok, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="mmlu")
    p.add_argument("--batch_size", default="8")
    p.add_argument("--limit", type=float, default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model, tok, meta = build(a.ckpt)
    short = {k: meta.get(k) for k in ("method", "architecture", "num_experts", "r", "alpha", "top_k")
             if meta.get(k) is not None}
    print(f"Loaded {a.ckpt}: {short}", flush=True)

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size,
              max_length=4096)
    res = lm_eval.simple_evaluate(model=lm, tasks=a.tasks.split(","),
                                  limit=a.limit, log_samples=False)
    print(make_table(res))
    if "groups" in res:
        print(make_table(res, "groups"))
    try:
        from model import Ours_LoRA_MoE_V3 as V3
        rs = [l.shared_expert_router for l in V3.shared_router_layers(model)]
        if rs and getattr(rs[0], "_residual_stats", None) is not None:
            res["residual_rate_per_layer"] = [r._residual_stats[0] / max(1, r._residual_stats[1]) for r in rs]
            print("residual rate per layer:", [round(x, 3) for x in res["residual_rate_per_layer"]])
        elif rs and getattr(rs[0], "_record_residual", False):
            res["residual_rate_per_layer"] = [
                (r._residual_hits / r._residual_total) if getattr(r, "_residual_total", 0) else None
                for r in rs]
            print("residual rate per layer:", [round(x, 3) if x is not None else None
                                               for x in res["residual_rate_per_layer"]])
    except Exception as exc:  # stats only
        print("residual stats skipped:", exc)
    res["trace_checkpoint"] = a.ckpt
    res["trace_meta"] = short
    with open(os.path.join(a.out, f"results_{a.tasks.replace(',', '_')}.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)


if __name__ == "__main__":
    main()
