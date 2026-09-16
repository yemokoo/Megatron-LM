"""Backbone (plain Llama-3.1-8B-Instruct) BoS-only generations with vLLM.

Prompt = the single <|begin_of_text|> token; sampling T=1.0, top_p=0.95.
Output: records.jsonl lines {"text": ..., "n_tokens": ...}.
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--num-seqs", type=int, default=4000)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    llm = LLM(model=a.model, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=0.85,
              max_model_len=1024, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    bos = tok.bos_token_id
    n_req = int(a.num_seqs * 1.15)
    # no per-request seed: identical seeds would make every sequence identical;
    # the engine-level seed above keeps the whole batch reproducible.
    params = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=a.max_new_tokens,
                            skip_special_tokens=False)
    prompts = [{"prompt_token_ids": [bos]} for _ in range(n_req)]
    outs = llm.generate(prompts, params)
    kept = 0
    with open(a.out, "w") as f:
        for i, o in enumerate(outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            if len(ids) < a.min_tokens:
                continue
            text = tok.decode(ids, skip_special_tokens=True)
            if not text.strip():
                continue
            f.write(json.dumps({"text": text, "n_tokens": len(ids), "finish": c.finish_reason},
                               ensure_ascii=False) + "\n")
            kept += 1
            if kept >= a.num_seqs:
                break
    print(f"requested {n_req} kept {kept} -> {a.out}")


if __name__ == "__main__":
    main()
