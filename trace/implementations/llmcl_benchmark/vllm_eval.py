#!/usr/bin/env python
"""
TRACE 8-benchmark zero-shot evaluation with a vLLM offline backend.

This bypasses the DeepSpeed / continual-learning checkpoint loop in
inference/infer_single.py and evaluates an *untrained* base model directly
on the 8 TRACE tasks. Generation is done by vLLM in OFFLINE batch mode
(from vllm import LLM) -- no server, no external network calls.

Scoring reuses the repo's own metrics: evaluations/*.py + metrics.py, so the
numbers are directly comparable to the paper's protocol.

Prompt format matches the repo's standard (non-ICL) inference path:
the raw `prompt` string from test.json is fed to the model as-is
(no chat template), see utils/data/data_collator.py:decoder_call.

Network notes (relevant for offline / corporate sessions):
  * Model weights load from a LOCAL path -> no network.
  * All tasks score locally EXCEPT 20Minuten, whose SARI metric calls
    datasets.load_metric("sari") which fetches a script from the HF Hub.
    Use --skip_sari (default) to avoid that call; pass --with_sari only if
    the HF metric is cached locally.
Set these before running to be safe:
  export HF_HUB_OFFLINE=1 ; export TRANSFORMERS_OFFLINE=1
"""
import argparse
import json
import os
import re
import sys

# make `from evaluations import ...` and `from metrics import ...` resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluations import (
    eval_CStance, eval_FOMC, eval_MeetingBank, eval_NumGLUE_cm,
    eval_NumGLUE_ds, eval_Py150, eval_ScienceQA, eval_20Minuten,
)

ALL_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
             "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

# Conservative per-task generation ceilings derived from an exhaustive token
# count of every test answer with both the Qwen3 and OLMoE tokenizers. Short
# exact-match tasks otherwise get dragged to the uniform 512-token ceiling when
# just one sequence in a batch fails to emit EOS. Long-generation tasks retain
# the original protocol's 512-token ceiling.
TASK_MAX_NEW_TOKENS = {
    # GT labels are one Qwen token. Eight also covers terse wrappers such as
    # "The answer is B" and FOMC's semantic labels without allowing a long
    # free-form continuation.
    "C-STANCE": 8,
    "FOMC": 8,
    "MeetingBank": 512,
    "Py150": 160,
    "ScienceQA": 512,
    "NumGLUE-cm": 16,
    "NumGLUE-ds": 16,
    "20Minuten": 256,
}

# Match the task-specific continuation cuts in the repository's original
# inference path (inference/ICL.py). These are scoring boundaries, not merely
# speed optimizations: generated text after the boundary is usually the next
# training-example header or a second code line and must not be compared with
# the one-line target.
TRACE_STOP_MARKERS = {
    "MeetingBank": "Meeting transcripts",
    "Py150": "<EOL>",
    "ScienceQA": "Question:",
    "NumGLUE-cm": "\n",
    "NumGLUE-ds": "\n",
    "20Minuten": "Paragraph",
}
TRACE_SCORING_PROTOCOL = "trace-task-stop-v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True,
                   help="Local path to a downloaded HF model (or hub id if online).")
    p.add_argument("--data_path", required=True,
                   help="Root dir containing one subfolder per task with test.json.")
    p.add_argument("--inference_tasks", default=",".join(ALL_TASKS),
                   help="Comma-separated task names. Default: all 8.")
    p.add_argument("--inference_output_path", required=True,
                   help="Where to write per-task result json + summary.json.")
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--max_ans_len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (recommended for reproducible benchmarking). "
                        "Paper used 0.1 w/ sampling.")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Set to number of GPUs to shard a large model across.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=None,
                   help="vLLM context window; default = model config. "
                        "Should be >= max_prompt_len + max_ans_len.")
    p.add_argument("--apply_chat_template", action="store_true",
                   help="Wrap each prompt with the tokenizer's chat template "
                        "(for instruct/chat models). OFF by default to match the "
                        "repo's raw-prompt protocol.")
    p.add_argument("--with_sari", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compute local offline SARI for 20Minuten (default: on).")
    p.add_argument(
        "--task_generation_limits", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use task-specific generation ceilings (default: on).")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: only evaluate the first N samples per task.")
    return p.parse_args()


def load_task(data_path, task):
    fp = os.path.join(data_path, task, "test.json")
    assert os.path.exists(fp), f"test.json not found for task '{task}': {fp}"
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [d["prompt"] for d in data]
    gts = [d["answer"] for d in data]
    return prompts, gts


# The repo's accuracy metric is exact string equality (caculate_accuracy in
# metrics.py), which the paper's protocol satisfies because models are SFT'd to
# emit a bare label. A raw/zero-shot chat model instead explains its answer
# ("...the stance is **B**...") -- same underlying judgment, but a 0% exact-match
# score. These extractors pull the label back out so scoring reflects whether the
# model actually got the answer right, not whether it obeys the terse output format.
CHOICE_LETTERS = {"C-STANCE": "ABC", "FOMC": "ABC", "ScienceQA": "ABCD"}
FOMC_SEMANTIC_LABELS = {
    "dovish": "A",
    "hawkish": "B",
    "neutral": "C",
}
_CUE_RE_CACHE = {}
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_choice(text, letters):
    if letters not in _CUE_RE_CACHE:
        _CUE_RE_CACHE[letters] = (
            re.compile(r"^\s*([" + letters + r"])(?:\s*$|[\s:.)-])", re.IGNORECASE),
            re.compile(r"(?:answer|choice|option|stance)\W{0,15}\b([" + letters + r"])\b", re.IGNORECASE),
        )
    leading_re, cue_re = _CUE_RE_CACHE[letters]
    m = leading_re.search(text)
    if m:
        return m.group(1).upper()
    m = cue_re.search(text)
    if m:
        return m.group(1).upper()
    return text  # nothing found; leave as-is (counts as wrong, doesn't crash)


def extract_number(text):
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return text
    return matches[-1].replace(",", "")  # last number = the model's final answer


def extract_fomc_choice(text):
    """Accept both the prompt's A/B/C labels and their semantic equivalents."""
    semantic = re.match(r"^\s*(?:answer\s*[:=\-]?\s*)?(dovish|hawkish|neutral)\b",
                        text, flags=re.IGNORECASE)
    if semantic:
        return FOMC_SEMANTIC_LABELS[semantic.group(1).lower()]
    return extract_choice(text, CHOICE_LETTERS["FOMC"])


def truncate_prediction(task, text):
    """Cut leaked continuation at the same boundary as TRACE inference."""
    marker = TRACE_STOP_MARKERS.get(task)
    if marker is None:
        return text
    return text.split(marker, 1)[0]


def normalize_scienceqa_prediction(text):
    """Put the answer in char 0 without duplicating an existing answer prefix."""
    # The benchmark scorer reads answer=text[0], reasoning=text[2:]. If the model
    # already emitted "A\nreason" (the SFT target format), prepending "A\n" again
    # would incorrectly score "A\nreason" as the reasoning. Trust this explicit
    # leading label before searching the reasoning: a rationale may itself mention
    # phrases such as "answer A" even when the actual leading answer is B.
    prefix = re.match(
        r"^\s*([ABCD])(?:\s*[\n:.)-]\s*|\s+)(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if prefix:
        answer = prefix.group(1).upper()
        reasoning = prefix.group(2)
    else:
        answer = extract_choice(text, CHOICE_LETTERS["ScienceQA"])
        if answer not in CHOICE_LETTERS["ScienceQA"]:
            return text
        reasoning = text
    return f"{answer}\n{reasoning}"


def normalize_predictions(task, preds):
    """Apply TRACE task stops, then reshape predictions for the task scorer."""
    preds = [truncate_prediction(task, prediction) for prediction in preds]
    if task == "C-STANCE":
        return [extract_choice(p, CHOICE_LETTERS[task]) for p in preds]
    if task == "FOMC":
        return [extract_fomc_choice(p) for p in preds]
    if task == "ScienceQA":
        # eval_ScienceQA.resolve() expects "LETTER\nreasoning" (datium[0] / datium[2:]).
        return [normalize_scienceqa_prediction(prediction) for prediction in preds]
    if task in ("NumGLUE-cm", "NumGLUE-ds"):
        return [extract_number(p) for p in preds]
    return preds


def score(task, prompts, preds, gts, with_sari):
    if task == "C-STANCE":
        return eval_CStance.eval(preds, gts)
    if task == "FOMC":
        return eval_FOMC.eval(preds, gts)
    if task == "MeetingBank":
        return eval_MeetingBank.eval(preds, gts)
    if task == "Py150":
        return eval_Py150.eval(preds, gts)
    if task == "ScienceQA":
        return eval_ScienceQA.eval(preds, gts)
    if task == "NumGLUE-cm":
        return eval_NumGLUE_cm.eval(preds, gts)
    if task == "NumGLUE-ds":
        return eval_NumGLUE_ds.eval(preds, gts)
    if task == "20Minuten":
        if with_sari:
            return eval_20Minuten.eval(prompts, preds, gts)
        # SARI omitted to avoid HF Hub network call; report BLEU/ROUGE only.
        from metrics import caculate_bleu, caculate_rouge
        return {
            "bleu-1": caculate_bleu(preds, gts, 1),
            "bleu-4": caculate_bleu(preds, gts, 4),
            "rouge-L": caculate_rouge(preds, gts),
            "sari": "skipped (use --with_sari)",
        }
    return {}


def main():
    args = parse_args()
    tasks = [t for t in args.inference_tasks.split(",") if t]
    os.makedirs(args.inference_output_path, exist_ok=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    summary = {}
    for task in tasks:
        prompts, gts = load_task(args.data_path, task)
        if args.limit:
            prompts, gts = prompts[:args.limit], gts[:args.limit]

        if args.apply_chat_template:
            model_inputs = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]
        else:
            model_inputs = prompts

        max_tokens = (
            min(args.max_ans_len, TASK_MAX_NEW_TOKENS.get(task, args.max_ans_len))
            if args.task_generation_limits else args.max_ans_len
        )
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            truncate_prompt_tokens=args.max_prompt_len,  # left-truncate, matches repo
        )
        outputs = llm.generate(model_inputs, sampling)
        raw_preds = [o.outputs[0].text for o in outputs]
        preds = normalize_predictions(task, raw_preds)

        result = score(task, prompts, preds, gts, args.with_sari)
        summary[task] = result
        print(f"[{task}] n={len(preds)} -> {result}", flush=True)

        out = {"eval": result, "prompts": prompts, "results": raw_preds,
               "results_scored": preds, "labels": gts,
               "scoring_protocol": TRACE_SCORING_PROTOCOL}
        with open(os.path.join(args.inference_output_path, f"results-{task}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

    with open(os.path.join(args.inference_output_path, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ SUMMARY ================")
    for task, res in summary.items():
        print(f"{task:14s} {res}")
    print(f"\nSaved to {args.inference_output_path}/summary.json")


if __name__ == "__main__":
    main()
