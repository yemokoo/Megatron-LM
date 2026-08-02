# How SLoRA actually uses TRACE

Audit target: SLoRA `0faf15fd6562ed3e146e8733205f59ef9742ba0f`
against TRACE `462e39f616134f4f819efeb3baea8638c03c7db4`.

## Bottom line

SLoRA does **not** simply run the official TRACE implementation for every
method. The paper says EWC, LwF, GEM and similar regularization baselines use
official TRACE, while O-LoRA, SD-LoRA, Seq-LoRA, RCL and SLoRA are run in the
authors' unified framework. The released SLoRA repository contains only that
unified LoRA/SLoRA training and evaluation path. It does not contain runnable
commands for the official TRACE baselines, Qwen models, or Llama 3.1.

Therefore the control needed before interpreting SLoRA is the authors'
**Seq-LoRA path**, using exactly the same model, converted data, chat format,
target modules, batch profile and evaluation parser as SLoRA. Running
TRACE's old `CL_method=lora` is not an equivalent control: it uses rank 8,
alpha 32, a different collator, and a different prompt/tokenization path.

## Model coverage in the release

The paper's main Table 1 uses Llama 3.1 8B Instruct and Qwen2.5 7B Instruct;
Llama 3 8B appears in the appendix, and Qwen2.5 14B/32B in Table 2. Public
scripts hard-code only `Meta-Llama-3-8B-Instruct`, `MODEL=llama3`, and
machine-specific paths. There are no Qwen2.5 or Llama 3.1 launch scripts.

The Python trainer is architecture-generic through `AutoModelForCausalLM`.
All reported Llama/Qwen architectures expose the seven hard-coded LoRA
targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, and
`down_proj`. Model variation is otherwise implicit:

- training relies on each tokenizer's own chat template through TRL
  `SFTTrainer` and a `messages` column;
- Llama's missing pad token is manually assigned ID 128004 only when the
  free-form `--model` string contains `llama3`;
- Qwen keeps its tokenizer-defined pad token;
- evaluation bypasses `tokenizer.apply_chat_template` and uses custom,
  separately maintained Llama3/Qwen conversation templates.

This means train/eval formatting parity is not guaranteed.

## Data path mismatch

Official TRACE JSON records are `{"prompt": ..., "answer": ...}`. Every local
5,000-example task has that schema. Both SLoRA trainers and the SLoRA
evaluation loader instead index:

```text
record["conversations"][0]["value"]
record["conversations"][1]["value"]
```

The SLoRA repository ships no data and no conversion script. Consequently the
public command cannot consume the official TRACE assets as released. A
compatibility layer must accept both schemas without altering text. Any
materialized conversion must preserve record order and source hashes.

## Training semantics

For each task SLoRA creates:

1. system: `You are a helpful assistant.`
2. user: the TRACE prompt
3. assistant: the TRACE answer

`SFTTrainer` formats this with the model tokenizer's chat template. It trains
all seven attention/MLP projection families with rank 64 and alpha 128.

Seq-LoRA starts from the frozen base for each task, loads every previous task
adapter in order, merges each into the model, creates a fresh adapter, trains
it, and saves it. SLoRA-Pre does the same except each task adapter is denoised
immediately and the denoised adapter is merged before the next task.
SLoRA-Post trains raw Seq-LoRA adapters and denoises them only while loading
for final evaluation.

## Confirmed execution defects affecting a control

1. **TRACE schema incompatibility**: released data cannot be read.
2. **Evaluation crash**: `tokenizer_with_prefix_space` is `None` when
   `use_logit_bias=False`, but evaluation unconditionally sets its
   `padding_side`.
3. **Broken wrapper**: an undefined `DROP_MODE` is used under `set -u`, and
   wrappers call a nonexistent `scripts/rebuttal/.../eval_cl.sh`.
4. **Missing DeepSpeed file**: training references absent
   `scripts/zero2.json`.
5. **Candidate-rank implementation**: the public search compares arbitrary
   null-space padding for candidates below the original rank.
6. **LoRA scaling changes after denoising**: raw adapters use alpha/r =
   128/64 = 2. `load_denoised_lora` overwrites alpha with the retained rank,
   giving alpha/r = 1. Because denoising decomposes unscaled `B @ A`, every
   denoised update is loaded at half the trained LoRA scale. This is a
   score-affecting confound, not stated in the paper.
7. **Double PEFT construction risk**: both trainers call `get_peft_model`
   before passing a second `peft_config` to `SFTTrainer`. Behavior must be
   checked against the exact TRL/PEFT versions.
8. **Qwen2.5 evaluation contamination**: the released Qwen evaluation path
   prepends `/no_think`, a control directive associated with later reasoning
   model templates and not described for Qwen2.5 in the paper.

Official-public and corrected runs must be retained separately. The first
successful plumbing target should be corrected Seq-LoRA on one task, followed
by official-scaling and corrected-scaling SLoRA variants.

## TRACE official baseline mismatch

TRACE's official loader:

- consumes raw `prompt/answer`;
- concatenates prompt and answer directly;
- adds BOS/EOS manually;
- masks prompt tokens and trains only answer tokens;
- uses left padding and left truncation;
- has no model chat template or SLoRA's helpful-system turn.

SLoRA's TRL conversational SFT path can therefore differ in special tokens,
system prompt, truncation and loss masking. The paper only says metrics and
epoch schedule follow TRACE; it does not specify these transformations.

## Evaluation path

SLoRA copied TRACE task metrics but added output parsing, task-specific
instructions, thought-block stripping, and custom chat wrappers. Generation
is deterministic at temperature 0, top-p 1, one beam, up to 1,024 new tokens.
Llama terminates on EOS or `<|eot_id|>`; Qwen only on EOS.

For Py150 it appends an output-only-code instruction; for NumGLUE it appends a
final-answer-only instruction. Qwen additionally receives `/no_think` in the
current release. These prompt additions are absent from the training data and
from official TRACE inference, so they must be frozen in provenance.

## Reproduction order

1. Validate both model tokenizers, special IDs and exact rendered chat text.
2. Normalize TRACE schema without changing prompt/answer strings.
3. Run one-task Seq-LoRA and verify nonzero trainable LoRA parameters,
   optimizer steps, adapter save/reload and deterministic evaluation.
4. Run all eight Seq-LoRA tasks and compare the paper's per-task final row.
5. Only then run SLoRA-Pre/Post, preserving official-public and corrected
   rank/scaling variants.
6. Treat EWC/LwF/GEM as separate TRACE-porting work; their published numbers
   cannot validate SLoRA until the unified Seq-LoRA control matches.
