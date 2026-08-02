# SLoRA reproducibility audit

> **Archived model-free phase:** this report preserves the earlier structure-only audit.
> Statements that no model/runtime is available are superseded by
> [`implementation_status.md`](implementation_status.md), where both backbones, the isolated
> runtime, entry-point smoke tests and full validation matrix are verified ready.


Generated for the clean snapshots and local assets verified on 2026-07-26.
This is an environment/audit result, not an experimental score claim.

## Provenance and preservation

| Repository | URL | pinned commit | commit date | role |
|---|---|---|---|---|
| SLoRA | `https://github.com/alina1031/SLoRA` | `0faf15fd6562ed3e146e8733205f59ef9742ba0f` | 2026-04-08 | pristine paper code |
| O-LoRA | `https://github.com/cmnfriend/O-LoRA` | `07117e1fc4a5f5ad9308a815a42cee8f46502dc8` | 2024-06-15 | pristine baseline |
| TRACE | `https://github.com/BeyonderXX/TRACE` | `462e39f616134f4f819efeb3baea8638c03c7db4` | 2024-01-24 | pristine baseline |

The existing `LLM-continual-learning` and `TRACE` worktrees are not reset,
cleaned, merged, or patched. `manifests/repositories.json` records their
remotes, branches, HEADs, status paths, and diff stats. The clean snapshots
remain separate, and proposed corrections are files under `patches/`.
`manifests/upstream_heads.json` records a 2026-07-26 `git ls-remote`
verification that every snapshot still matches its named upstream branch.

## Hardware and environment

The verified host has four NVIDIA A100 80 GB PCIe GPUs, driver 575.51.03,
CUDA runtime 12.9, CUDA toolkit 12.5, Python 3.10.12, GCC 11.4, 80 logical
CPUs, about 1 TiB RAM, and about 10 TB free space. This matches the paper's
reported count and A100 family, but the paper does not disclose driver,
toolkit, CPU, RAM, A100 form factor, or exact software versions.

`requirements-audit.lock` is the tested, minimal offline audit environment.
`requirements-full-candidate.txt` is explicitly a derived candidate because
the SLoRA repository has no lock; it must be integration-tested with an
approved backbone before any full result claim. Existing GPU jobs make a full
run inappropriate during this setup.

## Models

No backbone model was downloaded. Placeholder directories contain only
human-readable README files and cannot pass the readiness check. The central
registry records:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`

Meta license acceptance and Hugging Face access are prerequisites for the
Meta entries. License/access must be checked again for all models in the later
download phase. Llama-2 is a legacy O-LoRA example only.

## TRACE 500 versus 5,000

**Conclusion: use 5,000 training records per task for the paper-faithful TRACE
profile. There is no evidence that `batch 10` means 500 records.**

| Evidence | Finding |
|---|---|
| SLoRA paper Appendix B.1, Table 7 | all eight TRACE datasets contain 5,000 samples |
| local JSON contents | each of the eight selected `LLM-CL-Benchmark_5000` train files has exactly 5,000 list items |
| local alternative assets | separate 500 and 1,000 variants each have exactly those cardinalities |
| SLoRA scripts | use the eight-task order and epochs 5/3/7/5/3/5/5/7; do not truncate to 500 |
| O-LoRA Llama scripts | standard classification, one epoch, micro-batch 1, accumulation 8, eight GPUs, `logging_steps=10` |
| O-LoRA committed metadata | train samples DBpedia 14,000; Amazon 5,000; Yahoo 10,000; AG News 4,000 |
| current O-LoRA loader | default cap 10,000 only when sampling is `random`; committed DBpedia 14,000 metadata therefore reflects different code/config/cache history |

TRACE and standard O-LoRA classification are distinct experiments. The
current O-LoRA repository provides no support for a 500-example claim in its
standard order scripts. `logging_steps=10` controls reporting frequency, not
data cardinality or batch size.

All 72 local TRACE JSON files across the 500/1,000/5,000 variants were parsed.
Their actual counts, first-record schemas, byte sizes, and SHA-256 hashes are
in `manifests/preflight.json`.

## Effective batches and steps

For 5,000 examples, the public SLoRA-Pre launcher uses one process, per-device
micro-batch 2, and accumulation 8: effective batch 16, 2,500 dataloader
batches and 313 optimizer steps per epoch (ceiling semantics). Public
SLoRA-Post uses two processes: effective batch 32, 1,250 batches and 157
steps. A four-GPU harmonized profile would have effective batch 64, 625
batches and 79 steps, but this is **not proven to be the paper's batch
configuration**. The paper reports four A100s without its batch details.

The official O-LoRA Llama standard-CL script uses micro-batch 1 × 8 GPUs ×
accumulation 8 = effective batch 64. Its `logging_steps=10` is unrelated.
The runtime preflight prints and records all these values.

## Paper-to-code matrix

| Item | Paper | clean public code | audit status |
|---|---|---|---|
| TRACE order | C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten | same | matched |
| epochs | 5/3/7/5/3/5/5/7 | same in SLoRA runners | matched |
| LoRA | rank 64, alpha 128 | same | matched |
| target modules | q/k/v/o/up/gate/down | same | matched |
| optimizer | AdamW, lr 2e-4, warmup 3%, cosine | same launcher values | matched; AdamW variant/version not fully specified |
| dropout/init | not completely documented | library-dependent | unresolved; freeze required |
| reference | frozen pretrained weights | builder receives base weights | intended match |
| candidate rank | top-c update and top-c reference, squared Frobenius overlap | reconstructs rank-c then slices top-rank; unsquared norm | rank/null-space defect confirmed; omitted square does not change argmax |
| randomized SVD | randomized approximation | fresh unseeded Gaussian projection in each search | method match; determinism not ensured |
| Pre/Post timing | after each task / after full sequence | separate Pre/Post paths | conceptual match |
| batch/distribution | four A100s; batch undocumented | Pre world 1, Post world 2, hard-coded GPUs | contradiction/undocumented |
| DeepSpeed | not fully specified | references absent `scripts/zero2.json` | missing public artifact; local explicit zero-2 config supplied |
| model paths | paper model IDs | machine-specific `/LLMs` paths | portability patch/config required |
| evaluation wrapper | TRACE metrics | nonexistent wrapper path and undefined `DROP_MODE` under `set -u` | confirmed launch defects |
| generation | task evaluation | code-specific defaults | exact paper decoding parameters unresolved |
| metrics | accuracy, ROUGE-L, SARI, edit similarity | TRACE-derived metric implementations | names match; package/tokenization versions must be frozen |
| adapter lifecycle | accumulated task updates; Pre/Post denoising | adapter files loaded/rewritten per path | needs real-model integration test |

## GEM audit

Classification: **confirmed mathematical translation bug**, plus substantial
implementation deviations.

qpth solves `min 1/2 v'Pv + q'v` subject to `Gv <= h`. For the GEM correction
`g' = g + M'v` with `v >= margin`, the dual needs `q=M g`, `G=-I`,
`h=-margin`. TRACE instead sets `q=-M g`, `G=I`, `h=margin`. In the
deterministic one-dimensional conflict `g=-1, M=1`, its translation returns
`g'=-2`, worsening the constraint, while the reference projection returns
zero for margin zero.

Further deviations:

- TRACE stores the final observed mini-batch gradient at task transition, not
  gradients recomputed from an explicit episodic memory buffer.
- It projects each named parameter independently rather than one global
  gradient vector.
- It stores CPU bfloat16 tensors and hard-codes `.cuda()`, creating precision,
  device, and distributed/ZeRO risks.
- `safe_get_full_grad` helps gather ZeRO gradients but does not establish
  cross-rank consistency of the per-parameter projection.

The sign correction is isolated in `patches/trace-gem-qpth-sign.patch`.
Official and corrected run profiles remain separate. A faithful global,
episodic-memory GEM rewrite is larger than a sign patch and must not be
silently labeled official TRACE.

## O-LoRA audit

Classification: **confirmed paper-to-code mismatch** for the loss; other
reported concerns remain separately labeled.

The paper constrains its update-column matrices and defines a sum of squared
overlap entries. With PEFT's `delta_W = lora_B @ lora_A`, the corresponding
column basis is `lora_B`. Public O-LoRA instead computes an L1 overlap on
PEFT `lora_A`. A deterministic test constructs identical update-column spaces
whose public penalty is zero, while the paper-mapped penalty is positive.
The proposed correction is `patches/olora-paper-orthogonality.patch`.

Previous adapters are frozen and new adapters are trainable. Concatenating
old/new A rows and B columns preserves the sum of low-rank updates
algebraically. That lifecycle is an expected implementation choice, though a
real-model save/reload/merge test is blocked by the deliberately absent
model. Open GitHub issue
[`#41`](https://github.com/cmnfriend/O-LoRA/issues/41) reports zero
orthogonality loss, and issue
[`#42`](https://github.com/cmnfriend/O-LoRA/issues/42) questions decoder
source-plus-target truncation. These are corroborating reports, not
independent proof, and neither issue has a linked fix. The two currently open
pull requests (`#26` NaN loss and `#27` Llama special tokens) do not address
the audited orthogonality formula.

## SLoRA code defects and patches

- `slora-candidate-rank.patch`: implements top-c comparison and avoids
  arbitrary null-space padding; also uses selected `U_c` in reconstruction.
- `slora-portability.patch`: fixes undefined `DROP_MODE`, documents the
  nonexistent eval path correction, and requires configured paths/GPUs.
- `config/deepspeed_zero2.json`: supplies the missing referenced artifact.

Patches are not applied to pristine source. Score-changing patches must use a
distinct corrected provenance label.

## Result comparison and expected values

`config/expected_paper_results.json` transcribes the paper's main TRACE
Seq-LoRA/SLoRA values for all five backbones, standard-CL order averages, and
labels them as references only. `compare_results.py` requires a provenance
record, validates the continual score-matrix shape, computes final per-task
scores and average, and implements Appendix B.7 forgetting.

## Smoke result

The offline smoke profile passed data parsing/cardinality/hash, model absence,
repository cleanliness, deterministic mathematical tests, a NumPy plumbing
double for one optimizer step, adapter save/reload/merge, rank denoising,
greedy generation, metric dispatch, result schema, serialization, final
average, and AFR calculation. The synthetic fixture is
explicitly marked `smoke_only_not_a_paper_result`.

It cannot validate tokenizer/chat templates, transformer forward/backward,
real PEFT adapter lifecycle, actual SLoRA denoising on a transformer, model
generation, or exact task-metric parity without a compatible approved model.
The SARI smoke is explicitly a dispatch double, not the paper implementation.
Those are recorded
as remaining model-phase gates, not falsely claimed as completed experiments.

## Full-run gate

Before any full run:

1. separately obtain licenses/access and place an exact registered model at
   the planned or overridden path;
2. run strict preflight and record model file hashes;
3. install and freeze the full candidate environment, then validate imports;
4. copy a pristine snapshot into a disposable run tree and apply only the
   profile-listed patches;
5. record code/data/model/config hashes, seed, world size, precision, package
   freeze, and launch command in the output;
6. run a real one-step adapter save/reload/merge/generation/metric smoke test;
7. launch the expensive sequence only after those gates pass.

No score should be called reproduced unless its complete provenance and metric
implementation are attached.
