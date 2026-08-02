# Reproduction implementation status

Updated: 2026-07-26

## Ready

- Clean upstream snapshots are pinned for SLoRA, TRACE and O-LoRA.
- Separate editable worktrees exist at:
  - `implementations/SLoRA-repro`
  - `implementations/TRACE-repro`
  - `implementations/O-LoRA-repro`
- TRACE 500/1,000/5,000 assets were audited; the paper profile uses 5,000
  training records for every task.
- Llama-3.1 and Qwen2.5 model paths are centralized.
- Seq-LoRA, SLoRA-Pre and SLoRA-Post have train/eval launchers.
- EWC and LwF have modern-backbone TRACE launchers.
- GEM and O-LoRA have separately named upstream and corrected variants.
- The existing FFN LoRAMoE code is wired as a clearly labeled local compatible
  port for both paper backbones and TRACE-5000; it is not SLoRA-author code.
- SLoRA/TRACE methods render through `scripts/run_experiment.sh`.
- Backbone-specific complete catalogs render through
  `scripts/baselines/llama31.sh` and `scripts/baselines/qwen25_7b.sh`.
- The legacy nine-method matrix remains available through `scripts/run_suite.sh`.
- Evaluation artifacts are normalized to an 8×8 score matrix by
  `scripts/collect_results.py`.
- Final average and AFR are computed by `scripts/compare_results.py`.
- Twenty local unit/math/plumbing/model-integrity tests pass.
- MeetingBank longest-task training passes on both paper backbones at the selected four-GPU MB 8 / accumulation 2 profile.
- Disk-backed dataset fingerprints prevent repeated expensive SFT preprocessing on every distributed rank.

## Audited corrections

### SLoRA

- Accept official TRACE `prompt`/`answer` records.
- Remove the Qwen2.5-incompatible `/no_think` prefix.
- Avoid double PEFT wrapping by letting `SFTTrainer` add the new adapter.
- Compare candidate and reference subspaces at the selected candidate rank.
- Reconstruct with the selected `U_c`.
- Preserve the original LoRA `alpha/r` scaling after rank reduction.
- Snapshot the seven target projections from fixed pretrained `theta_0` before any adapter merge, and use that immutable anchor for Pre/Post denoising.
- Fix evaluation wrapper variables, missing DeepSpeed config and optional
  prefix-tokenizer handling.

### TRACE

- Use `AutoTokenizer` for both Llama-3.1 and Qwen2.5.
- Do not insert a `None` BOS token for Qwen2.5.
- Remove the legacy Llama FlashAttention monkey patch.
- Load local models in bfloat16 without changing vocabulary size.
- Honor cosine scheduler and 3% warmup settings.
- Use the actual tokenizer pad ID during generation.
- Repair 20Minuten SARI JSON serialization.

### GEM

- `gem_upstream` preserves the public qpth sign/constraint translation.
- `gem_corrected` uses the audited dual translation.
- Both results must be reported because the correction changes optimization.

### O-LoRA

- `olora_upstream` uses the public A/L1 overlap penalty.
- `olora_corrected` uses update-column B matrices and squared Frobenius
  overlap.
- Adapter merging is generalized from hard-coded q/v Llama layers to the
  seven configured projection modules.

## Verified execution readiness

- Llama-3.1-8B-Instruct: 4/4 shards, 291 indexed tensors and 16,060,522,496 tensor bytes verified.
- Qwen2.5-7B-Instruct: 4/4 shards, 339 indexed tensors and 15,231,233,024 tensor bytes verified.
- TRACE 500/1,000/5,000 variants pass schema/count checks; the launchers require 5,000 samples per task.
- The isolated `.venv-runtime` passes `pip check` with torch 2.4.1+cu124, transformers 4.51.3, PEFT 0.12.0, TRL 0.16.1, datasets 3.2.0 and DeepSpeed 0.16.9.
- Local tokenizer/config loading passes for both backbones on four visible A100 80GB GPUs.
- SLoRA and TRACE train/eval entry-point import and argument-parser smoke tests pass.
- Both backbones pass real bfloat16 CUDA loading and finite-logit forward tests on GPU 0.
- Real MeetingBank three-step training peaks at 34,817 MiB for Llama-3.1 and 36,379 MiB for Qwen2.5 with MB 8 / accumulation 2; effective batch remains 64.
- Offline 20Minuten SARI metric execution passes.
- The legacy suites render all nine SLoRA/TRACE methods; the new backbone-specific
  catalog adds released-code SLoRA-Pre and the LoRAMoE local port, for eleven
  runnable entries per backbone. Shell syntax, representative dry-runs, LoRAMoE
  math/sparse-routing tests and Llama/Qwen meta-device structure checks pass.

## Remaining experimental work

1. Run the long eight-task training/evaluation jobs and collect the 8x8 matrices.
2. Compare repeated-seed results with the paper; no score is called reproduced yet.
3. SD-LoRA, RCL and the SLoRA authors' unified O-LoRA runner remain unavailable in the public SLoRA repository.
4. TRACE-port O-LoRA upstream/corrected results must remain labeled controlled comparisons, not exact paper-author O-LoRA reproduction.

Ready commands:

```bash
cd /home/work/Agent_HJ/30_flame_agent/slora_repro
./scripts/run_suite.sh validate llama31
./scripts/run_experiment.sh all seq_lora llama31
```
