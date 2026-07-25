# G2 FFN-only — Pre-Code Expert-Init Distillation (9 runs)

Test whether initializing the 8 newly added experts (wiki 8→16 expansion) by
**distilling the expanded student from the pre-expansion 8-expert wiki teacher**
— before Code training — prevents the sharp wiki-probe collapse seen with
random/copy-init expansion.

## Pipeline (per mode, A → B → C)

| Stage | What | Data | Trainable | mb | iters |
|-------|------|------|-----------|----|-------|
| **A** | expand 8→16 + KL distill | wiki | new experts(8–15) + new router rows | 48 | 1800 |
| **B** | code training from A's ckpt (no re-expansion) | code | new experts + new router rows | 96 | 1800 |
| **C** | router-only retune | wiki+code | all router rows (experts frozen) | 96 | 1800 (→step 3600) |

Modes (distill objective in Stage A):
- `logits` — final-logit KL only
- `logits_hidden` — + transformer layer-output MSE
- `logits_hidden_router` — + router softmax KL (teacher 8-way zero-padded to 16)

Stage A/B: LM loss coeff 0, aux 0, z 0 (Stage A) — pure teacher-student align.
All runs: 4 GPUs (DP=4), gbs 2304, wandb offline.

Baseline for comparison (reused, not run here):
- Stage B baseline: `code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh`
- Stage C baseline: `run_g2matched_phase3_router_only_retune_mha.sh ffn-only`

## Run

```bash
bash scripts/experiment/a100/g2_ffn_only_distill_init/launch.sh
```

`launch.sh` starts `run_all9.sh` detached (nohup) and prints the PID + log paths.
Runs are **sequential** (stage-major: all A → all B → all C). Completed stages are
auto-skipped via their checkpoint tracker, so re-launching **resumes** after a
failure. The driver stops on the first failure (`set -e`).

### Monitor / stop
```bash
tail -f .local/logs/g2_ffn_only_distill_init/run_all9_*.log   # master log
watch -n5 nvidia-smi                                          # gpu
kill $(cat .local/logs/g2_ffn_only_distill_init/run_all9.pid) && pkill -f torch.distributed.run
```

### Knobs (env overrides for run_all9.sh)
`MODES`, `MB_A` (48), `MB_BC` (96), `ITERS` (1800), `RETUNE` (1800),
`CUDA_VISIBLE_DEVICES` (0,1,2,3), `NPROC_PER_NODE` (4).

## Outputs
```
.local/weights/a100/mha/g2-checkpoints/code/
  expansion_distill_init/   <- Stage A (16-expert distill-init)
  from_distill_init/        <- Stage B (code-trained)
  from_distill_init_phase3/ <- Stage C (router-retuned, final)
```

## Notes
- Environment: system `/usr/bin/python` + `--user` packages (no venv). If a fresh
  session breaks the tokenizer with *"transformers must be installed"*, it is the
  `huggingface_hub` version — pin it: `pip install --user 'huggingface_hub<1.0'`
  (transformers 4.33.1 needs `hub>=0.15,<1.0`).
- Verified by smoke (2026-07-09): freeze mask (new experts+router only, 0
  unexpected trainable, gradient-checked), teacher load, and all three modes'
  loss terms (kd / hidden mse / router kl) finite with no NaN.
