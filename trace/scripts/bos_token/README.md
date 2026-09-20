# Task-conditioning tokens for anchor-free self-generated replay

Goal: generate task-j replay from the continual model with **one token**
(`<BoS_j>`) instead of the hand-written instruction anchor.

## Scripts

| file | purpose |
|---|---|
| `train_bos_token.py` | Train one conditioning parameter on a frozen V3 checkpoint. `--method bos_token` (default): one input-embedding row, substituted into `inputs_embeds` (embed_tokens/lm_head/experts/router never touched). `--method last_bias`: zero-init bias at the last decoder layer (`--site layer_input` or `router_logits`; router_logits is a no-op with one expert). Supports `torchrun` (grad all-reduce; global batch = micro_batch x grad_accum kept fixed), `--init-file` (start from a saved token), `--plain-base` (train on bare Llama, no experts). |
| `gen_doc.py` | Generate whole chat documents from a bare start prompt (`<BoS_j>` via `--bos-token-file`, or N x BOS), stop at the 3rd `<|eot_id|>`, split system/user/assistant, write `text.jsonl` (user turn only) so `scripts/analysis/answer_pass_v3_fix.py` can run stage B unchanged. |
| `../analysis/bos_sample_v3.py` | `--mode bos_token --bos-token-file`, `--last-bias-file`, `--bos-repeat` added. |
| `runs/run_fomc_arms.sh` | Full pipeline: 640 BoS docs -> stage B -> 500 clean records; production anchor replay (8 GPUs); FOMC round from `1phase_kd_rep/0` for both arms (8 GPU, serial); eval C-STANCE+FOMC (1 GPU per arm). |
| `runs/chain_ep4.sh` | 8-GPU 4-epoch token training + generation probes. |
| `runs/run_lr_lanes.sh` | lr lanes (1e-3 / 1e-2 / 3e-4). |
| `runs/GOAL_bos_cstance_fomc.md`, `runs/RESULT_fomc_arms.md` | experiment spec and result. |

Token ids: reserved Llama-3.1 special tokens (`<|reserved_special_token_0|>` = 128002 for
`<BoS_cstance>`, `_1` = 128003 `<BoS_fomc>`, `_2`/`_3`/`_4` for variants). No vocab growth, no
tokenizer change; `load_v3_checkpoint(forbid_vocab_growth=True)` stays valid.

Training documents = the exact SLoRA chat documents used for TRACE training
(`SLoRATraceDataCollator`, `apply_chat_template` + `add_special_tokens=True`): they start with
**two** `<|begin_of_text|>` and carry the "Cutting Knowledge Date / Today Date" system lines.
The leading BOS run is replaced by the token; loss = full-document LM. Note `scripts/analysis/bos_sample_v3.py`'s
`CHAT_HEADER` (single BOS, no date lines) differs from the training documents.

Setup on this host: start checkpoint `/data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/0`
(V3 qkvo+ffn r64, 1phase, kd_init on, C-STANCE only). Outputs under
`/data2/seonghyeonnoh/paper/bos_token/{cstance_1phase_kd_rep,fomc_arms,after_fomc}`.
lr 1e-3 (Adam, 5% warmup); 1e-2 moves the vector by ~0.64/step (BOS row norm 0.48) and diverges.

## Findings (2026-09-20/21)

Round-0 model (C-STANCE only), 50 docs each: BOS alone already yields C-STANCE 50/50; `<BoS_cstance>`
only cleans the format (clean docs 0.86 -> 0.94, leading-BOS repeats 0.96 -> 0.58). eval NLL 2.409 -> 2.387.
Exact-anchor generation and `<BoS_cstance>` generation are equivalent in template match / length / diversity.

Production anchor generation (`run_arm_gen.sh` gen_chunk, `run_hp_cell.sh`) captures the anchor with bash
`$(...)`, which strips its trailing newline: every generated C-STANCE prompt lacks the newline after
`文本：` (0/50 template match vs 50/50 with the exact anchor). Existing anchor-replay runs were trained on
that format. Not fixed (baseline arm reproduces it on purpose).

FOMC round trained from `1phase_kd_rep/0` with 500 replay records (1phase, kd_init on, identical HP):

| replay source | C-STANCE@r1 | FOMC@r1 |
|---|---|---|
| `<BoS_cstance>` docs (500 clean of 768) | 56.85 | 73.99 |
| production anchor docs (640, trainer picks 500) | 57.55 | 71.37 |
| real C-STANCE (reference) | 57.85 | 71.17 |

**Document start is winner-take-all.** After the FOMC round, teacher-forced P(first user token | training
header) is `What` 0.999 / `判断` 0.000 (log P of the whole C-STANCE anchor -16), for both generated- and
real-replay models; round 0 was `判断` 1.000. Cause (code): replay LM loss trains the router only
(`_router_only_replay`), all tasks share the identical header prefix, top-1 hard routing can pick one expert at
the header position, and the new expert (trained with the new-task LM loss) wins. Expert 0 still knows
C-STANCE (anchor prompts work) but the entrance at the header is closed.

Generation on the post-FOMC model (`fomc_arms/bos_gen/model/1`), 50 docs, C-STANCE / FOMC / broken:

| prompt | C-STANCE | FOMC | broken |
|---|---|---|---|
| BOS x2 | 0 | 47 | 3 |
| `<BoS_cstance>` from round 0, 1 epoch | 0 | 41 | 9 |
| `<BoS_cstance>` from round 0, **4 epochs** (8 GPU) | 0 | 20 | 30 |
| `<BoS_cstance_base>` trained on bare Llama (`--plain-base`) | 0 | 9 | 41 |
| `<BoS_fomc>` trained on this model | 0 | 47 | 3 |
| `<BoS_cstance>` **re-tuned on this model**, real C-STANCE | 25 | 24 | 1 |
| `<BoS_cstance2>` re-tuned, 500 generated replay x10 ep, BOS init | 26 | 23 | 1 |
| `<BoS_cstance2c>` re-tuned, 500 generated replay x10 ep, init from old token | **31** | 18 | 1 |

Conclusions: a token trained on a model that only knows its task carries no task information (there is no
contrast; LM loss is already at its floor), so more epochs / sequence-level loss do not help and the
bare-base token does not transfer. Re-tuning on the later model works with replay only (CL-legal) but tops
out around 50-60% with one 4096-d vector.

## Next (proposed "A": token inside the training documents)

Replace the leading BOS run by `<BoS_t>` for a fraction (e.g. 50%) of task-t documents during the task's own
round, and by `<BoS_j>` for task-j replay documents; the rest keep BOS so BOS-prompted evaluation is unchanged.
The router then learns `<BoS_j>` -> expert j in the main training, the prefixes are no longer identical
(no winner-take-all), and every later round's replay re-teaches the mapping. Change: one line in the training
collator (prefix substitution), optional trainable embedding row per task, and `<BoS_j>` as the selfgen prompt.
Validate with rounds 0-1 (C-STANCE, FOMC) and `<BoS_cstance>` generation on model/1 (currently 0/50).
