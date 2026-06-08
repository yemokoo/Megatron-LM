# G2 Phase4 Conversation After Router Finetune Checkpoints

Date: 2026-06-08

## Goal

After the wiki -> code -> router-finetune pipeline is complete, train a third task on the OpenSubtitles-v2018 English Conversation Corpus and track wiki/code retention.

This phase starts from the router-finetuned checkpoints, not from code-only checkpoints.

## Conversation Dataset

- Dataset: OpenSubtitles-v2018 English Conversation Corpus
- Preprocessing: extract English subtitle text, remove duplicated English sentences across subtitle pairs such as En-Fr and En-Ko.
- KT path: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/data/conversation/train`
- Size: `1,403,196,153` tokens
- Training budget: `1800` steps
- Global batch size: `2304`
- Sequence length: `512`
- Tokens consumed by 1800 steps: `1800 * 2304 * 512 = 2,123,366,400`
- Effective epochs: `2,123,366,400 / 1,403,196,153 ~= 1.51`

## Common Phase4 Settings

- Execution script: `scripts/experiment/a100/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh`
- Run mode: `WANDB_MODE=offline`
- Run order: FFN-only -> Experiment 1 freeze-wiki -> Experiment 2 unfreeze-wiki
- Logical W&B step range: `5400 -> 7200`
- Local training steps per run: `1800`
- Physical checkpoint tracker in each new phase4 directory: `latest_checkpointed_iteration.txt = 1800`
- Save interval: every `600` local steps, so checkpoints at local `600`, `1200`, `1800`
- Probe interval: every `50` local steps
- Primary probe: `code_probe`
- Secondary probe: `wiki_probe`
- Tertiary probe: `conversation_probe`
- Conversation task learning signal: train LM loss on conversation data plus `conversation_probe`
- New task expansion: `16 -> 24` experts
- Router top-k: `4`
- MoE FFN hidden size: `352`

## Run 1: FFN-Only Baseline

User-facing label: `FFN only expert baseline`

Source checkpoint:

```text
/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/code/phase3/g2matched-attn-freeze-phase3-router-only-retune-wikicode-no-reinit-mb96-1800
```

Expected source state:

- Physical latest checkpoint: `3600`
- Previous router-finetune final code probe: about `0.662843`
- Previous router-finetune final wiki probe: about `0.457432`

Phase4 output checkpoint:

```text
/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/conversation/phase4/g2-ffn-only-phase4-conversation-from-router-retuned-e16to24-mb96-1800
```

Training mode:

- Existing FFN experts and existing router rows are frozen.
- Newly added conversation FFN experts and new router rows are trained.
- Attention remains frozen.
- No old-model KD.

## Run 2: Experiment 1 Freeze-Wiki

User-facing label: `실험1: code 학습 시 wiki expert 전체 freeze`

Important naming note: in code paths this checkpoint uses `g2-exp2...`, but in the experiment narrative this is Experiment 1.

Source checkpoint:

```text
/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/code/phase3/g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-no-reinit-mb72-1800
```

Expected source state:

- Physical latest checkpoint: `3600`
- Previous router-finetune final code probe: about `0.683746`
- Previous router-finetune final wiki probe: about `0.395259`

Phase4 output checkpoint:

```text
/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/conversation/phase4/g2-exp1-freeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800
```

Training mode:

- Shared-router hybrid FFN expert + attention expert model.
- Existing experts are frozen.
- Newly added conversation experts are trained.
- All router rows are trainable.
- This preserves the freeze-wiki style used during the code-learning stage while allowing router adaptation.

## Run 3: Experiment 2 Unfreeze-Wiki

User-facing label: `실험2: code 학습 시 wiki expert/router unfreeze`

Important naming note: in code paths this checkpoint uses `g2-exp1...`, but in the experiment narrative this is Experiment 2.

Source checkpoint:

```text
/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/code/phase3/g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-no-reinit-mb72-1800
```

Expected source state:

- Physical latest checkpoint: `3600`
- Previous router-finetune final code probe: about `0.696460`
- Previous router-finetune final wiki probe: about `0.339974`

Phase4 output checkpoint:

```text
/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/conversation/phase4/g2-exp2-unfreeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800
```

Training mode:

- Shared-router hybrid FFN expert + attention expert model.
- Existing wiki/code experts and newly added conversation experts are trainable.
- All router rows are trainable.
- This preserves the higher-plasticity unfreeze-wiki style used during the code-learning stage.

## KT Launch Command

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh
source ~/.config/wandb/env 2>/dev/null || true

if ps -ef | grep -E 'pretrain_gpt.py|torchrun|run_guarded_training' | grep -v grep; then
  echo "[ERROR] another training process is running"
  exit 1
fi

OUTLOG="g2_phase4_conversation_after_router_finetune_offline_chain_$(date +%Y%m%d_%H%M%S).log"

WANDB_MODE=offline \
nohup bash scripts/experiment/a100/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh \
  > "$OUTLOG" 2>&1 &

echo "[PID] $!"
echo "[OUTLOG] $OUTLOG"
tail -f "$OUTLOG"
```

## Log/Checkpoint Check

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

PHASE4_ROOT=".local/weights/a100/mha/g2-checkpoints/conversation/phase4"

for d in \
  "$PHASE4_ROOT/g2-ffn-only-phase4-conversation-from-router-retuned-e16to24-mb96-1800" \
  "$PHASE4_ROOT/g2-exp1-freeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800" \
  "$PHASE4_ROOT/g2-exp2-unfreeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800"
do
  echo
  echo "=== $d ==="
  cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo "missing tracker"
  find "$d/logs" -maxdepth 1 -type f -print 2>/dev/null
done
```
