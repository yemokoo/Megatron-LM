# G2 continual learning 코드 가이드

이 문서는 G2 실험의 expert 구성, KD 기반 expert 초기화, 1-phase joint LM 학습, 최신 shared-router 4-stage 체인을 한곳에 정리한다. 특히 다음 세 설정을 구분하는 것이 목적이다.

- FFN-only: FFN에만 task expert를 추가한다.
- Shared-router hybrid: FFN expert와 Q/K/V/O attention LoRA expert를 함께 추가하고 같은 router를 사용한다.
- Attention-only baseline: FFN은 dense로 두고 Q/K/V/O attention expert만 사용한다.

문서의 canonical 설정과 최신 실행 스냅샷은 2026-08-02 현재 작업 트리를 기준으로 한다. `.local/` 아래 로그와 checkpoint는 로컬 실행 산출물이므로 Git에 포함되지 않는다.

## 1. 한눈에 보는 canonical shared-router 학습

현재 주 실험은 아래 entry 하나로 실행되는 4-stage 연속 체인이다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

WANDB_MODE=offline \
  bash scripts/experiment/a100/run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh logits
```

Entry: [`run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh`](../scripts/experiment/a100/run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh)

| Stage | 목적 | expert 수 | 학습 데이터/gradient | step | micro batch |
|---|---|---:|---|---:|---:|
| 0 | Code expert KD 초기화 | 8 → 16 | Wiki logits KD | 600 | 48 |
| 1 | Code 1-phase | 16 | Code LM은 새 FFN/QKVO + 전체 router, Wiki LM은 전체 router만 | 1800 | 96 |
| 2 | Conversation expert KD 초기화 | 16 → 24 | Wiki+Code `equal_dataset` logits KD | 600 | 36 |
| 3 | Conversation 1-phase | 24 | Conversation LM은 새 FFN/QKVO + 전체 router, Wiki+Code LM은 전체 router만 | 1800 | 96 |

공통값은 다음과 같다.

- Global batch size: 2304
- Expert Top-K: 4
- Task당 추가 expert: 8개
- MoE layer: 총 9개 transformer layer 중 첫 layer를 제외한 8개 layer
- FFN expert hidden size: 352
- Attention expert: Q/K/V/O, LoRA rank 256, alpha 256
- FFN grouped GEMM과 attention LoRA grouped GEMM: 모두 활성화
- Router dtype: FP32
- LR ramp: 모든 canonical stage에서 비활성화
- W&B: 기본 offline
- Probe: 50 step 간격, 25 iteration, probe micro batch 32

Expert index는 task 순서와 직접 대응한다.

| Task | FFN expert row | QKVO attention expert row | shared-router row |
|---|---:|---:|---:|
| Wiki | `0:8` | `0:8` | `0:8` |
| Code | `8:16` | `8:16` | `8:16` |
| Conversation | `16:24` | `16:24` | `16:24` |

## 2. 모델 구성 비교

### 2.1 FFN-only

FFN-only에서는 각 MoE layer의 FFN에만 expert가 있다. Attention은 task별 routed expert가 아니며, standard MoE router는 FFN expert만 선택한다.

- 모델 설정: [`flame-moe-bf16-no-shared.sh`](../scripts/experiment/a100/flame-moe-bf16-no-shared.sh)
- Code KD: [`run_g2_ffn_only_code_expert_distill_init_mha.sh`](../scripts/experiment/a100/run_g2_ffn_only_code_expert_distill_init_mha.sh)
- Code 1-phase: [`run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh`](../scripts/experiment/a100/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh)
- Conversation KD: [`run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh`](../scripts/experiment/a100/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh)
- Conversation 1-phase: [`run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh`](../scripts/experiment/a100/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh)
- 기존 연속 체인: [`run_g2_ffn_only_joint_lm_code_kd_conversation_chain_mha.sh`](../scripts/experiment/a100/run_g2_ffn_only_joint_lm_code_kd_conversation_chain_mha.sh)

기존 FFN-only chain은 이미 만들어진 8→16 Code KD checkpoint를 source로 받아 Code 1-phase → Conversation KD → Conversation 1-phase의 3개 run을 수행한다. 따라서 최신 shared-router 4-stage chain처럼 Code KD부터 항상 새로 수행하는 entry는 아니다.

`run_g2_ffn_only_kd_ramp900_code_kd_ramp900_conversation_chain_mha.sh`는 LR ramp ablation이다. 현재 canonical 방법으로 사용하지 않는다.

### 2.2 FFN + attention expert, shared router

현재 canonical 실험이다. 각 layer에 다음 세 요소가 함께 있다.

1. FFN MoE expert
2. Q/K/V/O full-rank-capable LoRA expert
3. 두 expert family가 공유하는 layer-wise router와 Top-K routing map

즉 한 token에 대해 router가 고른 expert id가 FFN과 attention 양쪽에 동일하게 적용된다. Code row `8:16`이 선택되면 해당 FFN expert와 QKVO attention expert가 같은 route weight를 받는다.

- 모델 설정: [`flame-shared-router-hybrid-experts.sh`](../configs/model/flame-shared-router-hybrid-experts.sh)
- Transformer 구현: [`shared_router_hybrid.py`](../Megatron-LM/megatron/core/transformer/shared_router_hybrid.py)
- 공통 실행 backbone: [`continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh`](../scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh)
- Wiki source 학습: [`pretrain_wiki_shared_router_hybrid_local_bf16.sh`](../scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh)

`SharedFullRankLoraExperts`가 Q/K/V/O expert tensor와 grouped GEMM forward를 구현하고, `SharedRouterHybridTransformerLayer`가 router 결과를 attention과 FFN에 전달한다.

### 2.3 Attention-only baseline

Attention-only는 dense FFN과 routed QKVO attention expert를 조합한다. FFN hidden size 기본값 5632는 `16 × 352`와 같은 저장 parameter 규모를 맞추기 위한 baseline 설정이다.

- Entry: [`run_g2_attention_only_expert_mha.sh`](../scripts/experiment/a100/run_g2_attention_only_expert_mha.sh)
- 모델 설정: [`flame-attn-only-shared-router-qkvo-experts.sh`](../configs/model/flame-attn-only-shared-router-qkvo-experts.sh)

이 entry의 `wiki`, `code`, `phase3`, `all`은 별도 attention-only baseline 흐름이다. KD + joint-replay 4-stage canonical shared-router 체인과 같은 실험으로 해석하면 안 된다.

## 3. KD 기반 expert 초기화

### 3.1 목적

Expert 수를 늘리면 새 router row와 새 expert가 기존 경로를 방해하므로 expansion 직후 성능이 크게 떨어질 수 있다. KD stage는 새 expert family가 추가된 student가 이전 teacher의 출력을 재현하도록 초기화하는 단계다.

현재 실제 run은 `logits` 모드를 사용한다. 지원 모드는 세 가지다.

| Mode | 사용 loss |
|---|---|
| `logits` | output-logit KL |
| `logits_hidden` | output-logit KL + layer hidden MSE |
| `logits_hidden_router` | output-logit KL + layer hidden MSE + router probability KL |

KD stage에서는 LM loss coefficient가 0이고 aux/z loss도 0이다. 따라서 `logits` 모드의 최적화 objective는 token mask가 적용된 teacher-to-student KL이다.

```text
L_KD = coeff × T² × Σ_valid_token KL(p_teacher(T) || p_student(T))
```

Hidden mode는 지정 layer의 masked hidden-state MSE를 더한다. Router mode는 teacher router 확률을 student의 확장된 expert dimension 앞부분에 놓고 나머지를 0으로 padding한 target과 KL을 계산한다.

- Loss 합산과 reporting: [`pretrain_gpt.py`](../Megatron-LM/pretrain_gpt.py)
- Router KL: [`continual_learning_utils.py`](../Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py)
- CLI arguments: [`arguments.py`](../Megatron-LM/megatron/training/arguments.py)

### 3.2 Code KD: 8 → 16

Entry: [`run_g2_shared_router_code_expert_distill_init_mha.sh`](../scripts/experiment/a100/run_g2_shared_router_code_expert_distill_init_mha.sh)

- Teacher: Wiki shared-router 8-expert checkpoint
- Student: Wiki row를 복사한 16-expert 모델
- KD data: Wiki train only
- Trainable: 새 FFN `8:16`, 새 QKVO `8:16`, 새 router row `8:16`
- Frozen: 기존 FFN/QKVO `0:8`, 기존 router `0:8`, dense/shared trunk
- Canonical mode/step/MB: `logits`, 600, 48
- Optimizer state 저장: 안 함

Canonical Wiki source:

```text
.local/weights/a100/mha/g2-checkpoints/wiki/
g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800
```

### 3.3 Conversation KD: 16 → 24

Entry: [`run_g2_shared_router_conversation_expert_distill_init_wikicode_mha.sh`](../scripts/experiment/a100/run_g2_shared_router_conversation_expert_distill_init_wikicode_mha.sh)

- Teacher: Code 1-phase가 끝난 16-expert checkpoint
- Student: 기존 row를 복사한 24-expert 모델
- KD data: Wiki + Code `equal_dataset`
- Trainable: 새 FFN `16:24`, 새 QKVO `16:24`, 새 router row `16:24`
- Frozen: 기존 FFN/QKVO `0:16`, 기존 router `0:16`, dense/shared trunk
- Canonical mode/step/MB: `logits`, 600, 36
- Optimizer state 저장: 안 함

### 3.4 Expansion 초기값

기존 expert와 router row는 source checkpoint에서 그대로 복사한다. 새 attention LoRA expert의 A는 초기화되고 B는 0으로 시작하므로 최초 출력 delta는 0이다. 학습 초반에는 B가 먼저 유효 gradient를 받고 이후 A와 B가 함께 적응한다.

Expansion, row freeze, audit 구현은 [`continual_learning_utils.py`](../Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py)와 [`training.py`](../Megatron-LM/megatron/training/training.py)에 있다.

## 4. 1-phase joint LM 학습

### 4.1 핵심 의미

여기서 1-phase는 새 task LM과 old-data router replay를 서로 다른 phase/checkpoint로 나누지 않고, 한 optimizer update 안에서 두 backward 결과를 합치는 방식이다.

```text
primary task forward/backward
  └─ 새 task FFN + 새 task QKVO + 전체 router gradient 유지

old-data replay forward/backward
  ├─ 전체 router gradient는 primary gradient에 누적
  └─ router가 아닌 gradient는 replay 전 snapshot으로 복원

single optimizer.step()
```

구현 순서는 다음과 같다.

1. Primary LM forward/backward를 실행한다.
2. Router를 제외한 trainable parameter gradient를 snapshot한다.
3. Replay 동안 aux/z loss를 0으로 두고 old-data LM forward/backward를 실행한다.
4. Router가 아닌 gradient를 snapshot 값으로 복원한다.
5. Primary와 replay에서 누적된 router gradient를 포함해 optimizer step을 한 번 실행한다.

따라서 old data는 expert를 다시 학습하지 않고 router의 LM gradient만 제공한다. 핵심 구현은 [`training.py`](../Megatron-LM/megatron/training/training.py)의 `_snapshot_joint_replay_non_router_grads`, `_restore_joint_replay_non_router_grads`, `_activate_moe_joint_replay_optimizer`, train-step joint replay block이다.

### 4.2 Code 1-phase

Entry: [`run_g2_shared_router_code_wiki_joint_lm_allrouter_mha.sh`](../scripts/experiment/a100/run_g2_shared_router_code_wiki_joint_lm_allrouter_mha.sh)

| 입력 | 남는 gradient |
|---|---|
| Code primary LM | 새 FFN `8:16` + 새 QKVO `8:16` + router `0:16` |
| Wiki replay LM | router `0:16`만 |

- Code:Wiki backward 비율: 1:1
- KD: off
- LR ramp: off
- Step/MB: 1800 / 96
- Old FFN/QKVO와 dense/shared trunk: frozen

### 4.3 Conversation 1-phase

Entry: [`run_g2_shared_router_conversation_wikicode_joint_lm_allrouter_mha.sh`](../scripts/experiment/a100/run_g2_shared_router_conversation_wikicode_joint_lm_allrouter_mha.sh)

| 입력 | 남는 gradient |
|---|---|
| Conversation primary LM | 새 FFN `16:24` + 새 QKVO `16:24` + router `0:24` |
| Wiki+Code replay LM | router `0:24`만 |

Replay 안에서 Wiki와 Code는 `equal_dataset`으로 동일 비중이다. Primary 1회와 replay 1회를 합치므로 effective dataset 비율은 Wiki:Code:Conversation = 1:1:2다.

- KD: off
- LR ramp: off
- Step/MB: 1800 / 96
- Old FFN/QKVO와 dense/shared trunk: frozen

## 5. 최신 shared-router 4-stage 실행 구조

Chain script는 각 stage의 tracker를 확인하고 source checkpoint가 정확한 expected step에 도달했을 때만 다음 stage를 시작한다. 이미 완료된 stage는 skip할 수 있고, `RUN_ONLY_STAGE`로 한 단계만 실행할 수도 있다.

```bash
# 설정과 output path만 확인
PLAN_ONLY=1 \
  bash scripts/experiment/a100/run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh logits

# 특정 stage만 실행
RUN_ONLY_STAGE=conversation \
  bash scripts/experiment/a100/run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh logits
```

허용되는 `RUN_ONLY_STAGE` 값은 `all`, `code_init`, `code`, `conversation_init`, `conversation`이다.

Canonical checkpoint 경로는 다음과 같다. 긴 basename은 chain의 `FIX_TAG=attn-loadfix-v3-nosaveoptim`으로 격리되어 있다.

```text
Stage 0
.local/weights/a100/mha/g2-checkpoints/code/shared_router_expansion_distill_init/
g2-shared-router-e8to16-code-expert-init-logits-wiki-distill-qkvo-mha-a100-bf16-mb48-600-attn-loadfix-v3-nosaveoptim

Stage 1
.local/weights/a100/mha/g2-checkpoints/code/shared_router_joint_lm_replay/
g2-shared-router-code-wiki-joint-lm-allrouter-logits-init-mb96-1800-attn-loadfix-v3-nosaveoptim

Stage 2
.local/weights/a100/mha/g2-checkpoints/conversation/shared_router_expansion_distill_init_joint_code/
g2-shared-router-e16to24-conversation-expert-init-logits-wikicode-distill-qkvo-mha-a100-bf16-mb36-600-attn-loadfix-v3-nosaveoptim

Stage 3
.local/weights/a100/mha/g2-checkpoints/conversation/shared_router_joint_lm_replay/
g2-shared-router-conv-wikicode-joint-lm-allrouter-112-logits-init-mb96-1800-attn-loadfix-v3-nosaveoptim
```

### 실행 로그

```text
.local/logs/
g2_shared_router_joint_lm_code_kd_conversation_chain_attn-loadfix-v3-nosaveoptim/
```

각 로그를 독립적으로 보는 명령은 다음과 같다.

```bash
tail -n 100 -F .local/logs/g2_shared_router_joint_lm_code_kd_conversation_chain_attn-loadfix-v3-nosaveoptim/stage0_code_kd_init.log

tail -n 100 -F .local/logs/g2_shared_router_joint_lm_code_kd_conversation_chain_attn-loadfix-v3-nosaveoptim/stage1_code_wiki_joint.log

tail -n 100 -F .local/logs/g2_shared_router_joint_lm_code_kd_conversation_chain_attn-loadfix-v3-nosaveoptim/stage2_conversation_wikicode_kd.log

tail -n 100 -F .local/logs/g2_shared_router_joint_lm_code_kd_conversation_chain_attn-loadfix-v3-nosaveoptim/stage3_conversation_wikicode_joint.log
```

Directory와 filename 사이에 줄바꿈을 넣으면 `tail`이 directory 자체를 열려고 하므로 전체 파일 경로를 한 줄로 입력해야 한다.

## 6. 2026-08-02 최신 실행 스냅샷

아래 수치는 설계 기본값이 아니라 `attn-loadfix-v3-nosaveoptim` 실제 로그의 마지막 확인값이다.

| Stage | Tracker / 실행 상태 | 마지막 probe next-token accuracy |
|---|---|---|
| Code KD | 600 완료 | Wiki 0.467369, Code 0.415502 |
| Code 1-phase | 1800 완료 | Code 0.679828, Wiki 0.467134 |
| Conversation KD | 600 완료 | Wiki 0.466778, Code 0.679568, Conversation 0.293593 |
| Conversation 1-phase | 실행 중, tracker 1200 | Conversation 0.379707, Wiki 0.465497, Code 0.679066 |

Stage 3 process는 문서 작성 시점에도 실행 중이었다. 최종 1800 결과는 이 표의 1200-step snapshot으로 대체하면 안 된다.

## 7. Attention expert 추가 후 발견된 문제와 수정

### 7.1 Checkpoint load에서 기존 attention expert가 사라지던 문제

과거 finetune loader는 key 이름에 `full_rank_lora`가 있으면 일괄 제거했다. Wiki source checkpoint에 학습된 QKVO A/B tensor가 있어도 teacher와 student의 기존 attention expert가 checkpoint 값이 아니라 random A/zero B로 만들어졌다. 이 상태에서 KD가 수렴해도 원래 Wiki 모델의 동작을 보존하는 초기화가 될 수 없었다.

현재 [`checkpointing.py`](../Megatron-LM/megatron/training/checkpointing.py)는 distributed-checkpoint metadata를 먼저 읽고, checkpoint에 실제로 없는 adapter tensor만 선택적으로 제거한다. 정상적인 source load에서는 다음과 같은 로그가 나와야 한다.

```text
Finetune full-rank LoRA checkpoint filter: stripped 0 missing tensor(s); checkpoint-backed tensors will be loaded.
```

### 7.2 Gradient mask만으로 기존 row가 보존되지 않던 문제

Old/new expert가 한 tensor의 서로 다른 row에 들어갈 때 gradient hook으로 old row를 0으로 만들어도 AdamW의 decoupled weight decay는 old row를 변경할 수 있다.

현재 partial-row freeze tensor에는 `_exclude_from_weight_decay_for_frozen_rows` marker를 붙이고 optimizer group의 weight decay를 0으로 둔다. Freeze hook 설치 뒤 optimizer를 다시 만들어 marker가 param group에 반영되도록 했다.

- Freeze/marker: [`continual_learning_utils.py`](../Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py)
- Optimizer group: [`Megatron-LM/megatron/core/optimizer/__init__.py`](../Megatron-LM/megatron/core/optimizer/__init__.py)
- Freeze 후 optimizer rebuild: [`training.py`](../Megatron-LM/megatron/training/training.py)

### 7.3 KD loss 표시가 이중 정규화되던 문제

Optimization objective는 맞았지만 표시용 `kd loss`가 token count로 다시 나뉘어 실제보다 작게 기록되던 경로가 있었다. 현재 reporting은 `kl_loss_sum × T²`를 numerator로 사용하고 공통 denominator와 한 번만 결합한다. 구현과 regression test는 각각 [`pretrain_gpt.py`](../Megatron-LM/pretrain_gpt.py), [`test_kd_loss_reporting.py`](../tests/test_kd_loss_reporting.py)에 있다.

### 7.4 W&B relog의 빈 경로 문제

Shell 변수에 빈 checkpoint/log 경로가 들어간 상태를 relog script가 허용해 서로 다른 stage가 같은 데이터로 올라가거나 비정상 spike가 생길 수 있었다. Conversation KD에서 Wiki accuracy가 0.48987로 보인 기록은 실제 probe 결과가 아니라 이 relog 입력 문제에 의한 잘못된 기록이었다.

현재 [`relog_expansion_distill_pipeline_to_wandb.py`](../scripts/analysis/relog_expansion_distill_pipeline_to_wandb.py)는 빈 경로를 거부한다. Upload 전에는 모든 shell 변수가 실제 파일/디렉터리인지 확인해야 한다.

### 7.5 KD checkpoint optimizer-state 저장 문제

Partial expert row로 optimizer를 rebuild한 뒤 distributed checkpoint optimizer state의 sharding metadata가 맞지 않아 KD 종료 시 save가 실패한 run이 있었다. 다음 1-phase는 항상 optimizer를 reset하므로 KD-init checkpoint에 optimizer state가 필요하지 않다.

Canonical chain은 Stage 0과 Stage 2에 `NO_SAVE_OPTIM=1`을 강제한다. Stage 1과 Stage 3도 현재 wrapper 기본값은 no-save-optim이며 resume 시 optimizer를 로드하지 않는다.

## 8. Smoke test와 regression test

### 8.1 확인된 production smoke artifact

Attention load/expansion 수정 뒤 다음 로컬 artifact가 생성되었다.

```text
.local/smoke/g2_attn_loadfix_v2/weights/expansion_audit/
shared_router_hybrid_expand_8_to_16.json

.local/smoke/g2_attn_loadfix_v2/trainable_params_debug.json

.local/smoke/g2_attn_loadfix_v3_save/weights/
latest_checkpointed_iteration.txt
```

- 8개 shared-router layer의 기존 attention row가 source와 일치했다.
- Expansion audit의 최대 차이는 0이었다.
- Trainable parameter audit의 attention row check 32개가 통과했다.
- 1-step production train+save tracker가 1로 기록되었다.
- 저장된 checkpoint에서 QKV/O attention tensor를 다시 선택 load하는 smoke가 통과했다.

### 8.2 관련 unit/regression test

```text
Megatron-LM/tests/unit_tests/test_checkpointing_attn_full_rank_lora.py
Megatron-LM/tests/unit_tests/transformer/moe/test_shared_full_rank_lora_expansion_boundaries.py
Megatron-LM/tests/unit_tests/transformer/moe/test_continual_learning_utils.py
Megatron-LM/tests/unit_tests/test_joint_replay_lm.py
Megatron-LM/tests/unit_tests/transformer/moe/test_joint_replay_boundary.py
tests/test_kd_loss_reporting.py
tests/test_relog_expansion_distill_pipeline.py
```

이 테스트들은 최소한 다음 경계를 검증해야 한다.

- Checkpoint에 존재하는 기존 attention LoRA tensor를 제거하지 않는가
- Expansion 후 old FFN/QKVO/router row가 source와 같은가
- AdamW step 뒤 frozen old row가 bitwise 동일한가
- Replay backward 뒤 non-router gradient가 primary 값으로 복원되는가
- Replay router gradient는 primary router gradient에 누적되는가
- KD reporting이 실제 objective scale과 일치하는가
- Relog가 빈 경로를 거부하는가

## 9. 코드 위치 지도

| 기능 | 파일 |
|---|---|
| Canonical 4-stage orchestration | [`run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh`](../scripts/experiment/a100/run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh) |
| Shared-router 모델 args | [`flame-shared-router-hybrid-experts.sh`](../configs/model/flame-shared-router-hybrid-experts.sh) |
| QKVO expert와 shared routing forward | [`shared_router_hybrid.py`](../Megatron-LM/megatron/core/transformer/shared_router_hybrid.py) |
| Teacher/student forward, KD loss | [`pretrain_gpt.py`](../Megatron-LM/pretrain_gpt.py) |
| Expansion, copy, freeze, router KL, audit | [`continual_learning_utils.py`](../Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py) |
| Source load, expansion/resume, joint replay, LR ramp | [`training.py`](../Megatron-LM/megatron/training/training.py) |
| Attention tensor checkpoint filtering | [`checkpointing.py`](../Megatron-LM/megatron/training/checkpointing.py) |
| Frozen-row weight decay 제외 | [`Megatron-LM/megatron/core/optimizer/__init__.py`](../Megatron-LM/megatron/core/optimizer/__init__.py) |
| CLI argument와 validation | [`arguments.py`](../Megatron-LM/megatron/training/arguments.py) |
| Dataset staging, launch args, log formatting | [`continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh`](../scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh) |

## 10. 실행 전 점검표

1. Source tracker가 정확한 expected step인지 확인한다.
2. Source checkpoint metadata에 QKVO `full_rank_lora` tensor가 존재하는지 확인한다.
3. Load log에서 checkpoint-backed attention tensor가 strip되지 않았는지 확인한다.
4. Expansion audit에서 old FFN/QKVO/router row 차이가 0인지 확인한다.
5. Trainable audit에서 새 task row와 의도한 router row만 trainable인지 확인한다.
6. KD에서는 LM/aux/z loss가 0이고 선택한 distill loss만 objective에 들어가는지 확인한다.
7. 1-phase에서는 KD가 off이고 joint replay가 on인지 확인한다.
8. Canonical run에서는 `MOE_NEW_EXPERT_LR_RAMP_STEPS=0`인지 확인한다.
9. Code는 MB 48 → 96, Conversation은 MB 36 → 96인지 확인한다.
10. Probe step offset과 W&B step offset이 stage 누적 step과 맞는지 확인한다.
11. 완료 판정은 log 마지막 줄이 아니라 `latest_checkpointed_iteration.txt`와 expected step을 함께 확인한다.

## 11. Git push 전 주의사항

현재 기능은 outer repository의 shell/Python 파일과 `Megatron-LM` 내부 변경을 함께 사용한다. `Megatron-LM`이 submodule이라면 내부 commit을 먼저 만들고 outer repository에서 submodule pointer를 commit해야 한다.

또한 최신 shared-router stage script 일부가 현재 작업 트리에서 untracked일 수 있다. 문서만 push하고 실행 script를 누락하지 않도록 다음 명령으로 확인한다.

```bash
git status --short
git -C Megatron-LM status --short
```

기존 dirty change는 다른 실험 작업일 수 있으므로 일괄 `git add -A`보다 이번 기능에 필요한 파일을 명시적으로 stage하는 편이 안전하다.
