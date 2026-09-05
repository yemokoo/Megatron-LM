# Six-baseline 구현/학습 세션 인계 프롬프트

아래 작업을 이어받아라. 먼저 파일과 현재 프로세스를 읽기 전용으로 확인하고, 이미 실행 중인 체인을 중복 실행하지 마라. 기존 학습 경로와 사용자 변경은 보존해야 한다.

## 1. 절대 조건

- 저장소: `/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning`
- 기존 `Megatron-LM/pretrain_gpt.py`, `Megatron-LM/megatron/training/training.py`, 기존 `scripts/experiment/a100/*` launcher는 이 baseline 구현을 위해 수정하지 않았다. 계속 수정하지 마라.
- 새 구현은 `Megatron-LM/pretrain_gpt_baselines6.py`를 진입점으로 쓰는 독립 분기다. 이 프로세스 안에서만 hook을 설치하므로 기존 진입점은 새 모듈을 import하지 않는다.
- worktree에는 이 작업과 무관한 사용자 변경 및 untracked 파일이 매우 많다. reset, checkout, clean, 삭제를 하지 말고 `baselines6`와 `megatron/core/continual_learning` 범위만 다뤄라.
- GPU는 6,7만 사용한다. 0–5의 다른 작업을 건드리지 마라.
- old-model KD, replay, expansion은 지정한 경우 외에는 모두 금지다. `moe_old_model_kl_coeff`는 audit에서 0이어야 한다.

## 2. 구현 파일 지도

전용 진입점:

- `Megatron-LM/pretrain_gpt_baselines6.py`

핵심 구현:

- `Megatron-LM/megatron/core/continual_learning/branch.py`: 전용 argument와 process-local model/loss/training hook 설치
- `Megatron-LM/megatron/core/continual_learning/runtime.py`: task state, regularizer, optimizer-step hook, Fisher/terminal-gradient/merge/final audit
- `Megatron-LM/megatron/core/continual_learning/layer_specs.py`: Layer 1 FFN 5472, Layer 2–9 FFN 1408, SLoRA/O-LoRA adapter가 붙은 attention/MLP spec
- `Megatron-LM/megatron/core/continual_learning/lora_adapter.py`: checkpointable max-rank LoRA와 active-rank prefix
- `Megatron-LM/megatron/core/continual_learning/parameter_scope.py`: Wiki 및 post-Wiki trainability/freeze/checksum
- `Megatron-LM/megatron/core/continual_learning/ewc.py`: diagonal empirical-Fisher EWC 및 equal-lambda quadratic consolidation
- `Megatron-LM/megatron/core/continual_learning/trace_gem.py`: TRACE-compatible parameter별 terminal-gradient projection
- `Megatron-LM/megatron/core/continual_learning/slora.py`: immutable Wiki reference, candidate-rank denoising, merge/clear
- `Megatron-LM/megatron/core/continual_learning/olora.py`: output factor B의 squared-Frobenius 직교항과 slot norm audit
- `Megatron-LM/megatron/core/continual_learning/state.py`: TP/PP별 sidecar 저장/로드
- `Megatron-LM/megatron/core/continual_learning/audit.py`: 활성 parameter 수 invariant와 JSON audit

테스트:

- `Megatron-LM/tests/unit_tests/continual_learning/test_math.py`
- `Megatron-LM/tests/unit_tests/continual_learning/test_adapters.py`

실행/검증:

- `scripts/experiment/a100/baselines6/common.sh`
- `scripts/experiment/a100/baselines6/train_stage.sh`
- `scripts/experiment/a100/baselines6/train_dense_wiki.sh`
- `scripts/experiment/a100/baselines6/{ewc,trace_gem,slora_pre,olora,sequential_dense,fixed_moe}.sh`
- `scripts/experiment/a100/baselines6/run_all.sh`
- `scripts/experiment/a100/baselines6/smoke_all.sh`
- `scripts/experiment/a100/baselines6/launch_full_background.sh`
- `scripts/experiment/a100/baselines6/audit_runs.py`
- `scripts/experiment/a100/baselines6/README_KO.md`

## 3. 공통 실험 정의

- task 순서: Wiki → Code → Conversation
- hidden size 1024, 9 layers, attention heads 16, query groups 16
- Layer 1 FFN 5472 고정 구조
- dense Layer 2–9 SwiGLU FFN 1408
- BF16, seed 1234, sequence length 512
- GBS 2304, task당 1800 optimizer steps
- Wiki 이후 Layer 1, embedding, final norm, LM head 등 비대상 parameter는 bit-identical이어야 한다.
- EWC/TRACE-GEM/SLoRA-Pre/Sequential Dense는 동일한 `common_dense/wiki` checkpoint를 사용한다.
- O-LoRA와 Fixed MoE는 구조가 달라 별도의 Wiki checkpoint를 학습한다.
- layer별 projection activation parameter 수:
  - dense FFN: `3 × 1024 × 1408 = 4,325,376`
  - Fixed MoE top-4: `4 × 3 × 1024 × 352 = 4,325,376`
  - 최종 O-LoRA Q/V 3 slots: `3 × 2 × (1024×352 + 352×1024) = 4,325,376`
  - Fixed MoE router overhead 24,576/layer는 별도 audit한다.

데이터:

- `/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup`

기본 microbatch:

- EWC/TRACE-GEM/common dense Wiki: 64
- Sequential Dense: 72
- SLoRA-Pre: 24
- O-LoRA: 32
- Fixed MoE: 48

모두 DP=2에서 GBS 2304를 정확히 나누고 GPU 6,7 smoke에서 OOM 없이 검증됐다.

## 4. 방법별 정확한 구현

### EWC

- 원 논문식 task-boundary diagonal empirical Fisher를 사용한다.
- 마지막 minibatch gradient를 Fisher처럼 저장하는 TRACE EWC는 사용하지 않는다.
- Wiki 및 Code 종료 시 순수 LM NLL로 최대 100 minibatch gradient 제곱 평균을 계산한다.
- Layer 2–9 attention, FFN, RMSNorm의 48 tensors가 대상이다.
- lambda=400, penalty는 `0.5 * lambda * Σ F_i(θ_i-θ*_i)^2`다.
- Conv 전에 Wiki/Code quadratic을 Fisher-weighted anchor로 정확히 합친다.
- Fisher 계산 시간 절약을 위해 100 minibatch 상한, cyclic dataloader, GPU state cache를 사용한다.
- sidecar: `continual_state_ewc/state_tp00_pp00.pt`

### TRACE-GEM

- 이것은 canonical GEM의 episodic-example replay/global FP32 vector QP가 아니다. 사용자 합의에 따라 로컬 TRACE 구현 방식을 재현한 `TRACE-GEM`이다.
- task 마지막 optimizer gradient를 parameter별 BF16 CPU memory로 저장한다.
- 로컬 `trace/patches/trace-gem-qpth-sign.patch`의 수정 의미를 따른다: `q=M g`, `G=-I`, `h=-margin`, 즉 dual coefficient `v >= margin`이다.
- margin을 projected primal dot-product 하한으로 해석하면 안 된다.
- Wiki/Code/Conversation에서는 memory가 최대 2개만 projection에 쓰이므로 qpth 대신 정확한 lower-bound active-set 열거를 사용한다.
- sidecar memory 수는 Wiki=1, Code=2, Conversation=3이어야 한다.

### SLoRA-Pre

- 공통 Wiki dense weight를 immutable reference `theta_0`로 저장한다.
- Code는 fresh LoRA 하나를 rank 16/32/64/128/256 각각 별도 run으로 학습한다.
- 대상은 Layer 2–9 Q/K/V/O와 SwiGLU gate/up/down, 총 56 target이다.
- task 종료 시 trained delta SVD의 10–100% candidate rank를 검사해 denoise한 뒤 base에 merge하고 adapter를 clear한다.
- Conversation은 merge된 Code checkpoint에서 fresh rank-64 LoRA를 학습하고 다시 denoise/merge한다.
- alpha=128. 내부 parameter storage는 checkpoint shape 호환을 위해 max rank 256이며 forward/merge에는 active prefix만 사용한다.
- Code와 Conversation sidecar의 Wiki reference checksum은 동일해야 한다.

### O-LoRA

- Layer 2–9 Q/V에만 적용한다. router와 task id는 없다.
- Wiki는 base + slot0을 공동 학습한다.
- Code는 base/slot0을 freeze하고 slot1만 학습하며 slot0 output factor B와 직교시킨다.
- Conversation은 slot2만 학습하며 slot0, slot1의 output factor B 모두와 직교시킨다.
- regularizer는 L1이 아니라 squared Frobenius: `lambda * Σ ||B_old^T B_new||_F^2`에 해당하는 로컬 convention이다.
- rank=352, alpha=352, dropout=0.1, lambda=0.5.
- forward는 지금까지의 모든 slot을 합산한다: Wiki 1개, Code 2개, Conversation 3개.
- 최종 3-slot 활성 projection 수가 Fixed MoE top-4와 정확히 같다.
- `final.json`에 regularizer call/value와 세 slot A/B norm이 기록된다.

### Sequential Dense

- 별도 regularizer/replay/KD 없이 공통 Wiki dense를 시작점으로 쓴다.
- Code, Conversation에서 Layer 2–9 attention/FFN/RMSNorm을 순서대로 overwrite 학습한다.

### Fixed MoE

- Layer 1은 dense FFN 5472다.
- Layer 2–9는 처음부터 끝까지 24 experts × FFN 352, top-4 고정이다.
- 확장, old-model KD, replay가 없다.
- aux loss=0.01, router z-loss=0.001.
- Wiki→Code→Conversation에서 동일 구조를 overwrite 학습한다.

## 5. 중요하게 수정·검증한 기술 문제

- heterogeneous Layer 1/Layer 2–9 dense checkpoint key 충돌을 피하려고 dense config에 all-zero `moe_layer_freq` list를 checkpoint non-homogeneous marker로만 쓴다. 실제 MoE layer를 만들지 않는다.
- Fisher는 unwrapped model의 이름을 쓰되 forward/backward 및 grad-buffer lifecycle은 DDP wrapper를 사용한다.
- EWC Fisher 전에 finite iterator가 소진되지 않도록 전용 launcher는 cyclic dataloader를 쓴다.
- fresh SLoRA/O-LoRA reset은 checkpoint/optimizer 생성 후 발생하므로 `optimizer.reload_model_params()`로 BF16 optimizer의 FP32 master parameter를 반드시 동기화한다. 이 동기화가 없으면 첫 update가 fresh LoRA를 stale zero로 덮는다.
- 2-step smoke의 첫 WSD update는 실효 LR=0이므로 O-LoRA 비영 orthogonality를 확인할 때는 3-step smoke를 사용했다.
- Fixed MoE 로그의 theoretical parameter/memory estimator는 base dense FFN 5472를 참조해 과대 표기될 수 있다. 실제 constructor/audit parameter는 정상이며 smoke peak는 안전했다.
- 종료 시 PyTorch NCCL process-group warning과 flash-attn v3 미설치 warning이 나오지만 모든 검증 run은 exit code 0이었다.

## 6. 완료된 검증

- unit tests: 9 passed
- Python `py_compile`, 모든 baseline shell `bash -n`, `git diff --check` 통과
- smoke root: `/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_smoke_20260816`
- 15 stages 완료: 공통 Wiki 1 + EWC 2 + TRACE-GEM 2 + SLoRA rank16 2 + O-LoRA 3 + Sequential Dense 2 + Fixed MoE 3
- smoke audit 명령 결과: `OK: 15 stages, ranks=[16], train_iters=2, active=4,325,376`
- 모든 post-Wiki `frozen_bit_identical == true`, KD coefficient 0
- EWC Fisher 48 tensors, TRACE memory 1→2→3, SLoRA reference 56 targets/checksum, O-LoRA slot 1→2→3 확인
- corrected O-LoRA 3-step root: `/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_smoke_regularizer3_20260816`
- O-LoRA 비영 penalty: Code `3.809401e-05`, Conversation `6.053238e-04`
- Conversation 최종 O-LoRA slot 0/1/2 A/B norm 모두 비영

unit test 재실행:

```bash
cd /home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29877 \
PYTHONPATH=$PWD/Megatron-LM \
/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python -m pytest -q \
  Megatron-LM/tests/unit_tests/continual_learning/test_math.py \
  Megatron-LM/tests/unit_tests/continual_learning/test_adapters.py
```

## 7. 현재 전체 체인

- full output root: `/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816`
- PID file: `$FULL_ROOT/chain.pid`
- integrated log: `$FULL_ROOT/chain.log`
- 2026-08-17 00:02 KST 확인 당시 PID `529565`가 PPID 1로 살아 있었고 공통 dense Wiki `29/1800`, 약 333 TFLOP/s/GPU로 정상 진행 중이었다.
- 전체는 23 stages다: common Wiki 1 + EWC 2 + TRACE-GEM 2 + SLoRA 5 ranks×2 + O-LoRA 3 + Sequential Dense 2 + Fixed MoE 3.

먼저 다음처럼 중복 실행 여부를 확인하라.

```bash
FULL_ROOT=/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816
PID=$(tr -d '[:space:]' < "$FULL_ROOT/chain.pid")
ps -o pid,ppid,stat,etime,cmd -p "$PID"
tail -100 "$FULL_ROOT/chain.log"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

PID가 살아 있으면 절대 다시 launch하지 말고 모니터링만 해라. PID가 죽었을 때만 아래 launcher를 실행하라. 각 stage는 tracker와 final audit가 모두 완료된 경우 skip하고, 미완료 checkpoint는 resume한다.

```bash
cd /home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning
CUDA_VISIBLE_DEVICES=6,7 \
BASELINES6_OUTPUT_ROOT=/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816 \
bash scripts/experiment/a100/baselines6/launch_full_background.sh
```

## 8. 전체 완료 후 필수 작업

23 stages가 끝나면 다음 audit을 실행하라.

```bash
cd /home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning
/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python \
  scripts/experiment/a100/baselines6/audit_runs.py \
  /data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816 \
  --train-iters 1800 \
  --slora-ranks 16 32 64 128 256 \
  --require-regularizer-audit \
  --require-positive-olora
```

기대 출력은 다음과 같다.

```text
OK: 23 stages, ranks=[16, 32, 64, 128, 256], train_iters=1800, active=4,325,376
```

실패 시 해당 stage의 `logs/train.log`, `continual_audit/setup.json`, tracker, sidecar부터 확인하라. 기존 학습 파일을 고쳐 우회하지 말고 이 독립 branch 안에서만 원인을 해결하라. 전체 완료/audit 통과 후 method별 checkpoint 위치, final probe 성능, Fisher/regularizer/slot/rank audit 요약을 사용자에게 보고하라.
