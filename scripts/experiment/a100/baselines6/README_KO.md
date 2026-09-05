# Wiki → Code → Conversation 6개 baseline

이 디렉터리는 기존 `pretrain_gpt.py`, `megatron/training/training.py`, 기존 A100 launcher를 수정하거나 호출하지 않는 독립 분기다. 전용 entrypoint는 `Megatron-LM/pretrain_gpt_baselines6.py`이며 프로세스 내부에서만 baseline hook을 설치한다. 따라서 기존 launcher가 `pretrain_gpt.py`를 실행할 때는 이 구현을 import하지 않는다.

방법은 EWC, TRACE-GEM, SLoRA-Pre, O-LoRA, Sequential Dense, Fixed MoE다. 공통 dense Wiki checkpoint는 EWC Fisher와 TRACE terminal gradient를 모두 sidecar로 남기며 EWC/TRACE-GEM/SLoRA/Sequential Dense가 동일한 Wiki weight를 사용한다. O-LoRA와 Fixed MoE는 구조가 달라 별도 Wiki run을 사용한다.

공통 구조는 hidden 1024, 9 layers, MHA 16 groups다. Layer 1 FFN은 5472이고, dense Layer 2–9 FFN은 1408이다. Fixed MoE는 Layer 2–9에 24×352 experts, top-4를 처음부터 끝까지 고정한다. Wiki 이후에는 Layer 1, embedding, final norm, LM head가 checksum 기준 bit-identical이어야 한다.

## 방법별 정의

- EWC: Layer 2–9의 attention, FFN, RMSNorm 전체에 task-boundary diagonal empirical Fisher를 사용한다. Wiki와 Code 종료 시 각각 최대 100 minibatch의 순수 LM NLL gradient 제곱을 계산하고, 같은 lambda의 이전 quadratic들을 정확히 합쳐 다음 task에 적용한다. `lambda=400`, replay/KD 없음.
- TRACE-GEM: 로컬 TRACE 구현의 terminal-gradient memory 및 parameter별 projection 방식을 유지한다. 수정된 qpth dual 부호 `q=M g, G=-I, h=-margin`과 동일한 해를 두 개 이하의 memory에 대한 exact active-set으로 계산한다. episodic replay는 추가하지 않는다.
- SLoRA-Pre: 공통 Wiki dense weight를 immutable reference로 고정한다. Code에는 fresh LoRA를 rank 16/32/64/128/256으로 각각 학습하고, 10–100% candidate rank denoising 뒤 merge한다. Conversation에는 fresh rank-64 LoRA를 같은 방식으로 학습/merge한다. 대상은 Layer 2–9 Q/K/V/O와 gate/up/down이다.
- O-LoRA: Wiki는 base와 slot 0을 함께 학습하고, Code는 slot 1의 output factor B가 slot 0과 수직이 되도록, Conversation은 slot 2가 slot 0/1과 수직이 되도록 squared-Frobenius regularizer를 적용한다. Layer 2–9 Q/V만 사용하며 rank/alpha=352, dropout=0.1, lambda=0.5다. 세 slot은 누적 합산되고 router/task-id는 없다.
- Sequential Dense: 공통 Wiki dense checkpoint에서 시작해 Layer 2–9 attention/FFN/RMSNorm을 Code, Conversation 순서로 그대로 overwrite 학습한다. 별도 정규화/replay/KD 없음.
- Fixed MoE: 별도 Wiki부터 Layer 2–9에 24 experts × FFN 352, top-4를 고정한다. 확장 없이 Wiki→Code→Conversation에서 동일 구조를 overwrite 학습한다. aux=0.01, z=0.001이며 replay/KD 없음.

Layer별 projection activation은 dense FFN, Fixed-MoE top-4, 최종 O-LoRA 3개 Q/V slot 모두 `4,325,376`으로 맞춘다. Fixed-MoE router의 layer별 24,576 parameter는 별도 overhead로 audit한다.

## 학습과 상태

모든 task는 BF16, seed 1234, GBS 2304, 1,800 optimizer steps이고 순서는 Wiki → Code → Conversation이다. 공통 dense Wiki는 한 번만 학습하여 EWC/TRACE-GEM/SLoRA-Pre/Sequential Dense가 동일 초기 weight를 사용한다. task 경계 상태는 checkpoint 내부 모델과 별개로 다음 sidecar에 기록한다.

- `continual_state_ewc`: Fisher 합과 anchor
- `continual_state_trace_gem`: task별 terminal gradient
- `continual_state_slora_pre`: immutable Wiki reference/checksum과 retained rank
- `continual_state_olora`: 활성 slot 수와 rank/alpha/lambda

각 stage는 `continual_audit/setup.json`, `final.json`을 남긴다. post-Wiki stage는 frozen checksum이 bit-identical하지 않거나 EWC/O-LoRA regularizer가 loss에서 한 번도 호출되지 않으면 실패한다. SLoRA/O-LoRA의 fresh adapter reset 뒤에는 BF16 optimizer의 FP32 master copy를 명시적으로 재동기화한다.

검증:

```bash
bash scripts/experiment/a100/baselines6/smoke_all.sh
```

Smoke는 기본 2 step이며 전체 구조/checkpoint/sidecar/freeze 검증용이다. O-LoRA의 비영 orthogonality 값까지 보려면 첫 warmup update가 0-LR인 점을 고려해 3 step으로 실행한다.

```bash
BASELINES6_SMOKE=1 SMOKE_TRAIN_ITERS=3 \
BASELINES6_SMOKE_ROOT=/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_smoke_olora3 \
bash scripts/experiment/a100/baselines6/olora.sh
```

전체 순차 실행:

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/experiment/a100/baselines6/run_all.sh
```

터미널과 분리해 실행:

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/experiment/a100/baselines6/launch_full_background.sh
```

PID는 출력 root의 `chain.pid`, 통합 stdout/stderr는 `chain.log`에 기록된다.

기본 microbatch는 dense regularizer 64, Sequential Dense 72, SLoRA 24, O-LoRA 32, Fixed MoE 48이다. 모두 GBS 2304를 정확히 나눈다. SLoRA는 기본적으로 Code rank 16/32/64/128/256 전부 실행하고 Conversation rank는 64로 고정한다.

기본 전체 출력은 `/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816`이다. 완료 판정은 `latest_checkpointed_iteration.txt == TRAIN_ITERS`와 `continual_audit/final.json`이 모두 있을 때만 성립하므로, 동일 명령을 다시 실행하면 완료 stage는 건너뛰고 미완료 stage는 자체 checkpoint에서 resume한다.
