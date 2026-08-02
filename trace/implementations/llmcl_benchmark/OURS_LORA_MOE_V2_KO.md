# Ours LoRA-MoE v1/v2: SLoRA 동일 설정 비교

## 비교 원칙

SLoRA와 Ours에 다음 공통 설정을 사용한다.

- backbone: Llama-3.1-8B-Instruct 또는 Qwen2.5-7B-Instruct
- TRACE-5000, task 순서와 epoch: `5,3,7,5,3,5,5,7`
- 4 GPU, micro batch 8, gradient accumulation 2 (global effective batch 64)
- BF16, rank 64, alpha 128, LoRA dropout 0.05
- learning rate `2e-4`, cosine, warmup ratio 0.03, weight decay 0
- combined train sequence cutoff 1024, global gradient checkpointing
- seed 2025
- SLoRA와 동일한 system/user/assistant chat template, right padding,
  1024-token cutoff, 모든 non-padding token에 대한 causal-LM label
- AdamW `betas=(0.9, 0.999)`, epsilon `1e-8`

실제 로컬 Llama-3.1/Qwen2.5 tokenizer로 공개 SLoRA의 TRL 0.16.1 처리 순서와
Ours의 token IDs/labels parity를 확인했다. 이 공통 profile은
`--train_format slora_chat_full`로 고정되며 checkpoint metadata와 `run.env`에
기록된다. Llama는 training pad `128004`, Qwen은 원래 pad `151643`을
보존하며 둘 다 right truncation을 사용한다.

평가도 공개 SLoRA 코드에 맞춘다. 학습용 HF template과 달리 공개 평가용
Llama3 template에는 날짜 header가 없으므로 자체 문자열을 그대로 재현한다.
Qwen template과 Py150/NumGLUE suffix도 원본과 문자열 parity를 확인했다.
평가는 greedy, beam 1, prompt 무강제절단, uniform max-new-tokens 1024이며
stop token도 공개 코드처럼 Llama는 EOS+EOT, Qwen은 EOS로 고정한다. Ours
wrapper의 평가 batch 4는 처리량을 위한 차이이다.

Ours 구조상 FFN gate/up/down LoRA expert와 token router를 사용한다. SLoRA의
attention LoRA와 Ours의 expert/router 구조 차이는 방법론 자체의 차이다.

## 공통 fixed replay memory

각 5,000-sample task에서 한 번만 deterministic fixed subset을 만들고 계속
보존한다.

- `--replay_subset_ratio 0.01`: task당 고유 50 samples
- `--replay_subset_ratio 0.1`: task당 고유 500 samples
- `--router_replay_exposure_samples 1000`: round당 정확히 1,000 global exposures
- `--replay_distribution equal_task`: 1,000 budget을 memory scope의 task에 균등 분배

Task가 늘어나면 task별 subset은 누적되지만 각 phase의 총 exposure는 항상
1,000으로 유지된다. seen task가 4개인 v1은 current를 포함해 각 250 exposure,
past task가 3개인 v2는 각 334/333/333 exposure를 배정한다. 1% subset의 고유
50개가 배정량보다 작으면 deterministic하게 반복한다. 이 예산은 new-task epoch
수와 무관하다.

각 task subset index hash는 `fixed_replay_memory/`, round별 실제 배분은
`replay_plans/`에 기록된다.

## v1

1. new data로 신규 expert와 이번에 추가된 신규 router row만 task epoch만큼 학습한다. 기존 router row는 고정한다.
2. 모든 expert를 freeze하고 전체 router row를 trainable로 연다.
3. current task를 포함한 모든 seen-task fixed subsets에서 정확히 1,000-exposure
   stream을 만들고 한 번 소비하며 router만 finetune한다. 첫 task도 50개를
   20회 노출한 것과 같은 1,000 budget으로 2단계를 수행한다.

seen task의 full train data를 누적하거나 new-task epoch 수에 비례해 replay를
늘리지 않는다.

## v2 KD-init + joint update

Task가 시작되면 current task fixed subset을 생성해 다음 task부터 past memory로
사용한다. Task 2부터 KD와 joint replay는 동일한 past-task scope, 동일 subset
indices, 동일 task별 allocation과 순서를 갖는 deterministic 1,000-record
stream을 각각 한 번 사용한다.

KD-init은 확장 직전 expert/router prefix를 frozen teacher로 사용한다.
output-logit KL로 신규 expert와 신규 router row만 업데이트하며 기존 expert와
기존 router row는 고정한다. 별도의 KD epoch multiplier는 없으며 총 KD exposure는
정확히 1,000이다.

Joint 학습에서는 1,000-record replay stream을 전체 new-task epoch의 primary
microsteps에 균등하게 배치한다. replay가 배정된 microstep은 다음과 같다.

1. `new data`: 신규 expert + router gradient
2. `past replay`: 모든 expert를 freeze하고 router gradient만 추가
3. 두 gradient를 같은 buffer에 합산
4. accumulation 경계에서 optimizer update 1회

replay가 없는 primary microstep은 new-data gradient만 계산한다. KD loss와 joint
replay loss coefficient의 기본값은 각각 1이며, 이후 loss-scale ablation은
`OURS_V2_KD_LOSS_COEFF`와 `OURS_V2_REPLAY_LOSS_COEFF`로 분리한다.

## 실행

```bash
cd /home/work/Agent_HJ/30_flame_agent/slora_repro

# 학습하지 않는 명령 검증
./scripts/baselines/llama31/ours_lora_moe_v1.sh validate
./scripts/baselines/llama31/ours_lora_moe_v2.sh validate

# task당 1% fixed subset, phase당 정확히 1,000 global exposures
OURS_REPLAY_SUBSET_RATIO=0.01 \
OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000 \
./scripts/baselines/llama31/ours_lora_moe_v1.sh train

OURS_REPLAY_SUBSET_RATIO=0.01 \
OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000 \
OURS_V2_KD_LOSS_COEFF=1 \
OURS_V2_REPLAY_LOSS_COEFF=1 \
./scripts/baselines/llama31/ours_lora_moe_v2.sh train
```

학습 산출물은 기본적으로
`slora_repro/results/full_runs/<model>/ours_lora_moe_v1|v2`에 저장된다.
각 task 및 전체 run의 sample/token exposure, forward/backward, optimizer update,
operator FLOPs, 학습/저장 시간은 `training_workload.json`에 누적된다. 기존
full-replay 또는 5:1-token checkpoint는 새 exact-budget run과 resume 호환되지
않는다.
