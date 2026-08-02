# 백본별 베이스라인 실행기

두 모델에서 같은 방법명과 결과 구조를 사용하되, 실제 파일은 백본별
폴더 아래 방법별로 분리한다.

- `scripts/baselines/llama31/<method>.sh`
- `scripts/baselines/qwen25_7b/<method>.sh`

상위 `llama31.sh`, `qwen25_7b.sh`는 이전 명령 호환용이다.

```bash
cd LLM-continual-learning/trace

# 전체 방법과 상태
./scripts/baselines/llama31/list.sh
./scripts/baselines/qwen25_7b/list.sh

# 학습을 시작하지 않는 명령 검증
./scripts/baselines/llama31/seq_lora.sh validate
./scripts/baselines/qwen25_7b/loramoe.sh validate

# 방법 하나 학습
./scripts/baselines/llama31/slora_pre.sh train
./scripts/baselines/qwen25_7b/gem_corrected.sh train
./scripts/baselines/llama31/loramoe.sh train

# 학습·평가·결과 수집
./scripts/baselines/qwen25_7b/olora_corrected.sh all

# 실행 가능한 전체 목록을 순차 검증 또는 실행
./scripts/baselines/llama31/suite.sh validate
./scripts/baselines/qwen25_7b/suite.sh all
```

## 이름 구분

- `slora_pre_released`: 저자가 공개한 코드 동작을 보존한 포트다.
- `slora_pre`: 수식 기준 감사에서 수정한 SLoRA-Pre다.
- `gem_upstream`, `olora_upstream`: 공개 구현 결함을 그대로 보존한 비교군이다.
- `gem_corrected`, `olora_corrected`: 감사 후 수정한 비교군이다.
Ours v1/v2 wrapper는 공개 SLoRA와 동일한 chat template/full-sequence
labels/right padding 및 AdamW betas를 사용한다. 이는 단순 raw prompt+answer
answer-only 학습과 다른 설정이다. 평가도 공개 SLoRA의 모델별
conversation template, task suffix, greedy/1024-token generation을 사용한다.

- `ours_lora_moe_v1`: new-task phase에서는 신규 expert와 신규 router row만
  학습하고 기존 row를 고정한다. 이후 current를 포함한 seen-task capped
  fixed-subset pool로 전체 router row를 finetune하는 naive 2-phase 성장형
  LoRA-MoE다.
- `ours_lora_moe_v2`: past-only KD-init 뒤, replay가 배정된 microstep에서
  `new(new expert+router) + past(router-only)` gradient를 합쳐 update한다.
  두 방법 모두 `OURS_REPLAY_SUBSET_RATIO=0.01`로 task당 50개를 보관한다.
  v1 router retune은 current-inclusive 1,000 exposures, v2 KD와 joint router
  replay는 동일한 past-only deterministic stream을 각각 정확히 1,000 exposures
  사용한다. 예산은 `OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES`로 조정한다.
  task별/global 연산량과 학습 시간은 run 폴더의 `training_workload.json`에 저장된다.
- `loramoe`: `implementations/llmcl_benchmark`의 FFN
  LoRA-MoE를 두 paper backbone과 TRACE-5000에 연결한 로컬 호환 포트다.
  SLoRA 저자가 공개한 통합 실행 코드가 아니므로 결과를 “공식 SLoRA
  LoRAMoE 재현”이라고 표기하지 않는다.

`gem`과 `olora`는 어느 구현을 뜻하는지 모호하므로 실행 별칭으로 허용하지
않는다. 반드시 upstream 또는 corrected를 지정한다.

`sd_lora`, `rcl`, `unified_olora`는 논문 비교 목록에는 남겨 두지만 공개 실행
코드가 없으므로 fail-fast 처리한다. 가짜 placeholder 학습은 실행하지 않는다.

## LoRAMoE 기본 비교 설정

LoRAMoE는 task당 5,000개, task 순서
`C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm →
NumGLUE-ds → 20Minuten`, epoch `5,3,7,5,3,5,5,7`, seed 2025를 사용한다.
SLoRA 비교 프로필에 맞춰 rank 64, alpha 128, learning rate 2e-4,
4 GPU × micro batch 2 × accumulation 8을 기본값으로 둔다. LoRAMoE 구조상
LoRA는 FFN gate/up/down projection에 적용된다.

환경변수로 덮어쓸 수 있다.

```bash
LORAMOE_GPUS=0,1 LORAMOE_MICRO_BATCH=4 LORAMOE_GRAD_ACCUM=8 \
  ./scripts/baselines/llama31/loramoe.sh train
```

LoRAMoE는 기본적으로 `trace/.venv-runtime`을 사용한다. 다른 Python을 쓰려면
`LORAMOE_PYTHON` 또는 `TRACE_PYTHON`을 지정한다.

## 공통 랜덤 50개와 Llama 사전 토큰 캐시

Ours V1/V2는 `manifests/replay/trace_seed2025_random50_per_task.json`에 저장된 task별 랜덤 비복원 50개를 공통으로 사용한다. Llama-3.1 Ours V1/V2와 SeqLoRA 학습은 `cache/tokenized/llama31_8b/slora_chat_full_len1024/`의 전체 40,000개 사전 토큰 캐시를 기본 사용한다. 캐시는 다음 명령으로 검증 또는 생성한다.

```bash
./scripts/data/prepare_llama31_trace_cache.sh
```

batch padding은 실행 시 동적으로 적용되므로 microbatch 변경 때문에 토큰 캐시를 다시 만들 필요는 없다. Qwen은 replay index만 공유하고 토큰 캐시는 별도로 만들어야 한다.
