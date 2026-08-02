# SLoRA 재현 및 신규 비교 실험 기록

이 문서는 `slora_repro`에서 수행하는 SLoRA 재현, 감사된 대조군, Ours LoRA-MoE 비교 실험의 단일 인덱스다. 구현이나 실행 조건이 바뀌거나 실험이 시작·중단·완료될 때마다 갱신한다.

- 마지막 갱신: 2026-07-30 UTC
- 현재 결론: Llama-3.1-8B-Instruct의 Ours v2.5는 8-task 학습과 sparse-15 평가를 완료했지만, v2보다 AA와 forgetting이 모두 악화되어 실패 실험으로 판정한다. 두 backbone 전체 방법의 학습·평가는 아직 완료되지 않았다. partial checkpoint를 논문 수치 재현으로 표기하지 않는다.
- 논문: SLoRA: Balancing Plasticity and Forgetting in Large Language Models for Continual Learning, ACL 2026

---

## 0. 문서 운영 규칙

### 0.1 언제 갱신하는가

다음 중 하나가 발생하면 이 문서를 같은 작업에서 갱신한다.

1. 구현, 수식, 데이터 범위, 하이퍼파라미터 또는 평가 protocol을 변경했다.
2. 새 실험을 시작하거나 resume했다.
3. task 하나 또는 전체 chain이 완료됐다.
4. OOM, NaN, 연결 종료, checkpoint 손상 등으로 실험이 중단됐다.
5. 평가 결과, final average, AFR/BWT 또는 비교표가 생성됐다.

### 0.2 상태 이름

| 상태 | 의미 |
|---|---|
| `planned` | 실행 조건과 경로만 정해짐 |
| `validated` | dry-run/preflight/test는 통과했지만 학습은 시작하지 않음 |
| `running` | 프로세스가 현재 실행 중 |
| `partial` | 일부 task/checkpoint만 완료 |
| `interrupted` | 오류나 사용자 중단으로 멈춤 |
| `trained` | 전체 task 학습 완료, 평가 전 |
| `evaluated` | 전체 평가와 정규화 결과 생성 완료 |
| `reproduced-candidate` | 반복 실행과 논문 비교까지 완료한 후보; 근거를 함께 기록 |

### 0.3 진실의 우선순위

값이 충돌하면 다음 순서로 판단한다.

1. 실제 실행 폴더의 `run.env`, `train.command.txt`, `*.contract.json`
2. 해당 실행의 checkpoint metadata와 log
3. 현재 runner script
4. 이 문서
5. `config/*.json`의 과거 후보 profile과 오래된 보고서

`config/trace_experiments.json`에는 공개 코드와 과거 batch 후보도 남아 있다. 현재 실제 기본값은 반드시 runner의 dry-run으로 다시 확인한다.

---

## 1. 현재 실험 목표

1. 공개 SLoRA 코드를 두 paper backbone과 TRACE-5000에서 가능한 범위까지 재현한다.
2. 공개 코드 동작 보존본과 수식 감사 후 수정본을 분리해 보고한다.
3. GEM과 O-LoRA는 upstream 동작과 corrected 동작을 별도 방법으로 유지한다.
4. Ours LoRA-MoE v1/v2를 SLoRA와 같은 데이터, backbone, task order, epoch, optimizer 및 입력/평가 protocol에 놓고 성능을 비교한다.
5. 공개 실행 코드가 없는 방법은 가짜 구현으로 채우지 않는다.

주 backbone:

- `models/Llama-3.1-8B-Instruct`
- `models/Qwen2.5-7B-Instruct`

TRACE 데이터:

- `/home/work/Agent_HJ/30_flame_agent/TRACE/data/LLM-CL-Benchmark_5000/_extract/TRACE-Benchmark/LLM-CL-Benchmark_5000`
- 각 task train split: 5,000 records

---

## 2. 기준 실험 계약

### 2.1 task 순서와 epoch

| 순번 | task | epoch |
|---:|---|---:|
| 1 | C-STANCE | 5 |
| 2 | FOMC | 3 |
| 3 | MeetingBank | 7 |
| 4 | Py150 | 5 |
| 5 | ScienceQA | 3 |
| 6 | NumGLUE-cm | 5 |
| 7 | NumGLUE-ds | 5 |
| 8 | 20Minuten | 7 |

### 2.2 SLoRA와 Ours v1/v2 공통 profile

| 항목 | 현재 기본값 |
|---|---|
| GPU | A100 4장 |
| precision | BF16 |
| micro batch/device | 8 |
| gradient accumulation | 2 |
| effective global batch | 64 |
| train cutoff | 1,024 tokens |
| LoRA rank / alpha | 64 / 128 |
| LoRA dropout | 0.05 |
| optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8` |
| LR / WD | `2e-4` / `0` |
| scheduler | cosine, warmup ratio `0.03` |
| seed | 2025 |
| checkpointing | 모든 task |

Ours의 `--train_format slora_chat_full`은 공개 SLoRA의 TRL 0.16.1 경로와 동일하게 system/user/assistant chat template을 적용하고 모든 non-padding token에 causal-LM label을 둔다. Llama training pad ID는 128004, Qwen의 원래 pad ID는 151643이며 right padding/right truncation을 사용한다. 두 실제 tokenizer에서 token ID와 label parity를 검증했다. 공개 SLoRA 명령은 dropout을 명시하지 않지만 실제로 거치는 TRL `ModelConfig`의 기본 `lora_dropout`이 0.05이므로 이후 3-run의 SeqLoRA/Ours V1/Ours V2도 모두 0.05를 명시한다. bare PEFT `LoraConfig`의 0.0 기본값과 혼동하지 않는다.

평가는 공개 SLoRA의 자체 conversation template을 사용한다. Llama 평가 template에는 학습용 HF template의 날짜 header가 없다는 공개 코드 동작까지 보존한다. Py150/NumGLUE suffix, greedy, beam 1, prompt 무강제절단, max-new-tokens 1024, Llama `EOS+EOT`/Qwen `EOS` stop 조건을 맞췄다. Ours 평가 batch 4는 공개 runner의 batch 1과 다른 처리량 설정이다.

### 2.3 방법론 자체의 차이

- SLoRA: attention과 FFN의 7개 projection LoRA를 순차 학습하고 SVD 기반 denoising을 적용한다.
- Ours: FFN `gate/up/down`에 성장형 LoRA expert와 token router를 추가한다.
- 따라서 target module, router, expert, KD/재학습 loss는 통제해야 할 하이퍼파라미터가 아니라 방법론 자체의 차이로 기록한다.

### 2.4 아직 완전히 harmonize되지 않은 runner

`loramoe` standalone port는 현재 MB 2/accumulation 8, dropout 0, 일부 task checkpointing과 기존 raw answer-only collator를 사용한다. global batch는 64지만 Ours/SLoRA의 최신 strict profile과 완전히 같지는 않다. 공식 비교표에 넣기 전에 `_run_loramoe.sh`를 별도 감사하거나 차이를 결과표에 명시한다. TRACE EWC/LwF/GEM/O-LoRA도 각 공개 구현의 알고리즘과 데이터 경로를 보존하므로 method별 `run.env`를 기준으로 비교한다.

---

## 3. 저장소 구조

```text
slora_repro/
├── EXPERIMENTS.md                 # 이 문서: 신규 실험의 단일 인덱스
├── upstream/                      # 수정하지 않는 pinned 공개 snapshot
├── implementations/
│   ├── SLoRA-repro/               # 수식 감사·수정 SLoRA
│   ├── SLoRA-upstream-port/       # 공개 동작 보존 + 실행 호환 port
│   ├── TRACE-repro/               # EWC/LwF/GEM/O-LoRA 현대 backbone port
│   └── O-LoRA-repro/              # 원본 참고용; TRACE 비교 runner와 별개
├── scripts/
│   ├── baselines/                 # backbone별 method wrapper와 registry
│   ├── run_experiment.sh          # SLoRA/TRACE 단일 방법 router
│   ├── run_suite.sh               # legacy suite
│   ├── preflight.py               # model/data/runtime 검사
│   ├── collect_results.py         # 평가 결과 정규화
│   └── compare_results.py         # final average/AFR 비교
├── config/                        # model registry, paper fact, run profile
├── docs/                          # 실행 가이드와 결과 schema
├── reports/                       # 구현/수식/batch 감사 보고서
├── models/                        # 로컬 backbone
└── results/
    ├── full_runs/                 # corrected/TRACE/Ours/port 실험
    ├── full_runs_upstream_code/   # 공개 SLoRA 동작 보존 실험
    ├── memory_smoke/              # batch 측정
    └── summaries/                 # 정규화 결과와 비교표
```

Ours 구현 본체는 저장소 밖의 `/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark`에 있고, `scripts/baselines/_run_ours_lora_moe.sh`가 이 구현을 호출한다.

---

## 4. 방법별 구현 및 실행 위치

| 방법 | 상태 | 구현 | backbone별 실행 파일 | 결과 기본 위치 |
|---|---|---|---|---|
| `seq_lora` | runnable | `implementations/SLoRA-repro` | `<model>/seq_lora.sh` | `full_runs/<model>/seq` |
| `slora_pre_released` | runnable | `implementations/SLoRA-upstream-port` | `<model>/slora_pre_released.sh` | `full_runs_upstream_code/<model>/pre` |
| `slora_pre` | runnable | `implementations/SLoRA-repro` | `<model>/slora_pre.sh` | `full_runs/<model>/pre` |
| `slora_post` | runnable | `implementations/SLoRA-repro` | `<model>/slora_post.sh` | `full_runs/<model>/post` |
| `ewc` | runnable | `implementations/TRACE-repro` | `<model>/ewc.sh` | `full_runs/<model>/ewc_upstream` |
| `lwf` | runnable | `implementations/TRACE-repro` | `<model>/lwf.sh` | `full_runs/<model>/lwf_upstream` |
| `gem_upstream` | audit | `implementations/TRACE-repro` | `<model>/gem_upstream.sh` | `full_runs/<model>/gem_upstream` |
| `gem_corrected` | runnable | `implementations/TRACE-repro` | `<model>/gem_corrected.sh` | `full_runs/<model>/gem_corrected` |
| `olora_upstream` | audit | `implementations/TRACE-repro` | `<model>/olora_upstream.sh` | `full_runs/<model>/olora_upstream` |
| `olora_corrected` | runnable | `implementations/TRACE-repro` | `<model>/olora_corrected.sh` | `full_runs/<model>/olora_corrected` |
| `ours_lora_moe_v1` | runnable | `../llmcl_benchmark` | `<model>/ours_lora_moe_v1.sh` | `full_runs/<model>/ours_lora_moe_v1` |
| `ours_lora_moe_v2` | runnable | `../llmcl_benchmark` | `<model>/ours_lora_moe_v2.sh` | `full_runs/<model>/ours_lora_moe_v2` |
| `loramoe` | local port | `../llmcl_benchmark` | `<model>/loramoe.sh` | `full_runs/<model>/loramoe` |
| `sd_lora` | unavailable | 공개 runner 없음 | fail-fast wrapper만 존재 | 없음 |
| `rcl` | unavailable | 공개 runner 없음 | fail-fast wrapper만 존재 | 없음 |
| `unified_olora` | unavailable | SLoRA 저자 runner 미공개 | fail-fast wrapper만 존재 | 없음 |

`<model>`은 `scripts/baselines/llama31` 또는 `scripts/baselines/qwen25_7b`다. `gem`, `olora` 같은 모호한 별칭은 금지하고 반드시 upstream/corrected를 명시한다.

`slora_post`는 별도 학습을 반복하지 않는다. `seq_lora`의 raw continual adapter를 재사용하고 평가 시 denoising을 적용한다.

---

## 5. 원본 대비 수정·추가 사항

### 5.1 원본 보존 정책

- `upstream/SLoRA`, `upstream/TRACE`, `upstream/O-LoRA`는 pinned snapshot이다.
- 실제 실행 수정은 `implementations/*-repro`에만 둔다.
- `SLoRA-upstream-port`는 완전히 손대지 않은 snapshot이 아니라, 현대 backbone/TRACE에서 실행되도록 portability를 추가하면서 공개 알고리즘 동작을 보존한 비교 경로다.
- corrected 결과와 upstream-behavior 결과를 같은 이름이나 폴더로 저장하지 않는다.

### 5.2 SLoRA

주요 파일:

- `implementations/SLoRA-repro/src/train/cl_train.py`
- `implementations/SLoRA-repro/src/train/cl_train_slora.py`
- `implementations/SLoRA-repro/src/model/builder.py`
- `implementations/SLoRA-repro/src/trace_data.py`
- `implementations/SLoRA-repro/src/eval/model_diverse_gen_batch.py`
- `implementations/SLoRA-repro/scripts/repro/{train_trace,eval_trace}.sh`

실행 호환 및 수정 내용:

- TRACE `prompt`/`answer` schema를 chat messages로 변환한다.
- Qwen2.5와 충돌하는 `/no_think` prefix를 제거했다.
- `SFTTrainer`가 새 adapter를 한 번만 추가하도록 double PEFT wrapping을 제거했다.
- Llama-3.1/Qwen2.5 tokenizer, pad, local model loading을 지원한다.
- DeepSpeed ZeRO-2 config와 task별 contract/log/checkpoint 경로를 추가했다.
- SLoRA denoising reference는 이전 adapter가 섞인 model이 아니라 고정 pretrained `theta_0`의 7개 target projection snapshot을 사용한다.
- candidate/reference subspace를 선택된 candidate rank에서 비교한다.
- 선택한 `U_c`로 복원하고 rank 축소 뒤에도 원래 LoRA `alpha/r` scaling을 보존한다.
- Post는 Seq-LoRA checkpoint를 재사용하도록 실행 chain을 분리했다.

공개 동작 보존본은 `SLoRA-upstream-port`, 수식 감사 수정본은 `SLoRA-repro`다.

### 5.3 TRACE 대조군

주요 파일:

- `implementations/TRACE-repro/model/Regular/{EWC,LwF,GEM,O_LoRA}.py`
- `implementations/TRACE-repro/training/main.py`
- `implementations/TRACE-repro/utils/data/data_collator.py`
- `implementations/TRACE-repro/utils/model/model_utils.py`
- `implementations/TRACE-repro/scripts/repro/{train_trace,eval_trace}.sh`

공통 port:

- Llama-3.1/Qwen2.5에 `AutoTokenizer`를 사용한다.
- Qwen에 `None` BOS를 삽입하지 않는다.
- legacy Llama FlashAttention monkey patch를 제거했다.
- local backbone을 BF16으로 로드하고 vocabulary 크기를 임의 변경하지 않는다.
- cosine scheduler와 3% warmup을 실제 trainer에 전달한다.
- generation pad ID와 20Minuten SARI 직렬화를 수정했다.

GEM:

- `gem_upstream`: 공개 qpth 부호/constraint translation을 보존한다.
- `gem_corrected`: 감사한 dual 부호와 제약 변환을 사용한다.
- 두 결과는 최적화 자체가 달라지므로 함께 남긴다.

O-LoRA:

- `olora_upstream`: 공개 TRACE의 A/L1 overlap penalty를 사용한다.
- `olora_corrected`: update-column B matrix와 squared Frobenius overlap을 사용한다.
- 이전 adapter merge를 q/v 고정 경로에서 7개 target projection으로 일반화했다.
- SLoRA 저자의 unified O-LoRA 재현이 아니라 TRACE 기반 통제 실험이다.

### 5.4 Ours LoRA-MoE

구현 파일:

- `../llmcl_benchmark/model/Ours_LoRA_MoE.py`
- `../llmcl_benchmark/training/main_Ours_LoRA_MoE.py`
- `../llmcl_benchmark/evaluate_Ours_LoRA_MoE.py`
- `../llmcl_benchmark/utils/data/data_collator.py`
- `../llmcl_benchmark/utils/eval_generation.py`
- `../llmcl_benchmark/scripts/test_ours_lora_moe_v2.py`
- `../llmcl_benchmark/scripts/test_lora_moe_sparse.py`

공통 구조:

- frozen dense backbone FFN의 `gate/up/down`에 성장형 LoRA expert를 붙인다.
- task마다 기본 1 expert를 추가하고 full-softmax top-1 token router를 확장한다.
- dense backbone, attention, 이전 expert는 freeze한다.
- task별 deterministic fixed subset을 한 번 만들고 index/hash와 replay allocation plan을 저장한다.
- task 수와 new-task epoch 수가 늘어도 router replay는 round당 정확히 1,000 global sample exposures로 고정한다.
- 작은 subset은 deterministic하게 반복해 exposure budget을 채운다.
- sparse dispatch, padding 제외 router loss, checkpoint metadata/resume validation, optimizer-update 기준 scheduler step을 구현했다.
- SLoRA training/evaluation tokenizer와 label/generation protocol을 별도 compatibility path로 추가했다.

v1:

1. 현재 task 전체 데이터로 신규 expert와 이번에 추가된 신규 router row만 학습하며, 기존 router row는 고정한다.
2. 모든 expert를 freeze한다.
3. 현재 task를 포함한 모든 seen-task fixed subset pool로 전체 router row를 별도 finetune한다.
4. 첫 task도 phase 2를 수행한다.

v2:

1. 확장 직전 expert/router prefix를 frozen teacher로 사용해 신규 expert를 output-logit KD로 초기화한다.
2. KD는 past-task 1% pool에서 task별 균등 배분한 deterministic 1,000-sample stream을 정확히 한 번 사용한다.
3. joint replay도 KD와 동일한 1,000-sample stream을 전체 new-task epoch에 균등 분산한다.
4. replay가 배정된 microstep에서는 `new(new expert+router)`와 `past(router-only)` gradient를 같은 buffer에 더하고 accumulation 경계에서 한 번 update한다. 기본 KD/replay loss coefficient는 각각 1이다.

v1은 current-inclusive seen pool이고 v2의 joint replay/KD pool은 past-only다. 둘을 혼동하지 않는다.

### 5.5 실행·감사 인프라

추가된 공통 기능:

- 모델 shard/index/tensor byte 검증
- TRACE 5,000-record schema/count contract
- 격리 `.venv-runtime`과 별도 `envs/train_env`
- real BF16 CUDA forward smoke
- MeetingBank 기준 batch benchmark
- backbone별 method registry와 fail-fast unavailable entry
- 실행별 command, environment, contract, log 보존
- 8×8 continual matrix 정규화와 final average/AFR 비교
- Ours sparse routing, router gradient, memory budget, joint two-backward invariant CPU tests

---

## 6. 실행 스크립트

모든 명령은 다음 위치에서 실행한다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/slora_repro
```

### 6.1 목록과 dry-run

```bash
./scripts/baselines/llama31/list.sh
./scripts/baselines/qwen25_7b/list.sh

./scripts/baselines/llama31/slora_pre.sh validate
./scripts/baselines/llama31/ours_lora_moe_v1.sh validate
./scripts/baselines/qwen25_7b/ours_lora_moe_v2.sh validate
```

`validate`는 학습을 시작하지 않는다.

### 6.2 train/eval/all

```bash
# 학습만
./scripts/baselines/llama31/slora_pre.sh train

# 평가만
./scripts/baselines/llama31/slora_pre.sh eval

# 학습 → 평가 → 결과 수집
./scripts/baselines/qwen25_7b/gem_corrected.sh all
./scripts/baselines/llama31/ours_lora_moe_v1.sh all
```

내부 routing:

- SLoRA/TRACE: `scripts/run_experiment.sh`
- standalone LoRAMoE: `scripts/baselines/_run_loramoe.sh`
- Ours v1/v2: `scripts/baselines/_run_ours_lora_moe.sh`
- backbone/method 선택: `scripts/baselines/_run_model.sh`

### 6.3 preflight와 테스트

```bash
python scripts/preflight.py --mode full \
  --models llama31_8b_instruct qwen25_7b_instruct

PYTHONPATH=../llmcl_benchmark \
  python ../llmcl_benchmark/scripts/test_ours_lora_moe_v2.py
PYTHONPATH=../llmcl_benchmark \
  python ../llmcl_benchmark/scripts/test_lora_moe_sparse.py
PYTHONPATH=../llmcl_benchmark \
  python ../llmcl_benchmark/scripts/test_paper_baselines.py
```

### 6.4 Ours 조절값

```bash
OURS_REPLAY_SUBSET_RATIO=0.01 \
OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000 \
./scripts/baselines/llama31/ours_lora_moe_v1.sh train

OURS_REPLAY_SUBSET_RATIO=0.01 \
OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000 \
OURS_V2_KD_LOSS_COEFF=1 \
OURS_V2_REPLAY_LOSS_COEFF=1 \
./scripts/baselines/llama31/ours_lora_moe_v2.sh train
```

주요 environment override는 해당 runner 상단을 기준으로 한다. 값을 바꾼 실험은 명령과 `run.env`를 레지스트리에 반드시 기록한다.

---

## 7. 산출물 구조와 완료 판정

### 7.1 SLoRA/TRACE 계열

```text
results/full_runs/<model>/<method-dir>/
├── order1 ... order8/             # task별 checkpoint
├── orderN.contract.json           # 데이터/배치/epoch 계약
├── orderN.command.txt
├── orderN.train.log
└── evaluation/orderN/<task>/      # triangular evaluation
```

### 7.2 Ours/standalone LoRAMoE

```text
results/full_runs/<model>/<method>/
├── run.env
├── train.command.txt
├── train.log
├── 0 ... 7/                       # task round checkpoint
├── fixed_replay_memory/           # task별 immutable subset metadata
├── replay_plans/                  # round/phase별 exposure allocation
├── training_workload.json         # task별/global 연산량·학습/저장 시간
└── evaluation/order1 ... order8/
```

### 7.3 정규화 결과

- `results/summaries/<model>/<method>.json`
- 비교 결과: `results/summaries/<model>/comparison*.json`
- schema: `docs/result_schema.md`

전체 완료로 표시하려면 다음이 모두 있어야 한다.

1. 8개 task checkpoint
2. 각 round에서 그때까지 본 task의 triangular evaluation
3. normalized 8×8 matrix
4. final average와 AFR 또는 명시한 CL metric
5. 실행 command/profile/log
6. task별/global workload 및 wall-clock 기록
7. NaN, 누락 task, 손상 checkpoint가 없다는 검사

---

## 8. 현재 상태 스냅샷

2026-07-26 UTC 기준:

| 항목 | 상태 | 근거/비고 |
|---|---|---|
| Llama-3.1 model | validated | 4/4 shard와 indexed tensor bytes 검증 |
| Qwen2.5 model | validated | 4/4 shard와 indexed tensor bytes 검증 |
| TRACE-5000 | validated | 8 task × train 5,000 schema/count 확인 |
| runtime | validated | `.venv-runtime`, torch/TRL/PEFT/DeepSpeed 검사 |
| CUDA forward | validated | 두 backbone BF16 finite logits |
| batch smoke | evaluated | Llama/Qwen MeetingBank MB 8/GA 2 통과 |
| `slora_pre_released` Llama | interrupted | order1–3 final checkpoint 존재, order4는 중간 checkpoint만 존재 |
| `slora_pre` Llama | partial | order1 checkpoint 존재; 전체 chain/evaluation 없음 |
| SLoRA Qwen full chain | planned | contract/빈 evaluation directory는 완료 결과가 아님 |
| TRACE baseline full chains | planned | 현재 run.env/data cache만 있는 폴더는 완료 결과가 아님 |
| Ours v1/v2 | validated | 구현·CPU test·Llama/Qwen dry-run 완료, 실제 학습 미시작 |
| full reproduced score | 없음 | 어떤 방법도 8-task train+eval 전체 완료로 판정하지 않음 |

기존 partial checkpoint를 resume하거나 새 profile로 다시 시작할 때는 checkpoint가 현재 tokenizer, batch, denoising anchor, replay schema와 호환되는지 먼저 확인한다. profile이 달라졌다면 같은 결과 폴더를 덮어쓰지 않는다.

---

## 9. 실험 레지스트리

새 run을 시작하기 전에 한 행을 추가하고, 상태가 바뀔 때 갱신한다.

| ID | 날짜(UTC) | model | method | profile/핵심 변경 | 상태 | 결과 경로 | 요약 |
|---|---|---|---|---|---|---|---|
| ENV-001 | 2026-07-26 | both | environment | model/data/runtime/CUDA 검사 | validated | `manifests/preflight.json` | 두 backbone 실행 준비 |
| BATCH-001 | 2026-07-26 | both | SLoRA smoke | MeetingBank MB8/GA2 | evaluated | `results/memory_smoke/` | effective batch 64 |
| SLP-REL-L31-001 | 2026-07-26 | llama31 | slora_pre_released | 공개 동작 port | interrupted | `results/full_runs_upstream_code/llama31/pre` | order1–3 완료, order4 불완전 |
| SLP-COR-L31-001 | 2026-07-26 | llama31 | slora_pre | corrected pre | partial | `results/full_runs/llama31/pre` | order1만 존재 |
| OURS-V1-L31-001 | — | llama31 | ours_lora_moe_v1 | SLoRA strict profile, 1% memory/exact 1000 replay | planned | `results/full_runs/llama31/ours_lora_moe_v1` | 미실행 |
| OURS-V2-L31-001 | — | llama31 | ours_lora_moe_v2 | same 1% past stream, KD1000/replay1000 | planned | `results/full_runs/llama31/ours_lora_moe_v2` | 미실행 |
| OURS-V1-Q25-001 | — | qwen25_7b | ours_lora_moe_v1 | SLoRA strict profile, 1% memory/exact 1000 replay | planned | `results/full_runs/qwen25_7b/ours_lora_moe_v1` | 미실행 |
| OURS-V2-Q25-001 | — | qwen25_7b | ours_lora_moe_v2 | same 1% past stream, KD1000/replay1000 | planned | `results/full_runs/qwen25_7b/ours_lora_moe_v2` | 미실행 |

### 권장 실행 우선순위

1. Llama `slora_pre` C-STANCE 한 task를 현재 corrected 코드로 다시 검증
2. Llama `seq_lora → slora_post`, `slora_pre` 전체 chain
3. Llama Ours v1, v2
4. Qwen에서 동일 순서 반복
5. TRACE EWC/LwF, GEM upstream/corrected, O-LoRA upstream/corrected
6. standalone `loramoe` profile 차이 감사 후 포함 여부 결정

---

## 10. 실행별 상세 로그 템플릿

아래 블록을 복사해 레지스트리 아래 또는 이 절 아래에 추가한다.

````markdown
### <RUN-ID> — <한 줄 목적>

- 시작/종료 UTC:
- 상태:
- model / method:
- 실행 주체와 GPU:
- 코드 기준:
  - repo commit:
  - dirty diff 또는 patch:
- 명령:
  ```bash
  <exact command>
  ```
- 공통 profile에서 바꾼 값:
- 데이터/subset seed:
- output:
- task 진행:
- peak VRAM / 시간:
- checkpoint 검사:
- 평가 결과:
- 오류/경고:
- 해석:
- 다음 조치:
````

실패도 삭제하지 않는다. OOM 위치, 마지막 정상 checkpoint, resume 가능 여부, 바꿀 값과 비교 공정성 영향을 함께 남긴다.

---

## 11. 알려진 주의점과 결정

- `slora_pre_released`는 공개 코드 동작 비교용이고 `slora_pre`는 수식 감사 수정본이다. 둘 중 하나로 다른 하나를 대체하지 않는다.
- GEM/O-LoRA upstream과 corrected를 모두 보고하고 이름을 생략하지 않는다.
- TRACE 기반 O-LoRA를 SLoRA 저자의 unified O-LoRA 재현이라고 부르지 않는다.
- `sd_lora`, `rcl`, `unified_olora`는 공개 runner가 없어 unavailable이다.
- Ours는 SLoRA 저자 코드가 아니라 동일 조건 비교를 위한 로컬 방법이다.
- Ours v1 phase 2는 current-inclusive seen-task memory, v2 replay/KD는 past-only다.
- fixed subset의 unique size와 exposure를 구분한다. 기본 1%는 task당 50 unique records이고, v1 router retune 및 v2 KD/router replay의 round별 예산은 각각 정확히 1,000 global exposures다.
- batch 변경이 global batch를 보존하더라도 padding 구성과 수치 오차가 달라질 수 있으므로 실제 MB/GA를 결과에 기록한다.
- partial checkpoint, smoke test, dry-run 결과를 논문 성능 재현으로 쓰지 않는다.
- model 다운로드 명령과 인증 정보는 이 실험 기록에 넣지 않는다.
- worktree가 dirty하므로 장기 run 전 commit hash와 `git diff` 또는 patch를 실행 폴더에 보존하는 것이 필요하다.

---

## 12. 고정 replay 표본과 Llama 토큰 캐시

2026-07-26에 seed 2025로 각 TRACE-5000 task 전체에서 50개를 무작위 비복원 추출했다. 앞 50개 절단이 아니며, 8개 task의 실제 index 400개를 공통 manifest에 저장했다. V1/V2는 실행 폴더별 재추출 대신 이 동일 manifest를 사용한다.

- replay manifest: `cache/replay_manifests/trace_seed2025_random50_per_task.json`
- Llama-3.1 token cache: `cache/tokenized/llama31_8b/slora_chat_full_len1024/`
- cache metadata: `cache/tokenized/llama31_8b/slora_chat_full_len1024/manifest.json`
- generator: `../llmcl_benchmark/scripts/prepare_trace_token_cache.py`
- wrapper: `scripts/data/prepare_llama31_trace_cache.sh`

Llama cache는 8 task × 5,000 = 40,000개 전체 train record를 `slora_chat_full`, system/user/assistant chat template, max length 1024로 토크나이즈한다. unpadded token ID만 저장하고 batch마다 right padding하므로 microbatch를 바꿔도 재생성하지 않는다. Ours용 memory-mapped Arrow와 `datasets` 3.x/5.x 공용 metadata-free Parquet sidecar를 함께 두며 총 크기는 약 58 MiB다. 최초 token 생성 시간은 50.7초였다.

검증 결과는 다음과 같다.

- task마다 50개 index가 모두 고유하고 범위 안에 있음
- 8개 task에서 replay 400개 전부를 기존 동적 SLoRA collator와 비교
- `input_ids`, `attention_mask`, `labels`가 전부 exact-equal
- 기존 Ours V1/V2 CPU 불변식 테스트 10개 통과
- Llama V1/V2 wrapper가 manifest와 token cache를 기본 인자로 전달
- tokenizer asset hash, format, max length가 다르면 학습 시작 전에 fail-fast
- Llama SeqLoRA dry-run 8개 task 모두 동일 cache 경로를 전달하고 실제 runtime에서 40,000개 load 통과
- Qwen SeqLoRA에는 Llama cache 인자가 전달되지 않음

재검증 또는 캐시가 없는 환경에서의 생성 명령:

```bash
cd /home/work/Agent_HJ/30_flame_agent/slora_repro
./scripts/data/prepare_llama31_trace_cache.sh
```

기존 캐시가 있으면 task별로 동등성을 검사하고 재사용한다. 의도적으로 다시 만들 때만 `--overwrite`를 전달한다. Qwen은 같은 replay manifest를 공유할 수 있지만 token ID는 공유하지 않으므로 별도 tokenizer cache가 필요하다.

---

## 13. Llama-3.1 연속 학습 체인

SeqLoRA, Ours LoRA-MoE V1, Ours LoRA-MoE V2를 다음 순서로 중단 없이 실행하기 위한 체인을 추가했다.

1. `scripts/baselines/llama31/seq_lora.sh`
2. `scripts/baselines/llama31/ours_lora_moe_v1.sh`
3. `scripts/baselines/llama31/ours_lora_moe_v2.sh`

메인 스크립트는 `scripts/chains/run_llama31_seq_ours_v1_v2.sh`다. 한 단계의 명령이나 완료 검사가 실패하면 즉시 종료하며, 실패한 단계 뒤의 방법은 실행하지 않는다. `flock` 기반 단일 실행 잠금을 사용하므로 같은 체인을 실수로 중복 실행하지 않는다.

### 실행 전 확인 및 명령

```bash
cd /home/work/Agent_HJ/30_flame_agent/slora_repro

./scripts/chains/run_llama31_seq_ours_v1_v2.sh status
./scripts/chains/run_llama31_seq_ours_v1_v2.sh plan
./scripts/chains/run_llama31_seq_ours_v1_v2.sh validate
./scripts/chains/run_llama31_seq_ours_v1_v2.sh train
```

- `status`: 각 방법의 완료 task/round 수만 조회한다. 로그 폴더나 `latest_run.txt`를 변경하지 않는다.
- `plan`: 현재 체크포인트를 기준으로 skip, fresh start, resume 결정을 출력한다. 이 명령도 읽기 전용이다.
- `validate`: 세 wrapper의 preflight와 dry-run을 실행하고 실제 학습은 시작하지 않는다.
- `train`: SeqLoRA → V1 → V2 순서로 실제 학습한다.

기본 결과 경로는 각각 다음과 같다.

- SeqLoRA: `results/full_runs/llama31/seq/`
- Ours V1: `results/full_runs/llama31/ours_lora_moe_v1/`
- Ours V2: `results/full_runs/llama31/ours_lora_moe_v2/`
- 체인 상태와 로그: `results/chains/llama31_seq_ours_v1_v2/runs/<UTC run id>/`

각 체인 run에는 `status.log`, 단계별 로그, `chain.pid`를 저장한다. 세 방법이 모두 완료 검사를 통과한 경우에만 `CHAIN_COMPLETE`를 만든다. `results/chains/llama31_seq_ours_v1_v2/latest_run.txt`에서 가장 최근 체인 로그 폴더를 확인할 수 있다.

### 완료 판정과 재개

- SeqLoRA task는 해당 `orderN`에 non-empty `adapter_config.json`과 `adapter_model.safetensors`가 모두 있을 때만 완료로 본다. `SLORA_SKIP_COMPLETED=1`로 완료 task를 건너뛰고 미완료 task부터 계속한다.
- Ours round는 해당 round에 non-empty `lora_moe_meta.json`과 `pytorch_model.bin`이 모두 있을 때만 완료로 본다.
- Ours는 round 0부터 연속해서 완성된 마지막 checkpoint만 resume 대상으로 선택한다. 중간 round가 빠진 상태에서 더 높은 orphan checkpoint가 있어도 사용하지 않는다.
- wrapper 명령이 exit 0이어도 8개 task/round 산출물이 모두 없으면 체인은 완료 실패(exit 20)로 처리한다.

### 2026-07-26 검증 기록

- 실제 cache와 model 경로를 사용한 전체 `validate`가 exit 0으로 끝났으며 세 wrapper 모두 dry-run을 통과했다. 학습은 시작되지 않았다.
- 임시 checkpoint로 SeqLoRA 2/8, V1 연속 round 0–1 및 orphan round 3 상태를 만들었을 때, SeqLoRA는 완료 task skip, V1은 round 1 resume, V2는 fresh start로 판정했다.
- 가짜 SeqLoRA wrapper를 exit 42로 실패시킨 테스트에서 체인은 같은 코드로 종료했고 V1/V2 wrapper는 호출되지 않았다.
- 현재 실제 full-run 상태는 SeqLoRA 0/8, V1 0/8, V2 0/8이다.
- 최종 점검 중 발견한 SeqLoRA skip 검사 변수 순서를 수정하여 미완료/부분 완료 run에서도 `set -u` 오류 없이 재개하도록 했다.

### SLoRA-Pre 종료 후 자동 시작

자동 시작 스크립트는 `scripts/chains/wait_for_slora_pre_then_run_chain.sh`다. adapter 파일만 보고 완료로 판단하지 않고, 시작 시 기록한 `/proc` start ticks로 PID 재사용을 방지하며 SLoRA 부모 PID와 전체 process group이 모두 사라진 후 order1–8 산출물을 검사한다. 사라진 PID나 다른 start ticks로 arm하려 하면 exit 23과 `ARM_REJECTED`로 거부한다.

2026-07-26 첫 검증 중 order8 adapter가 생성된 시점에 검증용 사라진 PID를 사용해 체인이 너무 일찍 호출된 사건이 있었다. 당시 실제 SLoRA process group은 계속 후처리 중이어서 SeqLoRA가 rendezvous port 29500 충돌(`EADDRINUSE`)로 학습 시작 전에 종료됐다. SeqLoRA/V1/V2 checkpoint는 모두 0/8로 유지되며, 해당 감사 기록은 trigger run `20260726T134324Z`에 보존했다.

이후 사라진 PID arm을 차단하고 실제 SLoRA 부모 PID 1490172, start ticks 204794319, PGID 1490172에 다시 연결했다. 일반 `nohup` 자식이 작업 cgroup 종료와 함께 정리되는 환경이므로 detached tmux 세션을 사용한다.

- tmux session: `slora-chain-trigger`
- trigger ID: `armed_tmux_20260726T134737Z`
- trigger status: `results/chains/llama31_seq_ours_v1_v2/trigger/runs/armed_tmux_20260726T134737Z/status.log`
- goal prompt: `SLORA_CHAIN_MONITOR_GOAL_PROMPT.txt`

현재 watcher는 SLoRA 부모와 GPU worker가 살아 있는 동안 `waiting_for_slora` 상태를 유지한다. 실제 체인은 아직 0/8이며 SLoRA process group이 완전히 종료된 뒤에만 validate와 train을 순서대로 호출한다.

---

## 14. 변경 이력

### 2026-07-26

- 신규 실험 관리 문서 생성.
- Llama-3.1 SeqLoRA → Ours V1 → Ours V2 연속 체인과 fail-stop, resume, 단일 실행 잠금, 단계별 로그 검증을 추가.
- 실제 SLoRA PID/start ticks/process group 전체 종료를 기다리는 tmux trigger와 10분 감사·복구용 goal prompt를 추가.
- backbone별 baseline registry, 수정 포트, 결과 폴더를 한 인덱스로 정리.
- Ours v1을 current-inclusive 2-phase로 확정.
- Ours v1/v2의 SLoRA chat/full-label/optimizer/evaluation parity 반영.
- 이후 3-run의 SeqLoRA/Ours V1/Ours V2 LoRA dropout을 공개 SLoRA의 실제 TRL 기본값과 같은 0.05로 명시.
- Ours v1/v2 replay를 epoch 비례/5:1 token 방식에서 exact 1,000 global exposures로 확정하고, v2 KD도 동일한 past-only stream 1,000 exposures로 통일.
- task별/global sample·token·forward/backward·optimizer update·operator FLOPs·학습 시간을 `training_workload.json`에 저장하도록 계측 추가.
- 기존 partial SLoRA checkpoint와 미실행 실험을 구분해 초기 상태 기록.

### 2026-07-27 연속 체인 운영 감사

- SLoRA-Pre order1–8은 모두 완료되었고, 실제 종료 확인 후 trigger가
  `after_slora_armed_tmux_20260726T134737Z` 체인을 시작했다.
- 첫 체인 시도(`after_slora_20260726T134324Z`)는 SLoRA 후처리와 겹쳐
  torchrun rendezvous 포트가 이미 사용 중인 `EADDRINUSE`로 학습 전에 종료됐다.
  checkpoint나 로그를 삭제하지 않고, 원인 확인 후 현재 detached tmux trigger가
  재개하도록 두었다.
- 현재 재개 run은 `results/chains/llama31_seq_ours_v1_v2/runs/after_slora_armed_tmux_20260726T134737Z/`이며,
  SeqLoRA order1–7 산출물이 완성되었고 order8이 4-GPU로 진행 중이다(감사 시점
  약 25/546 step). GPU worker 4개가 살아 있고 실제 메모리를 사용하며,
  traceback/OOM/NCCL timeout/NaN 오류는 확인되지 않았다.
- V1/V2는 SeqLoRA 8개 order 완료 후에만 시작하도록 fail-stop 체인에 의해
  대기 중이다. 다음 감사에서 order8 완료와 V1 round 진행 여부를 확인한다.

### 2026-07-27 30분 감사

- 현재 재개 run은 동일한 `after_slora_armed_tmux_20260726T134737Z`이며,
  SeqLoRA order8 `20Minuten`이 4-GPU로 약 317/546 step(epoch 약 4.01)까지
  정상 진행 중이다. 네 GPU worker가 각각 약 34.8 GiB를 사용하고 있다.
- 로그의 step/epoch, loss·learning-rate 기록과 파일 크기가 증가했고,
  traceback/OOM/NCCL timeout/SIGKILL/NaN 및 체인 실패 마커는 확인되지 않았다.
  SeqLoRA order8 완료 전이므로 V1/V2는 아직 시작하지 않았다.

### 2026-07-27 체인 복구 감사

- SLoRA-Pre와 SeqLoRA order1–8은 정상 완료되었지만, 체인의 V1 첫 task
  `C-STANCE` phase-1에서 GPU OOM으로 종료되었다. 각 GPU가 약 79.2 GiB를
  사용한 상태에서 backward 중 추가 58–60 MiB 할당에 실패했으며, 이는
  hyperparameter/데이터 오류가 아니라 Ours의 `micro_batch=8` 설정이 Llama-3.1
  8B 80-GiB GPU의 여유 공간을 넘은 실행 자원 문제였다.
- 기존 checkpoint·로그·완료 마커는 삭제하지 않았다. 과학적 effective global
  batch 64 (= micro-batch 8 × 4 GPU × accumulation 2)를 보존하기 위해 복구
  재실행에서는 `micro_batch=4`, `gradient_accumulation=4`를 사용한다. 이 값은
  V1/V2 wrapper에 환경변수로만 주입하며 LR, epoch, rank/alpha, replay/KD 등은
  변경하지 않는다. `run.env`와 `train.command.txt`에 실제 값을 남긴다.
- FLOP counter는 유지하여 task별 `training_workload.json`의 operator FLOPs
  계측 요구를 보존한다. 완료된 SeqLoRA는 skip하고 V1은 round 0부터, V2는
  V1 완료 후 fresh/resume 규칙으로 체인을 재개한다.
- `validate`는 `micro_batch=4`, `gradient_accumulation=4` 명령을 확인하고
  exit 0으로 통과했다. 2026-07-27 04:28 UTC에 중복 Ours 프로세스와 잠금을
  확인한 뒤 `slora-chain-recovery` detached tmux에서 체인을 재실행했다. SeqLoRA는
  8/8 skip되었고 V1 fresh start가 시작되었으며, 실행 로그는
  `results/chains/llama31_seq_ours_v1_v2/recovery_tmux.log`와 새 run 폴더에 있다.
### 2026-07-27 30분 감사: Ours V1 재시도 OOM

- 복구 실행(`micro_batch=4`, `gradient_accumulation=4`, effective global batch 64)이 C-STANCE 첫 task의 backward에서 다시 CUDA OOM으로 종료됨.
- 실패 지점은 `Ours_LoRA_MoE.py`의 routed hidden-state indexing 중 training FLOP counter dispatch였고, GPU당 남은 메모리는 약 13 MiB(18 MiB 할당 실패)였음.
- LR, epoch, LoRA rank/alpha/dropout, task order, replay/KD 및 FLOP 측정은 변경하지 않음. effective global batch 64를 보존하기 위해 다음 재개만 `micro_batch=2`, `gradient_accumulation=8`로 조정함.
- 기존 checkpoint/로그/완료 마커는 삭제하지 않았으며, validate 후 detached tmux에서 체인을 재개함.

### 2026-07-27 30분 감사: Ours V1 FLOP 계측 호환성 오류

- `micro_batch=2`, `gradient_accumulation=8` 재시도는 첫 C-STANCE phase-1의
  14번째 microbatch에서 CUDA OOM이 아니라 PyTorch `FlopCounterMode`의 SDPA
  shape assertion(`torch.utils.flop_counter.sdpa_flop_count`)으로 종료됨. 현재
  PyTorch/Transformers SDPA의 가변 시퀀스 길이 조합에서 계측기가 중단되는
  instrumentation 호환성 문제다.
- 과학적 학습 설정과 effective global batch 64는 유지한다. Ours wrapper의
  실행 기본값만 FLOP counter 비활성(`OURS_DISABLE_TRAINING_FLOP_COUNTER=1`)으로
  바꾸어 실제 학습이 중단되지 않게 했다. sample/token/forward/backward/update,
  wall-clock workload는 계속 `training_workload.json`에 기록되고,
  `counted_operator_flops_global`은 계측 비활성 run에서 null로 명시된다.
- 다음 재실행은 이 변경을 검증하고, 다시 OOM이면 동일 global batch를 보존하는
  `micro_batch=1`, `gradient_accumulation=16`으로만 조정한다. 기존 로그·산출물은
  삭제하지 않는다.

### 2026-07-27 30분 감사: Ours V1 재개 진행 중

- `slora-chain-recovery` detached tmux에서 중복 없이 Ours V1이 실행 중이다.
  현재 FOMC phase-1 epoch 1, 약 step 840/3750(약 23%)까지 진행되었고 로그
  mtime/크기와 GPU 사용량이 증가했다(4개 worker, GPU당 약 18.8 GiB, 16–17% util).
- SeqLoRA 산출물은 8/8 완료 상태이며, Ours V1 round0의
  `lora_moe_meta.json`과 `pytorch_model.bin`이 생성되었다. V2는 V1 전체 완료
  후 시작하도록 대기 중이다.
- 과거 OOM 및 FLOP-counter assertion은 이전 시도의 기록이며 현재 실행의
  마지막 로그에는 새 traceback/CUDA OOM/NCCL timeout/NaN이 없다. 현재 실행은
  `micro_batch=1`, `gradient_accumulation=16`, FLOP counter 비활성으로 effective
  global batch 64를 보존한다. trigger watcher tmux는 종료되어 있으나 동일
  체인의 recovery tmux가 유효하게 실행 중이므로 중복 watcher는 재기동하지 않았다.

### 2026-07-27 30분 감사: Ours V1 FOMC 진행 확인

- `slora-chain-recovery` 세션과 Ours V1 부모/4개 GPU worker가 살아 있고,
  GPU 사용량은 약 18.8 GiB/장, utilization 14–18%였다. 현재 로그는 FOMC
  phase-1 epoch 3, step 약 320, 2,831/3,750 microbatch 부근까지 진행 중이다.
- 직전 감사 대비 Ours V1 로그 크기/mtime이 증가했다. 새 OOM, traceback, NCCL
  timeout, NaN/Inf는 확인되지 않았고, NCCL GPU mapping 경고만 반복된다.
- SeqLoRA는 8/8(각 order의 두 adapter 파일) 완료, Ours V1은 round0 파일 2개,
  Ours V2는 아직 0개다. trigger watcher 세션은 종료 상태지만 recovery 세션이
  단일 체인을 계속 실행 중이므로 중복 watcher는 시작하지 않았다. 체인 완료
  마커는 아직 없다.

### 2026-07-26 30분 감사: Ours V1 MeetingBank 진행 확인

- 현재 체인 run `20260726T202818Z`는 SeqLoRA 8/8을 건너뛴 뒤 Ours V1을 계속
  실행 중이다. MeetingBank phase-1 epoch 1, 약 643/8,750 microbatch까지
  진행되었고 직전 확인보다 로그 mtime/크기가 증가했다.
- 4개 GPU worker가 각각 약 20.9 GiB를 사용 중이며 프로세스가 살아 있다. 현재
  로그 tail에서 새 traceback, OOM, NCCL timeout, NaN/Inf는 발견되지 않았다.
- Ours V1 round0과 round1 checkpoint가 생성되었고, V1 workload 파일도 존재한다.
  V2 checkpoint와 CHAIN_COMPLETE/CHAIN_SUCCEEDED는 아직 없다. 기존 recovery
  실행이 단일 체인으로 정상 진행 중이므로 watcher 또는 추가 프로세스는 시작하지
  않았다.

### 2026-07-26 23:30 UTC 30분 감사: Ours V1 MeetingBank 계속 진행

- recovery 체인(`slora-chain-recovery`)의 단일 Ours V1 실행과 4개 GPU worker가
  살아 있다. 현재 로그는 MeetingBank phase-1, epoch 2, 약 2,162/8,750
  microbatch까지 진행 중이며 GPU 메모리는 약 20.9 GiB/장이다.
- 직전 감사 대비 로그가 계속 진행되고 있다. 새 traceback, CUDA OOM, NCCL
  timeout/failure, NaN/Inf는 없으며 NCCL GPU-mapping warning만 관찰된다.
- SeqLoRA는 8/8 완료, Ours V1은 non-empty checkpoint 2개, Ours V2는 아직
  0개다. CHAIN_COMPLETE/CHAIN_SUCCEEDED는 아직 없으므로 goal을 유지한다.
  기존 recovery 실행이 정상 진행 중이라 watcher 재기동이나 중복 프로세스는
  만들지 않았다.

### 2026-07-27 00:00 UTC 30분 감사: Ours V1 MeetingBank 계속 진행

- 최신 체인 run `20260726T202818Z`는 SeqLoRA 8/8을 건너뛰고 Ours V1을 실행 중이다.
  `slora-chain-recovery`의 단일 부모 프로세스와 4개 GPU worker가 살아 있으며,
  현재 로그는 MeetingBank phase-1 epoch 3, 약 step 1,150/8,750 부근까지 진행됐다.
  직전 감사 대비 로그 mtime/크기가 증가했고 GPU worker는 약 20.9 GiB/장 사용 중이다.
- 현재 run에서 새 traceback, CUDA OOM, NCCL timeout/failure, NaN/Inf는 확인되지
  않았다. 이전 trigger 로그의 과거 OOM 기록은 이미 recovery 실행으로 대체된
  과거 실패이며 현재 활성 프로세스의 오류로 분류하지 않았다.
- SeqLoRA 산출물은 8/8, Ours V1은 non-empty checkpoint 2개(메타/가중치 쌍),
  V2는 아직 0개다. CHAIN_COMPLETE/CHAIN_SUCCEEDED는 아직 없고 workload 파일은
  V1에 존재한다. 기존 recovery가 단일 체인으로 진행 중이므로 watcher 재기동,
  checkpoint 삭제, 하이퍼파라미터 변경은 하지 않았다.

### 2026-07-27 00:30 UTC 30분 감사: Ours V1 MeetingBank 계속 진행

- 최신 run `20260726T202818Z`에서 SeqLoRA 8/8 이후 Ours V1 MeetingBank
  phase-1이 계속 실행 중이다. 단일 recovery 부모/4개 GPU worker가 살아 있고,
  로그는 약 epoch 5, step 5,120/8,750까지 진행되어 직전 감사보다 mtime/크기가
  증가했다. GPU worker는 약 20.9 GiB/장 사용 중이다.
- 현재 로그에서 새 traceback, CUDA OOM, NCCL timeout/failure, NaN/Inf는 발견되지
  않았다. NCCL GPU-mapping warning은 기존 비치명 경고다. trigger tmux watcher는
  현재 없지만 active recovery 체인이 정상 진행 중이므로 중복 watcher를 만들지
  않았다.
- SeqLoRA artifact는 8/8, Ours V1은 non-empty checkpoint 2개, V2는 0개다.
  CHAIN_COMPLETE/CHAIN_SUCCEEDED는 아직 없고 V1 `training_workload.json`은
  존재한다. checkpoint 삭제·마커 조작·과학적 하이퍼파라미터 변경은 하지 않았다.


### 2026-07-27 Py150 경계 task별 batch 전환 및 FLOP 계측 감사

- Ours V1 Py150 phase-1과 current-inclusive router retune을 기존 `micro_batch=1`, `gradient_accumulation=16`으로 완료하고 round 3 checkpoint 및 workload 기록을 확인한 뒤 체인을 종료했다.
- runner가 task별 micro-batch와 task별 gradient accumulation 목록을 함께 받아 모든 task에서 `micro_batch × 4 GPU × accumulation = effective global batch 64`를 검증하도록 수정했다. 재개 profile은 C-STANCE/FOMC/ScienceQA/NumGLUE-cm/NumGLUE-ds `2/8`, MeetingBank/Py150/20Minuten `1/16`이다.
- PyTorch 2.4.1 `FlopCounterMode`의 SDPA 공식이 Llama-3.1 GQA(32 query heads/8 KV heads)를 거부하던 assertion을 custom GQA 공식으로 수정했고 CPU GQA forward/backward 계측 불변식은 통과했다.
- 그러나 ScienceQA `micro_batch=2`에서 operator counter를 task 전체에 적용하면 11번째 microbatch에서 GPU 메모리가 약 79.2 GiB까지 증가하여 OOM이 발생했다. round 3 checkpoint에는 영향이 없고 해당 시도는 성능 결과에서 제외한다.
- V1은 동일 profile의 ScienceQA부터 operator counter 비활성으로 다시 재개했으며 첫 step 이후 GPU당 약 20-22 GiB, global batch 64로 정상 진행 중이다. sample/token/update/wall-time workload 기록은 유지한다. 전체 operator FLOP은 학습과 분리된 별도 안전 계측 없이는 현재 null로 유지한다.

### 2026-07-28 23:30 KST 평가 인수인계

- Llama-3.1의 다섯 방법을 모델 우선 순서로 sparse-15 평가 중이다. sparse-15는
  order1–7의 대각선 7개와 order8의 전체 task 8개, 즉 모델당 15개 셀이다.
  방법 순서는 `slora_pre_released -> slora_post -> seq_lora ->
  ours_lora_moe_v1 -> ours_lora_moe_v2`다.
- 23:30 KST 완료 현황은 `slora_pre_released 15/15`, `slora_post 15/15`,
  `seq_lora 15/15`이다. Ours V1은 일반 셀 11개와 order3 MeetingBank까지
  완료하여 12/15이며, 현재 order8 MeetingBank를 4-GPU sample shard로 시작했다.
  Ours V2 평가는 아직 시작하지 않았다. V1/V2 학습 checkpoint 자체는 모두
  완료된 상태다.
- 현재 활성 tmux는 `sparse15-gpu-queue`다. 전체 평가가 성공 문구
  `[QUEUE] all five sparse-15 evaluations complete`를 기록하면
  `agent-data-after-sparse15-eval` watcher가
  `/home/work/Agent_HJ/30_flame_agent/agent_data_make.py`를 실행한다.
- 메인 append 로그는
  `results/eval_chains/llama31_sparse15_five_20260728/gpu_queue_driver_method_major_v2.log`,
  합본 로그는 같은 폴더의 `combined_current.log`, 일반 셀별 로그는
  `gpu_queue/` 아래에 있다. append 로그에는 이미 복구된 과거 traceback도
  남아 있으므로 마지막 `[QUEUE]` 이후와 현재 PID를 기준으로 판단한다.
- 현재 Ours V1 order8 MeetingBank 로그는
  `results/full_runs/llama31/ours_lora_moe_v1/evaluation/order8/MeetingBank.shard{0,1,2,3}.log`
  다. 긴 절대경로가 터미널에서 줄바꿈되어 실패할 수 있으므로 해당 order
  디렉터리로 `cd`한 뒤 `tail -f MeetingBank.shard{0,1,2,3}.log | tr '\r' '\n'`
  으로 보는 것이 안전하다.
- 평가 처리량 설정은 일반 task batch 16, MeetingBank `4 shards x batch 4`,
  Py150 `4 shards x batch 8`이다. Ours V1 ScienceQA 두 셀은 batch 32로
  끝났고, 평가 코드에는 다음에 시작할 Ours V2 ScienceQA를 batch 64로 올리는
  task 분기가 들어갔다. ScienceQA order5/order8을 각 2-shard로 동시에 돌리는
  아이디어는 합의했지만 아직 스케줄러에 구현하지 않았으므로, 현재 큐를 그대로
  두면 V2에서도 셀당 GPU 한 장으로 시작한다.
- Ours V1 order3 MeetingBank 결과는 BLEU-1 54.08, BLEU-4 42.37,
  ROUGE-L 63.12다. 논문 대표 지표와 AA/BWT 계산에는 MeetingBank ROUGE-L만
  사용하고 BLEU는 보조 지표로 둔다.
- `slora_pre_released` 대표 결과는 C-STANCE 58.10->52.50, FOMC
  65.32->63.31, MeetingBank ROUGE-L 58.93->48.04, Py150
  62.06->57.11, ScienceQA 93.25->91.30, NumGLUE-cm 62.96->64.20,
  NumGLUE-ds 69.54->68.62, final 20Minuten SARI 41.64다. 모든 대표 지표를
  0–100으로 놓은 order8 평균 AA는 60.84이고, 여기서 사용한 BWT 정의는
  `최초 학습 직후 - order8`의 앞 7-task 평균이라 3.59다. 이 부호에서는
  양수가 망각이다.
- 평가 중 수정한 치명 병목/오류는 다음과 같다. 20Minuten SARI는
  Hugging Face metric을 반복 로드하지 않고
  `/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark/metrics.py`의 로컬
  구현을 사용한다. 긴 MeetingBank 출력의 `rouge==1.0.1` 재귀 LCS가 기본
  recursion limit를 넘던 문제는 입력 단어 길이에 따라 limit를 올리도록
  SLoRA 두 metrics.py에 동일 반영했다. prefix 0인 첫 4-way Py150 병합에서
  존재하지 않는 `infer.jsonl`을 읽던 문제는 `py150_remaining_parts.py`가
  빈 prefix를 허용하도록 고쳤고, 실패 당시 생성된 4x500 결과는 재추론 없이
  병합했다.
- V1에 남은 순서는 현재 order8 MeetingBank, order4 Py150, order8 Py150다.
  23:30 기준 V1 종료 예상은 03:00–04:00 KST다. 이후 V2는 현재 스케줄러
  기준 약 5.5–7시간으로 보아 전체 평가 종료는 대략 08:30–11:00 KST다.
  실제 시간은 Py150에서 1024 토큰 가까이 생성되는 샘플 수에 크게 좌우된다.
- base 성능 보존용 residual/no-op expert 아이디어는 개념만 검토했고 아직
  구현하지 않았다. 현재 결론 후보는 영구 zero/frozen expert를 index 0에
  한 번 추가하고 task expert warm-up 후 자연 routing시키는 안과, 일반 도메인
  base-teacher KD/router supervision을 더하는 안이다. 어떤 방식으로 학습할지는
  더 논의하기로 했으므로 다음 Codex 세션이 임의로 구현하면 안 된다.

### 2026-07-29 14:30 KST sparse-15 1차 평가 결과

- 여기서 `1차 평가`는 각 방법이 8개 task 학습을 모두 마친 뒤의 sparse-15 평가다. 앞 7개 task는 `최초 학습 직후(order1–7 대각선) -> 전체 학습 종료 후(order8)`를 비교하고, 마지막 20Minuten은 order8 점수만 사용한다.
- task별 대표 지표는 C-STANCE/FOMC/ScienceQA/NumGLUE-cm/NumGLUE-ds Accuracy, MeetingBank ROUGE-L, Py150 Similarity, 20Minuten SARI다. 모든 값은 0–100 스케일이다.
- `AA = order8의 8개 대표 지표 평균`.
- 아래 `BWT/Forgetting = mean(R_i,i - R_8,i), i=1..7`이다. 따라서 이 문서에서는 양수가 망각, 음수가 backward transfer를 뜻한다.

| Task / 대표 지표 | SLoRA-pre | SLoRA-post | Seq-LoRA | Ours v1 | Ours v2 |
|---|---:|---:|---:|---:|---:|
| C-STANCE / Accuracy | 58.10→52.50 | 57.10→45.65 | 57.05→45.90 | 57.00→52.35 | 56.20→56.95 |
| FOMC / Accuracy | 65.32→63.31 | 64.11→63.31 | 64.11→62.70 | 71.77→62.10 | 68.75→65.93 |
| MeetingBank / ROUGE-L | 58.93→48.04 | 64.72→32.90 | 64.73→33.50 | 63.12→60.18 | 64.28→62.75 |
| Py150 / Similarity | 62.06→57.11 | 62.53→33.62 | 62.74→33.38 | 54.83→52.32 | 58.72→55.50 |
| ScienceQA / Accuracy | 93.25→91.30 | 93.50→88.65 | 93.45→88.55 | 90.60→91.05 | 91.50→90.20 |
| NumGLUE-cm / Accuracy | 62.96→64.20 | 54.32→59.26 | 54.32→58.02 | 64.20→54.32 | 43.21→53.09 |
| NumGLUE-ds / Accuracy | 69.54→68.62 | 74.15→71.69 | 74.15→71.38 | 54.15→52.92 | 60.62→52.00 |
| 20Minuten / SARI | 41.64 | 42.52 | 41.87 | 41.50 | 40.77 |
| **AA** | **60.84** | **54.70** | **54.42** | **58.34** | **59.65** |
| **BWT/Forgetting** | **3.59** | **10.77** | **11.02** | **4.35** | **0.98** |

- SLoRA-post의 MeetingBank 두 `eval.log`는 과거 ROUGE 재귀 오류 때문에 최종 점수를 기록하지 못했다. 재추론하지 않고 보존된 692개 `infer.jsonl`에 수정된 동일 ROUGE-L 함수를 적용해 order3 `64.72`, order8 `32.90`을 복원했다.
- Ours v2는 전체 AA와 평균 망각은 다섯 방법 중 양호하지만 NumGLUE-cm의 최초 성능 `43.21`과 NumGLUE-ds의 최종 성능 `52.00`이 낮다. 따라서 평균 수치만으로 신규 expert 학습이 정상이라고 결론 내리면 안 된다.
- NumGLUE-cm 진단 대조군은 원래 v2 order6 자연 routing `43.21`이고, 같은 checkpoint에서 E5를 강제한 top-1 평가는 `50.62`다. 강제 평가가 개선되지만 절대 성능은 여전히 낮아, router 선택률과 expert 자체 학습량을 함께 분리해서 봐야 한다.
- checkpoint4에서 NumGLUE-cm을 다시 학습하며 E0–E5 자연 선택률을 기록하는 aux-loss 유지 대조 실험이 진행 중이다. 이 대조군 완료 뒤 다른 조건은 그대로 두고 `moe_aux_loss_coeff=0`만 바꾸는 round5 ablation을 실행한다. 이는 checkpoint4까지 aux loss를 사용한 부분 ablation이며, 처음부터 8개 task를 aux=0으로 학습한 완전한 `v2.5` 결과와는 구분한다.


### 2026-07-29 Ours v2.5 학습 완료 및 sparse-15 평가 진행

- Ours v2.5는 Ours v2와 동일한 학습 설정을 사용하되 router auxiliary loss와 router z-loss만 제거했다. 즉 `moe_aux_loss_coeff=0`, `moe_z_loss_coeff=0`이며, v2의 joint replay/KD 구성과 task별 학습 방식은 유지한다.
- Llama-3.1-8B-Instruct Ours v2.5의 8-task 순차 학습은 완료됐다. checkpoint root는 `/home/work/Agent_HJ/30_flame_agent/slora_repro/results/full_runs/llama31/ours_lora_moe_v2_5`이고, 최종 order8 checkpoint는 `/home/work/Agent_HJ/30_flame_agent/slora_repro/results/full_runs/llama31/ours_lora_moe_v2_5/7`이다. 최종 메타데이터 파일은 `7/lora_moe_meta.json`이다.
- 현재 학습 완료 checkpoint로 sparse-15 평가를 진행 중이다. 평가는 order1--7 대각선 7개와 order8 전체 8개로 총 15개 셀이다. 일반 셀 11개는 eval batch 16으로 GPU 4장에 병렬 분배하고, 긴 셀은 일반 셀 뒤에서 MeetingBank 2개를 각각 `4 sample shards x batch 4`, Py150 2개를 각각 `4 sample shards x batch 8`로 처리한다.
- 평가 큐는 완전한 결과 파일의 sample 수가 test set 기대 개수와 일치할 때만 해당 셀을 완료로 판정한다. 2026-07-29 확인 시 order7 NumGLUE-ds는 325/325, order8 FOMC는 496/496으로 정상 완료됐고 각각 accuracy 0.5292307692, 0.6129032258이다.
- 전체 진행 로그는 `/home/work/Agent_HJ/30_flame_agent/slora_repro/results/full_runs/llama31/ours_lora_moe_v2_5/eval_workers/sparse15_queue.log`다. 큐 이벤트뿐 아니라 일반 셀 generation 로그와 이후 MeetingBank/Py150 shard 로그도 이 파일로 실시간 합치도록 follower를 연결했다.
- 15개 셀 평가와 sparse-15 결과 취합이 모두 성공한 경우에만 `/home/work/Agent_HJ/30_flame_agent/agent_data_make.py`를 자동 실행한다. 후처리 로그는 `results/full_runs/llama31/ours_lora_moe_v2_5/eval_workers/agent_data_make.log`에 기록한다.

### 2026-07-30 Ours v2.5 sparse-15 평가 완료 — 실패 판정

- **최종 판정: 실패.** Llama-3.1-8B-Instruct Ours v2.5 sparse-15 평가는 15/15 셀 모두 완료되어 실행 실패나 partial 결과는 아니지만, 방법 개선 목표를 달성하지 못했다. 각 통합 결과의 sample 수를 test set 기대 개수와 대조했으며 모두 일치한다.
- 이전 재개에서 order8 MeetingBank를 shard당 batch 8로 실행해 4개 GPU 모두 OOM이 발생했다. 문서의 원래 계획대로 MeetingBank는 `4 sample shards x batch 4`, Py150은 `4 sample shards x batch 8`로 다시 실행해 완료했다.
- order3 MeetingBank는 173개씩 4개 shard, 총 692개 추론 결과가 이미 완전했지만 통합 파일이 없었다. 재추론 없이 sample index의 연속성·중복 여부를 검증한 뒤 동일 scoring 함수로 병합했다.

| Task / 대표 지표 | Ours v2.5 최초→최종 |
|---|---:|
| C-STANCE / Accuracy | 57.70→57.40 |
| FOMC / Accuracy | 66.53→61.29 |
| MeetingBank / ROUGE-L | 64.59→60.54 |
| Py150 / Similarity | 56.61→56.85 |
| ScienceQA / Accuracy | 90.20→89.65 |
| NumGLUE-cm / Accuracy | 35.80→34.57 |
| NumGLUE-ds / Accuracy | 52.92→49.23 |
| 20Minuten / SARI | 42.27 |
| **AA** | **56.47** |
| **BWT/Forgetting** | **2.12** |

- `AA`는 order8의 8개 대표 지표 평균이다.
- 이 표의 `BWT/Forgetting`은 위 1차 평가 표와 동일하게 `mean(R_i,i - R_8,i), i=1..7`로 계산한다. 양수는 망각이다. 집계 JSON의 `BWT`는 반대 부호인 `mean(R_8,i - R_i,i)`이므로 `-2.12`다.
- v2.5는 v2 대비 AA가 `59.65→56.47`로 `3.18`점 하락했고, forgetting은 `0.98→2.12`로 `1.14`점 악화됐다. 성능과 망각이 동시에 나빠졌으므로 v2 개선안으로 채택하지 않는다.
- 특히 최종 NumGLUE-cm은 `53.09→34.57`, NumGLUE-ds는 `52.00→49.23`으로 하락했다. router auxiliary loss와 z-loss를 처음부터 제거한 설정이 NumGLUE 문제를 해결하지 못했고 전체 평균까지 훼손했다.
- 따라서 v2.5는 후속 주력 설정이나 성공 결과표에 사용하지 않고, `aux/z loss 완전 제거`가 유효하지 않았음을 보여주는 실패 ablation으로만 보존한다.
- 정규화 결과는 `results/summaries/llama31/ours_lora_moe_v2_5.json`에 저장한다.
