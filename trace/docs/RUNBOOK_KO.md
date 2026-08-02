# SLoRA 논문 재현 실행 가이드

## 목표

두 paper backbone과 격리 runtime은 준비되어 있다. 하나의 명령으로 학습, 평가, 결과표 생성을
수행한다. 원본 공개 구현과 감사 과정에서 수정한 구현의 결과는 절대 같은
이름으로 저장하지 않는다.

주 실험 backbone은 다음 두 개다.

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

현재 두 모델의 4개 safetensors shard, tokenizer/config, TRACE 데이터, 4개 A100,
고정 패키지와 학습/평가 진입점 smoke test가 모두 통과했다. 두 backbone의
bfloat16 CUDA 실제 로드와 유한 logits forward도 통과했다.
SLoRA denoising reference는 논문 4.3절대로 adapter 병합 전 고정 `theta_0`의
7개 projection을 rank 0 CPU snapshot으로 보존한다. 아직 장시간 학습
점수는 생성하지 않았으므로 논문 수치 재현 완료를 주장하지 않는다.

TRACE 학습 데이터는 여덟 task 모두 5,000개를 사용하며 순서와 epoch는
다음과 같이 고정한다.

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

## 구현 상태

| 실행 이름 | 코드 경로 | 상태와 의미 |
|---|---|---|
| `seq_lora` | `implementations/SLoRA-repro` | SLoRA와 동일한 unified trainer 대조군 |
| `slora_pre_released` | `implementations/SLoRA-upstream-port` | 저자 공개 코드 동작 보존본 |
| `slora_pre` | `implementations/SLoRA-repro` | 수식 기준 감사·수정 후 task 직후 denoising |
| `slora_post` | `implementations/SLoRA-repro` | raw Seq-LoRA 학습 후 평가 시 denoising |
| `ewc` | `implementations/TRACE-repro` | 공개 TRACE EWC를 현대 backbone에 포팅 |
| `lwf` | `implementations/TRACE-repro` | 공개 TRACE LwF를 현대 backbone에 포팅 |
| `gem_upstream` | `implementations/TRACE-repro` | 공개 qpth 수식 그대로, 결함 비교용 |
| `gem_corrected` | `implementations/TRACE-repro` | 감사된 qpth 부호/제약 수정 |
| `olora_upstream` | `implementations/TRACE-repro` | TRACE O-LoRA의 A/L1 penalty 포팅 |
| `olora_corrected` | `implementations/TRACE-repro` | 논문 대응 B/Frobenius-squared penalty |
| `loramoe` | `implementations/llmcl_benchmark` | FFN LoRA-MoE의 두 백본/TRACE-5000 로컬 호환 포트 |

백본별 전체 카탈로그와 실행 진입점은 `scripts/baselines/README_KO.md`,
`scripts/baselines/llama31/<method>.sh`,
`scripts/baselines/qwen25_7b/<method>.sh`에 각각 분리되어 있다.
LoRAMoE는 SLoRA 저자의 공개 구현이 아니므로 공식 논문 수치 재현이 아니라
동일 데이터·백본 비교를 위한 로컬 포트로 보고한다.

중요: SLoRA 논문 저자의 unified-framework O-LoRA 실행 코드는 공개 SLoRA
저장소에 없다. 여기의 O-LoRA 두 실행은 공개 TRACE 구현을 동일한
rank/alpha/target module로 포팅한 통제 실험이다. 논문의 O-LoRA 수치를
정확히 재현했다고 표기하면 안 된다. 공개 실행 코드가 없는 SD-LoRA와
RCL도 현재 실행 가능 목록에 넣지 않는다.

## 0. 모델 다운로드와 상태 확인

방화벽 연결이 불안정한 서버에서는 구형 CLI로 shard를 하나씩 이어받는다.

```bash
cd LLM-continual-learning/trace
./scripts/download_paper_models_legacy.sh
```

다른 터미널에서 진행률을 확인한다.

```bash
./scripts/status.sh
```


## 1. 런타임 설치

```bash
cd LLM-continual-learning/trace
./scripts/setup_runtime.sh
source .venv-runtime/bin/activate
```

설치 스크립트는 저장소에 commit된 `config/requirements-runtime.lock`을 그대로
설치한다. 전역 `PYTHONPATH`를 차단하고 실행 스크립트가 이 가상환경을 자동
사용하므로 activation은 선택 사항이다. 같은 setup 명령을 다시 실행해도 이미
설치된 고정 버전을 재사용한다.

## 2. 모델과 데이터 확인

```bash
python scripts/preflight.py --mode full \
  --models llama31_8b_instruct qwen25_7b_instruct
```

모든 shard, tokenizer, config와 여덟 task의 5,000개 train record가
확인되기 전에는 전체 학습을 시작하지 않는다.

## 3. 실제 실행 전 명령 검증

```bash
./scripts/run_suite.sh validate llama31
./scripts/run_suite.sh validate qwen25_7b
```

`validate`는 학습을 시작하지 않는다. 데이터 계약과 최종 명령만 생성한다.

## 4. 방법 하나 실행

학습부터 8×8 평가와 정규화 결과 JSON 생성까지:

```bash
./scripts/run_experiment.sh all slora_pre llama31
```

학습만:

```bash
./scripts/run_experiment.sh train gem_corrected qwen25_7b
```

이미 학습된 checkpoint 평가만:

```bash
./scripts/run_experiment.sh eval seq_lora llama31
```

GPU 선택 예:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 \
  ./scripts/run_experiment.sh all slora_post llama31
```

TRACE 계열은 `GPU_LIST=0,1,2,3`을 사용한다.

## 4.1 최장 task 기준 배치 설정

MeetingBank가 raw token 기준 가장 긴 task이며, 실제 학습은 `max_length=1024`로
절단된다. 4개 A100에서 두 backbone을 실제 측정한 결과 SLoRA 기본값은
`MICRO_BATCH=8`, `GRAD_ACCUM=2`로 설정했다. 유효 global batch는 64로 기존
MB 2/accumulation 8 설정과 같아서 epoch당 optimizer step 수는 변하지 않는다.

Llama에서 MB 8은 최대 34.8GB와 평균 6.10초/step, Qwen에서는 최대
36.4GB와 5.63초/step이었다. MB 16은 Llama에서 52.6GB를 사용하면서 MB 8보다
약 3%만 빨라 기본값으로 채택하지 않았다. 상세 측정은
`reports/batch_benchmark.md`에 있다.

필요하면 실행 시 덮어쓸 수 있다.

```bash
MICRO_BATCH=4 GRAD_ACCUM=4 \
  ./scripts/run_experiment.sh train slora_pre llama31
```

## 5. 전체 suite 실행

한 backbone의 모든 구현을 순차 실행한다.

```bash
./scripts/run_suite.sh all llama31
```

이 명령은 매우 오래 걸리고 큰 checkpoint를 생성한다. 각 방법의 평가가 끝나면
`results/summaries/<model>/<method>.json`도 자동 생성한다. 두 backbone을
동시에 같은 GPU에서 실행하지 않는다.

## 6. 결과 수집 및 비교

평가 산출물을 논문식 continual matrix로 변환한다.

```bash
python scripts/collect_results.py --method slora_pre --model llama31
python scripts/collect_results.py --method gem_corrected --model llama31
```

두 결과의 최종 평균과 AFR을 비교한다.

```bash
python scripts/compare_results.py \
  results/summaries/llama31/slora_pre.json \
  results/summaries/llama31/gem_corrected.json \
  --output results/summaries/llama31/comparison.json
```

대표 metric은 C-STANCE/FOMC/ScienceQA/NumGLUE의 accuracy,
MeetingBank의 ROUGE-L, Py150의 edit similarity, 20Minuten의 SARI다.
모든 결과는 논문 표와 같이 0–100 point 단위로 저장한다.

## 결과 폴더

```text
results/full_runs/
└── <model>/
    ├── seq/
    ├── pre/
    ├── post/
    ├── ewc_upstream/
    ├── lwf_upstream/
    ├── gem_upstream/
    ├── gem_corrected/
    ├── olora_upstream/
    └── olora_corrected/
```

각 실행에는 command, 데이터 계약, checkpoint, train log, evaluation
artifact가 함께 저장된다. 최종 정규화 결과는
`results/summaries/<model>/`에 저장된다.
