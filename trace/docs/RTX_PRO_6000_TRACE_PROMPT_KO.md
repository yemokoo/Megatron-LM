# RTX PRO 6000 Blackwell 8장 서버용 TRACE 실행 프롬프트

아래 내용을 새 서버의 코딩 에이전트에게 그대로 전달한다. 현재 H100 서버의 실행 중인 파일과 환경은 수정하지 않는다.

---

당신은 `yemokoo/LLM-continual-learning` 저장소의 TRACE 실험을 RTX PRO 6000 Blackwell 96GB GPU 8장 서버에서 재현해야 한다. 목표는 **현재 v1 생성/답변 방식**을 유지하면서 두 개의 데이터 양 실험을 순서대로 학습·평가하는 것이다. 작업을 시작하기 전에 GPU 8장, 드라이버, 디스크 여유 공간, Python 경로를 확인하고 실제 경로를 기록하라. 막히는 인증이나 누락 파일은 정확히 보고하라. 학습 조건을 임의로 바꾸지 마라.

## 소스와 실행 경계

1. `https://github.com/yemokoo/LLM-continual-learning.git`의 `feature/trace-v3-shared-qkvo-experts` 브랜치를 clone하고 `trace/`로 이동한다.
2. 실행 진입점은 `scripts/bos_token/runs/v1_1000_chain.sh`와 `scripts/bos_token/runs/v1_1500_2000_rep1000_chain.sh`다. 두 스크립트는 생성, 학습, sparse15 평가까지 연결한다. 생성 작업 큐는 `scripts/bos_token/regen_queue_v1.py`다. 새 서버의 절대경로를 소스에 박아 넣지 말고 아래 환경변수를 쓴다.
3. 원본 v1 방식의 Stage A는 `scripts/bos_token/gen_doc.py`의 chat-header-only 프롬프트와 결정 위치의 expert 강제다. Stage B는 `scripts/analysis/answer_pass_v3_fix.py`의 기존 수동 헤더·cue 방식이다. **학습용 헤더로 Stage B를 교체하거나 답변 시작 expert를 강제하거나 누락 anchor를 앞에 붙이지 마라.** 생성 후 v1 anchor 필터를 적용한다.
4. 작업 큐는 GPU 8장에 샤드를 동적으로 배분한다. MeetingBank, Py150, ScienceQA는 태스크당 10샤드이고 나머지는 8샤드다. 이 샤드 구성을 두 실험에 동일하게 사용한다. 샤드 수는 생성 데이터와 seed별 표본을 바꿀 수 있으므로 기록한다.

## 런타임

1. 이 저장소의 `scripts/setup_runtime.sh` 및 `config/requirements-runtime.lock`은 **H100용 torch 2.4.1+cu124를 고정**하므로 Blackwell 서버에서 그대로 실행하지 마라.
2. 별도 Python 3.10 가상환경 `trace/.venv-runtime`을 만들고 NVIDIA 드라이버와 호환되는 **sm_120 지원 PyTorch CUDA 12.8 이상 빌드**를 설치한다. 이어 `config/requirements-runtime.txt`의 Python 의존성을 설치한다. 설치 과정에서 torch가 구버전으로 대체되지 않았는지 재확인한다. 기존 H100 서버 환경은 변경하지 않는다.
3. 최소 확인: `python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.device_count())'` 결과에 `sm_120`과 GPU 8장이 있어야 한다. 8개 GPU 각각에서 CUDA 텐서 연산, BF16 연산, `torch.distributed` NCCL 통신이 되는지 확인한다. 문제가 생기면 학습을 걸기 전에 드라이버/torch 설치를 고친다.
4. 이 실험은 일반 PyTorch DDP와 Transformers 기반이다. 현재 실행 경로에는 Megatron이나 사용자 정의 CUDA 확장 빌드가 없다. 그래도 새 서버의 조합에서 실제 모델 로드와 한 번의 forward가 되는지 확인한 후 전체 체인을 건다.

## 데이터와 모델

1. Hugging Face CLI 인증을 완료한다. TRACE 데이터는 비공개 `YeMoKoo/flamedata2` 저장소의 `trace/**`에서 받는다. 접근 권한이 없으면 사용자에게 요청한다. 명령 예: `TRACE_HF_LOCAL_DIR="$PWD/data" ./scripts/data/download_trace_from_hf.sh`. 8개 태스크의 `train.json`, `eval.json`, `test.json`이 모두 있는지 스크립트가 검증한다. 데이터는 Git에 올리지 않는다.
2. `meta-llama/Llama-3.1-8B-Instruct` 접근 권한을 확인하고 `trace/models/Llama-3.1-8B-Instruct`로 다운로드한다. `scripts/download_paper_models.sh`는 불필요한 Qwen 모델도 다운로드하므로 이 실험에서는 HF CLI로 Llama 모델만 받는다. 모델은 Git에 올리지 않는다.
3. **두 체인의 시작점은 기존 v1 C-STANCE `model/0` 체크포인트**다. 원 서버의 `/data2/seonghyeonnoh/paper/bos_token/header_forced/model/0/` 전체와 `model/training_workload.json`, `model/epoch_probe.jsonl`, `model/fixed_replay_memory/task_0_C-STANCE.json`을 같은 상대 경로로 새 서버의 `trace/seed_v1/model/`에 복사한다. 첫 체인은 이 파일들로 round 0의 이력과 replay 상태를 초기화한다. 체크포인트는 Git에 없으며, 없으면 동일한 시작점의 실험을 수행할 수 없다. 다른 체크포인트로 조용히 대체하지 마라.
4. 코드와 데이터 위치가 기본값과 다르면 `TRACE_PYTHON`, `TRACE_DATA_ROOT`, `SLORA_LLAMA31_PATH`, `TRACE_TOKEN_CACHE_DIR`, `V1_CSTANCE_MODEL_ROOT`, `TRACE_RUN_ROOT`를 지정한다. `V1_CSTANCE_MODEL_ROOT`는 `0/lora_moe_meta.json`이 들어 있는 `model` 디렉터리다.

## 실행 순서와 결과

1. 먼저 `bash scripts/bos_token/runs/v1_1000_chain.sh`를 실행한다. 매 라운드 이전 태스크당 1000개 생성, v1 anchor 필터 후 최대 500개 선택해 replay 학습에 사용한다. 부족하면 통과한 만큼만 쓴다. 태스크 순서는 C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten이다. C-STANCE는 옮겨온 v1 체크포인트를 사용한다.
2. 첫 체인의 `EVENT: ALL_DONE` 및 `model/sparse15_summary.json`을 확인한 뒤 `bash scripts/bos_token/runs/v1_1500_2000_rep1000_chain.sh`를 시작한다. MeetingBank·ScienceQA는 태스크당 2000개, 다른 이전 태스크는 1500개 생성한다. 같은 v1 필터 후 태스크당 최대 1000개를 저장·학습에 사용한다. 부족하면 통과한 만큼만 쓴다. 이 체인 역시 동일한 v1 C-STANCE 체크포인트에서 시작하는 독립적인 데이터 양 ablation이다.
3. 실행 로그는 각 결과 디렉터리의 `chain.log`, `logs/`, `gen/round_*/regen_queue.log`에 남는다. 프로세스는 SSH 종료에 영향받지 않도록 `tmux` 또는 서비스 매니저에서 실행한다. 실패하면 마지막 정상 체크포인트와 로그를 확인하고 같은 체인을 재실행해 이어간다. 두 체인을 동시에 돌려 GPU 메모리를 경쟁시키지 마라.
4. 결과 보고에는 태스크별 요청·통과·선택 수, 체크포인트, sparse15 AA/F/LA, 실행 환경(torch/CUDA/드라이버/GPU), 전체 시간, 오류와 재시작 이력을 포함한다. 원본 데이터와 모델 가중치는 Git에 commit하지 않는다.

---

저장소의 추적 코드에는 기존 작업 트리의 다른 실험 변경이 섞여 있을 수 있다. 이 재현에 필요한 파일만 사용하고, 학습 조건이나 필터를 수정해야 한다면 변경 이유와 기존 실험과의 차이를 먼저 명시하라.
