# 새 서버 런타임 재현

이 저장소는 서로 ABI와 라이브러리 세대가 다른 두 학습 스택을 포함한다. 한 환경에
합치지 않는다.

| 스택 | 권장 격리 | Python / PyTorch | 핵심 이유 |
|---|---|---|---|
| FLAME/G2 Megatron | Conda `flame-megatron-a100` | 3.10 / 2.4.1+cu124 | Apex, TransformerEngine, grouped-gemm, flash-attn을 같은 CUDA ABI로 빌드 |
| `trace/` SLoRA/TRACE | `trace/.venv-runtime` | 3.10 / 2.4.1+cu124 | PEFT/TRL/Transformers/DeepSpeed 버전을 FLAME과 분리 |

두 환경에서 공통으로 `PYTHONNOUSERSITE=1`을 사용한다. `pip install --user`를 쓰거나
한 환경이 활성화된 상태에서 다른 환경의 설치 스크립트를 실행하지 않는다.

## 호스트 전제

- Linux x86_64, NVIDIA A100
- NVIDIA driver가 CUDA 12.4 런타임을 지원할 것
- FLAME CUDA 확장 빌드용 CUDA toolkit 12.4 또는 12.5와 `nvcc`
- GCC/G++ 11 권장
- Git submodule을 포함한 clone

현재 체크포인트를 만든 서버는 driver 575.51.03, CUDA toolkit 12.5.82, GCC 11.4,
NGC PyTorch 24.07 계열이었다. 상세 캡처는
`flame-megatron/reference-runtime.json`에 있다.

## 1. FLAME/G2 Megatron

새 서버에서 다음 순서로 설치한다.

```bash
git submodule update --init --recursive
bash scripts/miscellaneous/install_a100_env.sh
conda activate flame-megatron-a100
source scripts/miscellaneous/activate_flame_env.sh
python scripts/miscellaneous/verify_flame_env.py --require-gpu
```

설치 스크립트는 다음을 한 묶음으로 구성한다.

- official PyTorch `2.4.1+cu124`, torchvision `0.19.1`, torchaudio `2.4.1`
- 현재 실험에서 사용한 Transformers `4.33.1`, Datasets `2.20.0`, W&B 등
- FFN MoE와 attention LoRA expert가 사용하는 grouped-gemm `1.1.4`
- 저장소에 고정된 Apex 0.1과 TransformerEngine 1.11.0 source
- 다른 커밋된 attention 경로가 요구하는 flash-attn `2.4.2`

`--skip-extensions`는 YAML/Python 패키지만 먼저 확인할 때만 사용한다. G2의
`--moe-grouped-gemm`, `--attn-lora-grouped-gemm` 실행 전에는 확장 빌드가 반드시
완료되어야 한다.

### NGC 24.07과의 차이

기존 서버의 `torch 2.4.0a0+...nv24.07`, Apex, TransformerEngine은 NVIDIA가 NGC
이미지용으로 만든 바이너리라 일반 Conda/PyPI에서 동일 artifact를 받을 수 없다.
bit-for-bit 환경이 필요하고 새 서버에서 container를 쓸 수 있다면
`nvcr.io/nvidia/pytorch:24.07-py3`가 가장 가까운 기준이다. 이 저장소의 Conda
설치는 서버 이전성이 높은 호환 재현 경로이며, torch `2.4.1+cu124` 위에 현재
submodule source를 다시 빌드한다. 따라서 전체 학습 전에 import 검증과 1~10 step
smoke를 거쳐야 한다.

## 2. TRACE/SLoRA

TRACE는 Conda가 아니라 프로젝트-local venv를 권장한다. CUDA 런타임이 포함된
official torch wheel을 사용하고, 순수 Python 의존성이 대부분이라 이 방식이 더
단순하고 기존 실행 환경과도 같다.

```bash
cd trace
./scripts/setup_runtime.sh
source .venv-runtime/bin/activate
python scripts/runtime_preflight.py --skip-gpu
python scripts/runtime_preflight.py --model llama31 --world-size 4
```

- 직접 의존성: `config/requirements-runtime.txt`
- HF 다운로드/업로드와 W&B 도구: `config/requirements-tools.txt`
- 전체 transitive lock: `config/requirements-runtime.lock`

설치 스크립트는 committed lock을 읽으며 lock 파일을 현재 시점의 최신 버전으로
다시 쓰지 않는다. 그래서 새 서버 설치 시점이 달라도 같은 dependency resolution을
사용한다.

## 환경 전환

FLAME에서 TRACE로 바꿀 때:

```bash
conda deactivate
unset PYTHONPATH FLAME_MOE_REPO_DIR
source trace/.venv-runtime/bin/activate
```

TRACE에서 FLAME으로 바꿀 때는 새 shell을 여는 것이 가장 안전하다.

```bash
conda activate flame-megatron-a100
source scripts/miscellaneous/activate_flame_env.sh
```

검증 출력의 Python executable, torch, CUDA, Transformers 버전을 로그와 함께
보관한다. 검증이 실패하면 checkpoint 학습을 시작하지 않는다.
