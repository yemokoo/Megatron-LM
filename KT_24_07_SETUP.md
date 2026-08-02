# KT / NGC 24.07 Setup

> 이전의 `/usr/bin/python + pip --user + torch 2.5.1` 복구 절차는 폐기했다.
> 사용자 전역 패키지와 NGC 기본 패키지가 섞여 checkpoint 재현성을 보장하지
> 못하기 때문이다. 새 서버의 정식 절차는
> [environments/README_KO.md](./environments/README_KO.md)다.

## 권장 선택

KT에서 NGC PyTorch image를 골라야 한다면 `24.07`을 base로 사용한다. Python
3.10과 PyTorch 2.4 계열이라 현재 G2 checkpoint를 만든 서버와 가장 가깝다.

- container artifact까지 최대한 같아야 할 때: NGC 24.07 안에서 stock runtime을
  유지하고 저장소 import/smoke를 먼저 검증한다.
- 서버를 옮겨도 다시 만들 수 있는 환경이 필요할 때: NGC 또는 호스트 위에 별도
  Conda `flame-megatron-a100`을 만든다.
- TRACE/SLoRA: 위 두 경우와 관계없이 `trace/.venv-runtime`을 따로 만든다.

## Portable Conda 절차

```bash
git clone https://github.com/yemokoo/LLM-continual-learning.git
cd LLM-continual-learning
git submodule update --init --recursive

bash scripts/miscellaneous/install_a100_env.sh
conda activate flame-megatron-a100
source scripts/miscellaneous/activate_flame_env.sh
python scripts/miscellaneous/verify_flame_env.py --require-gpu
```

검증 후 현재 G2 설정의 1~10 step smoke를 통과하기 전에는 장시간 학습을 시작하지
않는다.

## TRACE/SLoRA 절차

```bash
cd trace
./scripts/setup_runtime.sh
source .venv-runtime/bin/activate
python scripts/runtime_preflight.py --skip-gpu
```

과거 호환을 위해 `install_kt_24_07_no_conda.sh`와 `activate_kt_env.sh` 파일은
남겨 두었지만 신규 서버 재현에는 사용하지 않는다.
