# KT Session Restart Runbook

KT 서버에서 새 터미널/세션을 열었을 때 현재 shared-router QKVO 실험 환경을 복구하는 순서입니다.

## 0. 새 KT 세션 빠른 복구 커맨드

아래 블록을 새 KT 세션에서 순서대로 실행합니다. `gh` 없이 GitHub token을 `git credential`에 저장하고, 세션마다 사라지는 `grouped_gemm`은 git tag에서 다시 빌드합니다.

### 0.1 Runtime 확인

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

echo "=== runtime ==="
which python
which pip
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY
```

### 0.2 GitHub token 저장 + main pull

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

git config --global credential.helper store
printf "protocol=https\nhost=github.com\n\n" | git credential reject

echo -n "GitHub token: "
stty -echo
read GITHUB_TOKEN
stty echo
echo

printf "protocol=https\nhost=github.com\nusername=x-access-token\npassword=%s\n\n" "$GITHUB_TOKEN" | git credential approve
unset GITHUB_TOKEN

GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/yemokoo/LLM-continual-learning.git HEAD

git fetch origin main
git checkout main
git pull --ff-only origin main

git submodule sync
git submodule update --init --recursive

cd Megatron-LM
git checkout qv-lora-bf16-fix
git pull --ff-only origin qv-lora-bf16-fix
cd ..
```

### 0.3 W&B 설치/로그인

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --upgrade \
  --index-url https://pypi.org/simple \
  --disable-pip-version-check \
  --timeout 60 --retries 1 \
  "wandb>=0.27.0"

mkdir -p ~/.config/wandb
chmod 700 ~/.config/wandb

echo -n "W&B API key: "
stty -echo
read WANDB_API_KEY
stty echo
echo

cat > ~/.config/wandb/env <<EOF
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_ENTITY="yemoyemo010831-korea-university"
export WANDB_PROJECT="flame-continual-top2-qv-lora"
EOF
chmod 600 ~/.config/wandb/env
unset WANDB_API_KEY

source ~/.config/wandb/env

python - <<'PY'
import os, wandb, inspect
print("wandb", wandb.__version__, inspect.getfile(wandb))
print("WANDB_API_KEY length", len(os.environ.get("WANDB_API_KEY", "")))
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
print("wandb login ok")
PY
```

### 0.4 grouped-gemm 확인 및 복구

`flash-attn`은 현재 실험에서 새로 받지 않는 방향입니다. 여기서는 `grouped_gemm`과 TransformerEngine만 확인하고, 실패하면 기존에 성공했던 `fanshiqing/grouped_gemm@v1.1.4` 방식으로 복구합니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

python - <<'PY'
import grouped_gemm, transformer_engine.pytorch, inspect
print("grouped_gemm", inspect.getfile(grouped_gemm))
print("TE ok")
PY
```

위 import가 실패하면 아래를 실행합니다.

```bash
PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --upgrade \
  --index-url https://pypi.org/simple \
  ninja wheel "packaging<25"

MAX_JOBS=8 PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --no-build-isolation --no-cache-dir \
  git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

### 0.5 최종 확인

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh
source ~/.config/wandb/env

python - <<'PY'
import torch, wandb, grouped_gemm, transformer_engine.pytorch, inspect
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("wandb", wandb.__version__, inspect.getfile(wandb))
print("grouped_gemm", inspect.getfile(grouped_gemm))
print("all ok")
PY

PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile \
  Megatron-LM/megatron/training/arguments.py \
  Megatron-LM/megatron/training/training.py \
  Megatron-LM/pretrain_gpt.py
```

핵심 순서:

1. Repo 진입과 KT runtime 활성화
2. 의존성 확인/복구: `flash-attn`, `grouped_gemm`, `wandb`
3. W&B API token 저장/로드: 86자 token 지원을 위해 최신 W&B 사용
4. GitHub token 인증과 최신 코드 pull
5. 최종 import/test 확인
6. 현재 G2 teacher-student router KD 학습 실행

## 1. Repo 진입과 기본 런타임

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

echo "=== runtime ==="
which python
which pip
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY
```

## 2. 의존성 확인/복구

KT system pip config에는 `https://pypi.ngc.nvidia.com` extra index가 박혀 있어 설치가 멈출 수 있습니다. PyPI 설치 명령에는 기본적으로 `PIP_CONFIG_FILE=/dev/null`을 붙입니다.

### 2.1 W&B package shadowing 확인

repo root의 `./wandb` 디렉토리가 실제 W&B package를 가리는 경우가 있습니다. 먼저 확인합니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

python - <<'PY'
try:
    import wandb, inspect
    print("wandb", getattr(wandb, "__version__", "NO_VERSION"))
    print("path", getattr(wandb, "__file__", None) or inspect.getfile(wandb))
    print("init?", hasattr(wandb, "init"))
except Exception as exc:
    print("wandb import failed:", repr(exc))
PY
```

`NO_VERSION`, `init? False`, `path=None`, 또는 `ModuleNotFoundError`가 나오면 local `./wandb` 디렉토리를 백업 이름으로 바꿉니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

if [ -d wandb ]; then
  mv wandb "wandb_shadow_backup_$(date +%Y%m%d_%H%M%S)"
fi
```

### 2.2 W&B 최신 버전 설치

86자 W&B token은 구버전 W&B에서 `API key must be 40 characters long` 에러가 날 수 있습니다. `wandb==0.19.11`로 고정하지 말고 최신 버전 또는 최소 `0.27.0` 이상을 사용합니다.

```bash
PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --upgrade \
  --index-url https://pypi.org/simple \
  --disable-pip-version-check \
  --timeout 60 --retries 1 \
  "wandb>=0.27.0"

python - <<'PY'
import wandb, inspect
print("wandb", wandb.__version__)
print("path", inspect.getfile(wandb))
print("init?", hasattr(wandb, "init"))
PY
```

### 2.3 flash-attn / grouped_gemm 확인

이미 import가 되면 재설치하지 않아도 됩니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

python - <<'PY'
import flash_attn, grouped_gemm, inspect
print("flash_attn", flash_attn.__version__, inspect.getfile(flash_attn))
print("grouped_gemm", inspect.getfile(grouped_gemm))
PY
```

실패하면 아래를 실행합니다.

```bash
PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --upgrade \
  --index-url https://pypi.org/simple \
  ninja wheel "packaging<25"

MAX_JOBS=8 PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --force-reinstall --no-cache-dir \
  --index-url https://pypi.org/simple \
  --no-build-isolation --no-deps --no-binary flash-attn \
  flash-attn==2.4.2

PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --no-build-isolation --no-cache-dir \
  git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

## 3. W&B API token 저장/로드

`WANDB_API_KEY`는 86자 token을 그대로 저장합니다. 구버전 `wandb login`이 40자 제한으로 실패하면 W&B package를 먼저 업데이트해야 합니다.

```bash
mkdir -p ~/.config/wandb
chmod 700 ~/.config/wandb

if [ -f ~/.config/wandb/env ]; then
  source ~/.config/wandb/env
else
  echo -n "W&B API key: "
  read -s WANDB_API_KEY
  echo
  cat > ~/.config/wandb/env <<EOF
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_ENTITY="yemoyemo010831-korea-university"
export WANDB_PROJECT="flame-continual-top2-qv-lora"
EOF
  chmod 600 ~/.config/wandb/env
  source ~/.config/wandb/env
fi

echo "WANDB_API_KEY length: ${#WANDB_API_KEY}"
echo "WANDB_ENTITY=${WANDB_ENTITY:-}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-}"

python - <<'PY'
import os, wandb, inspect
print("wandb", wandb.__version__, inspect.getfile(wandb))
print("WANDB_API_KEY length", len(os.environ.get("WANDB_API_KEY", "")))
PY
```

온라인 실행 전 인증을 명시적으로 확인하고 싶으면 아래를 한 번 실행합니다.

```bash
python - <<'PY'
import os, wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
print("wandb login ok")
PY
```

## 4. GitHub token 인증과 최신 코드 pull

토큰이 이미 저장되어 있으면 바로 fetch/merge만 수행합니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
git config --global credential.helper store

if [ ! -f ~/.git-credentials ]; then
  echo -n "GitHub token: "
  read -s GITHUB_TOKEN
  echo
  printf "https://x-access-token:%s@github.com\n" "$GITHUB_TOKEN" > ~/.git-credentials
  chmod 600 ~/.git-credentials
fi

GIT_TERMINAL_PROMPT=0 git fetch --no-tags --recurse-submodules=no \
  https://github.com/yemokoo/LLM-continual-learning.git \
  slurm

git merge --ff-only FETCH_HEAD

git submodule sync
git submodule update --init --recursive

echo "=== parent ==="
git log --oneline -3
echo "=== Megatron-LM ==="
git -C Megatron-LM log --oneline -3
```

merge가 로컬 수정 때문에 막히면, 해당 수정은 지우지 말고 stash로 보관합니다.

```bash
git status --short
git stash push -m "kt-local-before-session-restart"
git merge --ff-only FETCH_HEAD
git submodule update --init --recursive
```

## 5. 최종 import / 문법 / unit test 확인

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh
source ~/.config/wandb/env 2>/dev/null || true

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY

python - <<'PY'
import apex
print("apex ok")
PY

python - <<'PY'
import transformer_engine.pytorch
print("transformer_engine ok")
PY

python - <<'PY'
import wandb, flash_attn, grouped_gemm, inspect
print("wandb", wandb.__version__, inspect.getfile(wandb))
print("flash_attn", flash_attn.__version__, inspect.getfile(flash_attn))
print("grouped_gemm", inspect.getfile(grouped_gemm))
PY

PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile \
  Megatron-LM/megatron/core/transformer/moe/router.py \
  Megatron-LM/megatron/core/transformer/transformer_config.py \
  Megatron-LM/megatron/training/arguments.py \
  Megatron-LM/megatron/training/training.py \
  Megatron-LM/pretrain_gpt.py
```

router row masking / all-router-row 학습 unit test:

```bash
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29871

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/Megatron-LM" \
python -m pytest \
  Megatron-LM/tests/unit_tests/transformer/moe/test_continual_learning_utils.py -q
```

기대 결과:

```text
3 passed
```

## 6. 현재 G2 teacher-student router KD 학습 실행

최신 코드에서는 아래 스크립트가 있어야 합니다.

```bash
ls -lh scripts/experiment/a100/run_g2_teacher_student_router_kd_fullwiki_mha.sh
bash -n scripts/experiment/a100/run_g2_teacher_student_router_kd_fullwiki_mha.sh
```

full Wiki train dataset 확인:

```bash
ls -lh data/wiki/train/train_text_document.*
```

현재 본 학습 설정:

- G2 shared-router hybrid, 8 -> 16 experts
- Code LM + Wiki teacher-student router KD joint update
- Code LM: old/new shared router row 전체 학습, 기존 wiki experts/LoRA는 freeze
- Wiki KD: student shared router만 학습
- `ROUTER_MEMORY_KL_COEFF=1.0`
- W&B online metric-only: checkpoint artifact 업로드 끔
- W&B/probe/log interval: 20 step
- local checkpoint save interval: 60 step

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh
source ~/.config/wandb/env

export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"

export RUN_ID="g2-ts-routerkd-allrouter-kl1p0-log20-save60-1800"
export TRAIN_WEIGHTS="$PWD/.local/weights/a100/mha/shared-router-granularity-qkvo/code/$RUN_ID"
export WANDB_EXP_NAME="G2 ts-routerKD allrouter kl1p0 log20 save60 - wiki to code"

LOG="g2_ts_routerkd_allrouter_kl1p0_log20_save60_online_$(date +%Y%m%d_%H%M%S).log"

WANDB_MODE=online \
WANDB_LOG_CHECKPOINTS=0 \
RUN_ID="$RUN_ID" \
TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
WANDB_EXP_NAME="$WANDB_EXP_NAME" \
ROUTER_MEMORY_KL_COEFF=1.0 \
MICRO_BATCH_SIZE=72 \
LOG_INTERVAL=20 \
SAVE_INTERVAL=60 \
PROBE_EVAL_INTERVAL=20 \
SECONDARY_PROBE_EVAL_INTERVAL=20 \
RUN_INITIAL_PROBE_EVAL=1 \
SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1 \
nohup bash scripts/experiment/a100/run_g2_teacher_student_router_kd_fullwiki_mha.sh \
  > "$LOG" 2>&1 &

echo "$LOG"
tail -f "$LOG"
```

시작 후 설정 확인:

```bash
RUN_DIR=".local/weights/a100/mha/shared-router-granularity-qkvo/code/g2-ts-routerkd-allrouter-kl1p0-log20-save60-1800"
RUNLOG="$RUN_DIR/logs/run.log"

grep -nE "router_memory_kl_coeff|save_interval|log_interval|probe_eval_interval|wandb_exp_name|wandb_run_id|wandb_log_checkpoints|shared_router_hybrid_train_all_router_rows" "$RUNLOG" | head -120

grep -nE "teacher-student router memory KD|iteration +[0-9]+/ +1800|probe code_probe|probe wiki_probe|saving checkpoint|after training is done" "$RUNLOG" | tail -80
```

기대값:

```text
router_memory_kl_coeff .......................... 1.0
save_interval ................................... 60
log_interval .................................... 20
wandb_log_checkpoints ........................... False
shared_router_hybrid_train_all_router_rows ...... True
```

## 7. ETA 확인

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

RUN_DIR=".local/weights/a100/mha/shared-router-granularity-qkvo/code/g2-ts-routerkd-allrouter-kl1p0-log20-save60-1800"
RUNLOG="$RUN_DIR/logs/run.log"

python - <<'PY'
import re, datetime
from pathlib import Path

runlog = Path(".local/weights/a100/mha/shared-router-granularity-qkvo/code/g2-ts-routerkd-allrouter-kl1p0-log20-save60-1800/logs/run.log")
pat = re.compile(r"\[(.*?)\]\s+iteration\s+(\d+)/\s*(\d+)\s+\|\s+elapsed time per iteration \(ms\):\s*([0-9.]+)")
rows = []
for line in runlog.read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        ts, it, total, ms = m.groups()
        rows.append((ts, int(it), int(total), float(ms)))

if not rows:
    raise SystemExit("no timing rows yet")

ts, it, total, ms = rows[-1]
remain = max(total - it, 0)
eta_sec = remain * ms / 1000.0
eta = datetime.datetime.now() + datetime.timedelta(seconds=eta_sec)

print(f"latest logged time: {ts}")
print(f"latest iteration: {it}/{total}")
print(f"last step time: {ms/1000:.2f} sec/iter")
print(f"remaining: {remain} iters")
print(f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"remaining time: {datetime.timedelta(seconds=int(eta_sec))}")
PY
```
