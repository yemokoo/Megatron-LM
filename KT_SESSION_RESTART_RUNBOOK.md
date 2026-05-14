# KT Session Restart Runbook

이 문서는 KT 서버에서 새 터미널/세션을 열었을 때 그대로 복붙해서 현재 shared-router QKVO 실험 환경을 복구하기 위한 최소 절차입니다.

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

## 2. GitHub 인증과 최신 코드 pull

토큰이 이미 저장되어 있으면 바로 pull만 수행합니다.

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

git submodule sync Megatron-LM
git -C Megatron-LM fetch origin qv-lora-bf16-fix
git submodule update --init Megatron-LM

echo "=== parent ==="
git log --oneline -3
echo "=== Megatron-LM ==="
git -C Megatron-LM log --oneline -3
```

만약 merge가 로컬 수정 때문에 막히면, 해당 수정은 지우지 말고 stash로 보관합니다.

```bash
git status --short
git stash push -m "kt-local-before-session-restart"
git merge --ff-only FETCH_HEAD
```

## 3. W&B shadowing 복구와 설치

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

`NO_VERSION`, `init? False`, `path=None`, 또는 `ModuleNotFoundError`가 나오면 아래를 실행합니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

if [ -d wandb ]; then
  mv wandb "wandb_shadow_backup_$(date +%Y%m%d_%H%M%S)"
fi

PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --force-reinstall \
  --index-url https://pypi.org/simple \
  --disable-pip-version-check \
  --timeout 60 --retries 1 \
  wandb==0.19.11

python - <<'PY'
import wandb, inspect
print("wandb", wandb.__version__)
print("path", inspect.getfile(wandb))
print("init?", hasattr(wandb, "init"))
PY
```

주의: KT system pip config에는 `https://pypi.ngc.nvidia.com` extra index가 박혀 있어 설치가 멈출 수 있습니다. PyPI-only 설치에는 항상 `PIP_CONFIG_FILE=/dev/null`을 붙입니다.

## 4. W&B token 저장/로드

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
```

## 5. flash-attn / grouped_gemm 복구

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
PIP_CONFIG_FILE=/dev/null python -m pip install --user -U ninja packaging wheel

MAX_JOBS=8 PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --force-reinstall --no-cache-dir \
  --index-url https://pypi.org/simple \
  --no-build-isolation --no-deps --no-binary flash-attn \
  flash-attn==2.4.2

PIP_CONFIG_FILE=/dev/null \
python -m pip install --user --no-build-isolation --no-cache-dir \
  --index-url https://pypi.org/simple \
  git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

## 6. 최종 import / 문법 확인

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

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

## 7. 현재 자주 쓰는 실행 확인

G2 teacher-student router KD full-Wiki replay script가 있어야 최신 코드입니다.

```bash
ls -lh scripts/experiment/a100/run_g2_teacher_student_router_kd_fullwiki_mha.sh
bash -n scripts/experiment/a100/run_g2_teacher_student_router_kd_fullwiki_mha.sh
```

full Wiki train dataset 확인:

```bash
ls -lh data/wiki/train/train_text_document.*
```

학습 실행 예:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh
source ~/.config/wandb/env 2>/dev/null || true

LOG="g2_teacher_student_router_kd_fullwiki_$(date +%Y%m%d_%H%M%S).log"

nohup bash scripts/experiment/a100/run_g2_teacher_student_router_kd_fullwiki_mha.sh \
  > "$LOG" 2>&1 &

echo "$LOG"
tail -f "$LOG"
```

초기 run log에서 아래 문구가 보이면 teacher/student를 계속 로드한 joint-update 경로가 켜진 것입니다.

```bash
RUN_DIR=".local/weights/a100/mha/shared-router-granularity-qkvo/code/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-teacher-student-router-kd-fullwiki-kl0p1-1800"

grep -nE "Loaded full old shared-router hybrid teacher|Teacher-student router-memory joint update|teacher-student router memory KD" \
  "$RUN_DIR/logs/run.log" | head -40
```
