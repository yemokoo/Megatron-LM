#!/usr/bin/env bash

set -euo pipefail

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_BASE="/home/work/Agent_HJ/30_flame_agent"
REPO_DIR="$PROJECT_BASE/LLM-continual-learning"
WHEEL_DIR="/home/work/Agent_HJ/wheels"
DATA_DIR="$REPO_DIR/data"



# ── 1. Git 설정 ──────────────────────────────────────────────────────────────
echo "==> [1/8] Git 설정"
git config --global user.email "yemokoo@gmail.com"
git config --global user.name "yemokoo"

# ── 2. Repo 클론 (이미 있으면 스킵) ──────────────────────────────────────────
echo "==> [2/8] Repo 클론"
if [ -d "$REPO_DIR/.git" ]; then
  echo "  repo 이미 존재 → git pull"
  cd "$REPO_DIR"
  git pull --no-recurse-submodules || true
else
  mkdir -p "$PROJECT_BASE"
  cd "$PROJECT_BASE"
  git clone --recursive -b slurm https://github.com/yemokoo/LLM-continual-learning.git
  cd "$REPO_DIR"
fi

# ── 3. Submodule 설정 ────────────────────────────────────────────────────────
echo "==> [3/8] Submodule 설정"
# Megatron-LM URL을 yemokoo fork으로 (이미 .gitmodules에서 바뀌었지만 안전하게)
git config submodule.Megatron-LM.url https://github.com/yemokoo/Megatron-LM.git

# Megatron-LM submodule fetch & checkout
cd "$REPO_DIR/Megatron-LM"
git fetch https://github.com/yemokoo/Megatron-LM.git multi-nodes 2>/dev/null || true
git checkout f504c822 2>/dev/null || true
cd "$REPO_DIR"

# 나머지 submodule
git submodule update --init apex
git submodule update --init TransformerEngine
git submodule update --init lm-evaluation-harness

# TE 내부 submodule (cuDNN frontend)
cd "$REPO_DIR/TransformerEngine"
git submodule update --init --recursive
cd "$REPO_DIR"

# ── 4. PYTHONPATH & bashrc 설정 ──────────────────────────────────────────────
echo "==> [4/8] PYTHONPATH 설정"
PYTHONPATH_LINE="export PYTHONPATH=$REPO_DIR/Megatron-LM:\${PYTHONPATH:-}"
if ! grep -q "Megatron-LM" ~/.bashrc 2>/dev/null; then
  echo "$PYTHONPATH_LINE" >> ~/.bashrc
fi
export PYTHONPATH="$REPO_DIR/Megatron-LM:${PYTHONPATH:-}"

# ── 5. pip 업그레이드 + PyTorch 설치 ─────────────────────────────────────────
echo "==> [5/8] PyTorch 및 의존성 설치"
PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# PyTorch (PyPI 기본 = cu124, download.pytorch.org 차단이라 이걸 써야 함)
"$PYTHON_BIN" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Megatron requirements (modelopt, resiliency-ext 제외 - torch>=2.6 요구)
"$PYTHON_BIN" -c "
from pathlib import Path
skip_prefixes = ['torch', 'nvidia-modelopt', 'nvidia-resiliency-ext']
req_path = Path('Megatron-LM/requirements/pytorch_24.10/requirements.txt')
lines = []
for line in req_path.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        continue
    if any(stripped.lower().startswith(s) for s in skip_prefixes):
        continue
    lines.append(line)
Path('/tmp/_megatron_reqs.txt').write_text('\n'.join(lines))
"
"$PYTHON_BIN" -m pip install -r /tmp/_megatron_reqs.txt
rm -f /tmp/_megatron_reqs.txt

# 추가 패키지
"$PYTHON_BIN" -m pip install transformers pybind11 tensorboard "numpy==1.26.4" wandb

# ── 6. Apex 설치 (wheel 캐시 우선) ───────────────────────────────────────────
echo "==> [6/8] Apex 설치"
APEX_WHL="$(find "$WHEEL_DIR" -name 'apex-0.1-cp310-cp310-linux_x86_64.whl' 2>/dev/null | head -1)"
if [ -n "$APEX_WHL" ] && [ -f "$APEX_WHL" ]; then
  echo "  캐시 wheel 사용: $APEX_WHL"
  "$PYTHON_BIN" -m pip install "$APEX_WHL"
else
  echo "  소스 빌드 (~15분)"
  pushd "$REPO_DIR/apex" >/dev/null
  # CUDA 12.4 vs 12.5 버전 체크 우회 (ABI 호환, 안전)
  sed -i '40,48d' setup.py
  sed -i '39a\    pass' setup.py
  "$PYTHON_BIN" -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./
  # wheel 캐싱
  mkdir -p "$WHEEL_DIR"
  "$PYTHON_BIN" -m pip wheel --no-build-isolation --no-deps \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    -w "$WHEEL_DIR" . 2>/dev/null || echo "  (wheel 캐싱 스킵)"
  popd >/dev/null
fi

# ── 7. TransformerEngine 설치 (wheel 캐시 우선) ──────────────────────────────
echo "==> [7/8] TransformerEngine 설치"
TE_WHL="$(find "$WHEEL_DIR" -name 'transformer_engine-*.whl' 2>/dev/null | head -1)"
if [ -n "$TE_WHL" ] && [ -f "$TE_WHL" ]; then
  echo "  캐시 wheel 사용: $TE_WHL"
  "$PYTHON_BIN" -m pip install "$TE_WHL"
else
  echo "  소스 빌드 (~20분)"
  pushd "$REPO_DIR/TransformerEngine" >/dev/null
  export NVTE_FRAMEWORK=pytorch
  export MAX_JOBS="$(nproc)"
  "$PYTHON_BIN" -m pip install --no-build-isolation .
  # wheel 캐싱
  mkdir -p "$WHEEL_DIR"
  "$PYTHON_BIN" -m pip wheel --no-build-isolation --no-deps -w "$WHEEL_DIR" . 2>/dev/null || echo "  (wheel 캐싱 스킵)"
  popd >/dev/null
fi

# ── 8. flash-attn 설치 (--no-deps 필수! torch 덮어쓰기 방지) ────────────────
echo "==> [8/8] flash-attn 설치"
"$PYTHON_BIN" -m pip install flash-attn==2.4.2 --no-build-isolation --no-deps

# ── torch 버전 검증 ──────────────────────────────────────────────────────────
INSTALLED_TORCH="$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)')"
if [[ "$INSTALLED_TORCH" != "2.5.1"* ]]; then
  echo "WARNING: torch 버전 불일치 ($INSTALLED_TORCH), 복구 중..."
  "$PYTHON_BIN" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
fi

# ── Megatron-LM dot_product_attention 패치 (view → reshape) ──────────────────
echo "==> Megatron-LM attention 패치"
DPA_FILE="$REPO_DIR/Megatron-LM/megatron/core/transformer/dot_product_attention.py"
sed -i 's/key = key\.view(/key = key.reshape(/g' "$DPA_FILE"
sed -i 's/value = value\.view(value\.size(0)/value = value.reshape(value.size(0)/g' "$DPA_FILE"

# ── 데이터셋 다운로드 (HF 접근 가능하면) ─────────────────────────────────────
if [ ! -d "$DATA_DIR/wiki" ]; then
  echo "==> 데이터셋 다운로드 시도"
  until huggingface-cli download YeMoKoo/flamedata --repo-type dataset --local-dir "$DATA_DIR"; do
    echo "  끊김 - 10초 후 재시도..."
    sleep 10
  done
else
  echo "==> 데이터셋 이미 존재"
fi

# ── 스모크 테스트 ────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  KT 24.07 전체 세션 세팅 완료!"
echo "============================================"
echo ""
"$PYTHON_BIN" -c "import torch; print(f'  torch:      {torch.__version__} (CUDA {torch.version.cuda})')"
"$PYTHON_BIN" -c "import apex; print('  apex:       ok')"
"$PYTHON_BIN" -c "import transformer_engine.pytorch; print('  TE pytorch: ok')"
"$PYTHON_BIN" -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')"
echo ""
echo "경로 요약:"
echo "  repo:     $REPO_DIR"
echo "  data:     $DATA_DIR"
echo "  wheels:   $WHEEL_DIR"
echo ""
echo "학습 실행 예시:"
echo "  cd $REPO_DIR"
echo "  PYTHONPATH=\$PWD/Megatron-LM:\${PYTHONPATH:-} \\"
echo "  CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \\"
echo "  MICRO_BATCH_SIZE=16 GLOBAL_BATCH_SIZE=2304 \\"
echo "  TRAIN_ITERS=1800 SAVE_INTERVAL=300 EVAL_INTERVAL=100 \\"
echo "  SEQ_LENGTH=512 \\"
echo "  TRAIN_DATASET=$DATA_DIR/wiki/train \\"
echo "  PROBE_DATASET=$DATA_DIR/wiki/test \\"
echo "  bash scripts/experiment/pretrain_wiki_dense_local_bf16.sh"