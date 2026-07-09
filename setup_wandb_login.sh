#!/usr/bin/env bash
# W&B API key 저장 + 로그인 (wandb 패키지는 이미 설치됨)
set -e
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

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

source ~/.config/wandb/env
python - <<'PY'
import os, wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
print("wandb login ok")
PY
