#!/usr/bin/env bash
# GitHub 토큰을 git credential store + 환경변수로 영구 저장 (unset 안 함)
set -e
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

git config --global credential.helper store
printf "protocol=https\nhost=github.com\n\n" | git credential reject || true

echo -n "GitHub token: "
stty -echo
read GITHUB_TOKEN
stty echo
echo

# 1) git credential store에 영구 저장 (git이 자동 사용)
printf "protocol=https\nhost=github.com\nusername=x-access-token\npassword=%s\n\n" "$GITHUB_TOKEN" | git credential approve

# 2) 환경변수로도 계속 쓰도록 파일에 저장
mkdir -p ~/.config/github
cat > ~/.config/github/env <<EOF
export GITHUB_TOKEN="$GITHUB_TOKEN"
export GH_TOKEN="$GITHUB_TOKEN"
EOF
chmod 600 ~/.config/github/env

# 3) 새 세션에서도 자동 로드
grep -q 'github/env' ~/.bashrc || echo 'source ~/.config/github/env 2>/dev/null' >> ~/.bashrc

echo
echo "=== 저장 완료. 인증 확인: ==="
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/yemokoo/LLM-continual-learning.git HEAD
echo
echo ">> 현재 셸에도 반영하려면:  source ~/.config/github/env"
