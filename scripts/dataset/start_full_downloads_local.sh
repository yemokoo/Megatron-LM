#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p .logs

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is required."
    exit 1
fi

if tmux has-session -t wiki-full 2>/dev/null; then
    echo "tmux session wiki-full already exists"
else
    tmux new-session -d -s wiki-full \
        "cd '$PROJECT_ROOT' && python scripts/dataset/download_wikipedia_full_local.py 2>&1 | tee .logs/wiki_full_download.log"
    echo "started tmux session wiki-full"
fi

if tmux has-session -t code-full 2>/dev/null; then
    echo "tmux session code-full already exists"
else
    tmux new-session -d -s code-full \
        "cd '$PROJECT_ROOT' && python scripts/dataset/download_code_full_local.py 2>&1 | tee .logs/code_full_download.log"
    echo "started tmux session code-full"
fi

echo "wiki log: $PROJECT_ROOT/.logs/wiki_full_download.log"
echo "code log: $PROJECT_ROOT/.logs/code_full_download.log"
echo "attach wiki: tmux attach -t wiki-full"
echo "attach code: tmux attach -t code-full"
