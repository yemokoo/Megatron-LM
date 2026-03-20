#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

tmux kill-session -t wiki-full-tokenize 2>/dev/null || true
tmux kill-session -t code-full-tokenize 2>/dev/null || true

tmux new -d -s wiki-full-tokenize "bash $PROJECT_ROOT/scripts/dataset/tokenize_wikipedia_full_local.sh"
tmux new -d -s code-full-tokenize "bash $PROJECT_ROOT/scripts/dataset/tokenize_code_full_local.sh"

tmux ls | grep -E 'wiki-full-tokenize|code-full-tokenize'
