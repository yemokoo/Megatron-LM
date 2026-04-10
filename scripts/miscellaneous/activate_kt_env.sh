#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

case ":${PATH:-}:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:${PATH:-}" ;;
esac

export PYTHONPATH="$REPO_DIR/Megatron-LM:${PYTHONPATH:-}"
export FLAME_MOE_KT_REPO_DIR="$REPO_DIR"

echo "KT runtime activated"
echo "repo:   $REPO_DIR"
echo "python: $(command -v python || true)"
echo "pip:    $(command -v pip || true)"
