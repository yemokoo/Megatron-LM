#!/usr/bin/env bash
# One-time setup for the general-ability eval (MMLU / GSM8K / PIQA, lm-evaluation-harness 0.4.8).
#   LMEVAL_VENV=<dir> LMEVAL_DATA=<dir> bash scripts/lmeval/setup_lmeval.sh
# 1) venv with the exact package set of the original runs (requirements-lmeval.txt, Python 3.10,
#    torch 2.4.1+cu124); uses uv when available, else python3.10 -m venv + pip
# 2) the offline dataset cache the table was scored with (assets/lmeval_datasets.tar.gz, 10 MB:
#    hails/mmlu_no_train, gsm8k main, piqa), unpacked to $LMEVAL_DATA/lmeval_datasets.
#    LMEVAL_DOWNLOAD=1 instead fetches the same datasets from the Hugging Face Hub.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${LMEVAL_VENV:?set LMEVAL_VENV=<venv dir>}
DATA=${LMEVAL_DATA:?set LMEVAL_DATA=<data dir>}

if [ ! -x "$VENV/bin/python" ]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.10 "$VENV"
    uv pip install --python "$VENV/bin/python" -r "$HERE/requirements-lmeval.txt" --index-strategy unsafe-best-match
  else
    python3.10 -m venv "$VENV"
    "$VENV/bin/pip" install -r "$HERE/requirements-lmeval.txt"
  fi
fi
"$VENV/bin/python" -c "import importlib.metadata as m; print('lm_eval', m.version('lm_eval'), 'datasets', m.version('datasets'))"

mkdir -p "$DATA"
if [ "${LMEVAL_DOWNLOAD:-0}" = 1 ]; then
  HF_DATASETS_CACHE="$DATA/lmeval_datasets" HF_DATASETS_TRUST_REMOTE_CODE=1 "$VENV/bin/python" - <<'PY'
from lm_eval.tasks import TaskManager, get_task_dict
get_task_dict(["mmlu", "gsm8k", "piqa"], TaskManager())   # instantiating the tasks downloads them
print("datasets downloaded")
PY
else
  (cd "$HERE/assets" && sha256sum -c <(echo "$(cat lmeval_datasets.tar.gz.sha256)  lmeval_datasets.tar.gz"))
  tar -xzf "$HERE/assets/lmeval_datasets.tar.gz" -C "$DATA"
fi
echo "done: LMEVAL_VENV=$VENV  HF_DATASETS_CACHE=$DATA/lmeval_datasets"
