#!/usr/bin/env python3
"""train_residual_v3_split.py + bos_guard.py: same split-gradient residual
recipe (residual-from-task-0, expert branch masks the residual, router-FT
branch trains every row on replay+BoS+primary-reuse), with one addition --
at every layer except the last, a BOS-token position is always routed to the
residual, so its hidden state entering the last layer is architecturally
identical to the plain backbone regardless of round/checkpoint.  See
bos_guard.py for the rationale and mechanism.

Run exactly like train_residual_v3_split.py (same argv, same env vars).
BOS_GUARD_HEADER=1 additionally guards the fixed chat-template header of every
document (body: residual at all layers; last header token: layers 1..L-1),
see bos_guard.header_masks.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMPL = REPO / "implementations" / "llmcl_benchmark"
for path in (IMPL, HERE, REPO / "scripts" / "selfgen"):
    sys.path.insert(0, str(path))

import train_residual_v3_split as SPLIT   # noqa: E402  installs the split-gradient patches
import bos_guard                          # noqa: E402
from model import Ours_LoRA_MoE_V3 as V3  # noqa: E402

_prev_add = V3.add_v3_experts   # SPLIT's add_experts_from_residual (growth + row copy)


def add_experts_and_guard(model, count):
    _prev_add(model, count)
    bos_guard.install_bos_guard(model)


V3.add_v3_experts = add_experts_and_guard


def _configure_header_guard():
    if os.environ.get("BOS_GUARD_HEADER", "0") != "1":
        return
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO / "scripts" / "bos_token"))
    from train_bos_token import guard_header_spec, chat_header_ids
    base = sys.argv[sys.argv.index("--model_name_or_path") + 1]
    tok = AutoTokenizer.from_pretrained(base, use_fast=False, trust_remote_code=True, local_files_only=True)
    decision = os.environ.get("BOS_GUARD_DECISION", "nn")
    bos_guard.HEADER_IDS, bos_guard.HEADER_ALL_FULL = guard_header_spec(tok, decision)
    print(f"[bos_guard] header guard ON: {len(bos_guard.HEADER_IDS)} header tokens, decision={decision}, "
          f"all_full={bos_guard.HEADER_ALL_FULL}", flush=True)
    if os.environ.get("HEADER_NO_LOSS", "0") == "1":
        # every training doc that starts with the full 37-token chat header (through "\n\n") gets
        # labels only from the first user-content token on: the header is never a prediction target
        from utils.data import data_collator
        data_collator.HEADER_LABEL_MASK_IDS = chat_header_ids(tok, decision="nn")
        print(f"[bos_guard] header loss OFF: first {len(data_collator.HEADER_LABEL_MASK_IDS)} tokens unlabeled", flush=True)


def main():
    _configure_header_guard()
    root = os.environ.get("SELFGEN_ROOT", "").strip()
    if root:
        import train_selfgen
        train_selfgen.install_selfgen_replay(root)
    SPLIT.install_bos_memory()
    import runpy
    os.chdir(IMPL)
    script = str(IMPL / "training" / "main_Ours_LoRA_MoE.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
