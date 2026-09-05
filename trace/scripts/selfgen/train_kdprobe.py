#!/usr/bin/env python3
"""KD-init probe: run ONE round's expansion KD-init, then score the expanded
model against its teacher on REAL held-out old-task data, and exit before the
joint (primary + replay) phase.

Question it answers: does KD-init on self-generated old-task records leave the
expanded model as faithful to the teacher (on real data) as KD-init on the real
memory does?  KD-init trains the NEW expert + router (experts of old tasks are
frozen), so this is the one place generated data shapes expert weights.

env:
  KDPROBE_SOURCE   real | gen         (gen additionally needs SELFGEN_ROOT)
  SELFGEN_ROOT     <run>/gen/round_<k> for the gen arm (train_selfgen patch)
  SELFGEN_CURRENT_TASK  task being added this round (e.g. NumGLUE-ds)
  KDPROBE_OUT      json path for the scores
  KDPROBE_EVAL_N   held-out records per old task (default 128, from test.json)

Metrics per old task (student = expanded model after KD-init, teacher = same
model restricted to the old experts, i.e. exactly what KD-init distilled):
  kl_all      mean per-token KL(teacher||student) over non-pad tokens (T=1),
              the KD objective itself, but on real data
  ce_ans_*    answer-span causal-LM CE for teacher and student
  acc_*       first-answer-token letter accuracy vs ground truth (label tasks)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMPL = REPO / "implementations" / "llmcl_benchmark"
sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(HERE))

from model.Ours_LoRA_MoE_V3 import (                       # noqa: E402
    Ours_LoRA_MoE_V3_New, limit_v3_experts, shared_router_layers)
from utils.data.data_collator import SLoRATraceDataCollator  # noqa: E402

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
LETTERS = {"C-STANCE": "ABC", "FOMC": "ABC", "ScienceQA": "ABCDE"}
DATA = Path("/data2/seonghyeonnoh/LLM-continual-learning-data/"
            "flamedata2.data2-verified-backup/trace")


def _score(self, current_task, out_path, eval_n):
    args = self.args
    model = self.raw_model
    model.eval()
    device = next(model.parameters()).device
    layers = shared_router_layers(model)
    n_experts = layers[0].num_experts
    old_count = n_experts - args.experts_per_task
    old_tasks = TASKS[:TASKS.index(current_task)]
    if len(old_tasks) != old_count:
        raise RuntimeError(
            f"expected {len(old_tasks)} old experts for {current_task}, model has {old_count}")
    tok = self.tokenizer
    coll = SLoRATraceDataCollator(tok, max_length=int(args.max_train_len or 1024),
                                  label_scope="answer")
    rank = int(getattr(args, "global_rank", 0) or 0)
    report = {"current_task": current_task, "old_expert_count": old_count,
              "source": os.environ.get("KDPROBE_SOURCE", "?"),
              "selfgen_root": os.environ.get("SELFGEN_ROOT", ""), "tasks": {}}
    for task in old_tasks:
        recs = json.load((DATA / task / "test.json").open())[:eval_n]
        letters = LETTERS.get(task)
        letter_ids = ([tok.encode(l, add_special_tokens=False)[0] for l in letters]
                      if letters else None)
        kl_sum = 0.0; kl_n = 0
        ce_s = 0.0; ce_t = 0.0; ce_n = 0
        acc_s = 0; acc_t = 0; acc_n = 0
        for r in recs:
            ids, label_start = coll._encode(r)
            if label_start >= len(ids):
                continue
            x = torch.tensor([ids], device=device)
            att = torch.ones_like(x)
            with torch.no_grad():
                s_logits = model(input_ids=x, attention_mask=att, use_cache=False).logits[0].float()
                with limit_v3_experts(model, old_count):
                    t_logits = model(input_ids=x, attention_mask=att, use_cache=False).logits[0].float()
            # KD objective on real tokens (all positions, T=1)
            t_lp = F.log_softmax(t_logits, -1); s_lp = F.log_softmax(s_logits, -1)
            kl_sum += float((t_lp.exp() * (t_lp - s_lp)).sum()); kl_n += int(x.shape[1])
            # answer-span CE (positions label_start-1 .. end-1 predict answer tokens)
            tgt = x[0, label_start:]
            ce_s += float(F.cross_entropy(s_logits[label_start - 1:-1], tgt, reduction="sum"))
            ce_t += float(F.cross_entropy(t_logits[label_start - 1:-1], tgt, reduction="sum"))
            ce_n += int(tgt.numel())
            if letter_ids:
                gt = r["answer"].strip()[:1]
                if gt in letters:
                    acc_n += 1
                    acc_s += int(letters[int(s_logits[label_start - 1, letter_ids].argmax())] == gt)
                    acc_t += int(letters[int(t_logits[label_start - 1, letter_ids].argmax())] == gt)
        report["tasks"][task] = {
            "n": len(recs), "kl_all": kl_sum / max(kl_n, 1),
            "ce_ans_teacher": ce_t / max(ce_n, 1), "ce_ans_student": ce_s / max(ce_n, 1),
            **({"acc_teacher": acc_t / max(acc_n, 1), "acc_student": acc_s / max(acc_n, 1)}
               if letter_ids else {})}
        if rank == 0:
            print(f"[kdprobe] {task}: " + json.dumps(report["tasks"][task]), flush=True)
    if rank == 0:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out_path, "w"), indent=1)
        print(f"[kdprobe] wrote {out_path}", flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    raise SystemExit(0)


def install_probe():
    current = os.environ.get("SELFGEN_CURRENT_TASK", "").strip()
    out_path = os.environ.get("KDPROBE_OUT", "").strip()
    eval_n = int(os.environ.get("KDPROBE_EVAL_N", "128"))
    if not current or not out_path:
        raise SystemExit("SELFGEN_CURRENT_TASK and KDPROBE_OUT are required")

    def _joint(self, *a, **k):
        _score(self, current, out_path, eval_n)

    def _primary(self, *a, **k):
        _score(self, current, out_path, eval_n)

    Ours_LoRA_MoE_V3_New._run_v2_joint_epochs = _joint
    Ours_LoRA_MoE_V3_New._run_v3_primary_epochs = _primary
    print(f"[kdprobe] post-KD scoring installed; will exit before joint phase "
          f"(task={current}, out={out_path})", flush=True)


def main():
    source = os.environ.get("KDPROBE_SOURCE", "real").strip()
    if source == "gen":
        from train_selfgen import install_selfgen_replay
        root = os.environ.get("SELFGEN_ROOT", "").strip()
        if not root:
            raise SystemExit("gen arm needs SELFGEN_ROOT")
        install_selfgen_replay(root)
    elif source != "real":
        raise SystemExit(f"KDPROBE_SOURCE must be real|gen, got {source!r}")
    install_probe()
    import runpy
    sys.argv[0] = str(IMPL / "training" / "main_Ours_LoRA_MoE.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
