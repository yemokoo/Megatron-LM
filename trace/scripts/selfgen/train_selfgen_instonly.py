#!/usr/bin/env python3
"""Instruction-only self-generated replay.

Same as train_selfgen.py (generated records replace the real replay memory) with ONE
change: for generated records the replay causal-LM loss covers the instruction tokens
only -- labels from the answer start onward are -100.  The router-only replay phase
therefore corrects routing on the prompt/instruction span and never fits the generated
answer; KD-init is unaffected (it uses the nonpad attention-mask scope, not labels), and
the current task's real data carries no `answer_start`, so it keeps full-sequence labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_selfgen as base                                   # noqa: E402
from utils.data.data_collator import (                         # noqa: E402
    PreTokenizedSLoRATraceDataCollator, SLoRATraceDataCollator)

_orig_getitem = base.GeneratedRecords.__getitem__


def _answer_encoder(self):
    # the training encoder is label_scope "full" (label_start 0); the answer start
    # comes from an "answer"-scoped twin over the same tokenizer/max_length
    enc = getattr(self, "_answer_enc", None)
    if enc is None:
        enc = SLoRATraceDataCollator(self._encoder.tokenizer, max_length=self._encoder.max_length,
                                     label_scope="answer")
        self._answer_enc = enc
    return enc


def _getitem(self, i):
    item = _orig_getitem(self, i)
    if self._encoder is not None:
        ids, label_start = _answer_encoder(self)._encode(self.records[i])
        if len(ids) != len(item["input_ids"]):
            raise RuntimeError("answer-scoped encoding length differs from the full encoding")
        item = dict(item, answer_start=int(label_start))
    return item


base.GeneratedRecords.__getitem__ = _getitem


def _mask_after_answer_start(batch, instances):
    labels = batch["labels"]
    n_masked = 0
    for row, inst in enumerate(instances):
        start = inst.get("answer_start") if isinstance(inst, dict) else None
        if start is None:
            continue
        labels[row, int(start):] = -100
        n_masked += 1
    if n_masked and labels.ne(-100).sum().item() == 0:
        raise RuntimeError("instruction-only mask left no supervised tokens in the batch")
    return batch


def _wrap(cls):
    orig = cls.__call__

    def __call__(self, instances, *a, **k):
        return _mask_after_answer_start(orig(self, instances, *a, **k), instances)
    cls.__call__ = __call__


_wrap(PreTokenizedSLoRATraceDataCollator)
_wrap(SLoRATraceDataCollator)
print("[selfgen-instonly] generated replay records: labels masked from answer_start "
      "(instruction tokens only)", flush=True)

if __name__ == "__main__":
    base.main()
