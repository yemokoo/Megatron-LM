"""Measure per-optimizer-step training FLOPs of a real Megatron launch.

Installed by ``site/sitecustomize.py`` when FLOP_PROBE_STEPS is set.  The
production launcher, model construction, checkpoint loading, freezing,
replay/KD passes and grad accumulation all run exactly as in training; only
``train_step`` is wrapped.

Why this shape (and why the old in-loop attempt leaked):
  * FlopCounterMode is a TorchDispatchMode.  Holding one open across a whole
    run keeps per-op bookkeeping and dispatch interception alive for thousands
    of steps; memory grows until the run dies.  Here a *fresh* counter wraps
    exactly one train_step, is read, and is dropped.
  * Only FLOP_PROBE_STEPS steps are measured after FLOP_PROBE_WARMUP warmup
    steps; then the process exits (os._exit after a barrier) so no checkpoint
    is ever written.  The FLOPs of one global step do not depend on the
    weights, so the run can start from any checkpoint with the right
    architecture.
  * Counted = what the dispatcher executed: forward + backward matmuls,
    attention, replay/KD/teacher passes, routed experts actually used.
    Custom CUDA extensions (grouped_gemm) are invisible to the dispatcher --
    run probes with MOE_GROUPED_GEMM=0 / ATTN_LORA_GROUPED_GEMM=0 (same math,
    plain matmuls).

Env:
  FLOP_PROBE_STEPS   number of measured steps (required)
  FLOP_PROBE_WARMUP  steps to skip first (default 10)
  FLOP_PROBE_OUT     json path; ".rank{R}.json" is appended per rank
"""
import json
import os
import time

import torch

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:  # pragma: no cover
    FlopCounterMode = None

_STATE = {"calls": 0, "records": [], "installed": False}


def _rank():
    return int(os.environ.get("RANK", "0"))


def _log(msg, **_ignored):
    """Launchers may filter stdout; always also append to a side log next to the json."""
    print(msg, flush=True)
    try:
        root, _ = os.path.splitext(os.environ.get("FLOP_PROBE_OUT", "/tmp/flop_probe.json"))
        os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
        with open(f"{root}.rank{_rank()}.log", "a") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass


def _out_path():
    base = os.environ.get("FLOP_PROBE_OUT", "/tmp/flop_probe.json")
    root, _ = os.path.splitext(base)
    return f"{root}.rank{_rank()}.json"


def _args_snapshot():
    try:
        from megatron.training import get_args
        a = get_args()
    except Exception:
        return {}
    keys = [
        "micro_batch_size", "global_batch_size", "seq_length", "world_size",
        "data_parallel_size", "num_layers", "hidden_size", "ffn_hidden_size",
        "num_experts", "moe_ffn_hidden_size", "moe_router_topk",
        "moe_grouped_gemm", "attn_lora_grouped_gemm", "attn_lora_rank",
        "attn_lora_num_experts", "attn_lora_topk", "shared_router_hybrid_model",
        "moe_joint_replay_lm", "moe_joint_replay_micro_batch_size",
        "recompute_granularity", "transformer_impl", "train_iters", "iteration",
    ]
    return {k: getattr(a, k, None) for k in keys}


def _write(final):
    payload = {
        "schema_version": 1,
        "rank": _rank(),
        "warmup": int(os.environ.get("FLOP_PROBE_WARMUP", "10")),
        "measured_steps": int(os.environ["FLOP_PROBE_STEPS"]),
        "final": final,
        "args": _args_snapshot(),
        "records": _STATE["records"],
    }
    if _STATE["records"]:
        fl = [r["flops"] for r in _STATE["records"]]
        payload["flops_per_step_mean"] = sum(fl) / len(fl)
        payload["flops_per_step_min"] = min(fl)
        payload["flops_per_step_max"] = max(fl)
    path = _out_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def install(training_module):
    if _STATE["installed"]:
        return
    if FlopCounterMode is None:
        raise RuntimeError("torch.utils.flop_counter.FlopCounterMode unavailable")
    n = int(os.environ["FLOP_PROBE_STEPS"])
    warm = int(os.environ.get("FLOP_PROBE_WARMUP", "10"))
    if n < 1:
        raise ValueError("FLOP_PROBE_STEPS must be >= 1")
    original = training_module.train_step

    def probed_train_step(*args, **kwargs):
        _STATE["calls"] += 1
        c = _STATE["calls"]
        if c <= warm:
            return original(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        counter = FlopCounterMode(display=False)
        with counter:
            ret = original(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total = int(counter.get_total_flops())
        by_op = {}
        try:
            g = counter.get_flop_counts().get("Global", {})
            by_op = {str(k): int(v) for k, v in g.items()}
        except Exception:
            pass
        del counter
        rec = {
            "call": c,
            "flops": total,
            "sec": dt,
            "by_op": by_op,
        }
        if torch.cuda.is_available():
            rec["peak_mem_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
        _STATE["records"].append(rec)
        _log(f"[FLOP-PROBE] rank{_rank()} step{c} flops={total:.4e} "
              f"sec={dt:.2f} peak={rec.get('peak_mem_alloc_gb', 0):.1f}GB", flush=True)
        done = c >= warm + n
        _write(final=done)
        if done:
            fl = [r["flops"] for r in _STATE["records"]]
            _log(f"[FLOP-PROBE] DONE rank{_rank()} mean={sum(fl)/len(fl):.4e} "
                  f"min={min(fl):.4e} max={max(fl):.4e} -> {_out_path()}", flush=True)
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except Exception:
                pass
            os._exit(0)  # never reach any checkpoint save
        return ret

    training_module.train_step = probed_train_step
    _STATE["installed"] = True
    _log(f"[FLOP-PROBE] installed: warmup={warm} measure={n} out={_out_path()}", flush=True)
