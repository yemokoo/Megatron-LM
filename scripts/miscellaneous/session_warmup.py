#!/usr/bin/env python3
"""Lightweight GPU session warmup for ephemeral KT sessions."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep selected GPUs lightly busy.")
    parser.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="Comma-separated GPU ids to use. Defaults to CUDA_VISIBLE_DEVICES or 0.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=2048,
        help="Square matmul size per cycle.",
    )
    parser.add_argument(
        "--compute-seconds",
        type=float,
        default=1.0,
        help="Approximate compute time per cycle.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=4.0,
        help="Sleep between compute cycles.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Tensor dtype for the dummy workload.",
    )
    return parser.parse_args()


def worker(local_rank: int, matrix_size: int, compute_seconds: float, sleep_seconds: float, dtype_name: str) -> None:
    import torch

    torch.cuda.set_device(local_rank)
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda", local_rank)

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)

    print(
        f"[warmup] local_rank={local_rank} device={torch.cuda.get_device_name(device)} "
        f"dtype={dtype_name} matrix={matrix_size}",
        flush=True,
    )

    while True:
        started = time.time()
        while time.time() - started < compute_seconds:
            c = a @ b
            a = b
            b = c
        torch.cuda.synchronize(device)
        time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"PyTorch import failed: {exc}") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; cannot run warmup.")

    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not devices:
        raise SystemExit("No devices selected.")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    print(
        "[warmup] starting on visible GPUs "
        f"{os.environ['CUDA_VISIBLE_DEVICES']} with {len(devices)} worker(s)",
        flush=True,
    )
    print("[warmup] stop with Ctrl-C after clone/install finishes", flush=True)

    procs: list[mp.Process] = []
    try:
        for local_rank in range(len(devices)):
            proc = mp.Process(
                target=worker,
                args=(
                    local_rank,
                    args.matrix_size,
                    args.compute_seconds,
                    args.sleep_seconds,
                    args.dtype,
                ),
            )
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        print("\n[warmup] stopping workers...", flush=True)
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)


if __name__ == "__main__":
    main()
