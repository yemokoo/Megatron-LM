#!/usr/bin/env python3
import json
import os
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
TEXT_KEY = "text"
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "20000"))
ROOT = Path(os.environ.get("OUTPUT_DIR", "/workspace/FLAME-MoE/.local/dataset/wikipedia-full/raw"))
STATE_PATH = ROOT / "_download_state.json"


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"kept": 0, "shard_idx": 0}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def flush_buffer(buf, shard_idx):
    out_path = ROOT / f"shard_{shard_idx:05d}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in buf:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    kept = int(state["kept"])
    shard_idx = int(state["shard_idx"])
    skipped = 0
    buf = []

    print(f"dataset={DATASET_NAME} config={DATASET_CONFIG}", flush=True)
    print(f"output_dir={ROOT}", flush=True)
    print(f"resume_kept={kept} shard_idx={shard_idx}", flush=True)

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)

    for ex in ds:
        text = ex.get(TEXT_KEY, "")
        if not text or not text.strip():
            continue

        if skipped < kept:
            skipped += 1
            if skipped % 100000 == 0:
                print(f"resume skip progress {skipped}/{kept}", flush=True)
            continue

        buf.append({"text": text})
        kept += 1

        if len(buf) >= SHARD_SIZE:
            out_path = flush_buffer(buf, shard_idx)
            print(f"wrote {out_path} kept={kept}", flush=True)
            shard_idx += 1
            buf = []
            save_state({"kept": kept, "shard_idx": shard_idx})

    if buf:
        out_path = flush_buffer(buf, shard_idx)
        print(f"wrote {out_path} kept={kept}", flush=True)
        shard_idx += 1

    save_state({"kept": kept, "shard_idx": shard_idx})
    print(f"download complete kept={kept} shards={shard_idx}", flush=True)


if __name__ == "__main__":
    main()
