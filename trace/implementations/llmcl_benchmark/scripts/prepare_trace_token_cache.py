#!/usr/bin/env python
"""Pre-sample fixed TRACE replay memory and pre-tokenize SLoRA train data."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import pyarrow.parquet as parquet
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.chat_templates import (  # noqa: E402
    ensure_llama31_chat_template,
    update_fingerprint_for_chat_template,
)
from utils.data.data_collator import (  # noqa: E402
    PreTokenizedSLoRATraceDataCollator,
    SLoRATraceDataCollator,
)

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--samples-per-task", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_source_fingerprint(model_path):
    digest = hashlib.sha256()
    found = False
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json"):
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            found = True
            digest.update(name.encode("utf-8") + b"\0")
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"no tokenizer assets found under {model_path}")
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(tokenizer_config_path):
        with open(tokenizer_config_path, encoding="utf-8") as handle:
            update_fingerprint_for_chat_template(digest, json.load(handle))
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def load_records(data_root, task):
    path = Path(data_root) / task / "train.json"
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid or empty TRACE train file: {path}")
    return path, records


def build_replay_manifest(args, tasks):
    payload = {
        "schema_version": 1,
        "sampling": "random_without_replacement",
        "base_seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "task_order": tasks,
        "tasks": {},
    }
    for task_index, task in enumerate(tasks):
        source_path, records = load_records(args.data_root, task)
        if args.samples_per_task > len(records):
            raise ValueError(
                f"cannot draw {args.samples_per_task} from {task}: {len(records)}")
        task_seed = args.seed + task_index * 1009
        generator = torch.Generator().manual_seed(task_seed)
        indices = torch.randperm(
            len(records), generator=generator)[:args.samples_per_task].tolist()
        payload["tasks"][task] = {
            "task_index": task_index,
            "source_path": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "source_samples": len(records),
            "seed": task_seed,
            "indices": indices,
            "indices_sha256": hashlib.sha256(
                ",".join(map(str, indices)).encode("utf-8")).hexdigest(),
        }
        print(f"[sample] {task}: {len(indices)} random unique records")
    write_json(args.replay_manifest, payload)
    print(f"[sample] manifest: {args.replay_manifest}")
    return payload


def encode_batch(tokenizer, records, max_length):
    texts = []
    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["answer"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    encoded = tokenizer(
        texts, truncation=True, max_length=max_length,
        padding=False, return_tensors=None)
    output = []
    for input_ids in encoded["input_ids"]:
        ids = list(input_ids)
        if (tokenizer.eos_token_id is not None
                and len(ids) < max_length
                and ids[-1] != tokenizer.eos_token_id):
            ids.append(tokenizer.eos_token_id)
        output.append(ids)
    return output


def validate_equivalence(tokenizer, records, cached, max_length, task):
    dynamic = SLoRATraceDataCollator(tokenizer, max_length=max_length)
    prepared = PreTokenizedSLoRATraceDataCollator(tokenizer)
    check_indices = sorted(set([0, len(records) // 2, len(records) - 1]))
    original = dynamic([records[index] for index in check_indices])
    restored = prepared([cached[index] for index in check_indices])
    for key in ("input_ids", "attention_mask", "labels"):
        if not torch.equal(original[key], restored[key]):
            raise AssertionError(f"{task}: cached {key} differs from dynamic collator")


def tokenize_task(args, tokenizer, task, records, task_dir):
    input_ids = []
    progress = tqdm(range(0, len(records), args.batch_size),
                    desc=f"tokenize {task}", unit="batch", dynamic_ncols=True)
    for start in progress:
        input_ids.extend(encode_batch(
            tokenizer, records[start:start + args.batch_size], args.max_length))
    cached = Dataset.from_dict({
        "source_index": list(range(len(records))),
        "input_ids": input_ids,
    })
    cached.save_to_disk(str(task_dir))
    validate_equivalence(tokenizer, records, cached, args.max_length, task)
    lengths = [len(ids) for ids in input_ids]
    return cached, cache_stats(args, task, cached, lengths)

def write_portable_parquet(cached, path):
    """Write without HF schema metadata for datasets 3.x/5.x compatibility."""
    table = cached.data.table.replace_schema_metadata(None)
    parquet.write_table(table, path, compression="zstd")




def cache_stats(args, task, cached, lengths):
    return {
        "path": task,
        "portable_path": f"{task}.parquet",
        "source_samples": len(cached),
        "cached_samples": len(cached),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "truncated_samples": sum(length == args.max_length for length in lengths),
        "equivalence_checks": 3,
    }


def main():
    args = parse_args()
    tasks = [task for task in args.tasks.split(",") if task]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay = build_replay_manifest(args, tasks)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True,
        local_files_only=True)
    if tokenizer.pad_token is None:
        if "llama" not in args.model_path.lower():
            raise ValueError("missing non-Llama pad token")
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    chat_template_source = ensure_llama31_chat_template(
        tokenizer, args.model_path)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    manifest = {
        "schema_version": 1,
        "format": "slora_chat_full",
        "label_scope": "all_nonpadding_tokens",
        "padding": "dynamic_right",
        "max_length": args.max_length,
        "system_prompt": SYSTEM_PROMPT,
        "chat_template_source": chat_template_source,
        "model_path": str(Path(args.model_path).resolve()),
        "tokenizer_fingerprint": tokenizer_source_fingerprint(args.model_path),
        "replay_manifest": str(Path(args.replay_manifest).resolve()),
        "replay_manifest_sha256": sha256_file(args.replay_manifest),
        "tasks": {},
    }
    started = time.monotonic()
    for task in tasks:
        _, records = load_records(args.data_root, task)
        expected = replay["tasks"][task]["source_samples"]
        if len(records) != expected:
            raise ValueError(f"source changed while caching {task}")
        task_dir = output_dir / task
        portable_path = output_dir / f"{task}.parquet"
        if task_dir.exists():
            if args.overwrite:
                shutil.rmtree(task_dir)
            else:
                cached = load_from_disk(str(task_dir))
                if len(cached) != len(records):
                    raise ValueError(
                        f"incomplete existing cache for {task}: {len(cached)}")
                validate_equivalence(
                    tokenizer, records, cached, args.max_length, task)
                lengths = [len(cached[index]["input_ids"])
                           for index in range(len(cached))]
                if not portable_path.is_file():
                    write_portable_parquet(cached, portable_path)
                manifest["tasks"][task] = cache_stats(
                    args, task, cached, lengths)
                write_json(output_dir / "manifest.json", manifest)
                print(f"[cache] {task}: reused {len(cached)} records")
                continue
        cached, stats = tokenize_task(args, tokenizer, task, records, task_dir)
        write_portable_parquet(cached, portable_path)
        manifest["tasks"][task] = stats
        write_json(output_dir / "manifest.json", manifest)
        print(f"[cache] {task}: {stats['cached_samples']} records, "
              f"mean={stats['mean_tokens']:.1f}, "
              f"truncated={stats['truncated_samples']}")
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["complete"] = True
    write_json(output_dir / "manifest.json", manifest)
    print(f"[done] cache: {output_dir}")
    print(f"[done] elapsed: {manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
