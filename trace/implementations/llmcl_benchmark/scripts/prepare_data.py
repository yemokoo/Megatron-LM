#!/usr/bin/env python
"""
Unzip the TRACE benchmark archive (downloaded from Google Drive, re-hosted on
your HF dataset repo) and normalize it into the layout vllm_eval.py expects:

    <out>/C-STANCE/test.json
    <out>/FOMC/test.json
    ...  (8 tasks)

It does NOT assume a fixed nesting depth. It walks the extracted tree, finds
every folder whose name is one of the 8 TRACE tasks AND that contains a
test.json, and links it into <out>. If the archive ships multiple size
variants (e.g. LLM-CL-Benchmark_500 / _1000 / _5000), pass --prefer to pick one
(default: 5000, the paper's setting).

Usage:
    python scripts/prepare_data.py --zip /path/downloaded.zip --out ./data/LLM-CL-Benchmark_5000
    # or, if already extracted:
    python scripts/prepare_data.py --src /path/extracted_dir --out ./data/LLM-CL-Benchmark_5000
"""
import argparse
import json
import os
import shutil
import zipfile

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", help="Path to the downloaded .zip archive.")
    g.add_argument("--src", help="Path to an already-extracted directory.")
    p.add_argument("--out", required=True, help="Normalized output data_path.")
    p.add_argument("--prefer", default="5000",
                   help="If multiple size variants exist, prefer the path "
                        "containing this substring. Default: 5000.")
    p.add_argument("--copy", action="store_true",
                   help="Copy task folders instead of symlinking.")
    return p.parse_args()


def find_task_dirs(root):
    """Return {task: [candidate_dir, ...]} for dirs named as a task w/ test.json."""
    found = {t: [] for t in TASKS}
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base in found and "test.json" in filenames:
            found[base].append(dirpath)
    return found


def pick(candidates, prefer):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer by the VARIANT folder (the task's parent dir), not any substring of
    # the full path -- the out dir itself may contain the prefer string.
    preferred = [c for c in candidates
                 if os.path.basename(os.path.dirname(c)).endswith("_" + prefer)]
    return (preferred or candidates)[0]


def count_samples(test_json):
    try:
        with open(test_json, "r", encoding="utf-8") as f:
            return len(json.load(f))
    except Exception as e:
        return f"?({e})"


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.zip:
        extract_dir = os.path.join(args.out, "_extract")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[unzip] {args.zip} -> {extract_dir}")
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(extract_dir)
        root = extract_dir
    else:
        root = args.src

    found = find_task_dirs(root)

    print("\n== task detection ==")
    missing, resolved = [], {}
    for t in TASKS:
        chosen = pick(found[t], args.prefer)
        if chosen is None:
            missing.append(t)
            print(f"  [MISSING] {t}")
        else:
            resolved[t] = chosen
            extra = f"  (+{len(found[t])-1} other variant(s))" if len(found[t]) > 1 else ""
            print(f"  [ok] {t:12s} <- {chosen}{extra}")

    print("\n== linking into out ==")
    for t, srcdir in resolved.items():
        dst = os.path.join(args.out, t)
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        if args.copy:
            shutil.copytree(srcdir, dst)
        else:
            os.symlink(os.path.abspath(srcdir), dst)
        n = count_samples(os.path.join(dst, "test.json"))
        print(f"  {t:12s} test.json n={n}")

    if missing:
        print(f"\n!! MISSING {len(missing)} task(s): {missing}")
        print("   -> inspect the archive layout and rerun; the folder names must "
              "match exactly: " + ", ".join(TASKS))
    else:
        print(f"\nAll 8 tasks ready under: {os.path.abspath(args.out)}")
        print("Set DATA to this path in scripts/run_all_models.sh")


if __name__ == "__main__":
    main()
