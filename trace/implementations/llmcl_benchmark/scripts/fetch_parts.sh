#!/bin/bash
# Fetch the split (non-LFS) zip parts of YeMoKoo/TRACE directly from
# huggingface.co (the only reachable host in this session), reassemble the
# zip, and hand it to prepare_data.py.
#
# Assumes the parts were uploaded as regular <10MB files named trace_part_*.
set -e
REPO="YeMoKoo/TRACE"
DL="data/_dl"
ZIP="$DL/TRACE-Benchmark.zip"
mkdir -p "$DL"

echo "[1/4] listing parts via huggingface.co API ..."
PARTS=$(curl -s --max-time 30 "https://huggingface.co/api/datasets/$REPO" \
  | python -c "import sys,json;d=json.load(sys.stdin);print('\n'.join(sorted(f['rfilename'] for f in d['siblings'] if f['rfilename'].startswith('trace_part_'))))")
if [ -z "$PARTS" ]; then echo "  no trace_part_* files found in repo. Upload them first."; exit 1; fi
echo "$PARTS" | sed 's/^/    /'

echo "[2/4] downloading parts from huggingface.co (non-LFS, reachable) ..."
for p in $PARTS; do
  for attempt in 1 2 3 4 5; do
    curl -fsSL --max-time 120 -o "$DL/$p" \
      "https://huggingface.co/datasets/$REPO/resolve/main/$p" && break
    echo "    retry $attempt for $p"; sleep 5
  done
  echo "    got $p ($(stat -c%s "$DL/$p") bytes)"
done

echo "[3/4] reassembling zip ..."
cat $(echo "$PARTS" | sed "s#^#$DL/#") > "$ZIP"
echo "    $ZIP -> $(stat -c%s "$ZIP") bytes"
unzip -tq "$ZIP" >/dev/null && echo "    zip integrity OK" || { echo "    ZIP CORRUPT"; exit 1; }

echo "[4/4] normalizing into data/LLM-CL-Benchmark_5000 ..."
python scripts/prepare_data.py --zip "$ZIP" --out data/LLM-CL-Benchmark_5000
