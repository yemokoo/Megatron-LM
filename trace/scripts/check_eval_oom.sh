#!/bin/bash
# Scan every evaluation log for OOM and for cells that were skipped because of
# one.  SPARSE15_CONTINUE_ON_CELL_ERROR=1 keeps a failed cell from stopping the
# sweep, which is what we want for a long unattended run -- but it also means an
# OOM shows up as a quietly missing result rather than a crash, so something has
# to go looking for it.
set -uo pipefail
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace
status=0

echo "=== OOM in evaluation logs ==="
# Per-cell logs live beside the results, not in the lane logs, which is where
# the first OOM hid: the sweep skipped the cell and carried on, so nothing in
# the lane output said anything had gone wrong.
hits=$(grep -rlE 'CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
  "${R}"/*/*_st_top1/evaluation "${R}/logs" "${R}/slora_pre_upstream" \
  "${R}/series_lanes" 2>/dev/null | sort -u)
if [[ -n "${hits}" ]]; then
  status=1
  while read -r f; do
    [[ -z "${f}" ]] && continue
    echo "  ${f}"
    grep -oE 'Tried to allocate [^;]+|GPU [0-9]+ has a total capacity[^;]+' "${f}" 2>/dev/null | tail -2 | sed 's/^/      /'
  done <<< "${hits}"
else
  echo "  none"
fi

echo
echo "=== per-run cell counts (15 expected) ==="
for d in "${R}"/v3_*/*_st_top1; do
  [[ -d "${d}/evaluation" ]] || continue
  # Exclude shard partials: a sharded cell writes one file per shard next to
  # the merged result, so counting them makes an unfinished cell look done.
  n=$(find "${d}/evaluation" -name 'results-*.json' ! -name '*shard*' \
        2>/dev/null | wc -l)
  printf '  %-46s %2s/15\n' "$(basename "$(dirname "${d}")")" "${n}"
  [[ "${n}" -lt 15 ]] && status=1
done
for f in "${R}"/slora_pre_upstream/llama31/pre/evaluation/failed_cells.shard*.tsv; do
  [[ -f "${f}" ]] || continue
  n=$(( $(wc -l < "${f}") - 1 ))
  printf '  %-46s %s failed cells\n' "slora $(basename "${f}")" "${n}"
  [[ "${n}" -gt 0 ]] && { status=1; sed 1d "${f}" | sed 's/^/      /'; }
done

echo
if [[ "${status}" -eq 0 ]]; then
  echo "OK: no OOM, no missing cells"
else
  echo "ATTENTION: lower the offending task's batch via SPARSE15_<TASK>_BATCH and re-run that cell"
fi
exit "${status}"
