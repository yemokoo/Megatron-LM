#!/bin/bash
# Stop hmse_kd200 on 2,3 and resume the rest of it on 0-3.
#
# MeetingBank is abandoned mid-round: its KD-init was nearly done but the joint
# half had not started, and redoing both on four GPUs costs about what
# finishing them on two would have.
#
# Resume, not restart: the completed rounds are already on disk and the trainer's resume
# path re-checks the stored memory indices against their sha256, so the replay
# stream cannot silently diverge.  The gate in run_v3_job.sh is skipped for the
# training half only because a resume writes resume_from_N.command.txt instead
# of train.command.txt; the eval half still goes through run_v3_job.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace
OUT="${RUNS}/v3_hmse_kd200/v3_new_hmse_kd200_st_top1"
JOB_PID="${JOB_PID:?export JOB_PID (the run_v3_job.sh holding 2,3)}"
GPUS="${TARGET_GPUS:-0,1,2,3}"
LOG="${RUNS}/ablation_lanes/move_hmse_kd200.log"

say() { printf '[MOVE %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"; }

say "stopping job ${JOB_PID} on 2,3"
pkill -TERM -P "${JOB_PID}" 2>/dev/null
kill -TERM "${JOB_PID}" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "${JOB_PID}" 2>/dev/null || break; sleep 5; done
pkill -KILL -f "main_Ours_LoRA_MoE.py --training_version v3_new_hmse_kd200" 2>/dev/null
sleep 20

for attempt in $(seq 1 40); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g="${GPUS}" 'index(","g",", ","$1",")>0 && $2>2000' | wc -l)
  [[ "${busy}" -eq 0 ]] && break
  say "waiting for ${busy} of ${GPUS} to free (attempt ${attempt})"
  sleep 15
done

export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Four GPUs: micro 16 x accum 1 keeps the effective batch at the baseline's 64
# with the same per-device batch the two-GPU half already used, so the resume
# contract sees an unchanged per_device_train_batch_size.
say "resuming training from round ${RESUME_ROUND:-1} on ${GPUS} (micro 16 x accum 1 = 64)"
OURS_ST_RUN_ROOT="${RUNS}/v3_hmse_kd200" \
OURS_LORAMOE_GPUS="${GPUS}" \
OURS_LORAMOE_MICRO_BATCH=16 OURS_LORAMOE_GRAD_ACCUM=1 \
OURS_LORAMOE_RESUME_CHECKPOINT="${OUT}/${RESUME_ROUND:-1}" \
  bash "${ROOT}/scripts/run_v3_hmse_kd200_st_top1_8gpu.sh" >> "${LOG}" 2>&1
status=$?
say "training exit=${status}"
[[ "${status}" -eq 0 ]] || exit "${status}"

say "starting sparse-15 eval on ${GPUS}"
SPARSE15_NUM_SAMPLE_SHARDS=4 \
  bash "${ROOT}/scripts/run_v3_job.sh" v3_hmse_kd200 v3_new_hmse_kd200 "${GPUS}" eval >> "${LOG}" 2>&1
say "eval exit=$?"
