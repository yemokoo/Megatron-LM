#!/bin/bash
# Force-kill ANY running training/eval job started via this repo's scripts
# (main_Ours_LoRA_MoE / main_Ours_MoE_FFN / main_baseline / their deepspeed
# launchers), then verify all GPUs actually freed. Use this before launching a
# new run -- a plain Ctrl-C or a narrow pkill pattern can leave DeepSpeed's
# spawned rank processes alive and holding GPU memory.
cd "$(dirname "$0")/../.."   # repo root, so logs/ path matches run.sh
mkdir -p logs
# Tell run.sh's supervisor this was an intentional kill -> do NOT start the
# session-keepalive daemon (agent_data_make.py). run.sh removes this at launch.
touch logs/.killed

echo "Killing training processes..."
pkill -9 -f "main_Ours_LoRA_MoE.py" 2>/dev/null
pkill -9 -f "main_Ours_MoE_FFN.py" 2>/dev/null
pkill -9 -f "main_baseline.py" 2>/dev/null
pkill -9 -f "agent_data_make.py" 2>/dev/null   # session-keepalive daemon
pkill -9 -f "torch.distributed.run" 2>/dev/null
pkill -9 -f "torchrun" 2>/dev/null
pkill -9 -f "deepspeed.launcher" 2>/dev/null
pkill -9 -f "bin/deepspeed" 2>/dev/null
pkill -9 -f "train_Ours_LoRA_MoE.sh" 2>/dev/null
pkill -9 -f "train_Ours_MoE_FFN.sh" 2>/dev/null
pkill -9 -f "baseline_.*\.sh" 2>/dev/null
sleep 3

echo
echo "================ processes (should be empty) ================"
pgrep -af "main_Ours_LoRA_MoE|main_Ours_MoE_FFN|main_baseline|torchrun|torch.distributed.run|deepspeed" | grep -v grep || echo "(none)"

echo
echo "================ GPU (memory.used should be ~0 on all) ================"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo
echo "================ compute apps (should be empty) ================"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
