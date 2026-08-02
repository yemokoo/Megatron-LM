# Full-run templates (render only)

These commands are not authorized for execution during scaffold setup. Run
strict preflight and `run_contract.py` first, and replace every angle-bracket
placeholder. Apply patches only in a disposable copy and record that copy's
commit/tree hash.

## SLoRA-Pre

For each task and its configured epoch count:

```bash
python scripts/run_contract.py \
  --train-json <TASK_TRAIN_JSON> --task <TASK> --expected-samples 5000 \
  --micro-batch 8 --world-size <VERIFIED_WORLD_SIZE> \
  --gradient-accumulation 2 --epochs <TASK_EPOCHS> --logging-steps 1

torchrun --nproc_per_node=<VERIFIED_WORLD_SIZE> src/train/cl_train_slora.py \
  --bf16 True --use_peft True --lora_r 64 --lora_alpha 128 \
  --deepspeed <SCAFFOLD_ROOT>/config/deepspeed_zero2.json \
  --model_name_or_path <APPROVED_MODEL_PATH> --model <llama3-or-qwen> \
  --dataset_name <TASK> --train_data_path <TASK_TRAIN_JSON> \
  --output_dir <PROVENANCE_OUTPUT>/order<TASK_INDEX> \
  --num_train_epochs <TASK_EPOCHS> --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 \
  --eval_strategy no --save_strategy steps --save_steps 100 \
  --save_total_limit 1 --learning_rate 2e-4 --weight_decay 0 \
  --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 \
  --gradient_checkpointing True --task_id <TASK_INDEX> --mode max
```

For SLoRA-Post use `src/train/cl_train.py` for sequential raw adapter training,
then invoke the patched Post denoising/evaluation path after all eight raw
updates exist. Seq-LoRA uses the same trainer without a denoising mode.

## TRACE official and corrected baselines

Run the clean snapshot for the official path and a disposable copy with
`trace-gem-qpth-sign.patch` only for corrected GEM:

```bash
deepspeed --include=localhost:<GPU_LIST> training/main.py \
  --data_path <VERIFIED_5000_ROOT> \
  --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
  --model_name_or_path <APPROVED_MODEL_PATH> \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 16 \
  --max_prompt_len 1024 --max_ans_len 512 --learning_rate 2e-4 \
  --weight_decay 0 --num_train_epochs 5,3,7,5,3,5,5,7 \
  --gradient_accumulation_steps 8 --lr_scheduler_type cosine \
  --seed 1234 --zero_stage 2 --deepspeed --print_loss \
  --CL_method <GEM-or-EWC-or-LwF> --output_dir <PROVENANCE_OUTPUT>
```

The upstream `train_seq_cl.sh` contains invalid shell assignment
`$cl_method="EWC"` and hard-coded eight-GPU paths; do not use it unchanged.
The exact learning rate for a paper-faithful TRACE baseline remains a
paper/code reconciliation gate—do not assume this template proves equivalence.

## Official and corrected O-LoRA

Use the pinned `scripts_llama/order_*.sh` as the official command source after
replacing model/data/output/GPU paths. It specifies one epoch, per-device
batch 1, accumulation 8, eight GPUs, and `logging_steps=10`. The corrected
profile changes only `olora-paper-orthogonality.patch`. This is the standard
classification benchmark, not TRACE.
