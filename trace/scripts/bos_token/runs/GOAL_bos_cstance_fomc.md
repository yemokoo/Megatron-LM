# GOAL: `<BoS_cstance>` 생성 replay vs 앵커 생성 replay — FOMC 라운드 학습 후 C-STANCE/FOMC 채점

## 배경 (이미 된 것)
- 시작 체크포인트: `/data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/0` (ours V3 qkvo+ffn r64, 1phase, kd_init on, C-STANCE 라운드0 완료. C-STANCE 58.35)
- v1 토큰: `<|reserved_special_token_0|>`(id 128002)를 `<BoS_cstance>`로 사용. 학습본: `/data2/seonghyeonnoh/paper/bos_token/cstance_1phase_kd_rep/lr1e-3/bos_token.pt` (lr 1e-3, 1 epoch, eval NLL 2.409→2.387)
- 생성 스크립트: `/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/scripts/bos_token/gen_doc.py` (토큰 1개 → system/user/assistant 문서 전체 생성, 3번째 <|eot_id|>에서 정지, user turn만 `text.jsonl`로 출력 → Stage B 호환)
- 50개 예비 생성: user turn 템플릿 정확 일치 50/50, 실제 데이터와 길이·다양성 동급 (`/data2/seonghyeonnoh/paper/bos_token/cstance_1phase_kd_rep/gen50/`)
- production 앵커 생성(`run_arm_gen.sh` gen_chunk)은 bash `$(...)`가 앵커 끝 개행을 잘라 `文本：` 뒤 개행이 없는 형식으로 생성됨. "기존 앵커 방식" baseline은 이 production 동작 그대로 사용한다(버그 포함, 기존과 동일 조건).

## 공통
```
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$TRACE/.venv-runtime/bin/python
CK0=/data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/0
ROOT=/data2/seonghyeonnoh/paper/bos_token/fomc_arms      # 이 실험의 루트 (새로 생성)
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
진행 로그: `$ROOT/progress.log` (모든 단계 시작/종료를 append). 사용자에겐 항상 `tail -F <절대경로>`로 로그를 알려줄 것.

## Step 1 — BoS 기반 C-STANCE replay 640 생성 → 좋은 500 선별
1. 640개 문서 생성 (GPU 1개, ~5분):
   ```
   CUDA_VISIBLE_DEVICES=0 $PY $TRACE/scripts/bos_token/gen_doc.py --checkpoint $CK0 \
     --bos-token-file /data2/seonghyeonnoh/paper/bos_token/cstance_1phase_kd_rep/lr1e-3/bos_token.pt \
     --num-seqs 640 --batch 64 --max-new-tokens 320 --seed 1 --out-dir $ROOT/bos_gen/stageA
   ```
   출력: `stageA/docs.jsonl`(전체 문서 + user/answer 파싱), `stageA/text.jsonl`(user turn, `{"anchor":"","text":...}`).
   주의: 문서 앞에 `<|begin_of_text|>` 반복이나 `<|start_header_id|>system<|end_header_id|>system<|end_header_id|>` 말더듬이 섞임 — gen_doc.py의 `system_ok` 플래그는 이를 실패로 세지만 실제 내용은 정상. 선별 시 user turn 기준으로만 판단.
2. Stage B: production과 동일하게 greedy 답 생성 (`answer_pass_v3_fix.py`, cue `\n态度：`):
   ```
   CUDA_VISIBLE_DEVICES=0 $PY $TRACE/scripts/analysis/answer_pass_v3_fix.py --checkpoint $CK0 \
     --stage-a $ROOT/bos_gen/stageA --out $ROOT/bos_gen/records.all.jsonl \
     --prompt-cue $'\n态度：' --max-answer-tokens 256 --batch 64
   ```
   (gen_doc.py가 뽑은 sampled answer는 docs.jsonl에 남겨두되, 학습 라벨은 production과 같은 greedy 답을 쓴다.)
3. 500개 선별 → `$ROOT/bos_gen/gen/round_1/C-STANCE/records.jsonl` (정확히 500줄, `{"prompt","answer"}`):
   - prompt가 정규식 `^<anchor>(.+?)\n对象：\n(.+?)\n态度：$` (DOTALL)에 완전 일치. anchor = `scripts/selfgen/assets/anchors.json`의 key "0" (끝 개행 포함, 파이썬에서 JSON으로 읽을 것 — bash `$(...)` 금지)
   - 본문 길이 20~400자, answer ∈ {A,B,C}, prompt 중복 제거
   - 조건 통과분이 500 초과면 `random.Random(0).sample`로 500. 500 미만이면 `--seed 2`로 추가 생성 후 반복.
   - 선별 통계(통과율, 답 분포 A/B/C, 본문 길이 평균)를 `$ROOT/bos_gen/select_stats.json`에 저장.

## Step 2 — 앵커(production) baseline replay 640 생성
`run_arm_gen.sh`의 gen_chunk와 동일 절차를 C-STANCE(i=0)에 대해 재현 (8 chunk × 80, seed 200+k, 앵커는 production처럼 `$($PY -c ...)`로 캡처 = 개행 잘림 그대로):
```
SG=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829; [ -f $SG/anchors.json ] || SG=$TRACE/scripts/selfgen/assets
PFX=$($PY -c "import json;print(json.load(open('$SG/anchors.json'))['0']['anchor'],end='')")
for k in 0..7 (GPU k):
  $PY $TRACE/scripts/analysis/bos_sample_v3.py --checkpoint $CK0 --mode anchor --prefix-text "$PFX" \
    --num-seqs 80 --max-seqs 200000 --max-new-tokens 256 --batch 256 --no-routing-probe --seed $((200+k)) \
    --out-dir $ROOT/anchor_gen/stageA.shard$k --label r1_C-STANCE_s$k
  $PY $TRACE/scripts/analysis/answer_pass_v3_fix.py --checkpoint $CK0 --stage-a $ROOT/anchor_gen/stageA.shard$k \
    --out $ROOT/anchor_gen/records.part$k.jsonl --prompt-cue $'\n态度：' --max-answer-tokens 256 --batch 64
cat records.part*.jsonl > $ROOT/anchor_gen/gen/round_1/C-STANCE/records.jsonl   (640줄; 트레이너가 seed 2025로 500 랜덤 선택 = production 동일)
```

## 실행 순서 (GPU 배정)
1. Step 1 (bos_gen 생성+StageB+선별): GPU 1개.  Step 2 (anchor_gen 생성): 8 chunk를 GPU 0~7에 1개씩 병렬, StageB도 같은 GPU에서 chunk별로 이어서. 두 Step은 동시에 돌려도 됨(Step 1은 GPU 0 사용 시 Step 2 chunk0과 겹치지 않게 Step 2를 GPU 1~7 + 0 순서로 대기시키거나, Step 1 완료 후 Step 2 시작).
2. Step 3 학습은 **8 GPU 전부 사용, 직렬**: bos_gen 학습 → 완료 확인 → anchor_gen 학습. 하나의 wrapper 스크립트(`$ROOT/run_chain.sh`, nohup)로 두 학습을 이어 붙이고 각 단계 시작/종료를 `$ROOT/progress.log`에 기록.
3. Step 4 평가는 두 학습이 모두 끝난 뒤 **arm당 GPU 1개씩 병렬** (bos_gen→GPU 0, anchor_gen→GPU 1, 참조용 `1phase_kd_rep/1`→GPU 2). C-STANCE+FOMC는 batch 16으로 몇 분이면 끝남. 평가도 wrapper 끝에 붙여서 사람 개입 없이 RESULT.md까지 나오게 할 것.

## Step 3 — FOMC 라운드(라운드 1) 학습, 두 arm (8 GPU, 직렬: bos_gen → anchor_gen)
각 arm `A in {bos_gen, anchor_gen}`에 대해 RUN=$ROOT/$A:
1. 레이아웃 준비: `mkdir -p $RUN/model $RUN/logs`; `cp -r $CK0 $RUN/model/0` (symlink 말고 복사); `cp -r /data2/seonghyeonnoh/paper/ablation/2phase_kd_gen/model/fixed_replay_memory $RUN/model/` (output_dir=$RUN/model 기준 상대경로이므로 반드시 model/ 밑에 위치해야 함 — $RUN/fixed_replay_memory 아님, 2phase_kd_gen에서 실제 레이아웃 확인함); `$RUN/gen/round_1/C-STANCE/records.jsonl` 존재 확인.
2. 학습 명령 = `run_arm_gen.sh`의 train_round(t=1)와 **완전히 동일한 하이퍼파라미터** (`$TRACE/scripts/ablation/run_arm_gen.sh` 62~95행 참조), 차이는 아래만:
   - `PHASE=1phase KD=on`, `--stop_after_task FOMC`, `--resume_checkpoint $RUN/model/0`, `--output_dir $RUN/model`, `--data_output_path $RUN/data_cache`
   - env `SELFGEN_ROOT=$RUN/gen/round_1 SELFGEN_CURRENT_TASK=FOMC`
   - env `RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch,replay_source` — 라운드0 체크포인트는 `ablation.replay_source=real`이라 `replay_source` 드리프트 허용이 필수.
   - 8 GPU, `--nproc_per_node 8`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, global batch 64 (1phase_kd_rep와 동일). 두 arm은 **직렬** 실행(동시 실행 시 4 GPU+accum 2로 바꿔야 해서 조건이 달라짐). 각 ~15분. 앞 arm의 `model/1/lora_moe_meta.json` 확인 후 다음 arm 시작.
   - 로그 `$RUN/logs/train_r1.log`. 시작 후 2분 내 `resume hyperparameter mismatch` / `SELFGEN` 관련 에러가 없는지 확인. 실패 시 원인 파악 후 재실행(멋대로 하이퍼파라미터 바꾸지 말 것).
3. 완료 조건: `$RUN/model/1/lora_moe_meta.json` 존재, `num_experts=2`, `ablation.replay_source=selfgen`, `selfgen_root`가 $RUN/gen/round_1.

## Step 4 — 채점 (각 arm의 model/1에 대해 C-STANCE, FOMC; arm당 GPU 1개, 병렬)
```
cd $TRACE/implementations/llmcl_benchmark
CUDA_VISIBLE_DEVICES=g $PY -u evaluate_Ours_LoRA_MoE.py --checkpoint_dir $RUN/model/1 \
  --base_model_name_or_path /data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct \
  --data_path $DATA --inference_tasks C-STANCE,FOMC --inference_output_path $RUN/evaluation/order1 \
  --summary_filename r1_cstance_fomc.summary.json --max_prompt_len 0 --max_ans_len 1024 \
  --no-task_generation_limits --slora_conv_mode llama3 --per_device_eval_batch_size 16 --temperature 0
```
bos_gen→GPU 0, anchor_gen→GPU 1, 참조 `1phase_kd_rep/1`→GPU 2 로 동시에 시작. 비교 기준(reference):
- 1phase_kd_rep(실 데이터 replay): C-STANCE 라운드0 58.35, FOMC 라운드1 71.98 (`/data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/sparse15_summary.json`). 이 run에는 라운드1 C-STANCE 점수가 없으니(sparse15) 같은 명령으로 `/data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/1`도 채점하되 출력은 `$ROOT/ref_real/evaluation/order1`에 쓴다(원본 run 디렉토리는 건드리지 말 것). 결과 표는 3열(bos_gen / anchor_gen / real replay).

## 산출물 / 보고
- `$ROOT/RESULT.md`: 표 [arm × {C-STANCE@r1, FOMC@r1, C-STANCE 망각 = 58.35 − C-STANCE@r1}] (bos_gen, anchor_gen, +real replay 참조), replay 통계(선별 통과율, 답 분포, 본문 길이), 학습 시간, 문제/특이사항.
- 결론 한 줄: BoS 생성 replay가 앵커(production) 생성 replay 대비 C-STANCE 망각을 줄이는가, FOMC 학습에 손해가 없는가.

## 하지 말 것
- 하이퍼파라미터·epoch·batch·seed 변경 금지(비교 실험). 앵커 개행 버그도 baseline에서는 고치지 말 것(기존 조건 재현).
- 이미 돌고 있는 다른 GPU 작업이 있으면 죽이지 말고 빈 GPU만 사용. 시작 전 `nvidia-smi`.
