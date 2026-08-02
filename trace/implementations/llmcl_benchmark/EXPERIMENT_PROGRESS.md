# LLMCL Benchmark Experiment Progress

이 문서는 `/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark`의 실험 진행 상황, 재현 설정, 실패 원인, 체크포인트와 다음 실행 계획을 한곳에서 관리하는 living document이다.

- 마지막 갱신: 2026-07-16 15:19 KST
- TRACE task 순서: `C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten`
- task checkpoint 번호는 0-based task index와 같다.

## 갱신 규칙

새 실험을 시작하거나 상태가 바뀌면 다음 내용을 이 파일에 추가한다.

1. 맨 위의 `마지막 갱신` 시각을 수정한다.
2. `현재 실행 상태`를 갱신한다.
3. `실험 이력`에 새 항목을 역순으로 추가한다.
4. 실행 명령, GPU 수, batch, accumulation, 입력 checkpoint, output, log를 반드시 기록한다.
5. 완료 시 최종 checkpoint와 평가 결과를 기록한다.
6. 실패 시 마지막 정상 checkpoint, 실패 phase/step, traceback 요약과 재개 방법을 기록한다.

권장 기록 템플릿:

```markdown
### YYYY-MM-DD HH:MM — 실험 이름

- 상태: RUNNING | COMPLETED | FAILED | STOPPED
- 목적:
- 모델/방법:
- GPU:
- 시작 checkpoint:
- 학습 범위:
- 주요 설정:
- output:
- log:
- 실행 명령:
- 결과 또는 현재 step:
- 실패 원인/다음 작업:
```

## 방법 요약

### Track 1 — Qwen3 + Growing LoRA-MoE

- dense Qwen3 backbone을 freeze한다.
- TRACE task마다 LoRA expert를 추가한다.
- task 학습 후 router replay를 수행하는 continual-learning track이다.

### Track 2 — OLMoE + Growing Full-FFN Experts

- base model: `/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125`
- OLMoE의 원본 64개 FFN expert, attention, embedding, norm, LM head는 freeze한다.
- 각 TRACE task마다 모든 MoE layer에 full `OlmoeMLP` expert를 `experts_per_task`개 추가한다.
- 현재 주 실험은 `experts_per_task=1`이다.
- 원본 top-8 routing을 유지하면서 router 크기는 layer마다 `64 → 64+n`으로 증가한다.

Phase 1:

- 현재 task의 새 expert와 대응하는 새 router row만 학습한다.
- 이전 task expert/router row와 원본 64 expert/router는 freeze한다.
- 기본 `phase1_new_expert_routing=force`에서는 현재 task의 새 expert를 모든 non-padding token의 top-8 슬롯 하나에 강제로 포함한다.

Phase 2:

- 모든 expert를 freeze한다.
- 원본 64 router row와 추가된 모든 router row를 학습한다.
- 지금까지 본 TRACE task 전체와 OLMoE pretraining sample 5,000개를 replay한다.
- Phase 2와 inference에서는 새 expert 강제 포함을 해제하고 전체 `64+n`에서 자연 top-8 routing을 사용한다.

## Track 2 Batch Survey

4×A100 80GB, `experts_per_task=1`, forced phase-1 routing 기준 survey 결과:

| Task | Index | Per-device batch | Global batch (4 GPU) | Gradient checkpointing |
|---|---:|---:|---:|---|
| C-STANCE | 0 | 32 | 128 | OFF |
| FOMC | 1 | 32 | 128 | OFF |
| MeetingBank | 2 | 28 | 112 | ON |
| Py150 | 3 | 28 | 112 | ON |
| ScienceQA | 4 | 32 | 128 | ON |
| NumGLUE-cm | 5 | 32 | 128 | OFF |
| NumGLUE-ds | 6 | 32 | 128 | OFF |
| 20Minuten | 7 | 28 | 112 | ON |

- survey 결과: `eval_out/batch_survey_track2_4gpu_task_growth_seed1234.json`
- Phase 2 batch 28은 task 6 누적 replay에서 OOM이 발생했다.
- 재개 실행에서는 Phase 2 `router_retune_batch_size=16`을 사용한다.

2-GPU 전환 원칙:

- per-device batch는 메모리 관점에서 그대로 유지한다.
- 4-GPU와 같은 effective global batch 및 optimizer-step 수를 유지하려면 gradient accumulation을 2로 설정한다.
- 즉, `4 GPU × batch × accum 1 = 2 GPU × batch × accum 2`이다.
- 현재 런처의 accumulation 환경변수화는 아직 필요하다.

## Checkpoint Map

Track 2 output root:

`output/track2_OLMoE_ept1_force_upper_5k_seed1234`

| Checkpoint | 완료 task | 추가 expert/layer | 상태 |
|---:|---|---:|---|
| 0 | C-STANCE | 1 | 완료 |
| 1 | FOMC | 2 | 완료 |
| 2 | MeetingBank | 3 | 완료 |
| 3 | Py150 | 4 | 완료 |
| 4 | ScienceQA | 5 | 완료 |
| 5 | NumGLUE-cm | 6 | 완료, 현재 resume source |
| 6 | NumGLUE-ds | 7 | 완료 |
| 7 | 20Minuten | 8 | 미실행 |

각 checkpoint에는 최소 다음 파일이 있어야 한다.

- `pytorch_model.bin`: learned expert/router delta
- `moe_ffn_meta.json`: base model, expert 수, routing 방식과 checkpoint format
- `config.json` 및 tokenizer 파일

## 현재 실행 상태

### 2026-07-16 — NumGLUE-ds task 6 resume 완료

- 상태: COMPLETED, checkpoint 6 저장 후 자동 중지 완료
- 완료 시각: 2026-07-16 15:19 KST
- 최종 phase/step: `NumGLUE-ds [phase2 router retune]`, `625/625`
- 최종 확인 loss: `0.7388`
- GPU: 4×A100 80GB
- 시작 checkpoint: `output/track2_OLMoE_ept1_force_upper_5k_seed1234/5`
- `start_task=6`
- NumGLUE-ds Phase 1은 처음부터 다시 학습 완료
- Phase 1 per-device batch: 32
- Phase 2 per-device router batch: 16
- output: 기존 Track 2 output root에 checkpoint 6 저장 완료
- checkpoint model: `output/track2_OLMoE_ept1_force_upper_5k_seed1234/6/pytorch_model.bin` (1,414,100,170 bytes)
- checkpoint metadata: `output/track2_OLMoE_ept1_force_upper_5k_seed1234/6/moe_ffn_meta.json`
- live log: `logs/track2_olmoe_resume_task6_rb16_live.log`
- stop log: `logs/track2_olmoe_resume_task6_rb16_stop.log`

실행 명령의 핵심 설정:

```bash
OUT=output/track2_OLMoE_ept1_force_upper_5k_seed1234 \
ROUTER_BATCH=16 \
RESUME_CHECKPOINT=output/track2_OLMoE_ept1_force_upper_5k_seed1234/5 \
START_TASK=6 \
PHASE1_ROUTING=force \
bash scripts/train_Ours_MoE_FFN.sh 1 \
  /home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125
```

자동 중지 조건:

- watchdog이 `output/track2_OLMoE_ept1_force_upper_5k_seed1234/6/moe_ffn_meta.json` 생성을 1초 간격으로 확인한다.
- checkpoint 6 metadata 생성 확인 후 `sync`를 수행하고 4-GPU torchrun에 `SIGTERM`을 보냈다.
- 목적은 checkpoint 6을 완전하게 저장하고 20Minuten Phase 1은 시작하지 않는 것이다.
- 15:19 KST에 torchrun PID 65931과 worker가 종료되었으며 task 7은 시작되지 않았다.
- 현재 남은 관련 프로세스는 사용자가 실시간 로그 확인용으로 실행한 `tail -f`뿐이다.
- 다음 2-GPU 세션에서 checkpoint 6을 불러와 task 7만 재개한다.

## 실험 이력

### 2026-07-16 15:19 — NumGLUE-ds checkpoint 6 완료 및 4-GPU 학습 중지

- 상태: COMPLETED
- Phase 2 router retune을 `625/625`, 최종 확인 loss `0.7388`로 완료했다.
- checkpoint 6의 `pytorch_model.bin`, `moe_ffn_meta.json`, config/tokenizer 파일 생성을 확인했다.
- watchdog이 metadata 저장을 확인한 뒤 `sync`하고 torchrun PID 65931에 `SIGTERM`을 보냈다.
- 로그 끝의 `SignalException: signal 15`는 OOM이나 학습 실패가 아니라 계획된 자동 중지 기록이다.
- task 7 `20Minuten`은 실행되지 않았으며 다음 2-GPU 세션에서 checkpoint 6으로부터 재개한다.

### 2026-07-16 — checkpoint 5에서 task 6 재개

- 상태: RUNNING
- 기존 checkpoint loader를 training resume에 연결했다.
- `--resume_checkpoint`와 `--start_task` 인자를 추가했다.
- checkpoint의 `num_new_experts`가 `start_task × experts_per_task`와 일치하는지 검증한다.
- 완료 task는 global index를 유지하면서 skip한다.
- checkpoint 5에는 layer당 추가 expert 6개가 있으며 `start_task=6`, `experts_per_task=1` 검증을 통과했다.
- background/nohup process가 실행 환경 정책으로 정리되는 문제가 있어 unified long-running session으로 실행했다.

### 2026-07-16 — 최초 Track 2 연속 실행 실패

- 상태: FAILED
- output checkpoint 0~5까지 정상 저장했다.
- task 6 `NumGLUE-ds` Phase 1은 완료했으나 Phase 2 router replay에서 OOM이 발생했다.
- 실패 지점: 기존 per-device router batch 28, 약 `197/358` step.
- checkpoint는 task의 Phase 1과 Phase 2가 모두 끝난 뒤 저장하므로 checkpoint 6은 생성되지 않았다.
- 복구 결정: checkpoint 5에서 NumGLUE-ds Phase 1부터 다시 학습하고 router batch를 16으로 낮춘다.
- 실패/과거 로그: `eval_out/track2_OLMoE_ept1_force_upper_5k_seed1234.log`

### 2026-07-15~16 — Track 2 batch survey 및 OLMoE replay 준비

- 상태: COMPLETED
- 4-GPU task-growth 조건에서 task별 Phase 1 batch와 checkpointing 조합을 측정했다.
- OLMoE 원본 routing 보존용 replay sample 5,000개를 준비했다.
- replay path: `data/OLMOE0125sampling_5000_seed1234/sample_5000.jsonl`
- sampling manifest: `data/OLMOE0125sampling_5000_seed1234/sampling_manifest.json`

### 2026-07-14~15 — Track 1 Qwen3 진행

- `output/track1_Qwen3-8B_full`: checkpoint 0~7 존재, 8-task 완료.
- `output/track1_Qwen3-8B_ept1`: checkpoint 0~3 존재, 중간 상태.
- 세부 평가 결과와 실패 원인은 후속 확인 시 이 문서에 보강한다.

## 다음 작업

1. 다음 2-GPU 세션에서 checkpoint 6 metadata의 `num_new_experts=7`을 최종 검증한다.
2. 2-GPU 런처에서 `gradient_accumulation_steps=2`를 환경변수로 전달하도록 수정한다.
3. checkpoint 6에서 `start_task=7`로 20Minuten Phase 1/2를 실행한다.
4. checkpoint 7 저장 후 TRACE 8-task 평가를 수행하고 결과를 이 문서에 기록한다.
