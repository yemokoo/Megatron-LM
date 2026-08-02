# Experiments — Growing-Expert Continual Learning on TRACE

이 문서는 구현 현황 · 실험 세팅 · 진행/결과를 추적한다. **실험이 진행되거나 결과가 나올 때마다 갱신**한다. (마지막 갱신: 2026-07-23 UTC)

---

## 1. 개요

제안 방법: **태스크마다 expert를 추가(성장)하고 backbone은 freeze한 채 추가 파라미터만 학습**하는 continual learning. 두 백본에서 각각 구현한다.

| 모듈 | 이름 | 백본 | 추가하는 것 |
|---|---|---|---|
| Track 1 | `Ours_LoRA_MoE` | Qwen (dense) | FFN에 token-routed **LoRA expert** |
| Track 2 | `Ours_MoE_FFN` | OLMoE-1B-7B-0125 (native MoE) | 기존 64 expert 위에 **full FFN expert** (router 64→64+n) |

학습 흐름(공통): 태스크마다 **(phase-1)** 새 expert + 라우터 학습 → **(phase-2)** 모든 expert freeze 후 라우터만 과거 태스크 replay로 재보정.

---

## 2. 벤치마크 / 하니스

- **벤치마크**: TRACE, 8 태스크 5000 샘플씩, 표준 순서
  `C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten`
- **학습 세팅** (run 계열별 차이는 아래 별도 표에 명시):

  | 항목 | 값 | 비고 |
  |---|---|---|
  | sequence length | 2048 (`--max_prompt_len 1536 --max_ans_len 512`) | collator가 prompt+answer를 2048에 동적 패킹 |
  | epoch/task | **2** | 현재 Qwen3-8B paper-baseline chain 기준 |
  | learning rate | 1e-4 | |
  | weight decay | **0.0 (OFF)** | 우리 방식은 wd 미사용 (plain) |
  | LR schedule | constant + warmup | plain (논문은 cosine이나 우리는 유지) |
  | optimizer | plain PyTorch AdamW, β=(0.9,0.95) | paper-baseline chain 기준 |
  | batch | task별 survey batch, **2-GPU DDP** | 아래 §6.1 참조 |
  | ZeRO | 미사용 | frozen backbone + adapter만 학습, plain DDP |

- **평가 지표**: **OP** (마지막 태스크 후 전 태스크 평균 성능) + **BWT** (backward transfer). `evaluate_Ours_*.py --all_rounds`가 task×round 매트릭스 + BWT 산출.

---

## 3. 구현 현황

| 구성요소 | Track 1 (LoRA) | Track 2 (MoE-FFN) |
|---|---|---|
| 모델 코어 | `model/Ours_LoRA_MoE.py` ✅ | `model/Ours_MoE_FFN.py` ✅ |
| 학습 엔트리 | `training/main_Ours_LoRA_MoE.py` ✅ | `training/main_Ours_MoE_FFN.py` ✅ |
| 체크포인트 로더+메타 | `load_lora_moe_checkpoint` ✅ | `load_moe_ffn_checkpoint` ✅ |
| 평가기 | `evaluate_Ours_LoRA_MoE.py` ✅ | `evaluate_Ours_MoE_FFN.py` ✅ |
| 학습 스크립트 | `scripts/train_Ours_LoRA_MoE.sh` ✅ | `scripts/train_Ours_MoE_FFN.sh` ✅ |

**논문 대조군 5종(Qwen3-8B, FFN-LoRA) 구현**:

- 모델/알고리즘: `model/continual_lora.py`, `model/paper_baselines.py`
- 명시적 모델 래퍼: `model/{SeqLoRA,LoRAMoE,EWC_LoRA,GEM_LoRA,O_LoRA_FFN}.py`
- 학습 엔트리: `training/main_paper_baselines.py`
- 개별 실행: `scripts/train_paper_baseline.sh`
- 5종 순차 실행: `scripts/train_all_paper_baselines_2gpu.sh`
- 평가 로더 통합: `evaluate_Ours_LoRA_MoE.py`가 `paper_baseline_meta.json`을 감지해 partial checkpoint 재구성

**공유 baseline 인프라**: `model/baselines.py` (`StaticBaseline`) + `training/main_baseline.py`
(`--baseline {finetune, static_lora_moe, static_moe_ffn} --static_experts N`).

### 검증 상태
- ✅ Track 1: 라우터 성장·freeze 패턴·라우터-only 학습·로더 왕복(로짓 일치)·생성 — CPU 검증
- ✅ Track 2: 실제 OLMoE로 성장/freeze/no-op·로더, tiny 합성으로 gradient step·DeepSpeed 학습 경로
- ✅ StaticBaseline: tiny OLMoE + DeepSpeed로 엔진1회·성장없음·저장 검증
- ✅ 논문 대조군 5종: CPU 수학/구조 test + 각 방법 **2-GPU 학습 smoke** + checkpoint/result 검증
- ✅ Qwen3-0.6B 비교군 6종(5 baseline + Track1): 8/8 task 학습 및 OP/BWT 평가 완료 (§9.4–§9.7)

---

## 4. 실험 매트릭스 (대조군)

### 4.1 OLMoE (Track 2)  — *우선 진행*
| # | 조건 | 스크립트 | 정의 |
|---|---|---|---|
| a | **Ours** | `scripts/train_Ours_MoE_FFN.sh N` | 태스크마다 FFN expert N개 추가 + 2-phase |
| b | Plain finetune | `scripts/baseline_olmoe_finetune.sh` | OLMoE 전체 파라미터 학습 |
| c | Static-8 | `scripts/baseline_olmoe_static.sh 8` | 8 expert 미리 추가, backbone freeze, 그것만 학습 |

### 4.2 Qwen3-8B (Track 1) — 논문 정렬 대조군

| # | 조건 | 현재 스크립트/구현 | 정의 |
|---|---|---|---|
| a | **Ours (MH-MoE)** | `scripts/train_Ours_LoRA_MoE.sh N` | 태스크마다 LoRA expert 추가 + router 재보정 |
| b | **SeqLoRA** | `train_paper_baseline.sh seqlora` | FFN에 공유 LoRA 1세트, backbone freeze, 8 task 순차 갱신 |
| c | **LoRAMoE** | `train_paper_baseline.sh loramoe` | FFN 공통 linear + 8개 LoRA residual expert + token top-1 gate |
| d | **EWC** | `train_paper_baseline.sh ewc` | SeqLoRA 위 TRACE EWC; 매 train step의 squared-gradient Fisher 누적 |
| e | **GEM** | `train_paper_baseline.sh gem` | SeqLoRA 위 TRACE GEM; task-gradient 저장 + parameter별 QP constraint |
| f | **O-LoRA** | `train_paper_baseline.sh olora` | task별 새 rank-8 LoRA와 이전 adapter에 대한 orthogonality regularizer |

활성 adapter 파라미터 수는 rank 8/top-1로 맞춘다. Qwen3-8B에서 SeqLoRA 및 LoRAMoE top-1의 활성 LoRA 파라미터는 모두 **14,155,776개/token**이다. LoRAMoE router는 별도 소규모 trainable overhead로 기록한다.

---

## 5. 실행 방법

```bash
# 학습 (GPU 4장 단일 run, batch 12). expert/task 수는 첫 인자.
bash scripts/train_Ours_MoE_FFN.sh  2         # Track2 Ours
bash scripts/baseline_olmoe_finetune.sh        # Track2 plain FT
bash scripts/baseline_olmoe_static.sh 8         # Track2 static-8

# 평가 (전 라운드 → OP/BWT)
python evaluate_Ours_MoE_FFN.py --all_rounds \
  --output_dir output/track2_OLMoE-1B-7B-0125_ept2 \
  --base_model_name_or_path /home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125 \
  --data_path data/LLM-CL-Benchmark_5000 \
  --inference_output_path eval_out/track2_ours

# Qwen3-8B paper baseline 5종: GPU 0,1에서 순차 실행
bash scripts/train_all_paper_baselines_2gpu.sh

# 현재 실행 세션 확인
tmux attach -t paper5_2gpu
```

---

## 6. 진행 로그 / 결과

> 실험을 돌릴 때마다 여기에 추가한다.

### 진행 상황
- 2026-07-12: 구현 완료(두 트랙 + baseline), 코드 검증 완료. **실제 학습은 GPU 대기 중, 미시작.**
- 2026-07-18: 논문 대조군 5종 구현 및 2-GPU smoke 완료. Qwen3-8B 전체 순차 chain 시작.
- 2026-07-19 00:42 KST: SeqLoRA 앞의 5개 task 완료, NumGLUE-cm 29% 진행 중. 오류/OOM/NaN 없음.
- 2026-07-19 01:18 KST: **SeqLoRA 8/8 완료**. 총 학습 11,918.76초, 31,421,762 non-padding tokens, 추정 1,863,147.81 TFLOPs.
- 2026-07-19 약 06:23 KST: LoRAMoE의 앞 4개 task(C-STANCE~Py150) 완료 후 ScienceQA epoch 1 `43/278`(약 15%)에서 **GPU 0 CUDA OOM**. 432 MiB 추가 할당 시 GPU 0 여유가 약 309 MiB였다. rank 1은 collective를 기다리다 06:33 NCCL timeout, 06:34 전체 프로세스 종료. ScienceQA checkpoint는 생성되지 않음.
- 2026-07-19: ScienceQA per-GPU batch를 **18→12**로 낮춰 5개 baseline 모두에 공통 적용. method 내부 task resume를 구현하고 CPU 회귀/Python compile/shell 문법 및 실제 resume target 판정(`loramoe/3 → task 4`) 검증 완료.
- 2026-07-19 12:09 KST: chain 재실행. 완료된 SeqLoRA는 검증 후 skip, LoRAMoE checkpoint `3`에서 resume 시작. 기존 checkpoint는 optimizer-state 저장 도입 전이므로 모델 가중치는 복원하되 optimizer는 이번 한 번 새로 시작한다.

### 6.1 Qwen3-8B paper-baseline 5종 연속 실험 설정

| 항목 | 값 |
|---|---|
| 모델 | `/home/work/Agent_HJ/00_models/Qwen3-8B` |
| GPU | A100 80GB PCIe × 2 (`CUDA_VISIBLE_DEVICES=0,1`) |
| 실행 순서 | `SeqLoRA → LoRAMoE → EWC → GEM → O-LoRA` |
| task 순서 | `C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten` |
| epoch | 모든 method/task에서 2 |
| FFN target | `gate_proj`, `up_proj`, `down_proj` |
| LoRA | rank 8, alpha 32, dropout 0 |
| sequence | prompt 1536 + answer 512 |
| LR / WD | `1e-4` / `0` |
| task별 per-GPU batch | `10,6,8,8,12,18,26,8` (ScienceQA 18→12, 전 baseline 공통) |
| gradient checkpointing | `MeetingBank, Py150, ScienceQA, 20Minuten` |
| LoRAMoE | 8 experts, top-1, full-softmax routing, aux `0.01`, z-loss `0.001` (중단된 legacy 8B run 설정) |
| EWC | lambda 400, task 종료 후 별도 Fisher pass (중단된 legacy 8B run 설정) |
| GEM | task당 memory 100, replay batch 4, margin 0 (중단된 legacy 8B run 설정) |
| O-LoRA | orthogonal lambda 0.5, L2 lambda 0 |

순차 스크립트는 `set -euo pipefail`로 실행하며, 각 method 종료 후 `result.json`, 8개 checkpoint, method/world-size를 검증한 뒤에만 다음 method로 넘어간다. 새 0.6B 구현의 체크포인트는 AdamW/scheduler 상태와 EWC 누적 Fisher/직전 parameter snapshot 또는 GEM task-gradient state를 `paper_baseline_state.pt`에 함께 저장·복원한다. 위 8B legacy EWC/GEM checkpoint는 알고리즘 schema가 달라 새 구현의 resume 입력으로 사용하지 않는다.

경로:

- tmux: `paper5_2gpu`
- 전체 출력: `output/paper_baselines_Qwen3-8B_r8_2epoch_2gpu/`
- 전체 chain 로그: `output/paper_baselines_Qwen3-8B_r8_2epoch_2gpu/logs/chain.log`
- method 로그: `output/paper_baselines_Qwen3-8B_r8_2epoch_2gpu/logs/<method>.log`
- 전체 요약: `output/paper_baselines_Qwen3-8B_r8_2epoch_2gpu/result.json`
- method별 결과: `output/paper_baselines_Qwen3-8B_r8_2epoch_2gpu/<method>/result.json`
- task checkpoint: `output/paper_baselines_Qwen3-8B_r8_2epoch_2gpu/<method>/<0..7>/`

새 TRACE 방식 EWC/GEM은 별도 Fisher/replay forward 없이 train gradient를 재사용한다. `result.json`의 FLOPs는 일반 학습 `6ND`, activation-checkpointed train `8ND`이며 Fisher 누적과 QP elementwise 연산은 제외한다.

### 6.2 현재 진행 현황 — 2026-07-19 12:11 KST

전체 상태: **RUNNING**. chain 및 DDP rank 2개가 정상 생존 중이며 `--resume_checkpoint .../loramoe/3`과 ScienceQA batch 12 전달을 확인했다. 12:11 KST 확인 시 checkpoint shard 5/5 로드 완료 후 모델/adapter 초기화 단계(GPU당 약 16.5GB); 새 OOM/traceback 없음. `df: /home/work/.triton/autotune: No such file or directory`와 ``torch_dtype` is deprecated``는 비치명적 초기화 경고다.

| Method | 상태 | 완료 task | 현재/다음 |
|---|---|---:|---|
| SeqLoRA | **완료** | 8/8 | 재실행 시 검증 후 자동 skip |
| LoRAMoE | **resume 실행 중** | 4/8 | `loramoe/3` 복원 후 ScienceQA(task 4)부터; 이후 NumGLUE-cm → NumGLUE-ds → 20Minuten |
| EWC | 대기 | 0/8 | LoRAMoE 다음 |
| GEM | 대기 | 0/8 | EWC 다음 |
| O-LoRA | 대기 | 0/8 | GEM 다음 |

완료 task 기준 전체 진행률은 **12/40 = 30%**다. 이는 task별 길이와 EWC Fisher/GEM replay 추가 연산을 반영하지 않은 단순 개수 기준이다. aggregate `result.json`은 SeqLoRA를 `completed`, LoRAMoE를 `running`, 나머지를 `pending`으로 기록한다.

완료된 SeqLoRA 결과:

| task | 학습 시간 | non-padding tokens | 추정 TFLOPs | checkpoint |
|---|---:|---:|---:|---|
| C-STANCE | 218.43s (3m 38s) | 1,111,052 | 50,540.51 | `seqlora/0` |
| FOMC | 287.50s (4m 47s) | 767,632 | 34,918.72 | `seqlora/1` |
| MeetingBank | 3,898.54s (1h 04m 59s) | 13,013,406 | 789,287.01 | `seqlora/2` |
| Py150 | 3,524.57s (58m 45s) | 5,333,614 | 323,493.50 | `seqlora/3` |
| ScienceQA | 1,220.95s (20m 21s) | 2,858,334 | 173,363.21 | `seqlora/4` |
| NumGLUE-cm | 123.28s (2m 03s) | 518,708 | 23,595.45 | `seqlora/5` |
| NumGLUE-ds | 83.60s (1m 24s) | 414,714 | 18,864.88 | `seqlora/6` |
| 20Minuten | 2,561.88s (42m 42s) | 7,404,302 | 449,084.54 | `seqlora/7` |
| **누적** | **11,918.76s (3h 18m 39s)** | **31,421,762** | **1,863,147.81** | 8개 |

OOM 전 완료된 LoRAMoE 결과:

| task | 학습 시간 | non-padding tokens | 추정 TFLOPs | checkpoint |
|---|---:|---:|---:|---|
| C-STANCE | 1,353.40s (22m 33s) | 1,111,052 | 50,579.83 | `loramoe/0` |
| FOMC | 2,236.12s (37m 16s) | 767,632 | 34,945.89 | `loramoe/1` |
| MeetingBank | 7,247.53s (2h 00m 48s) | 13,013,406 | 789,901.06 | `loramoe/2` |
| Py150 | 6,406.80s (1h 46m 47s) | 5,333,614 | 323,745.17 | `loramoe/3` |
| **누적** | **17,243.85s (4h 47m 24s)** | **20,225,704** | **1,199,171.95** | 4개 |

### 6.3 5종 2-GPU smoke 검증 결과

Qwen3-0.6B, `C-STANCE → FOMC`, method/task당 1 step으로 실제 2-GPU DDP 경로, task 전환, checkpoint, metadata 및 result 누적을 검증했다.

| method | smoke 학습 시간 | tokens | 추정 TFLOPs | 추가 검증 |
|---|---:|---:|---:|---|
| SeqLoRA | 1.722s | 385 | 1.024 | frozen backbone, 공유 adapter |
| LoRAMoE | 5.610s | 385 | 1.026 | sparse top-1 DDP graph, 활성 파라미터 일치 |
| EWC | 3.274s | 770 | 2.048 | 2 task Fisher/mean state 저장 |
| GEM | 2.225s | 551 | 1.465 | 두 번째 task replay token 및 memory state 저장 |
| O-LoRA | 2.052s | 385 | 1.026 | 이전 adapter freeze + 현재 adapter 학습 |

### 결과 (OP / BWT)

#### OLMoE (Track 2)
| 조건 | OP | BWT | expert/task | 비고 |
|---|---|---|---|---|
| Ours | — | — | — | (미측정) |
| Plain finetune | — | — | — | (미측정) |
| Static-8 | — | — | — | (미측정) |

#### Qwen3-8B (Track 1)
| 조건 | OP | BWT | expert/task | 비고 |
|---|---|---|---|---|
| Ours | — | — | — | (미측정) |
| LoRAMoE | — | — | — | (미측정) |
| SeqLoRA | — | — | — | (미측정) |

---

## 7. Phase-2 라우터 재보정 설계 (트랙별 상이) — *구현 대기(데이터 확보 후)*

두 트랙의 라우터가 **라우팅 대상**이 달라서 phase-2를 다르게 간다:
- **Track 1 (dense+LoRA)**: 라우터는 **새로 추가한 LoRA expert만** 라우팅 (dense base FFN은 항상 적용, 라우팅 대상 아님). → 원본 보존 이슈 없음.
- **Track 2 (OLMoE)**: 라우터는 **기존 64 expert + 새 expert 전체**를 라우팅. all rows 재보정 시 원본 라우팅이 TRACE로 쏠려 훼손될 수 있음.

**보존 논리**: expert가 전부 freeze이므로 **"라우팅만 보존하면 능력(원본 지식)도 보존"**. pretraining 규모(5133B)와 맞출 필요 없음 — frozen expert 위의 저차원 라우팅 함수만 calibrate 유지하면 됨.

**phase-2 miniset 구성**:
| | miniset (phase-2, all router rows 재보정) |
|---|---|
| Track 1 | TRACE replay (과거+현재 태스크 누적) |
| Track 2 | TRACE replay (과거+현재) **+ OLMoE 원본 sample (라우터 분포 보존용)** |

- 원본 sample은 **라우터 파인튜닝에만** 사용 (phase-1 태스크 학습엔 새 데이터만; expert는 freeze).
- 원본 데이터 = **`allenai/OLMoE-mix-0924`** (bulk pretraining) 에서 **streaming-shuffle random sample** (도메인 자동 커버). 필요시 `allenai/dolmino-mix-1124`(annealing) 소량 추가.
- **양 = ablation 노브**: ~**3–10M 토큰** 예상 (2.3B은 풀/상한, 실제 사용은 그 일부). 규모 아니라 **도메인 다양성 + TRACE 대비 비율**이 핵심. 너무 많으면(예: 2.3B 전량) 원본이 압도해 태스크 학습을 못 함.
- **보존 측정(권장)**: 일반 벤치(MMLU/HellaSwag 등)로 학습 전/후 비교해 원본량을 실증적으로 튜닝.

구현 필요물(데이터 확보 후): OLMoE-mix streaming 샘플러, plain-LM 로더(raw text→2048), Track2 `_build_replay_loader`에 원본 혼합 + `--router_preserve_tokens` 노브. miniset은 **현재 태스크 포함**(`[:i_task+1]`)으로 변경.

## 8. 결정 사항 / 메모
- 아래 항목은 기존 8B/4-GPU 탐색 당시의 기록이다. 논문 재현 조건으로 실행하는
  Qwen3-0.6B 6종 비교에는 §9의 설정을 우선 적용한다.
- weight decay는 우리 방식이 안 쓰므로 **OFF 유지**. 논문의 cosine/wd 0.01/batch 10은 그쪽 공통 하니스이며, 우리는 seq 2048 + epoch만 맞추고 나머지는 plain.
- batch 10은 4-GPU에서 정확히 불가 → **batch 12로 통일**(모든 run 동일해 내부 비교는 공정).
- static 대조군: 성장 없이 expert만 미리 추가 + backbone freeze + 추가 파라미터(expert+router)만 학습, **phase-2 없음**.
- phase-2 라우터 재보정: **all rows**(Track2는 원본 gate까지) 재보정. 원본 보존은 별도 distillation 없이 **원본 데이터를 miniset에 섞는 것만으로** 달성(frozen expert + LM loss).

## 9. Qwen3-0.6B 6종 단일-GPU 병렬 실험 (2026-07-19)

논문의 0.6B 조건(2 epoch/task, LR 1e-4 cosine, sequence length 2048,
AdamW weight decay 0.01, β=(0.9,0.95), ε=1e-6)을 사용한다. LoRAMoE는
논문과 같이 layer당 expert 4개/top-1이며, Track1은 expert/task=1이다.

### 9.1 실제 OOM probe

- 모델: Qwen3-0.6B bf16
- 장치: A100 80GB 한 장/모델
- 입력: 각 task train split에서 가장 긴 sample을 batch 크기만큼 복제
- 측정 상태: LoRAMoE 4 experts, O-LoRA task 8, Track1 최종 8 experts
- 최초 survey의 GEM은 별도 replay gradient까지 포함해 현재 TRACE-gradient 구현보다 보수적으로 측정됐다. 모든 task의 공통 batch 병목은 LoRAMoE 또는 Track1이므로 권장 공통 batch에는 영향이 없다.
- 후보당 forward/backward/AdamW step 2회, 79,000 MiB 한계
- raw 결과: `eval_out/qwen06_batch_probe/summary.json`

| task | SeqLoRA | LoRAMoE | EWC | GEM | O-LoRA | Track1 | 공통 안전 batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| C-STANCE | 92 | 78 | 92 | 92 | 92 | 78 | **56** |
| FOMC | 64 | 54 | 64 | 64 | 64 | 54 | **40** |
| MeetingBank | 10 | 8 | 10 | 10 | 10 | 8 | **6** |
| Py150 | 10 | 8 | 10 | 10 | 10 | 8 | **6** |
| ScienceQA | 18 | 14 | 18 | 18 | 18 | 8 | **6** |
| NumGLUE-cm | 172 | 138 | 172 | 172 | 172 | 148 | **100** |
| NumGLUE-ds | 262 | 194 | 262 | 260 | 260 | 208 | **144** |
| 20Minuten | 10 | 8 | 10 | 10 | 10 | 8 | **6** |

공통 batch는 각 방법 측정 한계의 75%를 4의 배수로 내린 뒤 여섯 방법 중
최솟값을 택했다. 따라서 실제 학습에서 task별 effective batch가 여섯 방법
모두 동일하고, 측정 한계 대비 최소 25%의 여유가 있다.

### 9.2 실행

```bash
# probe를 다시 수행할 때
bash scripts/probe_all_qwen06_batches.sh

# GPU당 모델 하나. 한 모델이 끝나면 빈 GPU가 즉시 다음 method를 가져가는 동적 큐
bash scripts/train_qwen06_six_parallel.sh
```

초기 큐 순서는 `SeqLoRA, LoRAMoE, EWC, GEM, O-LoRA, Track1`이며, 처음 두
method를 GPU 0/1에 올린 뒤 먼저 끝난 GPU가 다음 method를 즉시 가져간다. Paper
baseline은 task checkpoint를 찾아 자동 resume하며 완료된 모델은 건너뛴다.
출력 기본 경로는 `output/qwen06_2epoch_fast6`이다.

### 9.3 EWC/GEM/LoRAMoE 공식 로직 재정렬 (2026-07-19)

- **EWC**: `/TRACE/model/Regular/EWC.py`와 동일하게 별도 task-end Fisher
  forward를 제거했다. 매 train backward 직후 `nan_to_num(grad)^2 / len(loader)`를
  누적하고, task 종료 시 직전 parameter snapshot을 갱신한다. penalty는
  `0.5 * lambda * sum(F * (theta-theta_prev)^2)`이다.
- **GEM**: `/TRACE/model/Regular/GEM.py`에 맞춰 episodic sample replay를 제거했다.
  각 task의 마지막 minibatch에서 (이전 constraint 적용 후) gradient를 bf16으로
  저장하고, 이후 task의 매 step에서 parameter tensor별로 과거 task gradient들과
  QP projection한다. margin=0, QP eps=1e-3이다.
- **LoRAMoE**: `Ablustrund/LoRAMoE`의 `base + p(e|x) B_e A_e x`와
  `alpha/r` scaling을 유지한다. arXiv:2602.12587에 따라 MLP
  `gate_proj/up_proj/down_proj`, 4 experts/layer, top-1을 적용한다. 공식 repo의
  optional BLC 기본값과 target paper의 공개 설정에 맞춰 추가 aux/z loss는 0이다.

Qwen3-0.6B에서 세 방법 모두 `C-STANCE → FOMC`, task당 1 step 실제 torchrun
smoke를 통과했다. EWC Fisher nonzero/finite, GEM task-gradient state 2개,
LoRAMoE `4 experts + top-1 + full-softmax probability + no aux/z` metadata를
checkpoint에서 재검증했다.

### 9.4 Qwen3-0.6B 6종 학습 완료 (2026-07-20)

출력 루트는 `output/qwen06_2epoch_fast6`이며 모든 method에 숫자 checkpoint
`0..7`이 존재한다. Paper baseline은 `result.json status=completed`, Track1은
`7/lora_moe_meta.json`과 `7/pytorch_model.bin` 및 로그의
`Sucessful saving model after round 7`을 확인했다.

| method | 상태 | 완료 task | 기록된 train seconds | 비고 |
|---|---|---:|---:|---|
| SeqLoRA | 완료 | 8/8 | 5,010.99s | 공유 rank-8 FFN LoRA |
| LoRAMoE | 완료 | 8/8 | 14,197.27s | 4 experts/layer, top-1 |
| EWC | 완료 | 8/8 | 5,302.65s | online squared-gradient Fisher |
| GEM | 완료 | 8/8 | 17,149.18s | task 0~6 gradient constraint를 보존해 20M 완료 |
| O-LoRA | 완료 | 8/8 | 6,931.60s | task별 adapter + orthogonal penalty |
| Track1 | 완료 | 8/8 | 별도 phase 로그 | task당 expert 1개, phase1 + router-only phase2 |

중간 운영/수정 사항:

- 공용 dataset cache를 두 학습 프로세스가 동시에 다시 쓰던 race를 제거했다.
  완성된 cache는 재사용하며, 24개 `.pt`를 `torch.load`로 검증했다.
- 2-way wave barrier를 동적 GPU work queue로 교체했다. 한 method가 끝나면 다른
  method를 기다리지 않고 빈 GPU에서 다음 method를 시작한다.
- GEM 20M에서 과거 7개 gradient의 CPU→GPU 복사를 parameter/step마다 반복하고
  작은 active-set QP를 CUDA kernel 수천 개로 풀어 약 19.3s/step까지 느려졌다.
  gradient 7세트(168 parameter/task, 약 38.5MB)는 `gem/6` state에 그대로 보존하고,
  task 시작 시 GPU에 한 번 cache하며 동일한 2^7 active-set QP를 NumPy/LAPACK으로
  풀도록 최적화했다. 기존 solver와 30개 무작위 QP 비교 최대 절대 오차는
  `3.58e-7`; 재시작 후 약 3s/step으로 감소했다.
- Track1은 task 0~2를 1-GPU batch `56,40,6,...`로 마친 뒤 round 2 checkpoint에서
  resume했다. task 3~7은 A100 2장, per-GPU batch
  `28,20,3,3,3,50,72,3`으로 실행해 global batch를 이전과 동일하게 유지했다.
  resume loader는 expert/router 3개를 복원하고 task 0~2를 건너뛴다. optimizer는
  원래 각 phase 경계마다 재생성되므로 task 경계 resume에 optimizer-state 차이가 없다.
- Track1 `past_task_ratio=1.0`은 각 round phase2에서 지금까지 본 모든 task의
  5,000 samples를 전량 replay한다. 마지막 round는 global batch 6으로 6,667 step,
  약 3h37m이 걸렸다. 후속 실험에서는 동일 조건으로 `0.1` 또는 `0.2` ablation을
  권장하지만, 이번 비교 run은 전 round 1.0을 유지했다.

### 9.5 평가 프로토콜 및 파서 검증

논문의 headline 정의에 맞춰 모든 primary score를 0~100으로 통일한다.
Py150 similarity와 20Minuten SARI는 원래 0~100이며, accuracy/ROUGE-L은 저장값(0~1)에 100을 곱한다.

- **OP** = final checkpoint 7의 8개 task primary score 평균.
- **BWT** = task 0~6 각각에 대해 `final score - 해당 task 학습 직후 checkpoint score`를
  계산한 뒤 7개 평균(percentage points). 마지막 task는 이후 학습이 없으므로 제외한다.
- primary metric: CS/FM/SQ/NC/ND=`accuracy`, MB=`ROUGE-L`, PY=`similarity`, 20M=`SARI`.
- 8×8 전체 matrix(64 task evaluation/model) 대신 final 8개 + BWT diagonal 7개,
  즉 **15 task evaluation/model**로 같은 OP/BWT를 계산한다.
- generation은 greedy(`temperature=0`), prompt 1536, answer upper bound 512,
  length bucketing 및 eval batch 64를 사용한다.
- task별 generation ceiling: CS/FM=8, MB=512, PY=160, SQ=512,
  NC/ND=16, 20M=256. Qwen tokenizer로 test GT를 전수 확인했다
  (CS/FM max 1, PY max 121, NC/ND max 5, 20M max 199 tokens).
- 선택지/숫자 parser, task stop marker, generation 후 원래 sample 순서 복원을 검증했다.
  8개 task GT를 prediction으로 넣은 normalization round-trip은 전부 일치한다.
  ScienceQA reasoning 내부의 `answer A`나 관사 `a`를 label로 오인하지 않도록
  출력 맨 앞 label 또는 명시적 answer cue만 인정한다.
- 20M SARI는 외부 Hub/package 없이 Hugging Face/Tensor2Tensor 정의
  `(F1_add + F1_keep + P_delete)/3`, 1~4 gram 평균을 로컬 구현했다. 공식 예제
  `26.953601953601954` 및 self-score 100을 재현했다.

관련 파일:

- 2-GPU 6-model 평가 큐: `scripts/eval_qwen06_six_2gpu.sh`
- 결과 집계: `scripts/summarize_five_baselines.py`
- 평가 출력: `eval_out/qwen06_2epoch_fast6_six_models/`
- 최종 표: `six_model_results.{json,csv,md}` (평가 완료 시 생성)

### 9.6 6-model OP/BWT 평가 실행 기록

GPU당 모델 하나, 동적 2-GPU queue로 실행했다. 느린 Track1/LoRAMoE를 먼저
시작하고 빈 GPU가 `GEM → O-LoRA → EWC → SeqLoRA`를 가져갔다. 중단 후
재실행 시 이미 존재하는 `results-*.json`을 건너뛰도록 구성했다.

| method | 평가 상태 | 결과 파일 |
|---|---|---:|
| Track1 | 완료 | 15/15 |
| LoRAMoE | 완료 | 15/15 |
| GEM | 완료 | 15/15 |
| O-LoRA | 완료 | 15/15 |
| EWC | 완료 | 15/15 |
| SeqLoRA | 완료 | 15/15 |

6개 method 모두 완료됐으며 OOM/traceback/runtime error는 없다. 집계기는
task별 final score, OP, BWT, task별 BWT를 JSON, CSV, Markdown으로 기록했다.

### 9.7 Qwen3-0.6B 6-model 최종 성능 (평가 완료)

2026-07-20 21:56 UTC(2026-07-21 06:56 KST)에 6개 method 모두
`final 8개 + diagonal 7개 = 15/15` 평가를 완료했다. 평가 로그 전수 검색에서
OOM, traceback, runtime error, NaN은 발견되지 않았다. 모든 값은 0~100 scale이며
BWT 단위는 percentage point다. 굵은 값은 열별 최고 성능이다.

| Method | OP ↑ | BWT ↑ | C-STANCE | FOMC | MeetingBank | Py150 | ScienceQA | NumGLUE-cm | NumGLUE-ds | 20Minuten (SARI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SeqLoRA | 31.90 | -5.54 | 34.35 | 49.60 | 27.41 | 40.62 | 19.30 | 11.11 | 36.31 | 36.48 |
| LoRAMoE | 33.27 | -5.34 | 34.80 | 40.93 | 24.91 | 32.74 | 53.75 | 7.41 | 35.38 | 36.25 |
| EWC | 34.22 | -7.20 | 35.50 | 50.60 | 28.57 | 36.47 | 32.95 | 14.81 | 38.46 | 36.37 |
| GEM | 17.34 | -13.51 | 13.20 | 3.63 | 14.35 | 40.69 | 3.45 | 8.64 | 18.46 | 36.29 |
| O-LoRA | 41.64 | -3.77 | **50.85** | **59.88** | 25.24 | **49.15** | 48.70 | 18.52 | **43.38** | 37.42 |
| **Track1 (Ours)** | **44.84** | **-0.79** | 50.55 | 49.60 | **37.72** | 49.03 | **68.75** | **23.46** | 42.15 | **37.48** |

Task별 backward transfer(`final − task 학습 직후`)는 다음과 같다. 마지막 task인
20Minuten은 이후 학습이 없으므로 BWT 평균에서 제외한다.

| Method | C-STANCE | FOMC | MeetingBank | Py150 | ScienceQA | NumGLUE-cm | NumGLUE-ds | 평균 BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SeqLoRA | -17.85 | -17.94 | -11.58 | -0.15 | -15.60 | +7.41 | +16.92 | -5.54 |
| LoRAMoE | -18.85 | -19.96 | -14.87 | +0.78 | -2.65 | +1.23 | +16.92 | -5.34 |
| EWC | -16.70 | -17.34 | -9.40 | +2.12 | -11.55 | +1.23 | +1.23 | -7.20 |
| GEM | -39.00 | -51.21 | -25.28 | +6.32 | -6.65 | +6.17 | +15.08 | -13.51 |
| O-LoRA | -1.75 | -4.23 | -13.96 | +8.95 | -16.30 | -1.23 | +2.15 | -3.77 |
| **Track1 (Ours)** | **-2.50** | **-2.42** | **-0.73** | +2.53 | **+0.65** | -3.70 | +0.62 | **-0.79** |

핵심 관찰:

- Track1은 **OP 44.84**로 2위 O-LoRA(41.64)보다 **+3.20p**, 최고 baseline인
  O-LoRA 대비 BWT도 **+2.97p** 높아, 전체 성능과 망각 억제를 동시에 개선했다.
- Track1은 MeetingBank, ScienceQA, NumGLUE-cm, 20Minuten에서 최고이며 특히
  ScienceQA는 O-LoRA보다 +20.05p 높다. O-LoRA는 C-STANCE, FOMC, Py150,
  NumGLUE-ds에서 최고다.
- Track1의 task별 BWT는 7개 중 3개가 절대값 1p 미만이고 최악도 -3.70p다.
  반면 SeqLoRA/LoRAMoE/EWC는 초기 분류 task에서 약 -17~-20p의 망각이 나타난다.
- GEM은 OP 17.34/BWT -13.51로 명확한 outlier다. 평가 실패는 없으므로 현재 값은
  파싱 누락보다는 학습 결과를 반영한다. 특히 C-STANCE/FOMC/MeetingBank의 큰
  음의 BWT가 전체 성능 저하의 주원인이다.

원본 산출물:

- `eval_out/qwen06_2epoch_fast6_six_models/six_model_results.json`
- `eval_out/qwen06_2epoch_fast6_six_models/six_model_results.csv`
- `eval_out/qwen06_2epoch_fast6_six_models/six_model_results.md`

---

## 10. TreeLoRA 공식 코드 기반 LLaMA-2-7B TRACE 재현

### 10.1 목적과 코드 기준

기존 `llmcl_benchmark`의 GEM/EWC는 Qwen의 FFN linear module에 LoRA를 붙인
자체 비교 구현이므로, TRACE/TreeLoRA에서 보고된 full-parameter GEM/EWC와 직접
동일하지 않다. 공식 구현의 신뢰성과 논문 수치 재현 가능성을 먼저 확인하기 위해
TreeLoRA 공개 저장소의 코드를 수정하지 않고 LLaMA-2-7B-chat에 적용한다.

- 공식 저장소: `/home/work/Agent_HJ/30_flame_agent/TreeLoRA`
- 확인한 revision: `1c7260c42b34e1961283797c742f08b9c3842501`
- backbone: `/home/work/Agent_HJ/00_models/Llama-2-7b-chat-hf`
- 별도 환경: `/home/work/Agent_HJ/30_flame_agent/envs/treelora_env`
- 주요 환경: Python 3.10, PyTorch 2.4.1+cu121, Transformers 4.45.2,
  DeepSpeed 0.15.3, FlashAttention 2.6.3, BF16, ZeRO Stage 2
- 모델 파일과 config/tokenizer를 로컬에서 검증했으며 LLaMA hidden size 4096,
  32 layers, vocab size 32,000이다.

FlashAttention 확장의 BF16 forward/backward smoke test는 통과했다. 다만 현재
Transformers 4.45.2에서 실제 LLaMA attention class는 `LlamaSdpaAttention`으로
표시된다. 따라서 확장 설치 성공과 실제 학습 forward에서 FlashAttention kernel을
사용한다는 것은 구분해서 해석해야 한다.

### 10.2 논문과 공개 실행 설정의 불일치

TreeLoRA 논문 부록은 TRACE task당 5,000개, batch size 4라고 기술하지만, 공개
저장소에 포함된 데이터와 실행 스크립트는 모두 `LLM-CL-Benchmark_500`을 가리킨다.
공개 스크립트의 기본값은 4 GPU × micro-batch 1 × gradient accumulation 8로
effective global batch 32다. LR 설명도 부록 안에서 1e-4/1e-5와 1e-3이 충돌하며,
실제 공개 코드는 LoRA 계열 1e-4, GEM/EWC 1e-5를 사용한다.

현재 재현은 우선 공개 코드에서 직접 확인할 수 있는 TRACE-500 조건만 수행한다.
TRACE-5000 실험은 이 큐에 포함하지 않았다.

| 방법 | 구현 | task별 epoch | LR | samples/task |
|---|---|---|---:|---:|
| O-LoRA | 공식 q/v LoRA 구현 | 5,3,7,5,3,5,5,7 | 1e-4 | 500 |
| GEM | 공식 full-parameter GEM | 1,1,5,5,1,5,5,5 | 1e-5 | 500 |
| EWC | 공식 full-parameter EWC | 1,1,5,5,1,5,5,5 | 1e-5 | 500 |

세 방법 공통 실행 조건은 다음과 같다.

- GPU: 2 × A100 80GB (`GPU 0,1`)
- per-device batch: 2
- gradient accumulation: 8
- effective global batch: `2 GPUs × 2 × 8 = 32`
- max prompt/answer: 1024/512
- BF16, DeepSpeed ZeRO-2, cosine scheduler, seed 1234
- 학습 후 8-task continual evaluation과 `final_results_new.txt` 집계를 연속 수행

### 10.3 실행 큐와 자동 후속 작업

2026-07-21 20:14 KST 기준 아래 순서의 큐를 시작했다.

1. O-LoRA TRACE-500
2. GEM TRACE-500
3. EWC TRACE-500

- 큐 스크립트: `TreeLoRA/scripts/run_olora_gem_ewc_500_only.sh`
- 큐 PID/PGID: `2170551`
- 큐 로그: `TreeLoRA/logs/reproduction/olora_gem_ewc_500_only_queue.log`
- 출력 root: `TreeLoRA/outputs_LLM-CL/reproduction/`
- 상태: O-LoRA 실행 중, GEM/EWC 대기

OOM 로그(`CUDA out of memory`, `OutOfMemoryError`)가 확인되면 10초 후
`/home/work/Agent_HJ/30_flame_agent/agent_data_make.py`를 자동 실행한다. 세 실험과
평가가 모두 정상 완료되어 EWC의 `predictions/final_results_new.txt`가 생성된
경우에도 같은 스크립트를 실행한다. 중복 실행은 `pgrep`으로 차단한다.

이미 시작된 큐에도 별도 감시기
`TreeLoRA/scripts/watch_trace500_then_agent_data_make.sh`를 연결했다. 후속 작업 로그는
`TreeLoRA/logs/reproduction/agent_data_make.log`, 감시기 로그는
`TreeLoRA/logs/reproduction/agent_data_make_watcher.log`에 저장된다.

### 10.4 O-LoRA 학습 완료 및 GEM/EWC 학습 전용 재개

2026-07-21 20:56 KST에 O-LoRA의 8개 task 학습과 checkpoint 0~7 저장이
정상 완료됐다. 학습 로그에는 CUDA OOM이 없다. 이어진 평가는 checkpoint 0에서
예측 subprocess가 `/bin/dash: deepspeed: not found`로 실패했고, 빈 prediction을
accuracy로 집계하면서 `ZeroDivisionError`가 발생했다. 이는 학습 실패나 prediction
parser 문제가 아니라 `infer_multi_command.py`가 생성한 자식 shell에 TreeLoRA
가상환경의 `PATH`가 전달되지 않은 실행환경 문제다.

O-LoRA는 재학습하지 않고 checkpoint를 보존했으며, 2026-07-21 21:56 KST에
GEM → EWC 순서의 **학습 전용** 큐를 새로 시작했다.

- 큐 PID/PGID: `2235585`
- 큐 스크립트: `TreeLoRA/scripts/run_gem_ewc_500_train_only.sh`
- 큐 로그: `TreeLoRA/logs/reproduction/gem_ewc_500_train_only_queue.log`
- 현재 상태: GEM 실행 중, EWC 대기
- 두 방법 모두 `TRAIN_ONLY=1`로 평가를 건너뛴다.
- GEM/EWC 학습이 모두 끝나거나 CUDA OOM이 확인되면 기존 약속대로
  `agent_data_make.py`를 한 번만 실행한다.

동시에 O-LoRA 및 GEM/EWC 평가 wrapper가 전용 환경의 `deepspeed` 경로를
명시적으로 `PATH` 앞에 추가하도록 수정했다. 학습 완료 후 저장된 checkpoint를
사용해 세 방법의 평가만 별도로 수행할 수 있다.


---

## 11. LLaMA-2-7B Ours_LoRA_MoE same-rank 비교

### 11.1 TreeLoRA 공개 O-LoRA 재현

- LLaMA-2-7B-chat, O-LoRA rank 8/alpha 32/dropout 0.1, attention q/v
- TRACE-500, epoch `5,3,7,5,3,5,5,7`, LR 1e-4
- 2 GPU × batch 2 × accumulation 8 = global batch 32
- checkpoint 0~7 모두 저장, 학습 시간 약 43분(20:13:20~20:56:25 KST)
- 평가 subprocess의 `deepspeed` PATH 문제를 수정했다.
- OP/BWT 평가는 중간 diagonal과 final 8개만 생성하는 sparse 방식으로 변경했고
  inference batch를 공식 parser 기본값 4로 설정했다.
- 결과는 `predictions/final_results_sparse.txt`에 공식 clipped BWT(/8)와
  표준 signed BWT(/7)를 함께 기록한다.

공개 TRACE-500은 train 500/task, test 합계 781이고 로컬 TRACE-5000은
train 5,000/task, test 합계 7,794다. 현재 run은 논문 서술 대비 학습·평가 모두
전체 합계 기준 약 1/10인 공개 경량 설정이다.

공식 GEM은 full-parameter 학습이며 2×A100에서 첫 checkpoint 전에 명시적 CUDA
traceback 없이 SIGKILL(-9)로 종료됐다. EWC는 시작하지 못했다.

### 11.2 LLaMA-2-7B 성장형 FFN LoRA-MoE

`Ours_LoRA_MoE`를 LLaMA dense MLP의 gate/up/down에 적용한다.

| 항목 | 설정 |
|---|---|
| backbone | Llama-2-7b-chat-hf |
| data | 공개 TRACE-500 |
| epoch | 5,3,7,5,3,5,5,7 |
| optimizer | AdamW, LR 1e-4, cosine |
| expert | task당 1개, rank 8, alpha 32 |
| routing | top-1, full-softmax weight |
| phase 1 | 새 MLP LoRA expert + router |
| phase 2 | 모든 expert freeze, 누적 task 전량 replay, router-only 1 epoch |
| frozen | backbone 전체와 attention |
| batch | 2 GPU × micro 1 × accumulation 16 = global 32 |
| sequence | prompt 1024, answer 512 |
| resume | `RESUME_CHECKPOINT=.../N` |

이는 O-LoRA와 동일 rank 비교지만 target module이 달라 activated parameter는
같지 않다. O-LoRA q/v rank 8은 약 4.19M, 우리 MLP gate/up/down rank 8 expert는
약 11.6M(+router)다. parameter-matched 비교는 후속 rank 3 실험으로 분리한다.

학습 wrapper는 `scripts/train_Ours_LoRA_MoE_llama2_trace500.sh`, 기본 출력은
`output/ours_lora_moe_llama2_7b_trace500_r8_ept1_gb32`이다. 정상 완료, CUDA OOM/SIGKILL, 그 밖의 non-zero 종료를 포함해 학습 프로세스가
어떤 상태로 끝나더라도 `agent_data_make.py`를 중복 없이 자동 실행한다.
본 학습 전 forward/backward, expert 성장, phase-2 router-only, save/resume smoke
test를 수행한다.

2026-07-22 00:20 KST에 O-LoRA sparse 평가 PID `2347667`의 정상 종료를
감시하는 `scripts/watch_olora_eval_then_train_ours_llama2.sh`를 PID `2379221`로
연결했다. `final_results_sparse.txt`가 생성된 경우에만 위 학습 wrapper를 자동
시작하며, 평가 실패로 학습을 시작하지 못한 경우에도 `agent_data_make.py`를 즉시 실행한다.
동일 학습이 이미 실행 중이면 중복 학습/부하 프로세스를 만들지 않고, 기존 학습
wrapper가 종료 시 `agent_data_make.py`를 실행한다.
감시 로그는 `logs/watch_olora_eval_then_train_ours_llama2.log`, 학습 launcher 로그는
`logs/ours_lora_moe_llama2_trace500_launcher.log`다.


### 11.3 O-LoRA TRACE-500 최종 복구 결과

20Minuten 추론 100개는 완료됐지만 TreeLoRA 자체 `metrics.py`가
`datasets.load_metric("sari")`로 GitHub에 접근하면서 오프라인 환경에서 실패했다.
저장된 `check_output/20Minuten_O_LoRA/results_20260722_004832.json`의 source,
prediction, GT를 `llmcl_benchmark/metrics.py`의 HF/Tensor2Tensor-compatible 로컬
SARI로 재채점했으며 재추론은 하지 않았다. SARI는 **41.7758**이다.

| 지표 | 결과 |
|---|---:|
| OP / Last Average | 39.8561 |
| 공식 clipped BWT (/8) | -10.6379 |
| 표준 signed BWT (/7) | -12.1576 |

| Task | 학습 직후 | Final | Delta |
|---|---:|---:|---:|
| C-STANCE | 51.0000 | 46.0000 | -5.0000 |
| FOMC | 57.0000 | 44.0000 | -13.0000 |
| MeetingBank | 43.3658 | 16.2828 | -27.0830 |
| Py150 | 59.8100 | 39.5400 | -20.2700 |
| ScienceQA | 64.0000 | 58.0000 | -6.0000 |
| NumGLUE-cm | 35.0000 | 21.2500 | -13.7500 |
| NumGLUE-ds | 52.0000 | 52.0000 | +0.0000 |
| 20Minuten | 41.7758 | 41.7758 | +0.0000 |

최종 파일은 TreeLoRA O-LoRA prediction 디렉터리의
`final_results_sparse.txt`다. 우리 방법 LLaMA-2-7B 학습은 사용자 지시에 따라
보류했다.

---

## 12. 새 Codex 세션 인수인계 (2026-07-23 기준, 이 절을 우선 적용)

이 절은 대화 기록이 없는 새 세션에서 즉시 재개하기 위한 authoritative handoff다.
위 절의 과거 PID와 “실행 중/대기” 표시는 당시 이력이다. 새 세션에서는 이 절을
우선 읽고 실제 프로세스와 GPU를 다시 확인한다.

### 12.1 디렉터리, 환경, 모델과 데이터

주 작업 환경:

```bash
cd /home/work/Agent_HJ/30_flame_agent/llmcl_benchmark
source /home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
```

`train_env`: Python 3.10.12, torch 2.8.0+cu128, transformers 4.57.6,
deepspeed 0.16.9, peft 0.18.1, accelerate 1.14.0, datasets 5.0.0.
Qwen, OLMoE, 우리 방법 학습과 현재 평가에 이 환경을 사용한다.

TreeLoRA 공식 코드 재현 전용 환경:

```bash
cd /home/work/Agent_HJ/30_flame_agent/TreeLoRA
source /home/work/Agent_HJ/30_flame_agent/envs/treelora_env/bin/activate
export PATH=/home/work/Agent_HJ/30_flame_agent/envs/treelora_env/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
```

`treelora_env`: Python 3.10.12, torch 2.4.1+cu121, transformers 4.45.2,
deepspeed 0.15.3, accelerate 1.0.1, datasets 2.18.0, flash-attn 2.6.3.
TreeLoRA revision은 `1c7260c42b34e1961283797c742f08b9c3842501`이다. 이 코드를
`train_env`에서 실행하지 않는다.

| model | local path |
|---|---|
| Qwen3-0.6B | `/home/work/Agent_HJ/00_models/Qwen3-0.6B` |
| Qwen3-8B | `/home/work/Agent_HJ/00_models/Qwen3-8B` |
| LLaMA-2-7B-chat | `/home/work/Agent_HJ/00_models/Llama-2-7b-chat-hf` |
| OLMoE-1B-7B-0125 | `/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125` |

| TRACE | path | 용도 |
|---|---|---|
| TRACE-500 | `TreeLoRA/data/LLM-CL-Benchmark/LLM-CL-Benchmark_500` | 공개 경량 재현, train 500/task |
| TRACE-5000 | `llmcl_benchmark/data/LLM-CL-Benchmark_5000` | train 5,000/task, test 총 7,794 |

표준 task 순서는 `C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA →
NumGLUE-cm → NumGLUE-ds → 20Minuten`이다.

### 12.2 최종 평가 프로토콜

현재 최종 비교는 final checkpoint의 8개 task와 task 학습 직후 diagonal 7개,
총 15개만 추론하는 sparse 방식이다. 전체 8×8 행렬은 불필요하다.

- OP: final 8개 task 점수 평균
- signed BWT: 이전 7개 task의 `final - task 학습 직후` 평균
- 20Minuten=SARI, MeetingBank=ROUGE-L, Py150=similarity, 나머지=accuracy
- max prompt 1536, task별 generation limit, greedy(`temperature=0`), length bucketing
- 과거 20Minuten ROUGE-L OP와 현재 SARI OP를 섞지 않는다.
- LoRA/paper baseline: `evaluate_Ours_LoRA_MoE.py`
- OLMoE growing FFN: `evaluate_Ours_MoE_FFN.py`
- 공통 parser/scorer: `vllm_eval.py`, `metrics.py`, `utils/eval_generation.py`

### 12.3 Qwen3-0.6B 여섯 모델 — 모두 완료

학습 root `output/qwen06_2epoch_fast6`. TRACE-5000, task당 2 epoch, LR 1e-4,
rank 8. 집계 파일:

```text
eval_out/qwen06_2epoch_fast6_six_models/six_model_results.json
eval_out/qwen06_2epoch_fast6_six_models/six_model_results.csv
eval_out/qwen06_2epoch_fast6_six_models/six_model_results.md
```

| method | OP (20M=SARI) | BWT |
|---|---:|---:|
| SeqLoRA | 31.90 | -5.54 |
| LoRAMoE | 33.27 | -5.34 |
| EWC | 34.22 | -7.20 |
| GEM | 17.34 | -13.51 |
| O-LoRA-FFN 변형 | 41.64 | -3.77 |
| Track1 Ours | **44.84** | **-0.79** |

이 표의 O-LoRA는 원 논문 구현이 아니다. FFN `gate/up/down_proj`에 task별
rank-8 LoRA를 붙인 자체 변형이며 최종 adapter 약 22.02M이다. 공식 O-LoRA는
attention `q_proj/v_proj`다.

GEM도 원 논문의 episodic replay GEM이 아니다. `model/GEM_LoRA.py`는
`model.paper_baselines.GEMLoRA` wrapper이며 replay sample은 **0개**다. 각 task의
마지막 minibatch gradient 한 벌을 BF16/CPU로 저장한다. trainable LoRA
2,752,512개 기준 약 5.5MB/task, 8 tasks 약 44MB다. 이는 TRACE 저장소의
stored-gradient GEM 변형이므로 낮은 결과를 원 논문 GEM 성능으로 해석하지 않는다.

### 12.4 Qwen 공식형 attention O-LoRA — 완료, 성능 저하

학습 root: `output/olora_original_qwen06_trace5000_e2_2gpu`. attention q/v,
task마다 rank-8 하나, alpha 32, dropout 0.1, TRACE-5000, 각 2 epoch, LR 1e-4.
과거 LoRA는 freeze하고 전부 누적 적용한다. 최종 `r_sum=64`는 backbone merge가
아니라 과거 A/B concatenate다. checkpoint 0→7의 과거 prefix 변화량은 모두 0.0.

결과:

```text
eval_out/olora_original_qwen06_trace5000_e2_2gpu_prompt1536/
  olora_original_results_both.json
```

| metric | value |
|---|---:|
| OP (20M=SARI) | 27.80 |
| BWT | -16.56 |
| legacy OP (20M=ROUGE-L) | 25.91 |

C-STANCE `52.90→13.90`, ScienceQA `53.55→11.15`로 붕괴했다. prompt 1536에서도
같으므로 parser/truncation 문제가 아니다. 다음 진단은 각 checkpoint의
C-STANCE/ScienceQA cross-round 평가로 붕괴를 유발한 후속 task를 찾는 것이다.
데이터 증가만으로 원인을 단정하지 않는다.

FFN 변형(OP 41.64)과 공식형 q/v(OP 27.80)은 target, task당 용량(약 2.75M vs
1.15M), dropout(0 vs 0.1), batch, scheduler가 함께 다르다. 통제 실험 없이
target 하나만의 인과로 결론 내리지 않는다.

### 12.5 LLaMA-2-7B 공식형 O-LoRA — TRACE-500 완료

공식 확장 semantics 결과:

```text
/home/work/Agent_HJ/30_flame_agent/TreeLoRA/outputs_LLM-CL/reproduction/
  olora_llama2_7b_trace500_original_r8_mb2_gb32_seed1234/
  predictions_official_diagonal/final_results_sparse.txt
```

attention q/v, rank 8, alpha 32, dropout 0.1, epoch `5,3,7,5,3,5,5,7`,
LR 1e-4, 2 GPUs × micro 2 × accumulation 8 = global batch 32.

| metric | value |
|---|---:|
| OP | 41.5441 |
| 공식 clipped BWT (/8) | -2.8773 |
| 표준 signed BWT (/7) | -2.6098 |

`olora_llama2_7b_trace500_mb2_gb32_seed1234`는 확장되지 않은 과거 run(OP
39.8561/BWT -12.1576)이므로 혼동하지 않는다. LLaMA 기반 우리 LoRA-MoE는
사용자 지시로 보류되어 `output/ours_lora_moe_llama2_7b_trace500_r8_ept1_gb32/0`
하나만 존재하며 최종 결과가 아니다.

### 12.6 OLMoE Track2 Ours_MoE_FFN — 학습/평가 완료

코드: `model/Ours_MoE_FFN.py`, `training/main_Ours_MoE_FFN.py`,
`evaluate_Ours_MoE_FFN.py`. 학습 root:

```text
output/track2_OLMoE_ept1_force_upper_5k_seed1234
```

checkpoint 0~7 모두 존재한다. native expert 64개 위에 task당 full FFN expert
1개를 추가해 최종 new expert 8개다. phase-1은 새 expert routing force,
phase-2는 expert freeze 후 누적 task replay로 router를 재보정했다.

평가 재실행:

```bash
cd /home/work/Agent_HJ/30_flame_agent/llmcl_benchmark
source /home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/activate
EVAL_BATCH=16 bash scripts/eval_track2_olmoe_sparse_2gpu.sh
```

결과: `eval_out/track2_OLMoE_ept1_force_upper_5k_seed1234_sparse/cl_summary_sparse.json`.

| metric | value |
|---|---:|
| OP | **55.1038** |
| BWT | **-0.1037** |

| task | learned | final | delta |
|---|---:|---:|---:|
| C-STANCE | 50.45 | 42.85 | -7.60 |
| FOMC | 68.75 | 70.36 | +1.61 |
| MeetingBank | 49.21 | 48.88 | -0.34 |
| Py150 | 58.04 | 56.43 | -1.61 |
| ScienceQA | 75.95 | 77.00 | +1.05 |
| NumGLUE-cm | 41.98 | 45.68 | +3.70 |
| NumGLUE-ds | 57.54 | 60.00 | +2.46 |
| 20Minuten | — | 39.64 SARI | — |

driver의 완료 marker는 `OLMOE_SPARSE_EVALUATION_COMPLETE`다. 과거
`eval_out/track2_OLMoE_ept1_force_upper_5k_seed1234.log`의 OOM은 이전 실패
시도이며 최종 상태가 아니다.

### 12.7 새 세션 시작 점검과 세션 유지

```bash
cd /home/work/Agent_HJ/30_flame_agent/llmcl_benchmark
ps -eo pid,pgid,etime,cmd | grep -E \
  'train_|evaluate_|agent_data_make.py|torchrun|deepspeed' | grep -v grep
nvidia-smi
git status --short
```

`agent_data_make.py`는 GPU 세션 유지용이고 실행 중이면 GPU당 약 60GB를 쓸 수
있다. 새 학습/평가 전 실제 PGID를 확인해 정상 종료하고 과거 PID를 재사용하지
않는다. 작업 후 세션 유지가 필요하면:

```bash
cd /home/work/Agent_HJ/30_flame_agent
setsid python agent_data_make.py \
  >> llmcl_benchmark/eval_out/agent_data_make.log 2>&1 < /dev/null &
echo $!
```

worktree에는 사용자 실험 변경이 누적돼 있다. `git reset --hard`, 광범위한
checkout/삭제를 하지 않는다. 결과를 덮기 전 완료 marker와 기존 JSON을 확인하고,
가능하면 이미 존재하는 task result skip 기능을 사용한다.
