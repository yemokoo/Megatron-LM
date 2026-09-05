# TRACE self-generated replay — 2026-09-01 실험 일지 및 finding

목표: 실데이터 버퍼 없이 (self-generated replay) AA ≥ 61 / BWT ≥ −3. 기준 lm(실데이터 리플레이) 63.51 / −0.74.
구조: Llama-3.1-8B-Instruct + v3 LoRA-MoE (task당 expert 1, top-1, expert는 자기 task 후 고정, 공유 라우터).
리플레이는 라우터만 학습; 생성 데이터가 expert에 닿는 경로는 KD-init(새 expert+라우터, teacher=옛 expert만 쓴 같은 모델)뿐.

## 1. dawn 체인 평가 (`selfgen_cl_fix_20260901`, 매 라운드 전 task 재생성)
- 왜: 어제 고친 생성(prompt-cue, Py150 앵커 히스토그램, greedy 정답)이 forgetting을 줄이는지.
- 결과: 고친 두 task가 오히려 최악. MeetingBank −9.59 → −13.59, Py150 −1.55 → −5.73, FOMC 71.8→62.0(r1→r7).
- 이어진 것: "생성 데이터 vs 실제 리플레이 데이터" 직접 비교.

## 2. 생성 데이터 품질 비교 (task별, 첫 라운드 vs 마지막 라운드 vs 실데이터)
- 왜: 리플레이 데이터 품질 말고는 설명이 안 됨.
- 결과: 첫 생성은 실데이터와 맞음(라벨 분포·형식 100%·길이, unigram JS ≈ 노이즈 바닥). 재생성은 라운드마다 표류:
  FOMC 라벨 A 26→2%, C 49→89%; ScienceQA Choices 93→45% → letter 정답 97→55%; MeetingBank 정답 회의록 문체 0.2→16.5%, 길이 310→176; Py150 프롬프트 1,123→145자.
  지시 부분은 유지(JS 0.67–0.77×), 무너지는 건 정답/구조.
- 이어진 것: (a) 학습 설정 점검, (b) 지시 생성 자체 조사, (c) guard(fix2), (d) 생성 시점 고정(frozen).

## 3. 학습 설정 점검
- 왜: 리플레이 데이터 외 v3 lm과 같은지.
- 결과: dawn은 4 GPU(배치 32)로 lm(64)보다 업데이트 2배. → FOMC 이후 배치 64로 통일. 평가 배치는 점수에 무관.

## 4. 지시(stage A) vs 정답(stage B) 분리 조사 (C-STANCE object-first, base vs CL)
- 왜: 지시가 망가지는지 정답이 망가지는지.
- 결과: 영양 없음(중립 편향·object 언급은 실데이터와 무관하지 않음). 사용자 결정으로 중단.

## 5. fix2/fix2b (라벨 quota·reject guard)
- 왜: 라벨 붕괴를 분포 guard로 막을 수 있는지.
- 결과: round 3에서 중단. quota가 라벨–본문 불일치 ~10% 유발 → 답이 아님. 라벨 붕괴는 증상(정답 토큰은 리플레이 loss의 ≈1%).

## 6. frozen 체인 (`selfgen_cl_frozen_20260901`, task 직후 1회 생성 후 재사용)
- 왜: "첫 생성 품질"과 "재생성 표류"를 분리.
- 결과: **AA 62.04 / BWT −2.21 (목표 안)**. task별 BWT: C-STANCE −1.95, FOMC −1.61, MeetingBank −3.36, Py150 −2.55, ScienceQA −3.20, cm −1.23, ds −1.54; 20Minuten 42.3.
  단 생성 레코드를 저장(500/task = lm 메모리와 같은 크기)하므로 버퍼-프리가 아님 → 진단 결과이지 최종 해법이 아님.
- 이어진 것: 왜 dawn만 무너지는지 hidden/라우터/expert 수준에서 확인.

## 7. hidden drift / FOMC 결정 부분공간 (lm vs dawn vs frozen, model/k→model/7)
- 결과 rel-L2: C-STANCE .30/.50/.33, FOMC .51/.57/.50, MeetingBank .35/.61/.49, Py150 .42/.56/.50, ScienceQA .44/.52/.46, cm .53/.60/.57, ds .51/.57/.58.
  FOMC 결정 부분공간 KL .23/1.02/.12, C 예측 207/333/204. → dawn만 튐, frozen ≈ lm. forgetting = hidden drift 논리 유지.

## 8. KD-init 프로브 (같은 시작점, 메모리 소스 real / gen-frozen / gen-dawn, 실제 held-out에서 student vs teacher)
- 왜: 생성 데이터가 expert에 닿는 유일한 경로(KD-init)가 망가지는지.
- 결과: KL real 대비 gen-frozen 0.9–1.4× (ScienceQA 1.6×, cm 2.3×), gen-dawn 1.0–1.4× (ScienceQA 2.5×); 절대값 0.01–0.16, 정답 CE·acc 동일. → **expert 경로는 원인이 아님**; 라우터-only 리플레이 단계가 남음.

## 9. 라우터 교체 생성 진단 (frozen model/7 + model/k 라우터 / model/7 그대로 / 첫 생성 / dawn r7)
- 왜: 표류가 expert 손상인지 라우터 이동인지; 최종 모델이 첫 생성을 재현하는지.
- 결과:
  FOMC A/B/C — 첫 생성 26/22/52, 라우터 교체 25/24/51, **model/7 그대로 21/24/56**, dawn 2/9/89.
  MeetingBank 프롬프트 길이 — 첫 3,652, 교체 3,619, dawn 3,502 (실데이터 15,751).
  Py150 stage A 길이 / 100자 미만 — 첫 1,123 / 15.5%, 교체 1,162 / 14.4%, **그대로 1,037 / 27.8%**, dawn 145 / 79.8%.
  → expert_k 무손상(라우터_k만 끼우면 정확 재현). frozen의 최종 라우터도 거의 보존(FOMC 약간, Py150은 짧은 생성 15→28%).
  → dawn 붕괴는 최종 모델의 성질이 아니라 "오염 재생성 → 라우터 학습 → 더 오염"의 복리 루프. 라우터 되돌리기(A)는 불필요(방법 실패 인정과 같음) — 진단용으로만.

## 10. 지시-only 리플레이 (진행 중)
- 왜: 버퍼-프리(매 라운드 재생성) 설정에서 되먹임 통로를 끊기 위해. 재생성물에서 먼저 무너지는 게 정답 토큰인데, 지시-only면 그 토큰이 리플레이 loss로 라우터에 되먹임되지 않음. 지시 부분은 r7까지 분포 유지.
- 구현: `scripts/selfgen/train_selfgen_instonly.py` (생성 레코드에만 answer_start 부착, collator에서 그 이후 라벨 −100; 실데이터·KD-init 무영향). 단위검증(122토큰 중 120 지시 라벨 유지, `A<|eot_id|>` 마스킹) 및 스모크(joint replay=1.0629 vs frozen 1.0536, 4×16 OOM 없음) 통과.
- 드라이버: `run_selfgen_cl_regen_instonly.sh` (dawn 드라이버 분기, round 0은 frozen model/0 시딩). 23:40 GPU 2,3,4,7 (PDB 16, global 64) 시작 → `/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_regen_instonly_20260901`.
  대조군(후순위): `run_selfgen_cl_instonly.sh` (frozen + 지시-only, 마스크 단독 효과).

## 종합 finding
1. 구조 변경 불필요: expert 고정 + 라우터-only 리플레이 + KD-init은 생성 데이터로도 작동.
2. 실패 원인은 재생성 루프의 복리 되먹임. 루프를 끊으면(frozen) 목표 안, 하지만 frozen은 합성 버퍼를 저장하므로 이상적이지 않음.
3. 되먹임 통로는 라우터-only 리플레이의 LM loss(정답 토큰 포함). expert 경로(KD-init)는 견딤.
4. lm과의 잔여 격차(BWT −2.2 vs −0.7)는 생성 프롬프트 길이(MeetingBank 3.6k vs 15.7k자, 1024 토큰 cap; Py150 1.1k vs 2.4k)와 정답 토큰 ~1% 불일치.

## 이제 볼 것
1. 재생성+지시-only 체인 라운드별: Py150 재생성 길이가 ~1,100자를 유지하는지(dawn은 145로 붕괴), FOMC 라벨 분포(A 26%대 유지 vs C 89% 붕괴), ScienceQA Choices 비율. 실패 신호가 나오면 그 라운드에서 중단.
2. 체인 완료 후 sparse-15: AA/BWT를 frozen(62.04/−2.21)·lm(63.51/−0.74)·dawn과 비교. 성공 기준은 dawn 대비 회복 + 목표 안.
3. 성공 시: frozen+지시-only 대조군으로 마스크 단독 효과 분리; MeetingBank/Py150 프롬프트 길이 cap(1024) 완화 실험.
4. 실패 시: 복리 진입 지점 분리 — KD-init 메모리만 실데이터/지시-only 리플레이 등 한 경로씩 교체.
