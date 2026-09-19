# HP sensitivity (TRACE) — 학습 순서 / 태스크당 expert 개수

모듈 ablation(`scripts/ablation/`)이 "각 구성요소가 필요한가"를 봤다면, 이쪽은
**HP를 흔들었을 때 결과가 얼마나 흔들리는가**를 본다. 기준 모델은 ablation의
all-on 셀(= **1phase** + KD-init on + self-generated replay + residual) 이고, 아래
두 축만 바꾼다. 나머지 플래그는 `scripts/ablation/run_arm_gen.sh` 와 문자
단위로 동일하다.

## 셀

| 셀 | 축 | experts/task | rank | top-k | alpha | rank 합 | 순서 |
|---|---|---|---|---|---|---|---|
| (기준) `1phase` + KD + gen + residual | — | 1 | 64 | 1 | 128 | 64 | 정순 |
| `order_reverse` | 1. 학습 순서 | 1 | 64 | 1 | 128 | 64 | **역순** |
| `e2_r32` | 2. expert 개수 | 2 | 32 | 2 | 64 | 64 | 정순 |
| `e4_r16` | 2. expert 개수 | 4 | 16 | 4 | 32 | 64 | 정순 |
| `e8_r8` | 2. expert 개수 | 8 | 8 | 8 | 16 | 64 | 정순 |

- **rank 합 = experts × rank = 64 고정.** 태스크당 용량은 같고 granularity만 변한다.
- **alpha = 2 × rank** (기준 모델의 `alpha/rank = 128/64 = 2` 를 유지). LoRA 실효
  스케일을 고정하려는 선택이다. 다른 규약(`alpha` 를 128로 고정 → 전문가당 스케일이
  rank 축소에 반비례해 커짐)을 쓰려면 `ALPHA=128` 로 덮어쓰면 된다. 참고로 리포지토리의
  고정 프로필 `v3_new_top4` 는 4×rank16 에 alpha 128 을 쓴다 — 즉 두 규약이 공존하며,
  둘을 섞으면 축이 오염된다.
- 역순: `20Minuten → NumGLUE-ds → NumGLUE-cm → ScienceQA → Py150 → MeetingBank → FOMC → C-STANCE`,
  에폭도 같이 뒤집는다(`7,5,5,3,5,7,3,5`).
- **residual expert 는 셀별로 experts/task 개수만큼** 붙는다(2/4/8). residual row 는
  LoRA 쌍을 갖지 않으므로 rank 개념이 없고, 개수만 태스크 그룹 크기에 맞춘다.

## 실행

```bash
H=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/scripts/hp_sensitivity

# 0) 스모크 (GPU 8장 필요, 셀당 2장 / 2라운드 / 각 1에폭, ~40분)
bash $H/smoke_hp.sh
tail -F /data2/seonghyeonnoh/paper/hp_sens/smoke/smoke.log

# 1) 학습 (4셀 직렬, 셀마다 8장 전부 / global batch 64)
setsid nohup bash $H/run_host_hp.sh >> /data2/seonghyeonnoh/paper/hp_sens/host.log 2>&1 < /dev/null &
tail -F /data2/seonghyeonnoh/paper/hp_sens/progress.log

# 2) residual (셀당 GPU 1장, 학습 끝난 셀부터)
NAME=e4_r16 GPU=0 bash $H/add_residual.sh

# 3) sparse-15 평가 (학습 큐 종료를 기다린 뒤 셀당 8장 직렬)
setsid nohup bash $H/eval_hp.sh >> /data2/seonghyeonnoh/paper/hp_sens/eval.log 2>&1 < /dev/null &
```

출력 루트: `/data2/seonghyeonnoh/paper/hp_sens/<cell>/` (`model/<round>/`,
`model/<round>_prephase2/`, `gen/round_<t>/`, `logs/`, `cell.json`,
`residual/`).

## 검증 (옵션이 실제로 먹혔는지)

`run_hp_cell.sh` 가 라운드 7 뒤에 두 검증기를 자동 실행한다.

- `verify_hp_run.py <run>` — `cell.json` 대비 **매 라운드** meta 검사:
  `experts_per_task` / `r` / `attention_rank` / `alpha` / `top_k`,
  `num_experts == (round+1) × E`, `dataset_order`, `stop_after_task`,
  그리고 움직여서는 안 되는 스위치(phase/kd/replay).
- `scripts/ablation/verify_ablation_run.py` — 텐서 수준 검사(phase1 이전 라우터 행
  drift = 0, phase2 expert delta = 0, KD-init 역할 존재, selfgen 500건 라인).

`dataset_order` / `stop_after_task` 는 이번에 `save_v3_meta` 에 추가한 필드다
(2026-09-18). 체크포인트는 라운드 인덱스로만 이름이 붙어서, 이 필드 없이는
"정말 역순으로 학습했는가"를 사후에 증명할 방법이 없었다. **그 이전에 만들어진
체크포인트는 이 필드가 비어 있어 순서 검증이 FAIL 로 표시된다**(메시지에 그 사실을
같이 출력한다).

## 전문가 디스패치 (`DISPATCH=loop|dense`)

`loop`(기본, 기존 코드)은 **배치 전체에서 선택된 expert 의 합집합**을 돈다. 토큰 하나가
지불하는 연산은 정확히 top-k 개 전문가뿐이고(선택 안 된 전문가의 토큰은 gather 에
아예 들어가지 않는다), 늘어나는 것은 **커널 개수**다. k=8 / 64 전문가면 배치 안에서
64개가 모두 한 번씩은 선택되므로 projection 하나당 64회 × (gather + 2 GEMM + scatter)
→ 7 projection × 32 layer = 한 forward 에 1.4만 개 수준의 작은 커널이 뜬다. FLOPs 가
아니라 런치 오버헤드다.

`dense`는 모든 전문가의 LoRA 쌍을 쌓아 **rank 공간에서 두 번의 큰 GEMM**으로 계산하고,
선택되지 않은 전문가의 rank 블록을 라우팅 가중치 0으로 지운다. 커널 수가 전문가 수와
무관해진다. 대신 선택되지 않은 전문가의 연산도 지불하는데, rank 공간 폭이
`num_experts × rank` = 8태스크 × (E × rank = 64) = **라운드 7 에서 어느 셀이든 512** 라서
4096 폭 projection 대비 약 +25%이고 **E=1 이든 E=8 이든 같은 비용**이다. 즉 `e8_r8` 을
`e1_r64` 와 같은 속도로 돌릴 수 있다.

동치성: `scripts/hp_sensitivity/check_dispatch_equivalence.py` 가 E=1/2/4/8 × (residual
유무 × 패딩 토큰 유무) 16개 조합에서 attention/FFN 출력을 비교한다. dropout 0 에서
전부 통과하고 k=1 은 비트 동일(0.000e+00). **단 dropout>0 이면 차이가 하나 있다**:
`loop` 은 (expert, token) 마다 독립 dropout 마스크를 뽑고 `dense` 는 토큰당 하나를 뽑아
그 토큰의 k개 전문가가 공유한다. k=1 이면 같은 draw 이고, k>1 이면 "독립 k개" 대 
"공유 1개" 차이다. 축을 섞지 않으려면 **한 연구 안에서는 모든 셀이 같은 모드**여야 한다.

체크포인트 meta 에 `expert_dispatch` 가 기록되고 `verify_hp_run.py` 가 `cell.json` 과
대조한다. 기본값은 `loop` 이라 기존/진행 중인 학습·평가 동작은 바뀌지 않는다.

```bash
# 동치성 (GPU 불필요, ~10초)
.venv-runtime/bin/python scripts/hp_sensitivity/check_dispatch_equivalence.py
# 스모크가 e8_r8 을 loop/dense 양쪽으로 돌려 벽시계를 비교한다
CELLS="e8_r8 e8_r8@dense" PAIRS="0,1 2,3" bash scripts/hp_sensitivity/smoke_hp.sh
```

## KD-init 예산 (`KD_FRACTION`, 기본 0.5)

`--v3_kd_init_step_fraction` (2026-09-18 신규). KD 스트림과 primary 스트림은
samples-per-pass(5000)와 pass 수(태스크 epoch)가 같으므로, KD-init 의 full 업데이트 수는
**그 태스크 phase1 업데이트 수와 정확히 같다**. 따라서 이 값은 말 그대로
"이 태스크 학습 스텝의 몇 %" 이다. 0.5 면 `ceil(0.5 × full_updates)` 만 돌고 나머지 KD
스트림은 방문하지 않는다. accum window 단위로 올림해 반쯤 찬 window 에서 멈추지 않는다.

로그에 두 줄이 남는다(검증 지점):

```
<task> [v3 KD-init] step fraction 0.5: 198/395 optimizer updates (198/395 microsteps); ...
<task> [v3 KD-init] truncated: 198 updates, N local sample exposures
```

meta 의 `v2.kd_init_step_fraction` 에 기록되고 `verify_hp_run.py` 가 `cell.json` 과 대조한다.

**이전 실행값**: 2026-09-18 ablation arm(`1phase_kd_rep`, `2phase_kd_rep`, `2phase_kd_gen`)과
논문 모델의 조상 `selfgen_cl_frozen_20260901` 은 모두 fraction 1.0(=100%) 으로 돌았다.
라운드별 KD 업데이트 수가 phase1 과 같았던 것이 그 증거다(237/237, 553/553, 395/395 ...).
20%(=360/1800)는 Megatron wiki→code→conv 쪽 관례이고 TRACE 계보에는 적용된 적이 없다.
따라서 **HP 스윕(50%)과 기존 ablation 셀(100%)은 KD 예산이 다르다** — 표에 함께 쓰려면
ablation arm 을 20% 로 재학습하거나, 표/캡션에 두 값이 다르다고 적어야 한다.

실측 기반 절감(오늘 `2phase_kd_gen` 로그, 8 GPU / gb 64):

| 태스크 | KD 업데이트 100% → 50% | KD 시간 100% → 50%(추정) |
|---|---|---|
| FOMC | 237 → 119 | 4m28s → 2.2m |
| MeetingBank | 553 → 277 | 12m38s → 6.3m |
| Py150 | 395 → 198 | 11m39s → 5.8m |
| ScienceQA | 237 → 119 | 10m21s → 5.2m |
| NumGLUE-cm | 395 → 198 | 17m40s → 8.9m |
| NumGLUE-ds | 395 → 198 | 18m20s → 9.2m |
| 20Minuten | 553 → 277 | 27m00s → 13.5m |
| **합계** | | **1h42m → 51m** |

## 함정

1. **`anchors.json` 은 정순 인덱스(`"0"`~`"7"`)로 키가 잡혀 있다.** 순서를 바꾸면
   인덱스 조회가 다른 태스크의 anchor 를 집어온다. `run_hp_cell.sh` 는 `"task"`
   필드로 이름 조회를 한다(`anchor_for`).
2. **역순에서는 20Minuten 이 round 0 이라 replay 를 생성해야 한다.** ablation 러너에는
   20Minuten 의 생성 프로필(CAP/BS/CHUNK)이 아예 없었다(정순에서는 마지막이라 생성할
   일이 없음). 여기서는 CAP 1024(selfgen `caps.json` 기준) / BS 128 / CHUNK 240 을 넣었고,
   **CUE 는 없다** — 20Minuten 프롬프트는 문단 뒤에 바로 정답이 이어지고 `Answer:` 류
   마커가 없다.
3. **`GEN_FINAL=1`(기본)** 이면 라운드 7 뒤에 마지막 태스크의 replay 를 한 번 더 생성해
   `gen/round_8` 을 만든다. residual 튜닝이 8개 태스크 전부의 self-generated replay 를
   쓰기 위한 것이다. 이게 없으면 마지막 태스크만 남의 런 기록을 빌려야 한다
   (기존 `residual_router_20260915` 가 `oursgen_20Minuten` 을 빌려 쓴 이유).
4. **평가는 순서를 알아야 한다.** `run_ours_sparse15_optimized.py --task-order`
   (또는 `SPARSE15_TASK_ORDER`)로 넘긴다. 안 넘기면 역순 셀의 diagonal 이 전부
   엉뚱한 태스크에 매칭된다. `eval_hp.sh` 가 `cell.json` 에서 읽어 자동으로 넣는다.
5. **E>1 / top-k>1 은 이번이 첫 학습이다.** 지금까지의 V3 TRACE 런은 전부 E=1/top-1
   이었다. 라우터·디스패치는 제네릭하게 구현돼 있어 코드 변경은 필요 없지만
   (`routes_for` 가 (slot, token) 쌍으로 dispatch), 전문가 루프가
   `active_expert_ids()` 길이만큼 돌기 때문에 **`e8_r8` 라운드 7 에서는 최대 64회**
   커널 런치가 난다. 스모크가 라운드당 벽시계를 `cell.json.smoke_seconds_2rounds`
   에 적어주니 큐 걸기 전에 예산을 확인할 것.
6. **`loop` 의 64회 루프는 버그가 아니다.** 토큰별로는 top-k 만 계산한다(위 디스패치 절).
   다만 `dense` 로 바꾸면 dropout draw 방식이 달라지므로 셀들 사이에서 모드를 섞지 말 것.
7. `pkill -f` 로 죽이지 말 것(자기 명령까지 매칭). PID 로 kill 한다.

## ETA (참고)

ablation `2phase_kd_gen`(2phase, 8라운드 + 생성)이 약 7시간이었다. 1phase 는 phase2 라우터 리튠(라운드당 395 업데이트)이 빠져 더 짧다. 4셀 직렬이면
**약 28시간**, `e8_r8` 의 전문가 루프 오버헤드를 감안하면 30~36시간. residual 은
셀당 15~30분(GPU 1장), sparse-15 평가는 셀당 45분~1.5시간(8장) 이므로 평가 4셀에
3~6시간.
