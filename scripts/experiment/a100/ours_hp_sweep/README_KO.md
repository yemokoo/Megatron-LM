# Ours HP 스윕 (wiki→code→conv)

**목적**: baseline들은 HP에 민감해서 wiki 최적값이 TRACE로 전이되지 않는다는
주장을 세우려면, ours가 같은 축에서 평평하다는 대응 증거가 있어야 한다.
이 스윕이 그 대응 증거다.

**실행하지 않았다.** 구현만 되어 있다.

## 설계

두 축을 각각 3점씩, 중심을 공유하는 십자에 최소 코너 한 점을 더한다.
중심 (replay 0.1%, KD 20%)은 **이미 확정된 런**이라 큐에 없다.

| | KD 180 (10%) | KD 360 (20%) | KD 720 (40%) |
|---|---|---|---|
| replay **0.01%** | 신규 | 신규 | 신규 |
| replay **0.1%** | 신규 | ✅ 기존 | 신규 |
| replay **1%** | — | 신규 | — |

= **신규 6셀**.

replay 0.01% 라인이 3점 전부라, step 곡선을 두 개(0.1%, 0.01%) 얻는다.
최소 코너 1점만 두면 "무너졌다/안 무너졌다"만 알고 그 이유는 모른다.
두 축이 독립적으로 포화하면 "예산을 양쪽에서 함께 줄여도 된다"는 강한 주장이 되고,
따로 줄일 땐 괜찮은데 같이 줄이면 무너지면 상호작용이 있는 것이다.

### replay 축은 총 노출 고정

`0.01%×2000 / 0.1%×200 / 1%×20` — 전부 총 ~829,4xx 시퀀스(1 epoch의 20%)이고
**고유 데이터 양만** 바뀐다. seed1234 트리의 기존 서브셋(`0p1pctx200`, `1pctx20`,
`10pctx2`)이 이미 이 규약을 따른다(`SUBSETS_INDEX.json`의 note: "총 시퀀스는 모두
829,4xx(=1 epoch의 20%). 고유 데이터 수만 다름").

노출 횟수를 고정하고 비율만 바꾸는 대안은 "메모리 크기"와 "replay 연산량"을 동시에
움직여서, 평평하게 나와도 무엇에 robust한지 말할 수 없다. 저장 비용이 걸린 것은
고유 데이터이므로 이쪽을 축으로 잡는다. MoE-LPR이 **같은 서브셋**을 쓰므로 같은
축에 나란히 놓을 수 있다는 이점도 있다.

주의: replay 0.1%×200 = 1 epoch의 20%이고 KD 360/1800도 20%다. 숫자가 겹치니
표에서 라벨을 분리할 것.

### 설정: ffn+attn shared router, 1× 지점

`moe_ffn_hidden_size 352` / `attn_full_rank_lora_rank 256` / shared_router_hybrid /
expert 8→16→24 / top-k 4. 본 방법 설정이자 DoF 1× 지점이며, 중심 셀이 그 런 자체다.

wiki 소스는 `hf_g2_wiki_code_conversation/.../sources/ffn_attn_shared_router/wiki`
(중심 런과 동일). DoF의 width별 wiki 소스가 아니다.

## 파일

| 파일 | 역할 |
|---|---|
| `prepare_0p01pct_subset.sh` | 없는 서브셋 `0p01pctx2000` 하나만 생성 |
| `job_ours_hp.sh` | 셀 1개 = code kd → code 1phase → conv kd → conv 1phase |
| `sched_ours_hp.sh` | 빈 GPU 쌍을 잡아 6셀을 순차 실행 |
| `collect_ours_hp.py` | 셀별 AA/FM/LA 집계 + 항등식 검증 |

`job_ours_hp.sh`는 중심 런을 만든 `ours_hyb_sub_chain.sh`의 포크다. 그 스크립트는
replay 축(`SUB`)을 이미 파라미터로 받고 있었고, 여기서 추가한 것은 **KD step 축**뿐이다
(원본은 360이 5곳에 하드코딩: 두 kd 스테이지의 TRAIN_ITERS/SAVE_INTERVAL/완료검사,
두 1phase 스테이지의 DISTILL_SOURCE_REQUIRED_ITERS). 나머지 — wiki 소스, 스테이지
스크립트, micro batch, replay 예산, seed — 는 중심 런과 동일하다. 셀은 중심과
**스윕한 두 값에서만** 다르다.

기존 스크립트는 하나도 수정하지 않았다.

## 실행 순서

```bash
O=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/scripts/experiment/a100/ours_hp_sweep

# 0) 계획 확인 (GPU 안 씀)
DRY_RUN=1 bash $O/sched_ours_hp.sh

# 1) 없는 서브셋 1개 생성 (0.01% x 2000)
bash $O/prepare_0p01pct_subset.sh

# 2) 큐 실행 — 빈 카드 쌍이 생길 때까지 대기하며 6셀을 순차 처리
ALLOWED="0 1 2 3" bash $O/sched_ours_hp.sh

# 셀 하나만 직접
SUB_LABEL=0p01pctx2000 KD_ITERS=180 GPUS=0,1 NPROC=2 bash $O/job_ours_hp.sh

# 3) 집계
python $O/collect_ours_hp.py
```

로그: `<SWEEP_ROOT>/logs/<cell>.out`, 체인 진행은 `<SWEEP_ROOT>/<cell>/logs/chain.log`.
`tail -F /data2/seonghyeonnoh/LLM-continual-learning-runs/ours_hp_sweep_20260912/logs/sched.log`

## 안전장치

- 스케줄러는 `ALLOWED`에 적힌 카드만 보고, **연속 `IDLE_STREAK`(기본 3)회 폴링에서
  compute 프로세스가 없는** 쌍만 잡는다. 남의 작업을 밀어내지 않는다.
- `job_ours_hp.sh`는 시작 전에 `2304 % (mb × world)`와 `replay % (mb × world)`가
  0인지 검사하고 아니면 즉시 중단한다. 어긋나면 Megatron이 조용히 global batch를
  바꾼다.
- 각 스테이지는 `latest_checkpointed_iteration.txt`로 완료를 판정하므로 재실행이
  안전하다(완료분 skip).
- `prepare_0p01pct_subset.sh`는 이미 있으면 skip하고, `FORCE_REBUILD`를 켜지 않는다.
  기존 서브셋을 건드리지 않는다.

## 집계기 주의

a11(wiki 소스 정확도)은 어느 스테이지도 만들지 않는다. ours wiki 소스는 HF 이관
체크포인트 디렉터리라 probe 로그가 없다. 값은 별도 probe 패스에서 온다:

- **hybrid (ffn+attn) = 0.467627** ← 이 스윕이 쓰는 값
- ffn-only = 0.460047 ← 이 스윕에는 틀린 값

`wiki_source_probe_20260908/hybrid.probe`. 집계기는 `AA − (LA − ⅔FM)` 잔차를
찍는데, 중심 셀에서 **1.1e-16**(기계 정밀도)이 나오므로 a11이 맞다는 확인이 된다.

### 미확인 항목

이 a11로 계산한 중심 셀은 **AA 0.5097 / FM +0.0040 / LA 0.5124**인데, 현재 DoF 표의
ours 1× 행은 **AA .5097 / FM .0026 / LA .5114**다. AA는 a11을 쓰지 않으므로 정확히
일치하고, LA/FM만 어긋난다 → DoF 표가 다른 a11을 썼을 가능성. 그림의 헤드라인
숫자(AA)는 영향 없지만, **DoF ours 행의 LA/FM은 0.467627 기준으로 재확인**이 필요하다.
