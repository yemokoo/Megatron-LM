# Offline W&B Upload Workflow

오프라인 학습 후 W&B에 안전하게 올리는 운영 메모다. 이 문서는 특히 다음 상황을 기준으로 정리했다.

- 학습은 `WANDB_MODE=offline`으로 수행
- 체크포인트는 `.local/weights/...` 아래에 저장
- 종료 시 NCCL teardown 때문에 hang가 날 수 있음
- 나중에 W&B로 일괄 업로드해야 함
- continual run은 source run의 `1800` 지점과 자연스럽게 이어지게 보고 싶음

## 기본 원칙

- 학습 중에는 가능하면 `DIRECT_LOCAL_SAVE=1`을 사용한다.
- 최종 성공 판정은 로그보다 `latest_checkpointed_iteration.txt`를 우선한다.
- `/tmp/flame-moe`는 작업 디렉토리일 뿐 최종 보관소가 아니다.
- W&B 업로드는 `offline-run` 디렉토리 통째로 sync하는 것보다 `run-*.wandb` 파일 단위가 더 안전하다.
- 이미 삭제된 run id는 재사용되지 않는다. 이 경우 새 backup run id를 써야 한다.
- continual 그래프를 `1800`에서 매끈하게 잇고 싶으면 단순 `wandb sync`가 아니라 replay 스크립트를 사용해야 한다.

## 학습 시 권장 설정

체인 또는 개별 런에서 아래 값을 기본으로 둔다.

```bash
WANDB_MODE=offline
DIRECT_LOCAL_SAVE=1
```

이렇게 하면:

- W&B 기록은 각 run 디렉토리 아래 `wandb/wandb/offline-run-*`에 저장된다.
- 최종 체크포인트는 `.local/weights/...` 아래에 직접 쌓인다.
- `/tmp/flame-moe`는 source staging, 임시 작업 디렉토리, 로그 보조 공간으로만 쓴다.

## 종료 hang가 날 때

학습이 망한 것과 종료가 hang된 것은 구분해야 한다. 아래가 보이면 학습은 끝난 것이다.

```bash
cat /path/to/run/latest_checkpointed_iteration.txt
```

값이 목표 step이면 체크포인트는 이미 저장된 상태다. 이 경우:

1. `.local` 쪽 최종 저장 확인
2. 필요하면 `/tmp/flame-moe/<RUN_ID>/`에서 수동 `rsync`
3. 그 뒤 프로세스 정리

예시:

```bash
find /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights -name latest_checkpointed_iteration.txt | sort
```

```bash
pkill -f "<RUN_ID>"
pkill -f "torch.distributed.run --standalone"
```

## `/tmp/flame-moe` 정리 규칙

모든 학습 프로세스가 끝났고 `.local` 저장이 확인되면 `/tmp/flame-moe`를 비워도 된다.

확인:

```bash
pgrep -af "pretrain_gpt.py|torch.distributed.run|offline_chain"
```

```bash
find /tmp/flame-moe -mindepth 1 | head
du -sh /tmp/flame-moe
df -h /tmp
```

정리:

```bash
rm -rf /tmp/flame-moe/*
```

주의:

- 학습 중에는 지우지 않는다.
- `DIRECT_LOCAL_SAVE=1`이어도 `/tmp/flame-moe`는 비어 있지 않을 수 있다.
- `/tmp` 공간 부족으로 런이 시작조차 못 하는 경우가 있다.

## 오프라인 run 위치 확인

offline W&B run은 보통 아래에 있다.

```bash
find /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights -path '*/wandb/wandb/offline-run-*' | sort
```

`.wandb` 파일만 보고 싶으면:

```bash
find /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights -name '*.wandb' | sort
```

## 단순 업로드: `.wandb` 파일 단위 sync

가장 단순한 경우는 `.wandb` 파일 하나를 직접 sync하는 것이다.

```bash
wandb sync /abs/path/to/run-xxxx.wandb
```

이 방식을 권장하는 이유:

- `offline-run-*` 디렉토리 전체를 sync할 때 metric이 일부만 올라가거나 watcher가 summary만 계속 만지는 경우가 있었다.
- `.wandb` 파일 단위 sync가 더 예측 가능했다.

## 삭제된 run id 충돌

기존 run이 W&B에서 한 번 생성됐다가 삭제되면, 같은 run id로는 다시 업로드되지 않는다.

에러 예시:

```text
run <id> was previously created and deleted; try a new run id
```

해결:

```bash
wandb sync \
  --id <new_backup_id> \
  --project <project> \
  --entity <entity> \
  --skip-console \
  /abs/path/to/run-xxxx.wandb
```

권장 규칙:

- 원래 run id는 보존하지 말고 backup suffix를 붙인다.
- 예: `a2_unfreeze_mha_backup_20260404_1`

## 단순 sync의 한계

`wandb sync`는 원본 offline run을 복원하는 데는 좋지만, 아래는 못 한다.

- source run의 `1800` baseline point를 continual run 앞에 새로 삽입
- 학습 기록과 probe를 합쳐 step `1800 -> 1900 -> 2000 ...` 형태로 재구성

즉:

- train log만 필요하면 `.wandb` sync도 충분할 수 있다.
- `1800` 연결점이 중요하면 replay가 필요하다.

## 권장 업로드 전략

### 1. pretrain run

pretrain은 standalone replay 또는 raw sync 중 하나를 쓴다.

- 이미 online으로 올라간 run이면 재업로드하지 않아도 된다.
- offline pretrain이면 `replay_full_history_to_wandb.py --run-dir ...`가 가장 안전하다.

### 2. continual run

continual은 아래 방식이 가장 좋다.

- source run의 step `1800` scalar를 baseline으로 먼저 기록
- continual run의 full scalar history를 뒤에 replay
- textual log에 남은 probe 값으로 같은 step의 probe metric을 보강

이렇게 하면:

- train log
- probe
- `1800` 연결점

을 한 번에 맞출 수 있다.

## Full Replay 스크립트

스크립트:

- [replay_full_history_to_wandb.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/replay_full_history_to_wandb.py)

지원 모드:

- `--run-dir`
  - standalone run의 전체 scalar history를 그대로 replay
- `--source-run-dir --continual-run-dir --baseline-step`
  - source baseline + continual full history를 stitch

동작 요약:

- TensorBoard event file에서 scalar history를 읽는다.
- source run에서 `baseline-step` 이하의 마지막 값을 가져와 baseline payload를 만든다.
- continual run의 scalar history를 replay한다.
- 필요하면 `--continual-log`에서 probe metric을 읽어 덮어쓴다.

## 예시 명령

### standalone pretrain replay

```bash
python analysis/replay_full_history_to_wandb.py \
  --run-dir /abs/path/to/pretrain_run \
  --project flame-continual-top2-qv-lora \
  --entity <entity> \
  --run-name "E2 wiki replay-full" \
  --run-id e2_wiki_replay_full_20260404_1 \
  --save-dir /tmp/wandb-replay-e2-wiki-full
```

### continual replay with baseline stitching

```bash
python analysis/replay_full_history_to_wandb.py \
  --source-run-dir /abs/path/to/source_run \
  --continual-run-dir /abs/path/to/continual_run \
  --continual-log /abs/path/to/continual.log \
  --project flame-continual-top2-qv-lora \
  --entity <entity> \
  --run-name "A2 unfreeze replay-full" \
  --run-id a2_unfreeze_replay_full_20260404_1 \
  --save-dir /tmp/wandb-replay-a2-unfreeze-full \
  --baseline-step 1800
```

## E2 code처럼 체인 로그에서 구간만 뽑아야 할 때

개별 run log가 따로 없으면 체인 로그에서 필요한 구간만 잘라 쓴다.

```bash
awk '/\[START\] 5_E2_code/{flag=1; next} /\[END\] 5_E2_code/{flag=0} flag' \
  /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/offline_chain_2345f.log \
  > /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/e2_code_only_replay.log
```

## 업로드 전 체크리스트

- `.local`의 `latest_checkpointed_iteration.txt`가 목표 step인지 확인
- `/tmp/flame-moe` 정리가 필요한지 확인
- replay 대상 run 디렉토리에 `events.out.tfevents*`가 있는지 확인
- probe가 필요한 경우 `logs/*.log` 또는 체인 로그 구간이 있는지 확인
- 삭제된 run id인지 확인하고 필요하면 새 backup id를 사용

## 업로드 후 확인

로컬 W&B CLI 로그:

```bash
tail -n 100 /home/work/Agent_HJ/wandb/debug-cli.work.log
```

확인할 것:

- run 생성 성공
- step `1800` baseline 존재 여부
- step `1900+` train metric 존재 여부
- step `1900+` probe 존재 여부

## 추천 운영 요약

앞으로는 아래 순서를 기본 운영으로 둔다.

1. 학습은 `WANDB_MODE=offline`, `DIRECT_LOCAL_SAVE=1`
2. 최종 체크포인트는 `.local`에서 확인
3. hang면 저장 확인 후 프로세스만 정리
4. `/tmp/flame-moe`는 다음 런 전에 정리
5. 단순 복원은 `.wandb` 파일 단위 sync
6. 논문용 그래프나 continual 연결이 중요하면 `replay_full_history_to_wandb.py` 사용

이 패턴이면 오프라인 저장, 재업로드, continual stitching을 한 흐름으로 반복할 수 있다.
