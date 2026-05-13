# Codex Shared-Router QKVO Session Harness

이 문서는 새 Codex 세션이나 새 터미널을 열었을 때, 현재 진행 중인 shared-router QKVO continual-learning 실험을 바로 이어받기 위한 하네스입니다. KT 서버 환경, GitHub 인증, W&B 업로드, 핵심 코드 위치, 실험 흐름을 한 번에 복원하는 것을 목표로 합니다.

## 1. 기본 위치와 환경

로컬 Mac 작업 경로:

```bash
cd "/Users/yemokoo/miil/1. LLM-CL/LLM-continual-learning"
```

KT 서버 작업 경로:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
```

KT 런타임 활성화:

```bash
source scripts/miscellaneous/activate_kt_env.sh
```

기본 확인:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -c 'import apex; print("apex ok")'
python -c 'import transformer_engine.pytorch; print("TE pytorch ok")'
python -c 'import flash_attn, inspect; print(flash_attn.__version__, inspect.getfile(flash_attn))'
python -c 'import grouped_gemm; print("grouped_gemm ok")'
```

주의:

- KT 기준 Python은 보통 `/usr/bin/python`입니다.
- `bitsandbytes` CUDA/CPU 라이브러리 경고는 지금 실험에서는 대체로 non-fatal입니다.
- `wandb login`은 새 `wandb_v1_...` 토큰을 40자 old API key로 오해해서 실패할 수 있습니다. 로그인보다 `WANDB_API_KEY` 환경변수 방식이 안전합니다.
- `grouped_gemm`이 없으면 MoE grouped GEMM 최적화가 꺼질 수 있으니 반드시 확인합니다.

## 2. GitHub 토큰 인증

KT 서버는 HTTPS remote + `credential.helper store` 방식으로 인증합니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

git remote set-url origin https://github.com/yemokoo/LLM-continual-learning.git
git config --global credential.helper store

echo -n "GitHub token: "
read -s GITHUB_TOKEN
echo

printf "https://yemokoo:%s@github.com\n" "$GITHUB_TOKEN" > ~/.git-credentials
chmod 600 ~/.git-credentials
unset GITHUB_TOKEN

GIT_TERMINAL_PROMPT=0 git ls-remote origin | tail -3
```

서버에서 최신 `slurm` 브랜치와 `Megatron-LM` 서브모듈을 맞출 때:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

git fetch origin slurm
git checkout slurm
git pull --ff-only origin slurm

git submodule sync Megatron-LM
git submodule update --init Megatron-LM
```

서브모듈 주의:

- `Megatron-LM`은 parent repo 내부 파일처럼 보여도 서브모듈입니다.
- `Megatron-LM/...` 내부를 수정했다면 먼저 서브모듈 안에서 commit/push하고, parent repo에서는 submodule pointer를 commit해야 합니다.
- attention LoRA grouped GEMM 최적화가 들어간 기준은 parent `slurm`의 `e07d00f` 근처, submodule `844e0a7e` 근처입니다.

## 3. W&B 토큰과 업로드 방식

토큰은 로그인하지 말고 환경 파일로 고정합니다.

```bash
mkdir -p ~/.config/wandb
echo -n "W&B token: "
read -s WANDB_API_KEY
echo

cat > ~/.config/wandb/env <<EOF
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_ENTITY="yemoyemo010831-korea-university"
export WANDB_PROJECT="flame-continual-top2-qv-lora"
EOF

chmod 600 ~/.config/wandb/env
unset WANDB_API_KEY
```

새 세션에서:

```bash
source ~/.config/wandb/env
```

현재 shared-router QKVO 실험은 일반 `wandb sync`가 실패할 수 있었습니다.

- `AssertionError`가 뜬 경우가 있었습니다.
- 삭제된 run id를 재사용하면 `409 previously created and deleted`가 뜹니다.
- 따라서 최종 비교 그래프는 offline run을 그대로 sync하기보다, `run.log`의 probe 라인을 파싱해서 새 W&B run에 다시 로깅하는 relog 방식을 씁니다.

상세 업로드 절차와 relog 스크립트는 다음 문서에 저장되어 있습니다.

```bash
docs/offline_wandb_upload_workflow_ko.md
```

핵심 규칙:

- wiki run과 code run은 반드시 따로 올립니다.
- code run은 step 0부터 올리지 않습니다.
- code run의 첫 점은 wiki 최종 1800 probe 값으로 step 1800에 찍습니다.
- 이후 code local iteration 100, 200, ..., 1800은 step 1900, 2000, ..., 3600으로 이어 붙입니다.
- code stage의 local_iteration 0 expanded-baseline 점은 그래프 연결용으로 쓰지 않습니다.

권장 W&B run 이름:

```text
G1 - 실험1 - wiki - expert top2
G1 - 실험1 - wiki to code - expert top2
G2 - 실험2 - wiki - expert top4
G2 - 실험2 - wiki to code - expert top4
G3 - 실험3 - wiki - expert top8
G3 - 실험3 - wiki to code - expert top8
G4 - 실험4 - wiki - expert top16
G4 - 실험4 - wiki to code - expert top16
```

## 4. 핵심 실험 구조

모델은 shared-router hybrid MoE GPT입니다.

- 9 layers, hidden size 1024, sequence length 512, bf16.
- 2 x A100 80GB 기준으로 실행했습니다.
- MoE layer는 layer 1-8, 총 8개입니다.
- 각 MoE layer 안에 FFN MoE expert와 attention QKVO LoRA expert가 같이 있습니다.
- 같은 layer 안에서 router를 한 번 계산하고, 그 routing 결과를 FFN expert와 attention Q/K/V/O LoRA expert가 공유합니다.
- 단, 전 layer가 하나의 global router를 공유하는 구조는 아닙니다. layer별 shared router입니다.

attention LoRA expert는 개념적으로 다음 형태입니다.

```text
output[token] += router_prob[token, expert] * (x[token] @ A_expert @ B_expert)
```

code stage에서는 wiki checkpoint를 불러온 뒤 expert 수를 확장합니다.

- 기존 wiki expert/router row는 freeze됩니다.
- 새로 추가된 FFN expert, attention QKVO LoRA expert, expanded router row만 학습됩니다.
- backbone, embedding, dense trunk는 학습하지 않는 설정입니다.

## 5. 실험 세팅

wiki stage:

| 실험 | topk | experts | FFN hidden | attn rank | micro batch | checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| G1 | 2 | 4 | 704 | 512 | 72 | 완료 |
| G2 | 4 | 8 | 352 | 256 | 72 | 완료 |
| G3 | 8 | 16 | 176 | 128 | 48 | 완료 |
| G4 | 16 | 32 | 88 | 64 | 32 | 완료 |

code stage:

| 실험 | 확장 | topk | micro batch | 특이사항 | checkpoint |
| --- | --- | ---: | ---: | --- | --- |
| G1 | 4 -> 8 | 2 | 96 | 기본 | 완료 |
| G2 | 8 -> 16 | 4 | 72 | 기본 | 완료 |
| G3 | 16 -> 32 | 8 | 48 | 기본 | 완료 |
| G4 | 32 -> 64 | 16 | 32 | attention LoRA grouped GEMM 적용 | 완료 |

공통:

```text
train_iters=1800
global_batch_size=2304
seq_length=512
save_interval=1800
probe_eval_interval=100
```

G4 code baseline은 매우 느렸고, attention LoRA grouped GEMM 적용 후 크게 빨라졌습니다.

- 기존 G4 code는 대략 118-127초/iter 수준이었습니다.
- 최적화 후 G4 code mb32는 대략 44-48초/iter 수준이었습니다.

## 6. 완료된 code checkpoint와 최종 probe

G1:

```text
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g1-top2-e4to8-ffn704-r512-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb96-1800
code_probe final: acc=0.690588, ppl=5.182763
wiki_probe final: acc=0.307310, ppl=69.12850
```

G2:

```text
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-1800
code_probe final: acc=0.687688, ppl=5.276930
wiki_probe final: acc=0.316804, ppl=61.87322
```

G3:

```text
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g3-top8-e16to32-ffn176-r128-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb48-1800
code_probe final: acc=0.680358, ppl=5.562718
wiki_probe final: acc=0.320972, ppl=59.84755
```

G4:

```text
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g4-top16-e32to64-ffn88-r64-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb32-attn-groupedgemm-1800
code_probe final: acc=0.670963, ppl=5.930213
wiki_probe final: acc=0.320012, ppl=61.74783
```

G1은 checkpoint 저장 후 `ChildFailedError`가 로그에 남아 있지만, `[after training is done]`와 `latest_checkpointed_iteration.txt=1800`이 있으므로 완료로 취급했습니다.

## 7. 완료 여부 확인 커맨드

KT 서버에서 G1-G4 code checkpoint와 최종 probe 확인:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

for d in \
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g1-top2-e4to8-ffn704-r512-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb96-1800 \
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-1800 \
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g3-top8-e16to32-ffn176-r128-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb48-1800 \
.local/weights/a100/mha/shared-router-granularity-qkvo/code/g4-top16-e32to64-ffn88-r64-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb32-attn-groupedgemm-1800
do
  echo
  echo "=== $(basename "$d") ==="

  echo -n "checkpoint: "
  if [ -f "$d/latest_checkpointed_iteration.txt" ]; then
    tr -d '\n\r[:space:]' < "$d/latest_checkpointed_iteration.txt"
    echo
  else
    echo "MISSING"
  fi

  echo "last probes:"
  grep -E "probe (code_probe|wiki_probe) at iteration .*local_iteration: 1800" "$d/logs/run.log" | tail -2

  echo "done marker:"
  grep -E "\[after training is done\]|ended without|OutOfMemory|ChildFailedError" "$d/logs/run.log" | tail -5
done
```

현재 살아있는 학습 확인:

```bash
ps -ef | grep -E 'offline_chain_shared_router_granularity_qkvo|torchrun|pretrain_gpt.py|g[1-4]-top' | grep -v grep
```

최신 run.log 찾기:

```bash
RUNLOG=$(find .local/weights/a100/mha/shared-router-granularity-qkvo \
  -path '*/logs/run.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "$RUNLOG"
tail -n 120 "$RUNLOG"
```

## 8. 핵심 코드 위치

모델/설정:

```text
configs/model/flame-shared-router-hybrid-experts.sh
Megatron-LM/megatron/core/transformer/shared_router_hybrid.py
Megatron-LM/megatron/core/transformer/transformer_config.py
Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py
Megatron-LM/megatron/training/arguments.py
Megatron-LM/megatron/training/training.py
Megatron-LM/pretrain_gpt.py
Megatron-LM/megatron/core/models/gpt/shared_router_hybrid_layer_specs.py
```

실험 실행:

```text
scripts/experiment/a100/offline_chain_shared_router_granularity_qkvo_wiki_only_mha.sh
scripts/experiment/a100/offline_chain_shared_router_granularity_qkvo_code_only_mha.sh
scripts/experiment/a100/offline_chain_shared_router_granularity_qkvo_code_router_memory_earlystop_mha.sh
scripts/experiment/a100/offline_chain_shared_router_granularity_qkvo_mha.sh
scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh
scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh
scripts/experiment/a100/run_guarded_training.sh
```

W&B 업로드:

```text
docs/offline_wandb_upload_workflow_ko.md
```

분석:

```text
analysis/compare_shared_router_pair_transitions.py
analysis/eval_shared_router_split_probe.py
analysis/run_shared_router_pair_transition.sh
analysis/dump_shared_router_pairs.py
analysis/plot_shared_router_split_probe.py
analysis/eval_specialization_suite.py
```

## 9. Attention LoRA Grouped GEMM 최적화

목적:

- 기존 attention QKVO LoRA expert 경로는 expert/topk가 커질수록 작은 GEMM과 token routing loop가 많이 생겨서 병목이 커졌습니다.
- `--attn-lora-grouped-gemm`은 expert별 작은 LoRA projection을 묶어서 실행하는 경로를 추가해, 특히 G4 code처럼 `topk=16`, `attn_lora_num_experts=64`인 경우 속도를 크게 줄였습니다.

활성화 확인:

```bash
grep -n "attn_lora_grouped_gemm\|moe_grouped_gemm\|micro_batch_size\|num_experts\|moe_router_topk" "$RUNLOG" | head -80
```

기대 설정:

```text
attn_lora_grouped_gemm=True
moe_grouped_gemm=True
micro_batch_size=32
num_experts=64
moe_router_topk=16
```

## 10. Router-Memory KL Replay 실험

목적:

- code continual learning 중 old wiki input의 routing distribution이 새 expert 쪽으로 drift되는지 확인합니다.
- old task LM loss를 replay하지 않고, wiki memory batch에서 layer별 shared router output distribution만 distill합니다.
- teacher로 full old model을 계속 띄우지 않고, wiki checkpoint에서 layer별 old router weight만 복사해서 저장합니다.

구현 요약:

- 새 code batch는 기존과 동일하게 학습합니다. 즉 새 expert와 새 router row만 학습합니다.
- `ROUTER_MEMORY_KL_COEFF > 0`이면 every N code step마다 wiki memory batch로 router KL step을 한 번 수행합니다.
- memory step에서는 현재 모델 forward를 `torch.no_grad()`로 수행하면서 각 MoE layer의 shared-router input hidden을 capture합니다.
- capture되는 hidden은 input layernorm 이후, shared router에 들어가기 직전의 hidden입니다.
- KL 계산 전 hidden을 detach하므로, router KL gradient는 backbone, attention, FFN expert, 이전 layer로 흐르지 않습니다.
- memory step에서는 기존 continual-learning hook을 잠시 풀어 old/new router row 전체가 KL loss로 업데이트되게 합니다.

새 옵션:

```text
--router-memory-kl-coeff
--router-memory-fraction
--router-memory-interval
--router-memory-data-path
--router-memory-eval-data-path
--router-memory-eval-interval
--router-memory-eval-iters
--router-kl-stop-step
--router-kl-early-stop-enabled
--router-kl-early-stop-metric
--router-kl-patience
--router-kl-min-delta
--router-kl-warmup-steps
--router-kl-smoothing-window
```

스크립트 환경변수:

```bash
ROUTER_MEMORY_KL_COEFF=0.1
ROUTER_MEMORY_FRACTION=0.05
ROUTER_MEMORY_INTERVAL=20
ROUTER_MEMORY_DATASET="$PWD/data/wiki/router_memory_5pct"
ROUTER_MEMORY_EVAL_DATASET="$PWD/data/wiki/router_memory_5pct"
ROUTER_MEMORY_EVAL_INTERVAL=0
ROUTER_MEMORY_EVAL_ITERS=1
ROUTER_KL_EARLY_STOP_ENABLED=1
ROUTER_KL_EARLY_STOP_METRIC=fixed_probe_kl
ROUTER_KL_WARMUP_STEPS=300
ROUTER_KL_PATIENCE=3
ROUTER_KL_MIN_DELTA=0.01
ROUTER_KL_SMOOTHING_WINDOW=3
CODE_RUN_SUFFIX="-router-memory-kl0p1"
```

주의:

- router-memory set은 4개 실험이 같은 old-memory examples를 보도록 고정된 별도 indexed dataset을 사용합니다.
- 현재 기본 fixed memory 위치는 `data/wiki/router_memory_5pct`입니다.
- `ROUTER_MEMORY_INTERVAL=20`이면 code 20 step마다 wiki router-memory KL step 1번이고, 총 memory token 수가 code 학습 token 수의 약 5%가 됩니다.
- 기존 baseline checkpoint와 충돌하지 않게 `CODE_RUN_SUFFIX`를 반드시 붙입니다.
- 로그 metric prefix `router_memory/...`는 실제 optimizer update에 사용되는 train-memory KL입니다.
- 로그 metric prefix `router_memory_eval/...`는 매번 같은 fixed Wiki probe batch로 측정하는 diagnostic KL입니다.
- early-stop은 기본적으로 `router_memory_eval/kl`의 smoothed 값이 다시 상승하는지 보고 KL step을 끊습니다.
- KL이 stopped 상태가 되어도 code LM 학습 step 수는 줄지 않고 그대로 진행됩니다.
- stopped 이후에는 optimizer update용 wiki memory batch를 더 소비하지 않으며, fixed-probe eval은 동일 샘플을 반복 측정하는 진단용입니다.
- `SAVE_INTERVAL=300`을 지정하면 early-stop 변형은 300 step마다 checkpoint를 남깁니다.
- 주요 상태 로그는 `router_memory/kl_enabled`, `router_memory/kl_stopped`, `router_memory/kl_skipped_early_stop`, `router_memory/stop_reason_code`입니다.
- `router_memory_eval/new_expert_prob_mass`, `router_memory_eval/topk_overlap_with_old_router`, layer별 expert usage도 같이 기록됩니다.

fixed router-memory set 생성:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

python scripts/dataset/materialize_fixed_sample_stream.py \
  --input-dir data/wiki/train \
  --output-dir data/wiki/router_memory_5pct \
  --samples 207360 \
  --sequence-length 512 \
  --random-seed 1234 \
  --output-prefix train_text_document
```

`207360 = 1800 * 2304 * 0.05`이므로, G1-G4 code stage가 모두 같은 5% router-memory pool을 공유합니다.

50% router-memory replay 변형:

- G1 기준으로 `ROUTER_MEMORY_INTERVAL=2`를 사용하면 code 2 step마다 wiki router-memory KL step 1번이고, 총 memory token 수가 code 학습 token 수의 약 50%가 됩니다.
- 이 경우 fixed memory pool도 10배 커야 하므로 `data/wiki/router_memory_50pct`를 별도로 사용합니다.
- `ROUTER_MEMORY_EVAL_INTERVAL=100`으로 두면 같은 fixed mini-set에 대한 diagnostic KL을 100 code step마다 반복 측정합니다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

python scripts/dataset/materialize_fixed_sample_stream.py \
  --input-dir data/wiki/train \
  --output-dir data/wiki/router_memory_50pct \
  --samples 2073600 \
  --sequence-length 512 \
  --random-seed 1234 \
  --output-prefix train_text_document
```

`2073600 = 1800 * 2304 * 0.5`이므로, G1 50% replay run이 같은 샘플을 과도하게 재사용하지 않습니다.

## 11. Code-Train Wiki Expert Mask 실험

목적:

- Wiki checkpoint에서 Code continual learning을 할 때, 학습 중 Code token이 기존 Wiki expert로 라우팅되는 것을 막습니다.
- train step에서만 기존 expert `0..SOURCE_NUM_EXPERTS-1`를 shared-router top-k 후보에서 제외합니다.
- probe/eval/inference에서는 mask를 끄고 Wiki expert와 Code expert 전체를 다시 열어둡니다.
- 따라서 실험 질문은 “Code 학습 동안만 Code expert 사용을 강제하면, 최종 전체 expert routing에서 retention/plasticity가 어떻게 바뀌는가”입니다.

핵심 옵션:

```text
--shared-router-train-mask-existing-experts
--shared-router-train-mask-existing-experts-from-num-experts
```

스크립트 환경변수:

```bash
SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS=1
SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS_FROM_NUM_EXPERTS="$SOURCE_NUM_EXPERTS"
```

G1/G2 실행 wrapper:

```text
scripts/experiment/a100/offline_chain_shared_router_granularity_qkvo_code_train_mask_wiki_experts_g1_g2_mha.sh
```

주의:

- mask는 `TopKRouter` 내부에서 `self.training`일 때만 적용됩니다.
- `model.eval()`로 도는 code/wiki probe에서는 전체 expert가 선택 가능합니다.
- router replay/KL과는 별개 실험이며, 기본 wrapper는 router-memory KL을 켜지 않습니다.

## 12. 실험 해석 메모

expert 수가 커져도 항상 느려지는 것은 아니고, 실제 active compute는 주로 `topk`에 의해 커집니다. 하지만 이번 granularity sweep은 expert 수와 topk를 같이 키웠습니다.

```text
G1: topk 2
G2: topk 4
G3: topk 8
G4: topk 16
```

따라서 G4가 느린 주된 이유는 다음을 같이 설명하면 됩니다.

- topk 증가로 token당 선택 expert 수가 증가했습니다.
- FFN MoE와 attention QKVO LoRA가 같은 routing을 공유하지만, 둘 다 선택된 expert 개수만큼 계산 경로가 늘어납니다.
- expert hidden/rank를 줄여 총 파라미터 수를 비슷하게 맞췄더라도, 작은 expert 연산이 많이 쪼개지면 GPU 효율이 떨어집니다.
- dispatch, gather, unpermute, probability weighting 같은 MoE routing overhead가 topk/expert 수와 함께 커집니다.
- attention LoRA는 Q/K/V/O 네 projection에 적용되므로 topk 증가의 체감 비용이 FFN보다 크게 보일 수 있습니다.

교수님께 설명할 때는 “파라미터 수가 비슷해도, 학습 시간은 파라미터 수만이 아니라 active expert path 수, routing/dispatch overhead, small GEMM 효율에 크게 좌우된다”고 말하면 정확합니다.

## 13. 새 세션에서 가장 먼저 볼 것

1. 이 파일을 읽습니다.
2. KT에서는 `source scripts/miscellaneous/activate_kt_env.sh` 후 import check를 합니다.
3. `git status --short`로 로컬 변경과 서브모듈 상태를 봅니다.
4. `latest_checkpointed_iteration.txt`와 final probe로 G1-G4 완료 여부를 확인합니다.
5. W&B 업로드는 `docs/offline_wandb_upload_workflow_ko.md`의 relog 방식을 사용합니다.
