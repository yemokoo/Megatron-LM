# Ours LoRA-MoE V3

Llama V1/V2/V2.5/V3의 공통 backbone은
`meta-llama/Llama-3.1-8B` base다.
Base tokenizer에는 chat template이 없으므로 V2와 같은 일반
system/user/assistant Llama-3.1-Instruct template을 런타임에 주입한다. V3의
token cache도 모두 `cache/tokenized/llama31_8b_base/`를 사용한다.

V3는 V2의 continual-learning 학습 메커니즘을 유지하면서, 각 decoder
layer의 router를 attention 앞으로 이동하고 Q/K/V/O 및 FFN LoRA expert가
동일한 routing 결정을 공유하도록 확장한 버전이다.

## Layer 구조

```text
layer input
    │
input_layernorm
    │
shared router ────────────────┐
    │                         │
Q/K/V LoRA experts           │ same indices and weights
    │                         │
self-attention               │
    │                         │
O LoRA experts               │
    │                         │
attention residual           │
    │                         │
post_attention_layernorm     │
    │                         │
gate/up/down LoRA experts ◀──┘
    │
FFN residual
```

Router 입력은 각 layer의 attention 직전 normalized state다.

```math
h_{route}=\operatorname{InputLayerNorm}(h_{layer})
```

FFN expert는 attention 이후의 `post_attention_layernorm` 출력을 변환하지만,
expert index와 weight는 `h_route`에서 미리 계산한 값을 그대로 재사용한다.
Router aux loss와 z-loss도 layer당 한 번만 계산한다.

## Expert 계약

- target: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`
- 모든 projection은 같은 LoRA rank와 alpha를 사용한다.
- 모든 projection은 같은 expert 수를 가진다.
- task마다 모든 projection에 `experts_per_task`개를 동시에 추가한다.
- 새 expert의 B는 0으로 초기화하므로 확장 직후 모델 출력은 변하지 않는다.
- Llama 3.1 및 Qwen2.5 GQA의 서로 다른 Q/K/V 출력 크기는 projection별 B
  shape로 처리한다.

## V2에서 유지되는 학습 동작

- 과거 task의 고정 replay subset
- task별 고정된 전역 replay exposure budget
- 새 expert를 위한 pre-expansion teacher KD 초기화
- new data의 신규 expert+router gradient
- past data의 router-only joint replay gradient
- KD 중 이전 router row의 bit-exact 고정
- old expert prefix로 제한한 teacher forward
- task별 expert growth 후 optimizer/DDP 재구성

Teacher prefix에서는 router logits을 이전 expert 수까지 먼저 자른 뒤
softmax/top-k를 수행한다. 제한된 routing context가 old QKVO 및 old FFN
expert에 동시에 적용되므로 확장 전 모델을 정확히 복원한다.

## Freeze 경계

새 task의 primary/KD 학습:

- old QKVO/FFN experts: frozen
- new QKVO/FFN experts: trainable
- shared router: trainable
- KD 중 old router rows: optimizer step 전후 복원
- pretrained backbone: frozen

Past-data joint replay:

- 모든 QKVO/FFN experts: frozen
- shared router만 trainable
- 모든 optimizer update에서 new-data gradient와 함께 누적
- 저장 replay pool은 정확히 1,000개 유지
- joint phase에서는 같은 pool을 반복해 new:replay sample exposure를 5:1로 유지
- 5,000-sample task의 3/5/7 epochs에 joint replay는 3,000/5,000/7,000회 노출
- 8-GPU rank-sparse replay 후 combined gradient를 전 rank 평균하므로 replay가
  빠지는 global optimizer update는 없음
- active-rank 수로 replay gradient를 정규화해 sparse batch에서도 replay loss
  coefficient 1을 유지

## 실행

```bash
cd /home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace

# command와 preflight만 렌더링
./scripts/baselines/llama31/ours_lora_moe_v3.sh validate
./scripts/baselines/qwen25_7b/ours_lora_moe_v3.sh validate

# 학습
./scripts/baselines/llama31/ours_lora_moe_v3.sh train
./scripts/baselines/qwen25_7b/ours_lora_moe_v3.sh train
```

직접 실행할 경우 기존 entry point에 `--training_version v3`를 사용한다.
V3 launcher의 기본 rank/alpha/top-k 및 V2 KD/replay 설정은 기존 Ours
launcher와 동일하고, joint sample ratio 기본값은 5:1이다. QKVO expert가
추가되므로 기본 micro-batch는 8,
gradient accumulation은 2로 두어 전역 effective batch 64를 유지한다.

## 체크포인트

V3 checkpoint는 frozen backbone을 제외하고 다음만 저장한다.

- layer shared router
- Q/K/V/O LoRA experts
- gate/up/down LoRA experts
- `lora_moe_meta.json`

Metadata의 `architecture`는 `shared_router_qkvo_ffn`이며, 기존 FFN-only
checkpoint loader와 분리된다. 평가는 `evaluate_Ours_LoRA_MoE.py`가 metadata를
읽고 자동으로 V2 또는 V3 loader를 선택한다.

## 구조 검증

```bash
./.venv-runtime/bin/python \
  implementations/llmcl_benchmark/scripts/test_ours_lora_moe_v3.py
```

검증 범위:

- zero-init 확장 전후 exact output
- layer당 router 한 번 호출 및 QKVO/FFN context identity
- attention/FFN rank와 expert count 동기화
- old-prefix teacher bit-exact 복원
- old/new expert gradient 경계
- KD old-router-row bit-exact 복원
- V2 every-update exact-budget joint replay의 router-only 경계
- Llama/Qwen2 GQA forward 및 KV-cache generation
- partial checkpoint save/load round trip
