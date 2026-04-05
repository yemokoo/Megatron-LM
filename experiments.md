# Experiments Index

짧은 색인 문서입니다.  
실험별 **실행 진입점**, **공통 러너**, **핵심 모델 구현**만 빠르게 찾기 위한 용도입니다.

## 1. Common Entry Points

- Guard runner: [scripts/experiment/a100/run_guarded_training.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_guarded_training.sh)
- Base pretrain runner: [scripts/experiment/a100/run_base_moe_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_base_moe_a100_bf16.sh)
- Continual runner: [scripts/experiment/a100/run_continual_moe_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_continual_moe_a100_bf16.sh)
- Replay to W&B: [analysis/replay_full_history_to_wandb.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/replay_full_history_to_wandb.py)

## 2. FFN MoE Baselines

- A2 wiki pretrain: [scripts/experiment/a100/wiki_a_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_a_a2_mha_a100_bf16.sh)
- A2 wiki to code, unfreeze: [scripts/experiment/a100/a_to_b_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_a2_mha_a100_bf16.sh)
- A2 wiki to code, freeze: [scripts/experiment/a100/a_to_b_freeze_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_freeze_a2_mha_a100_bf16.sh)
- Base FFN config: [configs/model/flame-moe.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-moe.sh)

## 3. Shared-Router Hybrid LoRA Expert Family

- Wiki pretrain entry: [scripts/experiment/a100/wiki_e2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_e2_mha_a100_bf16.sh)
- Wiki pretrain stage script: [scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh)
- Wiki to code entry: [scripts/experiment/a100/code_from_wiki_e2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_e2_mha_a100_bf16.sh)
- Wiki to code stage script: [scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh)
- Model config: [configs/model/flame-shared-router-hybrid-experts.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-shared-router-hybrid-experts.sh)
- Core implementation: [Megatron-LM/megatron/core/transformer/shared_router_hybrid.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/shared_router_hybrid.py)
- Expert expand/freeze utilities: [Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py)

Notes:
- `E2`, `E3`, `E4`는 모두 같은 실행 경로를 공유하고, 주로 `RUN_ID`, `ATTN_LORA_RANK`, `ATTN_LORA_ALPHA`, `ATTN_LORA_INCLUDE_PROJ`, batch 설정만 바꿉니다.
- 이 경로는 현재 `shared router + FFN expert + attention LoRA expert` 구조입니다.

## 4. Dense / Standalone LoRA Path

- Dense wiki pretrain: [scripts/experiment/a100/wiki_a_dense_x_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_a_dense_x_a100_bf16.sh)
- Dense wiki to code stage: [scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh)
- Dense config: [configs/model/flame-dense.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-dense.sh)
- Standalone routed Q/V(/O) LoRA impl: [Megatron-LM/megatron/core/transformer/qv_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py)

## 5. Full-Rank LoRA Family

- F1, full-rank LoRA + unfreeze: [scripts/experiment/a100/code_from_wiki_f1_attn_full_rank_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_f1_attn_full_rank_lora_mha_a100_bf16.sh)
- F2, full-rank LoRA + freeze: [scripts/experiment/a100/code_from_wiki_f2_attn_full_rank_lora_freeze_shared_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_f2_attn_full_rank_lora_freeze_shared_mha_a100_bf16.sh)
- Full-rank LoRA config: [configs/model/flame-attn-full-rank-lora.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-attn-full-rank-lora.sh)
- Core implementation: [Megatron-LM/megatron/core/transformer/full_rank_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/full_rank_lora_attention.py)
- Shared freeze helper: [Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py)

Notes:
- `F1`: attention base weight는 freeze, full-rank LoRA는 train, shared backbone은 unfreeze.
- `F2`: attention base weight는 freeze, full-rank LoRA는 train, shared backbone도 freeze.

## 6. Quick Mapping by Experiment Name

- `A / A2`: FFN MoE baseline 계열
- `E2`: shared-router hybrid, attention LoRA expert rank 16
- `E3`: shared-router hybrid, attention LoRA expert rank 256
- `E4`: shared-router hybrid, attention LoRA expert with projection included (`ATTN_LORA_INCLUDE_PROJ=1`)
- `F1`: FFN expert + full-rank LoRA unfreeze
- `F2`: FFN expert + full-rank LoRA freeze

## 7. Operational Rules

- Online run은 항상 `LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0`
- `1800` baseline은 online에서 선업로드하지 않고, replay로 사후 보정
- 코드 변경 후에는 항상 `push/pull`까지 같이 진행
