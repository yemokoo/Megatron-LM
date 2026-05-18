# Experiments Index

이 문서는 현재 continual learning 실험의 빠른 색인 + 연구 메모입니다.

핵심 목적:
- `wiki -> code` continual learning에서 **code 적응 성능**을 최대화한다.
- 특히 `FFN expert + attention-side LoRA expert`를 같이 쓰는 하이브리드 구조가, 왜 기대보다 성능이 덜 나오는지 확인한다.
- 단순한 파라미터 수 문제가 아니라면, **routing**, **expert specialization**, **shared backbone과의 co-adaptation** 중 무엇이 병목인지 좁혀간다.

## 1. Current Research Question

현재 가장 중요한 문제는 아래입니다.

- `shared-router hybrid` 계열에서 `attention LoRA expert`를 붙였는데, 기대보다 code 성능 향상이 크지 않다.
- 오히려 `single dense / standalone LoRA` 쪽이 더 잘 나오는 경우가 있다.
- 따라서 지금은 "LoRA 파라미터 수가 적어서 망한다"보다는 아래 가설들을 비교 중이다.

현재 보는 가설:
- `shared router` 때문에 FFN expert와 attention LoRA expert가 서로 routing을 방해한다.
- sparse expert 구조 자체가 `single dense adapter`보다 code adaptation에 불리하다.
- expert만 추가해서는 부족하고, `shared backbone`이 같이 적응해야 성능이 나온다.
- 4개 expert에서 7개 expert로 확장할 때, **같은 wiki token**이 어떤 expert pair로 재배치되는지가 성능과 연결된다.

현재 원하는 인사이트:
- attention LoRA expert가 실제로 도움이 되는가, 아니면 noise를 추가하는가
- shared router가 진짜 병목인가
- code expert 추가 이후에도 wiki token이 old-old pair를 유지하는지, old-new / new-new로 이동하는지
- 좋은 모델은 어떤 routing transition 패턴을 갖는지

## 2. Common Entry Points

- Guard runner: [run_guarded_training.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_guarded_training.sh)
- Base pretrain runner: [run_base_moe_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_base_moe_a100_bf16.sh)
- Continual runner: [run_continual_moe_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_continual_moe_a100_bf16.sh)
- Replay to W&B: [replay_full_history_to_wandb.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/replay_full_history_to_wandb.py)

## 3. Main Experiment Families

### 3.1 FFN MoE Baselines

- A2 wiki pretrain: [wiki_a_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_a_a2_mha_a100_bf16.sh)
- A2 wiki to code, unfreeze: [a_to_b_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_a2_mha_a100_bf16.sh)
- A2 wiki to code, freeze: [a_to_b_freeze_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_freeze_a2_mha_a100_bf16.sh)
- Base FFN config: [flame-moe.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-moe.sh)

역할:
- 가장 기본이 되는 `FFN MoE continual learning` 기준선
- 이후 `E`, `F` 계열은 모두 이 기준선과 비교한다

### 3.2 Shared-Router Hybrid LoRA Expert Family

- Wiki pretrain entry: [wiki_e2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_e2_mha_a100_bf16.sh)
- Wiki pretrain stage script: [pretrain_wiki_shared_router_hybrid_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh)
- Wiki to code entry: [code_from_wiki_e2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_e2_mha_a100_bf16.sh)
- Wiki pretrain entry, full-rank QV expert variant: [wiki_e5_fullrank_qv_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_e5_fullrank_qv_mha_a100_bf16.sh)
- Wiki to code entry, full-rank QV expert variant: [code_from_wiki_e5_fullrank_qv_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_e5_fullrank_qv_mha_a100_bf16.sh)
- Wiki pretrain entry, full-rank QKVO expert variant: [wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh)
- Wiki to code entry, full-rank QKVO expert variant: [code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh)
- Offline chain, full-rank QV expert variant: [offline_chain_e5_fullrank_qv_mha.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/offline_chain_e5_fullrank_qv_mha.sh)
- Wiki to code stage script: [continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh)
- Model config: [flame-shared-router-hybrid-experts.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-shared-router-hybrid-experts.sh)
- Core implementation: [shared_router_hybrid.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/shared_router_hybrid.py)
- Expand/freeze utilities: [continual_learning_utils.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py)

구조 요약:
- `shared router + FFN expert + attention LoRA expert`
- FFN과 attention LoRA가 **같은 router decision**을 공유한다
- expert expansion은 주로 `4 experts -> 7 experts`

대표 실험명:
- `E2`: shared-router hybrid, attention LoRA expert rank 16
- `E3`: shared-router hybrid, attention LoRA expert rank 256
- `E4`: shared-router hybrid, attention LoRA expert with projection included
- `E5 QV`: shared-router hybrid, full-rank-style attention LoRA expert on `Q/V` only
- `E6 QKVO expert`: shared-router hybrid, attention full-rank expert on `Q/K/V/O`

stage별 probe 기본값:
- `wiki stage`: primary `wiki_probe`, secondary `code_probe`
- `code stage`: primary `code_probe`, secondary `wiki_probe`

현재 비교 철학:
- `wiki stage`는 전체 파라미터를 정상적으로 학습한다
- `code stage`는 `shared_router_hybrid_train_new_experts_and_router_only` 경로를 써서, 기존 shared/base는 freeze하고 새로 확장된 expert와 router 쪽만 학습한다
- attention 쪽은 shared router가 선택한 expert index를 FFN expert와 같이 공유한다
- `E6`에서는 attention expert가 `Q/K/V/O` projection 각각에 대응하는 full-rank update를 가진다

### 3.3 Dense / Standalone LoRA Path

- Dense wiki pretrain: [wiki_a_dense_x_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_a_dense_x_a100_bf16.sh)
- Dense wiki to code stage: [continual_code_from_wiki_dense_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh)
- Dense config: [flame-dense.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-dense.sh)
- Standalone routed LoRA impl: [qv_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py)

역할:
- sparse expert 구조 없이, 보다 dense한 adapter가 code adaptation에 유리한지 비교하는 축

### 3.4 Full-Rank LoRA Family

- F1, full-rank LoRA + unfreeze: [code_from_wiki_f1_attn_full_rank_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_f1_attn_full_rank_lora_mha_a100_bf16.sh)
- F2, full-rank LoRA + freeze shared: [code_from_wiki_f2_attn_full_rank_lora_freeze_shared_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_f2_attn_full_rank_lora_freeze_shared_mha_a100_bf16.sh)
- F3, rank-16 LoRA + unfreeze: [code_from_wiki_f3_attn_rank16_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_f3_attn_rank16_lora_mha_a100_bf16.sh)
- F-QVO: [code_from_wiki_fqvo_attn_full_rank_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_fqvo_attn_full_rank_lora_mha_a100_bf16.sh)
- F-QKV: [code_from_wiki_fqkv_attn_full_rank_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_fqkv_attn_full_rank_lora_mha_a100_bf16.sh)
- F-QV: [code_from_wiki_fqv_attn_full_rank_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_fqv_attn_full_rank_lora_mha_a100_bf16.sh)
- Full-rank LoRA config: [flame-attn-full-rank-lora.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-attn-full-rank-lora.sh)
- Core implementation: [full_rank_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/full_rank_lora_attention.py)

구조 요약:
- attention base weight는 freeze
- LoRA branch만 학습
- `F1/F3/F-Q*`는 shared backbone이 같이 적응하는 `unfreeze` 계열
- `F2`는 shared backbone도 freeze

projection variant 의미:
- `F-QVO`: `Q/V/O`에만 full-rank LoRA, `K`는 완전 freeze
- `F-QKV`: `Q/K/V`
- `F-QV`: `Q/V`

### 3.5 G2-Matched FFN Baselines

목적:
- 기존 A/F 계열과 G2 shared-router 계열의 비교축이 섞이지 않도록, FFN expert granularity를 G2와 맞춘다.
- 고정 변수는 `topk=4`, `wiki experts=8`, `code experts=16`, `moe_ffn_hidden_size=352`, `global_batch_size=2304`이다.
- 기본 실행은 wiki source를 `micro_batch_size=128`, code continual baselines를 `micro_batch_size=96`으로 둔다.
- 의도적으로 바꾸는 변수는 code continual stage의 attention adaptation뿐이다.

비교군:
- `attn-freeze`: FFN-MoE만 사용한다. code stage에서 새 FFN experts와 router rows만 학습하고 attention/base trunk는 freeze한다.
- `attn-fullrank-lora`: 같은 FFN-MoE 설정에 single dense full-rank LoRA를 attention Q/K/V/O에 추가한다. base attention weight는 freeze하고 LoRA parameter만 학습한다.

Entry points:
- Wiki source, G2-matched FFN-MoE: [wiki_ffn_moe_g2matched_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_ffn_moe_g2matched_mha_a100_bf16.sh)
- Code continual, attention freeze: [code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh)
- Code continual, attention full-rank LoRA: [code_from_wiki_ffn_moe_g2matched_attn_full_rank_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_ffn_moe_g2matched_attn_full_rank_lora_mha_a100_bf16.sh)
- Code-only sequential launcher after wiki source is complete: [run_g2matched_ffn_attention_code_baselines_mha.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_g2matched_ffn_attention_code_baselines_mha.sh)
- Sequential launcher for all three stages: [run_g2matched_ffn_attention_baselines_mha.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_g2matched_ffn_attention_baselines_mha.sh)

해석 가능 범위:
- 두 code runs의 차이는 attention adaptation을 아예 주지 않았을 때와 single full-rank LoRA를 줬을 때의 차이로 해석한다.
- 이 비교는 shared-router hybrid G2와 완전히 같은 구조 비교는 아니다. G2는 FFN과 attention expert가 router decision을 공유하지만, 이 baseline은 FFN-MoE router와 dense attention LoRA를 분리해서 본다.
- `ATTN_FULL_RANK_LORA_RANK` 기본값은 기존 F-family의 full-rank 의미를 따라 `1024`이다. capacity-matched ablation이 필요하면 실행 시 `ATTN_FULL_RANK_LORA_RANK=256`으로 바꾼다.

## 4. Analysis Scripts

### 4.1 Replay and W&B Fix-up

- Replay all history to W&B: [replay_full_history_to_wandb.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/replay_full_history_to_wandb.py)

용도:
- online run에서 `1800 baseline`을 미리 올리지 않은 경우, 사후 replay로 graph를 깔끔하게 이어붙인다
- `nobase` continual run을 기준선과 stitched replay로 복원한다

### 4.2 Shared Router vs Split Router Clone

- Eval: [eval_shared_router_split_probe.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/eval_shared_router_split_probe.py)
- Plot: [plot_shared_router_split_probe.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/plot_shared_router_split_probe.py)

왜 했는가:
- hybrid 모델이 잘 안 되는 이유가 `shared router` 때문인지 확인하기 위해
- 같은 checkpoint를 그대로 두고, inference에서만
  - `shared router`
  - `split router clone`
  를 비교한다

원하는 인사이트:
- FFN과 LoRA가 같은 router를 공유하는 것이 진짜 병목인지

현재 결과:
- `E2 QV` code checkpoint 기준
- `shared_router`: acc `0.66477`, ppl `6.234`
- `split_router_clone`: acc `0.65066`, ppl `6.883`

현재 해석:
- split이 더 나빴기 때문에, **shared router 자체가 주범일 가능성은 낮아졌다**
- 관심은 다시 `expert parameterization`과 `shared backbone co-adaptation` 쪽으로 이동했다

### 4.3 Same-Token Routing Transition: 4C2 -> 7C2

- Pair dump: [dump_shared_router_pairs.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/dump_shared_router_pairs.py)
- Transition compare: [compare_shared_router_pair_transitions.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/compare_shared_router_pair_transitions.py)
- Runner: [run_shared_router_pair_transition.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/run_shared_router_pair_transition.sh)

왜 했는가:
- `4 experts -> 7 experts` 확장 후, **같은 wiki token**이 어떤 expert pair로 이동하는지 보고 싶기 때문
- 단순 평균 usage가 아니라, 같은 token 기준으로
  - old-old 유지
  - old-new 이동
  - new-new 이동
  을 보고 싶다

원하는 인사이트:
- code expert를 추가했을 때 wiki token routing이 얼마나 안정적인지
- 특정 old pair가 특정 new expert로 일관되게 흘러가는지
- 좋은 continual model이 expert transition 측면에서 어떤 패턴을 가지는지

주의:
- shared-router-hybrid에서는 FFN과 attention LoRA가 router를 공유하므로, 이 분석은 **shared router top-2 pair**만 추적하면 된다

## 5. Current Known Runs and Storage

아래 경로는 KT 서버 기준 주요 저장 위치 메모입니다.

### 5.1 Baseline

- A2 wiki pretrain: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/wiki-a-moe-bf16/a2-wiki-ffn-moe-mha-a100-bf16-mb96-1800`

### 5.2 Hybrid Family

- E2 wiki: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/e2-wiki-ffn-attn-lora-single-router-moe-mha-a100-bf16-mb96-1800`
- E2 code: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/e2-wiki-to-code-ffn-attn-lora-single-router-moe-mha-a100-bf16-mb96-1800`
- E4 code: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/e4-wiki-to-code-ffn-attn-lora-qvo-r16-single-router-moe-mha-a100-bf16-mb96-1800`
- E5 wiki: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/e5-wiki-ffn-attn-lora-qv-r1024-single-router-moe-mha-a100-bf16-mb64-1800`
- E5 code: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/e5-wiki-to-code-ffn-attn-lora-qv-r1024-single-router-moe-mha-a100-bf16-mb64-1800`

### 5.3 Full-Rank / Standalone LoRA Family

- F3 code: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/f-attn-r16-lora-bf16/f3-wiki-to-code-ffn-moe-unfreeze-attn-r16-lora-mha-a100-bf16-mb64-1800-nobase`
- F-QVO code: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/f-attn-qvo-full-rank-lora-bf16/fqvo-wiki-to-code-ffn-moe-unfreeze-attn-full-rank-lora-mha-a100-bf16-mb64-1800-nobase`

### 5.4 Important Logs

- `E5` chain log: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/offline_chain_e5_fullrank_qv_mha.log`
- `F3` continual log: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/f3_code_rank16_lora_mb64_nobase.log`
- `F-QVO` continual log: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/fqvo_code_full_rank_lora_mb64_nobase.log`

## 6. Current Findings

현재까지 중요한 관찰은 아래입니다.

- `F3` 같은 `single / standalone LoRA` 계열이 hybrid LoRA expert보다 더 잘 나오는 경우가 있다
- 따라서 "LoRA rank가 작아서 hybrid가 약하다"는 해석만으로는 부족하다
- `shared vs split router clone` 결과는 split이 더 나빴다
- 그래서 지금은 `shared router`보다는
  - sparse expert structure 자체
  - expert specialization 부족
  - backbone co-adaptation 부족
  쪽이 더 유력하다

실무적으로 보고 싶은 다음 포인트:
- code expert 추가가 실제로 routing transition을 만들었는가
- 새 expert가 선택되기만 하고 도움이 안 되는가
- wiki token에 대해 old-old 조합이 유지되는가, 아니면 new expert가 끼어드는가

## 7. Quick Mapping by Experiment Name

- `A / A2`: FFN MoE baseline 계열
- `E2`: shared-router hybrid, attention LoRA expert rank 16
- `E3`: shared-router hybrid, attention LoRA expert rank 256
- `E4`: shared-router hybrid, QVO-side attention LoRA expert with projection
- `E5 QV`: shared-router hybrid, full-rank-style attention LoRA expert on Q/V only
- `E6 QKVO expert`: shared-router hybrid, full-rank attention expert on Q/K/V/O
- `F1`: full-rank LoRA + unfreeze
- `F2`: full-rank LoRA + freeze shared
- `F3`: rank-16 standalone LoRA + unfreeze
- `F-QVO`: full-rank LoRA on Q/V/O only
- `F-QKV`: full-rank LoRA on Q/K/V only
- `F-QV`: full-rank LoRA on Q/V only

## 8. Operational Rules

- online continual run은 항상 `LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0`
- 즉 `1800` baseline은 online에서 선업로드하지 않는다
- baseline은 나중에 [replay_full_history_to_wandb.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/analysis/replay_full_history_to_wandb.py)로 stitched replay 한다
- hybrid / F-family 실험에서는 실제 entry script와 stage script를 먼저 확인하고 수정한다
- 코드 변경 후에는 항상 `push/pull`까지 같이 진행한다

## 9. Experimental Design Rules

이번 라운드에서 확인한 가장 큰 문제:
- 결과가 안 좋은 것보다, **비교축이 섞여서 해석이 불가능한 실험**이 더 치명적이다
- `single vs hybrid`, `freeze vs unfreeze`, `QV vs QKVO`, `low-rank vs full-rank`가 동시에 바뀌면 결과를 보고도 원인을 특정할 수 없다
- 따라서 앞으로는 "커맨드 작성"보다 먼저 **실험 질문과 대조군 정의**를 고정해야 한다

앞으로 모든 실험은 아래 질문을 먼저 문서 기준으로 점검한다:
- 이 실험으로 정확히 무엇을 증명하거나 반박하려는가
- 대조군은 무엇인가
- 이번 실험에서 **의도적으로 바꾸는 변수는 정확히 하나인가**
- freeze / unfreeze, projection 위치, rank, router 수, top-k, KL 유무, wiki/code stage 학습 범위는 모두 고정되어 있는가
- 결과가 좋아지거나 나빠졌을 때, 어디까지 해석할 수 있는가

실험 카드 최소 템플릿:
- 목적:
- 대조군:
- 바꾸는 변수:
- 고정 변수:
- 기대 해석:

실험 제안이 들어오면 앞으로 아래 순서로 진행한다:
1. 먼저 이 문서를 기준으로 `목적 / 대조군 / 바꾸는 변수 / 고정 변수 / 해석 가능 범위`를 짧게 정리한다
2. 사용자가 그 비교축에 동의하면 그다음 코드 수정과 실행 커맨드를 만든다
3. 실행 전에는 run name, trainable parameter 범위, probe 구성, baseline stitching 여부를 다시 확인한다
4. 결과 해석도 반드시 처음 정의한 실험 질문 기준으로만 한다

특히 주의할 confound:
- `single`과 `hybrid`를 비교하면서 `freeze/unfreeze`까지 같이 바뀌는 경우
- `QV`, `QVO`, `QKV`, `QKVO`처럼 projection 위치가 동시에 달라지는 경우
- `rank 16`과 `rank 1024`처럼 adapter capacity가 동시에 달라지는 경우
- `QKV packed`와 `Q/K/V separate`처럼 attention adapter의 내부 parameterization 자체가 같이 바뀌는 경우
- `wiki stage full-train`과 `code stage freeze-train` 범위가 비교군마다 다른 경우
- `KL on/off`가 비교군마다 다른 경우

ablation 원칙:
- projection ablation을 할 때는 가능하면 `targets=qk`, `targets=qv`처럼 단순히 target 문자열을 바꿔서 parameterization까지 함께 바꾸지 않는다
- 특히 `QKV packed` 경로를 쓰는 실험에서는, `Q/K`만 보고 싶더라도 packed 구조는 유지하고 `V` 출력만 비활성화하는 식으로 **동일한 parameterization을 유지한 채** 비교하는 것을 기본 원칙으로 한다
- 즉 앞으로 projection ablation은 "어느 출력을 사용하느냐"를 바꾸는 실험으로 설계하고, "adapter 내부 구조 자체가 separate로 바뀌는 실험"은 별도 실험으로 분리해서 해석한다

run naming 규칙:
- 이름만 보고도 비교축이 드러나야 한다
- 최소한 아래 정보는 run name에 반영한다
- `single` 또는 `hybrid`
- `freeze` 또는 `unfreeze`
- `QV/QVO/QKV/QKVO`
- `lowrank/fullrank`
- `mbXX`
- `nobase` 여부

실험 관련 질문에 대한 작업 원칙:
- 앞으로 실험 설계, 비교, 해석 질문이 들어오면 **항상 이 문서를 먼저 참고해서 답변**한다
- 사용자가 빠르게 커맨드를 원하더라도, 비교축이 섞이면 먼저 그 위험을 짚고 정리한 뒤 진행한다
- 즉 실험 진행은 단순 실행이 아니라, **대조군과 변인 통제를 먼저 합의하는 과정**을 포함해야 한다
