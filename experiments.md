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
- checkpoint는 기본적으로 최종 step만 저장하고, wiki/code probe는 50 step마다 실행한다.
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

### 4.4 Same-Token Hidden Preservation: Wiki-only -> KD Init -> Code:Wiki 1:1

- Dump runner: `scripts/experiment/a100/run_g2_ffn_only_kd_1to1_hidden_space_probes_mha.sh`
- Joint-PCA plotter: `scripts/analysis/plot_hidden_space_kd_1to1.py`

목적:

- Wiki-only, Wiki output-logits KD 직후, Code:Wiki = 1:1 LM 학습 후 모델에 동일한 Wiki/Code token 순서를 입력한다.
- 각 Transformer layer에서 같은 token 위치의 hidden state가 KD 및 Code 학습 뒤에도 유지되는지 확인한다.
- Wiki hidden이 유지되는 token/subspace가 있으면, raw Wiki replay 없이 저장된 hidden prototype/anchor로 router logits 또는 old-expert group routing을 맞출 수 있는지 후속 실험의 근거로 사용한다.
- 저장된 hidden을 사용하는 방식은 raw-data replay는 아니지만 feature-space replay/regularization으로 해석한다.

판정 기준:

- probe별로 세 체크포인트 hidden을 함께 중심화하고 layer별 공동 PCA/KDE 분포를 비교한다. Wiki와 Code probe의 PCA는 서로 분리한다.
- PCA 분포 중첩만으로 보존을 결론 내리지 않고, 동일 token 기준 paired cosine, symmetric normalized L2, cosine threshold별 보존 비율을 함께 본다.
- 세 dump의 token id, sample index, token position이 완전히 같지 않으면 plot 생성을 중단한다.

2026-07-31 산출물:

- Root: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/analysis/g2-ffn-only-kd-1to1-hidden-space`
- Wiki/Code probe 각각 3개 checkpoint x 2048 동일 token, 9개 Transformer layer hidden dump 완료.
- Wiki probe token identity SHA-256: `8d2559043db6791da8d4e58bc95b22bce4dbd9d37905d5e1732ab6e2ce040599`
- Code probe token identity SHA-256: `f12ed0ddfdea184589c3b01febcd03ee31a4646ce4a3ba81bbba5bdb562a0224`

초기 결과:

- Wiki probe, Wiki-only vs Code:Wiki 1:1의 layer-average paired cosine은 `0.998732`, cosine >= 0.99 token 비율은 `0.993164`이다.
- Code probe, Wiki-only vs Code:Wiki 1:1의 layer-average paired cosine은 `0.622178`, cosine >= 0.99 token 비율은 `0.001465`이다.
- Wiki output-logits KD 직후는 Wiki/Code probe 모두 Wiki-only 대비 layer-average paired cosine이 `0.9997` 이상으로, KD 자체는 hidden을 거의 유지했다.
- 현재 결과는 Wiki 표현을 유지하면서 Code 표현이 깊은 layer에서 크게 재구성된 패턴을 보인다. 단, Code:Wiki 1:1 학습 자체가 Wiki replay를 사용했으므로 replay-free 보존의 증거는 아니며, raw replay를 hidden anchor로 대체하는 후속 실험의 근거로만 해석한다.

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

### 5.4 G2 FFN-only LPR / KD + Joint-Replay Runs (2026-07-30~31)

공통 Wiki source:

- Wiki-only, 8 experts, step 1800 완료: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800`

LPR chain, `1800 -> 360 -> 1800 -> 360`, 전 단계 완료:

- Code task, 8 -> 16 experts, step 1800: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/lpr_chain/code_task/g2-ffn-only-e8to16-code-lm-aux-z-mb96-1800-lpr-chain`
- Code router LPR, 추가 360 steps, 누적 tracker 2160: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/lpr_chain/code_router/g2-ffn-only-code-router-lpr-gamma0.1-equal-token-mb96-360`
- Conversation task, 16 -> 24 experts, step 1800: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/lpr_chain/conversation_task/g2-ffn-only-e16to24-conversation-lm-aux-z-mb64-1800-from-lpr`
- Conversation router LPR, 추가 360 steps, 누적 tracker 2160: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/lpr_chain/conversation_router/g2-ffn-only-conversation-router-lpr-gamma0.1-equal-token-mb64-360`

KD initialization + new-expert LR-ramp joint-replay chain, 전 단계 완료:

- Wiki output-logits KD로 Code expert 8 -> 16 초기화, step 1800, MB 48: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/code/expansion_distill_init/g2-ffn-only-e8to16-code-expert-init-logits-wiki-distill-mha-a100-bf16-mb48-1800`
- Code:Wiki = 1:1 joint LM, step 1800, MB 96, 새 expert LR ramp 1~900: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/code/joint_lm_replay_ramp/g2-ffn-only-code-wiki-joint-lm-allrouter-newexpert-ramp900-mb96-1800`
- Wiki:Code = 1:1 output-logits KD로 Conversation expert 16 -> 24 초기화, step 600, MB 32: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/conversation/expansion_distill_init_joint_code_ramp/g2-ffn-only-e16to24-conv-init-from-code-ramp900-logits-wikicode-kd-mb32-600`
- Wiki:Code:Conversation = 1:1:2 joint LM, step 1800, MB 96, 새 expert LR ramp 1~900: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/conversation/joint_lm_replay_ramp/g2-ffn-only-conv-wikicode-joint-lm-allrouter-112-newexpert-ramp900-mb96-1800`

다음 hidden-space 검증의 `KD + 1:1` 대상은 위 Code:Wiki = 1:1 joint LM 체크포인트이다. 동일 token 순서 비교를 위해 Wiki-only source와 KD 직후 체크포인트도 함께 보존한다.

### 5.5 Important Logs

- `E5` chain log: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/offline_chain_e5_fullrank_qv_mha.log`
- `F3` continual log: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/f3_code_rank16_lora_mb64_nobase.log`
- `F-QVO` continual log: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/fqvo_code_full_rank_lora_mb64_nobase.log`
- G2 LPR chain logs: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/g2-checkpoints/lpr_chain/*/*/logs`
- G2 KD + LR-ramp chain logs: `/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/logs/g2_kd_ramp900_code_kd_ramp900_conversation_chain`

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

## 10. 2026-07 G2 FFN-Only KD + Joint LM Continual Chain

### 10.1 핵심 아이디어

과거 `Code 1 step -> router 1 step`처럼 optimizer update를 분리한 1/1 실험은 학습이 불안정하게 망가진 사례가 있었다. 이번 방식은 task loss와 replay loss의 gradient를 **한 optimizer update 안에서 aggregate**한다.

Code stage:

```text
Code batch: new Code experts + all router rows에 gradient
Wiki batch: all router rows에만 gradient
두 backward gradient를 accumulate -> optimizer.step() 한 번

L_code_stage = L_code_LM + L_wiki_LM
```

- lambda는 둘 다 1이다.
- Wiki router loss도 KD가 아니라 일반 LM loss이다.
- 1800은 micro-step이 아니라 optimizer step 수이다.
- 기존 `expert update 1800 + router update 1800`을 1800개의 joint update로 합친 실험이다.

Conversation stage:

```text
Conversation full batch 1개
+ Wiki/Code가 반씩 들어간 replay full batch 1개
-> gradient aggregate -> optimizer.step() 한 번

L_conv_stage ~= L_conversation + 0.5 L_wiki + 0.5 L_code
Wiki : Code : Conversation sample 수 = 1 : 1 : 2
```

`1:1:2`는 loss coefficient를 직접 그렇게 설정한다는 뜻이 아니라 실제 sample 노출량이다.

공통 구조:
- Wiki source 8 experts -> Code 16 experts -> Conversation 24 experts
- FFN experts only, router top-k 4, `moe_ffn_hidden_size=352`
- GBS 2304, 기존 experts와 dense/attention trunk freeze
- 각 stage의 새 experts + all router rows 학습
- save interval 600 optimizer steps

Wiki 8E source:

```text
.local/weights/a100/mha/g2-checkpoints/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800
```

### 10.2 완료된 KD-init chain

Chain script:

```text
scripts/experiment/a100/run_g2_ffn_only_joint_lm_code_kd_conversation_chain_mha.sh
```

사전 존재 8 -> 16 output-only KD init:

```text
.local/weights/a100/mha/g2-checkpoints/code/expansion_distill_init/g2-ffn-only-e8to16-code-expert-init-logits-wiki-distill-mha-a100-bf16-mb48-1800
```

- Wiki 8E를 16E로 확장한 뒤 Wiki output logits KD로 experts 9~16의 초기점을 만든 checkpoint이다.
- KD는 continual task 학습 phase가 아니라 random init보다 나은 expert 초기점을 만들기 위한 phase이다.

#### Run 1: Code + Wiki joint LM — 완료

```text
.local/weights/a100/mha/g2-checkpoints/code/joint_lm_replay/g2-ffn-only-code-wiki-joint-lm-allrouter-mb64-1800
```

- source: 위 8 -> 16 KD-init checkpoint
- MB64, GBS2304, 1800 optimizer steps
- trainable: experts 9~16 + all router rows
- probe: Code, Wiki
- final Code acc/PPL: `0.662594 / 6.331106`
- final Wiki acc/PPL: `0.460090 / 18.22470`

Log:

```text
.local/weights/a100/mha/g2-checkpoints/code/joint_lm_replay/g2-ffn-only-code-wiki-joint-lm-allrouter-mb64-1800/logs/a_to_b_freeze.log
```

#### Run 2: 16 -> 24 Wiki/Code output-only KD — 완료

```text
.local/weights/a100/mha/g2-checkpoints/conversation/expansion_distill_init_joint_code/g2-ffn-only-e16to24-conv-init-from-joint-code-logits-wikicode-distill-mb32-600
```

- source: Run 1의 16E checkpoint
- MB32, 600 optimizer steps
- experts 17~24의 Conversation 학습 전 초기점

KD final probe (`local_iteration=600`; diagnostic display baseline은 5400):

```text
Wiki acc/PPL:         0.462014 / 18.00381
Code acc/PPL:         0.658937 / 6.432723
Conversation acc/PPL: 0.288416 / 65.47378
```

Log:

```text
.local/weights/a100/mha/g2-checkpoints/conversation/expansion_distill_init_joint_code/g2-ffn-only-e16to24-conv-init-from-joint-code-logits-wikicode-distill-mb32-600/logs/g2_ffn_only_conversation_expert_logits_init_freeze.log
```

#### Run 3: Conversation + Wiki/Code joint LM — 완료

최종 유효 run:

```text
.local/weights/a100/mha/g2-checkpoints/conversation/joint_lm_replay/g2-ffn-only-conv-wikicode-joint-lm-allrouter-112-mb128-1800
```

- source: Run 2 KD-600 checkpoint
- MB128, GBS2304, grad accumulation 9, 1800 optimizer steps
- trainable: experts 17~24 + all router rows
- Wiki:Code:Conversation sample 비율 1:1:2
- probe: Conversation, Code, Wiki

Final:

```text
Conversation acc/PPL: 0.382690 / 28.13186
Code acc/PPL:         0.664720 / 6.258869
Wiki acc/PPL:         0.458670 / 18.48050
skipped/nan:          0 / 0
final grad norm:      약 0.079
```

Run 3 reload-time -> final:

```text
Conversation: 0.287973 -> 0.382690
Code:         0.664733 -> 0.664720
Wiki:         0.458800 -> 0.458670
```

- 안정 구간 약 14.3 sec/step, 약 36 TFLOP/s/GPU
- 최대 관측 VRAM 약 74.1GB/80GB; MB128보다 더 올리지 않는다.

Log:

```text
.local/weights/a100/mha/g2-checkpoints/conversation/joint_lm_replay/g2-ffn-only-conv-wikicode-joint-lm-allrouter-112-mb128-1800/logs/code_to_conversation_freeze.log
```

중단된 MB64 Run 3:

```text
g2-ffn-only-conv-wikicode-joint-lm-allrouter-112-mb64-1800
```

- 약 200 step에서 Wiki tertiary probe 누락을 발견해 중단했다.
- 600-step save 전이므로 유효 중간 checkpoint가 없고 최종 비교에 사용하지 않는다.

### 10.3 결과 해석과 평가 단차

- 분리 1/1 optimizer update와 달리 joint aggregation은 NaN/폭주 없이 완료됐다.
- Conversation을 학습하면서 Code/Wiki가 거의 유지됐다.
- 초기 급변 후 flat한 곡선 자체는 문제라기보다 빠른 초기 적응/수렴일 수 있다.

중요한 evaluation discontinuity:

```text
KD final Wiki:     0.462014
Run 3 reload Wiki: 0.458800
Run 3 final Wiki:  0.458670
```

- KD final과 Run 3 reload 사이에는 optimizer update가 없다.
- 따라서 `0.462014 -> 0.458800`을 forgetting으로 해석하면 안 된다.
- MB/probe iterator/sample 차이에 따른 평가 단차일 가능성이 크다.
- 실제 Run 3 내부 변화는 `0.458800 -> 0.458670`이다.
- KD-600과 Run3-600/1200/1800을 동일 seed/sample/eval MB로 재평가해야 실제 drift를 판단할 수 있다.
- 실제 drift라면 all-router gradient conflict, Wiki output KD 병행, router L2 anchor, GEM/PCGrad를 후보로 본다.

학습 시간:

```text
Run 1: 약 7시간 38분
Run 2: 약 3시간 2분
Run 3: 약 7시간 27분
유효 run 합: 약 18시간 7분
smoke/중단 포함 wall time: 약 20시간 15분
```

## 11. Stepwise Diagnostic: KD Final -> First 100 Conv Steps

목적:
- KD 직후 Conversation joint 학습 초기 100 optimizer step에서 Conversation/Code/Wiki를 매 step 측정한다.
- 초반 Wiki 변화가 언제 발생하는지 본다.

설정:
- source: Run 2 KD-600 checkpoint
- MB128, GBS2304, train 100 steps
- 세 probe 모두 interval 1, 각 25 eval iterations
- 원래 1800-step 첫 구간을 재현하도록 `LR_DECAY_ITERS=1800`, `LR_WSD_DECAY_ITERS=180`, `LR_WARMUP_FRACTION=0.01`
- diagnostic display: baseline 5400, 학습 후 step 1은 5401, step 100은 5500

### 11.1 절대 지켜야 할 baseline 규칙

- step 5400은 checkpoint reload 후 재평가값이 아니다.
- step 5400은 반드시 Run 2 KD 로그의 최종값으로 대체한다.
- 아래 6개를 **하나의 W&B payload**로 한 번에 기록한다.

```text
conversation_probe/next_token_accuracy = 0.288416
conversation_probe/ppl                 = 65.47378
code_probe/next_token_accuracy         = 0.658937
code_probe/ppl                         = 6.432723
wiki_probe/next_token_accuracy         = 0.462014
wiki_probe/ppl                         = 18.00381
```

- `RUN_INITIAL_PROBE_EVAL=0`으로 실제 학습 probe는 5401부터 시작한다.
- W&B monotonic-step 제한 때문에 baseline을 여러 calls로 나누면 안 된다.

### 11.2 시도한 run 상태

```text
g2-ffn-only-conv-joint-kd-init-stepwise-probe-mb128-100
```

- probe가 600부터 찍혀 중단. INVALID.

```text
g2-ffn-only-conv-joint-kd-init-stepwise-probe-offset5400-mb128-100
```

- 5400에 reload-time probe를 기록했고 local step 8 부근에서 중단. INVALID.

```text
g2-ffn-only-conv-joint-kd-init-stepwise-probe-corrected-offset5400-mb128-100
```

- helper가 두 probe만 지원해 Wiki를 두 번째 call로 넣다가 monotonic-step 제한으로 거부됨. INVALID.

```text
g2-kdinit-conv-stepwise-exact-baseline-v2-5400-5500-mb128
```

- 6개 KD baseline을 single payload로 전송했고 W&B에 `uploading history steps 0-0`이 표시됨.
- finish가 오래 대기해 프로세스를 종료했으므로 서버 반영을 재확인하기 전에는 이어 쓰지 않는다.
- 실제 100-step 학습은 시작하지 않았다.

현재 상태: 관련 학습 프로세스 없음. 사용자 변경 사항 대기 중.

재개 체크리스트:
1. 사용자 변경 사항을 먼저 반영한다.
2. 새 clean W&B run ID를 권장한다.
3. 6개 KD baseline이 step 5400에 모두 보이는지 확인한다.
4. `RUN_INITIAL_PROBE_EVAL=0`으로 시작한다.
5. Conversation/Code/Wiki가 5401부터 매 step 찍히는지 확인한다.

## 12. Planned Ablation: No-KD Random Expansion Chain

목적: KD-init과 random-init의 초기 안정성, 보존, sample efficiency, 최종 성능을 비교한다.

계획한 2-run chain:

```text
Wiki 8E
-> random 8 -> 16 expansion
-> Code + Wiki joint aggregation 1800 (MB128)
-> Code 16E
-> random 16 -> 24 expansion
-> Conversation + Wiki/Code joint aggregation 1800 (MB128)
-> final 24E
```

Run A:
- source Wiki 8E; experts 9~16/random router rows init
- trainable experts 9~16 + all router rows
- `L_code + L_wiki`, Code:Wiki 1:1
- MB128, GBS2304, 1800 steps; Code/Wiki probes

Run B:
- source Run A 16E; experts 17~24/random router rows init
- trainable experts 17~24 + all router rows
- Conversation + equal Wiki/Code replay; sample 비율 1:1:2
- MB128, GBS2304, 1800 steps; Conversation/Code/Wiki probes

구현 주의:
- 기존 KD-init runners는 `LOAD_EXPANDED_SOURCE=1`이라 unexpanded source에 그대로 쓰면 안 된다.
- 새 분기에서 `LOAD_EXPANDED_SOURCE=0`으로 두고 Code는 `--moe-expand-from-num-experts 8`, Conversation은 `--moe-expand-from-num-experts 16`을 사용한다.
- 별도 random-init checkpoint 없이 각 run 시작 시 expansion한다.

상태: 설계만 완료. 아직 runner/chain 구현 및 실행 안 함. Stepwise diagnostic 이후 진행.

## 13. Immediate Session Handoff

새 세션에서 먼저 할 일:
1. 이 문서의 10~13절을 읽는다.
2. Stepwise diagnostic에 대한 사용자의 변경 사항을 확인한다.
3. GPU/process 상태를 확인한다.
4. partial/invalid W&B run을 최종 결과처럼 사용하지 않는다.
5. clean 100-step diagnostic을 완료한 뒤 no-KD 2-run chain을 새 분기로 구현한다.

하지 말아야 할 것:
- 사용자 변경 사항 확인 전에 diagnostic 자동 시작
- reload-time probe를 KD final 값으로 사용
- step 5400 baseline을 여러 W&B calls로 분리
- `LOAD_EXPANDED_SOURCE=1` runner에 unexpanded checkpoint 전달
- invalid W&B run을 유효 대조군으로 사용

## 14. Planned 4-Stage Old-Data KD Chain (H100)

실험 카드:

- 목적: 1-phase에서 raw old-data LM replay를 teacher-logits KD로 바꾸면 기존 task
  보존과 새 task 적응의 trade-off가 개선되는지 확인한다.
- 대조군: 동일한 KD initialization, expert expansion, freeze mask, all-router update,
  GBS 2304, new-expert LR ramp 900, old/new sample 노출을 쓰는 joint-LM replay chain.
- 바꾸는 변수: 1-phase의 old-data branch loss만 `LM -> output-logits KD`로 바꾼다.
- 고정 변수: FFN-only G2, `8 -> 16 -> 24` experts, top-k 4,
  `moe_ffn_hidden_size=352`, 기존 experts와 dense/attention trunk freeze, 새 experts와
  all router rows 학습, optimizer update당 primary/replay gradient aggregation 1회.
- gradient 범위: new experts는 new-task LM gradient만 받고, old-data KD branch의
  non-router gradient는 복원하여 all router rows에만 KD gradient를 누적한다.
- 해석 가능 범위: joint-LM replay 대비 차이는 old-data target을 hard next-token
  label에서 frozen teacher distribution으로 바꾼 효과로 해석한다.

4-stage 설정:

```text
Wiki 8E -> Code expansion KD       600 steps, MB48
Code LM + Wiki old-data KD        1800 steps, MB96
16E -> Conversation expansion KD   600 steps, MB36
Conversation LM + Wiki/Code KD    1800 steps, MB96
```

구현:

- Core flag: `--moe-joint-replay-old-data-kd`
- Code runner: `scripts/experiment/a100/run_g2_ffn_only_code_wiki_joint_old_data_kd_allrouter_mha.sh`
- Conversation runner: `scripts/experiment/a100/run_g2_ffn_only_conversation_wikicode_joint_old_data_kd_allrouter_mha.sh`
- Chain/plan: `scripts/experiment/a100/run_g2_ffn_only_4stage_old_data_kd_chain_mha.sh`

안전 조건:

- expanded student source와 pre-expansion frozen teacher checkpoint 경로를 분리한다.
- primary branch에서는 teacher forward/KD를 끄고 LM loss만 사용한다.
- replay branch에서는 LM coefficient를 0으로 바꾸고 teacher-logits KD만 사용한다.
- 두 backward를 한 optimizer step 전에 aggregate한다.

상태 (2026-08-02): 구현, shell/Python syntax, plan-only, H100 단위 테스트 16개 통과.
체크포인트가 아직 없으므로 model-forward smoke와 실제 학습은 시작하지 않았다.
