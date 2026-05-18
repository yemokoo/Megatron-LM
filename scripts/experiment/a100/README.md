# A100 MoE BF16 Experiments

Principles:
- Repackage the existing fp32 MoE continual-learning experiments into A100-oriented bf16 launchers
- Disable the MoE shared expert path
- Train routed experts only
- Apply output-level KL only when the shared trunk remains trainable
- Skip teacher-model loading entirely for freeze runs

Default model values:
- `NUM_LAYERS=9`
- `HIDDEN_SIZE=1024`
- `FFN_HIDDEN_SIZE=5472`
- `MOE_FFN_HIDDEN_SIZE=704`
- `NUM_EXPERTS=4` for base stage
- `NUM_EXPERTS=7` for continual stage

Experiment entrypoints:
- `wiki A`: [wiki_a_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_a_a100_bf16.sh)
- `wiki A dense-x`: [wiki_a_dense_x_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_a_dense_x_a100_bf16.sh)
- `wiki A2 MHA`: [wiki_a_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_a_a2_mha_a100_bf16.sh)
- `A2 MHA unfreeze`: [a_to_b_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_a2_mha_a100_bf16.sh)
- `A2 MHA freeze`: [a_to_b_freeze_a2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_freeze_a2_mha_a100_bf16.sh)
- `wiki E2 MHA`: [wiki_e2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_e2_mha_a100_bf16.sh)
- `E2 MHA continual`: [code_from_wiki_e2_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_e2_mha_a100_bf16.sh)
- `code B`: [code_b_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_b_a100_bf16.sh)
- `A -> B`: [a_to_b_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_a100_bf16.sh)
- `A -> B freeze`: [a_to_b_freeze_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/a_to_b_freeze_a100_bf16.sh)
- `B -> A`: [b_to_a_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/b_to_a_a100_bf16.sh)
- `B -> A freeze`: [b_to_a_freeze_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/b_to_a_freeze_a100_bf16.sh)
- `G2-matched FFN wiki`: [wiki_ffn_moe_g2matched_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/wiki_ffn_moe_g2matched_mha_a100_bf16.sh)
- `G2-matched FFN attention freeze`: [code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh)
- `G2-matched FFN attention full-rank LoRA`: [code_from_wiki_ffn_moe_g2matched_attn_full_rank_lora_mha_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/code_from_wiki_ffn_moe_g2matched_attn_full_rank_lora_mha_a100_bf16.sh)
- `G2-matched FFN attention baseline chain`: [run_g2matched_ffn_attention_baselines_mha.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_g2matched_ffn_attention_baselines_mha.sh)

Default continual rules:
- expansion: `4 -> 7`
- old routed experts/router: freeze
- shared trunk: train in the unfreeze baseline, freeze in the freeze baseline
- KL: enable only when the shared trunk remains trainable, with `lambda=1.0` and `temperature=1.0`

Reference implementation:
- shared helper: [common.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/common.sh)
- base runner: [run_base_moe_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_base_moe_a100_bf16.sh)
- continual runner: [run_continual_moe_a100_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_continual_moe_a100_bf16.sh)
- sequential launcher: [run_all_moe_a100_bf16_sequential.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/run_all_moe_a100_bf16_sequential.sh)
- no-shared-expert model config: [flame-moe-bf16-no-shared.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/flame-moe-bf16-no-shared.sh)

Operational notes:
- offline W&B training/upload workflow: [offline_wandb_upload_workflow_ko.md](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/docs/offline_wandb_upload_workflow_ko.md)
