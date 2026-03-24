# Experiment Entrypoints

Use these top-level scripts as the canonical entrypoints:

- `3090/a_local_fp32.sh`
- `3090/b_local_fp32.sh`
- `3090/a_to_b_local_fp32.sh`
- `3090/b_to_a_local_fp32.sh`
- `3090/a_to_b_new_only_local_fp32.sh`
- `3090/b_to_a_new_only_local_fp32.sh`
- `pretrain_wiki_dense_local_bf16.sh`
- `continual_code_from_wiki_dense_local_bf16.sh`
- `pretrain_wiki_qv_lora_local_bf16.sh`
- `continual_code_from_wiki_qv_lora_expand_local_bf16.sh`

The older `stage_*` scripts are retained as implementation details and
compatibility shims.
