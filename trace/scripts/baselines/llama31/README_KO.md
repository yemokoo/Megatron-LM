# Llama-3.1-8B 베이스라인

Ours LoRA-MoE V1/V2/V2-new/V2-new-top4/V2.5/V3는 모두 다음 base 모델로 고정된다.

`trace/models/Llama-3.1-8B`

Instruct 모델을 사용하지 않는다. 다른 upstream baseline launcher의 모델
경로는 각 구현의 재현 설정에서 별도로 관리한다.

```bash
cd LLM-continual-learning/trace

./scripts/baselines/llama31/list.sh
./scripts/baselines/llama31/seq_lora.sh validate
./scripts/baselines/llama31/slora_pre.sh train
./scripts/baselines/llama31/gem_corrected.sh all
./scripts/baselines/llama31/loramoe.sh train
./scripts/baselines/llama31/ours_lora_moe_v1.sh validate
./scripts/baselines/llama31/ours_lora_moe_v2.sh train
./scripts/baselines/llama31/ours_lora_moe_v2_new_top4.sh train
./scripts/baselines/llama31/suite.sh validate
```

각 방법 파일은 `validate`, `train`, `eval`, `all` 중 하나를 인자로 받는다.

`seq_lora.sh train`은 `cache/tokenized/llama31_8b/slora_chat_full_len1024/`의
사전 토큰 캐시를 기본 사용한다. tokenizer hash, format, max length 또는 task
cache가 맞지 않으면 학습 전에 실패한다.
`sd_lora.sh`, `rcl.sh`, `unified_olora.sh`는 공개 구현 부재를 기록하는
fail-fast 파일이며 학습을 실행하지 않는다.
