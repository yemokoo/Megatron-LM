# Qwen2.5-7B-Instruct 베이스라인

이 폴더의 모든 실행 파일은 다음 모델로 고정된다.

`trace/models/Qwen2.5-7B-Instruct`

```bash
cd LLM-continual-learning/trace

./scripts/baselines/qwen25_7b/list.sh
./scripts/baselines/qwen25_7b/seq_lora.sh validate
./scripts/baselines/qwen25_7b/slora_pre.sh train
./scripts/baselines/qwen25_7b/gem_corrected.sh all
./scripts/baselines/qwen25_7b/loramoe.sh train
./scripts/baselines/qwen25_7b/ours_lora_moe_v1.sh validate
./scripts/baselines/qwen25_7b/ours_lora_moe_v2.sh train
./scripts/baselines/qwen25_7b/suite.sh validate
```

각 방법 파일은 `validate`, `train`, `eval`, `all` 중 하나를 인자로 받는다.
`sd_lora.sh`, `rcl.sh`, `unified_olora.sh`는 공개 구현 부재를 기록하는
fail-fast 파일이며 학습을 실행하지 않는다.
