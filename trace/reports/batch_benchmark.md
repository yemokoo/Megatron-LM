# SLoRA batch and longest-task benchmark

Updated: 2026-07-26

## Conclusion

The longest TRACE training task by raw chat-template token count is MeetingBank. All production SLoRA runs truncate to `max_length=1024`, so the benchmark used MeetingBank with the actual 1,024-token training cap. The selected four-GPU runtime profile is micro-batch 8 per device with gradient accumulation 2. It preserves effective global batch 64.

## Raw token-length audit

Every task contains 5,000 training records. With the Llama-3.1 chat template, MeetingBank has mean 3,540.7 tokens, p95 14,590, p99 31,898, maximum 79,409, and 2,858 of 5,000 records exceed 1,024 tokens. Py150 is next at mean 854.2 and maximum 75,533, with 961 records over 1,024. 20Minuten has mean 778 and maximum 2,918, with 982 records over 1,024. The other task means are C-STANCE 177.6, FOMC 117.4, ScienceQA 325, NumGLUE-cm 90, and NumGLUE-ds 80.

Qwen2.5 gives the same ordering. MeetingBank has mean 3,570.7, p95 14,700, p99 32,457, maximum 80,163, and 2,880 records above 1,024.

This establishes MeetingBank as the conservative memory test. It also establishes that raw maximum length does not become the training sequence length because TRL truncates it to 1,024.

## Four-A100 measurements

All runs used Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct, bf16, DeepSpeed ZeRO-2, gradient checkpointing, LoRA rank 64 and alpha 128, seven projection families, MeetingBank, max length 1,024, and three optimizer steps. Effective global batch was fixed at 64.

| Backbone | Micro-batch | Accumulation | Peak GPU memory | Runtime | Mean optimizer step | Samples/s |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 4 | 4 | 31,445 MiB max | 19.553 s | 6.518 s | 9.819 |
| Llama-3.1-8B | 8 | 2 | 34,817 MiB | 18.308 s | 6.103 s | 10.487 |
| Llama-3.1-8B | 16 | 1 | 52,573 MiB | 17.755 s | 5.918 s | 10.814 |
| Qwen2.5-7B | 8 | 2 | 36,379 MiB | 16.882 s | 5.627 s | 11.373 |

MB 8 is about 6.4 percent faster than MB 4 on Llama. MB 16 is only about 3.0 percent faster than MB 8 while using about 17.3 GiB more peak GPU memory. MB 8 therefore provides the better operating margin on both paper backbones.

## Production setting

The SLoRA launcher now defaults to:

```bash
MICRO_BATCH=8
GRAD_ACCUM=2
WORLD_SIZE=4
```

The effective global batch remains `8 x 2 x 4 = 64`, so the task sample count, epoch count, and optimizer steps per epoch remain unchanged. Override values remain supported, for example:

```bash
MICRO_BATCH=4 GRAD_ACCUM=4 ./scripts/run_experiment.sh train slora_pre llama31
```

The public repository batch profiles are retained separately in `config/trace_experiments.json`; the selected setting is explicitly labeled as a measured runtime profile, not a hyperparameter claimed by the paper.

## Preprocessing correction

The trainers previously constructed an in-memory Dataset independently on every distributed rank. That caused ChatML conversion, template application, tokenization, and truncation to repeat four times. They now load a disk-backed JSON Dataset with a stable fingerprint. The expensive SFT preprocessing is cached once and reused by the other ranks. Record order, sample text, message structure, and sample count are unchanged.

## Evidence

- `results/memory_smoke/llama31/mb4-ga4-20260726T032534Z`
- `results/memory_smoke/llama31/mb8-ga2-20260726T032918Z`
- `results/memory_smoke/llama31/mb16-ga1-20260726T033308Z`
- `results/memory_smoke/qwen25_7b/mb8-ga2-20260726T033845Z`
- reusable benchmark: `scripts/benchmark_slora_batch.sh`

The earlier `mb4-ga4-20260726T032438Z` directory is an invalid launcher attempt and is excluded. These short runs verify execution, memory, and relative throughput only. They do not establish convergence or reproduce paper scores.
