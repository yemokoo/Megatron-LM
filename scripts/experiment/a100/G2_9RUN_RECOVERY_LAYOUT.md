# G2 old-data distillation 9-run recovery

Permanent output root:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808`

- `00_sources/wiki_ffn_only_e8_step1800/`: downloaded Hugging Face FFN-only Wiki source checkpoint.
- `01_common_kd_init/code_e8_to_e16_wiki_kd_step600/`: common FFN-only 8-to-16 expert output-KD initialization checkpoint.
- `02_nine_runs/local/weights/`: the nine experiment checkpoints, retaining the original G2 hierarchy.
- `03_logs/recovery_chain.log`: top-level wait/download/training chain log.
- `03_logs/nine_stages/{hidden_kl,hidden_mse,vocab_kl}/`: per-stage logs.
- `04_scratch/`: temporary dataset and checkpoint staging. This is also placed on persistent `/data2` so a reboot does not erase it.

The chain waits for the TRACE V2-new-top4 final checkpoint `7`, downloads the Wiki source, creates the missing common R1 checkpoint, then runs hidden MSE, hidden softmax-KL, and output-vocabulary KL in that order. Each objective has Code continual training, Conversation expansion KD, and Conversation continual training. After all nine stages, it runs the 15-cell TRACE evaluation and writes `sparse15_summary.json` in the TRACE run directory.
