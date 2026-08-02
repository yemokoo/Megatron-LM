# TRACE continual-learning and SLoRA reproduction

This directory is the self-contained TRACE-family subproject of
`LLM-continual-learning`. It vendors the exact local code snapshots used for TRACE,
SLoRA, LoRAMoE, and Ours LoRA-MoE v1/v2/v2.5. The supported launchers no longer
depend on sibling `TRACE`, `slora_repro`, `llmcl_benchmark`, or `envs` directories.

Large datasets, base models, token caches, checkpoints, and logs remain server-local
and are intentionally ignored by Git. See `docs/PORTABILITY_KO.md` for the migration
contract and `manifests/source_provenance.json` for the vendored source revisions.

The current Korean runbook and one-command launchers are:

- `EXPERIMENTS.md` for the implementation/change index and live experiment log
- `docs/RUNBOOK_KO.md`
- `docs/PORTABILITY_KO.md` for moving the project to a new server
- `scripts/run_experiment.sh` for one method
- `scripts/run_suite.sh` for the legacy SLoRA/TRACE method matrix
- `scripts/baselines/README_KO.md` and `scripts/baselines/<model>/<method>.sh`
  for model-separated baseline launchers, including the local LoRAMoE port
- `scripts/model_forward_smoke.py` for a non-training CUDA model-load check

## New server setup and quick validation

The code tree is portable; runtime files are restored explicitly after cloning.

```bash
cd LLM-continual-learning/trace
./scripts/setup_runtime.sh
hf auth login
./scripts/data/download_trace_from_hf.sh
./scripts/download_paper_models.sh
python scripts/preflight.py --mode full \
  --models llama31_8b_instruct qwen25_7b_instruct
./scripts/run_suite.sh validate llama31
./scripts/run_suite.sh validate qwen25_7b
```

Run one complete method, including normalized result collection:

```bash
./scripts/run_experiment.sh all slora_pre llama31
```

## Layout

- `implementations/llmcl_benchmark/`: vendored latest LoRAMoE and Ours LoRA-MoE code.
- `implementations/*-repro/`: runnable SLoRA, TRACE, and O-LoRA reproduction ports.
- `upstream/`: pristine, pinned SLoRA, TRACE, and O-LoRA snapshots.
- `patches/`: reviewable proposals; none are applied to pristine snapshots.
- `config/models.json`: exact paper model IDs and planned paths.
- `config/trace_experiments.json`: paper facts and explicitly labeled batch profiles.
- `config/run_profiles.json`: paper candidates, official baselines, corrected baselines, and smoke-only paths.
- `manifests/source_provenance.json`: source commits and snapshot roles.
- `manifests/preflight.json`: generated machine, model, and dataset evidence.
- `reports/implementation_status.md`: current verified readiness and remaining experimental work.
- `reports/batch_benchmark.md`: longest-task token audit and measured four-GPU batch selection.
- `reports/reproducibility_report.md`: archived model-free audit and formula/source analysis.
- `scripts/launch_plan.py`: renders a plan and provenance without launching training.
- `scripts/compare_results.py`: validates triangular score matrices and computes final average and paper-style AFR.

Example render-only launch plan:

```bash
./scripts/audit_python.sh scripts/launch_plan.py slora_pre_trace \
  --output results/slora_pre_launch_plan.json
```

The render-only launch plan is retained as audit evidence. Actual training uses
`scripts/run_experiment.sh`; it performs model/runtime preflight, training,
evaluation, and normalized result collection.

## Runtime artifact policy

Base models belong under `models/`, TRACE JSON under `data/trace`, generated token
caches under `cache/`, and experiment outputs under `results/`. These directories
are not committed. `TRACE_DATA_ROOT`, model path variables, and method-specific
output variables can override every default without editing source files.
