# KT 24.07 Setup

Use this when the KT server session must start from the allowed NGC PyTorch `24.07` image and creating a separate conda environment is not the preferred path.

## Why 24.07

Among the allowed presets (`25.05`, `25.01`, `24.07`, `23.09`), `24.07` is the closest conservative fit for this repo because:
- it keeps Python `3.10`
- it is less disruptive for Megatron-LM + TransformerEngine + Apex than the Python `3.12` images
- it is newer than `23.09` while still staying relatively close to the previously validated local stack

## Session Assumption

- one KT session = one Docker/container session
- do not create a separate conda env unless the platform explicitly supports and expects it
- install the project dependencies directly into the provided session environment

## Setup Steps

We treat the NGC `24.07` image as the base session, then recreate a project-local Python environment on top of it each time.

Recommended order on KT:

```bash
git clone <YOUR_GITHUB_URL>
cd FLAME-MoE
git checkout slurm
```

If the platform reclaims low-utilization sessions, start a tiny GPU warmup in another pane before or during install:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/miscellaneous/session_warmup.py
```

or

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/miscellaneous/run_session_warmup.sh
```

The default warmup is intentionally light: roughly 1 second of compute followed by 4 seconds of sleep. If KT still reclaims the session, increase the load a bit:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/miscellaneous/session_warmup.py --matrix-size 3072 --compute-seconds 2.0 --sleep-seconds 2.0
```

Leave warmup running while dependencies install, then stop it with `Ctrl-C`.

```bash
bash scripts/miscellaneous/install_kt_24_07_no_conda.sh
source .venv-kt2407/bin/activate
```

## Smoke Tests

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -c 'import transformer_engine, apex; print("imports ok")'
python Megatron-LM/pretrain_gpt.py --help >/tmp/pretrain_help.txt && tail -n 5 /tmp/pretrain_help.txt
```

## Important Runtime Note

Several eval/analysis wrapper scripts used to assume the local path `.conda/envs/flame3090/bin/python`.
They now fall back to `python` / `python3` automatically if that local conda env does not exist.

On KT, the preferred path is now:
- start from the NGC `24.07` session
- recreate `.venv-kt2407` with the helper script
- activate that venv before running training or eval

The setup script explicitly reinstalls a project-local PyTorch stack close to the previously validated local environment:
- Python `3.10`
- `torch 2.5.1+cu121`
- `torchvision 0.20.1+cu121`
- `torchaudio 2.5.1+cu121`
- local Apex and TransformerEngine builds

## Recommended Bring-Up Strategy

1. Start from NGC PyTorch `24.07`
2. Clone repo and checkout `slurm`
3. Run the no-conda install script
4. Confirm PyTorch + Apex + TransformerEngine imports
5. Run a tiny Megatron smoke test
6. Only then resume training/eval

## Precision Note

Because KT uses A100, `bf16` becomes realistic again. But for bring-up:
- first confirm the recreated venv imports and Megatron help path
- then run a tiny smoke test
- only after that switch the main training path to `bf16`
