# 새 서버 Codex용 전체 이전 프롬프트

아래 `Codex에게 전달할 프롬프트` 절을 새 서버의 Codex에게 그대로 전달한다. 이
프롬프트는 GitHub 소스, 두 개의 격리된 Python 환경, Hugging Face 데이터/최종
체크포인트, TRACE base model, W&B 인증과 smoke 검증까지 다룬다.

현재 HF 업로드가 완전히 끝나기 전에 새 서버 작업을 시작할 수도 있으므로, 원격
파일이 일부만 보이면 성공으로 간주하지 않고 기다리도록 구성했다. 인증 토큰과 API
key는 채팅, shell history, 로그, Git 파일에 출력하지 않는다.

## Codex에게 전달할 프롬프트

당신은 새 학습 서버에서 `LLM-continual-learning` 프로젝트를 재현해야 한다. 아래
순서를 실제로 수행하고, 각 단계의 명령과 결과를 짧게 기록하라. 단, **학습은 절대
시작하지 말고 환경·데이터·모델 준비와 read-only smoke/preflight까지만** 수행하라.

### 목표와 기준

- GitHub: `https://github.com/yemokoo/LLM-continual-learning.git`의 `main`
- 데이터: private dataset `YeMoKoo/flamedata2`
  - G2: `wiki/**`, `code/**`, `conversation/**`
  - TRACE: `trace/**`
- 최종 checkpoint: private model `YeMoKoo/LLM-continual-learning`
  - `g2_wiki_code_conversation/**`
  - `trace/**`
- TRACE base model:
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`
- FLAME/G2와 TRACE/SLoRA 환경은 절대로 합치지 않는다.
- `pip install --user`를 사용하지 않는다.
- 기존 파일을 삭제하거나 부분 다운로드를 지우지 않는다. HF 다운로드는 같은
  `--local-dir`로 재실행하여 resume한다.
- credential을 명령행 인자, `.env`, 문서, Git에 저장하지 않는다. 인증 명령을
  대화형으로 열고 사용자가 직접 token/key를 입력하게 한다.

예상 저장 공간은 대략 다음과 같다.

- dataset 약 22.7 GB
- 최종 checkpoint 약 89.1 GiB
- TRACE base model 2개 약 30~35 GB
- Conda/venv, 빌드 산출물과 HF cache 여유 공간

작업 대상 filesystem에 최소 220 GiB, 가능하면 300 GiB 이상이 비어 있는지 먼저
확인하라. 부족하면 대용량 다운로드 전에 멈추고 사용자에게 알린다.

### 0. 작업 경로와 호스트 사전 점검

먼저 사용자에게 새 서버에서 사용할 상위 작업 경로 하나만 확인한다. 아래에서
`MIGRATION_ROOT`는 그 절대경로다. 이전 서버의
`/home/work/Agent_HJ/30_flame_agent`를 새 서버에 하드코딩하지 않는다.

```bash
export MIGRATION_ROOT=/사용자가_지정한_절대경로
```

다음을 read-only로 확인하고 결과를 기록한다.

```bash
date -u
uname -a
uname -m
nvidia-smi
nvcc --version
gcc --version
g++ --version
df -h "${MIGRATION_ROOT}"
command -v git
command -v git-lfs || true
command -v gh || true
command -v conda || true
```

FLAME/G2 기준 호스트는 Linux x86_64, NVIDIA A100, CUDA 12.4 런타임을 지원하는
driver, CUDA toolkit 12.4 또는 12.5의 `nvcc`, GCC/G++ 11이다. GPU/driver/toolkit이
다르면 설치를 억지로 진행하지 말고 차이를 보고하라. H100 등 다른 GPU에서의 빌드나
학습은 별도 검증이 필요하다.

Conda가 없다면 자동으로 임의 배포판을 섞지 말고, 사용자에게 project-local
Miniforge 설치 허용과 설치 위치를 확인한 뒤 공식 Miniforge를 설치한다. 설치 후에는
반드시 그 Conda의 `conda`를 사용한다.

### 1. GitHub 인증과 소스 clone

GitHub CLI가 있다면 다음 대화형 인증을 사용한다. PAT를 shell에 `echo`하거나 URL에
넣지 않는다.

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth status
gh auth setup-git
```

`gh`가 없고 저장소가 public으로 접근 가능하면
`git clone --recurse-submodules https://github.com/yemokoo/LLM-continual-learning.git`
를 사용할 수 있다. private 접근이 필요하면 PAT를 URL에 넣는 방식으로 우회하지 말고
사용자 허가를 받아 GitHub CLI를 설치한 후 위의 대화형 로그인을 사용한다.

인증 후 다음을 수행한다.

```bash
cd "${MIGRATION_ROOT}"
gh repo clone yemokoo/LLM-continual-learning
cd LLM-continual-learning
git checkout main
git pull --ff-only origin main
git submodule sync --recursive
git submodule update --init --recursive
git status --short --branch
git submodule status --recursive
```

이미 clone되어 있다면 새로 clone하지 말고 remote가 위 URL인지 확인한 다음
`pull --ff-only`와 submodule update만 수행한다. `git reset --hard`, 강제 checkout,
로컬 변경 삭제는 하지 않는다.

다음 문서를 먼저 읽고 경로/버전 기준으로 삼는다.

```text
environments/README_KO.md
docs/hf_migration_final_checkpoints_ko.md
RESEARCH_CODE_HANDOFF.md
trace/docs/PORTABILITY_KO.md
trace/docs/RUNBOOK_KO.md
```

### 2. FLAME/G2 전용 Conda 환경

TRACE venv가 활성화되어 있지 않은 새 shell에서 저장소 root 기준으로 실행한다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
git submodule update --init --recursive
mkdir -p .local/logs/bootstrap
MAX_JOBS=8 bash scripts/miscellaneous/install_a100_env.sh \
  2>&1 | tee .local/logs/bootstrap/install_flame_env.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flame-megatron-a100
source scripts/miscellaneous/activate_flame_env.sh
python scripts/miscellaneous/verify_flame_env.py --require-gpu \
  2>&1 | tee .local/logs/bootstrap/verify_flame_env.log
```

두 로그는 Git에 add하지 않는다. 설치 스크립트는 다음을 고정 설치/빌드한다.

- Python 3.10
- PyTorch `2.4.1+cu124`, torchvision `0.19.1+cu124`, torchaudio
  `2.4.1+cu124`
- Transformers `4.33.1`, Datasets `2.20.0`, W&B `0.28.1`
- **grouped-gemm `1.1.4`, commit
  `172fada89fa7364fe5d026b3a0dfab58b591ffdd`**
- repository submodule의 Apex와 TransformerEngine `1.11.0`
- flash-attn `2.4.2`

grouped-gemm은 선택적인 장식이 아니다. FFN MoE와 attention LoRA expert 실행에서
각각 `--moe-grouped-gemm`, `--attn-lora-grouped-gemm` 경로가 사용된다. 검증기의
JSON에서 아래 조건을 모두 확인하라.

- `"ok": true`
- `grouped-gemm` version과 commit이 위 값과 동일
- `apex`, `transformer-engine`, `flash-attn` import 성공
- `megatron.core` import 성공
- `cuda_available: true`, 예상 GPU 개수와 모델명 확인
- Megatron-LM/Apex/TransformerEngine source commit 검증 통과

extension 빌드가 실패했는데 `--skip-extensions`로 우회하여 환경 완료로 보고하지
마라. G2 학습 환경은 전체 extension 검증이 통과해야 완료다.

### 3. TRACE/SLoRA 전용 venv

FLAME Conda 환경을 deactivate하고 가능하면 새 shell에서 실행한다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
conda deactivate || true
unset PYTHONPATH FLAME_MOE_REPO_DIR

cd trace
./scripts/setup_runtime.sh 2>&1 | tee ../.local/logs/bootstrap/install_trace_env.log
source .venv-runtime/bin/activate
export PYTHONNOUSERSITE=1
python scripts/runtime_preflight.py --skip-gpu \
  2>&1 | tee ../.local/logs/bootstrap/verify_trace_env.log
```

이 환경은 PyTorch `2.4.1+cu124`, Transformers `4.51.3`, PEFT `0.12.0`, TRL
`0.16.1`, DeepSpeed `0.16.9` 계열의 committed lock을 사용한다. FLAME 환경의
Transformers/Apex/TransformerEngine을 이 venv에 설치하거나 반대로 섞지 않는다.

### 4. Hugging Face와 W&B 로그인

TRACE venv의 CLI를 기준으로 HF에 대화형 로그인한다. private dataset/model을 읽을
수 있는 token이 필요하고, Llama를 받으려면 같은 HF 계정에서 Meta Llama 3.1 사용
조건을 미리 승인받아야 한다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning/trace"
source .venv-runtime/bin/activate
.venv-runtime/bin/hf auth login
.venv-runtime/bin/hf auth whoami
```

네트워크 reset으로 `whoami`만 실패했지만 login success와 credential 저장이 먼저
출력된 경우 token을 다시 노출하지 말고 `hf auth whoami`를 재시도한다.

W&B도 API key를 채팅이나 명령행에 붙이지 말고 대화형으로 입력한다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flame-megatron-a100
source scripts/miscellaneous/activate_flame_env.sh
wandb login --relogin
wandb status
```

기존 G2 프로젝트 기본값은 `flame-continual-top2-qv-lora`이고 사용 entity는
`yemoyemo010831-korea-university`다. 계정에 이 entity 접근권한이 있는지 확인한다.
검증/smoke 동안에는 아래처럼 offline을 유지한다. 사용자가 실제 학습을 승인한 뒤에만
`WANDB_MODE=online`으로 바꾼다.

```bash
export WANDB_ENTITY=yemoyemo010831-korea-university
export WANDB_PROJECT=flame-continual-top2-qv-lora
export WANDB_MODE=offline
export WANDB_CONSOLE=off
```

### 5. HF 업로드 완료 여부를 먼저 검증

다운로드 전에 다음 Python 검사를 실행한다. 일부 prefix만 존재하면 업로드 중일 수
있으므로 기다리고 재검사한다. 빈 모델 repo나 부분 업로드를 완료로 간주하지 않는다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
trace/.venv-runtime/bin/python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
info = api.repo_info(
    "YeMoKoo/flamedata2", repo_type="dataset", files_metadata=True
)
files = list(info.siblings)
expected = {
    "wiki/": (8, 10_150_425_783),
    "code/": (8, 9_511_983_218),
    "conversation/": (94, 2_853_559_927),
    "trace/": (24, 163_854_918),
}
bad = []
for prefix, (expected_count, expected_bytes) in expected.items():
    rows = [f for f in files if f.rfilename.startswith(prefix)]
    size = sum(int(f.size or 0) for f in rows)
    ok = len(rows) == expected_count and size == expected_bytes
    print(prefix, "files=", len(rows), "bytes=", size, "ready=", ok)
    if not ok:
        bad.append(prefix)
if bad:
    raise SystemExit("dataset upload is incomplete: " + ", ".join(bad))
print("dataset repository is complete")
PY
```

모델 repo는 local manifest의 enabled 항목마다 remote folder가 있는지 확인한다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
trace/.venv-runtime/bin/python - <<'PY'
import csv
from pathlib import Path
from huggingface_hub import HfApi

root = Path.cwd()
with (root / "scripts/hf/final_checkpoint_manifest.tsv").open() as f:
    rows = [r for r in csv.DictReader(f, delimiter="\t") if r["enabled"] == "1"]
info = HfApi().repo_info(
    "YeMoKoo/LLM-continual-learning", repo_type="model", files_metadata=True
)
names = {f.rfilename for f in info.siblings}
missing = []
for row in rows:
    prefix = row["path_in_repo"].rstrip("/") + "/"
    under = [n for n in names if n.startswith(prefix)]
    if row["kind"] == "megatron":
        ok = prefix + "latest_checkpointed_iteration.txt" in names and any(
            n.startswith(prefix + "iter_") for n in under
        )
    else:
        ok = any(
            n.endswith((".safetensors", ".bin", ".pt", ".pth")) for n in under
        )
    print(row["group"], row["name"], "ready=", ok, "files=", len(under))
    if not ok:
        missing.append(row["name"])
if missing:
    raise SystemExit(f"model upload is incomplete: {len(missing)} entries missing")
print(f"model repository is complete: {len(rows)} manifest entries")
PY
```

첫 번째 검사가 실패하면 dataset 다운로드를 시작하지 말고 기다린다. 두 번째만
실패하면 dataset과 환경 준비는 계속할 수 있지만 checkpoint 다운로드는 기다린다.

### 6. G2 Wiki/Code/Conversation 데이터 다운로드

저장소 root의 `data/wiki`, `data/code`, `data/conversation`이 되도록 반드시
`--local-dir data`를 쓴다. 연결이 끊기면 같은 명령을 재시도한다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
HF=trace/.venv-runtime/bin/hf
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1

until "${HF}" download YeMoKoo/flamedata2 \
  --repo-type dataset \
  --include "wiki/**" "code/**" "conversation/**" \
  --local-dir data \
  --max-workers 1; do
  echo "HF dataset download interrupted; retrying in 10 seconds" >&2
  sleep 10
done
```

다음으로 Megatron indexed dataset의 `.bin`/`.idx` stem이 정확히 짝을 이루는지
검증한다.

```bash
trace/.venv-runtime/bin/python - <<'PY'
from pathlib import Path

root = Path("data")
bad = []
for task in ("wiki", "code", "conversation"):
    for split in ("train", "test"):
        folder = root / task / split
        bins = {p.stem for p in folder.glob("*.bin")}
        idxs = {p.stem for p in folder.glob("*.idx")}
        ok = folder.is_dir() and bins and bins == idxs
        print(folder, "pairs=", len(bins & idxs), "ready=", ok)
        if not ok:
            bad.append(str(folder))
if bad:
    raise SystemExit("invalid indexed dataset folders: " + ", ".join(bad))
PY
du -sh data/wiki data/code data/conversation
```

### 7. TRACE 데이터와 base model 다운로드

TRACE 데이터는 지원 스크립트로 `trace/data/trace`에 받는다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning/trace"
source .venv-runtime/bin/activate
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1

until ./scripts/data/download_trace_from_hf.sh; do
  echo "TRACE dataset download interrupted; retrying in 10 seconds" >&2
  sleep 10
done
```

그다음 base model 두 개를 받는다. 스크립트 자체가 single-worker resume/retry와 full
preflight를 수행한다.

```bash
./scripts/download_paper_models.sh \
  2>&1 | tee ../.local/logs/bootstrap/download_trace_models.log
```

Llama gated access가 거부되면 다른 mirror로 바꾸지 말고, 사용자에게 HF license
승인이 필요하다고 보고한다.

### 8. 최종 checkpoint 다운로드

5단계의 model repository 검사가 완전히 통과한 뒤에만 실행한다. 원격 Hub의 정리된
폴더 구조를 보존하여 `.local/hf_models`에 받는다.

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
HF=trace/.venv-runtime/bin/hf
mkdir -p .local/hf_models
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1

until "${HF}" download YeMoKoo/LLM-continual-learning \
  --repo-type model \
  --include "g2_wiki_code_conversation/**" \
  --local-dir .local/hf_models \
  --max-workers 1; do
  echo "G2 checkpoint download interrupted; retrying in 15 seconds" >&2
  sleep 15
done

until "${HF}" download YeMoKoo/LLM-continual-learning \
  --repo-type model \
  --include "trace/**" \
  --local-dir .local/hf_models \
  --max-workers 1; do
  echo "TRACE checkpoint download interrupted; retrying in 15 seconds" >&2
  sleep 15
done
```

Megatron checkpoint는 각 root의 tracker와 단일 최종 iteration이 있어야 한다.
임의로 원래 서버 절대경로에 복사하지 말고, 이후 runner의 `WIKI_SOURCE`, `LOAD`,
`SOURCE_CHECKPOINT`류 override에 `.local/hf_models/...`의 정리된 경로를 명시한다.

```bash
find .local/hf_models/g2_wiki_code_conversation \
  -name latest_checkpointed_iteration.txt -print | sort
du -sh .local/hf_models/g2_wiki_code_conversation .local/hf_models/trace
```

### 9. 학습 없는 smoke와 plan 검증

FLAME/G2:

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flame-megatron-a100
source scripts/miscellaneous/activate_flame_env.sh
export WANDB_MODE=offline
python scripts/miscellaneous/verify_flame_env.py --require-gpu

PLAN_ONLY=1 bash \
  scripts/experiment/a100/run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh \
  logits
```

plan 출력에서 최신 attention shared-router chain 기본값이 아래와 같은지 확인한다.

- Code KD: 600 step, micro batch 48
- Code 1-phase: 1800 step, micro batch 96
- Conversation KD: 600 step, micro batch 36
- Conversation 1-phase: 1800 step, micro batch 96
- LR ramp 없음
- KD에는 LM coefficient 0, 1-phase에는 KD off

이것은 plan 출력만 보는 명령이다. 실제 stage를 시작하지 않는다.

TRACE/SLoRA:

```bash
cd "${MIGRATION_ROOT}/LLM-continual-learning/trace"
source .venv-runtime/bin/activate
export WANDB_MODE=offline
python scripts/runtime_preflight.py --skip-gpu
python scripts/runtime_preflight.py --model llama31 --world-size 4
./scripts/smoke.sh
./scripts/baselines/llama31/suite.sh validate
```

실제 model-forward smoke가 GPU 메모리를 크게 점유하거나 프로세스를 띄우는 경우에는
먼저 실행 명령과 예상 GPU 수를 사용자에게 보여주고 승인을 받은 뒤 수행한다. 어떤
경우에도 continual training, KD, 1-phase, W&B online run을 시작하지 않는다.

### 10. 최종 보고 형식

마지막에 다음 표를 제공한다.

| 항목 | 상태 | 실제 위치/버전 | 검증 근거 |
|---|---|---|---|
| Git main/submodule | OK/FAIL | commit | git status |
| FLAME Conda | OK/FAIL | Python/torch/CUDA | verify JSON |
| grouped-gemm | OK/FAIL | version + commit | verify JSON |
| Apex/TE/flash-attn | OK/FAIL | versions | import 결과 |
| TRACE venv | OK/FAIL | 주요 버전 | preflight |
| HF/W&B auth | OK/FAIL | account/entity만 | token은 출력 금지 |
| G2 datasets | OK/FAIL | 경로와 크기 | bin/idx pair 수 |
| TRACE dataset | OK/FAIL | 경로와 24 JSON | downloader check |
| base models | OK/FAIL | 두 경로 | full preflight |
| G2 checkpoints | OK/WAIT | 경로와 크기 | tracker 수 |
| TRACE checkpoints | OK/WAIT | 경로와 크기 | manifest prefix |
| smoke/plan | OK/FAIL | 실행 명령 | exit code |

실패한 단계가 있으면 오류의 마지막 50~100줄, 현재 disk/GPU 상태, 재시도 명령을
함께 보고하라. 성공하지 않은 단계를 성공으로 표현하지 말고, 사용자 승인 없이 학습을
시작하지 마라.
