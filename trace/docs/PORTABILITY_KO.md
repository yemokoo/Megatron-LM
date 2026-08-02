# TRACE/SLoRA 단일 저장소 이식 구조

`trace/`는 `LLM-continual-learning` 저장소만 clone해도 TRACE 계열 실험 코드를
실행할 수 있도록 구성한 하위 프로젝트다. 바깥 `TRACE`, `slora_repro`,
`llmcl_benchmark`, `envs/train_env` 디렉터리는 기본 실행에 필요하지 않다.

## 경로 규약

| 용도 | 기본 위치 | override |
|---|---|---|
| TRACE 데이터 | `trace/data/trace` | `TRACE_DATA_ROOT` |
| Python | `trace/.venv-runtime/bin/python` | `TRACE_PYTHON`, 방법별 Python 변수 |
| 모델 | `trace/models/<model>` | `SLORA_LLAMA31_PATH`, `SLORA_QWEN25_7B_PATH` |
| 최신 LoRA-MoE 코드 | `trace/implementations/llmcl_benchmark` | `OURS_LORAMOE_ROOT`, `LORAMOE_ROOT` |
| 결과 | `trace/results` | 방법별 output root 변수 |

`data`, `models`, `cache`, `results`, runtime은 대용량 서버 로컬 파일이므로 Git에
포함하지 않는다. 소스 snapshot 출처는 `manifests/source_provenance.json`에 있다.

## 새 서버 준비

```bash
git clone <repository-url>
cd LLM-continual-learning/trace

./scripts/setup_runtime.sh
hf auth login
./scripts/data/download_trace_from_hf.sh
./scripts/download_paper_models.sh

python scripts/preflight.py --mode full \
  --models llama31_8b_instruct qwen25_7b_instruct
./scripts/baselines/llama31/suite.sh validate
```

기본 데이터 저장소는 private `YeMoKoo/flamedata2`이며 그 안의 `trace/**`만
다운로드한다. 토큰은 코드나 설정 파일에 저장하지 않고 HF CLI credential을 쓴다.

## 실행 코드 경계

- `scripts/baselines/**`, `scripts/run_experiment.sh`: 지원되는 정식 진입점
- `implementations/*`: 실제 학습·평가 구현 snapshot
- `upstream/*`: 비교용 pristine source snapshot
- `patches`, `reports`, `EXPERIMENTS.md`: 감사와 과거 실험 근거

`implementations/llmcl_benchmark/scripts` 안의 예전 서버 전용 one-off 스크립트는
실험 기록 보존용이며 일부는 과거 절대경로를 포함한다. 현재 지원 실행은 반드시
`trace/scripts/baselines`의 래퍼를 통한다.
