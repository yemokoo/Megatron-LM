# Hugging Face 서버 이전용 최종 체크포인트 정리

대상 모델 저장소는 private `YeMoKoo/LLM-continual-learning`이다. 실제 업로드 목록의 단일 기준은
`scripts/hf/final_checkpoint_manifest.tsv`이며, 검증 및 업로드 도구는
`scripts/hf/upload_final_checkpoints.py`다.

## 최종 체크포인트의 기준

- Megatron G2: 체크포인트 root 전체를 올리지 않는다. `latest_checkpointed_iteration.txt`와 그 tracker가 가리키는 단 하나의 `iter_XXXXXXX/`만 올린다.
- TRACE/S-LoRA: continual task별 root에 저장된 최종 파일만 올린다. `checkpoint-*`, cache, evaluation, replay memory는 제외한다.
- S-LoRA의 `max.safetensors`는 denoised LoRA 로딩에 사용되므로 보존한다. 이미 merge된 `max.safetensors`와 중복인 `max.safetensors.rank*-of-*` 임시 shard는 제외한다.
- base model은 공개 Hugging Face 모델에서 다시 받을 수 있으므로 이 저장소에 복제하지 않는다.
- optimizer/log/W&B/TensorBoard 파일은 최종 모델 로딩에 필요하지 않으므로 제외한다. 단, Megatron distributed checkpoint shard 안에 함께 직렬화된 상태는 shard를 쪼갤 수 없으므로 그대로 둔다.

## Hub 폴더 구조

```text
YeMoKoo/LLM-continual-learning
├── g2_wiki_code_conversation/
│   ├── sources/
│   │   ├── ffn_only/wiki/
│   │   └── ffn_attn_shared_router/wiki/
│   ├── baseline/
│   │   ├── ffn_only/{code_task,code_router_retune,conversation_task}/
│   │   └── ffn_attn_shared_router/{code_task,code_router_retune,conversation_freeze_old}/
│   └── kd_1phase/
│       ├── ffn_only/{code,conversation}/{kd_init,one_phase}/
│       └── ffn_attn_shared_router/{code,conversation}/{kd_init,one_phase}/
└── trace/
    ├── pre_local/order1/
    ├── seq/order1..order8/
    ├── ours_lora_moe_v1/task0..task7/
    ├── ours_lora_moe_v2/task0..task7/
    ├── ours_lora_moe_v2_5/task0..task7/
    └── upstream_pre/order1..order8/
```

Wiki source checkpoint는 baseline과 KD/1-phase가 공유하므로 Hub에서 중복 복사하지 않는다.

## 2026-08-02 검증 상태

- G2 enabled 16개: tracker와 최종 iteration 디렉터리 검증 완료, 선택 용량 약 55.62 GiB.
- TRACE enabled 41개: 각 task root의 최종 weight 검증 완료, 선택 용량 약 33.51 GiB. 중간 `checkpoint-*`와 중복 rank shard는 선택되지 않는다.
- Attention load-fix v3의 마지막 Conversation 1-phase는 2026-08-02에 1800 step과 최종 checkpoint 저장이 완료되어 manifest에 포함했다.
- weight가 없는 EWC/LwF/GEM/O-LoRA 결과 디렉터리는 모델 업로드 대상이 아니다.

## 사용법

Hugging Face가 설치되고 로그인된 Python을 사용한다. 기본 실행은 read-only dry run이다.

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

HF_PY=/home/work/Agent_HJ/30_flame_agent/slora_repro/.venv-runtime/bin/python

$HF_PY scripts/hf/upload_final_checkpoints.py --group g2
$HF_PY scripts/hf/upload_final_checkpoints.py --group trace
```

미완료 항목까지 감사하려면 다음 명령을 사용한다.

```bash
$HF_PY scripts/hf/upload_final_checkpoints.py --group g2 --include-disabled
```

실제 업로드는 목록을 다시 검토한 뒤에만 `--execute`를 붙인다.

```bash
$HF_PY scripts/hf/upload_final_checkpoints.py --group g2 --execute
$HF_PY scripts/hf/upload_final_checkpoints.py --group trace --execute
```

일부 항목만 올리려면 manifest name glob을 쓴다.

```bash
$HF_PY scripts/hf/upload_final_checkpoints.py \
  --group g2 \
  --name 'g2_ffn_only_*' \
  --execute
```

도구는 업로드 전에 다음 조건을 다시 검사한다.

1. source 디렉터리가 존재하는가.
2. Megatron tracker가 manifest의 expected iteration과 정확히 같은가.
3. 해당 `iter_XXXXXXX`가 존재하는가.
4. TRACE task root에 실제 model weight가 있는가.

검증 실패가 하나라도 있으면 업로드 단계로 진행하지 않는다.
