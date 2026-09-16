# Table-1 baseline 러너

TRACE 8-task에서 Table 1의 6개 방법을 돌리는 코드. Ours 런과 동일한 계약
(global batch 64, r64/α128/dropout 0.05, lr 2e-4, cosine + warmup 0.03,
seed 2025, `slora_chat_full` 템플릿, Instruct token cache) 아래에서만 방법이
달라지도록 만들었다.

기존 코드는 손대지 않았다. `main_Ours_LoRA_MoE.py`, `Ours_LoRA_MoE*.py`,
`continual_lora.py`, `paper_baselines.py`는 그대로이고 전부 새 파일이다.

## 파일

| 경로 | 내용 |
|---|---|
| `implementations/llmcl_benchmark/model/tab1_lora.py` | target-set 기반 LoRA 부착, O-LoRA merge, EWC Fisher/anchor |
| `implementations/llmcl_benchmark/model/tab1_baselines.py` | seq_lora / ewc / olora / mtl 트레이너 |
| `implementations/llmcl_benchmark/model/tab1_moe.py` | lifelong_moe / moe_lpr 트레이너 + scope 어댑터 |
| `implementations/llmcl_benchmark/model/tab1_checkpoint.py` | 체크포인트 재구성 |
| `implementations/llmcl_benchmark/training/main_tab1.py` | 학습 진입점 |
| `implementations/llmcl_benchmark/evaluate_tab1.py` | 평가 진입점 (생성·채점은 기존 evaluator 재사용) |
| `scripts/baselines/_run_tab1.sh` | 러너 |
| `scripts/baselines/{llama31,qwen25_7b}/tab1_*.sh` | 방법별 wrapper |
| `scripts/baselines/methods_tab1.tsv` | 레지스트리 |

## 실행

```bash
export SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
ROOT=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace

# 명령만 확인 (GPU 안 씀)
bash $ROOT/scripts/baselines/llama31/tab1_moe_lpr.sh validate

# 학습
TAB1_GPUS=0,1,2,3 bash $ROOT/scripts/baselines/llama31/tab1_moe_lpr.sh train

# 2 GPU: micro 8 유지, accum은 자동으로 4가 되어 global 64 유지
TAB1_GPUS=0,1 bash $ROOT/scripts/baselines/llama31/tab1_ewc.sh train

# 평가 (전 라운드 -> CL matrix, BWT, final avg)
cd $ROOT/implementations/llmcl_benchmark
$ROOT/.venv-runtime/bin/python evaluate_tab1.py --all_rounds \
  --output_dir /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab1/llama31/moe_lpr \
  --base_model_name_or_path $SLORA_LLAMA31_PATH \
  --data_path $ROOT/data/trace \
  --inference_output_path /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab1/llama31/moe_lpr/eval
```

`GRAD_ACCUM`은 기본값이 아니라 global batch에서 **유도**한다. SLoRA 스크립트들은
repo마다 micro/accum을 하드코딩해 두어서 `CUDA_VISIBLE_DEVICES`만 바꾼 런이
조용히 global batch 16으로 학습된 적이 있는데(`slora_pre_upstream`,
`slora_pre_replay`), 여기서는 그 실패가 구조적으로 불가능하다.

## 방법별 결정 사항

### 1. seq_lora
TRACE/SLoRA 세팅 그대로. all7(q,k,v,o,gate,up,down) r64. Table-1 수치는 기존
`full_runs/<model>/seq` 런을 그대로 써도 계약이 같다. 여기 있는 것은 같은
트레이너 위의 대조군.

### 2. EWC — 자체 구현
TRACE `model/Regular/EWC.py`는 세 가지가 논문과 다르다.
1. Fisher를 **학습 루프 안에서** 매 step 누적한다. task 2부터는 loss에 EWC
   penalty가 이미 섞여 있으므로 측정하는 것이 task likelihood의 곡률이 아니다.
2. task 사이 리셋이 없고(`_regular_fisher()`는 주석 처리) 무한 누적된다.
3. 분모가 `len(dataloader)`, 즉 **배치 수**다.

새 구현은 각 task 학습이 끝난 뒤 penalty를 끈 별도 pass에서 diagonal Fisher를
추정하고 샘플 수로 정규화한다. `--ewc_mode online`(누적 Fisher + 최신 anchor,
기본값) / `per_task`(태스크별 penalty 합). `--ewc_fisher_samples 1000`이 기본,
0이면 전체 train split.

LoRA 위치는 seq_lora와 동일한 **all7**.

### 3. O-LoRA — 공식 구현 기준
손실은 `upstream/O-LoRA/src/uie_trainer_lora.py` 그대로:
`loss + λ1·Σ|A_prev A_newᵀ| + λ2·Σ‖θ_new‖₂`. **lora_A**에 대한 **L1**이고 L2는
제곱하지 않은 Frobenius다. TRACE-repro의 `olora_corrected`(B/제곱)는 별개 방법이지
버그 수정이 아니므로 여기 반영하지 않았다.

λ는 공식 `scripts/long.sh`가 위치별로 손튜닝하는데, **앞 8개 구간은 전부
λ1=0.5**이고 λ1=5는 10번째 task부터다. 따라서 8-task 기본값은 λ1=0.5, λ2=0
(task 1은 직교 대상이 없으므로 자동 0). 정확한 위치별 스케줄이 필요하면
`TAB1_OLORA_L1="0,0.5,0.5,0.5,0.5,0.5,0.5,0.5"`처럼 8개를 콤마로 준다.

rank는 **task당 64**. 매 시점 학습 파라미터가 seq_lora와 같아지고, 마지막에
backbone으로 merge하므로 추론 파라미터도 같아진다. 총 누적 rank 512는 merge 뒤
사라진다. `TAB1_OLORA_MERGE=0`으로 merge를 끌 수 있다.

> 참고: 공식 O-LoRA는 `LoraConfig`에 `target_modules`를 주지 않아 PEFT의 Llama
> 기본값인 **q_proj/v_proj**만 쓴다. 계약을 맞추려고 기본값은 all7으로 두었고,
> 공식 그대로가 필요하면 `TAB1_LORA_TARGETS=qv`.

### 4. MTL
8개 train split을 합쳐 셔플해 한 번에 학습. epoch 기본 5 →
40,000 × 5 = 200,000 샘플로, 순차 스케줄(5,000 × (5+3+7+5+3+5+5+7) = 200,000)과
샘플 수가 정확히 같다. 점수뿐 아니라 연산량 기준으로도 상한이 된다.
체크포인트는 마지막 라운드 인덱스에 저장되어 평가 경로가 다른 방법과 같다.

### 5. Lifelong-MoE
ours와 동일한 확장(top-1, task당 expert 1개) + 이전 expert·router row freeze +
출력 KD. teacher는 두 번째 8B를 띄우지 않고 **자기 자신**을 (a) expert prefix를
이전 개수로 제한하고 (b) shared adapter를 task 시작 시점 값으로 되돌려 만든다.

논문은 이전 expert/router row만 freeze하고 나머지를 전부 학습하는데, backbone이
frozen인 우리 세팅에서는 그 "나머지"를 LoRA로 흉내낸다. 기본값은
`--lifelong_shared_targets attn` (attention 4개).

**FFN에는 붙일 수 없다.** `LoRAMoEMLP.add_experts`가
`base_mlp.gate_proj.weight`를 직접 읽어서, gate/up/down을 `SeqLoRALinear`로
감싸면 첫 task의 expert 성장에서 죽는다. `ffn_attn` 스코프는 7개 projection이
전부 routed라 남는 자리가 아예 없다. 이 조합들은 조용히 아무것도 안 붙이는 대신
**설명과 함께 즉시 실패**하도록 해 두었다 (`validate_shared_path`).

**결정 (2026-09-12): backbone은 freeze, shared path는 LoRA로 흉내낸다.**
Lifelong만 8B를 풀면 그 행 하나의 학습 파라미터가 나머지보다 3~4자릿수 크고,
"forgetting이 메커니즘 탓인지 용량 탓인지" 구분이 안 된다.

단, **shared path를 비우면 안 된다.** Lifelong-MoE의 forgetting은 매 task
재학습되는 shared path에서 나온다 — wiki 스윕에서 FM이 .1665(0.25x) →
.2139(4x)로 **증가**하는데, 다른 확장 방법은 전부 평평하거나 감소한다. 이걸
빼면 "grow + freeze + distil"만 남아 ours 수준으로 안 잊고, 더 이상 논문의
방법이 아니다. 그래서 shared adapter가 0개인 Lifelong 런은 명시적 opt-in
(`--lifelong_allow_no_shared_path 1` / `TAB1_LIFELONG_ALLOW_NO_SHARED=1`)
없이는 러너와 트레이너 양쪽에서 거부된다. opt-in해도 그건 ablation 행이지
Lifelong-MoE 행이 아니다.

따라서 **Lifelong-MoE는 ffn 스코프로만 보고**하고, ours-v3와의 파라미터 매칭
슬롯은 `moe_lpr_attn`이 맡는다. MoE-LPR은 원래 shared path가 없으므로
ffn_attn 형태가 완전히 faithful하다.

### wiki(G2) 구현과의 차이 — 논문에 명시할 것

`Megatron-LM/pretrain_gpt_lifelongmoe.py:150-158`의 wiki 구현은 옛 expert/
router row만 freeze하고 **layer 2..9의 실제 shared dense/attention 파라미터를
학습**한다(embedding/output head/final norm/layer 1은 freeze). 정규화는 이전
모델에 대한 KL λ=1.0이고 L2 anchor는 0.

TRACE 구현은 같은 자리에 **LoRA**를 둔다. 메커니즘(계속 학습되는 shared path +
출력 KD)은 같지만 shared path의 용량이 훨씬 작다. 두 스터디의 Lifelong-MoE는
동일 구현이 아니므로 그렇게 주장하지 않는다.

Online L2 anchor(논문 Eq. 5)는 `--lifelong_l2_coeff`로 노출해 두었고 기본 0 —
wiki 포크와 동일하게 KL만 쓴다.

### 6. MoE-LPR
ours FFN 2-phase와 phase 1이 동일하다(새 expert + 새 router row, 이전 row 고정).
phase 2는 router만 열고 replay pool에서 `L = L_LM + γ·L_LPR`.

task당 expert가 1개이므로 논문의 group log-sum-exp는 **단일 expert CE**로
환원된다. task t의 replay 토큰은 expert t로 가도록 cross-entropy를 건다.
`experts_per_task != 1`이면 명시적으로 에러를 낸다 (그 경우 group 형태가 필요).

phase 2에서 aux/z loss는 0. router에게 특정 expert를 선호하라고 가르치는 중에
load-balancing 압력을 거는 것은 목적함수와 정면으로 충돌한다.

router logit은 layout 모듈을 수정하지 않고 forward pre-hook으로 입력을 잡아
같은 GEMM을 다시 계산해 얻는다. FFN layout은 `self.router`를 호출하지 않고
`F.linear(x, router.weight[:active])`를 직접 쓰기 때문에 output hook은 아예
발화하지 않는다.

γ 기본값 0.1은 wiki(G2) 스터디에서 확정된 값. TRACE에서는 재탐색이 필요할 수
있다 (`TAB1_LPR_GAMMA`).

### FFN / FFN+attn 두 버전
`moe_lpr`(ffn) / `moe_lpr_attn`(shared-router QKVO+FFN),
`lifelong_moe` / `lifelong_moe_attn`. FFN-only만 두면 ours와 학습 파라미터 수가
3/7로 달라져서, 표가 "메커니즘 차이"가 아니라 "용량 차이"를 보여주게 된다.

## 테스트

```bash
cd $ROOT/implementations/llmcl_benchmark
CUDA_VISIBLE_DEVICES="" $ROOT/.venv-runtime/bin/python scripts/test_tab1_baselines.py
CUDA_VISIBLE_DEVICES="" $ROOT/.venv-runtime/bin/python scripts/test_tab1_checkpoint_roundtrip.py
```

- `test_tab1_baselines.py` — 실제 tiny LlamaForCausalLM에서 8개 layout의 부착,
  freeze 패턴, forward/backward, O-LoRA merge 정확도, router logit capture,
  router prefix 고정, shared-path 검증
- `test_tab1_checkpoint_roundtrip.py` — 8개 layout 전부 저장→재로드 후 logit
  차이 0. 메타는 트레이너의 `write_meta`가 쓰고 로더가 읽으므로 필드 불일치가
  학습 시작 전에 잡힌다

두 스위트 모두 CPU에서 수 초.

### GPU 스모크 (2026-09-12, llama31, GPU 4-7)

8개 변형 전부 통과. 2 task(C-STANCE, FOMC), 1 epoch, phase당 16 micro-batch,
world=1 / micro 8 / accum 8 = global batch 64.

    /tmp/.../scratchpad/tab1_smoke/run_smoke.sh   (드라이버)
    /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab1_smoke/llama31/

round 0 → round 1 체크포인트 diff로 freeze 패턴을 확인했다.

| 방법 | 변화한 텐서 | 해석 |
|---|---|---|
| seq_lora / ewc | 448/448 | 공유 어댑터 전체 학습 |
| olora | 448/896 | slot 1만 학습, slot 0 동결 |
| lifelong_moe | 256/448 | shared attention LoRA 256개만 변화, task-0 expert 192개 동결 |
| lifelong_moe_attn | 0/448 | shared path 없음 → 옛 expert 전부 동결 (설계대로) |
| moe_lpr | 0/192 | 옛 expert 동결, router만 성장 |
| moe_lpr_attn | 0/448 | 동일 |

### 스모크에서 잡힌 결함 5개

1. **token cache 거부** — `tokenizer_source_fingerprint`를 재구현하면서 파일명
   구분자와 chat-template 기여분을 빠뜨려 지문이 달라졌다. 재구현을 지우고
   `main_Ours_LoRA_MoE`에서 import하도록 바꿨다. 캐시를 Ours 런과 공유하므로
   "같아 보이는 함수"가 아니라 같은 함수여야 한다.
2. **Fisher 추정이 GPU 유휴 상태로 정체** — accumulator를 CPU에 두어 micro-batch
   마다 ~0.7GB를 PCIe로 넘겼다. 디바이스에 두도록 수정. Fisher 값은 동일.
3. **EWC penalty가 매 step 1.3GB 호스트→디바이스 전송** — 11 s/step. 상태를
   가속기에 두어 1.98 it/s로 회복.
4. **EWC penalty가 bf16에서 정확히 0** — Fisher(fp32)를 파라미터 dtype(bf16)으로
   내려 캐스팅했다. 드리프트 ~1e-4, 제곱 ~1e-8, Fisher ~1e-7 → 항이 ~1e-15이고
   168M개를 bf16 8-bit 가수로 합하면 0으로 반올림된다. fp32로 계산하도록 수정하고
   회귀 테스트를 추가했다.
5. **Lifelong KD teacher forward가 gradient checkpointing을 깨뜨림** — teacher를
   student forward와 backward 사이에서 돌리면 non-reentrant checkpoint의
   saved-tensor 장부가 어긋나 `A different number of tensors was saved during
   the original forward and recomputation`로 죽는다. teacher를 student **앞**으로
   옮겼다(`before_forward_fn`).

이 밖에 상속한 Ours base class가 args를 직접 읽는 곳들(`training_version`,
`disable_training_flop_counter` 등)을 채웠다. 감사 스크립트가 처음에 놓친 이유는
같은 이름이 다른 곳에서 `getattr` 기본값으로도 쓰여 "안전" 목록에 들어갔기
때문이다 — 이름 단위가 아니라 접근 지점 단위로 봐야 한다.

## HP 출처

| 방법 | 값 | 출처 |
|---|---|---|
| 공통 계약 | r64/α128/dropout 0.05, lr 2e-4, cosine+warmup 0.03, seed 2025, global 64, all7 | 기존 TRACE seq_lora 런 `slora/instruct_hidden_mse_postkd_20260813/llama31/seq/order1.command.txt` + `SLoRA-repro/src/train/cl_train.py:139`. 정확히 일치 확인 |
| EWC λ | 400 | `upstream/TRACE/model/Regular/EWC.py:13` 하드코딩 기본값. 손실 형태 `0.5*λ*penalty`도 동일 |
| O-LoRA λ1/λ2 | 0.5 / 0 | 공식 `upstream/O-LoRA/scripts/long.sh`. 앞 8개 위치가 전부 λ1=0.5, λ1=5는 10번째부터 |
| O-LoRA 손실식 | \|A_prev A_newᵀ\| L1 + ‖θ‖₂ | `upstream/O-LoRA/src/uie_trainer_lora.py:96` |
| Lifelong KD λ | 1.0 | wiki 포크 `code_from_wiki_lifelongmoe_mha_a100_bf16.sh:39` (논문 Eq. 4) |
| Lifelong L2 λ | 0.0 | 같은 스크립트 42행 |
| MoE-LPR γ | 0.1 | wiki 2-GPU γ 스윕에서 AA 기준 확정 (`run_moelpr_gamma_2gpu.sh`). **TRACE 검증값 아님** |
| MTL epochs | 5 | 논문 값 없음. 40k × 5 = 순차 스케줄의 200k 샘플과 일치하도록 유도 |

**주의 — EWC λ=400은 TRACE의 Fisher 스케일에 붙은 숫자다.** TRACE는 태스크 간
리셋 없이 매 step 누적하고 분모가 배치 수라, 태스크 t 시점에서 대략 5×t배 부풀어
있다. 우리 Fisher는 샘플 정규화된 수렴 후 값이므로 같은 λ라도 정규화 강도가
다르다. penalty 항을 로그로 찍게 해 두었으니(`ewc raw ... term ... (N% of LM)`)
본런에서 그 비율을 보고 스윕 여부를 판단한다. 스모크(태스크 2, 업데이트 3회)에서는
raw 1.6e-07 / term 3.3e-05로 LM 손실의 0.0%인데, 이는 파라미터가 아직 거의 안
움직여서이지 λ가 작다는 증거는 아니다 — 항은 드리프트의 제곱에 비례한다.

## 남은 것

1. HP 탐색: MoE-LPR γ, Lifelong KD 계수, EWC λ. 세 값 모두 wiki 스터디 값을
   기본값으로 넣어두었지 TRACE에서 검증한 값이 아니다
3. GPU 스모크 후 본런
