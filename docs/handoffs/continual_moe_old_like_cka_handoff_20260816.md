# 새 세션 전달 프롬프트 — Continual MoE old-like fingerprint / CKA GT / TRACE

아래 내용을 새 세션의 GPT/Codex에 그대로 전달하라. 이 문서는 2026-08-16 기준 실제 repository, checkpoint, log, 분석 산출물을 조사해 정리한 handoff이다. 경로가 이동되었을 수 있으므로 작업을 시작할 때 모든 경로와 완료 상태를 다시 검증하라. 실행 중인 프로세스를 임의로 종료하거나 기존 결과를 덮어쓰지 말라.

> **2026-08-16 최종 갱신:** 이 문서의 8~11절은 CKA pilot의 역사적 설계와 결과이고, 11.8~11.13절은 그 뒤 완료된 **전체 Code train census, exact targeted pass, 최종 bundle95 lock**의 authoritative 결과다. 따라서 아래 오래된 문장 중 `full GT 미생성`, `threshold 미확정`, `sealed test 공개가 다음 단계`라는 내용은 pilot 당시 상태를 설명할 뿐 현재 상태가 아니다. 현재 최종 상태는 `full census 완료 → exact 95/97/99 생성 → bundle95 immutable lock 완료 → sealed pilot test는 의도적으로 계속 미공개`다.

---

## 0. 작업 환경과 기본 원칙

- workspace: `/home/seonghyeonnoh/yemokoo`
- 주요 repository: `/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning`
- Megatron-LM: `/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/Megatron-LM`
- 실행 스크립트: `scripts/experiment/a100/`
- 분석 스크립트: `scripts/analysis/`
- TRACE: `trace/`
- 주요 결과 root: `/data2/seonghyeonnoh/LLM-continual-learning-runs/`
- 정리된 FlameMoE checkpoint tree: `/home/seonghyeonnoh/yemokoo/data2_llmcl/LLM-continual-learning-runs/flamemoe/`
- GPU 소유권은 실행 직전에 `nvidia-smi`와 process owner로 다시 확인한다. 과거 기본은 0~3이었지만 현재 TRACE 평가 체인은 사용자 지시에 따라 GPU0~5를 사용하고 GPU6,7을 비워 둔다. 다른 사용자의 process는 GPU 번호와 무관하게 건드리지 않는다.
- checkpoint를 임의 기호로 부르지 말고 `Wiki 학습 후`, `expert 확장 KD-init 후`, `Code+Wiki replay 1-Phase 완료 후`처럼 실제 의미로 표현한다.
- fingerprint는 router 입력이나 forced-routing 신호가 아니다. 자연 routing을 유지하고, 어떤 token/representation에 어떤 objective를 적용할지 고르는 selector로만 쓴다.
- hidden 보존 대상은 Transformer layer가 attention/MLP residual add까지 마친 뒤 반환하는 residual-included `layer_output`이다. Layer 2~9를 사용하며 마지막 Layer 9를 반드시 포함한다. Layer 1은 unchanged dense layer라 분석/GT에서 제외한다.
- raw hidden 전체는 저장하지 않는다. paired forward 중 fp32 scalar/CKA metric만 저장한다.
- 기존 결과를 요약할 때 “semantic Wiki ground truth”라고 과장하지 말고, `replay-stable relational anchor` 또는 `old-like stability pseudo-GT`라고 표현한다.

## 1. 연구 목적

최종 목적은 continual MoE에서 실제 old replay data를 계속 저장하지 않고도 old knowledge를 보존할 수 있는가를 검증하는 것이다.

현재 구체적 질문은 다음과 같다.

1. new task 학습 전후에도 old representation의 구조가 유지되는 new-task contextual token occurrence가 존재하는가?
2. 그런 token을 old-like fingerprint/GT로 식별할 수 있는가?
3. 식별된 token의 원래 문맥을 유지한 채 router-only replay objective를 적용하면 실제 old replay를 대체하거나 줄일 수 있는가?
4. 어떤 token을 고를지(selector)와 그 token에서 무엇을 맞출지(KD preservation target)는 별개의 문제인가?
5. raw cosine보다 centered CKA/RSM 기반 구조 지표가 공통 방향·low-norm 오염을 줄이고 더 정밀한 GT를 주는가?

학습 개념은 다음과 같다.

- 확실한 old-like contextual occurrence: teacher 기반 replay objective를 적용한다.
- 나머지 new-task occurrence: new-task LM loss를 적용한다.
- replay branch는 router gradient만 남기고 expert/non-router replay gradient는 제거한다.
- primary new-task branch는 새 expert와 router를 학습한다.
- routing은 강제하지 않고 자연 routing을 유지한다.

주의: old-like occurrence가 위치 `p`의 hidden으로 정의되면 hidden KD는 위치 `p`에 적용된다. LM objective를 같은 위치 mask에 적용할 때는 representation `p`로 label token `p+1`을 예측한다. 이 의미를 기록하고 임의로 mask를 shift하지 않는다.

## 2. 모델 흐름과 authoritative checkpoint

기존 전체 흐름은 대략 다음이었다.

`Wiki 학습 → expert 확장 및 KD-init → Code 1-Phase 학습(+Wiki replay/router FT) → 다시 expert 확장/KD-init → Conversation 1-Phase 학습 → TRACE`

CKA pilot에서 실제로 사용한 before/after checkpoint는 다음과 같다.

### Before: Wiki 학습 후 E8→E16 expert 확장 및 KD-init 완료, step 600

`/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/expansion_kd_init/kd_init/full_training/g2_olddata_kd_9run_20260808__code_e8_to_e16_wiki_kd_step600`

- tracker step: 600
- exact checkpoint content digest: `dd242e9318b54b2768936b5896805f8499e221d6bb7a18303c09d3fa670bddba`

### After: Code LM + Wiki replay LM 1-Phase 완료, step 1800

`/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/no_replay/lm/full_training/flame_code_bootstrap_20260810__g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt-v2`

- tracker step: 1800
- exact checkpoint content digest: `3846c7895857a7c34736ae784af57d7942cdd84d5a946d3cd749102610b8c9f2`
- 디렉터리 분류명이 `no_replay/lm`이지만 실제 run metadata/log에는 Wiki joint replay LM이 켜져 있었다. 폴더 이름만 보고 no-replay run으로 오해하지 말라.

## 3. 왜 layer output인가

기존 KD-init과 old-data hidden KD가 맞추는 값은 residual 포함 `TransformerLayer` 반환값이다.

- hook: `Megatron-LM/pretrain_gpt.py::_capture_transformer_layer_outputs`
- layer 수집: `Megatron-LM/pretrain_gpt.py::_collect_transformer_layers`
- 실제 반환 위치: `Megatron-LM/megatron/core/transformer/transformer_layer.py::forward`
- self-attention residual과 MLP residual add가 끝난 `output`을 반환한다.
- hidden MSE: `Megatron-LM/pretrain_gpt.py::_masked_layer_hidden_mse`
- old-data replay orchestration/gradient restore: `Megatron-LM/megatron/training/training.py`

따라서 input hidden만 비교하면 마지막 MoE layer의 변화가 반영되지 않을 수 있으므로, fingerprint/GT/KD 모두 residual-included layer output L2~L9를 기준으로 한다.

## 4. 초기 stable-subspace 실험에서 배운 점

초기에는 Wiki hidden의 stable direction/subspace를 찾고 Code token을 해당 subspace projection으로 scoring했다.

- stable direction은 Code 학습 전후 Wiki에서 분산은 충분하지만 변화가 작은 방향이었다.
- 이 방향을 이용해 Code token 중 Wiki-like token을 고르는 selector에는 작은 신호가 있었다.
- 그러나 stable direction 자체가 보존해야 할 KD direction이라는 증거는 없었다.
- 같은 stable selector로 token을 고른 뒤 random-r64 방향에서 KD한 경우가 stable-r64보다 좋기도 했다.
- 따라서 `token selection`과 `KD preservation direction`은 분리해야 한다.
- PCA top-32/64나 고정 계수는 근거가 부족했고, “안 변한다”와 “성능에 필수다”는 같은 주장이 아니다.

핵심 결론:

> 변하지 않는 방향은 selector 신호일 수 있지만, 그 방향 자체가 반드시 보존 대상이라는 뜻은 아니다.

200-step 수치:

- LM only: Wiki 0.440876, Code 0.597439
- stable selector + stable-r64 KD: Wiki 0.449930
- stable selector + random-r64 KD: Wiki 0.451339
- stable selector + hard stable-r64 KD: Wiki 0.450045
- 동일 random-r64 basis/weight에서 stable token assignment: Wiki 0.454526, Code 0.589566
- within-batch permuted assignment: Wiki 0.450024, Code 0.589091
- stable assignment 이득: Wiki +0.004502

근거:

- `/data2/seonghyeonnoh/LLM-continual-learning-runs/layer_output_fingerprint_kd_20260810/report/PHASE4_200STEP_ROOT_CAUSE.md`
- 같은 root의 `SELECTOR_ASSIGNMENT_200STEP_AUDIT.md`, `FULL_HIDDEN_200STEP_AUDIT.md`

## 5. 기존 cosine 기반 old-like GT

### 5.1 paired full-Code extraction

동일한 Code contextual occurrence를 before/after checkpoint에 forward하여 L1~L9 scalar metric을 저장했다. GT에서는 L2~L9만 사용했다.

- Code sequence: 4,147,200
- sequence length: 512
- contextual token occurrence: 2,123,366,400
- residual-included layer outputs
- 저장 metric: cosine, relative L2, symmetric relative L2, log norm ratio, delta MSE, feature-centered cosine 등
- authoritative feasibility report: `scripts/analysis/reports/code_old_like_decision_report_20260811.md`
- 원래 Code raw paired metric root와 Conversation raw paired metric root는 GT 생성 후 용량 절감을 위해 삭제했다. 현재는 packed GT, downstream log/checkpoint, repo의 feasibility report를 우선 사용한다. 오래된 report에 남은 raw-root 경로가 실제로 존재한다고 가정하지 말라.

### 5.2 cosine 분석 결과

- Code L2~L9 mean cosine은 depth에 따라 크게 하락했지만 Wiki는 거의 그대로였다.
- Wiki raw cosine mean L2~L9: 대략 0.99884, 0.99849, 0.99804, 0.99784, 0.99739, 0.99695, 0.99660, 0.99633.
- Wiki L2~L9 raw cosine mean: 약 0.9976.
- raw cosine mean 기반 Wiki-vs-Code AUROC: 약 0.99937. 단, Code 안에 genuine anchor가 있을 수 있으므로 semantic binary-classification AUROC로 과장하면 안 된다.
- Code layer top-1% pairwise mean Jaccard: 0.4693.
- Code all-8 layer top-1%는 histogram-midrank 기준 0.1522%였으나 tie가 포함된 값이다.
- 실제 raw fp32 threshold를 `>=`로 적용한 exact GT는 2,658,787 / 2,123,366,400 = 0.125215648%다.
- single-layer high cosine은 충분하지 않다. L9 top-1%의 relative-L2 median은 약 0.405이며, low-reference-norm(bottom 5%) 오염이 30.1%였다.
- all-8 반복 선택은 더 안정적이었지만 Wiki all-8보다 relative L2가 여전히 컸다.
- raw cosine은 방향만 보므로 크기 변화는 relative L2와 log norm ratio로 따로 봐야 한다.

### 5.3 exact cosine GT 정의

GT root:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/code_old_like_gt_l2_l9_top1_raw_20260812`

정의:

- 각 layer L2~L9에서 Code calibration/full-census top-1% raw cosine cut을 만든다.
- 동일 occurrence가 8개 layer cut을 모두 `>=`로 통과하면 positive다.
- cut:
  - L2 0.9998499751
  - L3 0.9901499748
  - L4 0.9566500187
  - L5 0.9395499825
  - L6 0.9184499979
  - L7 0.8883500099
  - L8 0.8607500196
  - L9 0.8523499966
- exact selected count: 2,658,787
- exact selected fraction: 0.125215648%
- semantic claim: `Wiki-like stability pseudo-GT`, semantic Wiki ground truth가 아니다.

이 top-1% 선택은 임의의 “1%가 좋아 보여서”만은 아니다.

1. layer별 raw scale 차이가 있어 single-layer 절대 cut은 불안정했다.
2. 여러 layer에서 반복되는 high-stability occurrence가 독립 우연보다 훨씬 많았다.
3. Code top-1% cut을 실제 Wiki에 적용하면 Wiki가 강하게 통과했다.
4. all-8 교집합이 약 0.1%여서 기존 replay storage 규모와도 비슷한 고순도 operating point가 됐다.
5. 다만 dead/low-norm과 magnitude 문제를 최종적으로 해결한 selector는 아니었다.

## 6. 기존 cosine GT 기반 replay 학습에서 생긴 구현 교훈

처음에는 GT token을 포함한 sequence 전체를 replay subset처럼 취급하거나 GT token을 문맥 없이 이어붙인 miniset을 만들었다. 두 방식 모두 주의가 필요했다.

- sequence 포함만으로 subset을 만들면 실제 GT가 아닌 주변 token도 replay objective에 들어갈 수 있다.
- GT token만 이어붙이면 원래 attention context가 깨져 teacher/student hidden KD의 의미가 바뀐다.
- 최종적으로는 **GT occurrence가 있는 원래 512-token 문맥을 forward하고, loss mask는 GT 위치에만 적용**하는 contextual-occurrence 방식이 맞다.
- replay dataset/iterator는 primary new-task dataset과 분리한다.
- non-GT positions는 context-only이며 loss를 받지 않는다.
- replay exposure는 선택 token을 무작정 1회만 보는 것이 아니라, 기존 1-Phase 방식처럼 전체 train-token budget의 20%에 맞게 반복한다.
- Code/Conversation run manifest에서 `replay_input_tokens=424,673,280`, `replay_samples=829,440`으로 20% input budget을 맞췄다.
- primary와 replay gradient는 같은 optimizer step에서 accumulate한 뒤 한 번 update한다.
- replay pass 뒤 expert/non-router gradient를 복원/제거하여 router gradient만 남긴다.

여기서 `20% input budget`과 `GT-supervised occurrence exposure`를 혼동하면 안 된다.

- Code token-only miniset은 512칸을 모두 GT token으로 채웠기 때문에 424.7M supervised position / 2.659M unique GT ≈ 159.7 token epochs였다. 그러나 문맥과 원래 next-token target을 파괴했기 때문에 이 결과, 특히 LM의 Wiki score 약 0.3003은 올바른 contextual replay 결과로 해석할 수 없다.
- Conversation exact-axis contextual run은 replay item마다 원래 512-token 문맥을 넣고 GT anchor 1개만 supervise했다. 829,440 replay item / 1,311,043 GT ≈ 0.633 GT-occurrence epoch였다.
- 두 방식은 replay **input** 양은 20%로 같지만 supervised GT exposure는 약 253배 차이 난다. 결과를 직접 비교하면 안 된다.
- selected-token 평균 loss이므로 coverage 0.1%라고 한 microbatch의 replay gradient가 자동으로 0.001배가 되는 것은 아니다. 하지만 unique anchor coverage와 반복 횟수 차이는 그대로 남는다.

관련 구현:

- GT dataset wrapper: `Megatron-LM/megatron/core/datasets/old_like_gt_dataset.py`
- objective mask: `Megatron-LM/pretrain_gpt.py::_apply_old_like_gt_objective_mask`
- replay budget: `Megatron-LM/megatron/training/training.py::_moe_joint_replay_old_like_budget`
- paired iterator identity validation: `Megatron-LM/megatron/training/training.py::_MoeJointReplayDataIterator`
- Code GT/miniset builders: `scripts/analysis/build_old_like_gt_*.py`
- 4-objective launcher: `scripts/experiment/a100/run_g2_old_like_token_miniset_1phase_4objective_chain_mha.sh`

비교한 replay objective는 다음과 같다.

1. selected-occurrence LM
2. residual-included layer-output hidden MSE, L2~L9
3. hidden KL
4. output-vocabulary KL

대표 정리된 checkpoint tree:

- Code hidden MSE: `.../flamemoe/ffn_experts_only/code/replay/hidden_mse/full_training/`
- Code hidden KL: `.../flamemoe/ffn_experts_only/code/replay/hidden_kl/full_training/`
- Code vocab KL: `.../flamemoe/ffn_experts_only/code/replay/vocab_kl/full_training/`
- Code LM: `.../flamemoe/ffn_experts_only/code/replay/lm/full_training/`
- Conversation counterparts: `.../flamemoe/ffn_experts_only/conversation/replay/{hidden_mse,hidden_kl,vocab_kl,lm}/full_training/`

이 실험들에서 cosine selector로 고른 token replay는 old forgetting을 일부 줄일 수 있었지만 full old-data replay의 완전한 회복을 보장하지 않았다. Conversation에서는 Code보다 forgetting과 new-task acquisition 문제가 더 컸고, 단순 cosine GT가 충분히 “old-like”인지 의문이 남았다. 이것이 CKA 전환의 직접적인 이유다.

Code 1800-step 대표 결과:

| 조건 | Code | Wiki |
|---|---:|---:|
| Code only, no replay | 0.666697 | 0.410556 |
| 실제 Wiki replay LM oracle | 0.663842 | 0.460038 |
| 실제 Wiki replay hidden MSE c10 | 0.664464 | 0.457552 |
| 실제 Wiki replay hidden KL | 0.664877 | 0.457314 |
| 실제 Wiki replay vocab KL | 0.660796 | 0.457916 |
| cosine-GT token-only hidden MSE c10 | 0.664963 | 0.454481 |
| cosine-GT token-only hidden KL c1 | 0.666617 | 0.440760 |
| cosine-GT token-only vocab KL c1 | 0.664734 | 0.455324 |
| cosine-GT token-only LM | 0.662725 | 0.300334 |

마지막 LM 결과는 selector 실패가 아니라 synthetic next-token target 때문에 invalid하다. MSE/vocab-KL은 일부 보존을 보였지만 약 160 selected-token epoch의 synthetic-context 실험이므로 “0.125%만으로 old replay를 대체했다”고 결론내리면 안 된다. cosine 8/8 GT의 matched-count contextual random control도 아직 없다.

## 7. CKA로 전환한 이유

raw cosine의 문제:

- hidden의 공통 평균 방향/anisotropy에 의해 cosine이 높아질 수 있다.
- 크기 변화는 무시한다.
- 각 token의 절대 방향만 보며, 주변 token과의 관계 구조가 유지됐는지는 보지 못한다.

CKA를 사용하는 이유:

- centered representation으로 공통 평균 성분을 제거한다.
- chunk 내부 token-token relational structure를 비교한다.
- linear CKA는 p>n(예: token 128, hidden 1024)에서도 CCA류처럼 degenerate하지 않는다.
- eigenvalue/variance가 큰 지배적 구조에 더 민감하다.
- token 단위 GT는 CKA의 RSM contribution으로 다시 내린다.

단, CKA는 scale-invariant이므로 magnitude 보존을 대신할 수 없다. 따라서 rel-L2와 `|log(norm ratio)|` 조건을 별도로 유지한다.

## 8. CKA pilot 설계

### 8.1 입력 단위

- document 경계를 넘지 않는 512-token window를 사용한다.
- full window 512와 tail 256~511을 허용한다.
- right-aligned chunk를 추가하지 않는다.
- chunk scale/stride:
  - scale 128, stride 64
  - scale 256, stride 128
  - scale 512는 whole-window diagnostic control만 사용하고 GT threshold에는 쓰지 않는다.
- fixed grid로 덮이지 않는 suffix token은 ineligible이다.
- document를 calibration/selection/test = 40/30/30으로 분할하고 document overlap을 금지한다.
- seed 1234를 고정한다.

Pilot는 Code/Wiki 각 100,000 window를 uniform sample했다.

**pilot 당시 중요했던 구분:** pilot launcher의 `full`은 100K Code + 100K Wiki를 뜻했으며 당시에는 전 corpus GT가 아니었다. 이후 별도의 `cka_gt_full_census_20260816` root에서 Code train 전수를 처리하고 exact GT를 생성했다. pilot root와 full-census root를 혼동하지 말라.

### 8.2 CKA 수식

한 layer의 before/after chunk hidden을 `X,Y ∈ R^(n×p)`라 하자. token 축 평균을 제거한다.

```text
Xc = X - mean_token(X)
Yc = Y - mean_token(Y)
K = Xc Xc^T
L = Yc Yc^T
CKA = <K,L>_F / (||K||_F ||L||_F)
```

token i의 CKA contribution:

```text
c_i = sum_j K_ij L_ij / (||K||_F ||L||_F)
sum_i c_i = CKA
```

자기 대각 성분을 제외한 contribution:

```text
c_i_off = sum_(j != i) K_ij L_ij / (||K||_F ||L||_F)
CKA_off = sum_i c_i_off
s_i = c_i_off / (CKA_off / n)
```

- `s_i`는 해당 token이 chunk CKA 구조 보존에 평균 대비 얼마나 기여하는지 나타낸다.
- 음수 `s_i`는 T 조건에서 fail 처리한다.
- overlapping chunk에서는 token별 `CKA_min`, `s_min`을 scale×layer별로 사용한다.
- `r_i`: K와 L의 off-diagonal row Pearson correlation이며 GT에는 직접 쓰지 않고 `r_min/r_mean/r_max` 진단으로 저장한다.
- `diag_ratio`, `CKA_off`, `offdiag_warning`도 진단으로 저장한다.

Magnitude metric:

```text
relative_L2 = ||Y-X|| / (||X|| + eps)
log_r = log((||Y||+eps)/(||X||+eps))
```

### 8.3 candidate selector

Wiki calibration 분포로만 threshold를 만든다.

- bundle 95:
  - B=Wiki CKA lower 5% quantile
  - T=Wiki raw pre-aggregation s_i lower 5% quantile
  - rel-L2=Wiki upper 95% quantile
  - |log-r|=Wiki upper 95% quantile
- bundle 97: lower 3% / upper 97%
- bundle 99: lower 1% / upper 99%

Condition-specific consensus가 primary다.

```text
B: 각 scale(128,256)에서 layer 7/8 통과
T: 각 scale(128,256)에서 layer 7/8 통과
M-L2: layer 7/8 통과
M-log-r: layer 7/8 통과
최종: B128 AND B256 AND T128 AND T256 AND L2 AND log-r
```

- valid layer가 정확히 6개면 6/6 통과를 요구한다.
- same-layer 방식, 즉 같은 layer에서 모든 조건을 동시에 만족하는 count는 diagnostic으로만 계산한다.
- condition-specific과 same-layer selection Jaccard가 0.9 미만이면 human review한다.
- membership(Mahalanobis/K=64 prototype)은 현재 purity diagnostic일 뿐 GT condition으로 승격하지 않았다.
- membership 통계는 before-checkpoint Wiki calibration에서 만들었다: mean은 19,009,359 valid token 전체, Ledoit-Wolf covariance는 layer당 uniform 2M subsample, K=64 prototype은 layer당 200K reservoir를 사용했다. Mahalanobis는 square-root whitened distance다.
- test split은 raw metric을 물리적으로 `sealed_test/raw/`에 분리 저장했고 REPORT v1에서는 열지 않았다.

## 9. CKA 구현과 검증

주요 파일:

- numerical core: `scripts/analysis/cka_gt_pilot_core.py`
- document-window preparation: `scripts/analysis/cka_gt_pilot_windows.py`
- runtime/storage: `scripts/analysis/cka_gt_pilot_runtime.py`
- model runner: `scripts/analysis/run_cka_gt_pilot_mha.sh`
- 4-GPU chain: `scripts/analysis/launch_cka_gt_pilot_4gpu.sh`
- analyzer: `scripts/analysis/analyze_cka_gt_pilot.py`
- validator: `scripts/analysis/validate_cka_gt_pilot.py`
- tests: `scripts/analysis/test_cka_gt_pilot_*.py`, `scripts/analysis/test_analyze_cka_gt_pilot.py`, `scripts/analysis/test_validate_cka_gt_pilot.py`

검증 사항:

- `sum(c_i)=CKA` fp32 검산
- Y=X이면 CKA=1
- Y=XQ, Q orthogonal이면 CKA=1
- Y=2X이면 CKA=1이지만 rel-L2=1, log-r=log2
- permutation/random-pair null은 실제 pair보다 낮음
- document split overlap=0
- exact dataset `.idx/.bin` SHA, exact checkpoint iteration file SHA를 config/runtime에 bind
- raw hidden 미저장
- open calibration/selection과 sealed test를 물리적으로 분리
- resume-safe atomic Parquet/journal
- router L2~L9와 natural routing 확인
- final validation: `ok=true`, errors=[], warnings=[]

전체 CPU regression은 마지막 audit 기준 49 tests PASS였고, production pre-analysis deep validation과 final validation이 통과했다.

주의: 이 CKA 구현/테스트 스크립트들은 마지막 확인 당시 Git에서 `??` untracked 상태였다. cleanup 전에 반드시 보존하고 의도적으로 commit할지 결정한다.

## 10. CKA pilot authoritative output

Root:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1`

우선 읽을 파일:

- report: `REPORT.md`
- candidate table: `threshold_tables/candidate_bundles.csv`
- same-layer diagnostic: `threshold_tables/same_layer_diagnostic.csv`
- selector overlap: `selector_comparison/pairwise_jaccard.csv`
- exclusive profiles: `selector_comparison/exclusive_set_profiles.json`
- histogram: `histograms/chunk_cka_{128,256,512}_{2..9}.{svg,csv}`
- null summary: `histograms/chunk_cka_null_summary.csv`
- raw s_i safety: `histograms/raw_s_i_summary.csv`
- context candidates: `context_samples/`
- binding: `analysis_input_binding.json`
- final validator: `validation/08_final_after_analysis.json`

전체 artifact는 약 154GB다: open token metrics 약 74GB, open chunk metrics 약 34GB, sealed test 약 46GB. raw hidden은 없다.

Input scale:

- Code source documents: 703,256
- Wiki source documents: 2,922,937
- sampled windows: Code 100K, Wiki 100K
- calibration/selection/test: 40K/30K/30K each domain
- Code discarded tail fraction: 1.88985%
- Wiki discarded tail fraction: 10.8865%
- Code suffix-ineligible fraction: 약 1.16%
- Wiki suffix-ineligible fraction: 약 3.47%

## 11. CKA 핵심 결과

### 11.1 real CKA와 null

- Wiki real CKA는 L2~L9에서 대체로 약 0.999로 매우 높다.
- Code real CKA median은 scale128에서 약 0.94→0.71, scale256에서 약 0.94→0.68로 depth가 깊어질수록 낮아진다.
- scale512 diagnostic은 대략 0.93→0.65 경향이다.
- 128/256의 모든 B95/B97/B99 threshold가 permutation/random-pair null p95보다 높다.
- invalid chunk fraction=0, null missing count=0.

이는 CKA가 finite-sample null floor만 보고 높은 값을 내는 것은 아니며, 실제 Wiki relational stability와 Code drift를 구분하는 강한 신호임을 뜻한다.

### 11.2 candidate selection 결과 — selection split만

| bundle | Wiki combined recall | Code coverage | Code selected token |
|---|---:|---:|---:|
| 95 | 55.9703% | 0.00411584% | 618 |
| 97 | 70.9156% | 0.00648677% | 974 |
| 99 | 87.6554% | 0.00955034% | 1,434 |

해석:

- 세 selector 모두 기존 cosine all-layer top-1% 약 0.1%보다 훨씬 적은, 극단적으로 precision-first한 subset이다.
- `equal_prior_operational_precision`은 0.99989~0.99993이지만 이는 Code를 완전 negative로 가정한 source-domain operational precision일 뿐 semantic precision이 아니다.
- 가장 큰 bottleneck은 B(chunk CKA) 조건이다.
- bundle95 Code 독립 pass fraction: B 0.007246%, T 73.9241%, rel-L2 0.073146%, |log-r| 0.342633%.
- bundle95에서 CKA-only 877개 중 magnitude 조건이 259개(29.5%)를 제거하여 최종 618개가 됐다.

### 11.3 condition-specific vs same-layer

- Code bundle95: 618 vs 517, Jaccard 0.83657 → human review 필요
- Code bundle97: 974 vs 954, Jaccard 0.97947
- Code bundle99: 1434 vs 1430, Jaccard 0.99721

따라서 95는 가장 순도 지향적이지만 101개의 condition-specific-only occurrence를 반드시 육안/metric으로 검토해야 한다. 97은 더 안정적인 fallback이다.

### 11.4 legacy cosine과의 관계

selection split에서:

- legacy cosine: 16,409
- CKA-only95: 877
- CKA+M95: 618
- legacy ∩ CKA+M95: 200
- Jaccard: 0.0118857

즉 CKA selector는 단순히 cosine selector의 더 빡센 threshold가 아니다. 거의 다른 subset을 고른다. 이 때문에 다음 학습에는 legacy cosine과 matched-count random control이 반드시 필요하다.

### 11.5 routing face-validity

CKA+M95 selected 618 token ×8 layer=4,944 layer-occurrence에서:

- top-4 expert overlap=4: 4,284 (86.65%)
- top-4 overlap=3: 660 (13.35%)
- overlap 0~2: 0
- selected old full-mass delta mean: -0.01774
- selected top4 old-mass delta mean: -0.00212

nonselected:

- exact top-4 overlap=4 비율은 약 2.12%
- old selected-mass delta mean은 약 -0.568

따라서 selected occurrence는 routing 측면에서도 매우 안정적인 anchor라는 face-validity가 있다. 하지만 routing stability가 old knowledge의 causal importance를 증명하는 것은 아니다.

### 11.6 s_i 안전성

- finite raw s_i 중 negative fraction: 약 0.0866%
- `abs(s_i)>100`: 56,034
- `abs(s_i)>1000`: 0
- 이 중 93% 이상은 diagnostic-only scale512에 몰려 있다.
- selector scale에서는 scale128 5개, scale256 3,503개 정도의 `>100` 사례가 있다.
- offdiag warning incidence=0.

s_i explosion은 현재 diagnostic warning이며 자동 exclusion은 아니다. 최종 selector를 고르기 전에 해당 group과 context를 확인해야 한다.

### 11.7 가장 중요한 추가 감사: window/document concentration

REPORT 밖에서 open selection assignment를 read-only로 다시 집계한 결과다. sealed test는 열지 않았다.

- bundle95 618개는 정확히 **4 window / 4 document**에서만 나왔다.
  - window별 371, 105, 76, 66개
  - top window 하나가 60.03%를 차지
- bundle97 974개도 정확히 4 window/4 document다.
  - 414, 350, 116, 94개
  - top window share 42.51%
- bundle99 1,434개는 6 window/6 document이며 상위 4 window가 95.47%를 차지한다.

direct tokenizer decode로 bundle95의 4개 window를 확인한 결과:

1. 색상 약어 permutation을 반복한 list: 512 token 중 unique token ID가 9개뿐이며 618개 중 371개를 공급
2. Italian word list
3. English insult multiline string/docstring
4. GPL/Project Euler 설명 prose + numeric grid

bundle99는 긴 numeric list와 반복적인 `1552: None` mapping도 추가로 받아들인다.

해석:

- 일부는 자연어/license/problem description이라 old-like한 pocket으로 볼 수 있다.
- 그러나 반복 list/table/mapping의 relational pattern이 CKA를 높인 구조적 artifact도 매우 강하다.
- 현재 selector는 token-level precision이 높아 보이지만 실제 독립 context diversity가 거의 없다.
- 따라서 지금 단계에서 `Wiki-like GT` 또는 generic old-like GT로 승격하면 안 된다.
- 가장 안전한 표현은 계속 `replay-stable relational anchor candidate`다.

Membership diagnostic도 이 문제를 해결하지 못했다.

- CKA+M-only와 legacy-only의 mean Mahalanobis median은 약 35.32 vs 35.41로 거의 같다.
- early layer에서는 selected Code가 Wiki에 더 가깝다고 보기 어려웠다.
- membership을 GT AND 조건으로 넣기 전에 AUROC/PR 또는 density-ratio/domain probe 검증이 필요하다.

### 11.8 전체 Code train CKA census — 완료

Pilot의 window 집중 현상이 작은 표본 때문인지 확인하고 full-train 분포를 얻기 위해 별도의 threshold-free census를 수행했다. 이 단계에서는 threshold를 적용해 GT를 만들지 않고, 전체 분포와 후속 exact pass에 필요한 최소 metric만 저장했다.

Authoritative root:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1`

Merged census root:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1/full_census`

정확한 처리량:

- Code source document: 703,256
- document-bounded window: 4,161,493
- retained input token: 2,083,238,288
- selector-eligible token: 2,058,949,248
- worker: 4개 독립 GPU worker, 각 약 1.040M window
- model forward: BF16
- CKA/B/T/M metric accumulation: FP32, TF32 disabled
- layer: residual-included L2~L9
- selector scale: 128/stride64, 256/stride128
- raw hidden 저장: false
- sealed pilot test 접근: false
- merged summary `complete=true`

Full census는 저장량과 속도를 위해 pilot의 wide Parquet 경로를 반복하지 않았다.

- 저장: 4096-bin marginal histogram, exact chunk-B shard, deterministic 5M token reservoir
- 제외: raw hidden, router diagnostic, membership, scale512/null/r_i wide table
- exact chunk-B: 224 shard, 3,385,315,439 bytes
- 5M reservoir는 full joint B/T/M coverage와 diversity의 threshold 후보를 추정하는 용도였다.
- threshold는 이 단계에서 refit하지 않았다. Wiki pilot calibration에서 만든 95/97/99 cut을 그대로 overlay했다.

Provenance:

- census manifest content identity: `381bb7ef0d34e9adfafb69d8faccff5351da44f4cfeb2b375f1ab7401690536d`
- merged summary SHA256: `217477314244856def000ea517388aff8b7379eedfaaf5eedea3a347a153f342`
- four-worker model-binding set SHA256: `7f55b40c804bd3dd41546a8ca002471390a3fd0971e4ba31bf93b9b56ea84962`
- Code `.bin` SHA256: `9bb78940d9683901afbbed35a377cf8724e180c46ece878d10c701060613f72d`
- Code `.idx` SHA256: `63a1e6b8090bc8167682bd204d6fe0b2e5640a61f44e28d46178ec7f86f6145b`
- before checkpoint identity: `dd242e9318b54b2768936b5896805f8499e221d6bb7a18303c09d3fa670bddba`
- after checkpoint identity: `3846c7895857a7c34736ae784af57d7942cdd84d5a946d3cd749102610b8c9f2`

중요한 구현 교훈: worker1이 batch192 장시간 실행 후 vocab-logit 출력의 큰 allocation과 fragmentation으로 마지막 약 10만 window 지점에서 OOM이 났다. committed journal은 손상되지 않았고 재개하여 전체 census를 완료했다. CKA 자체가 18GB를 요청한 것이 아니라, layer hook만 필요한 census에서도 사용하지 않는 `[B,S,V]` logits를 생성하던 경로가 원인이었다. 향후 동일 extraction에서는 logits head를 건너뛰거나 forward subbatch를 사용한다.

### 11.9 Full-census reservoir threshold review

분석 report:

`full_census/threshold_review_v1/REPORT.md`

이 report는 5M uniform token reservoir로 joint selector coverage와 diversity를 추정했으며, exact GT는 아직 만들지 않는 단계였다.

| bundle | reservoir selected | coverage | estimated full count | document-bootstrap CI95 | same-layer Jaccard |
|---:|---:|---:|---:|---:|---:|
| 95 | 327 | 0.00654% | 134,655 | 61,946–213,308 | 0.957187 |
| 97 | 509 | 0.01018% | 209,601 | 109,147–331,240 | 0.954813 |
| 99 | 1,080 | 0.02160% | 444,733 | 248,101–664,401 | 0.978704 |

Reservoir 축의 조건별 통과율:

| bundle | B | T | M | B+T | B+M | B+T+M |
|---:|---:|---:|---:|---:|---:|---:|
| 95 | 0.01082% | 74.8293% | 0.07108% | 0.00906% | 0.00778% | 0.00654% |
| 97 | 0.01536% | 80.0153% | 0.09210% | 0.01328% | 0.01170% | 0.01018% |
| 99 | 0.02828% | 87.7663% | 0.17036% | 0.02590% | 0.02354% | 0.02160% |

여기서도 B가 가장 큰 bottleneck이며 T는 상대적으로 느슨하다. `M`은 rel-L2와 `|log-r|`의 AND다.

Pilot의 4-window concentration은 full reservoir에서는 완화됐다.

| bundle | active windows | active documents | max-window share | max-document share |
|---:|---:|---:|---:|---:|
| 95 | 245 | 48 | 1.5291% | 20.7951% |
| 97 | 366 | 72 | 0.9823% | 15.5206% |
| 99 | 735 | 144 | 0.4630% | 16.2963% |

따라서 “CKA가 오직 4개의 반복 window만 잡는다”는 pilot 관찰을 full corpus에 그대로 일반화하면 안 된다. 다만 selected document 수는 여전히 작고 특정 document share가 크므로 구조적/list/table/repetition bias가 완전히 사라졌다고도 말할 수 없다.

### 11.10 B-candidate extraction과 targeted second pass

전체 20.59억 eligible token에 대해 T/M까지 모두 다시 GPU forward하는 대신 두 단계로 나눴다.

1. full census에 이미 저장된 exact chunk-B로 bundle99 B 조건을 만족하는 candidate window union을 만든다.
2. 그 candidate window에만 before/after forward를 다시 수행하여 token-level T, rel-L2, `|log-r|`를 exact하게 계산한다.

Authoritative candidate root:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1/gt_candidates_b99_v2`

- candidate window: 2,134
- candidate eligible token axis: 1,082,624
- exact B95 token: 226,048
- exact B97 token: 320,128
- exact B99 token: 594,176
- B95 subset B97 subset B99: true
- candidate manifest content SHA256: `6f411cc70aaadd26145c7a6b020dad29a83204fb5f07ff79217fffbb97ad0388`
- authoritative B mask SHA256: `b0f8030b1a4f0a6f4153bae9a8e1a2d3915aead6de4ab730fbb6bcd886b79cd5`

Targeted output:

`full_census/exact_targeted_v2`

- GPU forward batch: logical 192, forward subbatch 128
- exact shards: 7, SHA 검증 완료
- selected source window: 2,083개를 no-padding으로 재검증
- source IndexedDataset token ID comparison: 771,929 occurrence
- packed mask와 occurrence list bijection: pass
- bundle nesting: 95 subset 97 subset 99, pass
- raw hidden: 저장하지 않음
- sealed pilot test: 열지 않음

Targeted GPU에서 recompute한 B count는 batch-shape 수치 차이로 authoritative full-census B보다 각각 -128/-832/-960이었다. 최종 판정은 전체 census에서 이미 계산·해시 고정된 authoritative B mask를 사용했고, targeted pass는 T/M을 계산했다. 이 차이를 숨기지 않고 `b_count_validation.json`에 기록했으며 authoritative counts와 candidate manifest의 exact equality를 검증했다.

### 11.11 Exact 95/97/99 결과

Exact review:

`full_census/exact_targeted_v2/exact_review_v1/REPORT.md`

| bundle | exact GT | full eligible coverage | windows | documents | same-layer Jaccard |
|---:|---:|---:|---:|---:|---:|
| 95 | 130,289 | 0.006328% | 832 | 107 | 0.939174 |
| 97 | 201,825 | 0.009802% | 1,092 | 160 | 0.957612 |
| 99 | 439,815 | 0.021361% | 2,083 | 372 | 0.972618 |

Reservoir estimate는 세 bundle 모두 exact count를 CI 안에 포함했고, exact count가 최종 authoritative 수치다.

Exact concentration:

| bundle | max-window share | top-5 window share | window HHI | max-doc share | top-5 doc share | doc HHI | token entropy | top-token share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 95 | 0.3761% | 1.7561% | 0.00186776 | 23.7518% | 62.8472% | 0.102294 | 9.0747 bits | 9.7100% |
| 97 | 0.2472% | 1.2144% | 0.00136395 | 17.9735% | 55.8068% | 0.076094 | 9.5269 bits | 9.8021% |
| 99 | 0.1162% | 0.5800% | 0.00074545 | 15.1693% | 43.5190% | 0.054157 | 9.8432 bits | 10.7447% |

이 결과의 올바른 해석:

- Pilot의 “4 window가 거의 전부” 문제는 full census에서는 명백히 완화됐다.
- 그러나 bundle95는 107 document뿐이고 상위 5 document가 62.85%를 차지한다. 완전히 일반적인 semantic old-domain token 집합은 아니다.
- token entropy는 pilot보다 높지만 반복/list/table/prose/numeric structure 편향은 남아 있다.
- 따라서 label은 `Wiki semantic GT`가 아니라 `replay-stable relational anchor pseudo-GT`다.
- CKA는 반복 그 자체만 보는 것이 아니라 before/after에서 centered token-token relational geometry가 보존되는지를 본다. 반복 구조가 높은 CKA를 만들기 쉬운 failure mode인 것은 맞지만, magnitude 조건과 token contribution 조건을 모두 통과해야 최종 GT가 된다.

### 11.12 최종 bundle95 GT lock — 완료

최종 threshold는 precision-first 원칙으로 가장 엄격한 Wiki-calibrated `bundle95`를 선택해 immutable create-only lock했다.

Lock root:

`full_census/gt_locked_bundle_95_v1`

Decision record:

`full_census/gt_locked_bundle_95_v1/threshold_lock.json`

최종 selector:

```text
B128: layer L2~L9 중 7/8 이상 Wiki-95 lower cut 통과
B256: layer L2~L9 중 7/8 이상 Wiki-95 lower cut 통과
T128: layer L2~L9 중 7/8 이상 Wiki-95 lower cut 통과
T256: layer L2~L9 중 7/8 이상 Wiki-95 lower cut 통과
rel-L2: layer L2~L9 중 7/8 이상 Wiki-95 upper cut 이하
|log-r|: layer L2~L9 중 7/8 이상 Wiki-95 upper cut 이하
final GT = B128 AND B256 AND T128 AND T256 AND rel-L2 AND |log-r|
```

- valid layer가 정확히 6개면 6/6을 요구한다.
- 조건별 consensus가 primary이며 same-layer rule은 diagnostic이다.
- 최종 GT: 130,289 contextual occurrence
- denominator: 2,058,949,248 selector-eligible Code train token
- coverage: 0.006327936% (`0.006328%`)
- windows: 832
- documents: 107
- same-layer count: 122,364
- same-layer Jaccard: 0.939174, 0.9 이상이라 manual-review gate 통과
- packed-mask popcount = occurrence count, exact match
- sealed pilot test opened: false

최종 bundle95 artifact:

- occurrence list: `full_census/exact_targeted_v2/bundle_95_occurrences.npy`
  - SHA256 `a2eb014878397c5b91b10ebbaf6126c40c7c1c8fcc2c3d5224d3aded7fbfdbe3`
- packed mask: `full_census/exact_targeted_v2/bundle_95_packed.npy`
  - SHA256 `3534449c0bbec8c25f53c63917f1df09b97fbc78a41186fccd48d2dcea9185c4`
- exact summary SHA256: `9e3d9b3464f47680f2904de8d2c463bf585ba4257086f5bbe27d21c19413f5c9`

Bundle95의 실제 layer별 threshold는 `threshold_lock.json`이 authoritative하다.

- B128 lower: `[0.99920744, 0.99599200, 0.99415994, 0.99576092, 0.99708658, 0.99829960, 0.99731964, 0.99661773]`
- B256 lower: `[0.99918997, 0.99440843, 0.99208587, 0.99595082, 0.99702859, 0.99693966, 0.99421597, 0.99573022]`
- T128 lower: `[0.23416853, 0.29503015, 0.33653691, 0.25049138, 0.29534033, 0.34353572, 0.34388340, 0.25895900]`
- T256 lower: `[0.24612959, 0.30889857, 0.35070214, 0.22404544, 0.29037237, 0.34938255, 0.34994474, 0.25751927]`
- rel-L2 upper: `[0.06810204, 0.08499919, 0.09641258, 0.10021828, 0.10968129, 0.11881047, 0.12585881, 0.13152219]`
- `|log-r|` upper: `[0.01645426, 0.02303496, 0.02532124, 0.02703403, 0.02964939, 0.03216640, 0.03513953, 0.03791299]`

### 11.13 무엇이 검증됐고 무엇은 아직 아닌가

검증된 것:

1. Wiki before/after relational structure는 Code보다 훨씬 안정적이며 real CKA는 null보다 유의하게 높다.
2. CKA selector는 legacy raw-cosine selector와 거의 다른 집합을 고른다.
3. Pilot의 심한 4-window 집중은 full corpus에서 완화됐다.
4. 최종 bundle95는 direction/structure만이 아니라 token contribution과 magnitude까지 보존된 130,289 occurrence다.
5. dataset/checkpoint/window/B/T/M/mask/occurrence lineage가 SHA로 고정돼 있다.
6. raw hidden 없이 full census와 exact GT를 만들었다.

아직 검증되지 않은 것:

1. 이 anchor가 semantic Wiki membership을 뜻하는가 — 아니다. CKA stability만으로 membership은 증명되지 않았다.
2. 이 GT가 matched-count random이나 legacy cosine보다 causal하게 old forgetting을 줄이는가 — 아직 학습 ablation이 필요하다.
3. 0.006328%의 매우 작은 unique anchor set을 20% replay input budget으로 반복할 때 overfit 없이 도움이 되는가 — exposure 설계가 필요하다.
4. old-task 성능에 필요한 causal token인가, 단지 routing/representation이 안정적인 token인가 — 학습으로 검증해야 한다.
5. sealed pilot test 결과 — threshold lock은 완료했지만 test는 의도적으로 아직 열지 않았다. 독립 real test set을 쓸 계획이면 sealed pilot test를 영원히 보조 검증으로 남겨도 된다.

### 11.14 Full CKA 산출물과 구현 파일 지도

분포와 threshold를 직접 확인할 때는 다음 파일을 우선한다.

- full review: `full_census/threshold_review_v1/REPORT.md`
- full analysis JSON: `full_census/threshold_review_v1/analysis.json`
- threshold table: `full_census/threshold_review_v1/tables/candidate_thresholds.csv`
- condition marginals: `tables/marginal_pass_fractions.csv`
- B→T→M sequential contribution: `tables/sequential_B_T_M.csv`
- concentration: `tables/concentration.csv`
- repetition: `tables/token_id_repetition.csv`
- same-layer diagnostic: `tables/same_layer_diagnostic.csv`
- exact result: `full_census/exact_targeted_v2/summary.json`
- exact human review: `full_census/exact_targeted_v2/exact_review_v1/REPORT.md`
- final immutable decision: `full_census/gt_locked_bundle_95_v1/threshold_lock.json`

Full-distribution/zoom histogram:

- `histograms/raw_b_128_{full,zoom}.svg`
- `histograms/raw_b_256_{full,zoom}.svg`
- `histograms/token_min_b_128_{full,zoom}.svg`
- `histograms/token_min_b_256_{full,zoom}.svg`
- `histograms/token_min_t_128_{full,zoom}.svg`
- `histograms/token_min_t_256_{full,zoom}.svg`
- `histograms/relative_l2_{full,zoom}.svg`
- `histograms/abs_log_r_{full,zoom}.svg`

경로 기준점은 모두 `full_census/threshold_review_v1/`이다.

Full-census 및 exact-GT 구현:

- streaming census core/merge: `scripts/analysis/cka_gt_full_census.py`
- model runner: `scripts/analysis/run_cka_gt_full_census_mha.sh`
- 4-GPU launcher: `scripts/analysis/launch_cka_gt_full_census_4gpu.sh`
- full analysis: `scripts/analysis/analyze_cka_gt_full_census.py`
- B candidate extraction: `scripts/analysis/extract_cka_gt_b_candidate_windows.py`
- targeted exact writer/core: `scripts/analysis/cka_gt_targeted_gt.py`
- targeted runner: `scripts/analysis/run_cka_gt_targeted_gt_mha.sh`
- exact review: `scripts/analysis/review_cka_gt_exact_targeted.py`
- immutable lock: `scripts/analysis/lock_cka_gt_bundle.py`
- 관련 tests: 동일 디렉터리의 `test_*full_census*`, `test_*targeted*`, `test_extract_cka_gt_b_candidate_windows.py`, `test_review_cka_gt_exact_targeted.py`

마지막 targeted e2e audit 기준 관련 CPU test 55/55, py_compile, bash syntax, PLAN_ONLY가 통과했다. CKA 관련 파일 다수가 아직 Git에서 untracked일 수 있으므로 cleanup 전에 반드시 보존/commit 여부를 확인한다.

## 12. 아직 결론 내리지 않은 것

threshold, full coverage, exact count는 이제 확정됐다. 남은 연구 질문은 다음이다.

1. locked bundle95가 matched-count random contextual occurrence보다 old-task preservation에 실제로 유리한가
2. legacy cosine GT와 CKA GT 중 어느 쪽이 Wiki forgetting을 더 잘 줄이는가
3. CKA-only B/T와 CKA+M full selector 중 magnitude gate의 causal 기여가 있는가
4. 130,289 unique occurrence를 어떤 input/anchor exposure budget으로 반복해야 overfit을 피할 수 있는가
5. Mahalanobis/prototype 또는 별도 density-ratio membership이 semantic old-domain purity를 더하는가
6. MSE/hidden KL/vocab KL/LM 중 어느 preservation objective가 CKA anchor에 가장 맞는가
7. selected 107 document와 상위 5 document 62.85% 집중이 학습 결과를 지배하는가
8. sealed pilot test를 열 필요가 있는가, 아니면 독립 real downstream test만으로 causal evaluation을 할 것인가
9. Code에서 효과가 확인된 뒤 Conversation의 Wiki+Code dual-old source에 어떻게 확장할 것인가

이미 확정되어 다시 논의할 필요가 없는 것:

- final bundle: 95
- primary consensus: condition-specific 7/8, valid layer가 정확히 6이면 6/6
- final GT count: 130,289
- full eligible coverage: 0.006328%
- raw hidden 미저장
- 원문맥 유지
- label 의미: replay-stable relational anchor pseudo-GT

## 13. 권장 다음 순서

### 단계 A — CKA GT용 contextual replay dataset/loader

현재 bundle95 mask는 document-window axis다. 기존 cosine `OldLikeGTDataset`은 GPTDataset outer-sample axis이므로 그대로 연결하면 안 된다.

구현 원칙:

1. `bundle_95_occurrences.npy`와 packed mask의 SHA를 lock record와 대조한다.
2. selected occurrence가 속한 원래 document-bounded 512-token window를 가져온다.
3. window는 절대 이어붙이지 않고 원문맥/원 token order를 유지한다.
4. replay loss mask는 GT 위치에만 1, 나머지는 context-only 0이다.
5. 한 window에 GT가 여러 개면 같은 forward에서 모두 supervise할 수 있으나, exposure 계산은 occurrence 단위로 별도 기록한다.
6. primary Code LM과 replay batch는 기존 1-Phase처럼 같은 optimizer step에 gradient를 accumulate한다.
7. replay branch에서는 router gradient만 남기고 expert/non-router replay gradient를 제거한다.

### 단계 B — budget을 두 축으로 고정

최종 GT는 130,289개로 매우 작다. 기존 20%를 그대로 “input token”만으로 맞추면 수백~수천 GT epoch가 될 수 있다. 각 run metadata에 반드시 둘 다 기록한다.

- replay input/context tokens
- supervised GT occurrence presentations 및 unique-GT epochs

Primary comparison은 `같은 replay input token`과 `같은 supervised occurrence presentation`을 동시에 맞춘다. 불가능하면 어느 축을 고정했는지 run 이름과 report에 명시한다.

### 단계 C — 200-step causal training ablation

full 1800-step 전에 작은 200-step으로 다음 arm을 비교한다.

1. new-task LM only
2. 실제 Wiki replay upper bound
3. legacy cosine GT + hidden MSE
4. CKA+M locked GT + hidden MSE
5. CKA-only GT + hidden MSE
6. matched-count random contextual occurrence + hidden MSE
7. 가능하면 CKA GT + hidden KL 또는 vocab KL

모든 arm에서:

- 같은 before/student/teacher checkpoint
- 같은 primary new-task data/order
- 같은 selected-token count 또는 replay token budget
- 같은 원문 512-token context
- GT 위치만 replay loss mask
- natural routing
- replay gradient는 router-only
- primary LM은 새 expert+router
- hidden MSE L2~L9, L9 포함

평가:

- Wiki preservation
- Code acquisition
- hidden drift
- router 변화
- expert usage
- learned routing/forced-routing diagnostic 분리
- selector coverage
- random 대비 우위
- storage cost
- old replay 제거 성능

Go 기준은 matched-count random보다 CKA GT가 일관되게 old preservation을 높이고, Code acquisition 손실이 허용 범위인 경우다. 그렇지 않으면 CKA stability가 causal preservation anchor라는 가설을 기각하거나 selector/objective를 분리 재검토한다.

### 단계 D — sealed test 정책

bundle95 lock은 이미 완료됐다. sealed pilot test를 공개한다면 단 한 번만 열어 REPORT v2를 만들고 threshold는 바꾸지 않는다. 다만 사용자가 독립 test set을 authoritative evaluation으로 쓰기로 했으므로, sealed pilot test를 당장 열지 않고 보조 sanity set으로 계속 보존하는 것도 타당하다. 현재 상태는 `sealed_test_opened=false`다.

### 단계 E — full 1800 및 Conversation 확장

200-step에서 CKA GT가 random/legacy보다 우월할 때만 full 1800으로 확장한다. 그 다음에만 Conversation에서 Wiki+Code old source를 함께 다루는 CKA/membership 설계를 진행한다.

## 14. Conversation 확장 관련 교훈

기존 cosine 방식으로 Conversation contextual GT도 생성했다.

- GT root: `/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812/analysis/conversation_old_like_gt_l2_l9_top1`
- total valid token: 1,402,994,688
- selected: 1,311,043
- selected fraction: 0.093446%
- threshold는 Conversation 자체 layer별 top-1% 교집합으로 Code와 절대값은 다르다.

Conversation에서는 새 task acquisition과 old preservation이 Code보다 좋지 않았고, token만 이어붙인 miniset/문맥 처리 문제도 드러났다. 다음 CKA 실험은 Code에서 causal 효과를 먼저 검증한 뒤 Conversation으로 확장한다.

Conversation exact-axis 1800-step 결과(Conversation/Wiki/Code 순서):

| 조건 | Conversation | Wiki | Code |
|---|---:|---:|---:|
| no replay | 0.385744 | 0.415828 | 0.516702 |
| 실제 old replay hidden MSE c10 | 0.382280 | 0.457400 | 0.663995 |
| 실제 old replay hidden KL | 0.383017 | 0.456858 | 0.664056 |
| 실제 old replay vocab KL | 0.378944 | 0.458086 | 0.660571 |
| cosine old-like MSE c10 | 0.376219 | 0.448175 | 0.611544 |
| cosine old-like hidden KL c1 | 0.384224 | 0.433720 | 0.447568 |
| cosine old-like vocab KL c1 | 0.379552 | 0.449542 | 0.592046 |

- old-like MSE c10은 no-replay보다 Wiki/Code를 보존하지만 실제 old replay보다 크게 낮다.
- hidden KL c1은 Conversation을 배우지만 Code forgetting이 매우 컸다.
- objective branch마다 시작 checkpoint/expansion KD-init가 다를 수 있어 최종 절대값만으로 공정한 ranking을 하면 안 되고 start→end 변화도 함께 봐야 한다.
- MSE c0.3은 100 step에서 Code 0.6649→0.6096, Wiki 0.4541→0.4328로 c10보다 나빠 중단했다. 여기서 C는 hidden-MSE loss coefficient이며 stable-subspace 보존계수와 다른 개념이다.

Conversation에 적용할 때 old-positive source가 Wiki+Code 두 개가 되므로, source membership을 Wiki/Code별로 구분할 필요가 있다. CKA window context와 Mahalanobis/prototype diagnostic을 이때 활용할 수 있지만, 현재 pilot에서 membership은 아직 GT 조건이 아니다.

## 15. 현재 TRACE/GRPO 후속 체인

2026-08-16 문서 최종화 시점에 평가는 background systemd service에서 계속 실행 중이다. 학습은 끝났으며 평가만 진행 중이다.

이미 완료된 것:

- scalar GRPO: 1107/1107, adapter 저장
- discrete GRPO: 1107/1107, adapter 저장
- TRACE V3 hidden-MSE c10 training: 8 task 모두 완료
- V3 final: `/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3/instruct_hidden_mse_postkd_c10_20260814/v3_new_hidden_mse_c10/7`

첫 TRACE sparse-15 평가는 Py150 batch8 마지막 124/125 부근에서 V2/V3 모두 CUDA OOM으로 실패했다. 학습 실패가 아니라 평가 OOM이었다. batch2로 재개하여 sparse-15를 진행했고, 이후 V3-new top4 lower-triangle와 trained SLoRA 평가 체인으로 넘어갔다.

현재 authoritative service:

`trace-after-v3c10-top4-lower-slora-pre-eval-gpu012345-20260816.service`

- 2026-08-16 22시대 재구성 시 MainPID: `438704`; 새 세션에서는 다시 조회할 것
- lower-triangle worker pool: physical GPU 0,1,2,3,4,5
- GPU 6,7은 사용자 요청에 따라 비워 둠
- 기존 lower-triangle 완료 summary는 보존/skip
- 중단 당시 완전히 끝나지 않은 task만 재시작
- lower-triangle shard format은 기존 4-shard를 유지하여 artifact 호환성을 보존
- 한 evaluation task가 실패하면 해당 model×task 단위만 skip하고 다음 task를 계속함
- lower-triangle 완료 후 trained SLoRA evaluation을 GPU0~5에서 6-way shard로 수행
- SLoRA: `EVAL_SHARD_COUNT=6`, worker index 0~5, sparse15/all-round evaluation
- GRPO와 V3 c10 학습은 이미 끝났으므로 재실행하지 않음

2026-08-16 22:58 KST snapshot:

- top4 lower-triangle는 order8까지 진입
- 이전 완료 artifact를 포함한 summary JSON 수는 재편 직전 78개
- 새 6-GPU invocation status: active=6, pending=9
- GPU0~3: MeetingBank shard1~4 진행
- GPU4~5: Py150 shard1~2 진행
- 관찰 progress: MeetingBank 약 25~41%, Py150 약 16~31%
- GPU6,7: free
- 이 수치는 live snapshot이지 최종 score가 아니다.

로그:

- current lower-triangle: `/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/logs/instruct_hidden_mse_postkd_c10_20260814/after_v3_c10_top4_lower_then_slora_pre_20260816/01_v3_new_top4_lower_triangle.log`
- chain status/output root: 같은 디렉터리 `after_v3_c10_top4_lower_then_slora_pre_20260816/`
- 이전 sparse-15 status: `/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/logs/instruct_hidden_mse_postkd_c10_20260814/train_then_parallel_eval_20260815/status.log`

간단한 확인 명령:

```bash
systemctl --user status trace-after-v3c10-top4-lower-slora-pre-eval-gpu012345-20260816.service --no-pager
tail -f /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/logs/instruct_hidden_mse_postkd_c10_20260814/after_v3_c10_top4_lower_then_slora_pre_20260816/01_v3_new_top4_lower_triangle.log
```

새 세션 시작 시 systemd state, summary JSON, GPU release를 다시 확인하고 이 문서의 최종 상태 기록을 우선한다.

## 16. 새 세션에서 바로 할 일

1. 이 문서를 읽고 경로/프로세스/최종 validation을 read-only로 확인한다.
2. CKA full census `threshold_review_v1/REPORT.md`, exact review, bundle95 lock record를 authoritative 순서로 읽는다.
3. bundle95 lock이 이미 완료됐으므로 threshold를 다시 refit하거나 덮어쓰지 않는다.
4. 사용자 승인 없이 sealed test를 열지 않는다.
5. CKA GT contextual loader와 exposure-matched 200-step causal ablation을 다음 구현 대상으로 삼는다.
6. full-corpus GT는 이미 생성됐으므로 다시 전체 forward하지 않는다.
7. CKA를 semantic old-domain classifier라고 부르지 않는다. 현재 가장 강한 표현은 `replay-stable relational anchor`다.
8. 기존 cosine GT와 CKA GT의 matched-count 비교를 반드시 유지한다.
9. old-like replay는 원문맥 forward + GT-position loss mask를 사용하고, token-only concatenated miniset은 hidden KD에 사용하지 않는다.

## 17. 새 세션 답변 형식

처음 답변에서는 다음을 짧고 명확하게 보고하라.

1. 현재 완료 상태와 실행 중 프로세스
2. cosine 방식에서 확인된 사실과 한계
3. CKA pilot과 full-census exact selector 정의
4. exact 95/97/99 수량 및 locked95=130,289 결과
5. 아직 test가 봉인되어 있다는 사실
6. 다음 causal-training decision fork
7. 다음 한 단계만 제안

설계만 추상적으로 말하지 말고 실제 파일·checkpoint·log에 근거하라. 확인하지 않은 수치를 추측하지 말라.

---

## 최종 상태 기록란

TRACE 재개 체인은 이 문서 전달 시점에도 background에서 돌고 있다. 새 세션은 완료 후 아래 항목을 실제 값으로 갱신한다.

- service state at latest handoff: `active/running`, service `trace-after-v3c10-top4-lower-slora-pre-eval-gpu012345-20260816.service`, launch-time MainPID 438704
- V2/V3 sparse-15: 평가 재개 체인을 지나 top4 lower-triangle 단계로 전환됨
- top4 lower-triangle: order8 진행 중; 6 GPU worker(0~5), completed artifacts preserved
- trained SLoRA evaluation: lower-triangle 뒤 6 GPU/6 shard로 예약됨
- GPU6,7: free by design
- failure/retry notes: `Py150 batch8 OOM 후 batch2 평가 재개; 이후 lower-triangle worker pool을 0~5로 재편`
