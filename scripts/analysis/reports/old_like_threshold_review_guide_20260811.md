# Old-like token threshold review guide

## 먼저 결론: 이것을 replay data라고 부를 수 있는가?

현재 결과만으로 “선택된 Code token을 replay하면 Wiki 성능이 유지된다”고 결론 내릴 수는 없다.

현재 확보한 것은 다음과 같다.

- Wiki 학습 후 확장·KD-init 모델과 Code+Wiki replay 학습 완료 모델 사이에서 representation이 거의 유지된 Code token occurrence를 찾을 수 있다.
- 실제 Wiki token이 같은 stability score에서 매우 높은 값을 보이므로, 이러한 Code token을 **Wiki-like stability pseudo-label** 후보로 삼을 수 있다.
- 이 Code token을 이후 학습에서 router layer-output KD가 적용될 surrogate input으로 사용할 수 있다.

따라서 용어를 구분해야 한다.

- Wiki 문장을 저장해 LM loss를 다시 주는 것: 실제 old replay
- 선택된 Code token에서 teacher router output을 KD하는 것: old-like surrogate rehearsal
- 선택된 Code sequence 자체를 별도 buffer에 저장해 다시 흘리는 것: Code-derived replay이지 Wiki replay는 아님

성능 보존 가능성은 다음 KD 실험에서 확인해야 한다. 이번 결과는 selector feasibility까지만 증명한다.

## 정확히 어떤 cosine을 계산했는가?

동일한 contextual token occurrence `t`를 두 checkpoint에 넣고 Layer 2–9의 residual-add 완료 `layer_output`을 비교했다.

각 layer `l`에서:

`c(t,l) = cosine(h_reference(t,l), h_current(t,l))`

- `reference`: Wiki 학습 후 expert 확장·KD-init, step 600
- `current`: Code LM + Wiki replay LM 1-Phase 완료, step 1800
- Layer 1은 동일한 dense layer라 제외
- token ID type이 아니라 sample ID와 position으로 구분되는 contextual occurrence 단위

## 여러 layer를 어떻게 합쳤는가?

hidden vector를 평균내거나 concatenate한 뒤 cosine을 계산하지 않았다. Layer별 cosine scalar 8개를 먼저 계산한 다음 합쳤다.

### 1. Raw cosine mean

`raw_cosine_mean(t) = mean_l=2..9 c(t,l)`

가장 단순한 산술평균이다. Wiki-positive AUROC는 0.99937로 가장 높았다.

### 2. Layer-normalized percentile

Layer마다 Code cosine 분포가 다르므로 각 cosine을 해당 layer의 Code empirical CDF percentile로 변환했다.

`p(t,l) = F_Code,l(c(t,l))`

예를 들어 `p=0.99`는 그 layer의 전체 Code token 중 상위 약 1%라는 의미다.

그 후 다음을 계산했다.

- `percentile_mean_l2_l9 = mean(p(t,2)..p(t,9))`
- `percentile_median_l2_l9 = median(p(t,2)..p(t,9))`
- `percentile_min_l2_l9 = min(p(t,2)..p(t,9))`
- `late_percentile_mean_l7_l9 = mean(p(t,7),p(t,8),p(t,9))`

Wiki도 Wiki 자체 CDF가 아니라 같은 Code CDF로 변환했다. 그래야 두 domain의 위치를 비교할 수 있다.

### 3. Stable-layer count

각 layer에서 Code 상위 `p%` 안에 드는지를 이진화한 뒤 8개 layer 중 몇 개에서 반복되는지 센다.

`K_top-p(t) = sum_l 1[p(t,l) >= 1-p/100]`

예: `top 1%, at least 6 layers`는 Layer 2–9 중 최소 6개에서 각각 Code 상위 1%에 든 token이다.

## 사용자가 직접 볼 자료

### A. Layer별 raw cosine

- [Layer 2–9, cosine 0.9–1 확대, log histogram](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/old_like_threshold_review/layer_raw_cosine_hist_zoom_0p9_1_log.png)
- [Layer 2–9, 전체 -1–1, log histogram](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/old_like_threshold_review/layer_raw_cosine_hist_log.png)

파란색은 Code, 빨간색은 Wiki다. 그림은 가독성을 위해 100개 bin으로 묶었으며 정확한 계산은 원래 20,000-bin count를 사용했다.

### B. 여러 layer를 합친 global score

- [Global score 0.8–1 확대, log histogram](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/old_like_threshold_review/global_score_hist_zoom_0p8_1_log.png)
- [Global score 전체 0–1, log histogram](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/old_like_threshold_review/global_score_hist_log.png)

### C. Threshold를 직접 고르기 위한 정확한 표

- [Global score threshold 표](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/old_like_threshold_review/compact_threshold_review.csv)
- [Stable-layer count 표](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/old_like_threshold_review/stable_layer_count_review.csv)
- [전체 threshold curve](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/code_wiki_old_like_calibration/score_threshold_curves.csv)
- [Threshold curve 그림](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/code_wiki_old_like_calibration/score_threshold_curves.png)

표의 의미:

- `threshold_ge`: 이 값 이상을 old-like 후보로 둘 때의 threshold
- `actual_code_coverage`: 전체 Code occurrence 중 선택되는 비율
- `wiki_recall`: Wiki positive token 중 선택되는 비율
- `balanced_prior_precision`: Code/Wiki를 1:1 prior로 놓은 비교용 precision
- 모든 `selected_as_gt`는 False이며 아직 어떤 행도 GT로 선택하지 않았다.

## 대표적인 비교 지점 — 선택한 threshold가 아님

| Score | Threshold | Code 선택률 | Wiki recall |
|---|---:|---:|---:|
| percentile mean L2–9 | 0.978 | 1.002% | 98.516% |
| percentile median L2–9 | 0.987 | 1.025% | 99.740% |
| late percentile mean L7–9 | 0.988 | 1.033% | 99.799% |
| raw cosine mean L2–9 | 0.908 | 1.010% | 99.765% |

Stable-layer count의 대표 지점:

| 조건 | Code 선택률 | Wiki recall |
|---|---:|---:|
| layerwise top 1%, 최소 4/8 layers | 0.963% | 99.726% |
| layerwise top 1%, 최소 6/8 layers | 0.652% | 99.529% |
| layerwise top 1%, 8/8 layers | 0.152% | 59.672% |

## Cosine만 보고 선택하면 안 되는 이유

single-layer high cosine에는 방향만 유지되고 크기는 달라진 token이나 low-norm token이 섞인다.

확인용 자료:

- [High-cos relative L2 표](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/cross_layer_stability_l2_l9/high_cos_relative_l2.csv)
- [High-cos norm ratio 표](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/cross_layer_stability_l2_l9/high_cos_norm_ratio.csv)
- [High-cos reference norm 표](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/cross_layer_stability_l2_l9/high_cos_reference_norm.csv)
- [Stable-layer count별 drift 표](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/cross_layer_stability_l2_l9/stable_layer_diagnostics.csv)
- [Cosine percentile vs relative L2 joint plot](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/cross_layer_stability_l2_l9/cosine_percentile_vs_relative_l2_joint.png)
- [Cosine percentile vs log norm ratio joint plot](/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/cross_layer_stability_l2_l9/cosine_percentile_vs_log_norm_ratio_joint.png)

특히 Code의 Layer 9 top-1% cosine 집합은 relative L2 중앙값이 0.405이고, reference norm bottom-5% 비율이 30.1%다. 따라서 `Layer 9 cosine ≥ threshold`만으로 old-like를 정의하면 안 된다.

## Threshold 검토 순서 제안

1. `layer_raw_cosine_hist_zoom_0p9_1_log.png`에서 layer별 분포 차이를 확인한다.
2. `global_score_hist_zoom_0p8_1_log.png`에서 Wiki peak와 Code tail이 어디서 만나는지 본다.
3. `compact_threshold_review.csv`에서 원하는 Code coverage와 Wiki recall의 trade-off를 고른다.
4. 해석이 쉬운 조건이 필요하면 `stable_layer_count_review.csv`를 본다.
5. 선택 후보가 정해지면 relative L2, norm ratio, reference norm 조건을 추가할지 결정한다.
6. 그 뒤에만 hard GT 또는 soft weight를 고정하고 KD 실험으로 넘어간다.

현 단계에서는 `percentile median/late mean` 또는 `top-1%에서 최소 4–6 layers`가 사람이 검토하기 좋은 후보지만, 어느 것도 아직 최종 기준으로 선택하지 않았다.
