# Code token old-like stability feasibility — decision report

Date: 2026-08-11 (Asia/Seoul)

## Executive decision

Final decision: **A — a global token-level old-like stability score is feasible for this checkpoint pair.**

Here, “old-like” has a deliberately narrow operational meaning: a contextual Code token occurrence has a Layer 2–9 stability profile resembling actual Wiki token occurrences under the same reference/current checkpoint comparison. It does **not** prove semantic Wiki membership or causal importance to Wiki task performance. No GT label or training threshold was assigned in this analysis.

The data do not support cosine-only, single-layer labeling. The supported structure is:

1. one global cross-layer token selector;
2. plus explicit low-drift, norm-preservation, and non-low-activity guards;
3. with layer-specific diagnostics retained, but no token×layer mask required for the selector itself.

## 1. Data, checkpoints, and validation

Reference model — Wiki training followed by expert expansion and KD initialization, step 600:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600`

Current model — Code LM plus Wiki replay LM, one-phase training complete, step 1800:

`/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_code_bootstrap_20260810/g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt-v2`

Both forwards used residual-included Transformer `layer_output`. Layer 1 was excluded because it is the unchanged dense layer. All conclusions use Layers 2–9.

Code full census:

- 4,147,200 sequences × 512 = 2,123,366,400 contextual token occurrences
- 8 workers, 864 metric shards, approximately 478 GB
- exact partition coverage and token identity
- no missing or duplicate partition
- no NaN/Inf
- deep extraction validation passed
- cross-layer validation passed with exactly 2,123,366,400 observations per layer

Wiki positive-reference extraction:

- Wiki train dataset: `/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/wiki/train/train_text_document`
- 21,504 sequences × 512 = 11,010,048 contextual token occurrences
- four contiguous, non-overlapping workers on GPUs 0–3
- 8 metric shards, 2,477,455,920 bytes
- identical Wiki token inputs for the two checkpoints
- deep validation passed for token hashes, shard ranges, layer shapes, finite/range checks, progress, and reservoirs

The first background launch was terminated by the execution environment. A brief duplicate-launch overlap later left rank 0 with final shards at 5,376 samples but progress at 4,800. That rank was not accepted. It was preserved under `collision_archive` and regenerated cleanly; the final four-worker root then passed deep validation.

## 2. Layer 2–9 percentile definition

Each layer’s cosine was mapped using the Code full-census empirical CDF:

`p_l(t) = F_Code,l(cos_l(t))`

The CDF uses fixed cosine bins over `[-1,1]` with width `1e-4`, and the bin midrank is used as the percentile. Consequently, percentile values are approximate within the tied histogram bin. The largest midrank tie error is about 0.00327 in Layer 2; the other Layers 3–9 are approximately 0.00014–0.00022.

Wiki was deliberately transformed with the **Code CDF**, not a Wiki-self CDF. A Wiki-self percentile would be uniform by definition and would destroy the domain comparison.

## 3. Cross-layer consistency in Code

The approximate Spearman correlation is Pearson correlation of the histogram-CDF percentile ranks.

- mean off-diagonal Layer 2–9 Spearman: **0.6610**
- adjacent-layer Spearman:
  - L2–L3: 0.7538
  - L3–L4: 0.8700
  - L4–L5: 0.9208
  - L5–L6: 0.8473
  - L6–L7: 0.9145
  - L7–L8: 0.9178
  - L8–L9: 0.8481

Early-to-late correlation is weaker—for example L2–L9 is about 0.243—but local and adjacent coherence is strong.

The Layer 2–9 percentile mean is robust to removing any single layer. Its leave-one-layer-out correlation is 0.9929–0.9979. Therefore, one global score is not being driven by a single layer.

## 4. Top-p overlap and repeated stability

Code top-1% sets show strong overlap far above independent selection:

- mean pairwise Jaccard: **0.4693**
- mean independence lift: **59.7×**
- stable in at least 6/8 layers: **13,847,590 tokens, 0.6522%**
- stable in all 8/8 layers: **3,232,030 tokens, 0.1522%**
- stable in all L7–L9: **0.7558%**

For wider layerwise sets:

| Layerwise set | Code ≥6/8 | Code 8/8 | Mean Jaccard |
|---|---:|---:|---:|
| top 1% | 0.6522% | 0.1522% | 0.4693 |
| top 5% | 2.7059% | 0.8502% | 0.4140 |
| top 10% | 6.1345% | 2.1927% | 0.4647 |
| top 20% | 13.5883% | 6.2802% | 0.5205 |

This rejects the hypothesis that high stability is only an unrelated, layer-local accident.

## 5. Relative L2: high cosine is not sufficient

Within each Code layer’s top-1% cosine set, median relative L2 grows with depth:

- L2: 0.015
- L3: 0.095
- L4: 0.190
- L5: 0.216
- L6: 0.261
- L7: 0.278
- L8: 0.315
- L9: 0.405

At L9, only 80.0% of the top-1% cosine set has relative L2 ≤ 0.5. Thus a late-layer token can retain direction while still moving materially in Euclidean distance.

Repeated cross-layer selection is much better behaved. For Code top-1% membership:

- stable in 0 layers: median mean relative L2 0.918; late L7–L9 relative L2 1.109
- stable in 6 layers: 0.278; late 0.380
- stable in all 8 layers: **0.139; late 0.216**

Actual Wiki all-8 top-1% tokens are more stable still:

- median mean relative L2: **0.042**
- median late L7–L9 relative L2: **0.054**

Therefore stable-layer repetition is useful, but the final selector should still retain a relative-L2 guard.

## 6. Norm preservation and scaling artifacts

For Code layerwise top-1% sets, the median norm ratio is near one, but the fraction within `[0.9,1.1]` declines from 100% at L2 to 78.0% at L9. Layerwise top-5% is worse at L9: only 65.8% lies within `[0.9,1.1]`.

Cross-layer repetition removes much of this scaling contamination:

- Code top-1%, 0 stable layers: median maximum absolute log norm ratio 0.305
- Code top-1%, 6 stable layers: 0.070
- Code top-1%, all 8 layers: **0.039**
- Wiki top-1%, all 8 layers: **0.023**

Hence a global selector is defensible only with a norm-preservation guard; raw cosine by itself is not.

## 7. Low-activity/dead-like diagnostic

Reference hidden RMS was estimated from stored scalars as:

`reference_rms = sqrt(delta_mse) / relative_l2`

The proxy was checked against raw hidden reservoirs from all eight Code workers:

- 32,768 Layer 2–9 token-layer entries
- 32,768 valid
- median relative error: 0
- p99 relative error: 8.91e-8
- maximum relative error: 1.11e-7

Single-layer high-cosine sets do contain low-reference-norm enrichment. In the Code top-1% cosine set, the fraction in that layer’s global bottom-5% reference RMS is:

- L7: 11.5%
- L8: 15.3%
- L9: **30.1%**

For Wiki positive tokens under the same Code top-1% thresholds, the corresponding values are approximately 9.6%, 5.2%, and 5.9%. The L9 Code enrichment is therefore a genuine warning against single-layer cosine labeling, not a general property of all stable Wiki tokens.

## 8. Wiki positive-reference calibration

Wiki representation is nearly unchanged between the two checkpoints:

- Layer 2–9 raw cosine means: approximately 0.99884, 0.99849, 0.99804, 0.99784, 0.99739, 0.99695, 0.99660, 0.99633
- global Code-CDF percentile mean: **0.9960**
- percentile median: **0.9986**
- late L7–L9 percentile mean: **0.9988**
- raw cosine mean: **0.9976**

When using Code layerwise top-1% thresholds, Wiki positive-reference tokens are selected at:

- L2: 59.8%
- L3: 98.1%
- L4–L9: 99.5%–99.8%
- stable in at least 6/8 layers: **10,958,217 / 11,010,048 = 99.53%**
- stable in all 8/8 layers: **59.67%**

By contrast, Code stable in at least 6/8 layers is only 0.652%. This is a 152.6× Wiki-over-Code enrichment with balanced-prior precision 0.9935.

## 9. AUROC and global-score comparison

Wiki is treated as an analysis positive reference. Code is not assumed to be a pure negative class, because the purpose is precisely to locate a Wiki-like subset inside Code. Consequently, these AUROCs measure distributional separability, not semantic classification ground truth.

| Global score | Binned Wiki-positive AUROC |
|---|---:|
| raw cosine mean, L2–L9 | **0.99937** |
| late L7–L9 percentile mean | **0.99920** |
| percentile median | **0.99888** |
| percentile mean | **0.99844** |
| percentile minimum | 0.99533 |

Layerwise raw-cosine AUROC ranges from 0.9778 at L2 to 0.99935 at L9. Relative-L2 AUROC, where lower is Wiki-like, ranges from 0.9780 at L2 to 0.99929 at L9. Absolute log-norm-ratio AUROC is weaker but still informative at 0.8227–0.9433.

Representative operating points—not selected GT thresholds—are:

- percentile mean ≥0.979: Wiki recall 98.45%, Code selected 0.9567%
- percentile median ≥0.989: Wiki recall 99.71%, Code selected 0.8778%
- late L7–L9 percentile mean ≥0.989: Wiki recall 99.79%, Code selected 0.9520%
- raw cosine mean ≥0.908: Wiki recall 99.76%, Code selected 1.0102%

These are histogram-resolution examples for feasibility. They are not final hard thresholds.

## 10. Global score versus token×layer mask

A global token selector is supported because:

1. the same Code token repeatedly appears in high-stability sets across layers;
2. global percentile scores are stable under single-layer dropout;
3. Wiki positive tokens concentrate near the top of those global scores;
4. global AUROC is 0.9984–0.9994;
5. repeated cross-layer selection suppresses scaling and low-activity artifacts.

A token×layer mask is therefore not required to answer “which Code token occurrence is old-like?” It may still be useful later to control which layer receives KD, because the amount of drift differs by layer. Token selection and layerwise KD targeting should remain separate decisions.

## 11. Recommended selector structure for the next goal

Do not use a single-layer or cosine-only condition. The supported candidate is:

`global stability score + relative-L2 guard + norm-preservation guard + non-low-reference-norm guard`

Two reasonable global score families remain for the next calibration stage:

- performance-first: Layer 2–9 raw cosine mean, which has the best observed AUROC;
- rank-robust: percentile median or late L7–L9 percentile mean, which respects layer-specific cosine scales and is robust to layer dropout.

The threshold and whether these are combined as hard gates or a soft calibrated score must be selected in the next goal using an explicitly chosen Wiki-recall/Code-coverage operating point and then validated by downstream preservation/acquisition performance.

## 12. Additional Code forward decision

No additional full Code forward for `cos(mean hidden L2–L9)` is needed now.

The existing per-layer scalar census already shows strong cross-layer agreement, global-score robustness, and near-perfect Wiki/Code separation. The raw-hidden reservoir estimate for `cos(mean hidden)` is not promoted to a full-census claim, but it is no longer decision-critical. Re-forwarding 2.12B Code tokens would add cost without changing the present feasibility decision.

## 13. Final conclusion

**Selected outcome: A.**

There is a large, reproducible subset of Code contextual token occurrences whose representation remains stable across many Layers 2–9 and whose global stability score is strongly calibrated by actual Wiki tokens. This supports a global old-like token selector for this checkpoint pair.

The claim is not that stability alone equals old knowledge. Rather:

- stability provides the selector signal;
- Wiki positive calibration gives it an old-domain interpretation;
- relative L2, norm ratio, and reference norm prevent obvious direction-only, scaling, and low-activity false positives;
- no preservation direction, KD target, GT threshold, or router behavior is inferred here.

## 14. Gate for the next KD experiment

Proceed to a new KD-training goal only after fixing one operating point from the saved threshold curves. The next experiment should:

1. freeze this selector definition before looking at downstream evaluation;
2. apply router layer-output KD only to selected old-like Code token occurrences;
3. use expert + router LM loss on the remaining Code occurrences;
4. keep natural routing and never force dispatch from the fingerprint;
5. compare hard and soft weighting;
6. compare against random token selection with matched token count and matched KD budget;
7. compare no replay and Wiki replay settings;
8. report Wiki preservation, Code acquisition, hidden drift, routing/expert usage, and selected-token coverage separately.

The selector should be rejected or revised if matched-budget random selection performs equally, if the chosen non-dead/low-drift guards do not improve preservation, or if Wiki preservation comes at unacceptable Code acquisition cost.

## Authoritative artifacts

- Code cross-layer analysis: `/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/cross_layer_stability_l2_l9`
- Wiki extraction validation: `/data2/seonghyeonnoh/LLM-continual-learning-runs/wiki_token_representation_positive_20260811/validation.json`
- Wiki Code-CDF cross-layer analysis: `/data2/seonghyeonnoh/LLM-continual-learning-runs/wiki_token_representation_positive_20260811/analysis/cross_layer_stability_l2_l9_code_cdf`
- Code/Wiki calibration: `/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811/analysis/code_wiki_old_like_calibration`
