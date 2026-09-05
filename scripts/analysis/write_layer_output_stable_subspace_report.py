#!/usr/bin/env python3
"""Write the evidence-grounded Korean report for the stable layer-output study."""

import argparse
import csv
import json
import os


DISPLAY = {
    "code_only_no_replay_no_router_ft": "Code-only (old replay 없음, router FT 없음)",
    "code_lm_plus_wiki_layer_output_hidden_kl_router_gradient": "Code LM + Wiki layer-output hidden-KL (router gradient 포함)",
    "code_only_then_wiki_code_router_only_lm_ft": "Code-only 후 Wiki+Code router-only LM FT",
}


def mib(value):
    return value / 1024 ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result_path = os.path.join(args.root, "analysis/stable_subspace_results.json")
    with open(result_path, encoding="utf-8") as handle:
        result = json.load(handle)
    with open(os.path.join(args.root, "analysis/streaming_validation.json"), encoding="utf-8") as handle:
        validation = json.load(handle)
    with open(os.path.join(args.root, "analysis/lineage_and_loss_audit.json"), encoding="utf-8") as handle:
        audit = json.load(handle)

    report_dir = os.path.join(args.root, "report")
    os.makedirs(report_dir, exist_ok=True)
    code_only = result["wiki_pairs"]["code_only_no_replay_no_router_ft"]
    mixed = result["wiki_pairs"]["code_lm_plus_wiki_layer_output_hidden_kl_router_gradient"]
    restored = result["wiki_pairs"]["code_only_then_wiki_code_router_only_lm_ft"]

    rank_rows = []
    for rank in (8, 16, 32, 64, 128, 256):
        rows = []
        for layer in range(2, 10):
            q = code_only["layers"][str(layer)]["ranks"][str(rank)]
            stable = q["stable"]["test"]
            random_hi = q["random"]["test"]["variance_to_drift_ratio"]["ci95"][1]
            rows.append((stable["variance_to_drift_ratio"] / random_hi,
                         stable["old_variance_fraction"], stable["drift_energy_fraction"]))
        storage = result["fingerprint_storage_bytes"][str(rank)]
        rank_rows.append({
            "rank": rank,
            "min_advantage": min(row[0] for row in rows),
            "variance_min": min(row[1] for row in rows),
            "variance_max": max(row[1] for row in rows),
            "drift_min": min(row[2] for row in rows),
            "drift_max": max(row[2] for row in rows),
            "fp16_mib": mib(storage["fp16"]),
            "fp32_mib": mib(storage["fp32"]),
        })
    with open(os.path.join(report_dir, "rank_sweep_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rank_rows[0].keys())
        writer.writeheader(); writer.writerows(rank_rows)

    drift_rows = []
    for label, pair in result["wiki_pairs"].items():
        for layer in range(2, 10):
            row = pair["layers"][str(layer)]
            drift_rows.append({
                "condition": label, "layer": layer,
                "mean_cosine": row["token_drift_test"]["mean_cosine"],
                "mean_relative_l2": row["token_drift_test"]["mean_relative_l2"],
                "linear_cka": row["full_alignment_test"]["linear_cka"],
                "normalized_procrustes": row["full_alignment_test"]["normalized_procrustes"],
                "ffn_mean_cosine": row["ffn_output_diagnostic_test"]["mean_cosine"],
                "ffn_mean_relative_l2": row["ffn_output_diagnostic_test"]["mean_relative_l2"],
            })
    with open(os.path.join(report_dir, "condition_layer_drift.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=drift_rows[0].keys())
        writer.writeheader(); writer.writerows(drift_rows)

    lines = [
        "# Stable residual layer-output direction/subspace 존재 분석",
        "",
        "## 결론",
        "",
        "1. **존재한다.** old replay와 router FT 없이 Code만 학습한 뒤에도 Wiki의 residual-included Transformer `layer_output`에는 held-out에서 재현되는 stable subspace가 layer 2–9 모두 존재한다.",
        "2. rank-32 기준 stable subspace는 Wiki variance의 11.4–15.6%를 담으면서 전체 drift energy의 0.32–2.05%만 담았다. stability ratio는 모든 layer에서 dimension-matched random 95% 상한보다 최소 6.4배 높다.",
        "3. 마지막 layer 9에서도 stable rank-32의 variance/drift ratio는 18.13이고 random 95% CI는 1.82–2.31이다. 따라서 마지막 layer가 빠지는 input/router-input fingerprint가 아니라 residual `layer_output`을 보존 대상으로 두는 판단이 맞다.",
        "4. 실제 mixed branch(Code LM + Wiki layer-output hidden-KL, router gradient 포함)는 full Wiki drift를 매우 작게 유지한다. layer 2/5/9 cosine은 0.9988/0.9979/0.9960이다.",
        "5. Code-only 뒤의 Wiki+Code router-only LM FT는 깨진 Wiki representation을 실질적으로 복원한다. layer 2/5/9 cosine은 0.9525/0.8871/0.8506에서 0.9969/0.9936/0.9898로 회복된다.",
        "6. **그러나 Code update가 Wiki stable subspace를 선호해서 사용한다는 증거는 없다.** Code token의 hidden update projection은 random subspace보다 작다. stable 공간의 존재와 new-task update의 자연 사용은 별도 사실이다.",
        "",
        "## 1. 비교 대상과 lineage/loss audit",
        "",
        "기준은 expert 확장 및 Wiki logits-KD init이 끝난 step-600 모델이다. 임의 기호 대신 실제 의미를 사용했다.",
        "",
        "| 조건 | checkpoint step | 실제 loss / update 구조 |",
        "|---|---:|---|",
    ]
    for key, stage in audit["stages"].items():
        lines.append(f"| {key.replace('_', ' ')} | {stage['step']} | {stage.get('loss', stage.get('meaning'))}; {stage.get('updates', '')} |")
    lines += [
        "",
        "hidden-KL production branch는 `pretrain_gpt.py::_capture_transformer_layer_outputs`로 TransformerLayer 반환값을 hook한다. 이 반환값은 attention residual과 MoE/MLP residual add가 끝난 `output`이므로 residual이 포함된다. `ffn_output`은 이번 분석에서 진단만 했고 `router_input`은 제외했다.",
        "",
        "## 2. 데이터와 통계",
        "",
        "- Wiki train corpus 10,000,000 valid tokens, Code train corpus 10,000,000 valid tokens을 각 checkpoint pair에 동일하게 사용했다.",
        "- seed 1234, sequence length 512, 동일 in-memory batch를 reference/current model에 연속 forward했다.",
        "- 각 domain을 2M-token 블록 5개로 나눴다. 앞 3개(6M)는 subspace discovery, 4번째(2M)는 validation, 5번째(2M)는 최종 test다.",
        "- 전체 activation은 저장하지 않았다. 블록별 count, mean, XX/YY/XY second moment, delta moment, drift histogram, threshold count와 layer별 4,096-token deterministic reservoir만 저장했다.",
        "- 각 sample은 global sample index, valid token sequence SHA256, valid-position SHA256로 기록했다. 세 조건의 `samples.jsonl`과 5개 block token hashes가 domain별로 완전히 일치한다.",
        f"- 무결성 검사 결과: `passed={validation['passed']}`.",
        "",
        "## 3. '변하지 않는 방향'의 정확한 정의",
        "",
        "Wiki reference hidden을 x, 비교 checkpoint hidden을 y, Δ=y−x라 두고 discovery split에서 다음 generalized eigenproblem을 푼다.",
        "",
        "`Σ_wiki u = ρ (E[ΔΔᵀ] + εI) u`",
        "",
        "ρ가 큰 방향은 Wiki variance는 크고 실제 checkpoint drift energy는 작다. ε은 layer별 `trace(E[ΔΔᵀ])/hidden_size × 1e-4`이며 수치 문제가 있을 때만 1e-2 배율로 올린다. eigenvector span을 QR로 Euclidean-orthonormalize한 뒤 held-out split에서 다시 평가했다.",
        "",
        "## 4. no-replay / no-router-FT에서의 존재 증거 (test, rank 32)",
        "",
        "| layer | stable variance | stable drift | ratio | random ratio 95% CI | 3-block overlap 최솟값 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in range(2, 10):
        row = code_only["layers"][str(layer)]
        q = row["ranks"]["32"]
        stable = q["stable"]["test"]
        ci = q["random"]["test"]["variance_to_drift_ratio"]["ci95"]
        overlap = min(x["mean_squared_cosine"] for x in row["block_reproducibility"].values())
        lines.append(
            f"| {layer} | {stable['old_variance_fraction']:.3f} | {stable['drift_energy_fraction']:.4f} | "
            f"{stable['variance_to_drift_ratio']:.2f} | {ci[0]:.2f}–{ci[1]:.2f} | {overlap:.3f} |"
        )
    lines += [
        "",
        "block overlap은 3개 discovery block에서 각각 다시 구한 rank-64 subspace와 전체 6M discovery subspace 사이의 squared canonical-cosine 평균이다. layer 2–9 최솟값이 모두 0.968 이상이므로 특정 2M block 우연으로 보기 어렵다.",
        "",
        "## 5. rank sweep과 저장량",
        "",
        "| rank | random 상한 대비 최소 ratio 우위 | Wiki variance 범위 | drift 범위 | fp16 | fp32 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rank_rows:
        lines.append(
            f"| {row['rank']} | {row['min_advantage']:.2f}× | {row['variance_min']:.3f}–{row['variance_max']:.3f} | "
            f"{row['drift_min']:.4f}–{row['drift_max']:.4f} | {row['fp16_mib']:.2f} MiB | {row['fp32_mib']:.2f} MiB |"
        )
    full = result["fingerprint_storage_bytes"]["full"]
    lines += [
        "",
        f"저장량은 8개 layer의 projection basis와 centering mean을 포함한다. full은 fp16 {mib(full['fp16']):.2f} MiB, fp32 {mib(full['fp32']):.2f} MiB다. rank 32는 유효하지만 기본값으로 확정하지 않는다. 8→256 전 구간이 random보다 유의하게 안정적이며 variance 보유량과 drift/selectivity의 trade-off가 연속적으로 변한다.",
        "",
        "## 6. mixed 학습과 router FT 복원",
        "",
        "| 조건 | layer | cosine | relative L2 | linear CKA | Procrustes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, pair in result["wiki_pairs"].items():
        for layer in (2, 5, 9):
            row = pair["layers"][str(layer)]
            drift = row["token_drift_test"]
            align = row["full_alignment_test"]
            lines.append(
                f"| {DISPLAY[label]} | {layer} | {drift['mean_cosine']:.4f} | "
                f"{drift['mean_relative_l2']:.4f} | {align['linear_cka']:.4f} | {align['normalized_procrustes']:.4f} |"
            )
    lines += [
        "",
        "router-only FT의 relative-L2 감소율(code-only 대비)은 layer 2/5/9에서 각각 "
        f"{(1-restored['layers']['2']['token_drift_test']['mean_relative_l2']/code_only['layers']['2']['token_drift_test']['mean_relative_l2'])*100:.1f}%/"
        f"{(1-restored['layers']['5']['token_drift_test']['mean_relative_l2']/code_only['layers']['5']['token_drift_test']['mean_relative_l2'])*100:.1f}%/"
        f"{(1-restored['layers']['9']['token_drift_test']['mean_relative_l2']/code_only['layers']['9']['token_drift_test']['mean_relative_l2'])*100:.1f}%다. 성능 복원이 아니라 동일 Wiki token의 hidden representation 복원을 직접 확인한 값이다.",
        "",
        "## 7. Code update vector가 Wiki stable subspace를 사용하는가",
        "",
        "Code input에서 `h_after−h_reference`를 update vector로 정의하고, Wiki code-only discovery stable basis에 대한 energy fraction을 4,096-token reservoir에서 측정했다.",
        "",
        "| layer/rank | stable projection mean | random 95% CI | stable fraction ≥0.10 | random fraction ≥0.10 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    projection = result["code_update_projection_on_wiki_stable_subspace"]["code_only_no_replay_no_router_ft"]
    for layer in (2, 5, 9):
        for rank in (32, 128, 256):
            row = projection[str(layer)][str(rank)]
            stable = row["stable"]
            random = row["random_control"]
            mean_ci = random["mean_projection_energy_fraction"]["ci95"]
            fraction_ci = random["fraction_ge_0.10"]["ci95"]
            lines.append(
                f"| L{layer}/r{rank} | {stable['mean_projection_energy_fraction']:.4f} | "
                f"{mean_ci[0]:.4f}–{mean_ci[1]:.4f} | {stable['fraction_ge_0.10']:.4f} | "
                f"{fraction_ci[0]:.4f}–{fraction_ci[1]:.4f} |"
            )
    lines += [
        "",
        "projection은 0이 아니므로 stable 공간 성분을 가진 Code update vector 자체는 존재한다. 그러나 모든 표시 조건에서 mean projection이 random 95% CI보다 낮다. 특히 높은 rank에서 `fraction ≥ 0.10`이 커지는 것은 차원 증가 효과이며, random control이 더 크다. 따라서 'Code가 Wiki stable subspace를 자연스럽게 활용한다'는 강한 주장은 기각하고, 'Code update가 stable space를 침범하지 않는 경향'으로 해석하는 편이 정확하다.",
        "",
        "## 8. layer_output과 ffn_output 진단",
        "",
        "Code-only의 layer 2/5/9에서 residual layer-output cosine은 0.953/0.887/0.851인 반면 FFN branch-only cosine은 0.564/0.760/0.793이다. residual이 포함된 최종 layer output에서 보존 가능한 공통 geometry를 정의해야 하며, FFN output만 fingerprint로 삼으면 attention/residual 경로와 마지막 layer 전체 표현을 놓친다.",
        "",
        "## 9. 판정 및 다음 단계",
        "",
        "- **존재 판정: 통과.** layer 2–9 모두 held-out stable ratio가 random 95% 상한을 크게 넘고, 독립 2M block 재현성도 높다.",
        "- **mixed 실제 구조 판정: 통과.** router gradient가 포함된 Wiki layer-output hidden-KL branch에서 representation 보존이 직접 확인된다.",
        "- **router FT 복원 판정: 통과.** Code-only 후의 Wiki representation drift가 router-only FT 뒤 크게 줄었다.",
        "- **Code update의 stable-space 선호 사용 판정: 불통과.** nonzero component는 있지만 random보다 작다.",
        "- 다음 실험으로 넘어간다면 rank를 32로 고정하지 말고 16/32/64를 우선 후보로 유지해야 한다. fingerprint KD/gating은 이번 범위에서 구현하거나 실행하지 않았다.",
        "",
        "## 10. 산출물",
        "",
        f"- 최종 수치 JSON: `{result_path}`",
        f"- lineage/loss audit: `{os.path.join(args.root, 'analysis/lineage_and_loss_audit.json')}`",
        f"- streaming validation: `{os.path.join(args.root, 'analysis/streaming_validation.json')}`",
        f"- rank-256 bases + means: `{result['basis_files']['code_only_no_replay_no_router_ft']}`",
        f"- 실행 로그: `{os.path.join(args.root, 'logs')}`",
        "- 재실행 코드: `scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh`, `scripts/analysis/analyze_layer_output_stable_subspace.py`",
        "",
        "## 제한",
        "",
        "이 분석은 이미 완료된 checkpoint의 paired forward 분석이며 causal preservation training 실험은 아니다. sample identity는 Megatron dataset이 직접 원본 document ID를 batch에 노출하지 않아 global sample index와 exact valid-token/position hash로 고정했다. 또한 Code update projection의 token-level 분포는 전체 10M이 아니라 deterministic bounded reservoir 4,096개로 추정했으며, aggregate moments와 stable-subspace 존재 검정은 전체 10M/held-out 2M을 사용했다.",
    ]
    report_path = os.path.join(report_dir, "REPORT_KO.md")
    with open(report_path + ".inprogress", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(report_path + ".inprogress", report_path)
    commands_path = os.path.join(args.root, "commands/README.md")
    commands = f"""# Reproduction commands

All checkpoint reads are evaluation-only (`--skip-train` inside the launcher). Use only a free GPU among 0–3.

```bash
GPU=0 TARGET_LABEL=code_only_no_replay_no_router_ft_after_expansion_kd_init TARGET_LOAD=/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/checkpoints/D_B_to_Code_only_no_olddata_mb48_gbs2304_step1800 DOMAIN=wiki bash scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh
GPU=1 TARGET_LABEL=code_only_no_replay_no_router_ft_after_expansion_kd_init TARGET_LOAD=/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/checkpoints/D_B_to_Code_only_no_olddata_mb48_gbs2304_step1800 DOMAIN=code bash scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh
GPU=2 TARGET_LABEL=code_lm_plus_wiki_layer_output_hidden_kl_router_gradient TARGET_LOAD=/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/02_nine_runs/local/weights/a100/mha/g2-checkpoints/code/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/r2-code-hiddenkl-c10-fixed-l2to9-post16-mb48-1800-probe3i100 DOMAIN=wiki bash scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh
GPU=3 TARGET_LABEL=code_lm_plus_wiki_layer_output_hidden_kl_router_gradient TARGET_LOAD=/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/02_nine_runs/local/weights/a100/mha/g2-checkpoints/code/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/r2-code-hiddenkl-c10-fixed-l2to9-post16-mb48-1800-probe3i100 DOMAIN=code bash scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh
GPU=0 TARGET_LABEL=code_only_then_wiki_code_router_only_lm_ft TARGET_LOAD=/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/checkpoints/CodeWiki_router_only_LM_1800_from_D DOMAIN=wiki bash scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh
GPU=1 TARGET_LABEL=code_only_then_wiki_code_router_only_lm_ft TARGET_LOAD=/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/checkpoints/CodeWiki_router_only_LM_1800_from_D DOMAIN=code bash scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh
```

The exact final analysis invocation is recoverable from `scripts/analysis/analyze_layer_output_stable_subspace.py --help`; inputs are the six semantic directories under `{args.root}/streaming_stats`. Per-run stdout/stderr is under `{args.root}/logs`.
"""
    with open(commands_path + ".inprogress", "w", encoding="utf-8") as handle:
        handle.write(commands)
    os.replace(commands_path + ".inprogress", commands_path)
    print(report_path)


if __name__ == "__main__":
    main()
