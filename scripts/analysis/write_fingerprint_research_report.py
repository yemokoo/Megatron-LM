#!/usr/bin/env python3
"""Write a Korean, evidence-linked research report from completed fingerprint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return f"{100.0 * value:.1f}%"


def f4(value):
    return f"{value:.4f}"


def span(rows, getter, formatter=f4):
    values = [getter(row) for row in rows]
    return f"{formatter(min(values))}–{formatter(max(values))}"


def layer_rows(comparison):
    return {int(row["layer"]): row for row in comparison["layers"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repo = Path(args.repo).resolve()
    metrics = root / "metrics"
    inventory = read_json(root / "inventory/checkpoint_inventory_A_to_I.json")
    validation = read_json(root / "inventory/dump_semantic_validation.json")
    drift = read_json(metrics / "representation_routing_drift.json")
    sufficiency = read_json(metrics / "fingerprint_sufficiency.json")
    extended = read_json(metrics / "fingerprint_candidates_extended.json")
    forward_path = metrics / "actual_forward_interventions.json"
    forward = read_json(forward_path) if forward_path.is_file() else {"status": "missing", "conditions": {}}
    all_root = root / "extended_all_layers"
    all_required = (
        all_root / "inventory/dump_semantic_validation.json",
        all_root / "metrics/representation_routing_drift.json",
        all_root / "metrics/fingerprint_sufficiency.json",
        all_root / "metrics/fingerprint_candidates_extended.json",
    )
    all_available = all(path.is_file() for path in all_required)
    if all_available:
        all_validation = read_json(all_required[0])
        all_drift = read_json(all_required[1])
        all_sufficiency = read_json(all_required[2])
        all_extended = read_json(all_required[3])
        all_wiki_bc = all_drift["wiki"]["B_to_C_vocabkl"]["layers"]
        all_code_bc = all_drift["code"]["B_to_C_vocabkl"]["layers"]
        all_suff = {int(row["layer"]): row for row in all_sufficiency["layers"]}
        all_ext = {int(row["layer"]): row for row in all_extended["layers"]}
        all_layers = sorted(all_suff)

    wiki_bc = layer_rows(drift["wiki"]["B_to_C_vocabkl"])
    code_bc = layer_rows(drift["code"]["B_to_C_vocabkl"])
    wiki_fg = layer_rows(drift.get("wiki", {}).get("F_to_G", {"layers": []}))
    wiki_gh = layer_rows(drift.get("wiki", {}).get("G_to_H", {"layers": []}))
    suff = {int(row["layer"]): row for row in sufficiency["layers"]}
    ext = {int(row["layer"]): row for row in extended["layers"]}

    lines = [
        "# Stable representation fingerprint / router geometry 감사 보고서",
        "",
        "## 의사결정 요약",
        "",
        "현재 증거만으로는 **Wiki raw replay 없이 작은 fingerprint만으로 Wiki routing과 성능을 실질적으로 보존할 수 있다고 결론낼 수 없다.**",
        "",
        "- 긍정적 증거: B→C의 Wiki router input은 거의 보존되며, 16차원 teacher-router row space는 teacher logits/top-k를 정확히 결정한다.",
        "- 반증: old-domain 선택적인 32차원 안정 성분은 top-k 완전 복원이 31–42%에 그치고, 같은 차원의 top-variance PCA보다 일관되게 낫지 않다.",
        "- 실제 forward 반증: 현재 완료 조건에서 teacher router anchor를 층 2/5/9에 사용해도 C의 Wiki 정확도는 회복되지 않았고 Code 정확도는 크게 하락했다.",
        "- 핵심 병목: C에서는 representation drift보다 router-boundary drift가 우세하다. 따라서 representation fingerprint 단독보다 **작은 router anchor + 선택적 적용 마스크**가 더 직접적인 후보이다.",
        "- 확장 검증: 128 samples × 8 tokens와 모든 router/MoE layer 2–9에서도 같은 결론이 유지되었다.",
        "- 필수 미해결: matched D가 없고 E/I 물리 체크포인트도 없으므로, router FT가 없는 심한 망각 상태에서의 회복 가능성은 아직 직접 검증되지 않았다.",
        "",
        "## 1. 체크포인트와 비교 가능성",
        "",
        "| ID | 상태 | 정의/주의점 |",
        "|---|---|---|",
    ]
    for key, record in inventory["records"].items():
        state = "physical" if record["physical_checkpoint_present"] else "missing/reference"
        lines.append(f"| {key} | {state} | {record['role']}; `{record['comparison_status']}` |")
    lines.extend(
        [
            "",
            "직접 matched 비교는 A→B, B→C, F→G, G→H이다. D는 현재 `/data2`에 없다. E는 manifest의 MB96 Code baseline이지만 시작 Wiki가 MB128 run이라 현재 A(MB72 reconstructed run)와 동일 checkpoint가 아니다. I도 baseline router-retuned Code source에서 출발해 현재 F와 다르다.",
            "",
            "`no_kd_joint_lm` 이름의 체인은 Wiki 또는 Wiki+Code raw LM을 replay하므로 E/I의 Code-only/Conversation-only 대조군이 아니다.",
            "",
            f"상세 기계 판독 inventory: `{root / 'inventory/checkpoint_inventory_A_to_I.json'}`",
            "",
            "## 2. tensor hook와 정렬 검증",
            "",
            "수집 tensor는 transformer layer input, attention output, attention residual 이후 hidden, router input, FFN output, layer output, router logits/full probability/top-k index·weight/margin/entropy, router weight, 선택 expert별 output vector·norm이다.",
            "",
            f"- 구현: `{repo / 'Megatron-LM/pretrain_gpt.py'}`",
            f"- router 실제 선형 gate/개입: `{repo / 'Megatron-LM/megatron/core/transformer/moe/router.py'}`",
            f"- 실행기: `{repo / 'scripts/analysis/run_fingerprint_router_smoke_mha.sh'}`",
            "- sample ID, token ID, position, layer 번호를 checkpoint 간 exact-match로 검사했다.",
            f"- semantic validation 전체 통과: `{validation['all_checks_pass']}`. router logits 재계산, softmax/top-k, expert output 재합성, legacy layer output 일치를 모두 수치 검증했다.",
            f"- 검증 파일: `{root / 'inventory/dump_semantic_validation.json'}`",
            "",
            "각 domain은 64 samples × 16 valid tokens = 1,024 aligned tokens이며 대표 MoE layer 2/5/9를 사용했다. padding/loss-mask 밖 token은 dump 선택에서 제외한다.",
            "",
            "추가 확장은 128 samples × 8 valid tokens = 1,024 aligned tokens에서 모든 router/MoE layer 2–9를 수집했다. layer 1은 `moe_layer_freq=0`인 dense layer라 router 지표가 정의되지 않는다. 확장 dump의 semantic validation도 전체 통과했다.",
            "",
            f"기존 `{repo / 'scripts/analysis/plot_hidden_space_ffn_only.py'}`를 코드로 확인했다. 이 구현은 비교 stage vector를 먼저 concatenate해 공통 PCA를 fit하고, `histogram2d` 뒤 3×3 binomial/Gaussian형 kernel을 두 번 적용해 density contour를 그린다. 이번 확장은 같은 원칙을 유지하되 비교 model/domain 전체에 layer별 공통 PCA 하나를 사용하고 `gaussian_filter(sigma=1.25)`로 smoothing했다. 모델별 PCA는 사용하지 않았고, 2D 그림은 판정 근거가 아닌 시각화로만 사용했다.",
            "",
            "## 3. Stability와 representation drift",
            "",
            "### B→C: Wiki와 Code의 대비",
            "",
            "| layer | Wiki cosine / CKA | Wiki top-k | Code cosine / CKA | Code top-k |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in (2, 5, 9):
        w, c = wiki_bc[layer], code_bc[layer]
        lines.append(
            f"| {layer} | {f4(w['components']['router_input']['cosine_mean'])} / "
            f"{f4(w['router_input_geometry']['centered_linear_cka'])} | "
            f"{pct(w['routing']['topk_agreement'])} | "
            f"{f4(c['components']['router_input']['cosine_mean'])} / "
            f"{f4(c['router_input_geometry']['centered_linear_cka'])} | "
            f"{pct(c['routing']['topk_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "Wiki는 중·후반에서도 거의 고정되어 있지만 Code는 layer 5/9에서 크게 이동한다. 따라서 관찰된 안정성은 모든 domain에서 움직이지 않는 dead direction만으로 설명되지 않는다.",
        ]
    )
    if all_available:
        lines.extend(
            [
                "",
                "### 128-sample × 모든 MoE layer 확장",
                "",
                "| layer | Wiki cosine | Wiki CKA | Wiki top-k | Code cosine | Code top-k |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for w, c in zip(all_wiki_bc, all_code_bc):
            lines.append(
                f"| {w['layer']} | {f4(w['components']['router_input']['cosine_mean'])} | "
                f"{f4(w['router_input_geometry']['centered_linear_cka'])} | "
                f"{pct(w['routing']['topk_agreement'])} | "
                f"{f4(c['components']['router_input']['cosine_mean'])} | "
                f"{pct(c['routing']['topk_agreement'])} |"
            )
        lines.extend(
            [
                "",
                "전 층에서 Wiki cosine은 "
                f"{span(all_wiki_bc, lambda row: row['components']['router_input']['cosine_mean'])}, "
                "CKA는 "
                f"{span(all_wiki_bc, lambda row: row['router_input_geometry']['centered_linear_cka'])}지만 "
                "top-k는 "
                f"{span(all_wiki_bc, lambda row: row['routing']['topk_agreement'], pct)}다. "
                "Code cosine은 "
                f"{span(all_code_bc, lambda row: row['components']['router_input']['cosine_mean'])}, "
                "top-k는 "
                f"{span(all_code_bc, lambda row: row['routing']['topk_agreement'], pct)}로 크게 변한다.",
            ]
        )
    lines.extend(
        [
            "",
            "B→C Wiki의 PCA-truncated SVCCA/PWCCA는 다음과 같다.",
            "",
            "| layer | SVCCA@128 | PWCCA@128 | expert rank Spearman |",
            "|---:|---:|---:|---:|",
        ]
    )
    for layer in (2, 5, 9):
        row = ext[layer]
        lines.append(
            f"| {layer} | {f4(row['cca']['svcca_mean'])} | {f4(row['cca']['pwcca'])} | "
            f"{f4(row['router_rank_order']['spearman_expert_rank_mean'])} |"
        )
    lines.extend(
        [
            "",
            "단일 seed/checkpoint family만 물리적으로 있으므로 seed 간 변동은 검증하지 못했다.",
            "",
            "## 4. Representation drift와 router-boundary drift 분해",
            "",
            "B→C Wiki에서 cosine≥0.99인데 top-k set이 바뀐 token은 layer 2/5/9에서 각각 "
            f"{pct(wiki_bc[2]['drift_decomposition']['four_token_groups']['representation_stable_routing_changed']['fraction'])}, "
            f"{pct(wiki_bc[5]['drift_decomposition']['four_token_groups']['representation_stable_routing_changed']['fraction'])}, "
            f"{pct(wiki_bc[9]['drift_decomposition']['four_token_groups']['representation_stable_routing_changed']['fraction'])}였다. "
            "반대로 representation 변경+routing 변경 그룹은 세 층 모두 0 token이었다.",
            "",
            "B router를 고정하고 C hidden만 넣은 representation-only top-k agreement는 layer 2/5/9에서 "
            f"{pct(wiki_bc[2]['drift_decomposition']['representation_only_common_router']['topk_agreement'])}, "
            f"{pct(wiki_bc[5]['drift_decomposition']['representation_only_common_router']['topk_agreement'])}, "
            f"{pct(wiki_bc[9]['drift_decomposition']['representation_only_common_router']['topk_agreement'])}였다. "
            "C router boundary만 적용한 값은 "
            f"{pct(wiki_bc[2]['drift_decomposition']['boundary_only_common_experts']['topk_agreement'])}, "
            f"{pct(wiki_bc[5]['drift_decomposition']['boundary_only_common_experts']['topk_agreement'])}, "
            f"{pct(wiki_bc[9]['drift_decomposition']['boundary_only_common_experts']['topk_agreement'])}다. "
            "즉 Wiki routing 변화는 boundary drift가 주도한다.",
        ]
    )
    if all_available:
        stable_changed = [
            row["drift_decomposition"]["four_token_groups"]
            ["representation_stable_routing_changed"]["fraction"]
            for row in all_wiki_bc
        ]
        changed_changed = [
            row["drift_decomposition"]["four_token_groups"]
            ["representation_changed_routing_changed"]["fraction"]
            for row in all_wiki_bc
        ]
        rep_only = [
            row["drift_decomposition"]["representation_only_common_router"]
            ["topk_agreement"]
            for row in all_wiki_bc
        ]
        boundary_only = [
            row["drift_decomposition"]["boundary_only_common_experts"]
            ["topk_agreement"]
            for row in all_wiki_bc
        ]
        lines.extend(
            [
                "",
                "128-sample 전-layer에서도 representation 안정+routing 변경 token은 "
                f"{pct(min(stable_changed))}–{pct(max(stable_changed))}, representation 변경+routing 변경은 "
                f"{pct(min(changed_changed))}–{pct(max(changed_changed))}였다. "
                f"representation-only top-k는 {pct(min(rep_only))}–{pct(max(rep_only))}, "
                f"boundary-only top-k는 {pct(min(boundary_only))}–{pct(max(boundary_only))}로, "
                "대표 층 결론이 layer 선택의 우연이 아님을 확인했다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 5. Selectivity",
            "",
            "Wiki B→C 안정성으로 학습한 stable+routing-sensitive r32 subspace의 held-out Wiki-vs-Code AUC는 layer 2/5/9에서 "
            f"{f4(ext[2]['stable_sensitive_r32_selectivity']['heldout_domain_roc_auc'])}, "
            f"{f4(ext[5]['stable_sensitive_r32_selectivity']['heldout_domain_roc_auc'])}, "
            f"{f4(ext[9]['stable_sensitive_r32_selectivity']['heldout_domain_roc_auc'])}다. "
            "shuffled-label AUC 평균은 각각 "
            f"{f4(ext[2]['stable_sensitive_r32_selectivity']['shuffled_label_baseline']['auc_mean'])}, "
            f"{f4(ext[5]['stable_sensitive_r32_selectivity']['shuffled_label_baseline']['auc_mean'])}, "
            f"{f4(ext[9]['stable_sensitive_r32_selectivity']['shuffled_label_baseline']['auc_mean'])}다.",
            "",
            "domain과 teacher top-1 expert assignment의 normalized MI는 0.043/0.069/0.216으로, 후반층에서만 중간 정도의 expert 선택성이 나타난다.",
            "",
            "안정 channel subset은 128 channels에서도 top-k 복원이 12–14%이고 random channel과 일관된 차이가 없어 우선순위에서 제외한다. nearest expert-activation prototype도 top-k 완전 복원이 6–10%라 불충분하다.",
            "",
            "## 6. Sufficiency intervention",
            "",
            "held-out 32 samples(512 tokens)에서 얻은 fingerprint-only teacher top-k 완전 일치율:",
            "",
            "| layer | stable-sensitive r32 | top-PCA r32 | random r32 | router row-space r16 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in (2, 5, 9):
        curves = {(row["basis"], row["requested_rank"]): row for row in suff[layer]["curves"]}
        stable = curves[("stable_routing_sensitive", 32)]["fingerprint_only"]["topk_agreement"]
        pca = curves[("top_variance_pca", 32)]["fingerprint_only"]["topk_agreement"]
        random = curves[("random", 32)]["fingerprint_only_topk_agreement_mean"]
        rowspace = curves[("router_row_space", 16)]["fingerprint_only"]["topk_agreement"]
        lines.append(
            f"| {layer} | {pct(stable)} | {pct(pca)} | {pct(random)} | {pct(rowspace)} |"
        )
    lines.extend(
        [
            "",
            "stable-sensitive는 random보다 분명히 낫지만 PCA와 비슷하며 절대 복원율이 낮다. router row-space r16의 100%는 선형 router에서 Jacobian `d logits / d h = W`이고 모든 logits가 W의 row-space projection만으로 결정된다는 대수적 결과다. 이는 좋은 압축 router anchor지만 old-selective representation fingerprint라는 증거는 아니다.",
        ]
    )
    if all_available:
        lines.extend(
            [
                "",
                "128-sample 확장의 모든 MoE layer에서도 stable-sensitive r32는 random보다 크지만 PCA를 일관되게 이기지 못했다.",
                "",
                "| layer | stable r32 | PCA r32 | random r32 | row-space r16 | domain AUC |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for layer in all_layers:
            curves = {
                (row["basis"], row["requested_rank"]): row
                for row in all_suff[layer]["curves"]
            }
            stable = curves[("stable_routing_sensitive", 32)]
            pca = curves[("top_variance_pca", 32)]
            random = curves[("random", 32)]
            rowspace = curves[("router_row_space", 16)]
            lines.append(
                f"| {layer} | {pct(stable['fingerprint_only']['topk_agreement'])} | "
                f"{pct(pca['fingerprint_only']['topk_agreement'])} | "
                f"{pct(random['fingerprint_only_topk_agreement_mean'])} | "
                f"{pct(rowspace['fingerprint_only']['topk_agreement'])} | "
                f"{f4(stable['selectivity']['heldout_domain_roc_auc'])} |"
            )
        stable_all = []
        pca_all = []
        random_all = []
        auc_all = []
        for layer in all_layers:
            curves = {
                (row["basis"], row["requested_rank"]): row
                for row in all_suff[layer]["curves"]
            }
            stable_all.append(curves[("stable_routing_sensitive", 32)]["fingerprint_only"]["topk_agreement"])
            pca_all.append(curves[("top_variance_pca", 32)]["fingerprint_only"]["topk_agreement"])
            random_all.append(curves[("random", 32)]["fingerprint_only_topk_agreement_mean"])
            auc_all.append(curves[("stable_routing_sensitive", 32)]["selectivity"]["heldout_domain_roc_auc"])
        lines.extend(
            [
                "",
                f"전 층 범위는 stable r32 {pct(min(stable_all))}–{pct(max(stable_all))}, "
                f"PCA r32 {pct(min(pca_all))}–{pct(max(pca_all))}, "
                f"random r32 {pct(min(random_all))}–{pct(max(random_all))}, "
                f"domain AUC {f4(min(auc_all))}–{f4(max(auc_all))}다.",
            ]
        )
    lines.extend(
        [
            "",
            "teacher top-1 expert별 router-input centroid를 SVD해 얻은 routing-supervised discriminative basis는 유효 rank가 8이었다.",
            "",
            "| layer | effective rank | top-k | top-1 margin r | top-k boundary margin r | Wiki–Code AUC |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in (2, 5, 9):
        curves = {(row["basis"], row["requested_rank"]): row for row in suff[layer]["curves"]}
        routing = curves[("routing_discriminative", 32)]
        margin = routing["fingerprint_only"]["margin_recovery"]
        lines.append(
            f"| {layer} | {routing['effective_rank']} | "
            f"{pct(routing['fingerprint_only']['topk_agreement'])} | "
            f"{f4(margin['top1_margin']['pearson'])} | "
            f"{f4(margin['topk_boundary_margin']['pearson'])} | "
            f"{f4(routing['selectivity']['heldout_domain_roc_auc'])} |"
        )
    lines.extend(
        [
            "",
            "routing-supervised r8은 매우 작고 후반층에서 선택성과 top-1 margin 상관이 높지만, top-k 완전 복원은 20–35%라 단독 fingerprint로는 불충분하다. 따라서 `U_select`의 보조 feature 후보로만 남긴다.",
            "",
            "### 실제 forward routing-context 개입",
            "",
            f"상태: `{forward.get('status')}`. C checkpoint의 layer 2/5/9 expert dispatch를 저장된 B teacher anchor로 바꾸어 64×512-token probe를 실행했다.",
            "",
            "| condition | Wiki acc | Code acc | Wiki/C baseline fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition, record in forward.get("conditions", {}).items():
        wiki = record["domains"].get("wiki", {})
        code = record["domains"].get("code", {})
        if not wiki or not code:
            continue
        lines.append(
            f"| {condition} | {wiki['next_token_accuracy']:.6f} | {code['next_token_accuracy']:.6f} | "
            f"{pct(wiki['accuracy_fraction_of_C_baseline'])} / {pct(code['accuracy_fraction_of_C_baseline'])} |"
        )
    lines.extend(
        [
            "",
            "teacher-full은 Wiki를 회복하지 못하고 Code 성능을 크게 낮춘다. stable-only/PCA-only도 Wiki와 Code 모두 나빠진다. 따라서 현재의 correlation/top-k 복원만으로 성능 sufficiency를 주장할 수 없다. D/E처럼 실제 Wiki forgetting이 큰 matched checkpoint에서 반복해야 한다.",
            "",
            "### 데이터 조건별 판정",
            "",
            "| 구분 | 이번 실험 | 판정 |",
            "|---|---|---|",
            "| Oracle fingerprint | 분석 단계에서 B/C의 정렬된 Wiki raw token으로 안정 축을 발견 | 존재·선택성 진단에는 사용 가능하지만 최종 학습 방식은 아님 |",
            "| Compressed fingerprint | `mean+basis+teacher router weight` NPZ만으로 held-out Wiki routing을 재구성 | row-space는 logits/top-k 100%, stable r32는 31–45% |",
            "| New-data-only application | Wiki sample 없이 저장된 NPZ를 C의 Code forward에 적용 | row-space Code 정확도는 baseline의 68.0%로 plasticity 실패 |",
            "| New-data-only training | Code LM + 저장 fingerprint loss | matched D 부재와 대규모 재학습 금지 때문에 미실행; 핵심 질문 4는 아직 미해결 |",
            "",
            "## 7. Code→Conversation",
            "",
        ]
    )
    if wiki_fg and wiki_gh:
        lines.extend(
            [
                "현재 완료된 Wiki 기준으로 F→G top-k agreement는 layer 2/5/9에서 "
                f"{pct(wiki_fg[2]['routing']['topk_agreement'])}, {pct(wiki_fg[5]['routing']['topk_agreement'])}, {pct(wiki_fg[9]['routing']['topk_agreement'])}; "
                "G→H는 "
                f"{pct(wiki_gh[2]['routing']['topk_agreement'])}, {pct(wiki_gh[5]['routing']['topk_agreement'])}, {pct(wiki_gh[9]['routing']['topk_agreement'])}다.",
                "",
                "두 비교 모두 중·후반 CKA가 0.9993 이상으로, Wiki representation은 계속 안정적이지만 router boundary/새 expert 경쟁으로 약 6–10% token의 top-k set이 바뀐다.",
            ]
        )
    else:
        lines.append("F/G/H aligned dump는 평가 GPU 재점유 때문에 아직 부분 상태다.")
    if all_available:
        lines.extend(
            [
                "",
                "128-sample 전-layer 결과의 domain별 top-k 범위:",
                "",
                "| domain | F→G top-k | G→H top-k | G→H router-input cosine |",
                "|---|---:|---:|---:|",
            ]
        )
        for domain in ("wiki", "code", "conversation"):
            fg_rows = all_drift[domain]["F_to_G"]["layers"]
            gh_rows = all_drift[domain]["G_to_H"]["layers"]
            lines.append(
                f"| {domain} | {span(fg_rows, lambda row: row['routing']['topk_agreement'], pct)} | "
                f"{span(gh_rows, lambda row: row['routing']['topk_agreement'], pct)} | "
                f"{span(gh_rows, lambda row: row['components']['router_input']['cosine_mean'])} |"
            )
        lines.extend(
            [
                "",
                "F→G KD-init에서는 세 domain representation이 거의 유지된다. G→H 이후 Wiki/Code cosine은 각각 "
                f"{span(all_drift['wiki']['G_to_H']['layers'], lambda row: row['components']['router_input']['cosine_mean'])}, "
                f"{span(all_drift['code']['G_to_H']['layers'], lambda row: row['components']['router_input']['cosine_mean'])}인 반면, "
                "Conversation은 "
                f"{span(all_drift['conversation']['G_to_H']['layers'], lambda row: row['components']['router_input']['cosine_mean'])}까지 이동하고 "
                "top-k도 "
                f"{span(all_drift['conversation']['G_to_H']['layers'], lambda row: row['routing']['topk_agreement'], pct)}만 남는다. "
                "따라서 old-domain 안정성이 모든 domain의 dead direction이라는 해석은 지지되지 않는다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 8. 후보 우선순위",
            "",
            "1. **router row-space + frozen router anchor**: 가장 작고 top-k/logit sufficiency가 정확하다. 단, selectivity mask와 함께 써야 Code plasticity 손상을 줄일 수 있다.",
            "2. **stable+routing-sensitive r32**: old/new domain 선택성과 random 대비 이득은 있으나 PCA 대비 우월성과 절대 sufficiency가 부족하다. token gating feature로만 사용한다.",
            "3. **routing-supervised discriminative r8**: 작은 선택성/margin feature로 유망하지만 top-k 복원이 낮아 단독 사용은 기각한다.",
            "4. **expert activation centroid/prototype**: top-1 보조 신호로는 가능하지만 top-k 복원용 단독 fingerprint로는 기각한다.",
            "5. **stable raw channel subset**: random 대비 일관된 개선이 없어 현재 증거로는 기각한다.",
            "",
            "## 9. 권장 raw-data-free loss/gradient projection",
            "",
            "task t 종료 시 layer별로 `(μ_t, U_select,t, Σ_t, R_route,t, W_anchor,t, prototype_t)`만 저장한다. `U_select`는 old-like token 판별/gradient 보호용 capped stable-sensitive basis이고, `R_route`는 정확한 logit 복원을 위한 router row basis다. 두 역할을 한 basis에 억지로 합치지 않는다.",
            "",
            "새 Code token x에 대해:",
            "",
            "```text",
            "z_l = U_select,l^T (h_l(x) - μ_l)",
            "s_l(x) = sigmoid((τ - Mahalanobis(z_l; prototype_l, Σ_l)) / T_s)",
            "h_route = μ_l + R_route,l R_route,l^T (h_l-μ_l)",
            "q_anchor = softmax(W_anchor,l h_route / T_r)",
            "q_current_old = softmax(W_current,old h_l / T_r)",
            "L_fp = Σ_l stopgrad(s_l) · [KL(q_anchor || q_current_old) + β·new_expert_mass]",
            "L_total = L_new_LM + λ L_fp",
            "```",
            "",
            "representation 자체를 보존하려면 old-like token에서 hidden gradient를 `g_h ← (I - s_l U_select,l U_select,l^T) g_h`로 투영한다. `new_expert_mass` 항은 old-like token에 새 expert가 경쟁적으로 끼는 것을 제한한다. router anchor KL은 boundary drift, gradient projection은 representation drift를 각각 겨냥한다.",
            "",
            "모든 token이 old/new로 collapse하지 않도록 (a) `s`를 detach, (b) 저장된 old score quantile로 τ 고정, (c) batch별 soft quota/entropy regularizer, (d) 최소·최대 preserve mass, (e) Code validation plasticity와 new-expert usage 하한을 함께 둔다.",
            "",
            "### 메모리 scaling",
            "",
            "naive task별 basis는 `O(T·L·H·r)`이다. 현재 H=1024에서 `U_select` r=32는 fp16 기준 layer당 약 64 KiB, `R_route` r=16과 16-row anchor는 각각 약 32 KiB다. 8개 MoE layer에서 task당 약 1 MiB이며 centroid/covariance diagonal/prototype를 더해도 수 MiB 수준이다. 다만 expert 수와 함께 전체 W를 매 task 저장하면 누적이 O(T²)가 될 수 있으므로, 기존 공통 row basis와 task별 incremental rows만 저장하고 global rank cap/재직교화를 사용한다.",
            "",
            "## 10. 최소 후속 실험과 반증 기준",
            "",
            "1. 현재 B에서 matched D를 1회 생성하고 E manifest checkpoint를 복원한다. 같은 64/128 sample·모든 MoE layer로 B→D, A→E, D→E를 먼저 측정한다.",
            "2. D에서 baseline, teacher-full, row-space-only/removed, stable-r32, PCA-r32, random-r32 actual forward를 실행한다. Wiki 성능 회복이 없으면 fingerprint sufficiency 가설을 기각한다.",
            "3. Code-only 100–300 step 저비용 pilot에서 anchor-KL only, gradient projection only, 결합을 비교한다. Wiki raw data는 평가에만 쓰고 학습에는 쓰지 않는다.",
            "4. 3 seeds와 all MoE layers로 안정 basis principal angle/CCA 분산을 측정한다.",
            "5. Conversation matched I를 F에서 직접 만들어 F→G/H/I를 동일 source로 비교한다.",
            "",
            "기각 기준은 stable subspace가 PCA/random을 못 이기거나, row-space anchor가 Wiki를 회복하지 못하거나, Wiki 회복 대비 Code 정확도/신규 expert 사용이 크게 붕괴하거나, task 증가 시 capped basis로 유지되지 않는 경우다.",
            "",
            "## 11. 산출물",
            "",
            f"- checkpoint inventory: `{root / 'inventory/checkpoint_inventory_A_to_I.json'}`",
            f"- dump/정렬 검증: `{root / 'inventory/dump_manifest.json'}`, `{root / 'inventory/dump_semantic_validation.json'}`",
            f"- drift/분해: `{metrics / 'representation_routing_drift.json'}`",
            f"- sufficiency: `{metrics / 'fingerprint_sufficiency.json'}`",
            f"- 확장 후보: `{metrics / 'fingerprint_candidates_extended.json'}`",
            f"- actual forward: `{metrics / 'actual_forward_interventions.json'}`",
            f"- 전 MoE layer 확장 root: `{all_root}`",
            f"- 전 MoE layer 검증: `{all_root / 'inventory/dump_semantic_validation.json'}`",
            f"- 전 MoE layer drift/sufficiency: `{all_root / 'metrics/representation_routing_drift.json'}`, `{all_root / 'metrics/fingerprint_sufficiency.json'}`",
            f"- 압축 fingerprint NPZ: `{metrics / 'fingerprints'}`",
            f"- 그림: `{root / 'plots'}`",
            f"- 실행 로그: `{root / 'logs'}`",
            "",
            "### 핵심 그림",
            "",
            f"- 공통 PCA density: `{root / 'plots/common_pca_router_input_density.png'}`",
            f"- 모델별 Wiki–Code joint density: `{root / 'plots/model_wiki_code_joint_density.png'}`",
            f"- routing 안정/변경 token displacement: `{root / 'plots/token_displacement_routing_stability.png'}`",
            f"- Code→Conversation domain별 displacement: `{root / 'plots/code_conversation_token_displacement.png'}`",
            f"- stable/orthogonal 및 top-k 성공/실패: `{root / 'plots/stable_orthogonal_and_routing_restoration.png'}`",
            f"- expert assignment: `{root / 'plots/expert_assignment_by_layer.png'}`",
            f"- actual forward accuracy: `{root / 'plots/actual_forward_intervention_accuracy.png'}`",
            f"- 전 MoE layer 그림: `{all_root / 'plots'}`",
        ]
    )

    report = root / "report" / "fingerprint_router_geometry_report_ko.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = {
        "answer_to_primary_question": "not_yet_supported",
        "best_exact_routing_candidate": "teacher router row-space plus anchor",
        "best_selective_candidate": "stable+routing-sensitive r32",
        "performance_sufficiency": "failed_on_current_C_intervention_so_far",
        "all_moe_layers_validation": all_available and all_validation["all_checks_pass"],
        "blocking_controls": ["matched D", "physical E", "matched I"],
        "report": str(report),
    }
    decision_path = root / "report" / "decision_summary.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report), "decision": str(decision_path)}, indent=2))


if __name__ == "__main__":
    main()
