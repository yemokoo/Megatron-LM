#!/usr/bin/env python3
"""Build an evidence-linked A--I checkpoint inventory for fingerprint analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--source-root",
        default="/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808",
    )
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def checkpoint_facts(path: Path, expected_step: int):
    tracker = path / "latest_checkpointed_iteration.txt"
    actual = tracker.read_text().strip() if tracker.is_file() else None
    shards = sorted(path.glob("iter_*/mp_rank_*/model_optim_rng.pt"))
    if not shards:
        shards = sorted(path.glob("iter_*/mp_rank_*_model_states.pt"))
    if not shards:
        shards = sorted(path.glob("iter_*/*.distcp"))
    return {
        "path": str(path),
        "physical_checkpoint_present": path.is_dir() and actual is not None,
        "expected_step": expected_step,
        "actual_step": int(actual) if actual and actual.isdigit() else actual,
        "step_matches": actual == str(expected_step),
        "tensor_shard_count": len(shards),
        "tensor_shard_bytes": sum(item.stat().st_size for item in shards),
    }


def metadata(path: Path):
    meta = path / "logs" / "run_metadata.json"
    return json.loads(meta.read_text()) if meta.is_file() else None


def main():
    args = args_parser()
    repo = Path(args.repo).resolve()
    root = Path(args.source_root).resolve()
    out = Path(args.out_root).resolve() / "inventory"
    out.mkdir(parents=True, exist_ok=True)

    a = root / "00_sources/wiki_ffn_only_e8_step1800"
    b = root / "01_common_kd_init/code_e8_to_e16_wiki_kd_step600"
    c = root / (
        "02_nine_runs/local/weights/a100/mha/g2-checkpoints/code/joint_old_data_kd/"
        "post_kd_teacher_c10_fixed/r2-code-vocabkl-c10-fixed-post16-mb48-1800-probe3i100"
    )
    g = root / (
        "02_nine_runs/local/weights/a100/mha/g2-checkpoints/conversation/"
        "expansion_distill_init_3objective_c10_l2to9/vocab_kl/"
        "r3-conv-expand-outputkd-c1-vocabkl-branch-e16to24-mb32-600-probe3i100"
    )
    h = root / (
        "02_nine_runs/local/weights/a100/mha/g2-checkpoints/conversation/joint_old_data_kd/"
        "post_kd_teacher_c10_fixed/r4-conv-vocabkl-c10-fixed-post24-mb36-1800-probe3i100"
    )
    d_expected = root / "missing_controls/D_code_from_B_no_router_ft"
    e_manifest = (
        "/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/"
        "g2matched-ffn-moe-attn-freeze-bf16/"
        "g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-freeze-mha-a100-bf16-mb96-1800"
    )
    i_manifest = (
        "/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning/.local/weights/a100/mha/"
        "g2-checkpoints/conversation/phase4/"
        "g2-ffn-only-phase4-conversation-from-router-retuned-e16to24-mb48-1800"
    )

    common_model = {
        "layers": 9,
        "hidden_size": 1024,
        "dense_ffn_hidden_size": 5472,
        "moe_ffn_hidden_size": 352,
        "top_k": 4,
        "precision": "bf16",
        "global_batch_size": 2304,
        "learning_rate": 3e-4,
        "minimum_learning_rate": 3e-5,
        "lr_schedule": "WSD",
    }
    evidence = {
        "chain": str(repo / "scripts/experiment/a100/run_g2_9stage_old_replay_3objective_c10_l2to9_postkd_chain_mha.sh"),
        "d_recipe": str(repo / "scripts/experiment/a100/run_g2_ffn_only_code_from_distill_init_mha.sh"),
        "e_recipe": str(repo / "scripts/experiment/a100/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh"),
        "manifest": str(repo / "scripts/hf/final_checkpoint_manifest.tsv"),
        "conversation_baseline_recipe": str(repo / "scripts/experiment/a100/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh"),
        "joint_no_kd_not_code_only": str(repo / "scripts/experiment/a100/run_g2_ffn_only_no_kd_joint_lm_code_conversation_chain_mha.sh"),
    }

    records = {
        "A": {
            **checkpoint_facts(a, 1800),
            "role": "Wiki completion before Code expansion",
            "comparison_status": "direct_physical",
            "experts": "8",
            "data": "Wiki exact",
            "micro_batch_size": 72,
            "trainable_parameters": "Wiki pretraining parameters; no expansion freeze boundary",
            "metadata": metadata(a),
            "caveat": "Reconstructed provenance for the current physical checkpoint; tensors unchanged.",
        },
        "B": {
            **checkpoint_facts(b, 600),
            "role": "8->16 expansion plus Wiki output-logit KD initialization",
            "comparison_status": "direct_physical",
            "source": str(a),
            "experts": "8->16",
            "data": "Wiki exact only",
            "micro_batch_size": 36,
            "objective": "teacher vocabulary KL coefficient 1; LM/aux/z disabled by recipe",
            "trainable_parameters": "new experts 8:16 and new router rows 8:16; shared/attention/old experts/old router rows frozen",
            "metadata": metadata(b),
            "expansion_audit": str(b / "expansion_audit/expand_8_to_16.json"),
        },
        "C": {
            **checkpoint_facts(c, 1800),
            "role": "normal Code phase with per-update old Wiki router FT via vocabulary KL",
            "comparison_status": "direct_physical_canonical_vocab_kl",
            "source": str(b),
            "experts": "16 (already expanded)",
            "data": "Code LM + Wiki replay, equal-dataset scheduling",
            "micro_batch_size": 48,
            "objective": "Code LM + Wiki teacher vocabulary KL coefficient 10",
            "trainable_parameters": "new experts 8:16 plus all router rows; replay gradients are router-only",
            "metadata": metadata(c),
            "log": str(root / "03_logs/nine_stages/vocab_kl/r2.log"),
            "physical_variants": {
                "hidden_kl_c10_layers_2_9": "present",
                "hidden_mse_c10_layers_2_9": "present",
            },
        },
        "D": {
            **checkpoint_facts(d_expected, 1800),
            "role": "B-initialized Code-only phase without old replay/router FT",
            "comparison_status": "missing_matched_checkpoint_recipe_only",
            "source_required": str(b),
            "experts": "16 (already expanded)",
            "data": "Code exact only",
            "micro_batch_size": 96,
            "objective": "Code LM; KD/replay off",
            "trainable_parameters": "new experts 8:16 and new router rows 8:16 only",
            "evidence": evidence["d_recipe"],
            "caveat": "No physical D checkpoint was found under current /data2; no 1800-step retraining was launched.",
        },
        "E": {
            "path": e_manifest,
            "physical_checkpoint_present": Path(e_manifest).is_dir(),
            "expected_step": 1800,
            "actual_step": None,
            "step_matches": False,
            "tensor_shard_count": 0,
            "tensor_shard_bytes": 0,
            "role": "random 8->16 expansion, no KD, no old replay/router FT, Code-only",
            "comparison_status": "repository_reference_checkpoint_not_local_not_A_matched",
            "source": "manifest Wiki source g2matched...mb128-1800 (not current physical A mb72 run)",
            "experts": "8->16",
            "data": "Code only",
            "micro_batch_size": 96,
            "objective": "Code LM; old-model KL coefficient 0",
            "trainable_parameters": "recipe/code path: new FFN experts and new router rows only; actual checkpoint trainable log unavailable",
            "evidence": [evidence["e_recipe"], evidence["manifest"]],
            "caveat": "Use only as an architecture-level reference until the manifest checkpoint is restored; A->E is not a same-source direct comparison.",
        },
        "F": {
            **checkpoint_facts(c, 1800),
            "role": "Code completion source for the vocabulary-KL Conversation branch",
            "comparison_status": "direct_physical_same_tensor_checkpoint_as_C",
            "source": str(b),
            "experts": "16",
            "data": "Code LM + Wiki replay KD",
            "micro_batch_size": 48,
            "trainable_parameters": "new Code experts plus all router rows",
            "metadata": metadata(c),
        },
        "G": {
            **checkpoint_facts(g, 600),
            "role": "16->24 expansion plus Wiki+Code output-logit KD initialization",
            "comparison_status": "direct_physical_vocab_kl_branch",
            "source": str(c),
            "experts": "16->24",
            "data": "Wiki+Code equal mixture; no Conversation samples",
            "micro_batch_size": 32,
            "objective": "teacher vocabulary KL coefficient 1; LM/aux/z disabled",
            "trainable_parameters": "new experts 16:24 and new router rows 16:24 only",
            "metadata": metadata(g),
        },
        "H": {
            **checkpoint_facts(h, 1800),
            "role": "normal Conversation phase with per-update Wiki+Code router FT",
            "comparison_status": "direct_physical_vocab_kl_branch",
            "source": str(g),
            "experts": "24 (already expanded)",
            "data": "Conversation LM + Wiki+Code replay (1:0.5:0.5)",
            "micro_batch_size": 36,
            "objective": "Conversation LM + old-task vocabulary KL coefficient 10",
            "trainable_parameters": "new Conversation experts plus all router rows; replay gradients are router-only",
            "metadata": metadata(h),
            "log": str(root / "03_logs/nine_stages/vocab_kl/r4.log"),
        },
        "I": {
            "path": i_manifest,
            "physical_checkpoint_present": Path(i_manifest).is_dir(),
            "expected_step": 1800,
            "actual_step": None,
            "step_matches": False,
            "tensor_shard_count": 0,
            "tensor_shard_bytes": 0,
            "role": "repository Conversation-only random-expansion reference without Conversation KD/replay",
            "comparison_status": "repository_reference_checkpoint_not_local_source_not_F_matched",
            "source": "baseline router-retuned Code checkpoint at logical step 3600, not current F",
            "experts": "16->24",
            "data": "Conversation only",
            "micro_batch_size": 48,
            "objective": "Conversation LM; old-model KL coefficient 0",
            "trainable_parameters": "new Conversation experts and new router rows; old/shared/attention frozen",
            "reference_final_metrics": {
                "wiki_next_token_accuracy": 0.407037,
                "code_next_token_accuracy": 0.347415,
                "conversation_next_token_accuracy": 0.385033,
            },
            "evidence": [evidence["conversation_baseline_recipe"], evidence["manifest"]],
            "caveat": "Reference only: source and initialization are not matched to F/G/H.",
        },
    }

    inventory = {
        "schema_version": 1,
        "source_root": str(root),
        "common_model": common_model,
        "records": records,
        "comparison_eligibility": {
            "direct_now": ["A_to_B", "B_to_C", "F_to_G", "G_to_H"],
            "blocked_missing_matched_checkpoint": ["B_to_D", "D_to_E", "C_to_D", "C_to_E", "G_to_I"],
            "reference_only": ["A_to_E", "F_to_I"],
        },
        "negative_control_audit": {
            "no_kd_joint_lm_script_is_not_E_or_I": True,
            "reason": "It replays old Wiki (and later Wiki+Code) LM data, violating the Code-only/Conversation-only control definition.",
            "evidence": evidence["joint_no_kd_not_code_only"],
        },
        "evidence": evidence,
    }
    json_path = out / "checkpoint_inventory_A_to_I.json"
    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Fingerprint checkpoint inventory A–I",
        "",
        "| ID | availability | experts | source/data | direct use |",
        "|---|---|---:|---|---|",
    ]
    for key, record in records.items():
        availability = "physical" if record["physical_checkpoint_present"] else "missing/reference"
        lines.append(
            f"| {key} | {availability} | {record['experts']} | {record.get('data', '')} | "
            f"{record['comparison_status']} |"
        )
    lines.extend(
        [
            "",
            "Direct matched comparisons currently possible: A→B, B→C, F→G, G→H.",
            "D is absent. E and I are repository references whose physical checkpoints and matched sources are not local.",
            "The repository's `no_kd_joint_lm` chain is not E/I because it replays old raw LM data.",
            "",
            f"Machine-readable inventory: `{json_path}`",
        ]
    )
    md_path = out / "checkpoint_inventory_A_to_I.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
