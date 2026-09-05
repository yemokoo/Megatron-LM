#!/usr/bin/env python3
"""Emit a fail-fast checkpoint lineage/loss/update audit for the stability study."""

import argparse
import json
import os


REFERENCE = "/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600"
CODE_ONLY = "/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/checkpoints/D_B_to_Code_only_no_olddata_mb48_gbs2304_step1800"
MIXED = "/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/02_nine_runs/local/weights/a100/mha/g2-checkpoints/code/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/r2-code-hiddenkl-c10-fixed-l2to9-post16-mb48-1800-probe3i100"
ROUTER_FT = "/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/checkpoints/CodeWiki_router_only_LM_1800_from_D"


def checkpoint(path, expected):
    tracker = os.path.join(path, "latest_checkpointed_iteration.txt")
    with open(tracker, encoding="utf-8") as handle:
        actual = int(handle.read().strip())
    if actual != expected:
        raise RuntimeError(f"checkpoint step mismatch: {path}: {actual} != {expected}")
    return actual


def require(log, needles):
    with open(log, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise RuntimeError(f"missing audit strings in {log}: {missing}")
    return {needle: text.count(needle) for needle in needles}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    code_log = os.path.join(CODE_ONLY, "logs/a_to_b_freeze.log")
    mixed_log = os.path.join(MIXED, "logs/a_to_b_freeze.log")
    router_log = os.path.join(ROUTER_FT, "logs/phase3_moe_router_run.log")
    reference_log = os.path.join(REFERENCE, "logs/g2_ffn_only_code_expert_logits_init_freeze.log")
    audit = {
        "representation_target": {
            "name": "residual-included Transformer layer_output",
            "layers": [2, 3, 4, 5, 6, 7, 8, 9],
            "last_layer_included": True,
            "ffn_output": "diagnostic only",
            "router_input": "excluded",
            "code_evidence": {
                "hook": "Megatron-LM/pretrain_gpt.py:_capture_transformer_layer_outputs",
                "residual_add": "Megatron-LM/megatron/core/transformer/transformer_layer.py:mlp_bda(..., residual, ...)",
                "hidden_kl": "Megatron-LM/pretrain_gpt.py:_masked_layer_hidden_kl",
            },
        },
        "stages": {
            "expansion_kd_init_complete_reference": {
                "path": REFERENCE, "step": checkpoint(REFERENCE, 600),
                "meaning": "Code expert expansion 8->16 followed by Wiki logits-KD initialization",
                "log": reference_log,
                "evidence": require(reference_log, ["moe_expand_from_num_experts", "moe_old_model_kl_coeff"]),
            },
            "code_only_no_replay_no_router_ft": {
                "path": CODE_ONLY, "step": checkpoint(CODE_ONLY, 1800),
                "parent": REFERENCE,
                "loss": "Code LM only",
                "updates": "new experts 8:16 and their router rows; no Wiki replay/KD/router-FT",
                "log": code_log,
                "evidence": require(code_log, [
                    "moe_joint_replay_lm ............................. False",
                    "moe_old_model_kl_coeff .......................... 0.0",
                    "moe_train_new_experts_and_router_only ........... True",
                ]),
            },
            "code_lm_plus_wiki_layer_output_hidden_kl_router_gradient": {
                "path": MIXED, "step": checkpoint(MIXED, 1800),
                "parent": REFERENCE,
                "loss": "Code LM + Wiki residual layer_output hidden KL (coefficient 10, layers 2-9)",
                "updates": "new experts plus all 16 router rows; replay gradients reach router rows",
                "log": mixed_log,
                "evidence": require(mixed_log, [
                    "moe_joint_replay_lm ............................. True",
                    "moe_joint_replay_old_data_hidden_kl ............. True",
                    "moe_old_hidden_kl_coeff ......................... 10.0",
                    "moe_old_hidden_kl_layers ........................ 2,3,4,5,6,7,8,9",
                    "joint_replay/router_grad/global/all_rows/replay_scaled_norm",
                ]),
            },
            "code_only_then_wiki_code_router_only_lm_ft": {
                "path": ROUTER_FT, "step": checkpoint(ROUTER_FT, 3600),
                "parent": CODE_ONLY,
                "loss": "Wiki+Code mixed LM, 1800 additional steps",
                "updates": "MoE routers only; experts/backbone frozen",
                "log": router_log,
                "evidence": require(router_log, [
                    "phase3 MoE router-only mixed wiki train dataset",
                    "phase3 MoE router-only mixed code train dataset",
                    "moe_train_router_only ........................... True",
                    "moe_joint_replay_lm ............................. False",
                ]),
            },
        },
        "analysis_constraints": {
            "checkpoint_mutation": False,
            "optimizer_steps": 0,
            "forced_routing": False,
            "fingerprint_kd_or_gating": False,
            "domains": {"wiki_valid_tokens": 10000000, "code_valid_tokens": 10000000},
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + ".inprogress"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
