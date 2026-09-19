#!/usr/bin/env python3
"""Compute OA_set, OA_exact, NewSlot (Eq. routing) from two dump_full_routing.py outputs.

For a probe token u at layer l, let T and T_ref be the sets of experts selected
by the final and reference (wiki) checkpoints, and vecT/vecT_ref the same lists
ordered by descending router logit (== mixture-weight rank). Expert identities
are retained across expansion, so E_ref at layer l is simply the reference
checkpoint's expert range {0, ..., num_experts_ref[l]-1}.

    OA_set    = < |T ∩ T_ref| / K >
    OA_exact  = < 1[vecT == vecT_ref] >
    NewSlot   = < |T \\ E_ref| / K >

averaged over layers l in L and probe tokens u in X_j (i.e. a flat mean over
every (layer, token) pair from both dumps).

usage:
  python compute_routing_overlap.py --final <final.pt> --ref <wiki_ref.pt> \
      [--json-out result.json]
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", required=True, type=Path,
                        help="dump_full_routing.py output for f (final checkpoint)")
    parser.add_argument("--ref", required=True, type=Path,
                        help="dump_full_routing.py output for f_ref (wiki checkpoint)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    final = torch.load(args.final, map_location="cpu", weights_only=False)
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)

    if final["token_count"] != ref["token_count"]:
        raise ValueError(
            f"token_count mismatch: final={final['token_count']} "
            f"ref={ref['token_count']} -- the two dumps were not run on the "
            "same deterministic probe stream (same data-path/split/seed/"
            "batch sizes/eval-iters)")
    if final["layers"] != ref["layers"]:
        raise ValueError(f"layer set mismatch: final={final['layers']} ref={ref['layers']}")
    K_final, K_ref = final["topk"], ref["topk"]
    if K_final != K_ref:
        raise ValueError(f"topk mismatch: final={K_final} ref={K_ref}")
    K = K_final

    oa_set_sum = 0.0
    oa_exact_sum = 0.0
    new_slot_sum = 0.0
    total_pairs = 0
    per_layer = {}

    for layer in final["layers"]:
        T = final["expert_idx"][layer].long()          # [N, K], desc-logit order
        T_ref = ref["expert_idx"][layer].long()         # [N, K]
        n_experts_ref = int(ref["num_experts"][layer])
        n = T.shape[0]

        # Set overlap |T ∩ T_ref| / K, order-independent.
        T_onehot = torch.zeros(n, max(int(T.max()), int(T_ref.max())) + 1,
                               dtype=torch.bool)
        T_onehot.scatter_(1, T.clamp_min(0), True)
        Tref_onehot = torch.zeros_like(T_onehot)
        Tref_onehot.scatter_(1, T_ref.clamp_min(0), True)
        intersection = (T_onehot & Tref_onehot).sum(dim=1).float()
        oa_set = intersection / K

        # Exact ordered match (same experts, same rank order).
        oa_exact = (T == T_ref).all(dim=1).float()

        # NewSlot: fraction of T's slots outside the reference expert pool.
        is_new = (T >= n_experts_ref).float()
        new_slot = is_new.mean(dim=1)

        oa_set_sum += float(oa_set.sum())
        oa_exact_sum += float(oa_exact.sum())
        new_slot_sum += float(new_slot.sum())
        total_pairs += n
        per_layer[f"layer_{layer:02d}"] = {
            "OA_set": float(oa_set.mean()),
            "OA_exact": float(oa_exact.mean()),
            "NewSlot": float(new_slot.mean()),
            "token_count": n,
        }

    result = {
        "final_label": final.get("label"),
        "ref_label": ref.get("label"),
        "topk": K,
        "token_count": total_pairs // len(final["layers"]),
        "num_layers": len(final["layers"]),
        "OA_set": oa_set_sum / total_pairs,
        "OA_exact": oa_exact_sum / total_pairs,
        "NewSlot": new_slot_sum / total_pairs,
        "per_layer": per_layer,
    }
    # Sanity check from Eq. (routing_order): 1 - NewSlot >= OA_set >= OA_exact.
    result["order_check_1_minus_newslot_ge_oaset"] = (
        (1 - result["NewSlot"]) >= result["OA_set"] - 1e-9)
    result["order_check_oaset_ge_oaexact"] = (
        result["OA_set"] >= result["OA_exact"] - 1e-9)

    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
