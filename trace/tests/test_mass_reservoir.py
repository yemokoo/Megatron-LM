"""Deterministic mass-transfer reservoir (scripts/residual/mass_reservoir.py).

Invariants: the reservoir is never a Top-K candidate, it only adds alpha*exp(z_res) to the
full-softmax denominator; expansion copies r_res exactly into the new row and moves alpha
1 -> 0, so the denominator, the Top-K set (where z_res < K-th logit) and the model output are
unchanged; warm-up routes the primary pass to the new expert only; the router-correction
margin max(0, z_res - stopgrad(z_th) + delta) only moves r_res.
Tiny CPU Llama, same fixture as test_residual_split.
"""
import copy
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "implementations" / "llmcl_benchmark"
for path in (IMPL, ROOT / "scripts" / "residual", ROOT / "scripts" / "selfgen"):
    sys.path.insert(0, str(path))

import train_residual_v3_split as S            # noqa: E402  installs the patches
import train_residual_v3 as TR                 # noqa: E402
import mass_reservoir as MR                    # noqa: E402
from model import Ours_LoRA_MoE_V3 as V3       # noqa: E402

ON = MR.MassReservoirConfig(enabled=True, delta=0.5, margin_weight=0.1, warmup_frac=0.05)


def tiny_config(vocab=128):
    return LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                       tie_word_embeddings=False)


def make_v3(experts=2, top_k=1, cfg=ON, mode="full_softmax"):
    MR.set_active(cfg)
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_config()).eval()
    V3.attach_shared_qkvo_lora_moe(model, r=4, alpha=8, top_k=top_k, aux_loss_coeff=0.01,
                                   z_loss_coeff=0.001, routing_weight_mode=mode, dropout=0.0)
    V3.add_v3_experts(model, experts)        # a rebuild (EXPANDING off): alpha stays 1
    return model


def routers(model):
    return MR.routers(model)


def randomize(model, std=0.5):
    """Non-zero LoRA B and random router/skip/reservoir rows so routing is non-trivial."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (".experts." in name and name.endswith(".B")) or "shared_expert_router" in name:
                p.copy_(torch.randn(p.shape, generator=g) * std)


def ids(n=3, length=10):
    torch.manual_seed(2)
    return torch.randint(5, 128, (n, length))


def expand(model, count=1):
    MR.EXPANDING.update(on=True, probe=None)
    try:
        V3.add_v3_experts(model, count)
    finally:
        MR.EXPANDING.update(on=False, probe=None)


def logits_of(router, h):
    real = router._active_count()
    rows = torch.cat([router.router.weight[:real], router.residual_router.weight,
                      router.mres_reservoir.weight], 0)
    return F.linear(h.reshape(-1, h.shape[-1]).float(), rows.float())


class Base(unittest.TestCase):
    def tearDown(self):
        MR.set_active(MR.MassReservoirConfig())
        MR.EXPANDING.update(on=False, probe=None)


class Routing(Base):
    def test_requires_full_softmax(self):
        with self.assertRaises(ValueError):
            make_v3(mode="straight_through_topk")

    def test_reservoir_is_never_a_topk_candidate(self):
        model = make_v3(experts=2, top_k=2)
        randomize(model)
        r = routers(model)[0]
        with torch.no_grad():
            r.mres_reservoir.weight.mul_(50.0)          # reservoir logit dwarfs every candidate
        h = torch.randn(64, 32)
        ctx = r(h)
        real = r._active_count()
        self.assertTrue(bool((ctx.expert_indices <= real).all()))    # ids 0..real only
        z = logits_of(r, h)
        cand, z_res = z[:, :real + 1], z[:, real + 1]
        expected = torch.exp(cand.gather(-1, ctx.expert_indices)) / (
            torch.exp(cand).sum(-1, keepdim=True) + torch.exp(z_res)[:, None])
        torch.testing.assert_close(ctx.expert_weights, expected, rtol=1e-5, atol=1e-7)

    def test_alpha_enters_only_the_denominator(self):
        model = make_v3(experts=3, top_k=1)
        randomize(model)
        r = routers(model)[1]
        h = torch.randn(40, 32)
        z = logits_of(r, h)
        real = r._active_count()
        cand, z_res = z[:, :real + 1], z[:, real + 1]
        for alpha in (0.0, 0.3, 1.0):
            MR.set_alpha(model, alpha)
            ctx = r(h)
            self.assertTrue(torch.equal(ctx.expert_indices, cand.topk(1, -1).indices))
            denom = torch.exp(cand).sum(-1) + alpha * torch.exp(z_res)
            expected = torch.exp(cand.gather(-1, ctx.expert_indices))[:, 0] / denom
            torch.testing.assert_close(ctx.expert_weights[:, 0], expected, rtol=1e-5, atol=1e-7)
        MR.set_alpha(model, 0.0)
        self.assertEqual(r._mres_log_alpha, float("-inf"))

    def test_selected_weights_are_not_renormalised(self):
        model = make_v3(experts=3, top_k=2)
        randomize(model)
        r = routers(model)[0]
        ctx = r(torch.randn(30, 32))
        self.assertTrue(bool((ctx.expert_weights.sum(-1) < 1.0 - 1e-4).all()))

    def test_primary_excludes_skip_and_reservoir(self):
        model = make_v3(experts=3, top_k=3)
        randomize(model)
        r = routers(model)[0]
        TR._set_log_alpha(model, float("-inf"))        # split: skip masked in the primary pass
        MR.set_flag(model, "_mres_primary", True)
        ctx = r(torch.randn(25, 32))
        self.assertTrue(bool((ctx.expert_indices < r._active_count()).all()))
        torch.testing.assert_close(ctx.expert_weights.sum(-1), torch.ones(25), rtol=0, atol=1e-6)

    def test_warmup_routes_to_new_expert_only(self):
        model = make_v3(experts=3, top_k=1)
        randomize(model)
        TR._set_log_alpha(model, float("-inf"))
        MR.set_flag(model, "_mres_primary", True)
        MR.set_flag(model, "_mres_force_new", True)
        for r in routers(model):
            ctx = r(torch.randn(50, 32))
            self.assertTrue(bool((ctx.expert_indices == r._active_count() - 1).all()))
            torch.testing.assert_close(ctx.expert_weights, torch.ones_like(ctx.expert_weights))


class Expansion(Base):
    def _prepared(self, reservoir_scale):
        model = make_v3(experts=2, top_k=1)
        randomize(model)
        for r in routers(model):
            with torch.no_grad():
                r.mres_reservoir.weight.mul_(reservoir_scale)
        MR.set_alpha(model, 1.0)                        # end of the previous task
        return model

    def test_exact_copy_gives_bit_identical_logits(self):
        model = self._prepared(1.0)
        expand(model)
        for r in routers(model):
            real = r._active_count()
            self.assertTrue(torch.equal(r.router.weight[real - 1], r.mres_reservoir.weight[0]))
            z = logits_of(r, torch.randn(80, 32))
            self.assertTrue(torch.equal(z[:, real - 1], z[:, real + 1]))
            self.assertEqual(r._mres_log_alpha, float("-inf"))

    def test_expansion_preserves_denominator_topk_and_output(self):
        model = self._prepared(1.0)
        for r in routers(model):                        # z_res = 0 < |z_E0| = max(z_E0, z_skip)
            with torch.no_grad():
                r.mres_reservoir.weight.zero_()
                r.residual_router.weight.copy_(-r.router.weight[:1])
        batch = {"input_ids": ids(), "attention_mask": torch.ones(3, 10, dtype=torch.long)}
        batch["labels"] = batch["input_ids"].clone()
        before = MR.run_probe(model, batch)
        expand(model)
        after = MR.run_probe(model, batch)
        rep = MR.compare_probes(before, after)["summary"]
        self.assertEqual(rep["margin_satisfied_rate_min"], 1.0)
        self.assertLess(rep["denominator_rel_diff_max"], 1e-6)
        self.assertEqual(rep["topk_change_rate_all_max"], 0.0)
        self.assertLess(rep["hidden_rel_l2_max"], 1e-6)
        self.assertLess(rep["logits_max_abs_diff"], 1e-5)

    def test_violations_are_the_only_topk_changes(self):
        model = self._prepared(3.0)                     # reservoir often above the threshold
        batch = {"input_ids": ids(), "attention_mask": torch.ones(3, 10, dtype=torch.long)}
        before = MR.run_probe(model, batch)
        expand(model)
        rep = MR.compare_probes(before, MR.run_probe(model, batch))
        self.assertLess(rep["summary"]["margin_satisfied_rate_min"], 1.0)
        self.assertGreater(rep["summary"]["topk_change_rate_all_max"], 0.0)
        self.assertLess(rep["summary"]["denominator_rel_diff_max_same_input"], 1e-6)
        first = rep["layers"][0]                        # layer 0 input is always unchanged
        self.assertEqual(first["same_input_rate"], 1.0)
        self.assertEqual(first["topk_change_rate_margin_ok"], 0.0)

    def test_first_task_keeps_random_row(self):
        MR.set_active(ON)
        torch.manual_seed(0)
        model = LlamaForCausalLM(tiny_config()).eval()
        V3.attach_shared_qkvo_lora_moe(model, r=4, alpha=8, top_k=1, aux_loss_coeff=0.01,
                                       z_loss_coeff=0.001, routing_weight_mode="full_softmax")
        expand(model)                                   # task 0: nothing to copy
        for r in routers(model):
            self.assertFalse(torch.equal(r.router.weight[0], r.mres_reservoir.weight[0]))
            self.assertEqual(r._mres_log_alpha, float("-inf"))

    def test_rebuild_does_not_copy_or_reset_alpha(self):
        model = make_v3(experts=3)
        for r in routers(model):
            self.assertEqual(r._mres_log_alpha, 0.0)
            self.assertFalse(torch.equal(r.router.weight[-1], r.mres_reservoir.weight[0]))

    def test_new_lora_starts_at_zero_output(self):
        model = self._prepared(1.0)
        expand(model)
        for layer in V3.shared_router_layers(model):
            for proj in layer.attention_expert_projections:
                self.assertTrue(bool((proj.experts[-1].B == 0).all()))


class Margin(Base):
    def test_hinge_value_and_gradient_only_on_reservoir(self):
        model = make_v3(experts=2, top_k=1)
        randomize(model)
        model.train()
        r = routers(model)[0]
        MR.set_flag(model, "_mres_margin", True)
        h = torch.randn(4, 6, 32)
        r(h)
        margin = MR.pop_margin_per_sample(model, 4)
        z = logits_of(r, h)
        real = r._active_count()
        th = z[:, :real + 1].max(-1).values
        expected = F.relu(z[:, real + 1] - th + ON.delta).reshape(4, 6).mean(-1) / 1  # one router here
        per_layer_count = sum(1 for rr in routers(model) if rr._mres_margin_buf == [])
        self.assertEqual(per_layer_count, len(routers(model)))
        torch.testing.assert_close(margin.detach(), expected.detach(), rtol=1e-5, atol=1e-6)
        margin.sum().backward()
        self.assertIsNotNone(r.mres_reservoir.weight.grad)
        self.assertGreater(float(r.mres_reservoir.weight.grad.abs().sum()), 0.0)
        for p in (r.router.weight, r.residual_router.weight):
            self.assertTrue(p.grad is None or bool((p.grad == 0).all()))

    def test_margin_is_added_to_router_correction_losses(self):
        MR.set_active(ON)
        model = make_v3(experts=2, top_k=1)
        randomize(model)
        model.train()

        class Fake:
            raw_model = model
        batch = ids(2, 8)
        MR.set_flag(model, "_mres_margin", True)
        out = model(input_ids=batch)
        base = S._stock_per_sample_losses(out.logits, batch)
        with_margin = S._per_sample_losses_mres(Fake(), out.logits, batch)
        self.assertTrue(bool((with_margin >= base - 1e-6).all()))
        self.assertTrue(bool((with_margin > base).any()))
        self.assertTrue(all(r._mres_margin_buf == [] for r in routers(model)))


    def test_current_slice_margin_can_be_switched_off(self):
        MR.set_active(MR.MassReservoirConfig(enabled=True, delta=0.5, margin_weight=0.1,
                                             margin_current=False))
        model = make_v3(experts=2, top_k=1, cfg=MR.ACTIVE)
        randomize(model)
        model.train()

        class Fake:
            raw_model = model
            _router_ft_sources = 2
        fake = Fake()
        batch = ids(4, 8)
        slices = S._extra_router_replay_batches(fake, {"input_ids": batch, "labels": batch})
        MR.set_flag(model, "_mres_margin", True)
        for extra in slices:                            # iterating marks the current slice
            out = model(input_ids=extra["input_ids"])
            base = S._stock_per_sample_losses(out.logits, extra["labels"])
            got = S._per_sample_losses_mres(fake, out.logits, extra["labels"])
            self.assertTrue(torch.equal(got, base))
        self.assertFalse(fake._mres_current_slice)


class HeaderExclusion(Base):
    """Header/BOS-guarded positions never use the computed routing, so they are left out of
    the margin and the stats (the guard's own rule: header at every layer, BOS below the last)."""

    def setUp(self):
        import bos_guard
        self.bg = bos_guard
        self.saved = dict(bos_guard._bos_mask)

    def tearDown(self):
        self.bg._bos_mask.update(self.saved)
        super().tearDown()

    def test_margin_and_stats_skip_guarded_tokens(self):
        model = make_v3(experts=2, top_k=1)
        randomize(model)
        model.train()
        r = routers(model)[0]
        r._is_last_layer = False
        B, T = 2, 6
        header = torch.zeros(B, T, dtype=torch.bool); header[:, :3] = True
        bos = torch.zeros(B, T, dtype=torch.bool); bos[:, 3] = True
        self.bg._bos_mask.update(full=header, value=bos)
        h = torch.randn(B, T, 32)
        z = logits_of(r, h)
        real = r._active_count()
        hinge = F.relu(z[:, real + 1] - z[:, :real + 1].max(-1).values + ON.delta).reshape(B, T)
        MR.set_flag(model, "_mres_margin", True)
        MR.enable_stats(model, "router_ft")
        r(h)
        margin = MR.pop_margin_per_sample(model, B)
        routed = ~(header | bos)                        # below the last layer: header and BOS guarded
        expected = (hinge * routed).sum(-1) / routed.sum(-1)
        torch.testing.assert_close(margin.detach(), expected.detach(), rtol=1e-5, atol=1e-6)
        self.assertEqual(r._mres_stats["tokens"], float(routed.sum()))

        r._is_last_layer = True                         # last layer: only the header is guarded
        r._mres_stats_all = {}
        MR.enable_stats(model, "router_ft")
        r(h)
        margin = MR.pop_margin_per_sample(model, B)
        routed = ~header
        expected = (hinge * routed).sum(-1) / routed.sum(-1)
        torch.testing.assert_close(margin.detach(), expected.detach(), rtol=1e-5, atol=1e-6)
        self.assertEqual(r._mres_stats["tokens"], float(routed.sum()))

    def test_unguarded_model_excludes_nothing(self):
        model = make_v3(experts=2, top_k=1)
        r = routers(model)[0]
        self.assertFalse(hasattr(r, "_is_last_layer"))
        self.assertIsNone(MR.guarded_positions(r, 12))


class ScheduleAndMeta(Base):
    def test_alpha_schedule(self):
        total, warm = 100, MR.warmup_updates(100, 0.05)
        self.assertEqual(warm, 5)
        self.assertEqual(MR.alpha_schedule(0, total, warm), 0.0)
        self.assertEqual(MR.alpha_schedule(5, total, warm), 0.0)
        self.assertAlmostEqual(MR.alpha_schedule(52, total, warm), 47 / 95)
        self.assertEqual(MR.alpha_schedule(100, total, warm), 1.0)

    def test_meta_roundtrip(self):
        cfg = MR.MassReservoirConfig(enabled=True, delta=1.0, margin_weight=0.2, warmup_frac=0.1,
                                     margin_current=False)
        meta = {MR.META_KEY: cfg.to_meta(alpha_end=1.0)}
        self.assertEqual(MR.config_from_meta(meta), cfg)
        self.assertFalse(MR.config_from_meta({}).enabled)

    def test_reservoir_is_saved_with_the_router(self):
        self.assertIn(MR.RESERVOIR_KEY, V3.Ours_LoRA_MoE_V3.save_key_substrings)
        model = make_v3(experts=2)
        keys = [k for k in model.state_dict() if MR.RESERVOIR_KEY.strip(".") in k]
        self.assertEqual(len(keys), len(routers(model)))

    def test_disabled_config_attaches_nothing(self):
        model = make_v3(experts=2, cfg=MR.MassReservoirConfig(), mode="straight_through_topk")
        self.assertFalse(any(MR.enabled(r) for r in routers(model)))
        ctx = routers(model)[0](torch.randn(10, 32))
        torch.testing.assert_close(ctx.expert_weights, torch.ones_like(ctx.expert_weights))


if __name__ == "__main__":
    unittest.main()
