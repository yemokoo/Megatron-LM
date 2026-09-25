"""Mass-reservoir ablation arms (scripts/residual/{mass_reservoir,train_residual_v3_split}.py).

  random row   MRES_NEW_ROW=random: the only difference from the method is that the new router
               row keeps the stock init instead of the r_res clone.
  no margin    MRES_LAMBDA=0: the router-FT gradient equals the margin-free one exactly.
  post-hoc     RESIDUAL_ROUTER_FT_TIMING=posthoc: the joint loop split into a primary-only walk
               and a router-only walk that forwards exactly the joint arm's router-FT batches.
  distill      RESIDUAL_ROUTER_FT_OBJECTIVE=distill: replay/BoS records use the per-layer router
               KL to the pre-expansion router (zero-row or zero-prob padded); the current-task
               slice keeps LM loss; the margin stays.
Tiny CPU Llama, same fixture as test_mass_reservoir.
"""
import copy
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_mass_reservoir import (  # noqa: E402  (also installs the residual/split patches)
    ON, MR, S, TR, V3, expand, ids, logits_of, make_v3, randomize, routers)


class Base(unittest.TestCase):
    def setUp(self):
        self.saved = (S.ROUTER_FT_TIMING, S.ROUTER_FT_OBJECTIVE, S.DISTILL_PAD)

    def tearDown(self):
        S.ROUTER_FT_TIMING, S.ROUTER_FT_OBJECTIVE, S.DISTILL_PAD = self.saved
        MR.set_active(MR.MassReservoirConfig())
        MR.EXPANDING.update(on=False, probe=None)


def with_cfg(**kw):
    base = dict(enabled=True, delta=0.5, margin_weight=0.1, warmup_frac=0.05)
    base.update(kw)
    return MR.MassReservoirConfig(**base)


def prepared(cfg):
    model = make_v3(experts=2, top_k=1, cfg=cfg)
    randomize(model)
    MR.set_alpha(model, 1.0)
    return model


# ------------------------------------------------------------------ random row
class RandomRow(Base):
    def _expand_both(self):
        out = {}
        for arm in ("reservoir", "random"):
            model = prepared(with_cfg(new_row=arm))
            torch.manual_seed(123)                  # same RNG entering growth in both arms
            expand(model)
            out[arm] = model
        return out["reservoir"], out["random"]

    def test_only_the_new_router_rows_differ(self):
        res, rnd = self._expand_both()
        sd_res, sd_rnd = res.state_dict(), rnd.state_dict()
        self.assertEqual(sd_res.keys(), sd_rnd.keys())
        differing = [k for k in sd_res if not torch.equal(sd_res[k], sd_rnd[k])]
        self.assertTrue(differing)
        for key in differing:
            self.assertTrue(key.endswith("shared_expert_router.router.weight"), key)
            a, b = sd_res[key], sd_rnd[key]
            self.assertTrue(torch.equal(a[:-1], b[:-1]))           # old rows identical
        for r_res, r_rnd in zip(routers(res), routers(rnd)):
            self.assertTrue(torch.equal(r_res.router.weight[-1], r_res.mres_reservoir.weight[0]))
            self.assertFalse(torch.equal(r_rnd.router.weight[-1], r_rnd.mres_reservoir.weight[0]))
            self.assertEqual(r_res._mres_log_alpha, r_rnd._mres_log_alpha)   # alpha 1 -> 0 both

    def test_random_row_is_the_stock_growth_init(self):
        _, rnd = self._expand_both()
        stock = prepared(MR.MassReservoirConfig(enabled=True))
        torch.manual_seed(123)
        MR.set_active(MR.MassReservoirConfig(enabled=True))
        V3.add_v3_experts(stock, 1)                 # rebuild path: growth only, no copy
        for a, b in zip(routers(rnd), routers(stock)):
            self.assertTrue(torch.equal(a.router.weight, b.router.weight))

    def test_env_and_meta(self):
        import os
        old = dict(os.environ)
        try:
            os.environ.update(MRES_ENABLE="1", MRES_NEW_ROW="random")
            self.assertEqual(MR.config_from_env().new_row, "random")
            os.environ["MRES_NEW_ROW"] = "zeros"
            with self.assertRaises(ValueError):
                MR.config_from_env()
            os.environ.pop("MRES_NEW_ROW")
            self.assertEqual(MR.config_from_env().new_row, "reservoir")
        finally:
            os.environ.clear()
            os.environ.update(old)
        cfg = with_cfg(new_row="random")
        self.assertEqual(MR.config_from_meta({MR.META_KEY: cfg.to_meta(1.0)}), cfg)
        legacy = {k: v for k, v in with_cfg().to_meta(1.0).items() if k != "new_row"}
        self.assertEqual(MR.config_from_meta({MR.META_KEY: legacy}).new_row, "reservoir")


# ------------------------------------------------------------------ margin 0
class NoMargin(Base):
    def _router_ft_grads(self, cfg, margin_on=True):
        MR.set_active(cfg)
        model = prepared(cfg)
        expand(model)
        MR.set_alpha(model, 0.4)
        model.train()

        class Fake:
            raw_model = model
        batch = ids(3, 9)
        MR.set_flag(model, "_mres_margin", margin_on)
        out = model(input_ids=batch)
        S._per_sample_losses_mres(Fake(), out.logits, batch).sum().backward()
        return {n: (p.grad.clone() if p.grad is not None else None)
                for n, p in model.named_parameters()}

    def test_lambda_zero_equals_no_margin_term(self):
        free = self._router_ft_grads(with_cfg(margin_weight=0.0), margin_on=False)
        zero = self._router_ft_grads(with_cfg(margin_weight=0.0), margin_on=True)
        with_margin = self._router_ft_grads(with_cfg(margin_weight=0.1), margin_on=True)
        for name in free:
            if free[name] is None:
                self.assertTrue(zero[name] is None or bool((zero[name] == 0).all()), name)
            else:
                self.assertTrue(torch.equal(free[name], zero[name]), name)
        res = [n for n in free if "mres_reservoir" in n]
        self.assertTrue(any(not torch.equal(with_margin[n], zero[n]) for n in res))


# ------------------------------------------------------------------ distill
class Distill(Base):
    def _expanded(self, pad, new_row="reservoir"):
        S.ROUTER_FT_OBJECTIVE, S.DISTILL_PAD = "distill", pad
        cfg = with_cfg(new_row=new_row)
        MR.set_active(cfg)
        model = prepared(cfg)
        before = [(r.router.weight[:2].detach().clone(), r.residual_router.weight.detach().clone())
                  for r in routers(model)]
        MR.EXPANDING.update(on=True, probe=None)
        try:
            S.add_experts_from_residual(model, 1)
        finally:
            MR.EXPANDING.update(on=False, probe=None)
        return model, before

    def test_teacher_is_the_pre_expansion_router(self):
        for pad in MR.DISTILL_PADS:
            model, before = self._expanded(pad)
            for r, (rows, skip) in zip(routers(model), before):
                t = r._rd_teacher
                self.assertEqual((t["old"], t["pad"]), (2, pad))
                self.assertTrue(torch.equal(t["rows"], rows.float()))
                self.assertTrue(torch.equal(t["skip"], skip.float()))
                with torch.no_grad():                   # training moves the rows, not the teacher
                    r.router.weight.add_(1.0)
                self.assertTrue(torch.equal(t["rows"], rows.float()))

    def test_row_pad_kl_matches_manual(self):
        model, _ = self._expanded("row")
        r = routers(model)[0]
        model.train()
        h = torch.randn(2, 5, 32)
        r._rd_on = True
        r(h)
        kl, _ = r._rd_buf[-1]
        z = logits_of(r, h)
        real = r._active_count()
        student = z[:, :real + 1]
        t = r._rd_teacher
        x = h.reshape(-1, 32)
        teacher = torch.cat([x @ t["rows"].T, torch.zeros(x.shape[0], 1), x @ t["skip"].T], -1)
        expected = F.kl_div(F.log_softmax(student, -1), F.log_softmax(teacher, -1),
                            log_target=True, reduction="none").sum(-1)
        torch.testing.assert_close(kl.detach(), expected, rtol=1e-5, atol=1e-6)

    def test_prob_pad_kl_matches_manual(self):
        model, _ = self._expanded("prob")
        r = routers(model)[1]
        model.train()
        h = torch.randn(2, 5, 32)
        r._rd_on = True
        r(h)
        kl, _ = r._rd_buf[-1]
        z = logits_of(r, h)
        real = r._active_count()
        log_s = F.log_softmax(z[:, :real + 1], -1)
        t = r._rd_teacher
        x = h.reshape(-1, 32)
        log_t = F.log_softmax(torch.cat([x @ t["rows"].T, x @ t["skip"].T], -1), -1)
        on_t = torch.cat([log_s[:, :2], log_s[:, real:real + 1]], -1)   # new expert slot dropped
        expected = (log_t.exp() * (log_t - on_t)).sum(-1)
        torch.testing.assert_close(kl.detach(), expected, rtol=1e-5, atol=1e-6)

    def test_kl_is_zero_when_student_equals_padded_teacher(self):
        model, _ = self._expanded("row")
        for r in routers(model):
            with torch.no_grad():
                r.router.weight[-1].zero_()             # the zero-padded teacher row
        model.train()
        MR.set_flag(model, "_rd_on", True)
        model(input_ids=ids(2, 7))
        kl = MR.pop_distill_per_sample(model, 2)
        torch.testing.assert_close(kl, torch.zeros(2), rtol=0, atol=1e-6)

    def test_replay_uses_kl_current_slice_uses_lm_margin_kept(self):
        model, _ = self._expanded("row")
        model.train()

        class Fake:
            raw_model = model
            _router_ft_sources = 2
        fake = Fake()
        batch = ids(4, 8)

        def run(current):
            fake._mres_current_slice = current
            MR.set_flag(model, "_mres_margin", True)
            MR.set_flag(model, "_rd_on", True)
            out = model(input_ids=batch)                # deterministic: dropout 0
            kl_expected = MR.pop_distill_per_sample(model, 4)
            margin = MR.pop_margin_per_sample(model, 4)
            lm = S._stock_per_sample_losses(out.logits, batch)
            model(input_ids=batch)                      # refill the buffers for the real call
            got = S._per_sample_losses_mres(fake, out.logits, batch)
            return got, kl_expected, margin, lm

        got, kl, margin, lm = run(current=False)
        torch.testing.assert_close(got, kl + ON.margin_weight * margin, rtol=1e-5, atol=1e-6)
        got, kl, margin, lm = run(current=True)
        torch.testing.assert_close(got, lm + ON.margin_weight * margin, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(not r._rd_buf and not r._mres_margin_buf for r in routers(model)))

    def test_kl_gradient_reaches_router_rows_not_experts_or_teacher(self):
        model, _ = self._expanded("row")
        model.train()
        V3.freeze_v3_experts(model, None)
        MR.set_flag(model, "_rd_on", True)
        model(input_ids=ids(2, 8))
        MR.pop_distill_per_sample(model, 2).sum().backward()
        last = routers(model)[-1]
        self.assertGreater(float(last.router.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(last.residual_router.weight.grad.abs().sum()), 0.0)
        self.assertTrue(last.mres_reservoir.weight.grad is None
                        or bool((last.mres_reservoir.weight.grad == 0).all()))
        self.assertFalse(last._rd_teacher["rows"].requires_grad)

    def test_guarded_tokens_are_excluded(self):
        import bos_guard
        saved = dict(bos_guard._bos_mask)
        try:
            model, _ = self._expanded("row")
            model.train()
            r = routers(model)[0]
            r._is_last_layer = False
            B, T = 2, 6
            header = torch.zeros(B, T, dtype=torch.bool); header[:, :2] = True
            bos_guard._bos_mask.update(full=header, value=torch.zeros(B, T, dtype=torch.bool))
            r._rd_on = True
            r(torch.randn(B, T, 32))
            kl, routed = r._rd_buf[-1]
            self.assertTrue(torch.equal(routed, ~header.reshape(-1)))
        finally:
            bos_guard._bos_mask.update(saved)

    def test_lm_arm_captures_nothing(self):
        S.ROUTER_FT_OBJECTIVE = "lm"
        MR.set_active(ON)
        model = prepared(ON)
        MR.EXPANDING.update(on=True, probe=None)
        try:
            S.add_experts_from_residual(model, 1)
        finally:
            MR.EXPANDING.update(on=False, probe=None)
        self.assertTrue(all(getattr(r, "_rd_teacher", None) is None for r in routers(model)))


# ------------------------------------------------------------------ post-hoc
class _Records(list):
    pass


def _loop_harness(model, *, n_primary=12, batch=2, n_replay=18, sources=2, lr=1e-2):
    """A real V3 trainer object driving the real _run_v2_joint_epochs on a tiny model."""
    trainer = S.Trainer.__new__(S.Trainer)
    trainer.raw_model = trainer.model = model
    trainer.args = types.SimpleNamespace(
        gradient_accumulation_steps=1, v2_replay_forward_batch_size=2,
        v2_joint_new_to_replay_ratio=0, v2_max_replay_batches_per_step=0,
        loss_log_interval=10 ** 6, global_rank=-1, training_version="v3",
        v2_joint_replay_objective="lm", v2_joint_replay_loss_coeff=1.0,
        v2_hidden_mse_loss_coeff=1.0, replay_selection_mode="random")
    trainer.tokenizer = types.SimpleNamespace(pad_token_id=0)
    trainer._router_ft_sources = sources
    trainer._count_workload_batch = lambda *a, **k: None
    trainer._count_workload_update = lambda: None
    trainer._run_epoch_probe = lambda *a, **k: None
    trainer.saved = []
    trainer.save_model = trainer.saved.append
    trainer.steps = 0

    def reinit(updates=None):
        trainer.optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        step = trainer.optimizer.step

        def counted(*a, **k):
            trainer.steps += 1
            return step(*a, **k)
        trainer.optimizer.step = counted
        trainer.lr_scheduler = types.SimpleNamespace(step=lambda: None)
        trainer.reinit_calls = getattr(trainer, "reinit_calls", []) + [updates]
    reinit()
    trainer.reinit_calls = []
    trainer._reinit_engine = reinit

    g = torch.Generator().manual_seed(7)

    def rows(n, length):
        out = []
        for _ in range(n):
            x = torch.randint(5, 128, (length,), generator=g)
            out.append({"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()})
        return out
    primary = rows(n_primary, 8)
    replay = rows(n_replay, 6)
    primary_loader = DataLoader(primary, batch_size=batch, sampler=DistributedSampler(
        primary, num_replicas=1, rank=0, shuffle=True, seed=3))
    replay_loader = DataLoader(replay, batch_size=1, sampler=DistributedSampler(
        replay, num_replicas=1, rank=0, shuffle=True, seed=4))

    records = _Records()

    def hook(module, args, kwargs):
        experts_trainable = any(p.requires_grad for p in V3._v3_expert_parameters(model))
        records.append((
            "primary" if experts_trainable else "router",
            bool(getattr(trainer, "_mres_current_slice", False)),
            tuple(kwargs["input_ids"].reshape(-1).tolist())))
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    return trainer, primary_loader, replay_loader, records, handle


class PostHoc(Base):
    def _model(self):
        MR.set_active(ON)
        model = prepared(ON)
        expand(model)
        V3.freeze_v3_experts(model, {2})
        V3.freeze_v3_routers(model, True)
        model.train()
        return model

    def _run(self, branches=None, timing=None, epochs=2):
        model = self._model()
        trainer, pl, rl, records, handle = _loop_harness(model)
        try:
            if timing is not None:
                S.ROUTER_FT_TIMING = timing
                trainer._run_v2_joint_epochs(pl, rl, epochs, torch.device("cpu"), "t",
                                             task="T", i_task=1)
            else:
                trainer._joint_branches = branches
                S._prev_joint_epochs(trainer, pl, rl, epochs, torch.device("cpu"), "t",
                                     task="T", i_task=1)
        finally:
            handle.remove()
        return trainer, model, list(records), pl

    def test_split_walks_replay_exactly_the_joint_batches(self):
        _, _, joint, pl = self._run()
        t_p, _, prim, _ = self._run(branches=("primary",))
        t_r, _, rout, _ = self._run(branches=("router",))
        joint_primary = [r for r in joint if r[0] == "primary"]
        joint_router = [r for r in joint if r[0] == "router"]
        self.assertEqual(len(joint_primary), 2 * len(pl))
        self.assertTrue(any(r[1] for r in joint_router))          # current slices present
        self.assertTrue(any(not r[1] for r in joint_router))      # replay records present
        self.assertEqual(prim, joint_primary)
        self.assertEqual(rout, joint_router)
        self.assertEqual(t_p.steps, 2 * len(pl))
        self.assertEqual(t_r.steps, 2 * len(pl))

    def test_posthoc_timing_runs_primary_then_router_on_a_fresh_engine(self):
        t_j, _, joint, pl = self._run(timing="joint")
        t_h, _, posthoc, _ = self._run(timing="posthoc")
        joint_primary = [r for r in joint if r[0] == "primary"]
        joint_router = [r for r in joint if r[0] == "router"]
        self.assertEqual(posthoc, joint_primary + joint_router)
        self.assertEqual(t_j.steps, 2 * len(pl))
        self.assertEqual(t_h.steps, 2 * 2 * len(pl))              # primary pass + router pass
        self.assertEqual(t_h.reinit_calls, [2 * len(pl)])          # one fresh engine, same length
        self.assertIsNone(t_h._joint_branches)
        self.assertEqual(t_h.saved, ["1_prephase2"])               # pre-correction checkpoint
        self.assertEqual(t_j.saved, [])

    def test_posthoc_router_selection_is_even(self):
        for total, frac in ((553, 0.2), (237, 0.2), (12, 0.25), (10, 1.0)):
            keep, select = S.posthoc_router_selection(total, frac)
            chosen = [u for u in range(total) if select(u)]
            self.assertEqual(len(chosen), keep)
            self.assertEqual(keep, max(1, round(total * frac)))
            gaps = {b - a for a, b in zip(chosen, chosen[1:])}
            self.assertLessEqual(max(gaps, default=0) - min(gaps, default=0), 1)   # evenly spread

    def test_posthoc_fraction_forwards_a_subset_of_the_joint_router_batches(self):
        _, _, joint, pl = self._run(timing="joint")
        saved = S.POSTHOC_ROUTER_FRAC
        try:
            S.POSTHOC_ROUTER_FRAC = 0.25
            t_h, _, posthoc, _ = self._run(timing="posthoc")
        finally:
            S.POSTHOC_ROUTER_FRAC = saved
        n = 2 * len(pl)
        keep = round(n * 0.25)
        joint_router = [r for r in joint if r[0] == "router"]
        post_router = [r for r in posthoc if r[0] == "router"]
        self.assertEqual([r for r in posthoc if r[0] == "primary"],
                         [r for r in joint if r[0] == "primary"])
        it = iter(joint_router)                                   # ordered subsequence
        self.assertTrue(all(any(r == j for j in it) for r in post_router))
        self.assertTrue(0 < len(post_router) < len(joint_router))
        self.assertEqual(t_h.steps, n + keep)                     # primary N + router K updates
        self.assertEqual(t_h.reinit_calls, [keep])                # schedules sized to K
        self.assertIsNone(t_h._joint_router_update_select)

    def test_branch_gradients(self):
        # primary-only never touches r_res / skip; router-only never touches the experts
        _, m_p, _, _ = self._run(branches=("primary",))
        _, m_r, _, _ = self._run(branches=("router",))
        ref = self._model()
        for model, frozen in ((m_p, ("mres_reservoir", "residual_router")),
                              (m_r, (".experts.",))):
            for (name, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
                if any(f in name for f in frozen):
                    self.assertTrue(torch.equal(p, q), name)
        moved = [n for (n, p), (_, q) in zip(m_r.named_parameters(), ref.named_parameters())
                 if "mres_reservoir" in n and not torch.equal(p, q)]
        self.assertTrue(moved)

    def test_default_is_joint(self):
        self.assertEqual(S.ROUTER_FT_TIMING, "joint")
        self.assertEqual(S.ROUTER_FT_OBJECTIVE, "lm")
        self.assertEqual(MR.MassReservoirConfig().new_row, "reservoir")


if __name__ == "__main__":
    unittest.main()
