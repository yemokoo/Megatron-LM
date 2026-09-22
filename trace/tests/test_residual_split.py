"""Unit tests for scripts/residual/train_residual_v3_split.py.

Each mechanism of the split-gradient residual recipe is checked on a tiny
Llama (2 layers, hidden 32) so the tests run on CPU in seconds; the
checkpoint/loader roundtrip runs on GPU 0 when one is available.
"""
import copy
import inspect
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "implementations" / "llmcl_benchmark"
for path in (IMPL, ROOT / "scripts" / "residual", ROOT / "scripts" / "selfgen"):
    sys.path.insert(0, str(path))

import train_residual_v3_split as S            # noqa: E402  installs the patches
import train_residual_v3 as TR                 # noqa: E402
from model import Ours_LoRA_MoE_V3 as V3       # noqa: E402

LLAMA_TOKENIZER = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
torch.manual_seed(0)


def tiny_config(vocab=128):
    return LlamaConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False)


def make_v3(vocab=128, experts=1):
    torch.manual_seed(0)
    base = LlamaForCausalLM(tiny_config(vocab)).eval()
    model = copy.deepcopy(base)
    V3.attach_shared_qkvo_lora_moe(
        model, r=4, alpha=8, top_k=1, aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    V3.add_v3_experts(model, experts)      # patched: grow + residual + row copy
    return base, model


def routers(model):
    return [layer.shared_expert_router for layer in V3.shared_router_layers(model)]


def batch(vocab=128, n=2, length=12):
    ids = torch.randint(5, vocab, (n, length))
    mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}


def lm_and_router_loss(model, b):
    V3.set_v3_router_token_mask(model, b["attention_mask"])
    out = model(**b, use_cache=False)
    moe = V3.collect_v3_moe_losses(model)
    V3.set_v3_router_token_mask(model, None)
    return out.loss + (moe if moe is not None else 0.0)


def fake_trainer(model, **args):
    defaults = dict(global_rank=0, ablation_kd_init="off", ablation_phase_mode="1phase")
    defaults.update(args)
    return types.SimpleNamespace(raw_model=model, args=types.SimpleNamespace(**defaults))


class GrowthAndInit(unittest.TestCase):
    def test_residual_attached_and_new_row_copies_residual(self):
        _, model = make_v3()
        for router in routers(model):
            self.assertIsNotNone(router.residual_router)
            self.assertEqual(router.router.out_features, 1)
            # residual starts at zero, the copied row therefore is zero too
            self.assertTrue(torch.equal(router.router.weight[0], router.residual_router.weight[0]))
        # train the residual row a bit, then grow: the new row must equal it
        for router in routers(model):
            with torch.no_grad():
                router.residual_router.weight.normal_()
        old_rows = [r.router.weight.detach().clone() for r in routers(model)]
        V3.add_v3_experts(model, 1)
        for router, old in zip(routers(model), old_rows):
            self.assertEqual(router.router.out_features, 2)
            self.assertTrue(torch.equal(router.router.weight[:1], old))
            self.assertTrue(torch.equal(router.router.weight[1], router.residual_router.weight[0]))
            self.assertFalse(torch.equal(router.router.weight[1], router.router.weight[0]))

    def test_new_expert_and_residual_have_equal_logits(self):
        _, model = make_v3()
        for router in routers(model):
            with torch.no_grad():
                router.residual_router.weight.normal_()
        V3.add_v3_experts(model, 1)
        hidden = torch.randn(7, 32)
        for router in routers(model):
            expert_logits = router.router(hidden)
            res = router.residual_router(hidden)
            self.assertTrue(torch.allclose(expert_logits[:, 1:2], res))

    def test_new_expert_lora_b_zero_makes_model_a_noop(self):
        base, model = make_v3()
        for layer in V3.shared_router_layers(model):
            for pool in [layer.mlp.experts] + [p.experts for p in layer.attention_expert_projections]:
                for expert in pool:
                    for name, param in expert.named_parameters():
                        if name.endswith(".B") or name == "B":
                            self.assertEqual(float(param.abs().max()), 0.0)
        b = batch()
        with torch.no_grad():
            ref = base(**b, use_cache=False).logits
            got = model(**b, use_cache=False).logits
        self.assertTrue(torch.allclose(ref, got, atol=1e-5))

    def test_kd_off_is_enforced(self):
        _, model = make_v3()
        fake = fake_trainer(model, ablation_kd_init="on")
        with self.assertRaises(ValueError):
            S.Trainer.train_one_task(fake, "C-STANCE", 0, 1)
        fake = fake_trainer(model, ablation_phase_mode="2phase")
        with self.assertRaises(ValueError):
            S.Trainer.train_one_task(fake, "C-STANCE", 0, 1)


class SplitGradient(unittest.TestCase):
    def setUp(self):
        _, self.model = make_v3()
        for router in routers(self.model):
            with torch.no_grad():
                router.residual_router.weight.normal_(std=0.5)
                router.router.weight.normal_(std=0.5)
        with torch.no_grad():                       # a zero-B expert has no routing
            for p in V3._v3_expert_parameters(self.model):   # gradient from the LM loss
                p.normal_(std=0.2)
        V3.freeze_v3_experts(self.model, {0})
        V3.freeze_v3_routers(self.model, True)       # patched: residual trainable too
        self.model.train()
        self.fake = fake_trainer(self.model)

    def residual_grads(self):
        return [r.residual_router.weight.grad for r in routers(self.model)]

    def router_grads(self):
        return [r.router.weight.grad for r in routers(self.model)]

    def expert_params(self):
        return V3._v3_expert_parameters(self.model)

    def test_zero_b_expert_gives_no_router_gradient_from_lm_loss(self):
        _, fresh = make_v3()                          # B == 0 everywhere
        V3.freeze_v3_routers(fresh, True); fresh.train()
        with V3.Ours_LoRA_MoE_V3._suppress_replay_router_losses(fake_trainer(fresh)):
            lm_and_router_loss(fresh, batch()).backward()
        for router in routers(fresh):
            for g in (router.router.weight.grad, router.residual_router.weight.grad):
                self.assertTrue(g is None or float(g.abs().sum()) == 0.0)

    def test_primary_pass_masks_residual_out_and_drops_its_gradient(self):
        # make the residual the preferred choice so masking is actually tested
        for router in routers(self.model):
            with torch.no_grad():
                router.residual_router.weight.copy_(router.router.weight[0] + 3.0)
            router._capture_probe_routing = True
        self.fake._gradient_memory_enabled = lambda: False
        S.Trainer._begin_gradient_memory_batch(self.fake)          # primary forward starts
        for router in routers(self.model):
            self.assertEqual(router._residual_log_alpha, float("-inf"))
        loss = lm_and_router_loss(self.model, batch())
        self.assertTrue(torch.isfinite(loss))
        for router in routers(self.model):
            self.assertFalse(bool((router._last_probe_indices == -1).any()),
                             "residual must never be selected in the primary pass")
        loss.backward()
        for g in self.residual_grads():                            # -inf logit => no gradient
            self.assertTrue(g is None or float(g.abs().sum()) == 0.0)
        S.Trainer._after_primary_backward(self.fake)               # primary backward done
        for router in routers(self.model):
            self.assertEqual(router._residual_log_alpha, 0.0)
        for g in self.residual_grads():
            self.assertTrue(g is None or float(g.abs().sum()) == 0.0)
        self.assertTrue(all(g is not None and float(g.abs().sum()) > 0 for g in self.router_grads()))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in self.expert_params()))
        # with the mask lifted the same input routes to the residual again
        with torch.no_grad():
            lm_and_router_loss(self.model, batch())
        self.assertTrue(any(bool((r._last_probe_indices == -1).any()) for r in routers(self.model)))

    def test_after_primary_hook_zeroes_leftover_residual_gradient(self):
        lm_and_router_loss(self.model, batch()).backward()       # unmasked: residual gets grad
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0 for g in self.residual_grads()))
        S.Trainer._after_primary_backward(self.fake)
        for g in self.residual_grads():
            self.assertEqual(float(g.abs().sum()), 0.0)

    def test_router_only_pass_trains_residual_and_not_experts(self):
        b = batch()
        with V3.Ours_LoRA_MoE_V3._router_only_replay(self.fake), \
                V3.Ours_LoRA_MoE_V3._suppress_replay_router_losses(self.fake):
            lm_and_router_loss(self.model, b).backward()
        for p in self.expert_params():
            self.assertIsNone(p.grad)
            self.assertTrue(p.requires_grad)          # restored after the context
        self.assertTrue(all(g is not None and float(g.abs().sum()) > 0 for g in self.residual_grads()))
        self.assertTrue(all(g is not None and float(g.abs().sum()) > 0 for g in self.router_grads()))

    def test_extra_router_batches_reuse_one_source_share_of_primary(self):
        primary = batch(n=8)
        for sources, expect in ((1, 8), (2, 4), (4, 2), (8, 1), (16, 1)):
            self.fake._router_ft_sources = sources
            extra = S.Trainer._extra_router_replay_batches(self.fake, primary)
            self.assertEqual(len(extra), 1)
            self.assertEqual(extra[0]["input_ids"].shape[0], expect)
            self.assertTrue(torch.equal(extra[0]["labels"], primary["labels"][:expect]))

    def test_joint_loop_invokes_both_hooks(self):
        src = inspect.getsource(V3.Ours_LoRA_MoE_V3._run_v2_joint_epochs)
        self.assertIn("self._after_primary_backward()", src)
        self.assertIn("self._extra_router_replay_batches(primary)", src)
        self.assertLess(src.index("self._after_primary_backward()"),
                        src.index("self._extra_router_replay_batches(primary)"))
        # the extra forward is router-only and its gradient is the batch mean
        extra = src[src.index("self._extra_router_replay_batches(primary)"):]
        self.assertIn("self._router_only_replay()", extra[:1200])
        self.assertIn("* replay_loss_scale", extra[:1400])
        self.assertIn("* extra_losses.sum()).backward()", extra[:1400])


class BackboneBoSMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
        if cls.tok.pad_token_id is None:        # the trainer does the same
            cls.tok.pad_token = cls.tok.eos_token
        cls.delegated = []

        def stub_ensure(self, task):
            cls.delegated.append(task)
            return f"subset:{task}"
        cls.prev_ensure = S.Trainer._ensure_fixed_task_subset
        cls.prev_names = S.Trainer._memory_task_names
        S.Trainer._ensure_fixed_task_subset = stub_ensure
        S.install_bos_memory()

    @classmethod
    def tearDownClass(cls):
        S.Trainer._ensure_fixed_task_subset = cls.prev_ensure
        S.Trainer._memory_task_names = cls.prev_names

    def fake(self, **args):
        base = dict(global_rank=0, use_pretokenized_train_cache=True, max_train_len=64)
        base.update(args)
        fake = types.SimpleNamespace(
            args=types.SimpleNamespace(**base),
            tokenizer=self.tok, _fixed_task_subsets={}, _fixed_task_subset_indices={},
            train_task_list=["C-STANCE", "FOMC", "MeetingBank"])
        fake._v2_new_persistent_samples_per_task = lambda: 500
        return fake

    def test_memory_task_names_append_bos_pseudo_task_from_task_zero(self):
        fake = self.fake()
        self.assertEqual(S.Trainer._memory_task_names(fake, 0), [S.BOS_TASK])
        self.assertEqual(S.Trainer._memory_task_names(fake, 2), ["C-STANCE", "FOMC", S.BOS_TASK])

    def test_bos_subset_is_500_bounded_token_records(self):
        fake = self.fake()
        ds = S.Trainer._ensure_fixed_task_subset(fake, S.BOS_TASK)
        self.assertEqual(len(ds), 500)
        self.assertEqual(fake._fixed_task_subset_indices[S.BOS_TASK], list(range(500)))
        for i in (0, 123, 499):
            item = ds[i]
            self.assertEqual(item["input_ids"][0], self.tok.bos_token_id)
            self.assertEqual(item["input_ids"][-1], self.tok.eos_token_id)
            self.assertLessEqual(len(item["input_ids"]), 64)
            self.assertTrue(item["prompt"].startswith(S.BOS_TASK))
        self.assertIs(S.Trainer._ensure_fixed_task_subset(fake, S.BOS_TASK), ds)   # cached
        # collates through the pretokenized path with full labels
        from utils.data.data_collator import PreTokenizedSLoRATraceDataCollator
        collated = PreTokenizedSLoRATraceDataCollator(self.tok)([ds[0], ds[1]])
        self.assertEqual(collated["input_ids"].shape[0], 2)
        valid = collated["labels"].ne(-100)
        self.assertTrue(torch.equal(valid, collated["attention_mask"].bool()))

    def test_real_tasks_delegate_to_previous_ensure(self):
        fake = self.fake()
        self.delegated.clear()
        self.assertEqual(S.Trainer._ensure_fixed_task_subset(fake, "FOMC"), "subset:FOMC")
        self.assertEqual(self.delegated, ["FOMC"])

    def test_bos_requires_pretokenized_cache(self):
        fake = self.fake(use_pretokenized_train_cache=False)
        with self.assertRaises(RuntimeError):
            S.Trainer._ensure_fixed_task_subset(fake, S.BOS_TASK)


@unittest.skipUnless(torch.cuda.is_available(), "loader builds on the current CUDA device")
class CheckpointRoundtrip(unittest.TestCase):
    def test_save_meta_and_residual_loader_rebuild_identical_model(self):
        tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
        tmp = Path(tempfile.mkdtemp(prefix="residual_split_"))
        try:
            base_dir, ckpt_dir = tmp / "base", tmp / "ckpt"
            base = LlamaForCausalLM(tiny_config(vocab=len(tok)))
            base.save_pretrained(base_dir); tok.save_pretrained(base_dir)
            model = copy.deepcopy(base)
            V3.attach_shared_qkvo_lora_moe(
                model, r=4, alpha=8, top_k=1, aux_loss_coeff=0.01, z_loss_coeff=0.001,
                routing_weight_mode="straight_through_topk", dropout=0.0)
            V3.add_v3_experts(model, 2)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if any(s in name for s in V3.Ours_LoRA_MoE_V3.save_key_substrings):
                        param.normal_(std=0.3)
            state = {k: v for k, v in model.state_dict().items()
                     if any(s in k for s in V3.Ours_LoRA_MoE_V3.save_key_substrings)}
            self.assertTrue(any(TR.RESIDUAL_KEY in k for k in state))
            ckpt_dir.mkdir()
            torch.save(state, ckpt_dir / "pytorch_model.bin")
            args = types.SimpleNamespace(
                training_version="v3", experts_per_task=1,   # v3_new meta needs live trainer state
                dataset_name="C-STANCE,FOMC", ablation_kd_init="off",
                ablation_phase_mode="1phase", ablation_replay_source="selfgen",
                v2_new_active_memory_cap=5000, v2_new_persistent_samples_per_task=500,
                replay_selection_mode="random", replay_subset_seed=2025,
                v2_memory_batch_size=0, v2_kd_loss_coeff=0.0, v2_kd_temperature=1.0,
                v2_kd_learning_rate=0, v2_kd_chunk_tokens=256, v2_kd_token_scope="nonpad",
                v2_joint_replay_loss_coeff=1.0, v2_max_replay_batches_per_step=0,
                v2_replay_forward_batch_size=8, v2_kd_memory_batch_size=8,
                v2_joint_new_to_replay_ratio=1, v2_joint_replay_objective="lm",
                v2_kd_pass_multiplier=1, v2_hidden_mse_loss_coeff=1.0,
                router_replay_exposure_samples=5000, replay_distribution="equal_task",
                replay_subset_ratio=0.1, replay_recency_power=1.0, router_retune_epochs=0,
                train_format="slora_chat_full", max_prompt_len=1024, max_ans_len=512,
                max_train_len=1024, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                resolved_dataset_order=["C-STANCE", "FOMC"], stop_after_task="",
                v3_kd_init_step_fraction=1.0)
            V3.save_v3_meta(model, str(ckpt_dir), args, trainer=None)
            meta = json.load(open(ckpt_dir / V3.V3_META_NAME))
            self.assertEqual(meta["residual_expert"]["variant"], "split_gradient")
            self.assertEqual(meta["residual_expert"]["count"], 1)
            self.assertEqual(meta["num_experts"], 2)
            import residual_expert as RE
            loaded, loaded_meta = RE.load_v3_residual_checkpoint(
                str(ckpt_dir), tok, str(base_dir), device="cuda", dtype=torch.float32)
            self.assertEqual(loaded_meta["residual_expert"]["variant"], "split_gradient")
            loaded_state = loaded.state_dict()
            for key, value in state.items():
                self.assertTrue(torch.equal(loaded_state[key].cpu(), value), key)
            b = {k: v.cuda() for k, v in batch(vocab=len(tok), n=2, length=10).items()}
            model = model.cuda().float().eval()
            with torch.no_grad():
                ref = model(**b, use_cache=False).logits.cpu()
                got = loaded(**b, use_cache=False).logits.cpu()
            self.assertTrue(torch.allclose(ref, got, atol=1e-4))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class LastRouterBias(unittest.TestCase):
    """--method last_bias --site router_logits on a residual-from-task-0 model."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts" / "bos_token"))
        import train_bos_token as TB
        self.TB = TB
        _, self.model = make_v3(experts=2)
        for router in routers(self.model):
            with torch.no_grad():
                router.router.weight.normal_(std=0.5); router.residual_router.weight.normal_(std=0.5)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def test_bias_size_covers_experts_plus_residual(self):
        self.assertEqual(self.TB.last_router_bias_size(self.model), 3)
        with self.assertRaises(ValueError):
            self.TB.install_last_bias(self.model, "router_logits", torch.zeros(2))

    def test_bias_moves_last_layer_routing_only_and_is_removable(self):
        last = routers(self.model)[-1]; first = routers(self.model)[0]
        for r in (first, last): r._capture_probe_routing = True
        b = batch(n=2, length=16)
        with torch.no_grad(): self.model(**b, use_cache=False)
        idx0_first, idx0_last = first._last_probe_indices.clone(), last._last_probe_indices.clone()
        bias = torch.zeros(3); bias[2] = 50.0                        # force the residual at the last layer
        handle = self.TB.install_last_bias(self.model, "router_logits", bias)
        with torch.no_grad(): self.model(**b, use_cache=False)
        self.assertTrue(bool((last._last_probe_indices == -1).all()))
        self.assertTrue(torch.equal(first._last_probe_indices, idx0_first))
        handle.remove()
        with torch.no_grad(): self.model(**b, use_cache=False)
        self.assertTrue(torch.equal(last._last_probe_indices, idx0_last))

    def test_only_the_bias_gets_gradient(self):
        bias = torch.nn.Parameter(torch.zeros(3))
        self.TB.install_last_bias(self.model, "router_logits", bias)
        self.model.train()
        lm_and_router_loss(self.model, batch(n=2, length=16)).backward()
        self.assertIsNotNone(bias.grad)
        self.assertGreater(float(bias.grad.abs().sum()), 0.0)
        others = [(n, p) for n, p in self.model.named_parameters() if not n.endswith("_logit_bias")]
        self.assertTrue(all(p.grad is None for _, p in others))
        self.assertEqual([n for n, p in self.model.named_parameters() if p.grad is not None],
                         ["model.layers.1.shared_expert_router._logit_bias"])


class AllLayerRouterBias(unittest.TestCase):
    """--method all_layer_bias: one bias row per shared-router layer."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts" / "bos_token"))
        import train_bos_token as TB
        self.TB = TB
        _, self.model = make_v3(experts=1)
        for router in routers(self.model):
            with torch.no_grad():
                router.router.weight.normal_(std=0.5); router.residual_router.weight.normal_(std=0.5)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def test_bias_shape_is_layers_by_slots(self):
        n_layers = len(routers(self.model))
        bias = torch.zeros(n_layers, 2)
        self.TB.install_all_layer_bias(self.model, bias)
        for router in routers(self.model):
            self.assertIsNotNone(router._logit_bias)
        with self.assertRaises(ValueError):
            self.TB.install_all_layer_bias(self.model, torch.zeros(n_layers, 5))

    def test_every_layer_gets_its_own_row_and_moves_routing(self):
        rs = routers(self.model)
        for r in rs: r._capture_probe_routing = True
        b = batch(n=2, length=10)
        with torch.no_grad(): self.model(**b, use_cache=False)
        baseline = [r._last_probe_indices.clone() for r in rs]

        n_layers = len(rs)
        bias = torch.zeros(n_layers, 2); bias[:, 1] = 50.0     # force residual everywhere
        handle = self.TB.install_all_layer_bias(self.model, bias)
        with torch.no_grad(): self.model(**b, use_cache=False)
        for r in rs:
            self.assertTrue(bool((r._last_probe_indices == -1).all()))
        handle.remove()
        with torch.no_grad(): self.model(**b, use_cache=False)
        for r, base in zip(rs, baseline):
            self.assertTrue(torch.equal(r._last_probe_indices, base))

    def test_gradient_flows_to_every_layer_row_independently(self):
        n_layers = len(routers(self.model))
        bias = torch.nn.Parameter(torch.zeros(n_layers, 2))
        self.TB.install_all_layer_bias(self.model, bias)
        self.model.train()
        lm_and_router_loss(self.model, batch(n=2, length=10)).backward()
        self.assertIsNotNone(bias.grad)
        self.assertTrue((bias.grad.abs().sum(dim=1) > 0).all(), "every layer row should get gradient")
        self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters() if p is not bias))


class BosGuard(unittest.TestCase):
    """bos_guard.install_bos_guard: force residual at BOS positions, layers != last."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts" / "residual"))
        import bos_guard
        cls.bg = bos_guard
        # the true un-guarded forward: if a guard was installed by an earlier test class, it is _prev_forward
        cls._clean_forward = staticmethod(bos_guard._prev_forward or V3.SharedExpertRouter.forward)

    @classmethod
    def tearDownClass(cls):
        # leave no guard behind for the other test classes (module-level forward patch + stale mask)
        V3.SharedExpertRouter.forward = cls._clean_forward
        cls.bg._prev_forward = None; cls.bg._bos_mask["value"] = None; cls.bg._bos_mask["full"] = None

    def setUp(self):
        V3.SharedExpertRouter.forward = self._clean_forward  # undo any previous test's install
        self.bg._prev_forward = None; self.bg._bos_mask["value"] = None
        _, self.model = make_v3(experts=1, vocab=200)
        for router in routers(self.model):
            with torch.no_grad():
                router.router.weight.normal_(std=0.5); router.residual_router.weight.normal_(std=0.5)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.BOS = 199
        self.bg.install_bos_guard(self.model, bos_token_id=self.BOS)

    def test_layers_tagged_only_last_is_last(self):
        rs = routers(self.model)
        flags = [r._is_last_layer for r in rs]
        self.assertEqual(flags, [False] * (len(rs) - 1) + [True])

    def test_non_last_layers_force_residual_at_bos_positions(self):
        rs = routers(self.model)
        for r in rs: r._capture_probe_routing = True
        ids = torch.tensor([[self.BOS, 5, self.BOS, 7]])
        att = torch.ones_like(ids)
        with torch.no_grad():
            self.model(input_ids=ids, attention_mask=att, use_cache=False)
        bos_positions = torch.tensor([True, False, True, False])
        for r in rs[:-1]:
            probe = r._last_probe_indices.reshape(-1)
            self.assertTrue(bool((probe[bos_positions] == -1).all()), "non-last layer must force residual at BOS")
        last_probe = rs[-1]._last_probe_indices.reshape(-1)
        # last layer is NOT forced -- routing there is whatever the trained router+bias decides
        self.assertTrue(True)  # no assertion on the last layer's choice itself

    def test_mask_persists_after_forward_for_checkpoint_recompute(self):
        # No post-hook clears the mask: gradient checkpointing recomputes a
        # layer's forward during backward by calling the LAYER directly, not
        # the top-level model, so the mask must still be the correct one at
        # that point -- clearing it right after the top-level call returns
        # would make the recompute take a different path than the original
        # forward and trip torch.utils.checkpoint's determinism check.
        ids = torch.tensor([[self.BOS, 5]])
        att = torch.ones_like(ids)
        with torch.no_grad():
            self.model(input_ids=ids, attention_mask=att, use_cache=False)
        self.assertIsNotNone(self.bg._bos_mask["value"])
        self.assertTrue(torch.equal(self.bg._bos_mask["value"], ids == self.BOS))

    def test_next_forward_overwrites_stale_mask(self):
        ids1 = torch.tensor([[self.BOS, 5]]); att1 = torch.ones_like(ids1)
        with torch.no_grad():
            self.model(input_ids=ids1, attention_mask=att1, use_cache=False)
        ids2 = torch.tensor([[5, 7]]); att2 = torch.ones_like(ids2)  # no BOS at all
        with torch.no_grad():
            self.model(input_ids=ids2, attention_mask=att2, use_cache=False)
        self.assertFalse(bool(self.bg._bos_mask["value"].any()))

    def test_no_bos_in_batch_leaves_routing_untouched(self):
        rs = routers(self.model)
        for r in rs: r._capture_probe_routing = True
        ids = torch.tensor([[5, 7, 9]])
        att = torch.ones_like(ids)
        with torch.no_grad():
            self.model(input_ids=ids, attention_mask=att, use_cache=False)
        for r in rs[:-1]:
            probe = r._last_probe_indices.reshape(-1)
            # with no BOS present the guard is a no-op: some real routing happens (not
            # all forced to residual), i.e. it should not be unconditionally -1 everywhere
            self.assertFalse(bool((probe == -1).all()))


class ChatHeaderPrompt(unittest.TestCase):
    """--loss-scope after_header (train_bos_token) / --prompt-mode chat_header (gen_doc):
    the fixed Llama-3.1 chat-template header is supplied, never learned or generated."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts" / "bos_token"))
        import train_bos_token as TB
        cls.TB = TB
        cls.tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)

    def test_header_ids_match_collator_prefix(self):
        from utils.data.data_collator import SLoRATraceDataCollator
        coll = SLoRATraceDataCollator(tokenizer=self.tok, max_length=1024)
        hdr = self.TB.chat_header_ids(self.tok)
        self.assertEqual(hdr[:2], [self.tok.bos_token_id] * 2)
        for rec in ({"prompt": "判断以下文本对指定对象的态度\n对象：\n甲\n态度：", "answer": "A"},
                    {"prompt": "What is the stance?\nStance:", "answer": "B"}):
            ids, _ = coll._encode(rec)
            self.assertEqual(ids[:len(hdr)], hdr)
        # the header ends exactly at the user-turn content
        self.assertTrue(self.tok.decode(hdr).endswith("<|start_header_id|>user<|end_header_id|>\n\n"))

    def test_pad_batch_label_start_masks_header_only(self):
        docs = [[7, 8, 9, 10, 11], [7, 8, 9]]
        ids, att, lab = self.TB.pad_batch(docs, pad_id=0, label_start=2)
        self.assertEqual(lab[0].tolist(), [-100, -100, 9, 10, 11])
        self.assertEqual(lab[1].tolist(), [-100, -100, 9, -100, -100])
        self.assertEqual(att[1].tolist(), [1, 1, 1, 0, 0])
        ids2, _, lab2 = self.TB.pad_batch(docs, pad_id=0)                 # default: full loss, unchanged
        self.assertEqual(lab2[0].tolist(), [7, 8, 9, 10, 11]); self.assertTrue(torch.equal(ids, ids2))

    def test_gen_doc_chat_header_prompt_parses_and_stops_once_fewer(self):
        import gen_doc as G
        hdr = self.TB.chat_header_ids(self.tok)
        given = G.SYSTEM_TEXT + G.USER_HDR
        self.assertEqual(self.tok.decode(hdr[2:]), given)
        self.assertEqual(3 - given.count("<|eot_id|>"), 2)               # user eot + assistant eot remain
        gen = "判断以下文本\n对象：\n甲\n态度：<|eot_id|>" + G.ASST_HDR + "A<|eot_id|>"
        rec = G.parse_doc(given + gen)
        self.assertTrue(rec["system_ok"]); self.assertTrue(rec["complete"])
        self.assertEqual(rec["user"], "判断以下文本\n对象：\n甲\n态度："); self.assertEqual(rec["answer"], "A")


class HeaderGuard(unittest.TestCase):
    """bos_guard with header_ids: header body -> residual at EVERY layer, last header token ->
    residual at layers 1..L-1 (last layer free), everything else -> plain BOS rule."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts" / "residual"))
        import bos_guard
        cls.bg = bos_guard
        cls._clean_forward = staticmethod(bos_guard._prev_forward or V3.SharedExpertRouter.forward)

    @classmethod
    def tearDownClass(cls):
        # leave no guard behind for the other test classes (module-level forward patch + stale mask)
        V3.SharedExpertRouter.forward = cls._clean_forward
        cls.bg._prev_forward = None; cls.bg._bos_mask["value"] = None; cls.bg._bos_mask["full"] = None

    def setUp(self):
        V3.SharedExpertRouter.forward = self._clean_forward
        self.bg._prev_forward = None; self.bg._bos_mask["value"] = None; self.bg._bos_mask["full"] = None
        _, self.model = make_v3(experts=1, vocab=200)
        for router in routers(self.model):
            with torch.no_grad():
                router.router.weight.normal_(std=0.5); router.residual_router.weight.normal_(std=0.5)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.BOS = 199
        self.HDR = [self.BOS, self.BOS, 150, 151, 152, 153]      # toy header: body = first 5, decision = 153
        self.bg.install_bos_guard(self.model, bos_token_id=self.BOS, header_ids=self.HDR)

    def test_header_masks_position_independent_and_absent_when_short(self):
        ids = torch.tensor([[0, 0] + self.HDR + [5, 6],           # left padded row
                            self.HDR + [7, 8, 9, 10],             # row starting with the header
                            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]) # no header
        full, part = self.bg.header_masks(ids, self.HDR)
        self.assertEqual(full[0].nonzero().flatten().tolist(), [2, 3, 4, 5, 6]); self.assertEqual(part[0].nonzero().flatten().tolist(), [7])
        self.assertEqual(full[1].nonzero().flatten().tolist(), [0, 1, 2, 3, 4]); self.assertEqual(part[1].nonzero().flatten().tolist(), [5])
        self.assertFalse(bool(full[2].any())); self.assertFalse(bool(part[2].any()))
        self.assertEqual(self.bg.header_masks(torch.tensor([[153]]), self.HDR), (None, None))   # decode step
        self.assertEqual(self.bg.header_masks(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), self.HDR), (None, None))

    def _routed(self, ids):
        """expert index chosen at every position per layer, as the guarded forward returns it."""
        outs = {}
        hs = [r.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o.expert_indices.reshape(-1).clone()))
              for k, r in enumerate(routers(self.model))]
        with torch.no_grad():
            self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        for h in hs: h.remove()
        return [outs[k] for k in range(len(hs))]

    def test_routing_header_body_all_layers_decision_token_non_last_only(self):
        n_exp = routers(self.model)[0].router.weight.shape[0]     # residual slot index == num_experts
        ids = torch.tensor([self.HDR + [5, 6, 7, 8]])
        body = torch.tensor([True] * 5 + [False] * 5)
        decision = torch.tensor([False] * 5 + [True] + [False] * 4)
        guarded = self._routed(ids)
        for sel in guarded[:-1]:
            self.assertTrue(bool((sel[body | decision] == n_exp).all()))
        self.assertTrue(bool((guarded[-1][body] == n_exp).all()), "header body must be residual at the last layer too")
        V3.SharedExpertRouter.forward = self._clean_forward            # same model, guard off
        free = self._routed(ids)
        V3.SharedExpertRouter.forward = self.bg._bos_guarded_forward
        self.assertEqual(int(guarded[-1][decision][0]), int(free[-1][decision][0]))   # decision token: last layer free
        for g, f in zip(guarded, free):                                            # content positions untouched
            self.assertTrue(torch.equal(g[~(body | decision)], f[~(body | decision)]))
        self.assertTrue(any(bool((f[body] != n_exp).any()) for f in free[:-1]), "toy router must not be trivially residual")

    def test_bos_outside_header_keeps_plain_rule(self):
        n_exp = routers(self.model)[0].router.weight.shape[0]
        ids = torch.tensor([[self.BOS, 5, 6, 7, 8, 9, 10]])          # backbone BoS doc, no header
        guarded = self._routed(ids)
        for sel in guarded[:-1]:
            self.assertEqual(int(sel[0]), n_exp)
        self.assertIsNone(self.bg._bos_mask["full"])

    def test_real_header_ids_fast_and_slow_tokenizer_agree(self):
        sys.path.insert(0, str(ROOT / "scripts" / "bos_token"))
        from train_bos_token import chat_header_ids
        slow = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER, use_fast=False)
        fast = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER, use_fast=True)
        self.assertEqual(chat_header_ids(slow), chat_header_ids(fast))
        self.assertEqual(len(chat_header_ids(slow)), 37); self.assertEqual(chat_header_ids(slow)[-1], 271)   # "\n\n"


class HeaderGuardInputsEmbeds(unittest.TestCase):
    """A guarded model called with inputs_embeds only gets no mask from the pre-hook; apply_input_ids
    installs it by hand and must give the same routing as the input_ids call."""

    def test_apply_input_ids_matches_input_ids_forward(self):
        sys.path.insert(0, str(ROOT / "scripts" / "residual"))
        import bos_guard as bg
        clean = bg._prev_forward or V3.SharedExpertRouter.forward
        V3.SharedExpertRouter.forward = clean; bg._prev_forward = None; bg._bos_mask["value"] = None; bg._bos_mask["full"] = None
        try:
            _, model = make_v3(experts=1, vocab=200)
            for router in routers(model):
                with torch.no_grad():
                    router.router.weight.normal_(std=0.5); router.residual_router.weight.normal_(std=0.5)
            model.eval()
            BOS, HDR = 199, [199, 199, 150, 151, 152, 153]
            bg.install_bos_guard(model, bos_token_id=BOS, header_ids=HDR)
            ids = torch.tensor([HDR + [5, 6, 7, 8]]); att = torch.ones_like(ids)
            outs = {}
            hs = [r.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o.expert_indices.reshape(-1).clone()))
                  for k, r in enumerate(routers(model))]
            with torch.no_grad():
                model(input_ids=ids, attention_mask=att, use_cache=False)
            ref = [outs[k] for k in range(len(hs))]
            bg.set_bos_position_mask(None, None)                      # stale state cleared
            emb = model.get_input_embeddings()(ids)
            with torch.no_grad():
                model(inputs_embeds=emb, attention_mask=att, use_cache=False)
            self.assertIsNone(bg._bos_mask["full"])                   # pre-hook saw no input_ids -> unguarded
            bg.apply_input_ids(model, ids)
            with torch.no_grad():
                model(inputs_embeds=emb, attention_mask=att, use_cache=False)
            for h in hs: h.remove()
            for k in range(len(hs)):
                self.assertTrue(torch.equal(outs[k], ref[k]))
        finally:
            V3.SharedExpertRouter.forward = clean; bg._prev_forward = None; bg._bos_mask["value"] = None; bg._bos_mask["full"] = None


class HeaderGuardDecisionToken(unittest.TestCase):
    def test_end_header_variant_drops_only_the_trailing_newlines(self):
        sys.path.insert(0, str(ROOT / "scripts" / "bos_token"))
        from train_bos_token import chat_header_ids
        tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER, use_fast=False)
        full, short = chat_header_ids(tok), chat_header_ids(tok, decision="end_header")
        self.assertEqual(full[:-1], short); self.assertEqual(len(short), 36)
        self.assertEqual(tok.decode(short[-1:]), "<|end_header_id|>"); self.assertEqual(tok.decode(full[-1:]), "\n\n")


class HeaderGuardNoneAndLabelMask(unittest.TestCase):
    """--guard-decision none: the whole 36-token header is residual at every layer, "\\n\\n" is free;
    HEADER_NO_LOSS: header tokens are never labels in the trainer's collators."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts" / "residual")); sys.path.insert(0, str(ROOT / "scripts" / "bos_token"))
        import bos_guard, train_bos_token
        cls.bg, cls.TB = bos_guard, train_bos_token
        cls._clean_forward = staticmethod(bos_guard._prev_forward or V3.SharedExpertRouter.forward)
        cls.tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER, use_fast=False)
        if cls.tok.pad_token_id is None:
            cls.tok.pad_token_id = cls.tok.eos_token_id

    @classmethod
    def tearDownClass(cls):
        V3.SharedExpertRouter.forward = cls._clean_forward
        cls.bg._prev_forward = None; cls.bg._bos_mask["value"] = None; cls.bg._bos_mask["full"] = None
        from utils.data import data_collator
        data_collator.HEADER_LABEL_MASK_IDS = None

    def test_spec_none_is_36_tokens_all_full(self):
        ids, all_full = self.TB.guard_header_spec(self.tok, "none")
        self.assertEqual(len(ids), 36); self.assertTrue(all_full)
        self.assertEqual(self.tok.decode(ids[-1:]), "<|end_header_id|>")
        ids2, f2 = self.TB.guard_header_spec(self.tok, "end_header"); self.assertEqual(ids, ids2); self.assertFalse(f2)
        self.assertEqual(len(self.TB.guard_header_spec(self.tok, "nn")[0]), 37)

    def test_header_masks_all_full_has_empty_part(self):
        HDR = [199, 199, 150, 151]
        ids = torch.tensor([[0] + HDR + [271, 5, 6]])
        full, part = self.bg.header_masks(ids, HDR, all_full=True)
        self.assertEqual(full[0].nonzero().flatten().tolist(), [1, 2, 3, 4]); self.assertFalse(bool(part.any()))

    def test_routing_none_mode_whole_header_residual_newline_free(self):
        V3.SharedExpertRouter.forward = self._clean_forward
        self.bg._prev_forward = None; self.bg._bos_mask["value"] = None; self.bg._bos_mask["full"] = None
        _, model = make_v3(experts=1, vocab=200)
        for router in routers(model):
            with torch.no_grad():
                router.router.weight.normal_(std=0.5); router.residual_router.weight.normal_(std=0.5)
        model.eval()
        HDR = [199, 199, 150, 151, 152]; NN = 153
        self.bg.install_bos_guard(model, bos_token_id=199, header_ids=HDR, header_all_full=True)
        n_exp = routers(model)[0].router.weight.shape[0]
        ids = torch.tensor([HDR + [NN, 5, 6, 7]])
        outs = {}
        hs = [r.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o.expert_indices.reshape(-1).clone())) for k, r in enumerate(routers(model))]
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        guarded = [outs[k] for k in range(len(hs))]
        V3.SharedExpertRouter.forward = self._clean_forward
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        free = [outs[k] for k in range(len(hs))]
        for h in hs: h.remove()
        hdr = torch.tensor([True] * 5 + [False] * 4)
        for g, f in zip(guarded, free):                       # every layer, including the last
            self.assertTrue(bool((g[hdr] == n_exp).all()))
            self.assertTrue(torch.equal(g[~hdr], f[~hdr]))    # "\n\n" and content untouched
        self.assertTrue(any(bool((f[hdr] != n_exp).any()) for f in free))

    def test_collator_header_label_mask_applies_to_header_docs_only(self):
        from utils.data import data_collator
        from utils.data.data_collator import SLoRATraceDataCollator, PreTokenizedSLoRATraceDataCollator
        hdr = self.TB.chat_header_ids(self.tok)                # 37 ids through "\n\n"
        rec = {"prompt": "判断以下文本\n对象：\n甲\n态度：", "answer": "A"}
        coll = SLoRATraceDataCollator(tokenizer=self.tok, max_length=256, label_scope="full")
        pre = PreTokenizedSLoRATraceDataCollator(self.tok)
        data_collator.HEADER_LABEL_MASK_IDS = None
        b0 = coll([rec]); self.assertEqual(int(b0["labels"][0, 0]), int(b0["input_ids"][0, 0]))   # full labels by default
        data_collator.HEADER_LABEL_MASK_IDS = hdr
        b1 = coll([rec])
        self.assertTrue(bool((b1["labels"][0, :37] == -100).all()))
        self.assertEqual(int(b1["labels"][0, 37]), int(b1["input_ids"][0, 37]))                    # first content token is a label
        self.assertTrue(torch.equal(b0["input_ids"], b1["input_ids"]))
        ids, _ = coll._encode(rec)
        b2 = pre([{"input_ids": ids, "prompt": "x"}, {"input_ids": [128000, 5, 6, 7, 128009], "prompt": "__backbone_bos__:0"}])
        self.assertTrue(bool((b2["labels"][0, :37] == -100).all()))
        self.assertEqual(b2["labels"][1, :5].tolist(), [128000, 5, 6, 7, 128009])                  # BoS pseudo-doc: untouched
        ev = SLoRATraceDataCollator(tokenizer=self.tok, max_length=256, label_scope="answer")([rec])
        self.assertTrue(bool((ev["labels"][0, :37] == -100).all()))                                # answer scope unchanged
        data_collator.HEADER_LABEL_MASK_IDS = None


class DecisionOnlyBias(unittest.TestCase):
    """--bias-positions decision: the logit bias/mask acts only at the first token after the header."""

    def test_bias_confined_to_decision_position(self):
        sys.path.insert(0, str(ROOT / "scripts" / "residual")); sys.path.insert(0, str(ROOT / "scripts" / "bos_token"))
        import bos_guard as bg, train_bos_token as TB
        clean = bg._prev_forward or V3.SharedExpertRouter.forward
        V3.SharedExpertRouter.forward = clean; bg._prev_forward = None
        for k in bg._bos_mask: bg._bos_mask[k] = None
        try:
            _, model = make_v3(experts=2, vocab=200)
            for router in routers(model):
                with torch.no_grad():
                    router.router.weight.normal_(std=0.5); router.residual_router.weight.normal_(std=0.5)
            model.eval()
            HDR = [199, 199, 150, 151, 152]; NN = 153
            bg.install_bos_guard(model, bos_token_id=199, header_ids=HDR, header_all_full=True)
            ids = torch.tensor([HDR + [NN, 5, 6, 7]])
            self.assertEqual(bg.decision_mask(ids, HDR)[0].nonzero().flatten().tolist(), [5])
            self.assertIsNone(bg.decision_mask(torch.tensor([[153]]), HDR))            # decode step
            # huge bias toward the residual, confined to the decision position: "\n\n" must be residual
            # at every layer, while the content positions (never biased) still pick real experts
            bias = torch.zeros(len(model.model.layers), 3); bias[:, 2] = 1e4
            TB.install_all_layer_bias(model, bias)
            for r in routers(model): r._logit_bias_positions = bg.decision_position_mask
            outs = {}
            hs = [r.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o.expert_indices.reshape(-1).clone())) for k, r in enumerate(routers(model))]
            with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            sel = [outs[k] for k in range(len(hs))]
            for r in routers(model): r._logit_bias_positions = None
            with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            everywhere = [outs[k] for k in range(len(hs))]
            for h in hs: h.remove()
            for g in sel: self.assertEqual(int(g[5]), 2)                                       # decision token -> residual
            self.assertTrue(any(int(g[i]) != 2 for g in sel for i in (6, 7, 8)))                # content: real experts still chosen
            self.assertTrue(all(int(g[i]) == 2 for g in everywhere for i in (5, 6, 7, 8)))      # same bias unconfined -> all residual
        finally:
            V3.SharedExpertRouter.forward = clean; bg._prev_forward = None
            for k in bg._bos_mask: bg._bos_mask[k] = None
