import math
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from model.base_model import CL_Base_Model
from utils.my_peft.tuners.lora import LoraLayer
from utils.utils import print_rank_0, to_device


class O_LoRA(CL_Base_Model):
    """Original O-LoRA semantics adapted to TreeLoRA's single-process loop."""

    def __init__(self, model, tokenizer, optimizer, train_task_list,
                 eval_task_list, test_task_list, args, lamda_1=0.5, lamda_2=0):
        super().__init__(model, tokenizer, optimizer, train_task_list,
                         eval_task_list, test_task_list, args)
        self.lamda_1, self.lamda_2 = lamda_1, lamda_2
        if args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device("cuda", args.local_rank)
        self._set_trainability()

    def _peft_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _layers(self):
        for module in self._peft_model().modules():
            if isinstance(module, LoraLayer):
                yield module

    def _set_trainability(self):
        for name, param in self._peft_model().named_parameters():
            if "loranew_" in name:
                param.requires_grad = True
            elif "lora_" in name:
                param.requires_grad = False

    def _regularization(self):
        orth = torch.zeros((), device=self.device)
        l2 = torch.zeros((), device=self.device)
        for layer in self._layers():
            key = layer.active_adapter
            if key not in layer.loranew_A:
                continue
            old_a = layer.lora_A[key].weight
            new_a = layer.loranew_A[key].weight
            new_b = layer.loranew_B[key].weight
            if old_a.shape[0]:
                orth = orth + torch.abs(old_a @ new_a.T).sum()
            l2 = l2 + torch.norm(new_a, p=2) + torch.norm(new_b, p=2)
        return orth, l2

    @staticmethod
    def _frozen_linear(weight):
        out_features, in_features = weight.shape
        result = nn.Linear(in_features, out_features, bias=False).to(
            device=weight.device, dtype=weight.dtype)
        result.weight.data.copy_(weight)
        result.weight.requires_grad = False
        return result

    def _clear_optimizer_state(self):
        """Reset Adam moments because the next task uses a fresh subspace.

        ZeRO-2 stores optimizer state against flattened partition parameters,
        not the original LoRA Parameter objects. All trainable parameters in
        O-LoRA are loranew_*, so clearing the wrapped optimizer states is the
        exact single-process equivalent of upstream O-LoRA starting a new
        optimizer for every task.
        """
        optimizer, visited = self.optimizer, set()
        while optimizer is not None and id(optimizer) not in visited:
            visited.add(id(optimizer))
            state = getattr(optimizer, "state", None)
            if isinstance(state, dict):
                state.clear()
            optimizer = (getattr(optimizer, "optimizer", None)
                         or getattr(optimizer, "base_optimizer", None))

    @torch.no_grad()
    def consolidate_task_adapter(self):
        """Append current rank-r adapter and initialize a fresh rank-r adapter."""
        accumulated_rank, count = None, 0
        for layer in self._layers():
            key = layer.active_adapter
            if key not in layer.loranew_A:
                continue
            old_a = layer.lora_A[key].weight.detach()
            old_b = layer.lora_B[key].weight.detach()
            new_a = layer.loranew_A[key].weight.detach()
            new_b = layer.loranew_B[key].weight.detach()
            combined_a = torch.cat((old_a, new_a), dim=0)
            combined_b = torch.cat((old_b, new_b), dim=1)
            layer.lora_A[key] = self._frozen_linear(combined_a)
            layer.lora_B[key] = self._frozen_linear(combined_b)

            # Keep these Parameter objects because DeepSpeed's optimizer owns them.
            nn.init.kaiming_uniform_(layer.loranew_A[key].weight, a=math.sqrt(5))
            nn.init.zeros_(layer.loranew_B[key].weight)

            rank = combined_a.shape[0]
            if accumulated_rank is not None and rank != accumulated_rank:
                raise RuntimeError(
                    f"O-LoRA ranks differ across layers: {accumulated_rank} vs {rank}")
            accumulated_rank, count = rank, count + 1

        if not count:
            raise RuntimeError("O-LoRA found no adapted LoraLayer")
        for config in self._peft_model().peft_config.values():
            config.r_sum = accumulated_rank
            # loranew_B is zero, so saving it preserves exact inference while
            # allowing a checkpoint to resume the next task.
            config.save_loranew = True
        self._clear_optimizer_state()
        self._set_trainability()
        return accumulated_rank, count

    def train_one_task(self, task, i_task, epochs):
        loader = self.train_task_list[task]
        bar = tqdm(total=epochs * len(loader), leave=True,
                   disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{epochs}, "
                f"Total Micro Batches {len(loader)}", self.args.global_rank)
            self.model.train()
            for step, batch in enumerate(loader):
                task_micro_step = epoch * len(loader) + step
                is_task_end = (epoch == epochs - 1 and step == len(loader) - 1)
                gas = self.args.gradient_accumulation_steps
                is_boundary = ((task_micro_step + 1) % gas == 0) or is_task_end
                if hasattr(self.model, "set_gradient_accumulation_boundary"):
                    self.model.set_gradient_accumulation_boundary(is_boundary)
                del batch["sources"]
                batch = to_device(batch, self.device)
                accuracy_loss = self.model(**batch, use_cache=False).loss
                orth, l2 = self._regularization()
                loss = accuracy_loss + self.lamda_1 * orth + self.lamda_2 * l2
                if self.args.global_rank == 0:
                    bar.update(1)
                    bar.set_description(
                        f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}",
                        refresh=False)
                self.model.backward(loss)
                self.model.step()
                if self.args.global_rank == 0 and step % 30 == 0:
                    print_rank_0(
                        f"orthogonal_loss: {orth.item()}; l2_loss: {l2.item()}; "
                        f"accuracy_loss: {accuracy_loss.item()}; "
                        f"total_loss: {loss.item()}; lambda1: {self.lamda_1}; "
                        f"lambda2: {self.lamda_2}", self.args.global_rank)
        bar.close()

    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(
                task, i_task, int(self.args.num_train_epochs[i_task]))
            rank, layers = self.consolidate_task_adapter()
            print_rank_0(
                f"O-LoRA consolidated task {i_task} ({task}): "
                f"accumulated_rank={rank}, adapted_layers={layers}",
                self.args.global_rank)
            self.save_model(i_task)

    def save_model(self, round):
        if self.args.output_dir is None:
            return
        print_rank_0(
            f"saving O-LoRA adapter to {self.args.output_dir}/{round}...",
            self.args.global_rank)
        if self.args.global_rank == 0:
            path = os.path.join(self.args.output_dir, str(round))
            os.makedirs(path, exist_ok=True)
            self._peft_model().save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        print_rank_0(
            f"Successfully saved O-LoRA after round {round}",
            self.args.global_rank)
