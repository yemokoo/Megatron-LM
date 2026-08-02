import os
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


class O_LoRA(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 lamda_1 = 0.5, lamda_2 = 0
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        '''
        orthological to previous adapters
        '''
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2

        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)


    def train_one_task(self, task, i_task, epochs):
        # if i_task > 0:
        #     self.lamda_2 = 0.1

        num_task = len(self.train_task_list)
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]

        #### TRAIN ####
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                ########################### Regularization ##########################
                orthogonal_loss = 0.
                parameters = dict(self.model.named_parameters())
                if self.args.implementation_variant == "corrected":
                    for name, param in parameters.items():
                        if "lora_B" in name:
                            new_name = name.replace("lora_B", "loranew_B")
                            if new_name in parameters:
                                overlap = torch.mm(param.T, parameters[new_name])
                                orthogonal_loss += overlap.pow(2).sum()
                else:
                    for name, param in parameters.items():
                        if "lora_A" in name:
                            new_name = name.replace("lora_A", "loranew_A")
                            if new_name in parameters:
                                orthogonal_loss += torch.abs(
                                    torch.mm(param, parameters[new_name].T)
                                ).sum()

                # l2-normalization for loranew_A/B
                l2_loss = 0.
                for name, param in self.model.named_parameters():
                    if "loranew_" in name:
                        l2_loss += torch.norm(param, p=2)

                print_rank_0(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {self.lamda_1}; λ2: {self.lamda_2}", self.args.global_rank)
                loss = loss + orthogonal_loss * self.lamda_1 + l2_loss * self.lamda_2
                ######################################################################
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()

        #### COMBINE lora with lora_new and INITIALIZE lora_new ####
        merged_modules = 0
        for module in self.model.modules():
            required = ("lora_A", "lora_B", "loranew_A", "loranew_B")
            if not all(hasattr(module, attr) for attr in required):
                continue
            if "default" not in module.lora_A or "default" not in module.loranew_A:
                continue
            with torch.no_grad():
                old_a = module.lora_A["default"].weight
                old_b = module.lora_B["default"].weight
                new_a = module.loranew_A["default"].weight
                new_b = module.loranew_B["default"].weight
                old_a.data = torch.cat((old_a.data, new_a.data), dim=0)
                old_b.data = torch.cat((old_b.data, new_b.data), dim=1)
                nn.init.kaiming_uniform_(new_a, a=math.sqrt(5))
                nn.init.zeros_(new_b)
                merged_modules += 1
        if merged_modules == 0:
            raise RuntimeError("No O-LoRA modules were found to merge")
        print_rank_0(f"Merged {merged_modules} O-LoRA modules", self.args.global_rank)

        #### RESET ####
        for name, param in self.model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False

        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(i_task))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_to_save.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Sucessfully saving the final model to {peft_model_id}', self.args.global_rank)


    def save_model(self, i_task):
        pass
