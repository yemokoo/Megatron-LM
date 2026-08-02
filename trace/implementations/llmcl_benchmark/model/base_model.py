import torch
from utils.utils import print_rank_0, to_device, save_hf_format, get_all_reduce_mean
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from transformers import GenerationConfig
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        
        
    def _optimizer_step(self, loss, accum_step=0, accum_steps=1):
        """Plain-PyTorch replacement for DeepSpeed's engine.backward()+engine.step().
        Repeatedly calling deepspeed.initialize() on the same model (needed every
        phase, since add_experts() creates new nn.Parameters and each phase trains a
        different frozen/trainable split) leaks GPU memory without bound -- confirmed
        via a synthetic repro where it grows every single re-init regardless of ZeRO
        stage, destroy()+gc.collect(), or bucket size, while plain optimizer/DDP
        re-creation on the same model showed zero growth. So training steps here are
        plain autograd + grad clipping, matching the DeepSpeed config's
        gradient_clipping=1.0.
        """
        (loss / accum_steps).backward()
        if (accum_step + 1) % accum_steps == 0:
            trainable_params = [p for p in self.raw_model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            self.optimizer.step()
            if getattr(self, 'lr_scheduler', None) is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity


    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self._optimizer_step(loss)


            # Evaluate perplexity on the validation set.
            # print_rank_0(
            #     f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs} *****",
            #     self.args.global_rank)
            # perplexity = self.perplexity_evaluation(eval_dataloader, device)
            # print_rank_0(f"ppl: {perplexity}", self.args.global_rank)
            # self.model.tput_timer.update_epoch_count()
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            if i_task < getattr(self.args, "start_task", 0):
                print_rank_0(f"Skipping completed task {i_task}: {task}", self.args.global_rank)
                continue
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)

    
    # Subclasses can set this to a list of state_dict-key substrings to save only
    # those (e.g. grown experts + router), skipping the frozen backbone. None = save
    # the full model.
    save_key_substrings = None

    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round),
                           key_substrings=self.save_key_substrings)

        print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
        
