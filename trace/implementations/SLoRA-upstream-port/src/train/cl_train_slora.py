import argparse
import json
import os
from tqdm import tqdm

import torch
from safetensors.torch import save_file, load_file

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from datasets import Dataset
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config
)

from src.model.builder import load_denoised_lora
from src.trace_data import to_sft_messages
from src.train.joint_replay_trainer import JointReplaySFTTrainer
from src.train.replay_memory import build_replay_dataset

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ScriptArguments(ScriptArguments):
    dataset_name: str = field(metadata={"help": "Dataset name."}, default=None)
    train_data_path: str = field(metadata={"help": "Path to the training data."}, default=None)
    val_data_path: str = field(metadata={"help": "Path to the validation data."}, default=None)
    task_id: int = field(default=0)
    mode: str = field(default='max', metadata={"help": "Mode for denoising."})
    # Replay control.  Absent these, training is the released SLoRA-Pre run.
    replay_v3_run_dir: str = field(default=None, metadata={"help":
        "v3 run directory holding fixed_replay_memory/ and replay_plans/; "
        "enables joint replay when set."})
    replay_data_root: str = field(default=None, metadata={"help":
        "TRACE data root the stored replay indices point into."})
    replay_loss_coeff: float = field(default=1.0)

@dataclass
class ModelArguments(ModelConfig):
    model: str = field(metadata={"help": "name of model."}, default=None)

def return_prompt_and_responses(samples):
    return [to_sft_messages(sample) for sample in samples]

def obtain_dataset(data_path):
    # A disk-backed source gives every distributed rank the same fingerprint.
    # File locking lets non-zero ranks reuse preprocessing cache instead of
    # independently tokenizing all 5,000 examples.
    dataset = Dataset.from_json(data_path)
    return dataset.map(
        to_sft_messages,
        remove_columns=dataset.column_names,
        desc="Converting TRACE records to messages",
    )

def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def calculate_similarity(U_c_r, base_U_r, mode=None):
    if mode == 'l2':
        return torch.norm(U_c_r - base_U_r, p=2).item()
    elif mode == 'cosine':
        return torch.cosine_similarity(U_c_r.flatten(), base_U_r.flatten(), dim=0).item()
    else:
        return torch.norm(torch.mm(U_c_r.T, base_U_r), p="fro").item()

def perform_similarity_search(delta_W_tensor, base_U_r, rank, mode):
    print(f"rank: {rank}", flush=True)
    ratio_candidates = [0.1 * i for i in range(1, 11)]
    max_similarity = -float("inf")
    min_similarity = float("inf")
    best_c = rank
    new_lora_A, new_lora_B = None, None

    for ratio in ratio_candidates:
        c = max(int(rank * ratio), 1)
        Omega = torch.randn(delta_W_tensor.shape[1], c, device=delta_W_tensor.device)
        Y = torch.mm(delta_W_tensor, Omega)
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        B_ = torch.mm(Q.T, delta_W_tensor)
        U_hat, Sigma, Vt = torch.linalg.svd(B_, full_matrices=False)
        U = torch.mm(Q, U_hat)

        if mode == 'minor':
            U_c = U[:, -c:]
            Sigma_c = Sigma[-c:]
            V_c = Vt[-c:, :]
        else:
            U_c = U[:, :c]
            Sigma_c = Sigma[:c]
            V_c = Vt[:c, :]

        delta_W_c = torch.mm(torch.mm(U_c, torch.diag(Sigma_c)), V_c)
        U_c, _, _ = torch.linalg.svd(delta_W_c, full_matrices=False)
        U_c_r = U_c[:, :rank]

        total_similarity = 0
        if base_U_r is not None:
            similarity_base = calculate_similarity(U_c_r, base_U_r, mode)
            total_similarity += similarity_base

        if mode == 'min':
            if total_similarity < min_similarity:
                min_similarity = total_similarity
                best_c = c
                new_lora_A = V_c
                new_lora_B = torch.mm(U, torch.diag(Sigma_c))           
        else:
            if total_similarity > max_similarity:
                max_similarity = total_similarity
                best_c = c
                new_lora_A = V_c
                new_lora_B = torch.mm(U, torch.diag(Sigma_c))
        print(f"Best c determined by Sim-Search: {best_c}", flush=True)
    return new_lora_A, new_lora_B

def denoising(base_model, delta_weights, mode='max', shard_rank=0, num_shards=1):
    if base_model is not None:
        with torch.no_grad():
            base_weights = dict(base_model.named_parameters())

    denoised_delta_weights = {}
    lora_pairs = [
        (name, delta_W)
        for name, delta_W in delta_weights.items()
        if "lora_A" in name
    ]
    for module_index, (name, delta_W) in enumerate(
        tqdm(lora_pairs, desc=f"Denoising shard {shard_rank + 1}/{num_shards}")
    ):
        if module_index % num_shards != shard_rank:
            continue

        lora_B_name = name.replace("lora_A", "lora_B")
        if lora_B_name not in delta_weights:
            print(f"Warning: {lora_B_name} not found in LoRA weights. Skipping this parameter.", flush=True)
            continue

        lora_A = delta_W.detach()
        lora_B = delta_weights[lora_B_name].detach()
        rank = lora_A.shape[0]
        base_U_r = None
        if base_model is not None:
            base_weight_name = name.replace("base_model.model.", "").replace(".lora_A.weight", ".base_layer.weight")
            base_weight_name_1 = name.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            if base_weight_name in base_weights:
                base_W_tensor = base_weights[base_weight_name].detach()
            elif base_weight_name_1 in base_weights:
                base_W_tensor = base_weights[base_weight_name_1].detach()
            else:
                print(f"Warning: {base_weight_name}  or {base_weight_name_1} not found in base weights. Skipping this parameter.", flush=True)
                
                continue
            base_U, _, _ = torch.linalg.svd(base_W_tensor.to(torch.float32), full_matrices=False)
            base_U_r = base_U[:, :rank]


        delta_W_tensor = torch.mm(lora_B, lora_A)
        delta_W_tensor_32 = delta_W_tensor.to(torch.float32)

        new_lora_A, new_lora_B = perform_similarity_search(delta_W_tensor_32, base_U_r, rank, mode)

        denoised_delta_weights[name] = new_lora_A.to(delta_W_tensor.dtype).contiguous()
        denoised_delta_weights[lora_B_name] = new_lora_B.to(delta_W_tensor.dtype).contiguous()

    return denoised_delta_weights


def distributed_denoising(base_model, adapter_path, output_path, mode):
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0

    if distributed:
        torch.distributed.barrier()

    state_dict = load_file(adapter_path)
    state_dict = {key: value.to(base_model.device) for key, value in state_dict.items()}
    denoised_state_dict = denoising(
        base_model,
        state_dict,
        mode,
        shard_rank=rank,
        num_shards=world_size,
    )

    shard_path = f"{output_path}.rank{rank}-of-{world_size}"
    save_file(denoised_state_dict, shard_path)

    if distributed:
        torch.distributed.barrier()

    if rank == 0:
        merged_state_dict = {}
        for shard_rank in range(world_size):
            current_shard_path = f"{output_path}.rank{shard_rank}-of-{world_size}"
            current_shard = load_file(current_shard_path)
            overlap = merged_state_dict.keys() & current_shard.keys()
            if overlap:
                raise RuntimeError(f"Duplicate tensors across denoising shards: {sorted(overlap)}")
            merged_state_dict.update(current_shard)
        save_file(merged_state_dict, output_path)

    if distributed:
        torch.distributed.barrier()


def train_continual_learning():
    global local_rank

    parser = make_parser()
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.dataset_num_proc = 8
    local_rank = training_args.local_rank
    model_config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    task_id = script_args.task_id
    rank0_print('task', script_args.train_data_path)

    ################
    # Model & Tokenizer
    ################
    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path, local_files_only=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, config=config,
        local_files_only=True, torch_dtype=torch.bfloat16,
    )

    model = base_model
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code, use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        if "llama3" in model_config.model:
            tokenizer.pad_token="<|finetune_right_pad_id|>"
            tokenizer.pad_token_id=128004
        else:
            raise NotImplementedError
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    output_dir_parent = os.path.dirname(training_args.output_dir)
    
    rank0_print(f"Training task {task_id}: {script_args.train_data_path}")  
    if task_id > 1:
        for prev_order in range(1, task_id):
            previous_task_model_path = os.path.join(output_dir_parent, f"order{prev_order}")
            lora_config = LoraConfig.from_pretrained(previous_task_model_path)
            previous_task_lora_path = os.path.join(previous_task_model_path, f"{script_args.mode}.safetensors")
            print('Load LoRA from previous task:', previous_task_lora_path, flush=True)
            state_dict = load_file(previous_task_lora_path)
            model = load_denoised_lora(model, state_dict, lora_config)
    rank0_print("SFTTrainer will add the LoRA adapter...")


   
    ################
    # Dataset
    ################
    train_data = obtain_dataset(script_args.train_data_path)
    eval_data = obtain_dataset(script_args.val_data_path) if script_args.val_data_path is not None else None

    ################
    # Training
    ################
    ################
    # Replay (optional): the same stored records a v3 run replayed at this
    # round, but its gradient reaches the whole new LoRA instead of routers.
    ################
    replay_data = None
    if script_args.replay_v3_run_dir and task_id > 1:
        if not script_args.replay_data_root:
            raise ValueError(
                "--replay_data_root is required with --replay_v3_run_dir")
        replay_data, replay_manifest = build_replay_dataset(
            script_args.replay_v3_run_dir, script_args.replay_data_root,
            task_id, num_proc=training_args.dataset_num_proc)
        rank0_print(f"[replay] {json.dumps(replay_manifest)}")
        if local_rank == 0:
            # SFTTrainer creates output_dir later; the manifest is written here
            # so a failed run still records what it was fed.
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(
                    training_args.output_dir, "replay_manifest.json"),
                    "w") as handle:
                json.dump(replay_manifest, handle, indent=2)
    elif script_args.replay_v3_run_dir:
        rank0_print("[replay] task 1 has no past tasks; primary only")

    trainer_class = SFTTrainer if replay_data is None else JointReplaySFTTrainer
    trainer_kwargs = {} if replay_data is None else {
        "replay_dataset": replay_data,
        "replay_coeff": script_args.replay_loss_coeff,
    }
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
        **trainer_kwargs,
    )

    trainer.train()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    ################
    # After Training - Denoising
    ################
    if os.environ.get("SLORA_SKIP_DENOISE") == "1":
        # Smoke tests train a handful of steps; the randomized SVD over 224
        # modules afterwards is the slow part and proves nothing about replay.
        rank0_print("[smoke] SLORA_SKIP_DENOISE=1; skipping denoising")
        return
    if training_args.local_rank == 0:
        rank0_print(f"Pruning LoRA weights after task {task_id}...")
    distributed_denoising(
        base_model,
        os.path.join(training_args.output_dir, "adapter_model.safetensors"),
        os.path.join(training_args.output_dir, f"{script_args.mode}.safetensors"),
        script_args.mode,
    )

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelArguments)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    train_continual_learning()
