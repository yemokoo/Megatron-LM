import argparse
import hashlib
import json
import pathlib
import os

import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Trainer, set_seed
from peft import PeftModel, get_peft_model
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
from src.trace_data import to_sft_messages

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
    tokenized_train_cache_dir: str = field(
        default=None, metadata={"help": "Optional pre-tokenized TRACE cache root."})

@dataclass
class ModelArguments(ModelConfig):
    model: str = field(metadata={"help": "name of model."}, default=None)

def return_prompt_and_responses(samples):
    return [to_sft_messages(sample) for sample in samples]


def tokenizer_source_fingerprint(model_path):
    digest = hashlib.sha256()
    found = False
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json"):
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            found = True
            digest.update(name.encode("utf-8") + bytes([0]))
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    if not found:
        raise FileNotFoundError(
            f"no tokenizer assets found under {model_path}")
    return digest.hexdigest()


def obtain_tokenized_dataset(cache_root, task_name, model_path, max_length):
    manifest_path = os.path.join(cache_root, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "complete": True,
        "format": "slora_chat_full",
        "max_length": max_length,
        "tokenizer_fingerprint": tokenizer_source_fingerprint(model_path),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible pre-tokenized cache: {mismatches}")
    task = manifest.get("tasks", {}).get(task_name)
    if task is None:
        raise KeyError(f"token cache has no task entry for {task_name}")
    portable_path = task.get("portable_path")
    if not portable_path:
        raise ValueError(
            f"token cache for {task_name} has no portable_path; "
            "rerun prepare_llama31_trace_cache.sh")
    dataset = Dataset.from_parquet(os.path.join(cache_root, portable_path))
    if len(dataset) != task["cached_samples"]:
        raise ValueError(
            f"incomplete token cache for {task_name}: "
            f"{len(dataset)} != {task['cached_samples']}")
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"token cache for {task_name} has no input_ids")
    rank0_print(
        f"Using pre-tokenized cache for {task_name}: "
        f"{cache_root} ({len(dataset)} records)")
    return dataset

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
    """收集模型状态字典并保存到磁盘"""

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

def train_continual_learning():
    global local_rank

    parser = make_parser()
    script_args, training_args, model_config = parser.parse_args_and_config()
    # training_args.seed = 2025
    set_seed(training_args.seed)
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
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, config=config,
        local_files_only=True, torch_dtype=torch.bfloat16,
    )
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

    rank0_print(f"Training task {task_id}: {script_args.train_data_path}")  
    if task_id > 1:
        for prev_order in range(1, task_id):
            output_dir_parent = os.path.dirname(training_args.output_dir)    
            previous_task_model_path = os.path.join(output_dir_parent, f"order{prev_order}")
            model = PeftModel.from_pretrained(model, previous_task_model_path)
            model = model.merge_and_unload()
    rank0_print("SFTTrainer will add the LoRA adapter...")

    ################
    # Dataset
    ################
    if script_args.tokenized_train_cache_dir:
        train_data = obtain_tokenized_dataset(
            script_args.tokenized_train_cache_dir,
            script_args.dataset_name,
            model_config.model_name_or_path,
            training_args.max_length)
    else:
        train_data = obtain_dataset(script_args.train_data_path)
    eval_data = obtain_dataset(script_args.val_data_path) if script_args.val_data_path is not None else None

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelArguments)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

if __name__ == "__main__":
    train_continual_learning()
