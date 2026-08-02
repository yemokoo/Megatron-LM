import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.train.cl_train_slora import distributed_denoising


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the released SLoRA denoising calculation across multiple GPUs."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mode", default="max")
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(torch.device("cuda", local_rank))
    base_model.eval()

    distributed_denoising(
        base_model,
        args.adapter_path,
        args.output_path,
        args.mode,
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
