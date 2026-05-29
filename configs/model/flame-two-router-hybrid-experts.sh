#!/bin/bash
# Two-router hybrid transformer: attention-LoRA experts and FFN MoE experts
# use independent routers while keeping the same G2 expert counts and ranks.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/flame-shared-router-hybrid-experts.sh"

MODEL_ARGS[2]=gpt_two_router_hybrid_local_spec
