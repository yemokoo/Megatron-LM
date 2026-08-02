"""Paper baseline: diagonal-Fisher EWC applied to SeqLoRA parameters."""
from model.continual_lora import attach_seq_lora
from model.paper_baselines import EWCLoRA

EWC = EWCLoRA
__all__ = ["EWC", "EWCLoRA", "attach_seq_lora"]
