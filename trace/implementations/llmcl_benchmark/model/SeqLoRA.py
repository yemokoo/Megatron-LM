"""Paper baseline: one shared FFN LoRA adapter across the task stream."""
from model.continual_lora import attach_seq_lora
from model.paper_baselines import SeqLoRA

__all__ = ["SeqLoRA", "attach_seq_lora"]
