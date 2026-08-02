"""Paper baseline: original O-LoRA rank growth ported to FFN LoRA."""
from model.continual_lora import attach_olora
from model.paper_baselines import OLoRA

O_LoRA = OLoRA
__all__ = ["O_LoRA", "OLoRA", "attach_olora"]
