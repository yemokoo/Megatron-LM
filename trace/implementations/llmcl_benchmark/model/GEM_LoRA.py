"""TRACE task-gradient GEM applied to SeqLoRA parameters."""
from model.continual_lora import attach_seq_lora
from model.paper_baselines import GEMLoRA, project_gem_gradient

GEM = GEMLoRA
__all__ = ["GEM", "GEMLoRA", "attach_seq_lora", "project_gem_gradient"]
