"""Small, checkpointable LoRA primitives for SLoRA and O-LoRA.

The parameter convention here is ``x @ lora_a @ lora_b`` with ``lora_a`` of
shape ``[in, rank]`` and ``lora_b`` of shape ``[rank, out]``, so ``lora_a``
compresses and ``lora_b`` expands.

Mapping this onto the O-LoRA paper needs care, because the paper writes its
forward pass as a left multiplication -- Eq. (2) is ``h = W_init x + A B x``
with ``W_init`` in ``R^{d x k}``.  There ``k`` is the *input* dimension and
``d`` the *output* one, so ``B`` in ``R^{r x k}`` is the compressor and ``A``
in ``R^{d x r}`` is the expander.  The paper's ``A`` is therefore this
module's ``lora_b`` transposed, i.e. the usual PEFT output-column factor,
exposed as ``output_factor()``; do not read the ``d x r`` shape as an
input-side factor.  O-LoRA orthogonality (Eq. 6) is accordingly
``lora_b_i @ lora_b_t.T``.
"""

from __future__ import annotations

from typing import Optional

import torch

from megatron.core.transformer.module import MegatronModule


class ContinualLowRankAdapter(MegatronModule):
    """A maximum-rank allocation with a runtime-selectable active prefix."""

    def __init__(
        self,
        config,
        input_size: int,
        output_size: int,
        max_rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(config=config)
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.max_rank = int(max_rank)
        self.alpha = float(alpha)
        self.active_rank = 0
        self.dropout = torch.nn.Dropout(float(dropout))

        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        dtype = config.params_dtype
        self.lora_a = torch.nn.Parameter(
            torch.zeros(self.input_size, self.max_rank, device=device, dtype=dtype)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(self.max_rank, self.output_size, device=device, dtype=dtype)
        )

    @property
    def scale(self) -> float:
        return self.alpha / float(max(self.active_rank, 1))

    @torch.no_grad()
    def reset_active(self, rank: int, init_std: float = 0.02) -> None:
        rank = int(rank)
        if not 0 <= rank <= self.max_rank:
            raise ValueError(f"active rank {rank} is outside [0, {self.max_rank}]")
        self.lora_a.zero_()
        self.lora_b.zero_()
        if rank:
            torch.nn.init.normal_(self.lora_a[:, :rank], mean=0.0, std=float(init_std))
        self.active_rank = rank

    def set_active(self, rank: int) -> None:
        rank = int(rank)
        if not 0 <= rank <= self.max_rank:
            raise ValueError(f"active rank {rank} is outside [0, {self.max_rank}]")
        self.active_rank = rank

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.active_rank == 0:
            return hidden_states.new_zeros(*hidden_states.shape[:-1], self.output_size)
        rank = self.active_rank
        original_shape = hidden_states.shape[:-1]
        hidden_flat = self.dropout(hidden_states).reshape(-1, hidden_states.shape[-1])
        hidden_flat = hidden_flat.to(self.lora_a.dtype)
        low_rank = hidden_flat @ self.lora_a[:, :rank]
        output = (low_rank @ self.lora_b[:rank, :]) * self.scale
        return output.reshape(*original_shape, self.output_size)

    def local_delta_weight(self, *, rank: Optional[int] = None) -> torch.Tensor:
        """Return the local linear weight update in PyTorch [out, in] layout."""
        active = self.active_rank if rank is None else int(rank)
        if not 0 <= active <= self.max_rank:
            raise ValueError(f"rank {active} is outside [0, {self.max_rank}]")
        if active == 0:
            return self.lora_b.new_zeros(self.output_size, self.input_size)
        return (
            self.lora_a[:, :active].float() @ self.lora_b[:active, :].float()
        ).T * (self.alpha / float(active))

    def output_factor(self) -> torch.Tensor:
        """PEFT-style output-column factor, the O-LoRA paper's A, as [out_local, rank]."""
        return self.lora_b[: self.active_rank, :].T

    @torch.no_grad()
    def clear(self) -> None:
        self.lora_a.zero_()
        self.lora_b.zero_()
        self.active_rank = 0
