from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_intermediate_size(dim: int) -> int:
    """LLaMA-style SwiGLU width: about 8/3 of ``dim``.

    Widths for ``dim >= 64`` round up to a multiple of 64. That turns the
    design example of hidden size 512 into intermediate size 1408.
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    hidden = int(8 * dim / 3)
    if dim >= 64:
        hidden = ((hidden + 63) // 64) * 64
    return max(hidden, dim)


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward block: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        if intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
