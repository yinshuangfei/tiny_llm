from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by the design-doc decoder block."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(variance + self.eps)
        return (self.weight * x_f).to(dtype)
