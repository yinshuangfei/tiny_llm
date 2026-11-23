from __future__ import annotations

import torch
import torch.nn as nn

from tiny_llm.model.attention import CausalSelfAttention
from tiny_llm.model.mlp import SwiGLU
from tiny_llm.model.rmsnorm import RMSNorm


class Block(nn.Module):
    """Pre-norm decoder block: RMSNorm, attention, residual, SwiGLU."""

    def __init__(
        self,
        dim: int,
        heads: int,
        intermediate_size: int,
        use_rope: bool = False,
        use_gqa: bool = False,
        *,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        sliding_window: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.use_gqa = use_gqa
        self.attn_norm = RMSNorm(dim)

        if use_rope or use_gqa:
            kv_heads = max(1, heads // 2) if use_gqa else heads
            self.attn = CausalSelfAttention(
                dim,
                heads,
                kv_heads,
                use_rope,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                sliding_window=sliding_window,
                qk_norm=qk_norm,
            )
        else:
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: dict | None = None,
    ) -> torch.Tensor:
        h = self.attn_norm(x)

        if isinstance(self.attn, CausalSelfAttention):
            attn = self.attn(h, positions, cache)
        else:
            seq_len = x.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)

        x = x + attn
        return x + self.mlp(self.mlp_norm(x))
