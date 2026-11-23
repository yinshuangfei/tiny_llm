from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_llm.model.rmsnorm import RMSNorm
from tiny_llm.model.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Causal attention with RoPE, grouped-query attention, and KV cache."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        use_rope: bool,
        *,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        sliding_window: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if sliding_window is not None and sliding_window < 1:
            raise ValueError("sliding_window must be positive")

        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.qk_norm = qk_norm
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: dict | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            q = apply_rope(
                q, positions, theta=self.rope_theta, scaling=self.rope_scaling,
            )
            k = apply_rope(
                k, positions, theta=self.rope_theta, scaling=self.rope_scaling,
            )

        # Cache K/V before GQA repeat_interleave: this keeps the cache small.
        prev_len = 0
        if cache is not None:
            prev_k = cache.get("k")
            prev_v = cache.get("v")
            if prev_k is not None:
                prev_len = prev_k.shape[2]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            cache["k"] = k
            cache["v"] = v

        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = self._attend(q, k, v, positions, prev_len)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        prev_len: int,
    ) -> torch.Tensor:
        seq_len = q.size(2)
        if self.sliding_window is None:
            if seq_len == 1 and prev_len == 0:
                return F.scaled_dot_product_attention(q, k, v)
            if prev_len == 0:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        query_positions = positions
        key_start = int(positions[0].item()) - prev_len
        key_positions = torch.arange(
            key_start, key_start + k.size(2), device=q.device,
        )
        mask = query_positions[:, None] >= key_positions[None, :]
        if self.sliding_window is not None:
            distance = query_positions[:, None] - key_positions[None, :]
            mask = mask & (distance < self.sliding_window)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
