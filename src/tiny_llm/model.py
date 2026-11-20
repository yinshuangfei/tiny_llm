from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to q/k of shape (B, H, T, D)."""
    if x.size(-1) % 2:
        raise ValueError("RoPE requires an even head_dim")

    bsz, heads, seq_len, head_dim = x.shape
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim)
    )
    freqs = positions.to(x.device).float()[:, None] * inv_freq[None, :]
    cos = freqs.cos()[None, None, :, :].expand(bsz, heads, seq_len, head_dim // 2)
    sin = freqs.sin()[None, None, :, :].expand(bsz, heads, seq_len, head_dim // 2)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """Modern attention with RoPE and grouped-query attention."""

    def __init__(self, dim: int, heads: int, kv_heads: int, use_rope: bool):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")

        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.use_rope = use_rope
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)

        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


class Block(nn.Module):
    """Pre-LayerNorm causal Transformer block."""

    def __init__(self, dim: int, heads: int, use_rope: bool = False,
                 use_gqa: bool = False):
        super().__init__()
        self.use_rope = use_rope
        self.use_gqa = use_gqa
        self.ln1 = nn.LayerNorm(dim)

        if use_rope or use_gqa:
            kv_heads = max(1, heads // 2) if use_gqa else heads
            self.attn = CausalSelfAttention(dim, heads, kv_heads, use_rope)
        else:
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)

        if isinstance(self.attn, CausalSelfAttention):
            attn = self.attn(h, positions)
        else:
            seq_len = x.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)

        x = x + attn
        return x + self.mlp(self.ln2(x))


class TinyCausalLM(nn.Module):
    """Tiny decoder-only language model."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_seq_len: int = 128,
        use_rope: bool = False,
        use_gqa: bool = False,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")

        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_gqa = use_gqa

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.position_embed = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            Block(dim, heads, use_rope=use_rope, use_gqa=use_gqa)
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embed(input_ids)
        if not self.use_rope:
            x = x + self.position_embed(positions)

        for block in self.blocks:
            x = block(x, positions)

        return self.lm_head(self.norm(x))
