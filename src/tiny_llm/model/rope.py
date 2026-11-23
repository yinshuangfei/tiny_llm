from __future__ import annotations

import torch


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float = 10000.0,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Rotate q/k of shape (B, H, T, D).

    ``scaling`` > 1 applies NTK-aware base adjustment so the same head
    dimension covers a longer context. ``scaling`` of 1 matches the
    original RoPE frequencies.
    """
    if x.size(-1) % 2:
        raise ValueError("RoPE requires an even head_dim")
    if theta <= 0:
        raise ValueError("rope theta must be positive")
    if scaling <= 0:
        raise ValueError("rope scaling must be positive")

    bsz, heads, seq_len, head_dim = x.shape
    if scaling != 1.0:
        theta = theta * scaling ** (head_dim / (head_dim - 2))
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim)
    )
    freqs = positions.to(x.device).float()[:, None] * inv_freq[None, :]
    cos = freqs.cos()[None, None, :, :].expand(bsz, heads, seq_len, head_dim // 2)
    sin = freqs.sin()[None, None, :, :].expand(bsz, heads, seq_len, head_dim // 2)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
