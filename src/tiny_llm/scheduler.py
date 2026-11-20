from __future__ import annotations

import math


def lr_factor(step: int, total_steps: int, *, warmup_steps: int = 0, scheduler: str = "cosine", min_lr_ratio: float = 0.1) -> float:
    """Return the learning-rate factor for a 1-based optimizer step."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if scheduler not in {"constant", "linear", "cosine"}:
        raise ValueError(f"unknown scheduler: {scheduler}")

    current = max(0, min(step, total_steps))
    if warmup_steps and current <= warmup_steps:
        return current / warmup_steps
    if scheduler == "constant":
        return 1.0

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (current - warmup_steps) / decay_steps))
    if scheduler == "linear":
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def apply_lr_factor(optimizer, factor: float) -> None:
    """Apply a factor while retaining each param group's original base LR."""
    if factor < 0:
        raise ValueError("learning-rate factor cannot be negative")
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
        group["lr"] = group["base_lr"] * factor
