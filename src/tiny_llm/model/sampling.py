from __future__ import annotations

from collections.abc import Sequence

import torch


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Apply the CTRL-style repetition penalty in place-free fashion."""
    if penalty == 1.0 or not generated_ids:
        return logits

    used_ids = torch.tensor(generated_ids, dtype=torch.long, device=logits.device)
    selected = logits[used_ids]
    penalized = torch.where(
        selected > 0,
        selected / penalty,
        selected * penalty,
    )
    logits = logits.clone()
    logits[used_ids] = penalized
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest token set whose cumulative probability >= top_p."""
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be in (0, 1]")

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    # Keep the first token that crosses the threshold.
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    mask = torch.zeros_like(remove, dtype=torch.bool)
    mask[sorted_indices[remove]] = True
    return logits.masked_fill(mask, float("-inf"))


def _sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    context_ids: Sequence[int] | None = None,
) -> int:
    logits = logits.clone()
    logits = _apply_repetition_penalty(
        logits, list(context_ids or []), repetition_penalty
    )

    if temperature <= 0:
        return int(torch.argmax(logits))

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, 1)])

    logits = _apply_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))
