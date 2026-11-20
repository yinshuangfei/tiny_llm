from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_llm.train import load_checkpoint


@torch.no_grad()
def evaluate(model, tokenizer, text: str, *, batch_size: int = 8,
             max_windows: int | None = None) -> dict[str, float | int]:
    """Evaluate next-token loss/perplexity on non-overlapping windows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    ids = tokenizer.encode(text)
    seq_len = model.max_seq_len
    if len(ids) < seq_len + 1:
        raise ValueError(
            f"evaluation text needs at least {seq_len + 1} tokens; got {len(ids)}"
        )

    # Drop the short tail instead of padding. This keeps the loss exactly
    # token-weighted and is sufficient for this small evaluation harness.
    windows = [
        ids[start:start + seq_len + 1]
        for start in range(0, len(ids) - seq_len, seq_len)
    ]
    if max_windows is not None:
        windows = windows[:max_windows]

    total_loss = 0.0
    total_tokens = 0
    for batch_start in range(0, len(windows), batch_size):
        batch_ids = windows[batch_start:batch_start + batch_size]
        batch = torch.tensor(batch_ids, dtype=torch.long)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss_sum)
        total_tokens += targets.numel()

    average_loss = total_loss / total_tokens
    return {
        "loss": average_loss,
        "perplexity": math.exp(average_loss),
        "tokens": total_tokens,
        "windows": len(windows),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint)
    text = Path(args.data).read_text(encoding="utf-8")
    metrics = evaluate(model, tokenizer, text, batch_size=args.batch_size,
                       max_windows=args.max_windows)
    print(
        f"loss={metrics['loss']:.4f} "
        f"perplexity={metrics['perplexity']:.4f} "
        f"tokens={metrics['tokens']} windows={metrics['windows']}"
    )


if __name__ == "__main__":
    main()
