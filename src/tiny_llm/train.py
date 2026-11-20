from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_llm.bpe import BPETokenizer
from tiny_llm.model import TinyCausalLM
from tiny_llm.tokenizer import CharTokenizer


def build_tokenizer(text: str, kind: str = "char", vocab_size: int | None = None):
    if kind == "char":
        return CharTokenizer.from_text(text)
    if kind == "bpe":
        return BPETokenizer.train(text, vocab_size=vocab_size or 300)
    raise ValueError(f"unknown tokenizer: {kind}")


def tokenizer_state(tokenizer) -> dict:
    if isinstance(tokenizer, CharTokenizer):
        return {"kind": "char", "chars": tokenizer.chars}
    if isinstance(tokenizer, BPETokenizer):
        return {"kind": "bpe", "merges": tokenizer.merges}
    raise TypeError(f"unsupported tokenizer: {type(tokenizer).__name__}")


def tokenizer_from_state(state: dict):
    kind = state.get("kind", "char")
    if kind == "char":
        return CharTokenizer(state["chars"])
    if kind == "bpe":
        return BPETokenizer(state["merges"])
    raise ValueError(f"unknown tokenizer kind: {kind}")


def train(text: str, *, steps=20, batch_size=4, seq_len=16, dim=32,
          layers=1, heads=2, lr=1e-3, seed=0, log_every=None,
          tokenizer="char", vocab_size=None, use_rope=False, use_gqa=False):
    torch.manual_seed(seed)
    tokenizer = build_tokenizer(text, tokenizer, vocab_size)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if tokens.numel() <= seq_len + 1:
        raise ValueError("text is too short for the requested seq_len")

    model = TinyCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        layers=layers,
        heads=heads,
        max_seq_len=seq_len,
        use_rope=use_rope,
        use_gqa=use_gqa,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    max_start = tokens.numel() - seq_len - 1

    for step in range(1, steps + 1):
        starts = torch.randint(0, max_start, (batch_size,))
        inputs = torch.stack([tokens[i:i + seq_len] for i in starts])
        targets = torch.stack([tokens[i + 1:i + seq_len + 1] for i in starts])

        logits = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if log_every and step % log_every == 0:
            print(f"step {step:04d} loss {loss.item():.4f}")

    return model, tokenizer, losses


def save_checkpoint(model, tokenizer, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tokenizer": tokenizer_state(tokenizer),
        "config": {
            "vocab_size": model.vocab_size,
            "dim": model.dim,
            "layers": len(model.blocks),
            "heads": model.heads,
            "max_seq_len": model.max_seq_len,
            "use_rope": model.use_rope,
            "use_gqa": model.use_gqa,
        },
        "model_state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: str | Path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "tokenizer" not in checkpoint:
        checkpoint["tokenizer"] = {"kind": "char", "chars": checkpoint["chars"]}
    config = checkpoint["config"]
    config.setdefault("use_rope", False)
    config.setdefault("use_gqa", False)
    model = TinyCausalLM(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer_from_state(checkpoint["tokenizer"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--out", default="checkpoints/tiny_char.pt")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--use-gqa", action="store_true")
    args = parser.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    model, tokenizer, losses = train(
        text,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        lr=args.lr,
        seed=args.seed,
        log_every=max(1, args.steps // 10),
        tokenizer=args.tokenizer,
        vocab_size=args.vocab_size,
        use_rope=args.use_rope,
        use_gqa=args.use_gqa,
    )
    save_checkpoint(model, tokenizer, args.out)
    print(
        f"saved {args.out}; tokenizer={args.tokenizer}; vocab={tokenizer.vocab_size}; "
        f"rope={args.use_rope}; gqa={args.use_gqa}; final_loss={losses[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
