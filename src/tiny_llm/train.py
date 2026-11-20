from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tiny_llm.bpe import BPETokenizer
from tiny_llm.model import TinyCausalLM
from tiny_llm.scheduler import apply_lr_factor, lr_factor
from tiny_llm.tokenizer import CharTokenizer


DEFAULTS = {
    "data": None,
    "out": "checkpoints/tiny_char.pt",
    "tokenizer": "char",
    "vocab_size": None,
    "steps": 100,
    "batch_size": 8,
    "seq_len": 32,
    "dim": 64,
    "layers": 2,
    "heads": 4,
    "lr": 3e-4,
    "seed": 0,
    "use_rope": False,
    "use_gqa": False,
    "warmup_steps": 0,
    "min_lr_ratio": 0.1,
    "scheduler": "cosine",
    "grad_accum": 1,
    "resume": None,
    "log_every": None,
    "log_file": None,
}


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


def load_training_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("training config must be a JSON object")
    unknown = set(config) - set(DEFAULTS) - {"config"}
    if unknown:
        raise ValueError(f"unknown training config keys: {sorted(unknown)}")
    return config


def resolve_training_args(args) -> SimpleNamespace:
    """Resolve config-file defaults, then apply explicit CLI values."""
    values = dict(DEFAULTS)
    if getattr(args, "config", None):
        values.update(load_training_config(args.config))
    for key, value in vars(args).items():
        if key == "config":
            continue
        # Flags cannot represent "not supplied"; they only turn settings on.
        if isinstance(value, bool):
            if value:
                values[key] = True
        elif value is not None:
            values[key] = value
    return SimpleNamespace(**values)


def _append_log(path: str | Path | None, record: dict) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _log_step(step: int, loss: float, optimizer, tokens: int, seconds: float,
              log_every: int | None, log_file: str | Path | None, total_steps: int) -> None:
    should_log = step == total_steps or (log_every is not None and log_every > 0 and step % log_every == 0)
    if not should_log:
        return
    lr = optimizer.param_groups[0]["lr"]
    tokens_per_second = tokens / seconds if seconds else 0.0
    record = {
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "lr": lr,
        "tokens": tokens,
        "tokens_per_second": tokens_per_second,
        "step_seconds": seconds,
    }
    print(
        f"step {step:04d}/{total_steps:04d} loss {loss:.4f} lr {lr:.3g} "
        f"tokens/sec {tokens_per_second:.1f} step_sec {seconds:.3f}"
    )
    _append_log(log_file, record)


def train(text: str, *, steps=20, batch_size=4, seq_len=16, dim=32,
          layers=1, heads=2, lr=1e-3, seed=0, log_every=None,
          tokenizer="char", vocab_size=None, use_rope=False, use_gqa=False,
          warmup_steps=0, min_lr_ratio=0.1, scheduler="cosine", grad_accum=1,
          resume_path=None, log_file=None, out=None):
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0 or seq_len <= 0 or grad_accum <= 0:
        raise ValueError("batch_size, seq_len and grad_accum must be positive")

    if resume_path is not None:
        model, tokenizer, optimizer, start_step = load_training_checkpoint(resume_path)
        if seq_len > model.max_seq_len:
            raise ValueError("seq_len exceeds the resumed model's max_seq_len")
    else:
        torch.manual_seed(seed)
        tokenizer = build_tokenizer(text, tokenizer, vocab_size)
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
        start_step = 0

    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if tokens.numel() <= seq_len + 1:
        raise ValueError("text is too short for the requested seq_len")

    losses = []
    max_start = tokens.numel() - seq_len - 1
    tokens_per_step = batch_size * grad_accum * seq_len
    model.train()

    for step in range(start_step + 1, steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        # Each optimizer step averages grad_accum equally sized microbatches.
        for _ in range(grad_accum):
            starts = torch.randint(0, max_start, (batch_size,))
            inputs = torch.stack([tokens[i:i + seq_len] for i in starts])
            targets = torch.stack([tokens[i + 1:i + seq_len + 1] for i in starts])
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            (loss / grad_accum).backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        factor = lr_factor(
            step,
            steps,
            warmup_steps=warmup_steps,
            scheduler=scheduler,
            min_lr_ratio=min_lr_ratio,
        )
        apply_lr_factor(optimizer, factor)
        optimizer.step()

        loss_value = step_loss / grad_accum
        losses.append(loss_value)
        _log_step(
            step,
            loss_value,
            optimizer,
            tokens_per_step,
            time.perf_counter() - started,
            log_every,
            log_file,
            steps,
        )

    final_step = max(start_step, steps)
    if out is not None:
        save_checkpoint(model, tokenizer, out, optimizer=optimizer, step=final_step)
    return model, tokenizer, losses


def save_checkpoint(model, tokenizer, path: str | Path, optimizer=None, step: int = 0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
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
        "training": {"step": int(step)},
    }
    if optimizer is not None:
        checkpoint["training"].update({
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        })

    # Atomic replacement avoids corrupting a checkpoint if training is interrupted.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def _read_checkpoint(path: str | Path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "tokenizer" not in checkpoint:
        checkpoint["tokenizer"] = {"kind": "char", "chars": checkpoint["chars"]}
    config = checkpoint["config"]
    config.setdefault("use_rope", False)
    config.setdefault("use_gqa", False)
    return checkpoint


def load_checkpoint(path: str | Path, map_location="cpu"):
    checkpoint = _read_checkpoint(path, map_location)
    model = TinyCausalLM(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer_from_state(checkpoint["tokenizer"])


def load_training_checkpoint(path: str | Path, map_location="cpu"):
    """Restore model, tokenizer, optimizer, RNG state, and completed step count."""
    checkpoint = _read_checkpoint(path, map_location)
    model = TinyCausalLM(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training = checkpoint.get("training", {})
    if "optimizer_state_dict" in training:
        optimizer.load_state_dict(training["optimizer_state_dict"])
    if "rng_state" in training:
        torch.set_rng_state(training["rng_state"])
    return model, tokenizer, optimizer, int(training.get("step", 0))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--use-gqa", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--scheduler", choices=["constant", "linear", "cosine"], default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    resolved = resolve_training_args(args)
    if not resolved.data:
        parser.error("data is required (or provide it in --config)")
    if resolved.log_every is None:
        resolved.log_every = max(1, resolved.steps // 10)

    text = Path(resolved.data).read_text(encoding="utf-8")
    model, tokenizer, losses = train(
        text,
        steps=resolved.steps,
        batch_size=resolved.batch_size,
        seq_len=resolved.seq_len,
        dim=resolved.dim,
        layers=resolved.layers,
        heads=resolved.heads,
        lr=resolved.lr,
        seed=resolved.seed,
        log_every=resolved.log_every,
        tokenizer=resolved.tokenizer,
        vocab_size=resolved.vocab_size,
        use_rope=resolved.use_rope,
        use_gqa=resolved.use_gqa,
        warmup_steps=resolved.warmup_steps,
        min_lr_ratio=resolved.min_lr_ratio,
        scheduler=resolved.scheduler,
        grad_accum=resolved.grad_accum,
        resume_path=resolved.resume,
        log_file=resolved.log_file,
        out=resolved.out,
    )
    print(
        f"saved {resolved.out}; tokenizer={resolved.tokenizer}; vocab={tokenizer.vocab_size}; "
        f"rope={resolved.use_rope}; gqa={resolved.use_gqa}; final_loss={losses[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
