from __future__ import annotations

import argparse
import sys
import time

import torch

from tiny_llm.train import load_checkpoint


def generate(model, tokenizer, prompt: str,
             max_new_tokens=20, temperature=1.0, top_k=None):
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt produced no tokens")
    input_ids = torch.tensor([ids], dtype=torch.long)
    output = model.generate(input_ids, max_new_tokens, temperature, top_k)
    return tokenizer.decode(output[0].tolist())


def stream_generate(model, tokenizer, prompt: str,
                    max_new_tokens=20, temperature=1.0, top_k=None):
    """Print tokens as they arrive and return (full text, generated count)."""
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt produced no tokens")

    print(prompt, end="", flush=True)
    generated = []
    input_ids = torch.tensor([ids], dtype=torch.long)
    for token in model.generate_stream(
        input_ids, max_new_tokens, temperature, top_k
    ):
        generated.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    return tokenizer.decode(ids + generated), len(generated)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stream", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint)

    start = time.perf_counter()
    if args.stream:
        text, generated = stream_generate(
            model, tokenizer, args.prompt, args.tokens,
            args.temperature, args.top_k,
        )
    else:
        text = generate(
            model, tokenizer, args.prompt, args.tokens,
            args.temperature, args.top_k,
        )
        generated = args.tokens
    elapsed = time.perf_counter() - start

    if generated:
        rate = generated / elapsed
        print(f"[generated {generated} tokens in {elapsed:.3f}s, {rate:.1f} tokens/s]",
              file=sys.stderr)


if __name__ == "__main__":
    main()
