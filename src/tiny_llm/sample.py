from __future__ import annotations

import argparse
import sys
import time

import torch

from tiny_llm.train import load_checkpoint


def _stop_cutoff(text: str, prompt_length: int, stop_sequences: list[str]) -> int | None:
    """Return the cutoff where generated text first completes a stop string.

    ``prompt_length`` is a character offset into ``text``, not a token count.
    """
    cutoffs = []
    for stop in stop_sequences:
        if not stop:
            continue
        # A stop sequence may start at the end of the prompt and be completed
        # by generated text, so search slightly before the prompt boundary.
        start = max(0, prompt_length - len(stop) + 1)
        found = text.find(stop, start)
        if found != -1:
            cutoffs.append(found)
    return min(cutoffs) if cutoffs else None


def generate(model, tokenizer, prompt: str,
             max_new_tokens=20, temperature=1.0, top_k=None,
             top_p=1.0, repetition_penalty=1.0,
             stop_sequences: list[str] | None = None,
             seed: int | None = None):
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt produced no tokens")
    if seed is not None:
        torch.manual_seed(seed)

    input_ids = torch.tensor([ids], dtype=torch.long)
    generated: list[int] = []
    stop_sequences = stop_sequences or []
    # Stop strings are matched on decoded text, so the boundary is in characters.
    prompt_length = len(tokenizer.decode(ids))

    for token in model.generate_stream(
        input_ids, max_new_tokens, temperature, top_k,
        top_p, repetition_penalty,
    ):
        generated.append(token)
        text = tokenizer.decode(ids + generated)
        cutoff = _stop_cutoff(text, prompt_length, stop_sequences)
        if cutoff is not None:
            return text[:cutoff]

    return tokenizer.decode(ids + generated)


def stream_generate(model, tokenizer, prompt: str,
                    max_new_tokens=20, temperature=1.0, top_k=None,
                    top_p=1.0, repetition_penalty=1.0,
                    stop_sequences: list[str] | None = None,
                    seed: int | None = None):
    """Print tokens as they arrive and return (full text, generated count)."""
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt produced no tokens")
    if seed is not None:
        torch.manual_seed(seed)

    stop_sequences = stop_sequences or []
    # Delay printing the tail while it could still become part of a stop string.
    holdback = max((len(stop) for stop in stop_sequences), default=1) - 1
    print(prompt, end="", flush=True)
    generated: list[int] = []
    input_ids = torch.tensor([ids], dtype=torch.long)
    prompt_length = len(tokenizer.decode(ids))
    printed_length = prompt_length

    for token in model.generate_stream(
        input_ids, max_new_tokens, temperature, top_k,
        top_p, repetition_penalty,
    ):
        generated.append(token)
        text = tokenizer.decode(ids + generated)
        cutoff = _stop_cutoff(text, prompt_length, stop_sequences)
        if cutoff is not None:
            end = min(cutoff, len(text))
            if end > printed_length:
                print(text[printed_length:end], end="", flush=True)
            print()
            return text[:end], len(generated)

        safe_end = max(0, len(text) - holdback)
        if safe_end > printed_length:
            print(text[printed_length:safe_end], end="", flush=True)
            printed_length = safe_end

    text = tokenizer.decode(ids + generated)
    if len(text) > printed_length:
        print(text[printed_length:], end="", flush=True)
    print()
    return text, len(generated)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint)

    start = time.perf_counter()
    if args.stream:
        text, generated = stream_generate(
            model, tokenizer, args.prompt, args.tokens,
            args.temperature, args.top_k, args.top_p,
            args.repetition_penalty, args.stop, args.seed,
        )
    else:
        text = generate(
            model, tokenizer, args.prompt, args.tokens,
            args.temperature, args.top_k, args.top_p,
            args.repetition_penalty, args.stop, args.seed,
        )
        generated = args.tokens
    elapsed = time.perf_counter() - start

    if generated:
        rate = generated / elapsed
        print(f"[generated {generated} tokens in {elapsed:.3f}s, {rate:.1f} tokens/s]",
              file=sys.stderr)


if __name__ == "__main__":
    main()
