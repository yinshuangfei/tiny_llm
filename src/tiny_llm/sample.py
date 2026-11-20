from __future__ import annotations

import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_checkpoint(args.checkpoint)
    print(generate(model, tokenizer, args.prompt, args.tokens,
                   args.temperature, args.top_k))


if __name__ == "__main__":
    main()
