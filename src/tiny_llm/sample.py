from __future__ import annotations

import argparse

import torch

from tiny_llm.train import load_checkpoint


def generate(model, tokenizer, prompt: str,
             max_new_tokens=20, temperature=1.0, top_k=None):
    model.eval()
    ids = tokenizer.encode(prompt)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([ids[-model.max_seq_len:]]))[0, -1]
            logits = logits / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = torch.softmax(values, dim=-1)
                next_id = int(indices[torch.multinomial(probs, 1)])
            else:
                next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

            ids.append(next_id)

    return tokenizer.decode(ids)


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
