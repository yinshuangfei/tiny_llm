import math

import torch

from tiny_llm.eval import evaluate
from tiny_llm.tokenizer import CharTokenizer
from tiny_llm.train import train


def test_evaluate_returns_finite_metrics():
    text = "abcabc" * 20
    torch.manual_seed(0)
    model, tokenizer, _ = train(
        text,
        steps=1,
        batch_size=2,
        seq_len=8,
        dim=16,
        layers=1,
        heads=2,
    )
    assert isinstance(tokenizer, CharTokenizer)
    metrics = evaluate(model, tokenizer, text, batch_size=4)
    assert metrics["tokens"] > 0
    assert metrics["windows"] > 0
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["perplexity"])
    assert metrics["perplexity"] >= 1.0
