import math

import pytest
import torch

from tiny_llm.bpe import BPETokenizer
from tiny_llm.sample import generate
from tiny_llm.train import load_checkpoint, save_checkpoint, train


def test_one_training_step_and_sampling(tmp_path):
    torch.manual_seed(0)
    model, tokenizer, losses = train(
        "abcabc" * 20,
        steps=2,
        batch_size=2,
        seq_len=8,
        dim=16,
        layers=1,
        heads=2,
    )
    assert len(losses) == 2
    assert all(math.isfinite(loss) for loss in losses)

    path = tmp_path / "tiny.pt"
    save_checkpoint(model, tokenizer, path)
    loaded_model, loaded_tokenizer = load_checkpoint(path)
    text = generate(loaded_model, loaded_tokenizer, "abc", max_new_tokens=3)
    assert text.startswith("abc")


def test_bpe_training_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    model, tokenizer, losses = train(
        "the tiny model learns to repeat." * 10,
        tokenizer="bpe",
        vocab_size=260,
        steps=2,
        batch_size=2,
        seq_len=8,
        dim=16,
        layers=1,
        heads=2,
    )
    assert isinstance(tokenizer, BPETokenizer)
    assert tokenizer.vocab_size == 260
    assert all(math.isfinite(loss) for loss in losses)

    path = tmp_path / "tiny_bpe.pt"
    save_checkpoint(model, tokenizer, path)
    loaded_model, loaded_tokenizer = load_checkpoint(path)
    assert isinstance(loaded_tokenizer, BPETokenizer)
    assert loaded_tokenizer.vocab_size == 260
