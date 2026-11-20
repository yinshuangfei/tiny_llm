import torch
import pytest

from tiny_llm.model import TinyCausalLM


def test_modern_attention_forward_and_causality():
    model = TinyCausalLM(
        vocab_size=9,
        dim=8,
        layers=1,
        heads=4,
        max_seq_len=8,
        use_rope=True,
        use_gqa=True,
    )
    model.eval()
    prefix = torch.tensor([[1, 2, 3]])
    changed = torch.tensor([[1, 2, 8]])
    with torch.no_grad():
        base = model(prefix)
        masked = model(changed)
    assert base.shape == (1, 3, 9)
    assert torch.allclose(base[:, :2], masked[:, :2])
