import pytest

torch = pytest.importorskip("torch")

from tiny_llm.model import TinyCausalLM


def test_logits_shape():
    model = TinyCausalLM(vocab_size=7, dim=8, layers=1, heads=2)
    input_ids = torch.randint(0, 7, (2, 5))
    assert model(input_ids).shape == (2, 5, 7)


def test_causal_attention():
    model = TinyCausalLM(vocab_size=9, dim=8, layers=1, heads=2)
    model.eval()
    prefix = torch.tensor([[1, 2, 3]])
    changed = torch.tensor([[1, 2, 8]])
    with torch.no_grad():
        base = model(prefix)
        masked = model(changed)
    assert torch.allclose(base[:, :2], masked[:, :2])
