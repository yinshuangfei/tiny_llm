import torch
import pytest

from tiny_llm.model import TinyCausalLM


def test_cached_forward_matches_full_forward():
    model = TinyCausalLM(
        vocab_size=11,
        dim=8,
        layers=2,
        heads=4,
        max_seq_len=8,
        use_rope=True,
        use_gqa=True,
    )
    model.eval()
    input_ids = torch.tensor([[1, 4, 2, 7, 3]])

    with torch.no_grad():
        full = model(input_ids)
        cache = [None] * len(model.blocks)
        prefix_logits = model(input_ids[:, :3], kv_cache=cache)
        suffix_logits = model(input_ids[:, 3:], kv_cache=cache)

    assert prefix_logits.shape == (1, 3, 11)
    assert suffix_logits.shape == (1, 2, 11)
    assert torch.allclose(full[:, 3:], suffix_logits, atol=1e-5, rtol=1e-5)


def test_cached_generation_shape_and_repetition():
    torch.manual_seed(7)
    model = TinyCausalLM(
        vocab_size=11,
        dim=8,
        layers=1,
        heads=4,
        max_seq_len=6,
        use_rope=True,
        use_gqa=True,
    )
    model.eval()
    output = model.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=8,
        temperature=0.1,
        top_k=3,
    )
    assert output.shape == (1, 11)
    assert all(0 <= token < 11 for token in output[0].tolist())
