import torch

from tiny_llm.model.mlp import SwiGLU, default_intermediate_size
from tiny_llm.model.rmsnorm import RMSNorm
from tiny_llm.model.rope import apply_rope
from tiny_llm.model.transformer import TinyCausalLM
from tiny_llm.train import load_checkpoint, save_checkpoint
from tiny_llm.tokenizer import CharTokenizer


def test_intermediate_size_matches_design_example():
    assert default_intermediate_size(512) == 1408
    assert default_intermediate_size(96) == 256


def test_block_uses_rmsnorm_and_swiglu():
    model = TinyCausalLM(
        vocab_size=11,
        dim=16,
        layers=1,
        heads=4,
        max_seq_len=8,
        use_rope=True,
        use_gqa=True,
    )
    block = model.blocks[0]
    assert isinstance(block.attn_norm, RMSNorm)
    assert isinstance(block.mlp_norm, RMSNorm)
    assert isinstance(block.mlp, SwiGLU)
    assert isinstance(block.attn.q_norm, RMSNorm)
    assert isinstance(block.attn.k_norm, RMSNorm)
    assert isinstance(model.norm, RMSNorm)
    assert model.intermediate_size == default_intermediate_size(16)
    assert not hasattr(model, "position_embed")

    logits = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 11)


def test_sliding_window_ignores_tokens_outside_the_window():
    model = TinyCausalLM(
        vocab_size=11,
        dim=16,
        layers=1,
        heads=4,
        max_seq_len=8,
        use_rope=True,
        use_gqa=True,
        sliding_window=1,
    )
    model.eval()
    base = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[9, 2, 3, 4]])
    with torch.no_grad():
        base_logits = model(base)
        changed_logits = model(changed)
    assert torch.allclose(base_logits[:, -1], changed_logits[:, -1])
    assert not torch.allclose(base_logits[:, 0], changed_logits[:, 0])


def test_ntk_scaling_changes_rope_frequencies():
    x = torch.randn(1, 2, 4, 8)
    positions = torch.arange(4)
    plain = apply_rope(x, positions, theta=10000.0, scaling=1.0)
    scaled = apply_rope(x, positions, theta=10000.0, scaling=4.0)
    assert plain.shape == x.shape
    assert not torch.allclose(plain, scaled)


def test_architecture_config_roundtrip(tmp_path):
    tokenizer = CharTokenizer.from_text("abcd")
    model = TinyCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=16,
        layers=1,
        heads=4,
        max_seq_len=8,
        use_rope=True,
        use_gqa=True,
        rope_theta=50000.0,
        rope_scaling=2.0,
        sliding_window=4,
    )
    path = tmp_path / "arch.pt"
    save_checkpoint(model, tokenizer, path)
    loaded, _ = load_checkpoint(path)
    assert loaded.intermediate_size == model.intermediate_size
    assert loaded.rope_theta == 50000.0
    assert loaded.rope_scaling == 2.0
    assert loaded.sliding_window == 4
    assert loaded.qk_norm is True
    assert torch.allclose(
        loaded.blocks[0].attn.q_norm.weight,
        model.blocks[0].attn.q_norm.weight,
    )
    assert torch.allclose(
        loaded.blocks[0].mlp.gate_proj.weight,
        model.blocks[0].mlp.gate_proj.weight,
    )
