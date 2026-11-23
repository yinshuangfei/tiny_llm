"""Decoder-only model: norms, attention, feed-forward, and sampling."""

from tiny_llm.model.sampling import _apply_top_p, _sample_token
from tiny_llm.model.transformer import TinyCausalLM

__all__ = ["TinyCausalLM", "_apply_top_p", "_sample_token"]
