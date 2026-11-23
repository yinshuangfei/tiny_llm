from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
import torch.nn as nn

from tiny_llm.model.block import Block
from tiny_llm.model.mlp import default_intermediate_size
from tiny_llm.model.rmsnorm import RMSNorm
from tiny_llm.model.sampling import _sample_token


class TinyCausalLM(nn.Module):
    """Decoder-only model: token embedding, pre-norm blocks, SwiGLU, LM head."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_seq_len: int = 128,
        use_rope: bool = False,
        use_gqa: bool = False,
        intermediate_size: int | None = None,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        sliding_window: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if rope_scaling <= 0:
            raise ValueError("rope_scaling must be positive")
        if sliding_window is not None and sliding_window < 1:
            raise ValueError("sliding_window must be positive")
        if sliding_window is not None and not (use_rope or use_gqa):
            raise ValueError("sliding_window requires the modern attention path")
        if rope_scaling != 1.0 and not use_rope:
            raise ValueError("rope_scaling requires use_rope")

        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_gqa = use_gqa
        self.intermediate_size = intermediate_size or default_intermediate_size(dim)
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.qk_norm = qk_norm

        self.token_embed = nn.Embedding(vocab_size, dim)
        if not use_rope:
            self.position_embed = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList(
            Block(
                dim,
                heads,
                self.intermediate_size,
                use_rope=use_rope,
                use_gqa=use_gqa,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                sliding_window=sliding_window,
                qk_norm=qk_norm,
            )
            for _ in range(layers)
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: list[dict | None] | None = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)

        if kv_cache is not None:
            if len(kv_cache) != len(self.blocks):
                raise ValueError("kv_cache length must match number of layers")
            if not self.use_rope and not self.use_gqa:
                raise ValueError("KV cache requires the modern attention path")

        if kv_cache is not None:
            for i, layer_cache in enumerate(kv_cache):
                if layer_cache is None:
                    kv_cache[i] = {}

        if kv_cache is not None and kv_cache[0].get("k") is not None:
            prev_len = kv_cache[0]["k"].shape[2]
            if prev_len + seq_len > self.max_seq_len:
                raise ValueError(
                    f"cached sequence length {prev_len + seq_len} exceeds max_seq_len {self.max_seq_len}"
                )
            positions = torch.arange(
                prev_len, prev_len + seq_len, device=input_ids.device
            )
        else:
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
                )
            positions = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embed(input_ids)
        if not self.use_rope:
            x = x + self.position_embed(positions)

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = block(x, positions, layer_cache)

        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_token_ids: Sequence[int] | None = None,
    ) -> Iterator[int]:
        """Yield token ids one at a time, using KV cache when available."""
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("generate currently supports batch size 1")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        ids = input_ids.tolist()[0]
        supports_cache = self.use_rope or self.use_gqa

        if not supports_cache:
            # Old checkpoints keep a simple full-context generation path.
            for _step in range(max_new_tokens):
                context = torch.tensor([ids[-self.max_seq_len:]])
                logits = self(context)[0, -1]
                next_id = _sample_token(
                    logits, temperature, top_k, top_p,
                    repetition_penalty, ids,
                )
                ids.append(next_id)
                if stop_token_ids and next_id in stop_token_ids:
                    return
                yield next_id
            return

        # Prefill once. If the prompt is longer than the model window, keep its tail.
        context = ids[-self.max_seq_len :]
        cache = [None] * len(self.blocks)
        logits = self(torch.tensor([context]), kv_cache=cache)[0, -1]

        for step in range(max_new_tokens):
            next_id = _sample_token(
                logits, temperature, top_k, top_p,
                repetition_penalty, ids,
            )
            ids.append(next_id)
            if stop_token_ids and next_id in stop_token_ids:
                return
            yield next_id

            if step + 1 == max_new_tokens:
                break

            cache_len = cache[0]["k"].shape[2] if cache[0] is not None else 0
            if cache_len >= self.max_seq_len:
                # Sliding-window fallback: rebuild cache from the newest window.
                cache = [None] * len(self.blocks)
                logits = self(torch.tensor([ids[-self.max_seq_len:]]), kv_cache=cache)[0, -1]
            else:
                logits = self(torch.tensor([[next_id]]), kv_cache=cache)[0, -1]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_token_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Generate one batch and return all prompt + generated ids."""
        ids = input_ids.tolist()[0]
        for next_id in self.generate_stream(
            input_ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            stop_token_ids,
        ):
            ids.append(next_id)
        return torch.tensor([ids], dtype=torch.long)
