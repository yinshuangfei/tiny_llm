import pytest
import torch

from tiny_llm.bpe import BPETokenizer
from tiny_llm.model import _apply_top_p, _sample_token, TinyCausalLM
from tiny_llm.sample import _stop_cutoff, build_parser, generate, stream_generate


def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 2.0, 1.0])
    for _ in range(5):
        assert _sample_token(logits, temperature=0) == 1


def test_top_p_limits_distribution():
    logits = torch.tensor([10.0, 1.0, -10.0, -20.0])
    samples = {
        _sample_token(logits, temperature=1.0, top_p=0.5)
        for _ in range(20)
    }
    assert samples == {0}

    filtered = _apply_top_p(logits, 0.5)
    assert filtered[0] == logits[0]
    assert torch.isinf(filtered[1:]).all()


def test_repetition_penalty_changes_choice():
    logits = torch.tensor([1.0, 1.2])
    assert _sample_token(
        logits, temperature=0, repetition_penalty=2.0, context_ids=[0]
    ) == 1


def test_generate_stop_token_is_not_yielded():
    torch.manual_seed(3)
    model = TinyCausalLM(
        vocab_size=6,
        dim=8,
        layers=1,
        heads=2,
        max_seq_len=8,
        use_rope=True,
        use_gqa=True,
    )
    model.eval()
    output = model.generate(
        torch.tensor([[2, 3]]),
        max_new_tokens=10,
        temperature=0,
        stop_token_ids=[1],
    )
    assert 1 not in output[0, 2:].tolist()


class ScriptedModel:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def generate_stream(self, input_ids, max_new_tokens, *args, **kwargs):
        for token in self.tokens[:max_new_tokens]:
            yield token


def test_stop_string_uses_character_offset_not_token_count():
    corpus = "User: hi\nAssistant: hello there.\n" * 8
    tokenizer = BPETokenizer.train(corpus, vocab_size=300)
    prompt = "User: hi\nAssistant:"
    ids = tokenizer.encode(prompt)
    assert len(ids) < len("User:")
    reply = " hello there."
    model = ScriptedModel(tokenizer.encode(reply))

    text = generate(
        model, tokenizer, prompt, max_new_tokens=20,
        stop_sequences=["User:"],
    )
    assert text == prompt + reply

    text, count = stream_generate(
        model, tokenizer, prompt, max_new_tokens=20,
        stop_sequences=["User:"],
    )
    assert text == prompt + reply
    assert count == len(tokenizer.encode(reply))

    spanned = "Use" + "r: done"
    assert _stop_cutoff(spanned, len("Use"), ["User:"]) == 0


def test_sample_parser_has_m6_flags():
    args = build_parser().parse_args([
        "ckpt.pt", "--top-p", "0.9", "--repetition-penalty", "1.2",
        "--stop", "the end", "--seed", "3", "--stream",
    ])
    assert args.top_p == 0.9
    assert args.repetition_penalty == 1.2
    assert args.stop == ["the end"]
    assert args.seed == 3
    assert args.stream is True
