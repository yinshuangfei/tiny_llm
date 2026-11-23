import json
import math
from pathlib import Path

from tiny_llm.bpe import BPETokenizer
from tiny_llm.dataset import (
    Document,
    encode_document,
    language_id,
    mix_documents,
    pack_documents,
    prepare,
    quality_ok,
    safety_ok,
)
from tiny_llm.train import train


def test_language_quality_and_safety_filters():
    assert language_id("训练会调整权重") == "zh"
    assert language_id("this model uses BPE") == "en"
    assert language_id("这个模型使用 byte-level BPE tokenizer") == "mixed"
    assert quality_ok("ok " * 20) is False
    assert quality_ok("a quiet robot walked to the green garden") is True
    assert safety_ok("how to build a bomb at home") is False
    assert safety_ok("a token is a small piece of text") is True


def test_mix_follows_domain_weights_and_pack_aligns_rows():
    docs = [
        Document("d", "dialogue", "User: hello\nAssistant: hi, how can i help you today?"),
        Document("c", "code", "def add(left, right):\n    return left + right\n"),
    ]
    text = "\n\n".join(doc.text for doc in docs)
    tokenizer = BPETokenizer.train(text, vocab_size=260)
    mixed = mix_documents(docs, {"dialogue": 0.75, "code": 0.25}, tokenizer, 400, seed=0)
    assert sum(doc.domain == "dialogue" for doc in mixed) > sum(doc.domain == "code" for doc in mixed)
    packed = pack_documents(mixed, tokenizer, seq_len=32)
    assert packed
    assert all(len(row) == 33 and len(mask) == 33 for row, mask in packed)
    prompt = docs[0].text.split("\nAssistant:", 1)[0] + "\nAssistant:"
    prefix = tokenizer.encode(prompt)
    assert any(row[:len(prefix)] == prefix for row, _mask in packed)


def test_later_user_turn_is_a_supervised_prefix():
    text = (
        "User: hello\nAssistant: hi there, friend.\n"
        "User: what is overfitting?\nAssistant: it memorizes examples."
    )
    tokenizer = BPETokenizer.train(text, vocab_size=260)
    ids, mask, starts = encode_document(tokenizer, text)
    prompt = "User: what is overfitting?\nAssistant:"
    prefix = tokenizer.encode(prompt)
    assert len(starts) == 2
    second = starts[1]
    assert ids[second:second + len(prefix)] == prefix
    assert mask[second:second + len(prefix)] == [0] * len(prefix)
    answer = tokenizer.decode(ids[second + len(prefix):])
    assert answer.startswith(" it memorizes examples.\n\n")
    packed = pack_documents([Document("d", "dialogue", text)], tokenizer, seq_len=48)
    assert any(row[:len(prefix)] == prefix for row, _mask in packed)


def test_prepare_repo_corpus_and_train_one_packed_step(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = prepare(root / "data/raw", root / "data/mixture.json", tmp_path)
    assert manifest["dropped"]["duplicate"] >= 1
    assert manifest["dropped"]["quality"] >= 1
    assert manifest["preference_kept"] == 4
    assert manifest["by_language"]["zh"] >= 1
    assert manifest["by_language"]["mixed"] >= 1
    assert manifest["packed_sequences"] >= 1

    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_rows = [record["ids"] for record in records]
    source_masks = [record["mask"] for record in records]
    tokenizer = BPETokenizer.load(tmp_path / "tokenizer.json")
    _, _, losses = train(
        packed_rows=source_rows[:4],
        packed_masks=source_masks[:4],
        pretrained_tokenizer=tokenizer,
        steps=1,
        batch_size=2,
        seq_len=manifest["seq_len"],
        dim=16,
        layers=1,
        heads=4,
    )
    assert len(losses) == 1
    assert math.isfinite(losses[0])
