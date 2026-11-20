from tiny_llm.tokenizer import CharTokenizer


def test_roundtrip_is_lossless():
    tokenizer = CharTokenizer.from_text("\u4f60\u597dabc")
    text = "abc\u4f60\u597d"
    ids = tokenizer.encode(text)
    assert tokenizer.vocab_size == 5
    assert len(ids) == len(text)
    assert tokenizer.decode(ids) == text


def test_save_and_load(tmp_path):
    tokenizer = CharTokenizer.from_text("hello")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.decode(tokenizer.encode("elloh")) == "elloh"
