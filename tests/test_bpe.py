from tiny_llm.bpe import BPETokenizer


def test_roundtrip_and_compression():
    tokenizer = BPETokenizer.train("abcabcabc", vocab_size=260)
    ids = tokenizer.encode("abcabc")
    assert tokenizer.vocab_size == 260
    assert len(ids) < len("abcabc".encode())
    assert tokenizer.decode(ids) == "abcabc"


def test_unicode_roundtrip():
    tokenizer = BPETokenizer.train("你好abc", vocab_size=300)
    text = "abc你好"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_save_and_load(tmp_path):
    tokenizer = BPETokenizer.train("hello world", vocab_size=270)
    path = tmp_path / "bpe.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.decode(tokenizer.encode("hello")) == "hello"
