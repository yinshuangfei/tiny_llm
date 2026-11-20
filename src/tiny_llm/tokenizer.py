from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Minimal character tokenizer with JSON persistence."""

    def __init__(self, chars: list[str]):
        if not chars:
            raise ValueError("vocab cannot be empty")
        if len(set(chars)) != len(chars):
            raise ValueError("vocab contains duplicates")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        missing = set(text) - set(self.stoi)
        if missing:
            preview = "".join(sorted(missing))[:10]
            raise KeyError(f"unknown characters: {preview!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.chars[i] for i in ids)
        except IndexError as exc:
            raise ValueError("token id is outside vocab") from exc

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])
