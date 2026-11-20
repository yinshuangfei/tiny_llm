from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Minimal byte-level BPE tokenizer."""

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(pair) for pair in merges]
        self.vocab = [bytes([i]) for i in range(256)]
        self.ranks: dict[tuple[int, int], int] = {}
        for rank, pair in enumerate(self.merges):
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
            self.ranks[pair] = rank

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        while 256 + len(merges) < vocab_size and len(ids) > 1:
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            best = max(counts, key=lambda pair: (counts[pair], pair))
            ids = _merge(ids, best, 256 + len(merges))
            merges.append(best)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            pairs = set(zip(ids, ids[1:]))
            ranked = [(self.ranks[pair], pair) for pair in pairs if pair in self.ranks]
            if not ranked:
                break
            _, pair = min(ranked)
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        try:
            data = b"".join(self.vocab[i] for i in ids)
        except IndexError as exc:
            raise ValueError("token id is outside vocab") from exc
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"merges": self.merges}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["merges"])
