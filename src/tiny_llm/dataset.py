"""Prepare a small training mixture the way a production stack does.

Raw JSONL documents go through exact dedup, language identification,
quality and safety filters, then domain weights, a byte-level BPE, and
fixed-length sequence packing. Preference pairs are kept as their own
split and are not mixed into the pretraining stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tiny_llm.bpe import BPETokenizer

_UNSAFE_PATTERNS = (
    "how to make a bomb",
    "how to build a bomb",
    "how to make explosives",
)


@dataclass(frozen=True)
class Document:
    id: str
    domain: str
    text: str
    lang: str = ""


@dataclass(frozen=True)
class PreferencePair:
    id: str
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class TrainingSource:
    text: str | None = None
    packed_rows: list[list[int]] | None = None
    packed_masks: list[list[int]] | None = None
    tokenizer: BPETokenizer | None = None


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def language_id(text: str) -> str:
    """Label a document as zh, en, mixed, or unknown from its letters."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + latin
    if total == 0:
        return "unknown"
    cjk_ratio = cjk / total
    latin_ratio = latin / total
    if cjk_ratio >= 0.2 and latin_ratio >= 0.2:
        return "mixed"
    if cjk_ratio >= 0.5:
        return "zh"
    return "en"


def quality_ok(text: str) -> bool:
    """Drop empty, tiny, or highly repetitive documents.

    Long natural text reuses a small alphabet, so the check looks at the
    most common character and at repeated words rather than alphabet size.
    """
    stripped = text.strip()
    if len(stripped) < 24:
        return False
    letters = [ch for ch in stripped if not ch.isspace()]
    if letters and max(Counter(letters).values()) / len(letters) > 0.4:
        return False
    words = stripped.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.25:
        return False
    return True


def safety_ok(text: str) -> bool:
    lowered = text.lower()
    return not any(pattern in lowered for pattern in _UNSAFE_PATTERNS)


def load_raw(raw_dir: str | Path) -> tuple[list[Document], list[PreferencePair]]:
    raw_dir = Path(raw_dir)
    documents: list[Document] = []
    preferences: list[PreferencePair] = []
    paths = sorted(raw_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no jsonl files in {raw_dir}")
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            domain = row.get("domain", "")
            row_id = row.get("id") or f"{path.stem}-{line_no}"
            if domain == "preference":
                preferences.append(PreferencePair(
                    id=row_id,
                    prompt=row.get("prompt", ""),
                    chosen=row.get("chosen", ""),
                    rejected=row.get("rejected", ""),
                ))
                continue
            documents.append(Document(
                id=row_id,
                domain=domain,
                text=row.get("text", ""),
            ))
    return documents, preferences


def dedup(documents: list[Document]) -> tuple[list[Document], int]:
    seen: set[str] = set()
    kept: list[Document] = []
    dropped = 0
    for doc in documents:
        key = hashlib.sha256(normalize_text(doc.text).encode("utf-8")).hexdigest()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(doc)
    return kept, dropped


def filter_documents(
    documents: list[Document],
) -> tuple[list[Document], Counter]:
    """Identify language, then drop documents that fail quality or safety."""
    kept: list[Document] = []
    dropped: Counter = Counter()
    for doc in documents:
        if not doc.domain:
            dropped["missing_domain"] += 1
            continue
        if not safety_ok(doc.text):
            dropped["safety"] += 1
            continue
        if not quality_ok(doc.text):
            dropped["quality"] += 1
            continue
        lang = language_id(doc.text)
        if lang == "unknown":
            dropped["language"] += 1
            continue
        kept.append(Document(doc.id, doc.domain, doc.text, lang))
    return kept, dropped


def filter_preferences(
    pairs: list[PreferencePair],
) -> tuple[list[PreferencePair], int]:
    kept: list[PreferencePair] = []
    dropped = 0
    for pair in pairs:
        body = f"{pair.prompt}\n{pair.chosen}\n{pair.rejected}"
        if not pair.prompt.strip() or not pair.chosen.strip() or not pair.rejected.strip():
            dropped += 1
            continue
        if not safety_ok(body) or not quality_ok(pair.chosen):
            dropped += 1
            continue
        kept.append(pair)
    return kept, dropped


def mix_documents(
    documents: list[Document],
    weights: dict[str, float],
    tokenizer: BPETokenizer,
    target_tokens: int,
    seed: int,
) -> list[Document]:
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"mixture weights must sum to 1, got {total}")
    by_domain: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        by_domain[doc.domain].append(doc)
    missing = sorted(set(weights) - set(by_domain))
    if missing:
        raise ValueError(f"mixture domain has no documents: {missing}")
    extra = sorted(set(by_domain) - set(weights))
    if extra:
        raise ValueError(f"domains missing from mixture weights: {extra}")

    rng = random.Random(seed)
    selected: list[Document] = []
    for domain, weight in weights.items():
        group = list(by_domain[domain])
        rng.shuffle(group)
        budget = max(1, int(round(target_tokens * weight)))
        filled = 0
        index = 0
        while filled < budget:
            doc = group[index % len(group)]
            selected.append(doc)
            filled += max(1, len(tokenizer.encode(doc.text)))
            index += 1
            if index > budget * 20:
                break
    return selected


def encode_document(
    tokenizer: BPETokenizer, text: str,
) -> tuple[list[int], list[int], list[int]]:
    """Encode one document into ids, a loss mask, and user-turn starts.

    Chat turns are split on the assistant marker so BPE cannot merge the
    prompt with the answer. Prompt tokens, including ``Assistant:``, are
    mask 0. Answer tokens are mask 1, and each answer is followed by a
    supervised blank line so generation can stop where chat stops. Each
    later user turn is also a start, so a follow-up question can sit at
    position 0 the same way chat asks it. Documents without an assistant
    marker keep a mask of all ones.
    """
    marker = "\nAssistant:"
    if marker not in text:
        ids = tokenizer.encode(text)
        return ids, [1] * len(ids), [0]

    ids: list[int] = []
    mask: list[int] = []
    starts = [0]
    remaining = text
    end_turn = tokenizer.encode("\n\n")
    while remaining:
        found = remaining.find(marker)
        if found == -1:
            piece = tokenizer.encode(remaining)
            ids.extend(piece)
            mask.extend([1] * len(piece))
            break
        cut = found + len(marker)
        piece = tokenizer.encode(remaining[:cut])
        ids.extend(piece)
        mask.extend([0] * len(piece))
        remaining = remaining[cut:]
        if not remaining:
            ids.extend(end_turn)
            mask.extend([1] * len(end_turn))
            break
        user_at = remaining.find("\nUser:")
        answer_text = remaining if user_at == -1 else remaining[:user_at]
        answer = tokenizer.encode(answer_text)
        ids.extend(answer)
        mask.extend([1] * len(answer))
        ids.extend(end_turn)
        mask.extend([1] * len(end_turn))
        if user_at == -1:
            break
        remaining = remaining[user_at + 1:]
        starts.append(len(ids))
    return ids, mask, starts


def pack_documents(
    documents: list[Document],
    tokenizer: BPETokenizer,
    seq_len: int,
) -> list[tuple[list[int], list[int]]]:
    """Pack documents into rows of ``seq_len + 1`` tokens and a loss mask.

    One row starts at every document and at every later user turn. Chat
    prompts then occupy position 0, including questions that were a second
    turn in the raw dialogue. Non-overlapping slices cover the rest of a
    long document. The mask travels with the tokens, including a wrap.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    width = seq_len + 1
    separator = tokenizer.encode("\n\n")
    stream: list[int] = []
    mask_stream: list[int] = []
    starts: list[int] = []
    for doc in documents:
        ids, mask, local_starts = encode_document(tokenizer, doc.text)
        if not ids:
            continue
        if stream and separator:
            stream.extend(separator)
            mask_stream.extend([0] * len(separator))
        base = len(stream)
        starts.extend(base + local for local in local_starts)
        stream.extend(ids)
        mask_stream.extend(mask)
    if not stream:
        raise ValueError("corpus is too small to pack one training sequence")
    if len(stream) < width:
        base_ids = list(stream)
        base_mask = list(mask_stream)
        while len(stream) < width:
            if separator:
                stream.extend(separator)
                mask_stream.extend([0] * len(separator))
            stream.extend(base_ids)
            mask_stream.extend(base_mask)

    def row_at(start: int) -> tuple[list[int], list[int]]:
        ids = stream[start:start + width]
        mask = mask_stream[start:start + width]
        if len(ids) < width:
            need = width - len(ids)
            ids = ids + stream[:need]
            mask = mask + mask_stream[:need]
        return ids, mask

    rows: list[tuple[list[int], list[int]]] = []
    seen: set[tuple[int, ...]] = set()

    def add(start: int) -> None:
        ids, mask = row_at(start)
        key = tuple(ids)
        if key not in seen:
            seen.add(key)
            rows.append((ids, mask))

    for start in starts:
        add(start)
    for start in range(0, len(stream) - width + 1, width):
        add(start)
    return rows


def prepare(
    raw_dir: str | Path,
    mixture_path: str | Path,
    out_dir: str | Path,
) -> dict:
    mixture = json.loads(Path(mixture_path).read_text(encoding="utf-8"))
    documents, preferences = load_raw(raw_dir)
    unique, duplicate_drops = dedup(documents)
    kept, dropped = filter_documents(unique)
    dropped["duplicate"] += duplicate_drops
    kept_preferences, preference_drops = filter_preferences(preferences)

    vocab_size = int(mixture.get("vocab_size", 512))
    seq_len = int(mixture["seq_len"])
    target_tokens = int(mixture.get("target_tokens", 8192))
    seed = int(mixture.get("seed", 0))
    weights = {key: float(value) for key, value in mixture["weights"].items()}

    tokenizer_text = "\n\n".join(doc.text for doc in kept)
    tokenizer = BPETokenizer.train(tokenizer_text, vocab_size=vocab_size)
    mixed = mix_documents(kept, weights, tokenizer, target_tokens, seed)
    packed = pack_documents(mixed, tokenizer, seq_len)
    rows = [ids for ids, _mask in packed]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    _write_jsonl(out_dir / "documents.jsonl", [
        {"id": doc.id, "domain": doc.domain, "lang": doc.lang, "text": doc.text}
        for doc in kept
    ])
    _write_jsonl(out_dir / "train.jsonl", [
        {"ids": ids, "mask": mask} for ids, mask in packed
    ])
    _write_jsonl(out_dir / "preference.jsonl", [
        {
            "id": pair.id,
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }
        for pair in kept_preferences
    ])
    train_text = tokenizer.decode([token for row in rows for token in row])
    (out_dir / "train.txt").write_text(train_text, encoding="utf-8")

    manifest = {
        "seq_len": seq_len,
        "vocab_size": tokenizer.vocab_size,
        "target_tokens": target_tokens,
        "documents_in": len(documents),
        "documents_kept": len(kept),
        "preference_in": len(preferences),
        "preference_kept": len(kept_preferences),
        "preference_dropped": preference_drops,
        "dropped": dict(dropped),
        "by_domain": dict(Counter(doc.domain for doc in kept)),
        "by_language": dict(Counter(doc.lang for doc in kept)),
        "packed_sequences": len(rows),
        "packed_tokens": len(rows) * (seq_len + 1),
        "weights": weights,
        "train": "train.jsonl",
        "train_text": "train.txt",
        "documents": "documents.jsonl",
        "preference": "preference.jsonl",
        "tokenizer": "tokenizer.json",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_training_source(path: str | Path) -> TrainingSource:
    """Load either a raw text file or a prepared mixture manifest."""
    path = Path(path)
    if path.name != "manifest.json":
        return TrainingSource(text=path.read_text(encoding="utf-8"))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    tokenizer = BPETokenizer.load(base / manifest["tokenizer"])
    rows: list[list[int]] = []
    masks: list[list[int]] = []
    saw_mask = False
    for line in (base / manifest["train"]).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append([int(token) for token in record["ids"]])
        if "mask" in record:
            saw_mask = True
            masks.append([int(value) for value in record["mask"]])
    if not rows:
        raise ValueError(f"packed training file is empty: {base / manifest['train']}")
    if saw_mask and len(masks) != len(rows):
        raise ValueError("packed masks do not cover every training row")
    return TrainingSource(
        packed_rows=rows,
        packed_masks=masks if saw_mask else None,
        tokenizer=tokenizer,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare the packed training mixture")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--mixture", default="data/mixture.json")
    parser.add_argument("--out", default="data/processed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = prepare(args.raw, args.mixture, args.out)
    print(
        f"kept {manifest['documents_kept']}/{manifest['documents_in']} documents; "
        f"packed {manifest['packed_sequences']} sequences; "
        f"preference {manifest['preference_kept']}; "
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
