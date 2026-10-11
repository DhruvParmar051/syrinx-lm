"""Tests for ``syrinx.data.selection``."""

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest

from syrinx.data.documents import assign_split, document_hash, normalize_text
from syrinx.data.selection import DEFAULT_MIN_CHARS, DocumentStats, iter_documents

ROOT = Path(__file__).resolve().parents[1]

DOCS: list[str] = [f"document number {i} " + "word " * (i % 60) for i in range(600)]
"""600 distinct documents, 16 to 315 characters long."""


def _run(
    raw: list[str], val_fraction: float = 0.2, min_chars: int = 0
) -> tuple[list[tuple[str, str]], DocumentStats]:
    stats = DocumentStats()
    return list(iter_documents(raw, val_fraction, min_chars, stats)), stats


def test_default_min_chars() -> None:
    """The agreed length filter is 200 characters."""
    assert DEFAULT_MIN_CHARS == 200


def test_yields_split_and_normalized_text() -> None:
    """Each item is (split, normalize_text(raw)), split by the content hash."""
    raw = [f"line\r\nnumber {i}\x00" for i in range(100)]
    items, _ = _run(raw)
    expected = []
    for r in raw:
        text = normalize_text(r)
        expected.append((assign_split(document_hash(text), 0.2), text))
    assert items == expected
    assert {split for split, _ in items} == {"train", "val"}


def test_matches_tokenizer_script_selection() -> None:
    """With the length filter off, the train documents are exactly the ones the Phase 1
    tokenizer script selects (same dedup, same normalization, same split)."""
    spec = importlib.util.spec_from_file_location(
        "train_tokenizer", ROOT / "scripts" / "train_tokenizer.py"
    )
    assert spec is not None and spec.loader is not None
    train_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_script)

    raw = [*DOCS, DOCS[3], DOCS[5].replace(" ", " \r\n", 1), "", "\x00"]
    items, stats = _run(raw, val_fraction=0.2, min_chars=0)
    script_stats = train_script.SelectionStats()
    script_train = list(train_script.select_train_texts(raw, 0.2, 10**12, script_stats))
    assert [text for split, text in items if split == "train"] == script_train
    assert stats.val_docs == script_stats.val
    assert stats.duplicates == script_stats.duplicates
    assert stats.empty == script_stats.empty


def test_skips_empty() -> None:
    """Documents empty after normalization are skipped and counted."""
    items, stats = _run(["", "\x00\x01", "hello"], val_fraction=0.000001)
    assert items == [("train", "hello")]
    assert stats.empty == 2


def test_skips_duplicates_after_normalization() -> None:
    """CRLF/LF variants and exact repeats count as the same document."""
    items, stats = _run(["a\nb", "a\r\nb", "a\rb", "a\nb"], val_fraction=0.000001)
    assert items == [("train", "a\nb")]
    assert stats.duplicates == 3


def test_min_chars_drops_short_documents() -> None:
    """Documents shorter than min_chars are dropped and counted; length is measured in
    characters of the normalized text, and a document exactly min_chars long is kept."""
    raw = ["x" * 9, "y" * 10, "é" * 9, "è" * 10, "z" * 11, "w" * 8 + "\r\n"]
    items, stats = _run(raw, val_fraction=0.000001, min_chars=10)
    assert [text for _, text in items] == ["y" * 10, "è" * 10, "z" * 11]
    # dropped: "x"*9; 9 accented letters (18 bytes, but 9 characters); "w"*8 + "\n"
    assert stats.too_short == 3


def test_duplicate_of_short_document_is_a_duplicate() -> None:
    """A dropped short document's hash is still remembered: its repeat is counted as a
    duplicate, not as too_short again."""
    _, stats = _run(["tiny", "tiny"], val_fraction=0.000001, min_chars=10)
    assert (stats.too_short, stats.duplicates) == (1, 1)


def test_stats_add_up() -> None:
    """seen == empty + duplicates + too_short + train_docs + val_docs."""
    raw = ["", DOCS[0], DOCS[0], *DOCS]
    items, stats = _run(raw, val_fraction=0.2, min_chars=100)
    assert stats.seen == len(raw)
    total = stats.empty + stats.duplicates + stats.too_short + stats.train_docs + stats.val_docs
    assert stats.seen == total
    assert stats.train_docs + stats.val_docs == len(items)
    assert stats.too_short > 0 and stats.val_docs > 0


def test_train_and_val_never_share_a_document() -> None:
    """Even with repeats and newline variants in the input, no normalized text (or hash)
    appears in both splits."""
    raw = [*DOCS, *DOCS[:50], *(d.replace(" ", "\r\n", 1) for d in DOCS[:50])]
    items, _ = _run(raw, val_fraction=0.3)
    train = {document_hash(t) for s, t in items if s == "train"}
    val = {document_hash(t) for s, t in items if s == "val"}
    assert train and val
    assert train.isdisjoint(val)


def test_is_lazy() -> None:
    """Reads exactly one raw document per step: taking the first item pulls no more raw
    documents than needed to produce it."""
    pulled: list[int] = []

    def stream() -> Iterator[str]:
        for i, doc in enumerate(DOCS):
            pulled.append(i)
            yield doc

    gen = iter_documents(stream(), 0.5, 0)
    _, first = next(gen)
    assert len(pulled) == DOCS.index(first) + 1


def test_stats_current_at_yield() -> None:
    """Stats already count a document when it is yielded."""
    stats = DocumentStats()
    gen = iter_documents(DOCS, 0.000001, 0, stats)
    next(gen)
    assert (stats.seen, stats.train_docs) == (1, 1)


def test_works_without_stats() -> None:
    """stats is optional."""
    assert len(list(iter_documents(DOCS[:3], 0.000001, 0))) == 3


def test_negative_min_chars_raises() -> None:
    """min_chars < 0 raises ValueError."""
    with pytest.raises(ValueError):
        list(iter_documents(DOCS, 0.2, -1))


def test_bad_val_fraction_raises() -> None:
    """An invalid val_fraction raises ValueError (from assign_split)."""
    with pytest.raises(ValueError):
        list(iter_documents(DOCS, 1.0, 0))


def test_salt_changes_assignment() -> None:
    """The salt is passed to assign_split."""
    default = [s for s, _ in iter_documents(DOCS, 0.5, 0)]
    other = [s for s, _ in iter_documents(DOCS, 0.5, 0, salt="other-salt")]
    assert default != other
