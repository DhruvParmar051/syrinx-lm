"""Tests for ``syrinx.data.documents``."""

import hashlib
import subprocess
import sys

import pytest

from syrinx.data.documents import assign_split, document_hash, normalize_text


def _hashes(n: int) -> list[str]:
    """Hashes of n distinct synthetic documents."""
    return [document_hash(f"document number {i}") for i in range(n)]


# --- normalize_text -------------------------------------------------------------------


def test_crlf_and_cr_become_lf() -> None:
    """``"a\\r\\nb\\rc"`` normalizes to ``"a\\nb\\nc"``."""
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_nfc_composes() -> None:
    """``"e\\u0301"`` (e + combining acute) becomes ``"\\u00e9"`` (é)."""
    assert normalize_text("e\u0301") == "\u00e9"


def test_control_chars_removed_but_newline_and_tab_kept() -> None:
    """NUL, BEL, ESC and DEL are removed; ``"\\n"`` and ``"\\t"`` are kept."""
    assert normalize_text("a\x00b\x07c\x1bd\x7fe\n\tf") == "abcde\n\tf"


def test_text_otherwise_unchanged() -> None:
    """Case, leading/trailing spaces, repeated spaces and emoji are untouched."""
    text = "  Hello   World 🙂  "
    assert normalize_text(text) == text


def test_normalize_is_idempotent() -> None:
    """Normalizing twice equals normalizing once, for a messy input."""
    messy = "x\r\n\x00e\u0301\r\x1b y"
    once = normalize_text(messy)
    assert normalize_text(once) == once


# --- document_hash --------------------------------------------------------------------


def test_hash_known_value() -> None:
    """``document_hash("")`` equals the published SHA-256 of empty input."""
    digest = document_hash("")
    assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert len(digest) == 64


def test_hash_uses_utf8_bytes() -> None:
    """Non-ASCII text is hashed as its UTF-8 bytes: "\\u00e9" hashes like b"\\xc3\\xa9"."""
    assert document_hash("\u00e9") == hashlib.sha256(b"\xc3\xa9").hexdigest()


def test_hash_stable_across_processes() -> None:
    """A fresh Python process computes the same hash (guards against ``hash()``,
    which is randomized per process)."""
    text = "the same document"
    code = f"from syrinx.data.documents import document_hash; print(document_hash({text!r}))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == document_hash(text)


def test_equivalent_texts_hash_equal_after_normalization() -> None:
    """NFD and NFC forms of "café" differ as raw strings but hash equal once normalized."""
    nfd = "cafe\u0301"
    nfc = "caf\u00e9"
    assert nfd != nfc
    assert document_hash(normalize_text(nfd)) == document_hash(normalize_text(nfc))


# --- assign_split ---------------------------------------------------------------------


def test_split_returns_train_or_val() -> None:
    """Every assignment is exactly "train" or "val"."""
    assert {assign_split(h, 0.5) for h in _hashes(200)} == {"train", "val"}


def test_split_is_deterministic() -> None:
    """The same hash always gets the same split."""
    doc_hash = document_hash("some document")
    first = assign_split(doc_hash, 0.1)
    assert all(assign_split(doc_hash, 0.1) == first for _ in range(10))


def test_split_pinned_assignments() -> None:
    """Pins the exact algorithm: salted SHA-256 of the hash, first 8 bytes big-endian,
    compared to val_fraction. If this fails, the split changed and every tokenizer and
    shard built so far is inconsistent with it."""
    expected = {
        "doc-a": "train",
        "doc-b": "train",
        "doc-c": "val",
        "doc-d": "train",
        "doc-e": "train",
        "doc-f": "val",
    }
    actual = {text: assign_split(document_hash(text), 0.5) for text in expected}
    assert actual == expected


def test_split_fraction_close_to_target() -> None:
    """Over 20,000 documents at val_fraction=0.05, the val share is within ±0.006
    (about 4 standard deviations of a binomial proportion)."""
    hashes = _hashes(20_000)
    share = sum(assign_split(h, 0.05) == "val" for h in hashes) / len(hashes)
    assert abs(share - 0.05) <= 0.006


def test_split_monotone_in_fraction() -> None:
    """A document in val at fraction 0.1 is still in val at 0.3: raising the fraction
    only moves documents from train into val."""
    hashes = _hashes(2_000)
    val_small = {h for h in hashes if assign_split(h, 0.1) == "val"}
    val_large = {h for h in hashes if assign_split(h, 0.3) == "val"}
    assert val_small
    assert val_small <= val_large


def test_salt_changes_split() -> None:
    """A different salt gives a different assignment for some documents."""
    hashes = _hashes(2_000)
    default = [assign_split(h, 0.5) for h in hashes]
    other = [assign_split(h, 0.5, salt="another-salt") for h in hashes]
    assert default != other


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_fraction_raises(fraction: float) -> None:
    """val_fraction outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError):
        assign_split(document_hash("x"), fraction)


@pytest.mark.parametrize("bad_hash", ["", "abc", "Z" * 64, "a" * 63, "a" * 65])
def test_invalid_hash_raises(bad_hash: str) -> None:
    """A doc_hash that isn't exactly 64 lowercase hex characters raises ValueError."""
    with pytest.raises(ValueError):
        assign_split(bad_hash, 0.1)
