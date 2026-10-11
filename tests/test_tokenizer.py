"""Tests for ``syrinx.tokenizer``.

These train a SMALL tokenizer (vocab 300) on a tiny synthetic corpus, which takes
milliseconds. The real 32,768-token tokenizer is checked by the Phase 1 run itself.
"""

import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers

from syrinx.tokenizer import (
    EOS_TOKEN,
    PAD_TOKEN,
    TokenizerError,
    build_tokenizer,
    bytes_per_token,
    decode,
    encode_document,
    encode_documents,
    file_sha256,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
    validate_tokenizer,
)

SMALL_VOCAB = 300
"""Trainable on CORPUS (which supports up to ~327 tokens)."""

_WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "while",
    "small",
    "birds",
    "sing",
    "near",
    "quiet",
    "river",
    "banks",
]
CORPUS: list[str] = [
    " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(8)) + "." for i in range(300)
]
"""ASCII-only on purpose: round-trip tests then use characters never seen in training."""

NUMBER_CORPUS: list[str] = [
    f"in {1900 + i % 120} we counted {i * 37} birds and {i % 7}0 dogs." for i in range(300)
]
"""Number-heavy text: plenty of repeated digit pairs that BPE would merge if allowed."""

NUMBER_TEXT = "year 2026, pi=3.14, x12y, -42 and 1e-5"

ASCII_DIGITS = frozenset("0123456789")
"""Not str.isdigit(): that is also True for byte symbols like superscript two."""


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    """A small trained tokenizer shared by the module's tests."""
    return train_tokenizer(CORPUS, vocab_size=SMALL_VOCAB, min_frequency=1)


# --- construction and training --------------------------------------------------------


def test_build_settings() -> None:
    """Pre-tokenizer is Sequence[Digits(individual), ByteLevel(no prefix space, GPT-2
    regex)] in that order; ByteLevel decoder; no normalizer."""
    t = build_tokenizer()
    pre = json.loads(t.to_str())["pre_tokenizer"]
    assert pre["type"] == "Sequence"
    digits, byte_level = pre["pretokenizers"]
    assert digits == {"type": "Digits", "individual_digits": True}
    assert byte_level["type"] == "ByteLevel"
    assert byte_level["add_prefix_space"] is False
    assert byte_level["use_regex"] is True
    assert isinstance(t.decoder, decoders.ByteLevel)
    assert t.normalizer is None


def test_digits_pretokenized_individually() -> None:
    """Every pre-token that contains a digit is exactly one digit, so no merge can ever
    join digits ("2026" is always 2, 0, 2, 6)."""
    pieces = [p for p, _ in build_tokenizer().pre_tokenizer.pre_tokenize_str(NUMBER_TEXT)]
    with_digits = [p for p in pieces if any(c in ASCII_DIGITS for c in p)]
    assert len(with_digits) == sum(c in ASCII_DIGITS for c in NUMBER_TEXT)
    assert all(len(p) == 1 for p in with_digits)


def test_no_digit_merges_after_training() -> None:
    """Trained on number-heavy text, the vocab still has no multi-character token
    containing a digit (without digit splitting this corpus learns e.g. space + "19")."""
    t = train_tokenizer(CORPUS + NUMBER_CORPUS, vocab_size=SMALL_VOCAB, min_frequency=1)
    digit_tokens = [s for s in t.get_vocab() if any(c in ASCII_DIGITS for c in s)]
    assert sorted(digit_tokens) == sorted(ASCII_DIGITS)
    text = "in 2026 we counted 1999 birds"
    assert decode(t, encode_document(t, text, append_eos=False)) == text


def test_vocab_size_exact(tok: Tokenizer) -> None:
    """The trained vocabulary has exactly the requested size."""
    assert tok.get_vocab_size() == SMALL_VOCAB


def test_special_token_ids(tok: Tokenizer) -> None:
    """``<|endoftext|>`` is id 0 and ``<|pad|>`` is id 1."""
    assert tok.token_to_id(EOS_TOKEN) == 0
    assert tok.token_to_id(PAD_TOKEN) == 1


def test_all_bytes_in_vocab(tok: Tokenizer) -> None:
    """All 256 ByteLevel symbols are in the vocab, though CORPUS is ASCII-only."""
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    assert len(alphabet) == 256
    assert all(tok.token_to_id(symbol) is not None for symbol in alphabet)


def test_undersized_vocab_raises() -> None:
    """Asking for more tokens than CORPUS supports raises instead of shrinking silently."""
    with pytest.raises(TokenizerError, match="1000"):
        train_tokenizer(CORPUS, vocab_size=1000, min_frequency=1)


def test_trains_from_a_generator() -> None:
    """Training accepts a one-pass iterator (the training script streams documents)."""
    t = train_tokenizer((text for text in CORPUS), vocab_size=SMALL_VOCAB, min_frequency=1)
    assert t.get_vocab_size() == SMALL_VOCAB


def test_training_is_deterministic(tok: Tokenizer) -> None:
    """Training again on the same corpus gives an identical tokenizer."""
    again = train_tokenizer(CORPUS, vocab_size=SMALL_VOCAB, min_frequency=1)
    assert again.to_str() == tok.to_str()


# --- validation -----------------------------------------------------------------------


def test_validate_rejects_wrong_size(tok: Tokenizer) -> None:
    """validate_tokenizer raises when the expected size differs."""
    with pytest.raises(TokenizerError):
        validate_tokenizer(tok, vocab_size=SMALL_VOCAB + 1)


def test_validate_rejects_misplaced_special_tokens() -> None:
    """A tokenizer whose special tokens got ids in the wrong order is rejected."""
    t = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=SMALL_VOCAB,
        min_frequency=1,
        special_tokens=[PAD_TOKEN, EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    t.train_from_iterator(CORPUS, trainer)
    with pytest.raises(TokenizerError):
        validate_tokenizer(t, vocab_size=SMALL_VOCAB)


# --- round-trip (Gate 1) --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        " leading space",
        "trailing space ",
        "multiple   spaces\tand\ttabs",
        "numbers: 2026, 3.14, -42, 1e-5, 0x1F, 1,000,000",
        "windows\r\nnewlines\n\n\nblank lines",
        "unicode: h\u00e9llo na\u00efve caf\u00e9",
        "CJK: \u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",
        "emoji: \U0001f642\U0001f44d\U0001f3fd and ZWJ \U0001f468\u200d\U0001f469\u200d\U0001f467",
        "",
    ],
)
def test_roundtrip_exact(tok: Tokenizer, text: str) -> None:
    """``decode(encode_document(text, append_eos=False)) == text``, byte for byte, for
    text full of characters that never appeared in training."""
    assert decode(tok, encode_document(tok, text, append_eos=False)) == text


def test_roundtrip_non_nfc_unchanged(tok: Tokenizer) -> None:
    """Decomposed "cafe" + combining accent round-trips unchanged: no normalization."""
    text = "cafe\u0301"
    assert decode(tok, encode_document(tok, text, append_eos=False)) == text


def test_ids_in_range(tok: Tokenizer) -> None:
    """Every id is a valid vocab index (no unknown token) and fits uint16."""
    ids = encode_document(tok, "\u0645\u0631\u062d\u0628\u0627 \u05e9\u05dc\u05d5\u05dd \U0001f680")
    assert ids
    assert all(0 <= i < SMALL_VOCAB for i in ids)
    assert max(ids) < 2**16


def test_empty_document_is_just_eos(tok: Tokenizer) -> None:
    """An empty document encodes to ``[0]`` with EOS, ``[]`` without."""
    assert encode_document(tok, "") == [0]
    assert encode_document(tok, "", append_eos=False) == []


# --- special tokens -------------------------------------------------------------------


def test_eos_appended_once(tok: Tokenizer) -> None:
    """With append_eos, the ids end with exactly one EOS id (0)."""
    ids = encode_document(tok, "the quick brown fox.")
    assert ids[-1] == 0
    assert ids.count(0) == 1
    assert encode_document(tok, "the quick brown fox.", append_eos=False) == ids[:-1]


def test_literal_special_string_is_not_special(tok: Tokenizer) -> None:
    """A document containing the text "<|endoftext|>" encodes it as ordinary tokens:
    the only id 0 is the appended one, and the text round-trips."""
    text = f"before {EOS_TOKEN} after {PAD_TOKEN}"
    ids = encode_document(tok, text)
    assert ids.count(0) == 1 and ids[-1] == 0
    assert 1 not in ids
    assert decode(tok, ids[:-1]) == text


@pytest.mark.parametrize("initial", [False, True])
def test_encode_document_has_no_side_effect(tok: Tokenizer, initial: bool) -> None:
    """``encode_special_tokens`` is restored to whatever it was before the call."""
    tok.encode_special_tokens = initial
    try:
        encode_document(tok, f"x {EOS_TOKEN} y")
        assert tok.encode_special_tokens is initial
    finally:
        tok.encode_special_tokens = False


def test_decode_keeps_special_tokens(tok: Tokenizer) -> None:
    """decode shows the EOS text where id 0 appears (the library default drops it)."""
    assert decode(tok, encode_document(tok, "hi")) == f"hi{EOS_TOKEN}"


# --- persistence ----------------------------------------------------------------------


def test_save_load_roundtrip(tok: Tokenizer, tmp_path: Path) -> None:
    """save_tokenizer returns the file's SHA-256; the loaded tokenizer encodes
    identically; parent directories are created."""
    path = tmp_path / "nested" / "tokenizer.json"
    digest = save_tokenizer(tok, path, vocab_size=SMALL_VOCAB)
    assert digest == file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_tokenizer(path, vocab_size=SMALL_VOCAB)
    text = "the lazy dog sings \u00e9 \U0001f642"
    assert encode_document(loaded, text) == encode_document(tok, text)


def test_save_refuses_overwrite(tok: Tokenizer, tmp_path: Path) -> None:
    """Saving to an existing path raises FileExistsError and leaves the file intact."""
    path = tmp_path / "tokenizer.json"
    path.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_tokenizer(tok, path, vocab_size=SMALL_VOCAB)
    assert path.read_text(encoding="utf-8") == "keep me"


def test_save_validates(tok: Tokenizer, tmp_path: Path) -> None:
    """Saving with the wrong expected size raises TokenizerError and writes nothing."""
    path = tmp_path / "tokenizer.json"
    with pytest.raises(TokenizerError):
        save_tokenizer(tok, path)  # default expects 32,768
    assert not path.exists()


def test_load_rejects_wrong_vocab(tok: Tokenizer, tmp_path: Path) -> None:
    """Loading with a mismatched expected size raises TokenizerError."""
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tok, path, vocab_size=SMALL_VOCAB)
    with pytest.raises(TokenizerError):
        load_tokenizer(path, vocab_size=999)


def test_load_missing_file(tmp_path: Path) -> None:
    """Loading a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_tokenizer(tmp_path / "nope.json")


def test_file_sha256_known_value(tmp_path: Path) -> None:
    """An empty file hashes to the published SHA-256 of empty input."""
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert file_sha256(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# --- compression ----------------------------------------------------------------------


def test_bytes_per_token_counts_utf8_bytes(tok: Tokenizer) -> None:
    """Total UTF-8 bytes (not characters) over total tokens, with no EOS counted."""
    texts = ["the quick brown fox.", "caf\u00e9 \U0001f642"]
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(encode_document(tok, t, append_eos=False)) for t in texts)
    assert n_bytes == 20 + 10
    assert bytes_per_token(tok, texts) == n_bytes / n_tokens


def test_bytes_per_token_compresses_training_text(tok: Tokenizer) -> None:
    """On text like the training corpus, merges give more than 1.5 bytes per token."""
    assert bytes_per_token(tok, CORPUS[:20]) > 1.5


@pytest.mark.parametrize("texts", [[], [""]])
def test_bytes_per_token_empty_raises(tok: Tokenizer, texts: list[str]) -> None:
    """No tokens at all raises ValueError instead of dividing by zero."""
    with pytest.raises(ValueError):
        bytes_per_token(tok, texts)


# --- batched encoding -----------------------------------------------------------------

BATCH_TEXTS: list[str] = [
    *CORPUS[:20],
    "",
    "numbers: 2026, 3.14, -42",
    f"literal {EOS_TOKEN} and {PAD_TOKEN} inside",
    "caf\u00e9 \u65e5\u672c \U0001f642 \U0001f468\u200d\U0001f469\u200d\U0001f467",
    " leading space\nand\ttabs ",
]


def test_encode_documents_matches_encode_document(tok: Tokenizer) -> None:
    """Batched ids equal per-document ids (without EOS), document by document, including
    empty text, digits, emoji and literal special-token strings."""
    expected = [encode_document(tok, t, append_eos=False) for t in BATCH_TEXTS]
    assert encode_documents(tok, BATCH_TEXTS) == expected


def test_encode_documents_no_eos_and_no_specials(tok: Tokenizer) -> None:
    """No id 0 or 1 appears: EOS is not appended and literal special strings stay text."""
    ids = encode_documents(tok, BATCH_TEXTS)
    assert all(0 not in doc and 1 not in doc for doc in ids)


def test_encode_documents_empty_list(tok: Tokenizer) -> None:
    """No documents gives no id lists."""
    assert encode_documents(tok, []) == []


@pytest.mark.parametrize("initial", [False, True])
def test_encode_documents_restores_setting(tok: Tokenizer, initial: bool) -> None:
    """``encode_special_tokens`` is restored to its previous value."""
    tok.encode_special_tokens = initial
    try:
        encode_documents(tok, [f"x {EOS_TOKEN} y"])
        assert tok.encode_special_tokens is initial
    finally:
        tok.encode_special_tokens = False


def test_encode_documents_returns_plain_lists(tok: Tokenizer) -> None:
    """The result is a list of lists of Python ints (not Encoding objects)."""
    ids = encode_documents(tok, ["the dog", "a fox"])
    assert isinstance(ids, list) and all(isinstance(doc, list) for doc in ids)
    assert all(isinstance(i, int) for doc in ids for i in doc)
