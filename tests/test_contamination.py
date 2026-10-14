"""Tests for ``syrinx.contamination``: normalization, hashing, index, scanning, examples."""

import os
import subprocess
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

from syrinx.contamination import (
    MULTIPLIER,
    BenchmarkIndex,
    N,
    build_index,
    example_text,
    init_worker,
    ngram_hashes,
    normalize_words,
    scan_document,
    scan_shard,
    scan_tokens,
    word_hashes,
)
from syrinx.data.shards import shard_path, write_shard_file
from syrinx.tokenizer import VOCAB_SIZE, encode_document, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "syrinx-bpe-32k" / "tokenizer.json"
QUESTION = (
    "The quick brown fox jumps over the lazy dog while the farmer sleeps under an old oak tree"
)
OTHER = "Photosynthesis converts light energy into chemical energy that plants store as sugar in their leaves every day"


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    return load_tokenizer(TOKENIZER, VOCAB_SIZE)


def _naive(words: list[str], n: int) -> list[int]:
    hs = [zlib.crc32(w.encode("utf-8")) for w in words]
    return [
        sum(hs[j + k] * pow(MULTIPLIER, n - 1 - k, 2**64) for k in range(n)) % 2**64
        for j in range(len(words) - n + 1)
    ]


# normalize_words ------------------------------------------------------------------------------


def test_normalize_case_and_punctuation() -> None:
    assert normalize_words("Hello, World!  It's 3.14") == ["hello", "world", "it", "s", "3", "14"]
    assert normalize_words("snake_case and--dashes\tand\nlines") == [
        "snake",
        "case",
        "and",
        "dashes",
        "and",
        "lines",
    ]


def test_normalize_unicode() -> None:
    assert normalize_words("ﬁne Ｆｕｌｌ") == ["fine", "full"]  # NFKC: ligature, full width
    assert normalize_words("café") == normalize_words("café") == ["café"]  # NFD == NFC
    assert normalize_words("ÉCOLE Straße") == ["école", "straße"]
    assert normalize_words("日本語 テキスト") == ["日本語", "テキスト"]


def test_normalize_empty() -> None:
    assert normalize_words("") == [] and normalize_words("...  !!") == []


# word_hashes / ngram_hashes ------------------------------------------------------------------


def test_word_hashes_are_crc32_uint64() -> None:
    h = word_hashes(["hello", "café", ""])
    assert h.dtype == np.uint64 and h.shape == (3,)
    assert h.tolist() == [zlib.crc32(b"hello"), zlib.crc32("café".encode()), zlib.crc32(b"")]
    assert word_hashes([]).shape == (0,)


def test_word_hashes_deterministic_across_processes() -> None:
    code = "from syrinx.contamination import word_hashes; print(word_hashes(['alpha','beta','gamma']).tolist())"
    outs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(
            subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
            ).stdout
        )
    assert outs[0] == outs[1] == str(word_hashes(["alpha", "beta", "gamma"]).tolist()) + "\n"


@pytest.mark.parametrize(("w", "n"), [(30, 13), (13, 13), (5, 3), (40, 1)])
def test_ngram_hashes_match_naive(w: int, n: int) -> None:
    words = [f"w{i % 7}x{i}" for i in range(w)]
    out = ngram_hashes(word_hashes(words), n)
    assert out.dtype == np.uint64 and out.shape == (w - n + 1,)
    assert out.tolist() == _naive(words, n)


def test_ngram_hashes_too_short_is_empty() -> None:
    out = ngram_hashes(word_hashes(["a"] * 12), 13)
    assert out.dtype == np.uint64 and out.shape == (0,)
    assert ngram_hashes(np.zeros(0, dtype=np.uint64)).shape == (0,)


def test_ngram_hashes_default_n_and_order_sensitive() -> None:
    words = normalize_words(QUESTION)
    assert ngram_hashes(word_hashes(words)).tolist() == _naive(words, N)
    swapped = [words[1], words[0], *words[2:]]
    assert ngram_hashes(word_hashes(swapped))[0] != ngram_hashes(word_hashes(words))[0]


def test_ngram_hashes_wrap_without_warnings() -> None:
    hashes = np.full(100, np.uint64(2**32 - 1), dtype=np.uint64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ngram_hashes(hashes)
    assert out.tolist() == [_naive_from_hashes([2**32 - 1] * 13)] * 88


def _naive_from_hashes(hs: list[int]) -> int:
    return sum(h * pow(MULTIPLIER, 12 - k, 2**64) for k, h in enumerate(hs)) % 2**64


# build_index -------------------------------------------------------------------------------------


def test_build_index_contents() -> None:
    index = build_index({"a": QUESTION, "b": OTHER, "short": "only five words in here"})
    assert index.n == 13 and index.unchecked == ["short"]
    qa, qb = normalize_words(QUESTION), normalize_words(OTHER)
    expected = {tuple(qa[j : j + 13]) for j in range(len(qa) - 12)} | {
        tuple(qb[j : j + 13]) for j in range(len(qb) - 12)
    }
    assert set(index.owners) == expected
    assert index.hashes.dtype == np.uint64
    assert index.hashes.tolist() == sorted(set(_naive(qa, 13) + _naive(qb, 13)))


def test_build_index_shared_ngram_has_both_owners() -> None:
    index = build_index({"x": QUESTION, "y": QUESTION + " again"})
    first = tuple(normalize_words(QUESTION)[:13])
    assert index.owners[first] == {"x", "y"}
    assert len(index.hashes) == len(set(index.hashes.tolist()))  # de-duplicated


def test_build_index_respects_n_and_handles_no_examples() -> None:
    index = build_index({"a": "one two three four"}, n=3)
    assert index.n == 3 and len(index.owners) == 2 and index.unchecked == []
    empty = build_index({})
    assert empty.hashes.shape == (0,) and empty.owners == {} and empty.unchecked == []


# scan_document -------------------------------------------------------------------------------------


def test_planted_contamination_found_despite_case_and_punctuation() -> None:
    index = build_index({"q": QUESTION, "o": OTHER})
    doc = "Some intro text. " + QUESTION.upper().replace(" over ", ", over: ") + "! And more text."
    hits = scan_document(doc, index)
    assert set(hits) == {"q"}
    assert hits["q"] == " ".join(normalize_words(QUESTION)[:13])  # first matching window


def test_twelve_word_overlap_not_flagged() -> None:
    index = build_index({"q": QUESTION})
    twelve = " ".join(normalize_words(QUESTION)[:12])
    assert scan_document(f"start {twelve} end", index) == {}


def test_hash_collision_rejected_by_word_check() -> None:
    words = normalize_words(QUESTION)
    forged = BenchmarkIndex(
        n=13, hashes=np.unique(ngram_hashes(word_hashes(words))), owners={}, unchecked=[]
    )
    assert scan_document(QUESTION, forged) == {}


def test_scan_document_short_or_empty_index() -> None:
    assert scan_document("too short", build_index({"q": QUESTION})) == {}
    assert scan_document(QUESTION, build_index({})) == {}


def test_scan_document_reports_every_contaminated_example() -> None:
    index = build_index({"q": QUESTION, "o": OTHER, "x": "x " * 20})
    hits = scan_document(f"{OTHER}. Then: {QUESTION}", index)
    assert set(hits) == {"q", "o"}


# scan_tokens / scan_shard ----------------------------------------------------------------------------


def _tokens(tokenizer, docs: list[str]) -> np.ndarray:  # type: ignore[no-untyped-def]
    ids: list[int] = []
    for doc in docs:
        ids += encode_document(tokenizer, doc)  # EOS appended
    return np.array(ids, dtype=np.uint16)


def test_no_match_across_document_boundary(tokenizer) -> None:  # type: ignore[no-untyped-def]
    index = build_index({"q": QUESTION})
    words = QUESTION.split()
    split = _tokens(tokenizer, ["intro " + " ".join(words[:7]), " ".join(words[7:]) + " outro"])
    assert scan_tokens(split, tokenizer, index)[2] == {}
    joined = _tokens(tokenizer, ["intro " + QUESTION + " outro"])
    assert set(scan_tokens(joined, tokenizer, index)[2]) == {"q"}


def test_scan_tokens_counts_and_max_docs(tokenizer) -> None:  # type: ignore[no-untyped-def]
    index = build_index({"q": QUESTION, "o": OTHER})
    docs = ["clean text here", OTHER, "more clean words", QUESTION]
    documents, words, hits = scan_tokens(_tokens(tokenizer, docs), tokenizer, index)
    assert documents == 4 and words == sum(len(normalize_words(d)) for d in docs)
    assert set(hits) == {"q", "o"}
    documents, words, hits = scan_tokens(_tokens(tokenizer, docs), tokenizer, index, max_docs=2)
    assert documents == 2 and set(hits) == {"o"}
    assert words == len(normalize_words(docs[0])) + len(normalize_words(docs[1]))


def test_scan_tokens_many_batches(tokenizer, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Every batch must be scanned: hits in the first and middle batches, words from all."""
    from syrinx import contamination

    monkeypatch.setattr(contamination, "DECODE_BATCH", 2)
    index = build_index({"q": QUESTION, "o": OTHER, "x": "x " * 20})
    docs = [QUESTION, "a b c", "d e f", OTHER, "g h", "i j k"]  # batches: [q, .], [., o], [., .]
    documents, words, hits = scan_tokens(_tokens(tokenizer, docs), tokenizer, index)
    assert documents == 6 and set(hits) == {"q", "o"}
    assert words == sum(len(normalize_words(d)) for d in docs)


def test_scan_shard_with_worker_state(tmp_path: Path, tokenizer) -> None:  # type: ignore[no-untyped-def]
    path = shard_path(tmp_path, "train", 0)
    docs = ["hello there", QUESTION]
    write_shard_file(path, _tokens(tokenizer, docs), len(docs), {})
    init_worker(build_index({"q": QUESTION}), TOKENIZER)
    scan = scan_shard(path)
    assert scan.shard == str(path) and scan.documents == 2 and set(scan.hits) == {"q"}
    assert scan.words == sum(len(normalize_words(d)) for d in docs) and scan.seconds >= 0
    assert scan_shard(path, max_docs=1).hits == {}


# example_text ------------------------------------------------------------------------------------------


def test_example_text_per_benchmark() -> None:
    assert (
        example_text("hellaswag", {"ctx": "A man", "endings": ["runs.", "sits."], "label": "0"})
        == "A man runs. sits."
    )
    arc = {
        "question": "Why?",
        "choices": {"text": ["Because", "No"], "label": ["A", "B"]},
        "id": "x",
    }
    assert example_text("arc_easy", arc) == "Why? Because No"
    assert (
        example_text("piqa", {"goal": "Open jar", "sol1": "twist", "sol2": "push", "label": 0})
        == "Open jar twist push"
    )
    assert example_text("lambada", {"text": "the end"}) == "the end"


def test_example_text_unknown_benchmark() -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        example_text("mmlu", {"question": "?"})


def test_exactly_n_words_is_checked() -> None:
    thirteen = " ".join(normalize_words(QUESTION)[:13])
    index = build_index({"edge": thirteen, "twelve": " ".join(thirteen.split()[:12])})
    assert index.unchecked == ["twelve"]
    assert set(scan_document(f"before {thirteen} after", index)) == {"edge"}
