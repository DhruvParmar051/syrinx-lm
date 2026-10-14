"""Benchmark contamination check: 13-gram overlap between benchmark examples and train text.

An example is *contaminated* if any of its word 13-grams also occurs inside one training
document. Both sides use the same normalization (NFKC, lowercase, runs of non-alphanumeric
characters as separators). Word hashes are ``crc32`` (deterministic across processes, unlike
Python's ``hash``); 13-gram hashes are a polynomial rolling hash in ``uint64`` that wraps
mod 2**64. A hash hit only counts after the 13 words themselves are compared, so a hash
collision can never be reported as contamination. Examples with fewer than ``n`` words cannot
be checked and are reported as *unchecked*, never as clean.
"""

import re
import time
import unicodedata
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from syrinx.data.shards import read_shard, split_documents
from syrinx.tokenizer import load_tokenizer

N = 13
"""Words per n-gram (GPT-3 / Llama convention)."""
MULTIPLIER = 0x100000001B3
"""Odd 64-bit multiplier for the rolling hash (the FNV-1 prime)."""
DECODE_BATCH = 1024
"""Documents per ``decode_batch`` call."""
_WORD = re.compile(r"[^\W_]+")


def normalize_words(text: str) -> list[str]:
    """Words of ``text`` after NFKC normalization and lowercasing.

    A word is a maximal run of Unicode letters and digits (``[^\\W_]+``); everything else
    (punctuation, underscores, whitespace) separates words.
    """
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower())


def word_hashes(words: Sequence[str]) -> np.ndarray:
    """``zlib.crc32`` of each word's UTF-8 bytes, as a (W,) ``uint64`` array."""
    return np.fromiter(
        (zlib.crc32(w.encode("utf-8")) for w in words), dtype=np.uint64, count=len(words)
    )


def ngram_hashes(hashes: np.ndarray, n: int = N) -> np.ndarray:
    """Rolling hash of every window of ``n`` consecutive word hashes.

    ``H[j] = sum_{k=0}^{n-1} hashes[j + k] * MULTIPLIER**(n - 1 - k)  (mod 2**64)``, for
    ``j = 0 .. W - n``. Returns a (W - n + 1,) ``uint64`` array, empty if ``W < n``.
    """
    count = len(hashes) - n + 1
    if count <= 0:
        return np.zeros(0, dtype=np.uint64)
    out = np.zeros(count, dtype=np.uint64)
    for k in range(n):
        power = np.uint64(pow(MULTIPLIER, n - 1 - k, 2**64))
        out += hashes[k : k + count] * power
    return out


@dataclass
class BenchmarkIndex:
    """All n-grams of all checkable benchmark examples."""

    n: int
    hashes: np.ndarray
    """Sorted, unique (M,) ``uint64`` n-gram hashes."""
    owners: dict[tuple[str, ...], set[str]]
    """Each n-gram's words -> ids of the examples containing it."""
    unchecked: list[str] = field(default_factory=list)
    """Ids of examples with fewer than ``n`` words."""


def build_index(examples: Mapping[str, str], n: int = N) -> BenchmarkIndex:
    """Index ``examples`` (id -> text).

    Each text goes through :func:`normalize_words`; examples with fewer than ``n`` words go to
    ``unchecked`` (in input order); every window of the others is added to ``owners`` and its
    :func:`ngram_hashes` value to ``hashes`` (sorted, de-duplicated).
    """
    owners: dict[tuple[str, ...], set[str]] = {}
    hashes: list[np.ndarray] = []
    unchecked: list[str] = []
    for example_id, text in examples.items():
        words = normalize_words(text=text)
        if len(words) < n:
            unchecked.append(example_id)
            continue
        hashes.append(ngram_hashes(word_hashes(words=words), n))
        for j in range(len(words) - n + 1):
            owners.setdefault(tuple(words[j : j + n]), set()).add(example_id)
    merged = np.unique(np.concatenate(hashes)) if hashes else np.zeros(0, dtype=np.uint64)
    return BenchmarkIndex(n=n, hashes=merged, owners=owners, unchecked=unchecked)


def scan_document(text: str, index: BenchmarkIndex) -> dict[str, str]:
    """Contaminated example ids found in one training document.

    Candidates are windows whose hash is in the sorted ``index.hashes`` (binary search with
    ``np.searchsorted``, never ``np.isin``, which re-sorts the whole index on every call); a
    candidate counts only if its words are a key of ``index.owners``. Windows never cross document boundaries
    (the caller passes one document at a time).

    Returns:
        ``{example_id: matched n-gram as space-joined words}``, the first match per example.
    """
    words = normalize_words(text=text)
    if len(words) < index.n or len(index.hashes) == 0:
        return {}
    window = ngram_hashes(word_hashes(words), index.n)
    pos = np.minimum(np.searchsorted(index.hashes, window), len(index.hashes) - 1)
    candidates = np.flatnonzero(index.hashes[pos] == window)
    hits: dict[str, str] = {}
    for j in candidates:
        gram = tuple(words[j : j + index.n])
        for example_id in index.owners.get(gram, ()):
            hits.setdefault(example_id, " ".join(gram))
    return hits


@dataclass(frozen=True)
class ShardScan:
    """Result of scanning one shard."""

    shard: str
    documents: int
    words: int
    hits: dict[str, str]
    seconds: float


_WORKER: dict[str, Any] = {}


def init_worker(index: BenchmarkIndex, tokenizer_path: Path) -> None:
    """Process-pool initializer: keep ``index`` and the loaded tokenizer in this process."""
    _WORKER["index"] = index
    _WORKER["tokenizer"] = load_tokenizer(path=tokenizer_path)


def scan_tokens(
    tokens: np.ndarray, tokenizer: Tokenizer, index: BenchmarkIndex, max_docs: int | None = None
) -> tuple[int, int, dict[str, str]]:
    """Split ``tokens`` into documents (:func:`split_documents`), keep the first ``max_docs``
    if given, decode them with ``tokenizer.decode_batch(..., skip_special_tokens=False)`` in
    batches of ``DECODE_BATCH``, and :func:`scan_document` each one.

    Returns:
        ``(documents scanned, words scanned, hits)``; hits keep the first match per example.
    """
    documents = split_documents(tokens=tokens)
    if max_docs is not None:
        documents = documents[:max_docs]
    hits: dict[str, str] = {}
    words = 0
    for start in range(0, len(documents), DECODE_BATCH):
        batch = [doc.tolist() for doc in documents[start : start + DECODE_BATCH]]
        for text in tokenizer.decode_batch(batch, skip_special_tokens=False):
            words += len(normalize_words(text))
            for example_id, gram in scan_document(text=text, index=index).items():
                hits.setdefault(example_id, gram)
    return len(documents), words, hits


def scan_shard(path: Path, max_docs: int | None = None) -> ShardScan:
    """Scan one shard with the index and tokenizer set by :func:`init_worker`."""
    start = time.perf_counter()
    documents, words, hits = scan_tokens(
        read_shard(path=path), _WORKER["tokenizer"], _WORKER["index"], max_docs
    )
    return ShardScan(
        str(path), documents=documents, words=words, hits=hits, seconds=time.perf_counter() - start
    )


BENCHMARK_FIELDS: dict[str, str] = {
    "hellaswag": "ctx + endings",
    "arc_easy": "question + choices.text",
    "piqa": "goal + sol1 + sol2",
    "lambada": "text",
}
"""What :func:`example_text` joins for each benchmark."""


def example_text(benchmark: str, row: Mapping[str, Any]) -> str:
    """The text checked for one benchmark row, parts joined by single spaces.

    - ``hellaswag``: ``row["ctx"]`` then every string in ``row["endings"]``.
    - ``arc_easy``: ``row["question"]`` then every string in ``row["choices"]["text"]``.
    - ``piqa``: ``row["goal"]``, ``row["sol1"]``, ``row["sol2"]``.
    - ``lambada``: ``row["text"]``.

    Raises:
        ValueError: If ``benchmark`` is not a key of :data:`BENCHMARK_FIELDS`.
    """
    if benchmark == "hellaswag":
        parts: Iterable[str] = [row["ctx"], *row["endings"]]
    elif benchmark == "arc_easy":
        parts = [row["question"], *row["choices"]["text"]]
    elif benchmark == "piqa":
        parts = [row["goal"], row["sol1"], row["sol2"]]
    elif benchmark == "lambada":
        parts = [row["text"]]
    else:
        raise ValueError(
            f"unknown benchmark {benchmark!r}; expected one of {sorted(BENCHMARK_FIELDS)}"
        )
    return " ".join(parts)
