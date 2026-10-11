"""The one document path every stage uses: normalize, dedup, filter, split.

Tokenizer training (Phase 1) and shard building (Phase 2) must agree on which documents
are train and which are val, or Gate 2's overlap check and the tokenizer's train-only
guarantee refer to different sets. This module is that single definition.
"""

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from syrinx.data.documents import SPLIT_SALT, Split, assign_split, document_hash, normalize_text

logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS = 200
"""Documents shorter than this (in characters, after normalization) are dropped
(docs/decisions.md, 2026-10-11)."""


@dataclass
class DocumentStats:
    """Counts from :func:`iter_documents`.

    Every document read is counted in exactly one of ``empty``, ``duplicates``,
    ``too_short``, ``train_docs`` or ``val_docs``, so ``seen`` equals their sum.
    """

    seen: int = 0
    empty: int = 0
    duplicates: int = 0
    too_short: int = 0
    train_docs: int = 0
    val_docs: int = 0


def iter_documents(
    raw_texts: Iterable[str],
    val_fraction: float,
    min_chars: int = DEFAULT_MIN_CHARS,
    stats: DocumentStats | None = None,
    salt: str = SPLIT_SALT,
) -> Iterator[tuple[Split, str]]:
    """Yield ``(split, normalized_text)`` for every document that survives the pipeline.

    For each raw document, in this order:
      1. ``normalize_text``; if the result is empty, count ``empty`` and skip.
      2. ``document_hash``; if the hash was seen before, count ``duplicates`` and skip.
         (The hash is remembered even if step 3 then drops the document.)
      3. If ``len(text) < min_chars`` (characters, not bytes), count ``too_short`` and
         skip.
      4. ``assign_split(hash, val_fraction, salt)``; count ``train_docs`` or
         ``val_docs``, then yield.
    Stats are updated before each yield. The function is lazy: it reads one raw document
    per yielded-or-skipped document and never reads ahead.

    With ``min_chars=0`` the train documents are exactly those the tokenizer training
    script selects (tested), which is what makes the two stages consistent.

    Args:
        raw_texts: Raw documents, e.g. a streamed dataset.
        val_fraction: Fraction of documents assigned to val, in (0, 1).
        min_chars: Minimum normalized length in characters; 0 disables the filter.
        stats: Updated in place if given.
        salt: Split namespace; keep the default.

    Yields:
        ``("train" | "val", normalized_text)``.

    Raises:
        ValueError: If ``min_chars`` is negative (raised on first iteration), or
            ``val_fraction`` is not in (0, 1) (from ``assign_split``).
    """
    if min_chars < 0:
        raise ValueError(f"min_chars must be >= 0, got {min_chars}")
    if stats is None:
        stats = DocumentStats()
    seen_hashes: set[str] = set()
    for raw in raw_texts:
        stats.seen += 1
        text = normalize_text(raw)
        if not text:
            stats.empty += 1
            continue
        doc_hash = document_hash(text=text)
        if doc_hash in seen_hashes:
            stats.duplicates += 1
            continue
        seen_hashes.add(doc_hash)
        if len(text) < min_chars:
            stats.too_short += 1
            continue
        split = assign_split(doc_hash=doc_hash, val_fraction=val_fraction, salt=salt)
        if split == "train":
            stats.train_docs += 1
        else:
            stats.val_docs += 1
        yield split, text
