"""Per-document text normalization, content hashing and the train/val split.

The split is decided once per document from a hash of its NORMALIZED text, so every
stage (tokenizer training in Phase 1, shard building in Phase 2, the Gate 2 overlap
check) agrees on which documents are validation, on any machine, in any order.

Order of operations for every document:
    raw text -> normalize_text -> document_hash -> assign_split
"""

import hashlib
import logging
import re
import unicodedata
from typing import Literal

logger = logging.getLogger(__name__)

Split = Literal["train", "val"]

SPLIT_SALT = "syrinx-split-v1"
"""Mixed into the split hash. Changing it re-draws the split: never change it once
shards or a tokenizer exist (bump the version and rebuild everything instead)."""


def normalize_text(text: str) -> str:
    """Normalize one document: newlines, Unicode NFC, control characters.

    Steps, in this order:
      1. ``"\\r\\n"`` and lone ``"\\r"`` become ``"\\n"``.
      2. Unicode NFC (``unicodedata.normalize("NFC", ...)``), so visually identical
         text has one byte representation (matters for dedup and the split hash).
      3. Remove characters in Unicode category ``Cc`` (control) except ``"\\n"`` and
         ``"\\t"``.
    No lowercasing, no whitespace collapsing, no stripping: the model should see
    real text.

    Args:
        text: Raw document text.

    Returns:
        The normalized text. Idempotent: ``normalize_text(normalize_text(x)) ==
        normalize_text(x)``.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def document_hash(text: str) -> str:
    """SHA-256 hex digest of the UTF-8 bytes of ``text``.

    Call on NORMALIZED text. Used for exact dedup (Phase 2), the split, and the
    Gate 2 train/val overlap check. Never use Python's ``hash()``: it is randomized
    per process (PYTHONHASHSEED), so splits would differ between runs.

    Args:
        text: Normalized document text.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_split(doc_hash: str, val_fraction: float, salt: str = SPLIT_SALT) -> Split:
    """Deterministically assign a document to ``"train"`` or ``"val"``.

    Hash ``salt`` + ``doc_hash`` again (e.g. SHA-256), read the first 8 bytes as an
    unsigned big-endian integer ``u``, and return ``"val"`` iff
    ``u / 2**64 < val_fraction``. Depends only on the arguments, so the same
    document always lands in the same split, and identical documents always share a
    split (they can never leak across it).

    Args:
        doc_hash: Output of :func:`document_hash`.
        val_fraction: Target fraction of documents in validation, in (0, 1).
        salt: Split namespace; keep the default.

    Returns:
        ``"train"`` or ``"val"``.

    Raises:
        ValueError: If ``val_fraction`` is not in (0, 1) or ``doc_hash`` is not a
            64-character hex string.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    if re.fullmatch(r"[0-9a-f]{64}", doc_hash) is None:
        raise ValueError(f"doc_hash must be 64 lowercase hex characters, got {doc_hash!r}.")
    digest = hashlib.sha256((salt + doc_hash).encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if position < val_fraction else "train"
