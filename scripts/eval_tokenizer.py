"""Gate 1 evaluation of the trained tokenizer on HELD-OUT text.

Corpora (both never seen by tokenizer training):
  * ``fineweb_edu_val``: FineWeb-Edu documents that the content-hash split assigns to
    "val" (same dataset, subset, revision and ``val_fraction`` as the training script).
  * ``tinystories_val``: the official TinyStories ``validation`` split.

For each corpus: exact round-trip of every document, bytes per token for Syrinx, and the
same text through the GPT-2 tokenizer for comparison. Prints one line of JSON and exits
1 if any document fails the round-trip (Gate 1 requires 100%).

Usage:
    uv run python scripts/eval_tokenizer.py
    uv run python scripts/eval_tokenizer.py --max-docs 200
"""

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from syrinx.data.documents import assign_split, document_hash, normalize_text
from syrinx.tokenizer import (
    EOS_TOKEN,
    VOCAB_SIZE,
    bytes_per_token,
    decode,
    encode_document,
    file_sha256,
    load_tokenizer,
)
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
DEFAULT_MAX_DOCS = 1000
"""Documents per corpus. ~1000 FineWeb-Edu val documents means streaming ~200K
documents (val is 0.5%), a few minutes."""

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSET = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_VAL_FRACTION = 0.005
"""These four must equal the training script's defaults (tested)."""

TINYSTORIES_DATASET = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
"""Dataset commit on the Hub (card checked 2026-10-11, docs/data.md)."""

REFERENCE_TOKENIZER = "openai-community/gpt2"
REFERENCE_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
"""GPT-2's tokenizer.json, pinned to a Hub commit (checked 2026-10-11)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Options and defaults: ``--tokenizer`` (DEFAULT_TOKENIZER, a ``Path``),
    ``--vocab-size`` (VOCAB_SIZE, int), ``--max-docs`` (DEFAULT_MAX_DOCS, int),
    ``--val-fraction`` (FINEWEB_VAL_FRACTION, float), ``--reference``
    (REFERENCE_TOKENIZER), ``--reference-revision`` (REFERENCE_REVISION).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``tokenizer``, ``vocab_size``, ``max_docs``, ``val_fraction``,
        ``reference`` and ``reference_revision``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--val-fraction", type=float, default=FINEWEB_VAL_FRACTION)
    parser.add_argument("--reference", default=REFERENCE_TOKENIZER)
    parser.add_argument("--reference-revision", default=REFERENCE_REVISION)
    return parser.parse_args(argv)


def stream_texts(
    dataset: str, subset: str | None, split: str, text_field: str, revision: str
) -> Iterator[str]:
    """Stream raw texts from one split of a Hugging Face dataset.

    Calls ``load_dataset(dataset, name=subset, split=split, streaming=True,
    revision=revision)`` (the module-level name, so tests can replace it) and yields
    ``row[text_field]``.

    Args:
        dataset: HF dataset id.
        subset: Config/subset name, or None.
        split: Split name ("train", "validation").
        text_field: Name of the text column.
        revision: Dataset commit to read.

    Yields:
        Raw document texts.

    Raises:
        KeyError: If a row has no ``text_field`` column (message names the column).
    """

    for row in load_dataset(dataset, name=subset, split=split, streaming=True, revision=revision):
        if text_field not in row:
            raise KeyError(f"Row has no {text_field!r} column; columns: {sorted(row)}")

        yield row[text_field]


def prepare_texts(
    raw_texts: Iterable[str], max_docs: int, val_fraction: float | None = None
) -> list[str]:
    """Normalize and collect up to ``max_docs`` evaluation documents.

    For each raw document: ``normalize_text``; skip it if empty; if ``val_fraction`` is
    given, skip it unless ``assign_split(document_hash(text), val_fraction) == "val"``
    (so FineWeb-Edu evaluation text is exactly the held-out split). Stop as soon as
    ``max_docs`` documents are collected, WITHOUT reading another raw document.

    Args:
        raw_texts: Raw documents.
        max_docs: Number of documents to return (fewer if the input runs out).
        val_fraction: If given, keep only val-split documents.

    Returns:
        Normalized documents, in input order.

    Raises:
        ValueError: If ``max_docs`` is not positive.
    """
    if max_docs <= 0:
        raise ValueError(f"max_docs must be positive, got {max_docs}.")
    texts: list[str] = []
    for raw in raw_texts:
        text = normalize_text(raw)
        if not text:
            continue
        if (
            val_fraction is not None
            and assign_split(document_hash(text), val_fraction=val_fraction) != "val"
        ):
            continue
        texts.append(text)
        if len(texts) >= max_docs:
            break
    return texts


def roundtrip_failures(tokenizer: Tokenizer, texts: list[str]) -> list[int]:
    """Indices of documents that do not round-trip exactly.

    A document passes iff BOTH hold:
      * ``decode(encode_document(text, append_eos=False)) == text``
      * ``decode(encode_document(text)) == text + EOS_TOKEN``

    Uses ``encode_document`` and ``decode`` by their module-level names (tests
    replace ``decode``).

    Args:
        tokenizer: Tokenizer under test.
        texts: Documents.

    Returns:
        Sorted indices of failing documents (empty list = 100% round-trip).
    """
    failures: list[int] = []
    for index, text in enumerate(texts):
        plain = decode(
            tokenizer=tokenizer,
            ids=encode_document(tokenizer=tokenizer, text=text, append_eos=False),
        )
        with_eos = decode(tokenizer=tokenizer, ids=encode_document(tokenizer=tokenizer, text=text))
        if plain != text or with_eos != text + EOS_TOKEN:
            failures.append(index)
    return failures


def evaluate_corpus(tokenizer: Tokenizer, reference: Tokenizer, texts: list[str]) -> dict:
    """Compression and round-trip numbers for one corpus.

    Token counts use ``encode_document(..., append_eos=False)`` for BOTH tokenizers, so
    special-token strings in the text are treated as plain text by both.

    Args:
        tokenizer: The Syrinx tokenizer.
        reference: The comparison tokenizer (GPT-2).
        texts: Evaluation documents.

    Returns:
        Dict with keys ``docs`` (int), ``bytes`` (int, UTF-8), ``tokens`` (int),
        ``reference_tokens`` (int), ``bytes_per_token`` (float),
        ``reference_bytes_per_token`` (float), ``relative_tokens`` (float,
        tokens / reference_tokens; below 1 means Syrinx needs fewer tokens) and
        ``roundtrip_failures`` (int, ``len(roundtrip_failures(tokenizer, texts))``,
        called by its module-level name).

    Raises:
        ValueError: If ``texts`` has no tokens (from :func:`bytes_per_token`).
    """
    n_bytes = sum(len(text.encode("utf-8")) for text in texts)
    tokens = sum(
        len(encode_document(tokenizer=tokenizer, text=text, append_eos=False)) for text in texts
    )
    reference_tokens = sum(
        len(encode_document(reference, text, append_eos=False)) for text in texts
    )

    return {
        "docs": len(texts),
        "bytes": n_bytes,
        "tokens": tokens,
        "reference_tokens": reference_tokens,
        "bytes_per_token": bytes_per_token(tokenizer=tokenizer, texts=texts),
        "reference_bytes_per_token": bytes_per_token(reference, texts),
        "relative_tokens": tokens / reference_tokens,
        "roundtrip_failures": len(roundtrip_failures(tokenizer=tokenizer, texts=texts)),
    }


def load_reference_tokenizer(name: str, revision: str) -> Tokenizer:
    """Load the comparison tokenizer from the Hub at a pinned revision.

    ``Tokenizer.from_pretrained(name, revision=revision)``. Kept separate so tests can
    replace it (no network in tests).

    Args:
        name: Hub model id.
        revision: Commit hash.

    Returns:
        The tokenizer.
    """
    return Tokenizer.from_pretrained(name, revision=revision)


def main(argv: list[str] | None = None) -> int:
    """Entry point: load both tokenizers, build both corpora, evaluate, print JSON.

    Steps: ``setup_logging()``; parse args; ``load_tokenizer(args.tokenizer,
    args.vocab_size)``; ``load_reference_tokenizer(args.reference,
    args.reference_revision)``; build the corpora with ``stream_texts`` ->
    ``prepare_texts``:
      * ``fineweb_edu_val``: ``stream_texts(FINEWEB_DATASET, FINEWEB_SUBSET, "train",
        "text", FINEWEB_REVISION)`` with ``val_fraction=args.val_fraction``
      * ``tinystories_val``: ``stream_texts(TINYSTORIES_DATASET, None, "validation",
        "text", TINYSTORIES_REVISION)`` with no val_fraction
    each limited to ``args.max_docs``; ``evaluate_corpus`` on each. Print ONE line of
    JSON (``sort_keys=True``) with keys ``tokenizer`` (str path), ``tokenizer_sha256``,
    ``reference``, ``reference_revision``, ``val_fraction``, ``max_docs`` and
    ``corpora`` ({corpus name: evaluate_corpus dict}). Call ``stream_texts``,
    ``load_reference_tokenizer`` and ``roundtrip_failures`` by their module-level
    names (tests replace them).

    Args:
        argv: Argument list.

    Returns:
        0 if every document in every corpus round-trips, else 1.
    """
    setup_logging()
    args = parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer, args.vocab_size)
    reference = load_reference_tokenizer(args.reference, args.reference_revision)
    corpora = {
        "fineweb_edu_val": prepare_texts(
            stream_texts(FINEWEB_DATASET, FINEWEB_SUBSET, "train", "text", FINEWEB_REVISION),
            args.max_docs,
            val_fraction=args.val_fraction,
        ),
        "tinystories_val": prepare_texts(
            stream_texts(TINYSTORIES_DATASET, None, "validation", "text", TINYSTORIES_REVISION),
            args.max_docs,
        ),
    }
    results = {
        name: evaluate_corpus(tokenizer, reference, texts) for name, texts in corpora.items()
    }
    summary = {
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "reference": args.reference,
        "reference_revision": args.reference_revision,
        "val_fraction": args.val_fraction,
        "max_docs": args.max_docs,
        "corpora": results,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if all(r["roundtrip_failures"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
