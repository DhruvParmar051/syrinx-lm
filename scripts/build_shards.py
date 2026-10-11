"""Build train/val token shards for one source (Phase 2).

Pipeline: stream -> ``iter_documents`` (normalize, dedup, min length, content-hash split)
-> batched ``encode_documents`` -> one ``ShardWriter`` per split. Writes
``<out-dir>/train/train_NNNN.bin``, ``<out-dir>/val/val_NNNN.bin`` (+ JSON sidecars) and
``<out-dir>/summary.json``, and prints the same summary as one JSON line.

Usage:
    uv run python scripts/build_shards.py --source tinystories
    uv run python scripts/build_shards.py --source fineweb-edu
    uv run python scripts/build_shards.py --source fineweb-edu --train-tokens 1e7 \\
        --out-dir data_cache/tokens/smoke
"""

import argparse
import itertools
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from syrinx.data.documents import SPLIT_SALT
from syrinx.data.selection import DEFAULT_MIN_CHARS, DocumentStats, iter_documents
from syrinx.data.shards import DEFAULT_SHARD_TOKENS, ShardWriter
from syrinx.tokenizer import VOCAB_SIZE, encode_documents, file_sha256, load_tokenizer
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
DEFAULT_OUT_ROOT = Path("data_cache/tokens")
DEFAULT_VAL_FRACTION = 0.005
DEFAULT_BATCH_DOCS = 1024
"""Documents per ``encode_documents`` call."""


@dataclass(frozen=True)
class SourceSpec:
    """A pinned Hub dataset and the splits streamed from it, in order."""

    dataset: str
    subset: str | None
    revision: str
    splits: tuple[str, ...]
    text_field: str
    default_train_tokens: int | None
    """Stop once train shards hold this many tokens; None = use every document."""


SOURCES: dict[str, SourceSpec] = {
    "fineweb-edu": SourceSpec(
        dataset="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        splits=("train",),
        text_field="text",
        default_train_tokens=2_500_000_000,
    ),
    "tinystories": SourceSpec(
        dataset="roneneldan/TinyStories",
        subset=None,
        revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        splits=("train", "validation"),
        text_field="text",
        default_train_tokens=None,
    ),
}
"""Revisions must match docs/data.md and the Phase 1 scripts (tested)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Options and defaults: ``--source`` (required, one of ``SOURCES``), ``--tokenizer``
    (DEFAULT_TOKENIZER, ``Path``), ``--vocab-size`` (VOCAB_SIZE, int), ``--out-dir``
    (``Path``; default None, meaning ``DEFAULT_OUT_ROOT / source``), ``--train-tokens``
    (default None, meaning the source's ``default_train_tokens``; accepts ``2.5e9``,
    stored as ``int``), ``--min-chars`` (DEFAULT_MIN_CHARS, int), ``--val-fraction``
    (DEFAULT_VAL_FRACTION, float), ``--shard-tokens`` (DEFAULT_SHARD_TOKENS; accepts
    ``1e8``, stored as ``int``), ``--batch-docs`` (DEFAULT_BATCH_DOCS, int).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with those fields; ``out_dir`` and ``train_tokens`` already resolved
        from the source when not given.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-tokens", type=lambda s: int(float(s)), default=None)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument(
        "--shard-tokens", type=lambda s: int(float(s)), default=DEFAULT_SHARD_TOKENS
    )
    parser.add_argument("--batch-docs", type=int, default=DEFAULT_BATCH_DOCS)
    args = parser.parse_args(argv)
    spec = SOURCES[args.source]
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_ROOT / args.source
    if args.train_tokens is None:
        args.train_tokens = spec.default_train_tokens
    return args


def stream_texts(
    dataset: str, subset: str | None, split: str, text_field: str, revision: str
) -> Iterator[str]:
    """Stream raw texts: ``load_dataset(dataset, name=subset, split=split,
    streaming=True, revision=revision)`` (module-level name, so tests can replace it),
    yielding ``row[text_field]``.

    Raises:
        KeyError: If a row has no ``text_field`` column (message names the column).
    """
    for row in load_dataset(dataset, name=subset, split=split, streaming=True, revision=revision):
        if text_field not in row:
            raise KeyError(f"Row has no {text_field!r} column; columns: {sorted(row)}")
        yield row[text_field]


def stream_source(spec: SourceSpec) -> Iterator[str]:
    """Raw texts of every split in ``spec.splits``, one after another, via the
    module-level ``stream_texts``."""
    for split in spec.splits:
        yield from stream_texts(
            dataset=spec.dataset,
            subset=spec.subset,
            split=split,
            text_field=spec.text_field,
            revision=spec.revision,
        )


def build(
    raw_texts: Iterable[str],
    tokenizer: Any,
    out_dir: Path,
    val_fraction: float,
    min_chars: int,
    train_tokens: int | None,
    shard_tokens: int,
    batch_docs: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Select, tokenize and write shards; return the summary.

    Open ``ShardWriter(out_dir / "train", "train", ...)`` and ``ShardWriter(out_dir /
    "val", "val", ...)`` (both with ``metadata``) BEFORE reading any document, so an
    existing output fails fast. Then take ``(split, text)`` pairs from
    ``iter_documents(raw_texts, val_fraction, min_chars, stats)`` in batches of
    ``batch_docs``; encode each batch with ``encode_documents``; add every document to
    its split's writer in input order. After each batch, stop if ``train_tokens`` is not
    None and the train writer holds ``>= train_tokens`` tokens (EOS included); stopping
    reads no further raw document. On any exception, both writers abort.

    Args:
        raw_texts: Raw documents.
        tokenizer: Trained tokenizer.
        out_dir: Output root.
        val_fraction: Split fraction.
        min_chars: Length filter (characters).
        train_tokens: Train token target, or None for all documents.
        shard_tokens: Target tokens per shard file.
        batch_docs: Documents per encode call; positive.
        metadata: Copied into every shard sidecar.

    Returns:
        ``{"stats": DocumentStats as dict, "splits": {"train": {...}, "val": {...}}}``
        where each split dict has ``docs``, ``tokens`` (EOS included) and ``shards``
        (file names, in order).

    Raises:
        ValueError: If ``batch_docs`` is not positive.
        FileExistsError: If shards already exist under ``out_dir``.
    """
    if batch_docs <= 0:
        raise ValueError(f"batch_docs must be positive, got {batch_docs}.")
    stats = DocumentStats()
    with (
        ShardWriter(out_dir / "train", "train", shard_tokens, metadata) as train_writer,
        ShardWriter(out_dir / "val", "val", shard_tokens, metadata) as val_writer,
    ):
        writers = {"train": train_writer, "val": val_writer}
        documents = iter_documents(raw_texts, val_fraction, min_chars, stats)
        while True:
            batch = list(itertools.islice(documents, batch_docs))
            if not batch:
                break
            ids = encode_documents(tokenizer, [text for _, text in batch])
            for (split, _), doc_ids in zip(batch, ids, strict=True):
                writers[split].add_document(doc_ids)
            if train_tokens is not None and train_writer.total_tokens >= train_tokens:
                break
    return {
        "stats": asdict(stats),
        "splits": {
            name: {
                "docs": writer.total_docs,
                "tokens": writer.total_tokens,
                "shards": [shard.path.name for shard in writer.shards],
            }
            for name, writer in writers.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Steps: ``setup_logging()``; parse args; ``load_tokenizer(args.tokenizer,
    args.vocab_size)``; ``metadata`` = ``tokenizer`` (str path), ``tokenizer_sha256``,
    ``source``, ``dataset``, ``subset``, ``revision``, ``val_fraction``,
    ``split_salt`` (SPLIT_SALT), ``min_chars``; ``build(stream_source(spec), ...)``;
    summary = metadata + ``train_tokens`` (the target, or None) + ``shard_tokens`` +
    the ``build`` result; write it to ``out_dir / "summary.json"`` (indent 2,
    sort_keys) and print it as ONE line (sort_keys). Call ``stream_texts`` by its
    module-level name (tests replace it).

    Args:
        argv: Argument list.

    Returns:
        0 on success.
    """
    setup_logging()
    args = parse_args(argv)
    spec = SOURCES[args.source]
    tokenizer = load_tokenizer(args.tokenizer, args.vocab_size)
    metadata = {
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "source": args.source,
        "dataset": spec.dataset,
        "subset": spec.subset,
        "revision": spec.revision,
        "val_fraction": args.val_fraction,
        "split_salt": SPLIT_SALT,
        "min_chars": args.min_chars,
    }
    result = build(
        stream_source(spec),
        tokenizer,
        args.out_dir,
        args.val_fraction,
        args.min_chars,
        args.train_tokens,
        args.shard_tokens,
        args.batch_docs,
        metadata,
    )
    summary = {
        **metadata,
        "train_tokens": args.train_tokens,
        "shard_tokens": args.shard_tokens,
        **result,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
