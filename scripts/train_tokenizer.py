"""Train the Syrinx 32K byte-level BPE tokenizer on TRAIN-split documents only.

Pipeline per document: stream -> normalize_text -> skip empty -> skip exact duplicate
(document_hash) -> assign_split; keep only "train" documents until ``--max-bytes`` of
UTF-8 text is collected, then train, validate, save, and print a one-line JSON summary
(SHA-256, settings, document counts) to paste into experiments/log.md.

Usage:
    uv run python scripts/train_tokenizer.py
    uv run python scripts/train_tokenizer.py --max-bytes 1e8 --out tokenizer/artifacts/t.json
"""

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset

from syrinx.data.documents import SPLIT_SALT, assign_split, document_hash, normalize_text
from syrinx.tokenizer import VOCAB_SIZE, save_tokenizer, train_tokenizer
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_SUBSET = "sample-10BT"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_VAL_FRACTION = 0.005
"""Must equal the value Phase 2 uses (docs/decisions.md)."""
DEFAULT_MAX_BYTES = 1_500_000_000
DEFAULT_OUT = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")


@dataclass
class SelectionStats:
    """Counts from :func:`select_train_texts`, for experiments/log.md.

    Every document read is counted in exactly one of ``empty``, ``duplicates``,
    ``val`` or ``train_docs``, so ``seen`` equals their sum.
    """

    seen: int = 0
    empty: int = 0
    duplicates: int = 0
    val: int = 0
    train_docs: int = 0
    train_bytes: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Options and defaults: ``--dataset`` (DEFAULT_DATASET), ``--subset``
    (DEFAULT_SUBSET), ``--text-field`` (DEFAULT_TEXT_FIELD), ``--val-fraction``
    (DEFAULT_VAL_FRACTION, float), ``--max-bytes`` (DEFAULT_MAX_BYTES; accepts
    scientific notation like ``1.5e9`` and is stored as an ``int``), ``--vocab-size``
    (VOCAB_SIZE, int), ``--min-frequency`` (2, int), ``--out`` (DEFAULT_OUT, a
    ``Path``).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``dataset``, ``subset``, ``text_field``, ``val_fraction``,
        ``max_bytes``, ``vocab_size``, ``min_frequency`` and ``out``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--max-bytes", type=lambda s: int(float(s)), default=DEFAULT_MAX_BYTES)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def stream_texts(dataset: str, subset: str | None, text_field: str) -> Iterator[str]:
    """Stream raw document texts from a Hugging Face dataset.

    Calls ``load_dataset(dataset, name=subset, split="train", streaming=True)`` (the
    module-level name, so tests can replace it) and yields ``row[text_field]`` for
    each row. Streaming downloads lazily: stopping the iteration stops the download.
    Check the dataset's license on its card and record it in docs/data.md first.

    Args:
        dataset: HF dataset id.
        subset: Config/subset name, or None.
        text_field: Name of the text column.

    Yields:
        Raw document texts.

    Raises:
        KeyError: If a row has no ``text_field`` column (message names the column).
    """
    for row in load_dataset(dataset, name=subset, split="train", streaming=True):
        if text_field not in row:
            raise KeyError(f"Row has no {text_field!r} column; columns: {sorted(row)}")
        yield row[text_field]


def select_train_texts(
    raw_texts: Iterable[str],
    val_fraction: float,
    max_bytes: int,
    stats: SelectionStats | None = None,
) -> Iterator[str]:
    """Normalize, dedup, split, and yield TRAIN documents until ``max_bytes`` is reached.

    For each raw document, in order:
      1. ``normalize_text``; if the result is empty, count it in ``empty`` and skip.
      2. ``document_hash``; if this hash was already seen, count ``duplicates``, skip.
      3. ``assign_split(hash, val_fraction)``; if "val", count ``val`` and skip.
      4. Count ``train_docs`` and ``train_bytes`` (UTF-8 bytes), then yield the
         normalized text.
      5. Stop as soon as ``train_bytes >= max_bytes``, WITHOUT reading another raw
         document (the input is a network stream).
    Stats are updated before each yield, so they are correct even if the consumer
    stops early.

    Args:
        raw_texts: Raw documents.
        val_fraction: Must be the same value Phase 2 uses.
        max_bytes: Stop once this many UTF-8 bytes of train text were yielded.
        stats: Updated in place if given.

    Yields:
        Normalized train-split texts.

    Raises:
        ValueError: If ``max_bytes`` is not positive (raised on first iteration),
            or ``val_fraction`` is not in (0, 1) (from ``assign_split``).
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}.")

    if stats is None:
        stats = SelectionStats()
    seen_hashes: set[str] = set()

    for raw in raw_texts:
        stats.seen += 1
        text = normalize_text(raw)
        if not text:
            stats.empty += 1
            continue
        doc_hash = document_hash(text)
        if doc_hash in seen_hashes:
            stats.duplicates += 1
            continue
        seen_hashes.add(doc_hash)
        if assign_split(doc_hash=doc_hash, val_fraction=val_fraction) == "val":
            stats.val += 1
            continue

        stats.train_docs += 1
        stats.train_bytes += len(text.encode("utf-8"))
        yield text
        if stats.train_bytes >= max_bytes:
            return


def main(argv: list[str] | None = None) -> int:
    """Entry point: select train texts, train, save, print a JSON summary.

    Steps: ``setup_logging()``; parse args; if ``out`` already exists raise
    ``FileExistsError`` BEFORE streaming anything; run ``stream_texts`` ->
    ``select_train_texts`` (with a :class:`SelectionStats`) -> ``train_tokenizer``
    -> ``save_tokenizer``; print ONE line of JSON (``json.dumps(..., sort_keys=True)``)
    to stdout with keys: ``out`` (str), ``sha256``, ``dataset``, ``subset``,
    ``text_field``, ``val_fraction``, ``max_bytes``, ``vocab_size``,
    ``min_frequency``, ``split_salt`` (SPLIT_SALT) and ``stats`` (the
    SelectionStats as a dict). Call ``stream_texts`` and ``train_tokenizer`` by their
    module-level names (tests replace them).

    Args:
        argv: Argument list.

    Returns:
        Process exit code (0 on success).

    Raises:
        FileExistsError: If the output file already exists.
    """
    setup_logging()
    args = parse_args(argv=argv)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}.")
    stats = SelectionStats()
    texts = select_train_texts(
        stream_texts(args.dataset, args.subset, args.text_field),
        args.val_fraction,
        args.max_bytes,
        stats=stats,
    )
    tokenizer = train_tokenizer(
        texts=texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    sha256 = save_tokenizer(tokenizer=tokenizer, path=args.out, vocab_size=args.vocab_size)
    logger.info("Selection stats: %s", stats)
    summary = {
        "out": str(args.out),
        "sha256": sha256,
        "dataset": args.dataset,
        "subset": args.subset,
        "text_field": args.text_field,
        "val_fraction": args.val_fraction,
        "max_bytes": args.max_bytes,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "split_salt": SPLIT_SALT,
        "stats": asdict(stats),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
