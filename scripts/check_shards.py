"""Gate 2 checks on built shards, read back from disk.

For a directory written by ``scripts/build_shards.py`` (``train/``, ``val/``,
``summary.json``):
  1. Every shard is well-formed: size matches its sidecar, ends with EOS, EOS count
     equals ``num_docs``, every id is below the vocabulary size.
  2. Zero train/val overlap: no document appears in both splits (compared by a hash of
     its token ids; the tokenizer is lossless, so equal ids <=> equal text).
  3. Counts match ``summary.json`` (docs, tokens, shard names per split), and every
     sidecar has the same provenance as the summary and the current tokenizer file.
  4. Decoding reproduces the source: re-stream the source, run the same selection, and
     the first documents of each split decode to exactly that text.
Writes ``<data-dir>/check.json``, prints it as one JSON line, exits 0 only if every
check passes.

Usage:
    uv run python scripts/check_shards.py --data-dir data_cache/tokens/tinystories
    uv run python scripts/check_shards.py --data-dir data_cache/tokens/fineweb-edu
"""

import argparse
import hashlib
import itertools
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from syrinx.data.selection import iter_documents
from syrinx.data.shards import (
    EOS_ID,
    decode_documents,
    list_shards,
    read_metadata,
    read_shard,
    split_documents,
)
from syrinx.tokenizer import VOCAB_SIZE, file_sha256, load_tokenizer
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
DEFAULT_VERIFY_DOCS = 20_000
"""Source documents re-streamed for the decode check (~100 MB of FineWeb-Edu)."""

SOURCE_SPLITS: dict[str, tuple[str, ...]] = {
    "fineweb-edu": ("train",),
    "tinystories": ("train", "validation"),
}
"""Hub splits streamed per source; must equal ``build_shards.SOURCES`` (tested)."""

PROVENANCE_KEYS = (
    "tokenizer_sha256",
    "dataset",
    "subset",
    "revision",
    "split_salt",
    "val_fraction",
    "min_chars",
)
"""Sidecar fields that must equal the summary's."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Options: ``--data-dir`` (required, ``Path``), ``--tokenizer`` (DEFAULT_TOKENIZER,
    ``Path``), ``--vocab-size`` (VOCAB_SIZE, int), ``--verify-docs``
    (DEFAULT_VERIFY_DOCS, int; 0 skips the source comparison).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``data_dir``, ``tokenizer``, ``vocab_size``, ``verify_docs``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--verify-docs", type=int, default=DEFAULT_VERIFY_DOCS)

    return parser.parse_args(argv)


def document_token_hashes(tokens: np.ndarray) -> list[bytes]:
    """SHA-256 digest of each document's token ids (EOS excluded).

    Each document from ``split_documents(tokens)`` is hashed as its ids in little-endian
    ``uint16`` bytes (``doc.astype("<u2").tobytes()``).

    Args:
        tokens: A shard's token stream.

    Returns:
        One 32-byte digest per document, in order.
    """
    return [
        hashlib.sha256(np.asarray(doc).astype("<u2").tobytes()).digest()
        for doc in split_documents(tokens)
    ]


def check_split(
    directory: Path, prefix: str, vocab_size: int
) -> tuple[dict[str, Any], set[bytes], list[str]]:
    """Validate every shard of one split and collect its document hashes.

    For each shard in ``list_shards(directory, prefix)``: open it with ``read_shard``
    (a ``ValueError`` or ``FileNotFoundError`` becomes a problem and the shard is
    skipped); then record a problem if it does not end with ``EOS_ID``, if its EOS count
    differs from the sidecar's ``num_docs``, or if any id is ``>= vocab_size``. If the
    shard ends with EOS, hash its documents with :func:`document_token_hashes` (a shard
    that doesn't cannot be split into documents, so it is not hashed); a hash already
    seen in this split counts as a duplicate. No shards at all is a problem.

    Args:
        directory: Split directory.
        prefix: Shard prefix ("train" / "val").
        vocab_size: Exclusive upper bound for ids.

    Returns:
        ``(report, hashes, problems)``: report has ``shards`` (list of file names),
        ``docs``, ``tokens`` and ``duplicate_docs`` over the readable shards; hashes is
        the set of document digests; problems is a list of human-readable strings,
        each naming the shard.
    """
    report: dict[str, Any] = {"shards": [], "docs": 0, "tokens": 0, "duplicate_docs": 0}
    hashes: set[bytes] = set()
    problems: list[str] = []
    paths = list_shards(directory, prefix)
    if not paths:
        problems.append(f"no {prefix} shards in {directory}")
    for path in paths:
        try:
            tokens = np.asarray(read_shard(path))
            meta = read_metadata(path)
        except (ValueError, FileNotFoundError) as exc:
            problems.append(f"{path.name}: unreadable ({exc})")
            continue
        report["shards"].append(path.name)
        report["tokens"] += len(tokens)
        eos_count = int((tokens == EOS_ID).sum())
        report["docs"] += eos_count
        ends_with_eos = len(tokens) > 0 and int(tokens[-1]) == EOS_ID
        if not ends_with_eos:
            problems.append(f"{path.name}: does not end with EOS")
        if eos_count != meta.get("num_docs"):
            problems.append(
                f"{path.name}: {eos_count} EOS tokens but sidecar num_docs={meta.get('num_docs')}"
            )
        if len(tokens) and int(tokens.max()) >= vocab_size:
            problems.append(f"{path.name}: id {int(tokens.max())} >= vocab_size {vocab_size}")
        if ends_with_eos:
            for digest in document_token_hashes(tokens):
                if digest in hashes:
                    report["duplicate_docs"] += 1
                hashes.add(digest)
    return report, hashes, problems


def check_summary(summary: dict[str, Any], reports: dict[str, dict[str, Any]]) -> list[str]:
    """Compare per-split ``docs``, ``tokens`` and ``shards`` with ``summary["splits"]``.

    Args:
        summary: Parsed ``summary.json``.
        reports: ``{"train": report, "val": report}`` from :func:`check_split`.

    Returns:
        One problem string per mismatching field (naming split, field, both values).
    """
    problems: list[str] = []
    for split, report in reports.items():
        expected = summary["splits"][split]
        for field in ("docs", "tokens", "shards"):
            if report[field] != expected[field]:
                problems.append(
                    f"{split} {field}: on disk {report[field]!r}, summary {expected[field]!r}"
                )
    return problems


def check_provenance(data_dir: Path, summary: dict[str, Any], tokenizer_sha256: str) -> list[str]:
    """Every sidecar must agree with the summary on ``PROVENANCE_KEYS``, and the
    summary's ``tokenizer_sha256`` must equal the current tokenizer file's.

    Args:
        data_dir: Build directory (with ``train/`` and ``val/``).
        summary: Parsed ``summary.json``.
        tokenizer_sha256: SHA-256 of the tokenizer file being used for the checks.

    Returns:
        Problem strings (naming the shard and key).
    """
    problems: list[str] = []
    if summary.get("tokenizer_sha256") != tokenizer_sha256:
        problems.append(
            f"tokenizer_sha256: summary {summary.get('tokenizer_sha256')} "
            f"!= tokenizer file {tokenizer_sha256}"
        )
    for split in ("train", "val"):
        for path in list_shards(data_dir / split, split):
            try:
                meta = read_metadata(path)
            except FileNotFoundError:
                continue  # already reported by check_split
            for key in PROVENANCE_KEYS:
                if meta.get(key) != summary.get(key):
                    problems.append(
                        f"{path.name}: {key} {meta.get(key)!r} != summary {summary.get(key)!r}"
                    )
    return problems


def check_against_source(
    raw_texts: Iterable[str],
    tokenizer: Tokenizer,
    data_dir: Path,
    summary: dict[str, Any],
    verify_docs: int,
) -> tuple[dict[str, int], list[str]]:
    """Decoding reproduces the source text.

    Take the first ``verify_docs`` items of ``iter_documents(raw_texts,
    summary["val_fraction"], summary["min_chars"])``. Within each split, the shards'
    first documents (in file order, decoded with ``decode_documents``) must equal that
    split's expected texts, in order. Decode only as many shards as needed.

    Args:
        raw_texts: The source, re-streamed.
        tokenizer: The tokenizer the shards were built with.
        data_dir: Build directory.
        summary: Parsed ``summary.json``.
        verify_docs: Number of selected documents to compare; must be positive.

    Returns:
        ``(verified, problems)``: verified is ``{"train": n, "val": m}``, the number of
        documents compared per split; one problem per split whose documents differ
        (the message contains the split name, the word "differs" and the first
        differing index) or whose shards hold fewer documents than expected.

    Raises:
        ValueError: If ``verify_docs`` is not positive.
    """
    if verify_docs <= 0:
        raise ValueError(f"verify_docs must be positive, got {verify_docs}.")
    selected = iter_documents(raw_texts, summary["val_fraction"], summary["min_chars"])
    expected: dict[str, list[str]] = {"train": [], "val": []}
    for split, text in itertools.islice(selected, verify_docs):
        expected[split].append(text)

    verified: dict[str, int] = {}
    problems: list[str] = []
    for split, texts in expected.items():
        decoded: list[str] = []
        for path in list_shards(data_dir / split, split):
            if len(decoded) >= len(texts):
                break
            decoded.extend(decode_documents(tokenizer, read_shard(path)))
        decoded = decoded[: len(texts)]
        verified[split] = len(decoded)
        if len(decoded) < len(texts):
            problems.append(
                f"{split}: shards hold {len(decoded)} documents, "
                f"source selects at least {len(texts)}"
            )
            continue
        for index, (got, want) in enumerate(zip(decoded, texts, strict=True)):
            if got != want:
                problems.append(f"{split}: document {index} differs from the source")
                break
    return verified, problems


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


def stream_source(summary: dict[str, Any]) -> Iterator[str]:
    """Re-stream the source a build came from: for each split in
    ``SOURCE_SPLITS[summary["source"]]``, ``stream_texts(summary["dataset"],
    summary["subset"], split, "text", summary["revision"])`` via the module-level name.

    Raises:
        KeyError: If the summary's source is unknown.
    """
    for split in SOURCE_SPLITS[summary["source"]]:
        yield from stream_texts(
            summary["dataset"], summary["subset"], split, "text", summary["revision"]
        )


def main(argv: list[str] | None = None) -> int:
    """Run every check and report.

    Steps: ``setup_logging()``; parse args; load ``summary.json`` from ``data_dir``;
    ``load_tokenizer(args.tokenizer, args.vocab_size)``; :func:`check_split` for
    "train" and "val" (directories ``data_dir / split``); the overlap is the number of
    hashes in both splits (any overlap is a problem); :func:`check_summary`;
    :func:`check_provenance` with ``file_sha256(args.tokenizer)``; if
    ``args.verify_docs > 0``, :func:`check_against_source` on ``stream_source(summary)``
    (else ``verified`` is ``{"train": 0, "val": 0}``).

    Report keys: ``data_dir`` (str), ``tokenizer_sha256``, ``splits`` ({split: report}),
    ``overlap_docs``, ``verified_docs``, ``problems`` (all problems, in check order)
    and ``passed`` (no problems). Write it to ``data_dir / "check.json"`` (indent 2,
    sort_keys; overwriting is fine) and print it as ONE line (sort_keys).

    Args:
        argv: Argument list.

    Returns:
        0 if ``passed``, else 1.
    """
    setup_logging()
    args = parse_args(argv)
    summary = json.loads((args.data_dir / "summary.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.tokenizer, args.vocab_size)

    problems: list[str] = []
    reports: dict[str, dict[str, Any]] = {}
    hashes: dict[str, set[bytes]] = {}
    for split in ("train", "val"):
        reports[split], hashes[split], split_problems = check_split(
            args.data_dir / split, split, args.vocab_size
        )
        problems.extend(split_problems)

    overlap = len(hashes["train"] & hashes["val"])
    if overlap:
        problems.append(f"train/val overlap: {overlap} documents appear in both splits")
    for split, report in reports.items():
        if report["duplicate_docs"]:
            problems.append(f"{split}: {report['duplicate_docs']} duplicate documents")

    problems.extend(check_summary(summary, reports))
    tokenizer_sha256 = file_sha256(args.tokenizer)
    problems.extend(check_provenance(args.data_dir, summary, tokenizer_sha256))

    if args.verify_docs > 0:
        verified, source_problems = check_against_source(
            stream_source(summary), tokenizer, args.data_dir, summary, args.verify_docs
        )
        problems.extend(source_problems)
    else:
        verified = {"train": 0, "val": 0}

    report = {
        "data_dir": str(args.data_dir),
        "tokenizer_sha256": tokenizer_sha256,
        "splits": reports,
        "overlap_docs": overlap,
        "verified_docs": verified,
        "problems": problems,
        "passed": not problems,
    }
    (args.data_dir / "check.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
