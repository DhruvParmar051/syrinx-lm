"""Check benchmark examples for 13-gram overlap with the training shards (Phase 8).

Benchmarks are read from local JSONL files (one HF row per line, exported at a pinned revision
before this runs), so the check itself never touches the network. Every train shard is
scanned in a process pool; the report lists, per benchmark, how many examples are
contaminated, unchecked (too short) or clean.

Usage:
    uv run python scripts/check_contamination.py \\
        --benchmark hellaswag=data_cache/benchmarks/hellaswag.jsonl \\
        --benchmark arc_easy=data_cache/benchmarks/arc_easy.jsonl \\
        --out experiments/contamination-2026-10-14.json
    # timing estimate first: --max-docs-per-shard 10000 on one shard directory
"""

import argparse
import json
import logging
import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from syrinx.contamination import N, build_index, example_text, init_worker, scan_shard
from syrinx.data.shards import list_shards
from syrinx.tokenizer import file_sha256
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
DEFAULT_SHARD_DIRS = (
    Path("data_cache/tokens/tinystories/train"),
    Path("data_cache/tokens/fineweb-edu/train"),
)
MAX_EXAMPLES_REPORTED = 5


def parse_benchmark(spec: str) -> tuple[str, Path]:
    """``"name=path"`` -> ``(name, Path(path))``.

    Raises:
        argparse.ArgumentTypeError: If there is no ``=`` or either side is empty.
    """
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options and defaults: ``--benchmark`` (repeatable ``NAME=PATH`` via
    :func:`parse_benchmark`, required), ``--shard-dir`` (repeatable ``Path``; none given =
    ``DEFAULT_SHARD_DIRS``), ``--tokenizer`` (DEFAULT_TOKENIZER), ``--n`` (N),
    ``--workers`` (``os.cpu_count()``), ``--max-docs-per-shard`` (int, None = all),
    ``--out`` (``Path``, None)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=parse_benchmark, action="append", required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max-docs-per-shard", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.shard_dir is None:
        args.shard_dir = list(DEFAULT_SHARD_DIRS)
    return args


def load_benchmark(name: str, path: Path) -> dict[str, str]:
    """``{example_id: example_text(name, row)}`` for every JSONL row of ``path``, in file
    order. ``example_id`` is ``f"{name}/{row['id']}"`` if the row has an ``id``, else
    ``f"{name}/{line_index}"`` (0-based), so ids are unique across benchmarks."""
    examples: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            example_id = f"{name}/{row['id']}" if "id" in row else f"{name}/{i}"
            examples[example_id] = example_text(name, row)
    return examples


def main(argv: list[str] | None = None) -> int:
    """``setup_logging()``; parse args; load every benchmark (:func:`load_benchmark`) and
    build ONE index over all of them (``build_index(..., args.n)``); list shards with
    ``list_shards(directory, "train")`` for each ``--shard-dir`` in order; scan them with a
    ``get_context("spawn").Pool(args.workers, init_worker, (index, args.tokenizer))``
    ``starmap``-ing ``scan_shard`` over ``(path, max_docs)`` (results in shard order); merge hits,
    keeping each example's first match in shard order.

    Record: ``{"n", "normalization" ("nfkc-lower-alnum-words"), "tokenizer_sha256",
    "max_docs_per_shard", "shards": [{"shard", "documents", "words", "seconds"}],
    "documents", "words", "seconds" (wall clock of the scan), "benchmarks": {name: {"path",
    "examples", "unchecked", "checked", "contaminated", "contaminated_pct" (100 *
    contaminated / checked, or None if checked == 0), "contaminated_ids" (sorted),
    "matches" (first MAX_EXAMPLES_REPORTED of the sorted ids: [{"id", "ngram"}])}}}``. If
    ``--out`` is given, write it as indented JSON (parents created; mode ``"x"``). Print it as
    ONE JSON line. Returns 0.

    Raises:
        FileNotFoundError: If a ``--shard-dir`` has no ``train_*.bin`` shards.
    """
    setup_logging()
    args = parse_args(argv)
    per_benchmark = {name: load_benchmark(name, path) for name, path in args.benchmark}
    all_examples = {k: v for examples in per_benchmark.values() for k, v in examples.items()}
    index = build_index(all_examples, args.n)
    shards: list[Path] = []
    for directory in args.shard_dir:
        found = list_shards(directory, "train")
        if not found:
            raise FileNotFoundError(f"no train_*.bin shards in {directory}")
        shards += found
    start = time.perf_counter()
    with get_context("spawn").Pool(args.workers, init_worker, (index, args.tokenizer)) as pool:
        scans = pool.starmap(scan_shard, [(path, args.max_docs_per_shard) for path in shards])
    seconds = time.perf_counter() - start
    hits: dict[str, str] = {}
    for scan in scans:
        for example_id, gram in scan.hits.items():
            hits.setdefault(example_id, gram)
    unchecked = set(index.unchecked)
    benchmarks: dict[str, Any] = {}
    for (name, path), examples in zip(args.benchmark, per_benchmark.values(), strict=True):
        ids = sorted(i for i in examples if i in hits)
        n_unchecked = sum(1 for i in examples if i in unchecked)
        checked = len(examples) - n_unchecked
        benchmarks[name] = {
            "path": str(path),
            "examples": len(examples),
            "unchecked": n_unchecked,
            "checked": checked,
            "contaminated": len(ids),
            "contaminated_pct": 100.0 * len(ids) / checked if checked else None,
            "contaminated_ids": ids,
            "matches": [{"id": i, "ngram": hits[i]} for i in ids[:MAX_EXAMPLES_REPORTED]],
        }
    record: dict[str, Any] = {
        "n": args.n,
        "normalization": "nfkc-lower-alnum-words",
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "max_docs_per_shard": args.max_docs_per_shard,
        "shards": [
            {"shard": s.shard, "documents": s.documents, "words": s.words, "seconds": s.seconds}
            for s in scans
        ],
        "documents": sum(s.documents for s in scans),
        "words": sum(s.words for s in scans),
        "seconds": seconds,
        "benchmarks": benchmarks,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
