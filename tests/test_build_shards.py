"""Tests for ``scripts/build_shards.py``. No network: streams are in-memory fakes."""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer

from syrinx.data.documents import SPLIT_SALT, document_hash
from syrinx.data.selection import iter_documents
from syrinx.data.shards import decode_documents, list_shards, read_metadata, read_shard
from syrinx.tokenizer import VOCAB_SIZE, file_sha256, save_tokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]

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
]
DOCS: list[str] = [
    " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(3 + i % 9)) + f" {i}." for i in range(300)
]
"""300 distinct documents, 16 to ~70 characters."""

SMALL_VOCAB = 280
MIN_CHARS = 30


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load("build_shards")


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_tokenizer(DOCS, vocab_size=SMALL_VOCAB, min_frequency=1)


def _build(
    script: ModuleType,
    tok: Tokenizer,
    out: Path,
    raw: Any = None,
    train_tokens: int | None = None,
    shard_tokens: int = 300,
    batch_docs: int = 16,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    return script.build(
        DOCS if raw is None else raw,
        tok,
        out,
        val_fraction,
        MIN_CHARS,
        train_tokens,
        shard_tokens,
        batch_docs,
        {"source": "test"},
    )


def _decoded(tok: Tokenizer, directory: Path, prefix: str) -> list[str]:
    return [d for p in list_shards(directory, prefix) for d in decode_documents(tok, read_shard(p))]


# --- settings -------------------------------------------------------------------------


def test_sources_are_pinned_and_consistent(script: ModuleType) -> None:
    """Revisions match the Phase 1 scripts and docs/data.md; FineWeb-Edu stops at 2.5B
    train tokens, TinyStories uses everything from both official splits."""
    train_tok = _load("train_tokenizer")
    eval_tok = _load("eval_tokenizer")
    data_md = (ROOT / "docs" / "data.md").read_text(encoding="utf-8")

    fw = script.SOURCES["fineweb-edu"]
    assert (fw.dataset, fw.subset) == (train_tok.DEFAULT_DATASET, train_tok.DEFAULT_SUBSET)
    assert fw.revision == train_tok.DEFAULT_REVISION
    assert fw.splits == ("train",)
    assert fw.default_train_tokens == 2_500_000_000

    ts = script.SOURCES["tinystories"]
    assert (ts.dataset, ts.subset) == ("roneneldan/TinyStories", None)
    assert ts.revision == eval_tok.TINYSTORIES_REVISION
    assert ts.splits == ("train", "validation")
    assert ts.default_train_tokens is None

    assert fw.revision in data_md and ts.revision in data_md
    assert script.DEFAULT_VAL_FRACTION == train_tok.DEFAULT_VAL_FRACTION


def test_parse_args_defaults_resolve_from_source(script: ModuleType) -> None:
    """--out-dir and --train-tokens default from the source."""
    args = script.parse_args(["--source", "fineweb-edu"])
    assert args.out_dir == Path("data_cache/tokens/fineweb-edu")
    assert args.train_tokens == 2_500_000_000
    assert args.tokenizer == Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
    assert args.vocab_size == VOCAB_SIZE
    assert args.min_chars == 200
    assert args.val_fraction == 0.005
    assert args.shard_tokens == 100_000_000
    assert args.batch_docs == 1024

    args = script.parse_args(["--source", "tinystories"])
    assert args.out_dir == Path("data_cache/tokens/tinystories")
    assert args.train_tokens is None


def test_parse_args_overrides(script: ModuleType) -> None:
    """Scientific notation is accepted for token counts and stored as int."""
    args = script.parse_args(
        [
            "--source",
            "tinystories",
            "--out-dir",
            "x/y",
            "--train-tokens",
            "1e7",
            "--shard-tokens",
            "5e6",
            "--min-chars",
            "0",
            "--batch-docs",
            "8",
        ]
    )
    assert args.out_dir == Path("x/y")
    assert args.train_tokens == 10_000_000 and isinstance(args.train_tokens, int)
    assert args.shard_tokens == 5_000_000 and isinstance(args.shard_tokens, int)
    assert (args.min_chars, args.batch_docs) == (0, 8)


def test_parse_args_requires_known_source(script: ModuleType) -> None:
    """--source is required and must be a known source."""
    with pytest.raises(SystemExit):
        script.parse_args([])
    with pytest.raises(SystemExit):
        script.parse_args(["--source", "wikipedia"])


# --- streaming ------------------------------------------------------------------------


def test_stream_texts_reads_cached_hub_files(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delegates to ``syrinx.data.sources.iter_hub_texts`` (download each parquet file
    once, read it locally); the remote streaming path is gone."""
    calls: list[tuple[Any, ...]] = []

    def fake(
        dataset: str, subset: str | None, split: str, text_field: str, revision: str
    ) -> Iterator[str]:
        calls.append((dataset, subset, split, text_field, revision))
        return iter(["a", "b"])

    monkeypatch.setattr(script, "iter_hub_texts", fake)
    assert list(script.stream_texts("d", "s", "train", "text", "r")) == ["a", "b"]
    assert calls == [("d", "s", "train", "text", "r")]
    assert not hasattr(script, "load_dataset")


def test_stream_source_chains_splits(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """All splits of the source, in order, each at the source's revision."""
    calls: list[tuple[Any, ...]] = []

    def fake(
        dataset: str, subset: str | None, split: str, text_field: str, revision: str
    ) -> Iterator[str]:
        calls.append((dataset, subset, split, text_field, revision))
        return iter([f"{split}-1", f"{split}-2"])

    monkeypatch.setattr(script, "stream_texts", fake)
    spec = script.SOURCES["tinystories"]
    texts = list(script.stream_source(spec))
    assert texts == ["train-1", "train-2", "validation-1", "validation-2"]
    assert calls == [
        (spec.dataset, None, "train", "text", spec.revision),
        (spec.dataset, None, "validation", "text", spec.revision),
    ]


# --- build ----------------------------------------------------------------------------


def test_build_roundtrip_matches_selection(
    script: ModuleType, tok: Tokenizer, tmp_path: Path
) -> None:
    """Gate 2: decoding each split's shards gives exactly that split's documents from
    iter_documents (same order), across several shard files."""
    summary = _build(script, tok, tmp_path)
    expected = list(iter_documents(DOCS, 0.2, MIN_CHARS))
    for split in ("train", "val"):
        assert _decoded(tok, tmp_path / split, split) == [t for s, t in expected if s == split]
    assert len(summary["splits"]["train"]["shards"]) > 1


def test_build_train_val_disjoint(script: ModuleType, tok: Tokenizer, tmp_path: Path) -> None:
    """Gate 2: no document hash appears in both splits, even with repeats and newline
    variants in the input."""
    raw = [*DOCS, *DOCS[:40], *(d.replace(" ", "\r\n", 1) for d in DOCS[:40])]
    _build(script, tok, tmp_path, raw=raw)
    train = {document_hash(t) for t in _decoded(tok, tmp_path / "train", "train")}
    val = {document_hash(t) for t in _decoded(tok, tmp_path / "val", "val")}
    assert train and val and train.isdisjoint(val)


def test_build_summary_counts_match_files(
    script: ModuleType, tok: Tokenizer, tmp_path: Path
) -> None:
    """Gate 2: per-split docs/tokens in the summary equal what is on disk; stats come
    from iter_documents and add up."""
    summary = _build(script, tok, tmp_path, raw=[*DOCS, DOCS[0], "", "short"])
    for split in ("train", "val"):
        paths = list_shards(tmp_path / split, split)
        info = summary["splits"][split]
        assert info["shards"] == [p.name for p in paths]
        assert info["tokens"] == sum(len(read_shard(p)) for p in paths)
        assert info["docs"] == sum(int((np.asarray(read_shard(p)) == 0).sum()) for p in paths)
    stats = summary["stats"]
    assert (stats["duplicates"], stats["empty"]) == (1, 1)
    assert stats["too_short"] >= 1
    assert stats["train_docs"] == summary["splits"]["train"]["docs"]
    assert stats["val_docs"] == summary["splits"]["val"]["docs"]
    json.dumps(summary)


def test_build_sidecars_carry_metadata(script: ModuleType, tok: Tokenizer, tmp_path: Path) -> None:
    """Every shard sidecar includes the metadata passed to build."""
    _build(script, tok, tmp_path)
    for split in ("train", "val"):
        for p in list_shards(tmp_path / split, split):
            assert read_metadata(p)["source"] == "test"


def test_build_stops_at_train_tokens(script: ModuleType, tok: Tokenizer, tmp_path: Path) -> None:
    """Stops after the first batch that brings train tokens to >= the target, and reads
    no further raw document."""
    pulled: list[int] = []

    def stream() -> Iterator[str]:
        for i, doc in enumerate(DOCS):
            pulled.append(i)
            yield doc

    summary = _build(script, tok, tmp_path, raw=stream(), train_tokens=500, batch_docs=10)
    train_tokens = summary["splits"]["train"]["tokens"]
    assert train_tokens >= 500
    assert len(pulled) < len(DOCS)

    # The same input without a limit, cut at the same point, matches exactly.
    items = list(iter_documents(DOCS[: len(pulled)], 0.2, MIN_CHARS))
    assert len(items) % 10 == 0  # stopped on a batch boundary
    train_texts = [t for s, t in items if s == "train"]
    assert _decoded(tok, tmp_path / "train", "train") == train_texts
    # ...and the batch before the last one had not yet reached the target
    last_batch_train = [t for s, t in items[-10:] if s == "train"]
    last_batch_tokens = sum(len(tok.encode(t).ids) + 1 for t in last_batch_train)
    assert train_tokens - last_batch_tokens < 500


def test_build_without_limit_uses_everything(
    script: ModuleType, tok: Tokenizer, tmp_path: Path
) -> None:
    """train_tokens=None reads the whole input."""
    summary = _build(script, tok, tmp_path)
    stats = summary["stats"]
    assert stats["seen"] == len(DOCS)


def test_build_refuses_existing_output_before_reading(
    script: ModuleType, tok: Tokenizer, tmp_path: Path
) -> None:
    """Existing shards make build fail before any raw document is read."""
    _build(script, tok, tmp_path)

    def must_not_read() -> Iterator[str]:
        raise AssertionError("read input although the output exists")
        yield ""  # pragma: no cover

    with pytest.raises(FileExistsError):
        _build(script, tok, tmp_path, raw=must_not_read())


def test_build_aborts_cleanly_on_error(script: ModuleType, tok: Tokenizer, tmp_path: Path) -> None:
    """If reading fails midway, the error propagates and no temporary files remain."""

    def failing() -> Iterator[str]:
        yield from DOCS[:50]
        raise RuntimeError("network died")

    with pytest.raises(RuntimeError):
        _build(script, tok, tmp_path, raw=failing(), shard_tokens=10**9)
    leftovers = [p.name for p in tmp_path.rglob("*") if p.is_file()]
    assert not any(name.endswith(".tmp") for name in leftovers)
    assert list_shards(tmp_path / "train", "train") == []


@pytest.mark.parametrize("batch_docs", [0, -1])
def test_build_rejects_bad_batch_docs(
    script: ModuleType, tok: Tokenizer, tmp_path: Path, batch_docs: int
) -> None:
    """batch_docs must be positive."""
    with pytest.raises(ValueError):
        _build(script, tok, tmp_path, batch_docs=batch_docs)


# --- main -----------------------------------------------------------------------------


def test_main_end_to_end(
    script: ModuleType,
    tok: Tokenizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Builds both splits from the source's streams, writes summary.json and prints the
    same summary as one JSON line, with provenance."""
    tok_path = tmp_path / "tok.json"
    save_tokenizer(tok, tok_path, vocab_size=SMALL_VOCAB)
    streams: list[tuple[Any, ...]] = []

    def fake(
        dataset: str, subset: str | None, split: str, text_field: str, revision: str
    ) -> Iterator[str]:
        streams.append((dataset, subset, split, text_field, revision))
        return iter(DOCS[:150] if split == "train" else DOCS[150:])

    monkeypatch.setattr(script, "stream_texts", fake)
    out = tmp_path / "out"
    argv = [
        "--source",
        "tinystories",
        "--tokenizer",
        str(tok_path),
        "--vocab-size",
        str(SMALL_VOCAB),
        "--out-dir",
        str(out),
        "--min-chars",
        str(MIN_CHARS),
        "--val-fraction",
        "0.2",
        "--shard-tokens",
        "300",
    ]
    assert script.main(argv) == 0

    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    on_disk = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert printed == on_disk

    spec = script.SOURCES["tinystories"]
    assert [s[2] for s in streams] == ["train", "validation"]
    assert printed["tokenizer_sha256"] == file_sha256(tok_path)
    assert printed["tokenizer"] == str(tok_path)
    assert printed["source"] == "tinystories"
    assert (printed["dataset"], printed["subset"]) == (spec.dataset, None)
    assert printed["revision"] == spec.revision
    assert printed["split_salt"] == SPLIT_SALT
    assert (printed["val_fraction"], printed["min_chars"]) == (0.2, MIN_CHARS)
    assert (printed["train_tokens"], printed["shard_tokens"]) == (None, 300)
    assert printed["stats"]["seen"] == len(DOCS)

    expected = list(iter_documents(DOCS, 0.2, MIN_CHARS))
    assert _decoded(tok, out / "train", "train") == [t for s, t in expected if s == "train"]
    meta = read_metadata(list_shards(out / "train", "train")[0])
    assert meta["tokenizer_sha256"] == file_sha256(tok_path)
    assert meta["revision"] == spec.revision
