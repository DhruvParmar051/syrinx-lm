"""Tests for ``scripts/check_shards.py``.

Each test builds a small, correct shard directory with ``build_shards.main`` (fake
stream, small tokenizer), then breaks one thing and expects the check to report it.
"""

import hashlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer

from syrinx.data.documents import assign_split, document_hash
from syrinx.data.shards import (
    list_shards,
    metadata_path,
    read_metadata,
    read_shard,
    write_shard_file,
)
from syrinx.tokenizer import (
    VOCAB_SIZE,
    file_sha256,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)

ROOT = Path(__file__).resolve().parents[1]

_WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "while", "small"]
DOCS: list[str] = [
    " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(4 + i % 7)) + f" {i}." for i in range(240)
]
"""240 distinct documents, ~20 to ~65 characters."""

SMALL_VOCAB = 280
MIN_CHARS = 25


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load("check_shards")


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load("build_shards")


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_tokenizer(DOCS, vocab_size=SMALL_VOCAB, min_frequency=1)


def _fake_stream(
    dataset: str, subset: str | None, split: str, text_field: str, revision: str
) -> Iterator[str]:
    """TinyStories-like: 'train' then 'validation' together are DOCS."""
    return iter(DOCS[:160] if split == "train" else DOCS[160:])


@pytest.fixture()
def built(
    builder: ModuleType, tok: Tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """A correct build of DOCS (tinystories layout) plus the paths the checks need."""
    tok_path = tmp_path / "tok.json"
    save_tokenizer(tok, tok_path, vocab_size=SMALL_VOCAB)
    monkeypatch.setattr(builder, "stream_texts", _fake_stream)
    data_dir = tmp_path / "data"
    builder.main(
        [
            "--source",
            "tinystories",
            "--tokenizer",
            str(tok_path),
            "--vocab-size",
            str(SMALL_VOCAB),
            "--out-dir",
            str(data_dir),
            "--min-chars",
            str(MIN_CHARS),
            "--val-fraction",
            "0.25",
            "--shard-tokens",
            "200",
        ]
    )
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["splits"]["train"]["shards"]) > 1
    return {"tok_path": tok_path, "data_dir": data_dir, "summary": summary}


def _argv(built: dict[str, Any], *extra: str) -> list[str]:
    return [
        "--data-dir",
        str(built["data_dir"]),
        "--tokenizer",
        str(built["tok_path"]),
        "--vocab-size",
        str(SMALL_VOCAB),
        *extra,
    ]


def _run(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(script, "stream_texts", _fake_stream)
    code = script.main(_argv(built, *extra))
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return code, report


def _problems_mention(report: dict[str, Any], *words: str) -> bool:
    return any(all(w in p for w in words) for p in report["problems"])


# --- settings -------------------------------------------------------------------------


def test_source_splits_match_build_script(script: ModuleType, builder: ModuleType) -> None:
    """The check re-streams exactly the splits the build streamed."""
    assert set(script.SOURCE_SPLITS) == set(builder.SOURCES)
    for name, spec in builder.SOURCES.items():
        assert script.SOURCE_SPLITS[name] == spec.splits


def test_parse_args(script: ModuleType) -> None:
    """--data-dir is required; defaults for the rest."""
    args = script.parse_args(["--data-dir", "d"])
    assert args.data_dir == Path("d")
    assert args.tokenizer == Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
    assert args.vocab_size == VOCAB_SIZE
    assert args.verify_docs == 20_000
    assert script.parse_args(["--data-dir", "d", "--verify-docs", "0"]).verify_docs == 0
    with pytest.raises(SystemExit):
        script.parse_args([])


# --- document_token_hashes ------------------------------------------------------------


def test_document_token_hashes(script: ModuleType) -> None:
    """One SHA-256 per document over its little-endian uint16 ids, EOS excluded; an
    empty document hashes like empty input."""
    tokens = np.array([3, 300, 0, 0, 3, 300, 0], dtype=np.uint16)
    hashes = script.document_token_hashes(tokens)
    expected_first = hashlib.sha256(bytes([3, 0, 44, 1])).digest()
    assert hashes == [expected_first, hashlib.sha256(b"").digest(), expected_first]


# --- the clean build passes -----------------------------------------------------------


def test_clean_build_passes(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A correct build passes every check; the report matches the summary and is also
    written to check.json."""
    code, report = _run(script, built, monkeypatch, capsys)
    assert report["problems"] == []
    assert report["passed"] is True and code == 0
    assert report["overlap_docs"] == 0
    for split in ("train", "val"):
        info = built["summary"]["splits"][split]
        r = report["splits"][split]
        assert (r["docs"], r["tokens"], r["shards"]) == (
            info["docs"],
            info["tokens"],
            info["shards"],
        )
        assert r["duplicate_docs"] == 0
    assert report["verified_docs"] == {
        "train": built["summary"]["splits"]["train"]["docs"],
        "val": built["summary"]["splits"]["val"]["docs"],
    }
    assert report["tokenizer_sha256"] == file_sha256(built["tok_path"])
    on_disk = json.loads((built["data_dir"] / "check.json").read_text(encoding="utf-8"))
    assert on_disk == report


def test_verify_docs_limits_comparison(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--verify-docs N compares only the first N selected documents; 0 skips it."""
    _, report = _run(script, built, monkeypatch, capsys, "--verify-docs", "20")
    assert sum(report["verified_docs"].values()) == 20
    assert report["passed"] is True

    def must_not_stream(*a: Any, **k: Any) -> Iterator[str]:
        raise AssertionError("streamed although --verify-docs 0")

    monkeypatch.setattr(script, "stream_texts", must_not_stream)
    code = script.main(_argv(built, "--verify-docs", "0"))
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and report["verified_docs"] == {"train": 0, "val": 0}


# --- each failure is reported ---------------------------------------------------------


def _first_val_doc(built: dict[str, Any]) -> np.ndarray:
    val_shard = read_shard(list_shards(built["data_dir"] / "val", "val")[0])
    end = int(np.flatnonzero(np.asarray(val_shard) == 0)[0])
    return np.asarray(val_shard[: end + 1])  # ids + EOS


def test_overlap_detected(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A val document that also appears in a train shard fails the check."""
    write_shard_file(built["data_dir"] / "train" / "train_9999.bin", _first_val_doc(built), 1, {})
    code, report = _run(script, built, monkeypatch, capsys, "--verify-docs", "0")
    assert code == 1 and report["passed"] is False
    assert report["overlap_docs"] == 1
    assert _problems_mention(report, "overlap")


def test_duplicate_within_split_detected(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same document twice in one split is counted as a duplicate."""
    write_shard_file(built["data_dir"] / "val" / "val_9999.bin", _first_val_doc(built), 1, {})
    _, report = _run(script, built, monkeypatch, capsys, "--verify-docs", "0")
    assert report["splits"]["val"]["duplicate_docs"] == 1
    assert report["passed"] is False


def test_truncated_shard_detected(script: ModuleType, built: dict[str, Any]) -> None:
    """A shard whose size disagrees with its sidecar is reported by name and skipped."""
    path = list_shards(built["data_dir"] / "train", "train")[0]
    path.write_bytes(path.read_bytes()[:-2])
    report, _, problems = script.check_split(built["data_dir"] / "train", "train", SMALL_VOCAB)
    assert any(path.name in p for p in problems)
    assert path.name not in report["shards"]


def test_missing_trailing_eos_detected(script: ModuleType, built: dict[str, Any]) -> None:
    """A shard that does not end with EOS is reported (and not hashed)."""
    bad = built["data_dir"] / "val" / "val_9999.bin"
    write_shard_file(bad, np.array([5, 6, 0, 7], dtype=np.uint16), 1, {})
    _, _, problems = script.check_split(built["data_dir"] / "val", "val", SMALL_VOCAB)
    assert any(bad.name in p for p in problems)


def test_eos_count_mismatch_detected(script: ModuleType, built: dict[str, Any]) -> None:
    """A sidecar whose num_docs disagrees with the EOS count is reported."""
    path = list_shards(built["data_dir"] / "train", "train")[0]
    meta = read_metadata(path)
    meta["num_docs"] += 1
    metadata_path(path).write_text(json.dumps(meta), encoding="utf-8")
    _, _, problems = script.check_split(built["data_dir"] / "train", "train", SMALL_VOCAB)
    assert any(path.name in p for p in problems)


def test_out_of_vocab_id_detected(script: ModuleType, built: dict[str, Any]) -> None:
    """An id >= vocab_size is reported."""
    bad = built["data_dir"] / "val" / "val_9999.bin"
    write_shard_file(bad, np.array([SMALL_VOCAB, 0], dtype=np.uint16), 1, {})
    _, _, problems = script.check_split(built["data_dir"] / "val", "val", SMALL_VOCAB)
    assert any(bad.name in p for p in problems)


def test_clean_split_has_no_problems(script: ModuleType, built: dict[str, Any]) -> None:
    """check_split on an untouched build: no problems, counts match the summary."""
    for split in ("train", "val"):
        report, hashes, problems = script.check_split(built["data_dir"] / split, split, SMALL_VOCAB)
        assert problems == []
        assert report["docs"] == built["summary"]["splits"][split]["docs"] == len(hashes)


def test_summary_mismatch_detected(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Counts on disk that disagree with summary.json are reported (split and field)."""
    summary = dict(built["summary"])
    summary["splits"] = json.loads(json.dumps(summary["splits"]))
    summary["splits"]["train"]["tokens"] += 1
    (built["data_dir"] / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _, report = _run(script, built, monkeypatch, capsys, "--verify-docs", "0")
    assert _problems_mention(report, "train", "tokens")
    assert report["passed"] is False


def test_provenance_mismatch_detected(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sidecar with a different revision (e.g. a shard from another build) is
    reported by shard and key."""
    path = list_shards(built["data_dir"] / "val", "val")[0]
    meta = read_metadata(path)
    meta["revision"] = "0" * 40
    metadata_path(path).write_text(json.dumps(meta), encoding="utf-8")
    _, report = _run(script, built, monkeypatch, capsys, "--verify-docs", "0")
    assert _problems_mention(report, path.name, "revision")


def test_different_tokenizer_detected(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Checking with a tokenizer file other than the one the shards were built with is
    reported (its SHA-256 differs)."""
    other = train_tokenizer([*DOCS, "zebra quagga"], vocab_size=SMALL_VOCAB, min_frequency=1)
    other_path = tmp_path / "other.json"
    save_tokenizer(other, other_path, vocab_size=SMALL_VOCAB)
    built = {**built, "tok_path": other_path}
    _, report = _run(script, built, monkeypatch, capsys, "--verify-docs", "0")
    assert _problems_mention(report, "tokenizer_sha256")
    assert report["passed"] is False


def test_source_mismatch_detected(
    script: ModuleType,
    built: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the re-streamed source differs from what was built in one document's text
    (chosen so it stays in the same split, so the counts still agree), the decode
    comparison reports that document."""
    original = DOCS[10]  # 7 words: kept by the length filter
    split = assign_split(document_hash(original), 0.25)
    changed_text = next(
        t
        for t in (f"{original} v{k}" for k in range(1000))
        if assign_split(document_hash(t), 0.25) == split
    )
    changed = [changed_text if i == 10 else d for i, d in enumerate(DOCS)]

    verified, problems = script.check_against_source(
        changed, script_tokenizer(built), built["data_dir"], built["summary"], 10**6
    )
    assert any("differs" in p and split in p for p in problems)
    assert verified == {
        "train": built["summary"]["splits"]["train"]["docs"],
        "val": built["summary"]["splits"]["val"]["docs"],
    }

    def stream(
        dataset: str, subset: str | None, split_name: str, text_field: str, revision: str
    ) -> Iterator[str]:
        return iter(changed[:160] if split_name == "train" else changed[160:])

    monkeypatch.setattr(script, "stream_texts", stream)
    assert script.main(_argv(built)) == 1
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["passed"] is False


def script_tokenizer(built: dict[str, Any]) -> Tokenizer:
    """The small tokenizer the build used, loaded from its file."""
    return load_tokenizer(built["tok_path"], vocab_size=SMALL_VOCAB)


def test_check_against_source_reports_short_shards(
    script: ModuleType, built: dict[str, Any], tok: Tokenizer
) -> None:
    """If the source yields more documents of a split than the shards hold, that is a
    problem (a build that stopped early must not pass as complete)."""
    extra = [*DOCS, *(f"brand new document number {i} with enough text" for i in range(50))]
    verified, problems = script.check_against_source(
        extra, tok, built["data_dir"], built["summary"], 10**6
    )
    assert problems
    assert verified["train"] <= built["summary"]["splits"]["train"]["docs"]


def test_check_against_source_rejects_non_positive(
    script: ModuleType, built: dict[str, Any], tok: Tokenizer
) -> None:
    """verify_docs must be positive."""
    with pytest.raises(ValueError):
        script.check_against_source(DOCS, tok, built["data_dir"], built["summary"], 0)


def test_missing_split_detected(script: ModuleType, tmp_path: Path) -> None:
    """check_split on a directory without shards reports a problem."""
    report, hashes, problems = script.check_split(tmp_path / "nothing", "train", SMALL_VOCAB)
    assert problems and not hashes
    assert report["docs"] == 0 and report["tokens"] == 0 and report["shards"] == []


# --- streaming ------------------------------------------------------------------------


def test_stream_source_uses_summary_provenance(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-streams the summary's dataset/subset/revision, split by split."""
    calls: list[tuple[Any, ...]] = []

    def fake(
        dataset: str, subset: str | None, split: str, text_field: str, revision: str
    ) -> Iterator[str]:
        calls.append((dataset, subset, split, text_field, revision))
        return iter([split])

    monkeypatch.setattr(script, "stream_texts", fake)
    summary = {"source": "tinystories", "dataset": "d", "subset": None, "revision": "r"}
    assert list(script.stream_source(summary)) == ["train", "validation"]
    assert calls == [("d", None, "train", "text", "r"), ("d", None, "validation", "text", "r")]
    with pytest.raises(KeyError):
        list(script.stream_source({**summary, "source": "unknown"}))


def test_stream_texts_reads_cached_hub_files(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delegates to ``syrinx.data.sources.iter_hub_texts`` (download each parquet file
    once, read it locally), same as the build script."""
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
