"""Tests for ``scripts/train_tokenizer.py``.

No network: ``load_dataset`` and ``stream_texts`` are replaced with in-memory fakes.
"""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from syrinx.data.documents import SPLIT_SALT, assign_split, document_hash, normalize_text
from syrinx.tokenizer import VOCAB_SIZE, file_sha256, load_tokenizer

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
    "near",
    "quiet",
    "river",
    "banks",
]
DOCS: list[str] = [
    " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(8)) + f" {i}." for i in range(400)
]
"""400 distinct documents (the trailing number makes each one unique)."""

SMALL_VOCAB = 300


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The script loaded as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "train_tokenizer", ROOT / "scripts" / "train_tokenizer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select(script: ModuleType, docs: list[str], val_fraction: float, max_bytes: int) -> Any:
    """Run select_train_texts to the end; return (texts, stats)."""
    stats = script.SelectionStats()
    texts = list(script.select_train_texts(docs, val_fraction, max_bytes, stats))
    return texts, stats


# --- parse_args -----------------------------------------------------------------------


def test_parse_args_defaults(script: ModuleType) -> None:
    """Defaults are the agreed Phase 1 settings."""
    args = script.parse_args([])
    assert args.dataset == "HuggingFaceFW/fineweb-edu"
    assert args.subset == "sample-10BT"
    assert args.text_field == "text"
    assert args.val_fraction == 0.005
    assert args.max_bytes == 1_500_000_000
    assert isinstance(args.max_bytes, int)
    assert args.vocab_size == VOCAB_SIZE
    assert args.min_frequency == 2
    assert args.out == Path("tokenizer/syrinx-bpe-32k/tokenizer.json")


def test_parse_args_overrides(script: ModuleType) -> None:
    """Every option can be overridden; --max-bytes accepts scientific notation."""
    args = script.parse_args(
        [
            "--dataset",
            "d",
            "--subset",
            "s",
            "--text-field",
            "body",
            "--val-fraction",
            "0.1",
            "--max-bytes",
            "2e9",
            "--vocab-size",
            "300",
            "--min-frequency",
            "1",
            "--out",
            "x/t.json",
        ]
    )
    assert (args.dataset, args.subset, args.text_field) == ("d", "s", "body")
    assert args.val_fraction == 0.1
    assert args.max_bytes == 2_000_000_000
    assert isinstance(args.max_bytes, int)
    assert (args.vocab_size, args.min_frequency) == (300, 1)
    assert args.out == Path("x/t.json")


@pytest.mark.parametrize("bad", ["abc", "1.5GB"])
def test_parse_args_rejects_bad_max_bytes(script: ModuleType, bad: str) -> None:
    """A non-numeric --max-bytes is a usage error (argparse exits)."""
    with pytest.raises(SystemExit):
        script.parse_args(["--max-bytes", bad])


# --- stream_texts ---------------------------------------------------------------------


def test_stream_texts_calls_load_dataset_streaming(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loads split "train" with streaming=True and yields the text column in order."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((args, kwargs))
        return [{"text": "a", "id": 1}, {"text": "b", "id": 2}]

    monkeypatch.setattr(script, "load_dataset", fake_load_dataset)
    assert list(script.stream_texts("org/ds", "sub", "text")) == ["a", "b"]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("org/ds",)
    assert kwargs == {"name": "sub", "split": "train", "streaming": True}


def test_stream_texts_missing_field_raises(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row without the text column raises KeyError naming the column."""
    monkeypatch.setattr(script, "load_dataset", lambda *a, **k: [{"content": "a"}])
    with pytest.raises(KeyError, match="body"):
        list(script.stream_texts("org/ds", None, "body"))


# --- select_train_texts ---------------------------------------------------------------


def test_select_yields_only_train_documents(script: ModuleType) -> None:
    """Yields exactly the train-split documents, in input order, none from val."""
    texts, stats = _select(script, DOCS, 0.2, 10**12)
    expected = [d for d in DOCS if assign_split(document_hash(d), 0.2) == "train"]
    assert texts == expected
    assert 0 < stats.val < len(DOCS)
    assert stats.train_docs == len(expected)
    assert stats.val == len(DOCS) - len(expected)


def test_select_normalizes_text(script: ModuleType) -> None:
    """Yielded text is normalize_text(raw), and the split uses that normalized text."""
    raw = [f"line one\r\nline two \x00{i}" for i in range(50)]
    texts, _ = _select(script, raw, 0.2, 10**12)
    normalized = [normalize_text(r) for r in raw]
    expected = [n for n in normalized if assign_split(document_hash(n), 0.2) == "train"]
    assert texts == expected
    assert all("\r" not in t and "\x00" not in t for t in texts)


def test_select_skips_empty_documents(script: ModuleType) -> None:
    """Documents that are empty after normalization are skipped and counted."""
    raw = ["", "\x00\x01", DOCS[0], "\r"]
    texts, stats = _select(script, raw, 0.000001, 10**12)
    assert texts == [DOCS[0], "\n"]  # "\r" normalizes to "\n", which is not empty
    assert stats.empty == 2
    assert stats.train_docs == 2


def test_select_skips_exact_duplicates(script: ModuleType) -> None:
    """A document identical (after normalization) to an earlier one is skipped."""
    raw = [DOCS[0], DOCS[1], DOCS[0], DOCS[0].replace("\n", "\r\n"), DOCS[1]]
    texts, stats = _select(script, raw, 0.000001, 10**12)
    assert texts == [DOCS[0], DOCS[1]]
    assert stats.duplicates == 3


def test_select_duplicate_check_after_normalization(script: ModuleType) -> None:
    """CRLF and LF versions of a document count as the same document."""
    raw = ["a\nb", "a\r\nb", "a\rb"]
    texts, stats = _select(script, raw, 0.000001, 10**12)
    assert texts == ["a\nb"]
    assert stats.duplicates == 2


def test_select_stops_at_max_bytes(script: ModuleType) -> None:
    """Stops at the first document that brings the total to >= max_bytes."""
    max_bytes = 1_000
    texts, stats = _select(script, DOCS, 0.2, max_bytes)
    sizes = [len(t.encode("utf-8")) for t in texts]
    assert sum(sizes) >= max_bytes
    assert sum(sizes) - sizes[-1] < max_bytes
    assert stats.train_bytes == sum(sizes)


def test_select_counts_utf8_bytes(script: ModuleType) -> None:
    """train_bytes counts UTF-8 bytes, not characters."""
    raw = ["café \U0001f642"]  # 6 characters, 10 bytes
    _, stats = _select(script, raw, 0.000001, 10**12)
    assert stats.train_bytes == 10


def test_select_does_not_read_past_max_bytes(script: ModuleType) -> None:
    """After max_bytes is reached, no further raw document is pulled from the stream."""
    pulled: list[int] = []

    def stream() -> Iterator[str]:
        for i, doc in enumerate(DOCS):
            pulled.append(i)
            yield doc

    texts, _ = _select(script, DOCS, 0.2, 1_000)
    needed = DOCS.index(texts[-1]) + 1  # raw documents up to the one that hit the limit
    list(script.select_train_texts(stream(), 0.2, 1_000))
    assert needed < len(DOCS)
    assert len(pulled) == needed


def test_select_stats_add_up(script: ModuleType) -> None:
    """seen == empty + duplicates + val + train_docs."""
    raw = ["", DOCS[0], DOCS[0], *DOCS]
    _, stats = _select(script, raw, 0.2, 10**12)
    assert stats.seen == len(raw)
    assert stats.seen == stats.empty + stats.duplicates + stats.val + stats.train_docs
    assert (stats.empty, stats.duplicates) == (1, 2)


def test_select_stats_current_when_consumer_stops_early(script: ModuleType) -> None:
    """Stats already include a document when it is yielded."""
    stats = script.SelectionStats()
    gen = script.select_train_texts(DOCS, 0.000001, 10**12, stats)
    first = next(gen)
    assert stats.train_docs == 1
    assert stats.train_bytes == len(first.encode("utf-8"))


def test_select_works_without_stats(script: ModuleType) -> None:
    """stats is optional."""
    assert list(script.select_train_texts(DOCS[:3], 0.000001, 10**12)) == DOCS[:3]


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_select_rejects_non_positive_max_bytes(script: ModuleType, max_bytes: int) -> None:
    """max_bytes <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        list(script.select_train_texts(DOCS, 0.2, max_bytes))


def test_select_rejects_bad_val_fraction(script: ModuleType) -> None:
    """An invalid val_fraction raises ValueError (from assign_split)."""
    with pytest.raises(ValueError):
        list(script.select_train_texts(DOCS, 0.0, 10**12))


# --- main -----------------------------------------------------------------------------


def _main_args(out: Path) -> list[str]:
    return [
        "--val-fraction",
        "0.2",
        "--vocab-size",
        str(SMALL_VOCAB),
        "--min-frequency",
        "1",
        "--out",
        str(out),
    ]


def test_main_end_to_end(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Trains, saves a loadable tokenizer, returns 0 and prints a JSON summary whose
    sha256 matches the file."""
    monkeypatch.setattr(script, "stream_texts", lambda *a, **k: iter(DOCS))
    out = tmp_path / "tok" / "tokenizer.json"
    assert script.main(_main_args(out)) == 0
    load_tokenizer(out, vocab_size=SMALL_VOCAB)

    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["sha256"] == file_sha256(out)
    assert summary["out"] == str(out)
    assert summary["val_fraction"] == 0.2
    assert summary["vocab_size"] == SMALL_VOCAB
    assert summary["min_frequency"] == 1
    assert summary["split_salt"] == SPLIT_SALT
    assert summary["dataset"] == "HuggingFaceFW/fineweb-edu"
    assert summary["subset"] == "sample-10BT"
    assert summary["text_field"] == "text"
    assert summary["max_bytes"] == 1_500_000_000
    stats = summary["stats"]
    assert stats["seen"] == len(DOCS)
    assert stats["train_docs"] + stats["val"] == len(DOCS)


def test_main_trains_only_on_train_split(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The texts handed to train_tokenizer are exactly the train-split documents:
    no validation document leaks into tokenizer training."""
    monkeypatch.setattr(script, "stream_texts", lambda *a, **k: iter(DOCS))
    real_train = script.train_tokenizer
    seen: list[str] = []

    def recording_train(texts: Any, *args: Any, **kwargs: Any) -> Any:
        materialized = list(texts)
        seen.extend(materialized)
        return real_train(materialized, *args, **kwargs)

    monkeypatch.setattr(script, "train_tokenizer", recording_train)
    script.main(_main_args(tmp_path / "tokenizer.json"))
    expected = [d for d in DOCS if assign_split(document_hash(d), 0.2) == "train"]
    assert seen == expected


def test_main_passes_stream_settings(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--dataset, --subset and --text-field reach stream_texts."""
    calls: list[tuple[str, str | None, str]] = []

    def fake_stream(dataset: str, subset: str | None, text_field: str) -> Iterator[str]:
        calls.append((dataset, subset, text_field))
        return iter(DOCS)

    monkeypatch.setattr(script, "stream_texts", fake_stream)
    argv = [*_main_args(tmp_path / "t.json"), "--dataset", "d", "--subset", "s"]
    script.main([*argv, "--text-field", "body"])
    assert calls == [("d", "s", "body")]


def test_main_refuses_existing_output_before_streaming(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An existing --out raises FileExistsError without touching the dataset, and the
    file is left intact."""

    def must_not_stream(*args: Any, **kwargs: Any) -> Iterator[str]:
        raise AssertionError("streamed although the output exists")

    monkeypatch.setattr(script, "stream_texts", must_not_stream)
    out = tmp_path / "tokenizer.json"
    out.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        script.main(_main_args(out))
    assert out.read_text(encoding="utf-8") == "keep me"
