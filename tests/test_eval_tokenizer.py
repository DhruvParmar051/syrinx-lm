"""Tests for ``scripts/eval_tokenizer.py``.

No network: ``load_dataset``, ``stream_texts`` and ``load_reference_tokenizer`` are
replaced with in-memory fakes, and small tokenizers stand in for the 32K one.
"""

import importlib.util
import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tokenizers import Tokenizer, normalizers

from syrinx.data.documents import assign_split, document_hash, normalize_text
from syrinx.tokenizer import (
    EOS_TOKEN,
    VOCAB_SIZE,
    encode_document,
    file_sha256,
    save_tokenizer,
    train_tokenizer,
)

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
"""400 distinct lowercase documents."""

SMALL_VOCAB = 300


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The script loaded as a module (scripts/ is not a package)."""
    return _load_script("eval_tokenizer")


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    """Stands in for the Syrinx tokenizer."""
    return train_tokenizer(DOCS, vocab_size=SMALL_VOCAB, min_frequency=1)


@pytest.fixture(scope="module")
def ref() -> Tokenizer:
    """Stands in for GPT-2: a smaller vocab, so it needs more tokens."""
    return train_tokenizer(DOCS, vocab_size=270, min_frequency=1)


def _is_hash(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value) is not None


# --- settings -------------------------------------------------------------------------


def test_parse_args_defaults(script: ModuleType) -> None:
    """Defaults evaluate the committed tokenizer against pinned GPT-2."""
    args = script.parse_args([])
    assert args.tokenizer == Path("tokenizer/syrinx-bpe-32k/tokenizer.json")
    assert args.vocab_size == VOCAB_SIZE
    assert args.max_docs == 1000
    assert args.val_fraction == 0.005
    assert args.reference == "openai-community/gpt2"
    assert args.reference_revision == script.REFERENCE_REVISION


def test_parse_args_overrides(script: ModuleType) -> None:
    """Every option can be overridden."""
    args = script.parse_args(
        [
            "--tokenizer",
            "x/t.json",
            "--vocab-size",
            "300",
            "--max-docs",
            "7",
            "--val-fraction",
            "0.2",
            "--reference",
            "some/tok",
            "--reference-revision",
            "abc",
        ]
    )
    assert args.tokenizer == Path("x/t.json")
    assert (args.vocab_size, args.max_docs, args.val_fraction) == (300, 7, 0.2)
    assert (args.reference, args.reference_revision) == ("some/tok", "abc")


def test_fineweb_settings_match_training_script(script: ModuleType) -> None:
    """Held-out FineWeb-Edu must come from the same dataset, subset, revision and split
    fraction the tokenizer was trained with, or "val" would mean different documents."""
    train = _load_script("train_tokenizer")
    assert script.FINEWEB_DATASET == train.DEFAULT_DATASET
    assert script.FINEWEB_SUBSET == train.DEFAULT_SUBSET
    assert script.FINEWEB_REVISION == train.DEFAULT_REVISION
    assert script.FINEWEB_VAL_FRACTION == train.DEFAULT_VAL_FRACTION


def test_revisions_are_pinned_commits(script: ModuleType) -> None:
    """Every Hub source is a 40-hex commit, never a moving branch; the dataset commits
    are recorded in docs/data.md."""
    assert _is_hash(script.FINEWEB_REVISION)
    assert _is_hash(script.TINYSTORIES_REVISION)
    assert _is_hash(script.REFERENCE_REVISION)
    data_md = (ROOT / "docs" / "data.md").read_text(encoding="utf-8")
    assert script.TINYSTORIES_REVISION in data_md


# --- stream_texts ---------------------------------------------------------------------


def test_stream_texts_calls_load_dataset(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streams the given split at the given revision and yields the text column."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((args, kwargs))
        return [{"text": "a"}, {"text": "b"}]

    monkeypatch.setattr(script, "load_dataset", fake_load_dataset)
    assert list(script.stream_texts("org/ds", None, "validation", "text", "rev1")) == ["a", "b"]
    assert calls == [
        (
            ("org/ds",),
            {"name": None, "split": "validation", "streaming": True, "revision": "rev1"},
        )
    ]


def test_stream_texts_missing_field_raises(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row without the text column raises KeyError naming the column."""
    monkeypatch.setattr(script, "load_dataset", lambda *a, **k: [{"story": "a"}])
    with pytest.raises(KeyError, match="body"):
        list(script.stream_texts("org/ds", None, "validation", "body", "rev1"))


# --- prepare_texts --------------------------------------------------------------------


def test_prepare_without_split_takes_first_docs(script: ModuleType) -> None:
    """With no val_fraction, the first max_docs documents are returned, as a list."""
    texts = script.prepare_texts(iter(DOCS), 5)
    assert texts == DOCS[:5]
    assert isinstance(texts, list)


def test_prepare_with_split_keeps_only_val(script: ModuleType) -> None:
    """With val_fraction, only val-split documents are kept, in input order."""
    expected = [d for d in DOCS if assign_split(document_hash(d), 0.2) == "val"]
    assert 5 < len(expected) < len(DOCS)
    assert script.prepare_texts(DOCS, 5, val_fraction=0.2) == expected[:5]


def test_prepare_normalizes_before_splitting(script: ModuleType) -> None:
    """Text is normalized, and the split is decided on the normalized text."""
    raw = [f"line\r\nnumber {i}\x00" for i in range(200)]
    normalized = [normalize_text(r) for r in raw]
    expected = [n for n in normalized if assign_split(document_hash(n), 0.2) == "val"]
    assert script.prepare_texts(raw, 1000, val_fraction=0.2) == expected


def test_prepare_skips_empty(script: ModuleType) -> None:
    """Documents empty after normalization are skipped and don't count toward max_docs."""
    assert script.prepare_texts(["", "\x00", "a", "", "b", "c"], 2) == ["a", "b"]


def test_prepare_returns_fewer_when_input_runs_out(script: ModuleType) -> None:
    """A short input returns what there is."""
    assert script.prepare_texts(["a", "b"], 10) == ["a", "b"]


def test_prepare_does_not_read_past_max_docs(script: ModuleType) -> None:
    """Once max_docs documents are collected, no further raw document is pulled."""
    pulled: list[int] = []

    def stream() -> Iterator[str]:
        for i, doc in enumerate(DOCS):
            pulled.append(i)
            yield doc

    texts = script.prepare_texts(stream(), 3, val_fraction=0.2)
    assert len(pulled) == DOCS.index(texts[-1]) + 1
    assert len(pulled) < len(DOCS)


@pytest.mark.parametrize("max_docs", [0, -1])
def test_prepare_rejects_non_positive_max_docs(script: ModuleType, max_docs: int) -> None:
    """max_docs <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        script.prepare_texts(DOCS, max_docs)


# --- roundtrip_failures ---------------------------------------------------------------


def test_roundtrip_all_pass(script: ModuleType, tok: Tokenizer) -> None:
    """A correct tokenizer has no failures, including on unseen Unicode."""
    texts = [*DOCS[:20], "café 日本 \U0001f642", f"literal {EOS_TOKEN} text", ""]
    assert script.roundtrip_failures(tok, texts) == []


def test_roundtrip_reports_failing_indices(script: ModuleType) -> None:
    """A tokenizer that lowercases (changes the text) fails exactly on the documents
    with uppercase letters."""
    lossy = train_tokenizer(DOCS, vocab_size=SMALL_VOCAB, min_frequency=1)
    lossy.normalizer = normalizers.Lowercase()
    texts = ["the dog", "The dog", "a fox", "A FOX"]
    assert script.roundtrip_failures(lossy, texts) == [1, 3]


def test_roundtrip_checks_eos_variant(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tok: Tokenizer
) -> None:
    """A document fails if the version WITH the EOS id does not decode to text + EOS."""
    real_decode = script.decode

    def eos_dropping_decode(tokenizer: Tokenizer, ids: list[int]) -> str:
        return real_decode(tokenizer, [i for i in ids if i != 0])

    monkeypatch.setattr(script, "decode", eos_dropping_decode)
    assert script.roundtrip_failures(tok, ["the dog", "a fox"]) == [0, 1]


def test_roundtrip_empty_list(script: ModuleType, tok: Tokenizer) -> None:
    """No documents, no failures."""
    assert script.roundtrip_failures(tok, []) == []


# --- evaluate_corpus ------------------------------------------------------------------


def test_evaluate_corpus_numbers(script: ModuleType, tok: Tokenizer, ref: Tokenizer) -> None:
    """Every number is computed from UTF-8 bytes and append_eos=False token counts."""
    texts = [*DOCS[:30], "café \U0001f642"]
    result = script.evaluate_corpus(tok, ref, texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tok = sum(len(encode_document(tok, t, append_eos=False)) for t in texts)
    n_ref = sum(len(encode_document(ref, t, append_eos=False)) for t in texts)
    assert result["docs"] == len(texts)
    assert result["bytes"] == n_bytes
    assert result["tokens"] == n_tok
    assert result["reference_tokens"] == n_ref
    assert result["bytes_per_token"] == pytest.approx(n_bytes / n_tok)
    assert result["reference_bytes_per_token"] == pytest.approx(n_bytes / n_ref)
    assert result["relative_tokens"] == pytest.approx(n_tok / n_ref)
    assert result["roundtrip_failures"] == 0


def test_evaluate_corpus_bigger_vocab_compresses_more(
    script: ModuleType, tok: Tokenizer, ref: Tokenizer
) -> None:
    """Sanity: the 300-token tokenizer beats the 270-token one on its own training text."""
    result = script.evaluate_corpus(tok, ref, DOCS[:50])
    assert result["bytes_per_token"] > result["reference_bytes_per_token"]
    assert result["relative_tokens"] < 1


def test_evaluate_corpus_is_json_serializable(
    script: ModuleType, tok: Tokenizer, ref: Tokenizer
) -> None:
    """The result goes straight into the JSON summary."""
    json.dumps(script.evaluate_corpus(tok, ref, DOCS[:5]))


@pytest.mark.parametrize("texts", [[], [""]])
def test_evaluate_corpus_no_tokens_raises(
    script: ModuleType, tok: Tokenizer, ref: Tokenizer, texts: list[str]
) -> None:
    """A corpus with no tokens raises ValueError instead of dividing by zero."""
    with pytest.raises(ValueError):
        script.evaluate_corpus(tok, ref, texts)


# --- main -----------------------------------------------------------------------------


@pytest.fixture()
def main_env(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tok: Tokenizer,
    ref: Tokenizer,
) -> dict[str, Any]:
    """A saved small tokenizer, a fake reference loader and a fake stream that record
    their calls."""
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tok, path, vocab_size=SMALL_VOCAB)
    env: dict[str, Any] = {"path": path, "streams": [], "reference_calls": []}

    def fake_stream(
        dataset: str, subset: str | None, split: str, text_field: str, revision: str
    ) -> Iterator[str]:
        env["streams"].append((dataset, subset, split, text_field, revision))
        return iter(DOCS)

    def fake_reference(name: str, revision: str) -> Tokenizer:
        env["reference_calls"].append((name, revision))
        return ref

    monkeypatch.setattr(script, "stream_texts", fake_stream)
    monkeypatch.setattr(script, "load_reference_tokenizer", fake_reference)
    env["argv"] = [
        "--tokenizer",
        str(path),
        "--vocab-size",
        str(SMALL_VOCAB),
        "--max-docs",
        "40",
        "--val-fraction",
        "0.2",
    ]
    return env


def test_main_end_to_end(
    script: ModuleType, main_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Returns 0 and prints one JSON line covering both corpora."""
    assert script.main(main_env["argv"]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert summary["tokenizer"] == str(main_env["path"])
    assert summary["tokenizer_sha256"] == file_sha256(main_env["path"])
    assert summary["reference"] == "openai-community/gpt2"
    assert summary["reference_revision"] == script.REFERENCE_REVISION
    assert summary["val_fraction"] == 0.2
    assert summary["max_docs"] == 40
    assert set(summary["corpora"]) == {"fineweb_edu_val", "tinystories_val"}

    val_docs = [d for d in DOCS if assign_split(document_hash(d), 0.2) == "val"][:40]
    assert summary["corpora"]["fineweb_edu_val"]["docs"] == len(val_docs)
    assert summary["corpora"]["fineweb_edu_val"]["bytes"] == sum(
        len(d.encode("utf-8")) for d in val_docs
    )
    assert summary["corpora"]["tinystories_val"]["docs"] == 40
    assert summary["corpora"]["tinystories_val"]["bytes"] == sum(
        len(d.encode("utf-8")) for d in DOCS[:40]
    )
    for corpus in summary["corpora"].values():
        assert corpus["roundtrip_failures"] == 0


def test_main_streams_the_right_sources(script: ModuleType, main_env: dict[str, Any]) -> None:
    """FineWeb-Edu train stream (split by hash) and the TinyStories validation split,
    each at its pinned revision; GPT-2 at its pinned revision."""
    script.main(main_env["argv"])
    assert sorted(main_env["streams"], key=str) == sorted(
        [
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", script.FINEWEB_REVISION),
            ("roneneldan/TinyStories", None, "validation", "text", script.TINYSTORIES_REVISION),
        ],
        key=str,
    )
    assert main_env["reference_calls"] == [("openai-community/gpt2", script.REFERENCE_REVISION)]


def test_main_returns_1_on_roundtrip_failure(
    script: ModuleType, main_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any round-trip failure makes the exit code 1 (Gate 1 requires 100%)."""
    monkeypatch.setattr(script, "roundtrip_failures", lambda tokenizer, texts: [0])
    assert script.main(main_env["argv"]) == 1


def test_main_rejects_wrong_vocab_size(script: ModuleType, main_env: dict[str, Any]) -> None:
    """The tokenizer file is validated on load (default expects 32,768)."""
    argv = [a for a in main_env["argv"] if a not in ("--vocab-size", str(SMALL_VOCAB))]
    with pytest.raises(ValueError):
        script.main(argv)
