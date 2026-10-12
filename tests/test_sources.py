"""Tests for ``syrinx.data.sources``. No network: the Hub calls are in-memory fakes and the
parquet files are written to ``tmp_path``."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from syrinx.data import sources

ROOT = Path(__file__).resolve().parents[1]

ACCENT = "caf" + chr(0xE9) + " " + chr(0x4E16) + chr(0x754C)
"""Non-ASCII text (Latin + CJK) built without escapes in the source file."""


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, texts: list[str], row_group_size: int = 1000) -> Path:
    """A parquet file with the text column plus an unrelated one (like the real data)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"id": [f"id-{i}" for i in range(len(texts))], "text": texts})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


class FakeHub:
    """Stands in for ``list_repo_files`` and ``hf_hub_download``; records every call
    into ``events`` so tests can check order and laziness."""

    def __init__(self, files: dict[str, Path], extra: list[str] | None = None) -> None:
        self.files = files
        self.extra = extra or []
        self.events: list[tuple[Any, ...]] = []

    def list_repo_files(self, repo_id: str, **kwargs: Any) -> list[str]:
        self.events.append(("list", repo_id, kwargs))
        names = list(self.files) + self.extra
        return sorted(names, reverse=True)  # deliberately not in order

    def hf_hub_download(self, **kwargs: Any) -> str:
        self.events.append(("fetch", kwargs["filename"], kwargs))
        return str(self.files[kwargs["filename"]])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sources, "list_repo_files", self.list_repo_files)
        monkeypatch.setattr(sources, "hf_hub_download", self.hf_hub_download)


# --- split_file_pattern ---------------------------------------------------------------


def test_every_build_split_has_a_pattern() -> None:
    """Each split the build script reads resolves to a parquet glob."""
    build = _load("build_shards")
    for spec in build.SOURCES.values():
        for split in spec.splits:
            pattern = sources.split_file_pattern(spec.dataset, spec.subset, split)
            assert pattern.endswith(".parquet")


def test_split_file_pattern_values() -> None:
    assert (
        sources.split_file_pattern("HuggingFaceFW/fineweb-edu", "sample-10BT", "train")
        == "sample/10BT/*.parquet"
    )
    assert (
        sources.split_file_pattern("roneneldan/TinyStories", None, "validation")
        == "data/validation-*.parquet"
    )


def test_split_file_pattern_unknown_raises() -> None:
    with pytest.raises(KeyError, match="sample-100BT"):
        sources.split_file_pattern("HuggingFaceFW/fineweb-edu", "sample-100BT", "train")
    with pytest.raises(KeyError, match="test"):
        sources.split_file_pattern("roneneldan/TinyStories", None, "test")


# --- list_split_files -----------------------------------------------------------------


def test_list_split_files_filters_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake(repo_id: str, **kwargs: Any) -> list[str]:
        calls.append((repo_id, kwargs))
        return [
            "README.md",
            "data/validation-00000-of-00001.parquet",
            "data/train-00001-of-00002.parquet",
            "data/train-00000-of-00002.parquet",
            "data/train-notes.txt",
        ]

    monkeypatch.setattr(sources, "list_repo_files", fake)
    got = sources.list_split_files("org/ds", "abc123", "data/train-*.parquet")
    assert got == ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
    assert calls == [("org/ds", {"repo_type": "dataset", "revision": "abc123"})]


def test_list_split_files_is_case_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching is case-sensitive (``fnmatchcase``)."""
    monkeypatch.setattr(
        sources,
        "list_repo_files",
        lambda repo_id, **k: ["data/TRAIN-0.parquet", "data/train-0.parquet"],
    )
    assert sources.list_split_files("d", "r", "data/train-*.parquet") == ["data/train-0.parquet"]


def test_list_split_files_raises_when_nothing_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sources, "list_repo_files", lambda repo_id, **k: ["README.md"])
    with pytest.raises(ValueError, match="r0"):
        sources.list_split_files("d", "r0", "data/train-*.parquet")


# --- fetch_file -----------------------------------------------------------------------


def test_fetch_file_downloads_at_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "x.parquet")

    monkeypatch.setattr(sources, "hf_hub_download", fake)
    got = sources.fetch_file("org/ds", "abc123", "data/x.parquet")
    assert got == tmp_path / "x.parquet" and isinstance(got, Path)
    assert calls == [
        {
            "repo_id": "org/ds",
            "filename": "data/x.parquet",
            "repo_type": "dataset",
            "revision": "abc123",
        }
    ]


# --- iter_parquet_texts ---------------------------------------------------------------


def test_iter_parquet_texts_reads_column_in_order(tmp_path: Path) -> None:
    """Every row, in order, across row groups and read batches; text kept exactly."""
    texts = [f"doc {i}" for i in range(2500)]
    texts[7] = ACCENT
    texts[8] = ""
    texts[9] = "line one\n\nline two\r\n  "
    path = _write(tmp_path / "a.parquet", texts, row_group_size=1000)
    assert list(sources.iter_parquet_texts(path, "text", batch_rows=333)) == texts
    assert list(sources.iter_parquet_texts(path, "text")) == texts


def test_iter_parquet_texts_reads_other_column(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.parquet", ["x", "y"])
    assert list(sources.iter_parquet_texts(path, "id")) == ["id-0", "id-1"]


def test_iter_parquet_texts_missing_column_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.parquet", ["x"])
    with pytest.raises(KeyError, match="story"):
        next(sources.iter_parquet_texts(path, "story"))


def test_iter_parquet_texts_is_lazy(tmp_path: Path) -> None:
    """A generator: taking the first rows does not require reading the whole file."""
    path = _write(tmp_path / "a.parquet", [f"d{i}" for i in range(5000)])
    it = sources.iter_parquet_texts(path, "text", batch_rows=10)
    assert [next(it) for _ in range(3)] == ["d0", "d1", "d2"]


# --- iter_hub_texts -------------------------------------------------------------------


def test_iter_hub_texts_reads_files_in_sorted_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = {
        "data/train-00001-of-00002.parquet": _write(tmp_path / "1.parquet", ["c", "d", "e"]),
        "data/train-00000-of-00002.parquet": _write(tmp_path / "0.parquet", ["a", "b"]),
        "data/validation-00000-of-00001.parquet": _write(tmp_path / "v.parquet", ["v"]),
    }
    hub = FakeHub(files, extra=["README.md"])
    hub.install(monkeypatch)
    got = list(sources.iter_hub_texts("roneneldan/TinyStories", None, "train", "text", "rev1"))
    assert got == ["a", "b", "c", "d", "e"]
    assert hub.events[0] == (
        "list",
        "roneneldan/TinyStories",
        {"repo_type": "dataset", "revision": "rev1"},
    )
    fetched = [e for e in hub.events if e[0] == "fetch"]
    assert [e[1] for e in fetched] == [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
    ]
    for _, _, kwargs in fetched:
        assert kwargs["repo_id"] == "roneneldan/TinyStories"
        assert (kwargs["repo_type"], kwargs["revision"]) == ("dataset", "rev1")


def test_iter_hub_texts_validation_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    files = {
        "data/train-00000-of-00001.parquet": _write(tmp_path / "t.parquet", ["t"]),
        "data/validation-00000-of-00001.parquet": _write(tmp_path / "v.parquet", ["v1", "v2"]),
    }
    FakeHub(files).install(monkeypatch)
    got = list(sources.iter_hub_texts("roneneldan/TinyStories", None, "validation", "text", "r"))
    assert got == ["v1", "v2"]


def test_iter_hub_texts_uses_subset_pattern(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FineWeb-Edu sample-10BT reads only sample/10BT/, not other subsets' files."""
    files = {
        "sample/10BT/000_00000.parquet": _write(tmp_path / "a.parquet", ["fw0"]),
        "sample/10BT/001_00000.parquet": _write(tmp_path / "b.parquet", ["fw1"]),
        "sample/100BT/000_00000.parquet": _write(tmp_path / "c.parquet", ["big"]),
        "data/CC-MAIN-2024-10/train-00000-of-00001.parquet": _write(tmp_path / "d.parquet", ["cc"]),
    }
    FakeHub(files).install(monkeypatch)
    got = list(
        sources.iter_hub_texts("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", "r")
    )
    assert got == ["fw0", "fw1"]


def test_iter_hub_texts_fetches_lazily(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """File k+1 is downloaded only after file k is exhausted; stopping early never
    downloads the remaining files."""
    files = {
        f"sample/10BT/{i:03d}_00000.parquet": _write(
            tmp_path / f"{i}.parquet", [f"f{i}a", f"f{i}b"]
        )
        for i in range(3)
    }
    hub = FakeHub(files)
    hub.install(monkeypatch)
    it = sources.iter_hub_texts("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", "r")
    assert not hub.events  # a generator: nothing happens before the first next()

    def fetched() -> list[str]:
        return [e[1] for e in hub.events if e[0] == "fetch"]

    assert next(it) == "f0a"
    assert fetched() == ["sample/10BT/000_00000.parquet"]
    assert next(it) == "f0b"
    assert fetched() == ["sample/10BT/000_00000.parquet"]
    assert next(it) == "f1a"
    assert fetched() == ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"]
    it.close()
    assert len(fetched()) == 2


def test_iter_hub_texts_missing_column_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = {"data/train-0.parquet": _write(tmp_path / "a.parquet", ["x"])}
    FakeHub(files).install(monkeypatch)
    with pytest.raises(KeyError, match="story"):
        list(sources.iter_hub_texts("roneneldan/TinyStories", None, "train", "story", "r"))


def test_iter_hub_texts_unknown_split_raises_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = FakeHub({})
    hub.install(monkeypatch)
    with pytest.raises(KeyError):
        list(sources.iter_hub_texts("roneneldan/TinyStories", None, "test", "text", "r"))
    assert not hub.events
