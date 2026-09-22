"""Tests for ``syrinx.utils.metrics``."""

from pathlib import Path

import numpy as np
import pytest
import torch

from syrinx.utils.metrics import JsonlWriter, read_jsonl


def test_roundtrip(tmp_path: Path) -> None:
    """Records written are read back in order, unchanged."""
    path = tmp_path / "m.jsonl"
    records = [{"step": 1, "loss": 10.4, "tag": "train"}, {"step": 2, "loss": 9.9, "ok": True}]
    with JsonlWriter(path) as log:
        for record in records:
            log.write(record)
    assert read_jsonl(path) == records


def test_converts_tensors_and_numpy(tmp_path: Path) -> None:
    """Scalar tensors and numpy scalars are stored as plain numbers."""
    path = tmp_path / "m.jsonl"
    with JsonlWriter(path) as log:
        log.write({"loss": torch.tensor(2.5), "tokens": np.int64(4096), "lr": np.float32(0.5)})
    assert read_jsonl(path) == [{"loss": 2.5, "tokens": 4096, "lr": 0.5}]


def test_non_finite_values_are_logged(tmp_path: Path) -> None:
    """NaN and infinities are kept (as strings), never dropped or written as invalid JSON."""
    path = tmp_path / "m.jsonl"
    with JsonlWriter(path) as log:
        log.write({"a": float("nan"), "b": float("inf"), "c": torch.tensor(float("-inf"))})
    assert read_jsonl(path) == [{"a": "nan", "b": "inf", "c": "-inf"}]
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_nested_and_paths(tmp_path: Path) -> None:
    """Nested mappings, lists and Paths are converted."""
    path = tmp_path / "m.jsonl"
    with JsonlWriter(path) as log:
        log.write({"mem": {"gib": 1.5}, "shape": (8, 1024), "ckpt": Path("a/b.pt")})
    assert read_jsonl(path) == [{"mem": {"gib": 1.5}, "shape": [8, 1024], "ckpt": "a/b.pt"}]


def test_each_record_flushed_immediately(tmp_path: Path) -> None:
    """A record is on disk before the writer is closed (a crash keeps it)."""
    path = tmp_path / "m.jsonl"
    log = JsonlWriter(path)
    log.write({"step": 1})
    assert read_jsonl(path) == [{"step": 1}]
    log.close()


def test_default_mode_refuses_overwrite(tmp_path: Path) -> None:
    """Mode ``"x"`` (default) raises if the file exists; ``"a"`` appends."""
    path = tmp_path / "m.jsonl"
    with JsonlWriter(path) as log:
        log.write({"step": 1})
    with pytest.raises(FileExistsError):
        JsonlWriter(path)
    with JsonlWriter(path, mode="a") as log:
        log.write({"step": 2})
    assert [r["step"] for r in read_jsonl(path)] == [1, 2]


def test_invalid_mode(tmp_path: Path) -> None:
    """Modes other than x/a (e.g. "w", which would truncate) raise ValueError."""
    with pytest.raises(ValueError, match="mode"):
        JsonlWriter(tmp_path / "m.jsonl", mode="w")  # type: ignore[arg-type]


def test_rejects_unloggable(tmp_path: Path) -> None:
    """Non-scalar tensors, arrays and arbitrary objects raise TypeError."""
    with JsonlWriter(tmp_path / "m.jsonl") as log:
        with pytest.raises(TypeError, match="scalar"):
            log.write({"x": torch.zeros(3)})
        with pytest.raises(TypeError, match="Arrays"):
            log.write({"x": np.zeros(3)})
        with pytest.raises(TypeError, match="Cannot log"):
            log.write({"x": object()})


def test_write_after_close_raises(tmp_path: Path) -> None:
    """Writing to a closed writer raises RuntimeError; close is idempotent."""
    log = JsonlWriter(tmp_path / "m.jsonl")
    log.close()
    log.close()
    with pytest.raises(RuntimeError, match="closed"):
        log.write({"step": 1})


def test_read_reports_bad_line(tmp_path: Path) -> None:
    """A corrupt line raises ValueError naming the line number; blank lines are skipped."""
    path = tmp_path / "m.jsonl"
    path.write_text('{"step": 1}\n\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":3:"):
        read_jsonl(path)


def test_read_rejects_non_object(tmp_path: Path) -> None:
    """A line holding a JSON array (not an object) raises ValueError."""
    path = tmp_path / "m.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_jsonl(path)
