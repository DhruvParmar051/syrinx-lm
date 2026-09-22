"""Tests for ``syrinx.utils.log``."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from syrinx.utils.log import setup_logging


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """Put the root logger's handlers and level back after each test."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers, root.level = handlers, level


def test_sets_level_by_name() -> None:
    """``"debug"`` (any case) sets the root level to DEBUG."""
    setup_logging("debug")
    assert logging.getLogger().level == logging.DEBUG


def test_unknown_level_raises() -> None:
    """An unknown level name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown log level"):
        setup_logging("LOUD")


def test_writes_to_file(tmp_path: Path) -> None:
    """With ``log_file``, messages land in the file (parent dirs created)."""
    log_file = tmp_path / "logs" / "run.log"
    setup_logging("INFO", log_file=log_file)
    logging.getLogger("syrinx.test").info("hello file")
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "hello file" in text
    assert "syrinx.test" in text


def test_idempotent() -> None:
    """Calling twice leaves exactly one stderr handler (no duplicate output)."""
    setup_logging()
    setup_logging()
    assert len(logging.getLogger().handlers) == 1
