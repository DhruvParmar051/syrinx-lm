"""Logging configuration for Syrinx entry points.

Library modules only create ``logging.getLogger(__name__)`` and never configure
handlers. Scripts call :func:`setup_logging` once at startup.
"""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger: stderr, plus an optional file.

    Replaces any handlers configured earlier, so calling it twice doesn't duplicate
    output.

    Args:
        level: Logging level name (``"DEBUG"``, ``"INFO"``, ...) or number.
        log_file: If given, also append log lines to this file (parents created).

    Raises:
        ValueError: If ``level`` is an unknown level name.
    """
    if isinstance(level, str):
        resolved = logging.getLevelNamesMapping().get(level.upper())
        if resolved is None:
            raise ValueError(f"Unknown log level {level!r}.")
        level = resolved

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers, force=True
    )
