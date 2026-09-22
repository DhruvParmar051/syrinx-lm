"""Append-only JSON Lines metrics files (one JSON object per line).

Every number reported in ``experiments/`` should trace back to a file written here.
Each record is flushed as soon as it is written, so a crashed run keeps every line
up to the crash.
"""

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, Self

import numpy as np
import torch

logger = logging.getLogger(__name__)

NON_FINITE = {"nan", "inf", "-inf"}
"""Strings used to store non-finite floats, which strict JSON cannot represent."""


def _to_jsonable(value: Any) -> Any:
    """Convert ``value`` into something ``json.dumps(allow_nan=False)`` accepts.

    Scalar tensors and numpy scalars become Python numbers; non-finite floats become
    ``"nan"``, ``"inf"`` or ``"-inf"`` (a NaN loss must still be logged); ``Path``
    becomes ``str``; mappings and sequences are converted recursively.

    Raises:
        TypeError: For non-scalar tensors/arrays or unsupported types.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"Only scalar tensors can be logged, got shape {tuple(value.shape)}.")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, np.ndarray):
        raise TypeError(f"Arrays cannot be logged, got shape {value.shape}.")

    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    raise TypeError(f"Cannot log value of type {type(value).__name__}.")


class JsonlWriter:
    """Write one JSON object per line, flushing after every record.

    The default mode ``"x"`` refuses to overwrite an existing file, so two runs can't
    silently share a log. Use ``"a"`` to continue a log when resuming a run.

    Example::

        with JsonlWriter(run_dir / "metrics.jsonl") as log:
            log.write({"step": 1, "loss": loss, "lr": lr})
    """

    def __init__(self, path: Path, mode: Literal["x", "a"] = "x") -> None:
        """Open ``path`` for writing (parents are created).

        Args:
            path: Destination ``.jsonl`` file.
            mode: ``"x"`` to create a new file (error if it exists) or ``"a"`` to append.

        Raises:
            ValueError: If ``mode`` is not ``"x"`` or ``"a"``.
            FileExistsError: If ``mode == "x"`` and ``path`` exists.
        """
        if mode not in ("x", "a"):
            raise ValueError(f"mode must be 'x' or 'a', got {mode!r}.")
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] | None = path.open(mode, encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        """Serialize ``record`` as one line and flush it.

        Args:
            record: Flat or nested mapping of loggable values.

        Raises:
            RuntimeError: If the writer is closed.
            TypeError: If a value cannot be logged.
        """
        if self._file is None:
            raise RuntimeError(f"JsonlWriter for {self.path} is closed.")
        line = json.dumps(_to_jsonable(record), allow_nan=False)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Self:
        """Return the writer."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the file."""
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every record from a JSON Lines file.

    Non-finite floats come back as the strings in :data:`NON_FINITE`.

    Args:
        path: File written by :class:`JsonlWriter`.

    Returns:
        The records, in file order.

    Raises:
        ValueError: If a line is not a JSON object; the message gives the line number.
    """
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(record, dict):
                # Bad file content, not a bad argument type, so ValueError.
                raise ValueError(f"{path}:{line_no}: expected a JSON object.")  # noqa: TRY004
            records.append(record)
    return records
