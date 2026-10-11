"""Tokenized shards: flat ``uint16`` files of ``ids(doc) + EOS`` per document.

Layout: ``<directory>/<prefix>_NNNN.bin`` (raw little-endian ``uint16``, no header) plus a
``<prefix>_NNNN.json`` sidecar with counts and provenance. A document never spans two
files, and every file ends with EOS. Files are written under a temporary name and
renamed only when complete, so a ``.bin`` file is never partial.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
from tokenizers import Tokenizer

from syrinx.tokenizer import decode

logger = logging.getLogger(__name__)

SHARD_DTYPE = np.uint16
"""On-disk token type. Valid because the vocabulary (32,768) is below 65,536."""

MAX_TOKEN_ID = 65535
"""Largest id a ``uint16`` can hold."""

EOS_ID = 0
"""Id of ``<|endoftext|>``; written after every document."""

DEFAULT_SHARD_TOKENS = 100_000_000
"""Target tokens per file (~200 MB). A file closes at the first document boundary at or
after this many tokens."""


@dataclass(frozen=True)
class ShardInfo:
    """One finished shard file."""

    path: Path
    num_tokens: int
    num_docs: int


def shard_path(directory: Path, prefix: str, index: int) -> Path:
    """Path of shard ``index``: ``directory / f"{prefix}_{index:04d}.bin"``."""
    return directory / f"{prefix}_{index:04d}.bin"


def metadata_path(bin_path: Path) -> Path:
    """The JSON sidecar of a shard: the same path with suffix ``.json``."""
    return bin_path.with_suffix(".json")


class ShardWriter:
    """Append documents to a sequence of shard files for one split.

    Use as a context manager: on a normal exit it calls :meth:`close`; if the block
    raises it calls :meth:`abort` (temporary files are removed; finished shards stay).
    """

    def __init__(
        self,
        directory: Path,
        prefix: str,
        shard_tokens: int = DEFAULT_SHARD_TOKENS,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Prepare a writer; nothing is written until the first shard fills or closes.

        Args:
            directory: Output directory (created with parents if missing).
            prefix: File name prefix, normally the split name ("train" / "val").
            shard_tokens: Target tokens per file; must be positive.
            metadata: Extra JSON-serializable fields copied into every sidecar
                (tokenizer SHA, dataset, revision, salt, ...).

        Raises:
            ValueError: If ``shard_tokens`` is not positive.
            FileExistsError: If ``directory`` already holds any ``{prefix}_*.bin``.
        """
        if shard_tokens <= 0:
            raise ValueError(f"shard_tokens must be positive, got {shard_tokens}.")
        if list_shards(directory=directory, prefix=prefix):
            raise FileExistsError(f"{directory} already holds {prefix} shards.")
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._prefix = prefix
        self._shard_tokens = shard_tokens
        self._metadata = dict(metadata or {})
        self._buffer: list[int] = []
        self._buffer_docs = 0
        self._shards: list[ShardInfo] = []
        self._closed = False

    def _flush(self) -> None:
        """Write the buffer as the next shard and empty it."""
        path = shard_path(self._directory, self._prefix, len(self._shards))
        tokens = np.array(self._buffer, dtype=SHARD_DTYPE)
        self._shards.append(
            write_shard_file(
                path=path,
                tokens=tokens,
                num_docs=self._buffer_docs,
                metadata=self._metadata,
            )
        )
        self._buffer = []
        self._buffer_docs = 0

    def add_document(self, ids: Sequence[int]) -> None:
        """Buffer one document's ids followed by ``EOS_ID``.

        After appending, if the buffer holds at least ``shard_tokens`` tokens, write it
        out as the next shard (so a shard always ends on a document boundary).

        Args:
            ids: The document's token ids WITHOUT EOS (may be empty).

        Raises:
            ValueError: If any id is outside ``[0, MAX_TOKEN_ID]`` or equals ``EOS_ID``
                (an EOS inside a document would create a fake boundary). Nothing is
                buffered in that case.
        """
        ids = list(ids)
        for token_id in ids:
            if token_id < 0 or token_id > MAX_TOKEN_ID or token_id == EOS_ID:
                raise ValueError(f"Invalid token id {token_id} in document.")
        self._buffer.extend(ids)
        self._buffer.append(EOS_ID)
        self._buffer_docs += 1
        if len(self._buffer) >= self._shard_tokens:
            self._flush()

    def close(self) -> list[ShardInfo]:
        """Write any buffered documents as a final shard and return all shards.

        Calling it again returns the same list without writing anything.

        Returns:
            Every shard written by this writer, in order (empty if no documents).
        """
        if not self._closed:
            if self._buffer:
                self._flush()
            self._closed = True
        return list(self._shards)

    def abort(self) -> None:
        """Discard the buffer and delete this writer's temporary files."""
        self._buffer = []
        self._buffer_docs = 0
        for temp in self._directory.glob(f"{self._prefix}_*.bin.tmp"):
            temp.unlink()
        self._closed = True

    @property
    def shards(self) -> list[ShardInfo]:
        """Shards finished so far."""
        return list(self._shards)

    @property
    def total_tokens(self) -> int:
        """Tokens added so far (written plus buffered), EOS included."""
        return sum(s.num_tokens for s in self._shards) + len(self._buffer)

    @property
    def total_docs(self) -> int:
        """Documents added so far (written plus buffered)."""
        return sum(s.num_docs for s in self._shards) + self._buffer_docs

    def __enter__(self) -> Self:
        """Return the writer."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """``close()`` on success, ``abort()`` if the block raised (the exception
        propagates)."""
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_shard_file(
    path: Path, tokens: np.ndarray, num_docs: int, metadata: dict[str, Any]
) -> ShardInfo:
    """Write one shard atomically: tokens to ``path`` + ".tmp", the JSON sidecar, then
    rename the temporary file to ``path``.

    The sidecar holds ``num_tokens``, ``num_docs``, ``dtype`` ("uint16"), ``eos_id`` and
    every key of ``metadata``. If any step fails, delete the temporary file (and a
    partially written sidecar) before re-raising, so nothing is left under either name.

    Args:
        path: Final ``.bin`` path; must not exist.
        tokens: 1-D ``uint16`` array ending with ``EOS_ID``.
        num_docs: Documents in ``tokens`` (equals its EOS count).
        metadata: Extra sidecar fields.

    Returns:
        The finished shard.

    Raises:
        FileExistsError: If ``path`` exists.
    """
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    temp = path.with_name(path.name + ".tmp")
    sidecar = {
        **metadata,
        "num_tokens": len(tokens),
        "num_docs": int(num_docs),
        "dtype": "uint16",
        "eos_id": EOS_ID,
    }
    try:
        np.asarray(tokens, dtype="<u2").tofile(temp)
        metadata_path(path).write_text(
            json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
        )
        temp.rename(path)
    except BaseException:
        temp.unlink(missing_ok=True)

        metadata_path(path).unlink(missing_ok=True)
        raise
    return ShardInfo(path=path, num_tokens=len(tokens), num_docs=int(num_docs))


def list_shards(directory: Path, prefix: str) -> list[Path]:
    """Finished shard files ``{prefix}_*.bin`` in ``directory``, sorted by name
    (temporary files are never listed).

    Args:
        directory: Shard directory.
        prefix: File name prefix.

    Returns:
        Sorted paths (empty if none or the directory doesn't exist).
    """
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{prefix}_*.bin"))


def read_shard(path: Path) -> np.ndarray:
    """Open a shard read-only as a ``uint16`` memory map, checked against its sidecar.

    Args:
        path: A ``.bin`` file written by :class:`ShardWriter`.

    Returns:
        ``np.memmap`` of shape ``(num_tokens,)``, dtype ``uint16``, mode "r".

    Raises:
        FileNotFoundError: If the file or its sidecar is missing.
        ValueError: If the file size is not ``2 * num_tokens`` bytes.
    """
    meta = read_metadata(path)
    num_tokens = int(meta["num_tokens"])
    size = path.stat().st_size
    if size != 2 * num_tokens:
        raise ValueError(f"{path} has {size} bytes, expected {2 * num_tokens}.")
    return np.memmap(path, dtype=SHARD_DTYPE, mode="r", shape=(num_tokens,))


def split_documents(tokens: np.ndarray) -> list[np.ndarray]:
    """Cut a shard's token stream at each EOS into documents (EOS removed).

    Args:
        tokens: 1-D token array that ends with ``EOS_ID``.

    Returns:
        One array per document, in order; an empty document gives an empty array.

    Raises:
        ValueError: If ``tokens`` is empty or does not end with ``EOS_ID``.
    """
    if len(tokens) == 0 or int(tokens[-1]) != EOS_ID:
        raise ValueError("A shard must be non-empty and end with EOS.")
    ends = np.flatnonzero(np.asarray(tokens) == EOS_ID)
    documents: list[np.ndarray] = []
    start = 0
    for end in ends:
        documents.append(np.asarray(tokens[start:end]))
        start = int(end) + 1

    return documents


def decode_documents(tokenizer: Tokenizer, tokens: np.ndarray) -> list[str]:
    """Decode every document in a shard (Gate 2: this must reproduce the source text).

    Args:
        tokenizer: The tokenizer the shard was built with.
        tokens: A shard's token stream.

    Returns:
        Decoded documents, in order, via :func:`split_documents` and ``decode``.
    """
    return [decode(tokenizer=tokenizer, ids=doc.tolist()) for doc in split_documents(tokens=tokens)]


def read_metadata(bin_path: Path) -> dict[str, Any]:
    """Load a shard's JSON sidecar.

    Args:
        bin_path: A shard's ``.bin`` path.

    Returns:
        The sidecar as a dict.

    Raises:
        FileNotFoundError: If the sidecar is missing.
    """
    return json.loads(metadata_path(bin_path=bin_path).read_text(encoding="utf-8"))
