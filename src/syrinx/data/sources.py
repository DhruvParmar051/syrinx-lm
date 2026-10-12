"""Raw text from pinned Hub parquet files, downloaded once and read from the local cache.

Streaming a split with ``load_dataset(..., streaming=True)`` reads the remote files over
HTTP range requests; one stalled connection stalls the whole build, and nothing is kept.
Instead each parquet file of a split is fetched with ``hf_hub_download`` (cached,
resumable, retried) at the pinned revision, then read locally with ``pyarrow``. Files are
fetched lazily, one at a time, so a build that stops early never downloads the rest.

Order is the same as Hub streaming: files sorted by repo path, rows in file order.
"""

import fnmatch
import logging
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)

SPLIT_FILES: dict[tuple[str, str | None, str], str] = {
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"): "sample/10BT/*.parquet",
    ("roneneldan/TinyStories", None, "train"): "data/train-*.parquet",
    ("roneneldan/TinyStories", None, "validation"): "data/validation-*.parquet",
}
"""``(dataset, subset, split)`` -> glob of its parquet files inside the repo. Checked
against the pinned revisions: the sorted matches equal the files ``datasets`` streams."""

DEFAULT_BATCH_ROWS = 1024
"""Rows per ``pyarrow`` record batch when reading a file."""


def split_file_pattern(dataset: str, subset: str | None, split: str) -> str:
    """The repo glob for one split, from ``SPLIT_FILES``.

    Raises:
        KeyError: If the split is not listed (message names dataset, subset and split).
    """
    key = (dataset, subset, split)
    if key not in SPLIT_FILES:
        raise KeyError(
            f"No parquet file pattern for dataset={dataset!r}, subset={subset!r}, "
            f"split={split!r}; add it to SPLIT_FILES."
        )
    return SPLIT_FILES[key]


def list_split_files(dataset: str, revision: str, pattern: str) -> list[str]:
    """Repo paths matching ``pattern`` at ``revision``, sorted.

    Calls ``list_repo_files(dataset, repo_type="dataset", revision=revision)`` (the
    module-level name, so tests can replace it) and keeps paths where
    ``fnmatch.fnmatchcase(path, pattern)`` holds.

    Raises:
        ValueError: If no file matches (message names the pattern and revision).
    """
    names = list_repo_files(dataset, repo_type="dataset", revision=revision)
    matched = sorted(name for name in names if fnmatch.fnmatchcase(name=name, pat=pattern))
    if not matched:
        raise ValueError(f"No files in {dataset} at revision {revision!r} match {pattern!r}.")
    return matched


def fetch_file(dataset: str, revision: str, filename: str) -> Path:
    """Local path of one repo file, downloading it on first use.

    Calls ``hf_hub_download(repo_id=dataset, filename=filename, repo_type="dataset",
    revision=revision)`` (module-level name) and returns its result as a ``Path``.
    """
    local = hf_hub_download(
        repo_id=dataset, filename=filename, repo_type="dataset", revision=revision
    )
    return Path(local)


def iter_parquet_texts(
    path: Path, text_field: str, batch_rows: int = DEFAULT_BATCH_ROWS
) -> Iterator[str]:
    """Every value of column ``text_field`` in a local parquet file, in row order.

    Reads only that column, ``batch_rows`` rows at a time
    (``pq.ParquetFile(path).iter_batches(batch_size=..., columns=[text_field])``), so
    memory stays bounded for multi-GB files.

    Raises:
        KeyError: If the file has no ``text_field`` column (message names the column),
            raised before anything is yielded.
    """
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    if text_field not in columns:
        raise KeyError(f"{path} has no {text_field!r} column; columns: {columns}")

    for batch in parquet.iter_batches(batch_size=batch_rows, columns=[text_field]):
        yield from batch.column(0).to_pylist()


def iter_hub_texts(
    dataset: str, subset: str | None, split: str, text_field: str, revision: str
) -> Iterator[str]:
    """Raw texts of one split: :func:`split_file_pattern`, then :func:`list_split_files`,
    then for each file in order :func:`fetch_file` and :func:`iter_parquet_texts`.

    Lazy: file ``k + 1`` is fetched only after every row of file ``k`` was yielded. Log
    each file as it is opened. Call the helpers by their module-level names (tests replace
    them).
    """
    pattern = split_file_pattern(dataset=dataset, subset=subset, split=split)
    filenames = list_split_files(dataset=dataset, revision=revision, pattern=pattern)
    for index, filename in enumerate(filenames):
        logger.info(
            "Reading %s %s file %d/%d: %s", dataset, split, index + 1, len(filenames), filename
        )
        path = fetch_file(dataset=dataset, revision=revision, filename=filename)
        yield from iter_parquet_texts(path=path, text_field=text_field)
