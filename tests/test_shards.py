"""Tests for ``syrinx.data.shards``."""

from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from syrinx.data.shards import (
    DEFAULT_SHARD_TOKENS,
    EOS_ID,
    MAX_TOKEN_ID,
    SHARD_DTYPE,
    ShardInfo,
    ShardWriter,
    decode_documents,
    list_shards,
    metadata_path,
    read_metadata,
    read_shard,
    shard_path,
    split_documents,
    write_shard_file,
)
from syrinx.tokenizer import encode_documents, train_tokenizer

_WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
TEXTS: list[str] = [
    " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(5)) + f" {i}." for i in range(60)
]
UNICODE_TEXTS = ["café \U0001f642", "", "line one\nline two\ttab", "日本語"]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_tokenizer(TEXTS, vocab_size=280, min_frequency=1)


def _docs(n: int, length: int, start: int = 2) -> list[list[int]]:
    """n documents of `length` ids each, ids never 0 (EOS)."""
    return [[start + (i * length + j) % 500 for j in range(length)] for i in range(n)]


def _stream(docs: list[list[int]]) -> list[int]:
    out: list[int] = []
    for d in docs:
        out.extend(d)
        out.append(EOS_ID)
    return out


# --- constants and paths --------------------------------------------------------------


def test_constants() -> None:
    """uint16 storage, EOS is id 0, ~100M tokens per file."""
    assert SHARD_DTYPE == np.uint16
    assert MAX_TOKEN_ID == 65535
    assert EOS_ID == 0
    assert DEFAULT_SHARD_TOKENS == 100_000_000


def test_paths(tmp_path: Path) -> None:
    """Four-digit zero-padded index; the sidecar swaps .bin for .json."""
    p = shard_path(tmp_path, "train", 7)
    assert p == tmp_path / "train_0007.bin"
    assert metadata_path(p) == tmp_path / "train_0007.json"


# --- write_shard_file / read_shard ----------------------------------------------------


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    """Tokens come back identical as a read-only uint16 memmap; the sidecar holds the
    counts, dtype, EOS id and extra metadata."""
    tokens = np.array(_stream(_docs(3, 4)), dtype=np.uint16)
    path = tmp_path / "val_0000.bin"
    info = write_shard_file(path, tokens, 3, {"split": "val", "tokenizer_sha256": "abc"})
    assert info == ShardInfo(path=path, num_tokens=15, num_docs=3)
    assert path.stat().st_size == 2 * 15

    data = read_shard(path)
    assert isinstance(data, np.memmap)
    assert data.dtype == np.uint16 and data.shape == (15,)
    assert not data.flags.writeable
    np.testing.assert_array_equal(data, tokens)

    meta = read_metadata(path)
    assert meta["num_tokens"] == 15 and meta["num_docs"] == 3
    assert meta["dtype"] == "uint16" and meta["eos_id"] == 0
    assert meta["split"] == "val" and meta["tokenizer_sha256"] == "abc"


def test_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    """After a successful write only the .bin and .json exist."""
    write_shard_file(tmp_path / "t_0000.bin", np.array([5, 0], dtype=np.uint16), 1, {})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t_0000.bin", "t_0000.json"]


def test_write_refuses_existing(tmp_path: Path) -> None:
    """An existing .bin is never overwritten."""
    path = tmp_path / "t_0000.bin"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        write_shard_file(path, np.array([5, 0], dtype=np.uint16), 1, {})
    assert path.read_bytes() == b"keep"


@pytest.mark.parametrize("change", ["truncate", "extend"])
def test_read_rejects_size_mismatch(tmp_path: Path, change: str) -> None:
    """A .bin whose size disagrees with its sidecar is rejected, whether it is shorter
    (truncated) or longer (extra data) than ``2 * num_tokens`` bytes."""
    path = tmp_path / "t_0000.bin"
    write_shard_file(path, np.array([5, 6, 0], dtype=np.uint16), 1, {})
    data = path.read_bytes()
    path.write_bytes(data[:-2] if change == "truncate" else data + b"\x07\x00")
    with pytest.raises(ValueError):
        read_shard(path)


def test_write_failure_leaves_nothing(tmp_path: Path) -> None:
    """If writing fails (here: unserializable metadata), neither the final .bin nor a
    temporary file is left behind."""
    path = tmp_path / "t_0000.bin"
    with pytest.raises(TypeError):
        write_shard_file(path, np.array([5, 0], dtype=np.uint16), 1, {"bad": object()})
    assert not path.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_read_missing_sidecar(tmp_path: Path) -> None:
    """A .bin without its sidecar is rejected."""
    path = tmp_path / "t_0000.bin"
    np.array([5, 0], dtype=np.uint16).tofile(path)
    with pytest.raises(FileNotFoundError):
        read_shard(path)


def test_file_is_raw_little_endian_uint16(tmp_path: Path) -> None:
    """The .bin is headerless little-endian uint16 (readable by np.fromfile)."""
    path = tmp_path / "t_0000.bin"
    write_shard_file(path, np.array([1, 300, 65535, 0], dtype=np.uint16), 1, {})
    assert path.read_bytes() == bytes([1, 0, 44, 1, 255, 255, 0, 0])


# --- ShardWriter ----------------------------------------------------------------------


def test_writer_single_shard(tmp_path: Path) -> None:
    """Below shard_tokens everything goes into one file at close: ids + EOS per doc."""
    docs = _docs(5, 3)
    with ShardWriter(tmp_path, "train", shard_tokens=1000, metadata={"split": "train"}) as w:
        for d in docs:
            w.add_document(d)
        assert w.total_docs == 5 and w.total_tokens == 20
        assert w.shards == []  # nothing written yet
    assert [s.path.name for s in w.shards] == ["train_0000.bin"]
    np.testing.assert_array_equal(read_shard(w.shards[0].path), _stream(docs))
    assert read_metadata(w.shards[0].path)["split"] == "train"


def test_writer_rolls_over_at_document_boundary(tmp_path: Path) -> None:
    """A file closes at the first document boundary at or after shard_tokens: every file
    ends with EOS, no document is split, and the concatenation is the full stream."""
    docs = _docs(10, 4)  # 5 tokens each with EOS
    writer = ShardWriter(tmp_path, "train", shard_tokens=12)
    for d in docs:
        writer.add_document(d)
    shards = writer.close()
    assert [s.num_tokens for s in shards] == [15, 15, 15, 5]
    assert [s.num_docs for s in shards] == [3, 3, 3, 1]
    assert [s.path.name for s in shards] == [f"train_{i:04d}.bin" for i in range(4)]
    joined = np.concatenate([np.asarray(read_shard(s.path)) for s in shards])
    np.testing.assert_array_equal(joined, _stream(docs))
    for s in shards:
        assert read_shard(s.path)[-1] == EOS_ID
        assert int((np.asarray(read_shard(s.path)) == EOS_ID).sum()) == s.num_docs


def test_writer_shard_written_when_full(tmp_path: Path) -> None:
    """A shard is written as soon as it fills, not only at close."""
    writer = ShardWriter(tmp_path, "val", shard_tokens=4)
    writer.add_document([7, 8, 9])
    assert len(writer.shards) == 1
    assert shard_path(tmp_path, "val", 0).exists()
    writer.close()


def test_writer_counts_include_eos(tmp_path: Path) -> None:
    """total_tokens counts the EOS after each document, written and buffered."""
    writer = ShardWriter(tmp_path, "train", shard_tokens=5)
    writer.add_document([3, 4, 5, 6])  # 5 tokens -> written
    writer.add_document([3])  # 2 tokens -> buffered
    assert (writer.total_tokens, writer.total_docs) == (7, 2)
    writer.close()


def test_writer_empty_document_is_just_eos(tmp_path: Path) -> None:
    """An empty document still gets its EOS."""
    with ShardWriter(tmp_path, "train") as w:
        w.add_document([])
        w.add_document([9])
    np.testing.assert_array_equal(read_shard(w.shards[0].path), [0, 9, 0])


@pytest.mark.parametrize("bad", [[-1], [65536], [5, 70000], [5, EOS_ID, 6]])
def test_writer_rejects_bad_ids(tmp_path: Path, bad: list[int]) -> None:
    """Negative ids, ids that don't fit uint16 (no silent wrap) and an EOS inside a
    document are rejected, and nothing from that document is buffered."""
    writer = ShardWriter(tmp_path, "train")
    writer.add_document([4])
    with pytest.raises(ValueError):
        writer.add_document(bad)
    assert (writer.total_tokens, writer.total_docs) == (2, 1)
    writer.close()
    np.testing.assert_array_equal(read_shard(shard_path(tmp_path, "train", 0)), [4, 0])


def test_writer_accepts_max_id(tmp_path: Path) -> None:
    """65535 is valid."""
    with ShardWriter(tmp_path, "train") as w:
        w.add_document([MAX_TOKEN_ID])
    assert int(read_shard(w.shards[0].path)[0]) == MAX_TOKEN_ID


def test_writer_close_without_documents(tmp_path: Path) -> None:
    """No documents, no files."""
    assert ShardWriter(tmp_path / "d", "train").close() == []
    assert list_shards(tmp_path / "d", "train") == []


def test_writer_close_is_idempotent(tmp_path: Path) -> None:
    """A second close writes nothing new and returns the same shards."""
    writer = ShardWriter(tmp_path, "train")
    writer.add_document([5])
    first = writer.close()
    assert writer.close() == first
    assert len(list_shards(tmp_path, "train")) == 1


def test_writer_refuses_existing_shards(tmp_path: Path) -> None:
    """A directory that already holds shards of this prefix is refused at construction;
    another prefix in the same directory is fine."""
    with ShardWriter(tmp_path, "train") as w:
        w.add_document([5])
    with pytest.raises(FileExistsError):
        ShardWriter(tmp_path, "train")
    ShardWriter(tmp_path, "val").close()


def test_writer_creates_directory(tmp_path: Path) -> None:
    """Missing parent directories are created."""
    with ShardWriter(tmp_path / "a" / "b", "train") as w:
        w.add_document([5])
    assert (tmp_path / "a" / "b" / "train_0000.bin").exists()


@pytest.mark.parametrize("bad", [0, -5])
def test_writer_rejects_non_positive_shard_tokens(tmp_path: Path, bad: int) -> None:
    """shard_tokens must be positive."""
    with pytest.raises(ValueError):
        ShardWriter(tmp_path, "train", shard_tokens=bad)


def test_writer_aborts_on_exception(tmp_path: Path) -> None:
    """If the with-block raises, buffered documents are discarded, no temporary file is
    left, finished shards stay, and the exception propagates."""
    with pytest.raises(RuntimeError), ShardWriter(tmp_path, "train", shard_tokens=3) as w:
        w.add_document([5, 6])  # 3 tokens -> shard 0 written
        w.add_document([7])  # buffered
        raise RuntimeError("boom")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train_0000.bin", "train_0000.json"]
    np.testing.assert_array_equal(read_shard(tmp_path / "train_0000.bin"), [5, 6, 0])


# --- list_shards ----------------------------------------------------------------------


def test_list_shards_sorted_and_filtered(tmp_path: Path) -> None:
    """Only finished {prefix}_*.bin files, sorted; temp files and other prefixes are
    ignored; a missing directory gives []."""
    for name in ["train_0001.bin", "train_0000.bin", "val_0000.bin", "train_0002.bin.tmp"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_shards(tmp_path, "train")] == [
        "train_0000.bin",
        "train_0001.bin",
    ]
    assert list_shards(tmp_path / "missing", "train") == []


# --- split_documents / decode_documents -----------------------------------------------


def test_split_documents() -> None:
    """Cuts at every EOS; empty documents give empty arrays."""
    docs = split_documents(np.array([3, 4, 0, 0, 5, 0], dtype=np.uint16))
    assert [d.tolist() for d in docs] == [[3, 4], [], [5]]


@pytest.mark.parametrize("bad", [[], [3, 4], [0, 5]])
def test_split_documents_requires_trailing_eos(bad: list[int]) -> None:
    """An empty stream or one not ending in EOS is not a valid shard."""
    with pytest.raises(ValueError):
        split_documents(np.array(bad, dtype=np.uint16))


def test_decode_documents_reproduces_source(tok: Tokenizer, tmp_path: Path) -> None:
    """Gate 2: writing encoded documents and decoding the shard gives back exactly the
    source texts, across several shard files."""
    texts = [*TEXTS[:20], *UNICODE_TEXTS]
    writer = ShardWriter(tmp_path, "val", shard_tokens=40)
    for ids in encode_documents(tok, texts):
        writer.add_document(ids)
    shards = writer.close()
    assert len(shards) > 1
    decoded = [d for s in shards for d in decode_documents(tok, read_shard(s.path))]
    assert decoded == texts
