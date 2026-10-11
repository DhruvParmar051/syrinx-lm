"""Tests for ``syrinx.data.loader``.

Shards here hold consecutive integers (shard i starts at i * 10_000), so a window's first
token tells exactly which shard and start it came from.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from syrinx.data.loader import RandomWindowLoader, load_split, val_batches
from syrinx.data.shards import ShardWriter

T = 8


def _shard(length: int, index: int = 0) -> np.ndarray:
    return np.arange(index * 10_000, index * 10_000 + length, dtype=np.uint16)


def _starts(x: torch.Tensor) -> list[tuple[int, int]]:
    """(shard index, local start) of each row of x."""
    first = x[:, 0].tolist()
    return [(v // 10_000, v % 10_000) for v in first]


# --- RandomWindowLoader ---------------------------------------------------------------


def test_batch_shapes_and_dtype() -> None:
    """x and y are (B, T) int64 on the requested device."""
    loader = RandomWindowLoader([_shard(100)], batch_size=4, seq_len=T, seed=0)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (4, T)
    assert x.dtype == y.dtype == torch.int64
    assert x.device.type == y.device.type == "cpu"


def test_windows_are_contiguous_and_targets_shifted() -> None:
    """Each row is stream[s:s+T]; its target is stream[s+1:s+T+1] for the same s."""
    loader = RandomWindowLoader([_shard(200)], batch_size=16, seq_len=T, seed=1)
    x, y = loader.next_batch()
    ramp = torch.arange(T)
    for row_x, row_y in zip(x, y, strict=True):
        assert torch.equal(row_x, row_x[0] + ramp)
        assert torch.equal(row_y, row_x + 1)


def test_windows_stay_in_bounds_and_cover_every_start() -> None:
    """Starts range over exactly 0 .. N - T - 1: the last valid start is reached and no
    window ever runs past the end."""
    n = T + 10  # valid starts 0..9
    loader = RandomWindowLoader([_shard(n)], batch_size=50, seq_len=T, seed=2)
    seen = set()
    for _ in range(40):
        x, y = loader.next_batch()
        seen.update(s for _, s in _starts(x))
        assert int(y.max()) <= n - 1
    assert seen == set(range(10))


def test_shard_of_exactly_t_plus_one_tokens() -> None:
    """With N = T + 1 the only valid start is 0."""
    loader = RandomWindowLoader([_shard(T + 1)], batch_size=8, seq_len=T, seed=3)
    x, y = loader.next_batch()
    assert {s for _, s in _starts(x)} == {0}
    assert int(y[0, -1]) == T


def test_windows_never_cross_shards() -> None:
    """A window comes from one shard: its tokens are consecutive within that shard."""
    shards = [_shard(T + 5, 0), _shard(T + 5, 1), _shard(T + 5, 2)]
    loader = RandomWindowLoader(shards, batch_size=64, seq_len=T, seed=4)
    x, y = loader.next_batch()
    assert torch.equal(x[:, -1] - x[:, 0], torch.full((64,), T - 1))
    assert torch.equal(y[:, -1] // 10_000, x[:, 0] // 10_000)
    assert {i for i, _ in _starts(x)} == {0, 1, 2}


def test_shards_sampled_in_proportion_to_their_starts() -> None:
    """Every valid window is equally likely, so a shard with 3x the valid starts is
    picked ~3x as often (4,000 draws; tolerance ~6 standard deviations)."""
    shards = [_shard(T + 100, 0), _shard(T + 300, 1)]
    loader = RandomWindowLoader(shards, batch_size=400, seq_len=T, seed=5)
    counts: Counter[int] = Counter()
    for _ in range(10):
        x, _ = loader.next_batch()
        counts.update(i for i, _ in _starts(x))
    share = counts[1] / sum(counts.values())
    assert abs(share - 0.75) < 0.04


def test_same_seed_same_batches() -> None:
    """Two loaders with the same seed produce identical batches."""
    a = RandomWindowLoader([_shard(500)], batch_size=4, seq_len=T, seed=7)
    b = RandomWindowLoader([_shard(500)], batch_size=4, seq_len=T, seed=7)
    for _ in range(5):
        xa, ya = a.next_batch()
        xb, yb = b.next_batch()
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_different_seed_different_batches() -> None:
    """Different seeds give different batches."""
    a = RandomWindowLoader([_shard(5000)], batch_size=8, seq_len=T, seed=7)
    b = RandomWindowLoader([_shard(5000)], batch_size=8, seq_len=T, seed=8)
    assert not torch.equal(a.next_batch()[0], b.next_batch()[0])


def test_state_dict_resume_is_exact() -> None:
    """Saving the state after k batches and restoring it into a FRESH loader (different
    seed) reproduces exactly the batches that followed."""
    loader = RandomWindowLoader([_shard(5000)], batch_size=8, seq_len=T, seed=11)
    for _ in range(3):
        loader.next_batch()
    state = loader.state_dict()
    expected = [loader.next_batch() for _ in range(4)]

    resumed = RandomWindowLoader([_shard(5000)], batch_size=8, seq_len=T, seed=999)
    resumed.load_state_dict(state)
    for xe, ye in expected:
        x, y = resumed.next_batch()
        assert torch.equal(x, xe) and torch.equal(y, ye)


def test_state_dict_is_a_snapshot() -> None:
    """The returned state does not change when the loader keeps going."""
    loader = RandomWindowLoader([_shard(5000)], batch_size=8, seq_len=T, seed=12)
    state = loader.state_dict()
    first = loader.next_batch()[0]
    loader.load_state_dict(state)
    assert torch.equal(loader.next_batch()[0], first)


@pytest.mark.parametrize(
    ("shards", "batch_size", "seq_len"),
    [
        ([], 4, T),
        ([np.arange(T, dtype=np.uint16)], 4, T),  # N = T: no valid window
        ([np.arange(100, dtype=np.uint16)], 0, T),
        ([np.arange(100, dtype=np.uint16)], 4, 0),
    ],
)
def test_invalid_arguments_raise(shards: list[np.ndarray], batch_size: int, seq_len: int) -> None:
    """No shards, a shard shorter than T + 1, or non-positive B / T raise ValueError."""
    with pytest.raises(ValueError):
        RandomWindowLoader(shards, batch_size=batch_size, seq_len=seq_len, seed=0)


def test_works_on_memmapped_shards(tmp_path: Path) -> None:
    """Shards written by ShardWriter and opened with load_split feed the loader."""
    with ShardWriter(tmp_path / "train", "train", shard_tokens=50) as w:
        for i in range(30):
            w.add_document(list(range(1 + i, 6 + i)))
    shards = load_split(tmp_path / "train", "train")
    assert len(shards) > 1
    x, y = RandomWindowLoader(shards, batch_size=4, seq_len=T, seed=0).next_batch()
    assert x.shape == y.shape == (4, T) and x.dtype == y.dtype == torch.int64


@pytest.mark.mps
def test_batch_on_mps(mps_device: torch.device) -> None:
    """Batches can be placed on MPS as int64."""
    loader = RandomWindowLoader([_shard(100)], batch_size=2, seq_len=T, seed=0, device=mps_device)
    x, y = loader.next_batch()
    assert x.device.type == "mps" and y.device.type == "mps"
    assert x.dtype == torch.int64


# --- val_batches ----------------------------------------------------------------------


def test_val_batches_fixed_non_overlapping_windows() -> None:
    """Windows start at k*T in each shard (K = (N - 1) // T), in order, batched across
    shards; the last batch may be smaller."""
    shards = [_shard(3 * T + 1, 0), _shard(2 * T + 5, 1)]  # K = 3 and K = 2
    batches = list(val_batches(shards, batch_size=2, seq_len=T))
    assert [b[0].shape[0] for b in batches] == [2, 2, 1]
    xs = torch.cat([b[0] for b in batches])
    ys = torch.cat([b[1] for b in batches])
    assert _starts(xs) == [(0, 0), (0, T), (0, 2 * T), (1, 0), (1, T)]
    assert torch.equal(ys, xs + 1)
    assert xs.dtype == torch.int64


def test_val_batches_identical_every_call() -> None:
    """Evaluation sees exactly the same tokens every time."""
    shards = [_shard(10 * T + 3)]
    first = [b[0] for b in val_batches(shards, 3, T)]
    second = [b[0] for b in val_batches(shards, 3, T)]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_val_batches_cover_stream_once() -> None:
    """The windows tile stream[0 : K*T] exactly, and targets reach stream[K*T]."""
    n = 4 * T + 1
    xs = torch.cat([b[0] for b in val_batches([_shard(n)], 3, T)])
    assert xs.flatten().tolist() == list(range(4 * T))


def test_val_batches_skip_too_short_shard() -> None:
    """A shard with no full window contributes nothing; others still do."""
    batches = list(val_batches([_shard(T, 0), _shard(T + 1, 1)], 4, T))
    assert len(batches) == 1 and _starts(batches[0][0]) == [(1, 0)]


def test_val_batches_no_windows_raises() -> None:
    """If no shard has a full window, ValueError."""
    with pytest.raises(ValueError):
        list(val_batches([_shard(T)], 4, T))


# --- load_split -----------------------------------------------------------------------


def test_load_split_opens_all_shards_in_order(tmp_path: Path) -> None:
    """Every shard of the split, as read-only uint16 arrays, in file order."""
    with ShardWriter(tmp_path, "val", shard_tokens=4) as w:
        for i in range(5):
            w.add_document([10 + i, 20 + i, 30 + i])
    shards = load_split(tmp_path, "val")
    assert len(shards) == 5
    assert [int(s[0]) for s in shards] == [10, 11, 12, 13, 14]
    assert all(s.dtype == np.uint16 for s in shards)


def test_load_split_empty_raises(tmp_path: Path) -> None:
    """A split with no shards raises ValueError."""
    with pytest.raises(ValueError):
        load_split(tmp_path, "train")
