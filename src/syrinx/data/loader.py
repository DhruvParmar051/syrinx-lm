"""Batches of (input, target) token windows from memory-mapped shards.

Train: uniform random windows over every valid start in every shard (a window never
crosses a shard boundary), drawn from a seeded ``np.random.Generator`` whose state can be
saved and restored for exact resume. Val: fixed, non-overlapping windows, identical on
every evaluation.

For a shard of N tokens and window length T, the valid starts are ``0 .. N - T - 1``
(``N - T`` of them): each window reads ``T + 1`` tokens, ``x = w[:-1]``, ``y = w[1:]``.
"""

import copy
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from syrinx.data.shards import list_shards, read_shard

logger = logging.getLogger(__name__)


class RandomWindowLoader:
    """Infinite stream of random training batches.

    Sampling, per batch: ONE call ``rng.integers(0, total_starts, size=batch_size)``
    gives global start indices over the concatenated valid starts of all shards (shard
    ``i`` contributes ``len(shard_i) - seq_len`` starts); each is mapped to (shard,
    local start). This makes every valid window equally likely and the sequence of
    batches a pure function of the generator state.
    """

    def __init__(
        self,
        shards: Sequence[np.ndarray],
        batch_size: int,
        seq_len: int,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Validate the shards and create the generator ``np.random.default_rng(seed)``.

        Args:
            shards: 1-D token arrays (e.g. from :func:`load_split`).
            batch_size: B, windows per batch; positive.
            seq_len: T, tokens per window input; positive.
            seed: Generator seed.
            device: Where batches are placed.

        Raises:
            ValueError: If there are no shards, ``batch_size``/``seq_len`` are not
                positive, or any shard has fewer than ``seq_len + 1`` tokens.
        """

        if not shards:
            raise ValueError("No shards given.")
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError(
                f"batch size and seq_len must be positive, got {batch_size} and {seq_len}"
            )
        for index, shard in enumerate(shards):
            if len(shard) < seq_len + 1:
                raise ValueError(
                    f"Shard {index} has {len(shard)} tokens; need at least seq_len + 1 = "
                    f"{seq_len + 1}."
                )
        self._shards = list(shards)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._device = torch.device(device=device)
        starts = np.array([len(shard) - seq_len for shard in shards], dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(starts)])
        self._total_starts = int(self._offsets[-1])
        self._rng = np.random.default_rng(seed=seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one batch.

        Returns:
            ``(x, y)``, each ``(B, T)`` ``int64`` on ``device``, with
            ``y[b] = stream[s_b + 1 : s_b + T + 1]`` and ``x[b] = stream[s_b : s_b + T]``
            for the same start ``s_b``.
        """
        global_starts = self._rng.integers(0, self._total_starts, size=self._batch_size)
        shard_ids = np.searchsorted(self._offsets, global_starts, side="right") - 1
        local_starts = global_starts - self._offsets[shard_ids]
        windows = np.stack(
            [
                np.asarray(self._shards[i][s : s + self._seq_len + 1])
                for i, s in zip(shard_ids, local_starts, strict=True)
            ]
        ).astype(np.int64)
        batch = torch.from_numpy(windows)
        return batch[:, :-1].to(self._device), batch[:, 1:].to(self._device)

    def state_dict(self) -> dict[str, Any]:
        """Generator state for checkpointing: ``{"bit_generator": rng.bit_generator.state}``."""
        return {"bit_generator": copy.deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state from :meth:`state_dict`; the next batches are then exactly the
        ones that followed when the state was saved."""
        self._rng.bit_generator.state = copy.deepcopy(state["bit_generator"])


def val_batches(
    shards: Sequence[np.ndarray],
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed, non-overlapping validation windows, in a deterministic order.

    For each shard in order, windows start at ``k * seq_len`` for ``k = 0 .. K - 1`` with
    ``K = (len(shard) - 1) // seq_len``; windows are grouped into batches of
    ``batch_size`` across shard boundaries (a window itself never crosses one); the last
    batch may be smaller.

    Args:
        shards: 1-D token arrays.
        batch_size: Windows per batch; positive.
        seq_len: T; positive.
        device: Where batches are placed.

    Yields:
        ``(x, y)``, each ``(b, T)`` ``int64`` with ``b <= batch_size``.

    Raises:
        ValueError: If the shards give no window at all (raised on first iteration).
    """
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(
            f"batch_size and seq_len must be positive, got {batch_size} and {seq_len}."
        )
    windows = [(shard, k * seq_len) for shard in shards for k in range((len(shard) - 1) // seq_len)]
    if not windows:
        raise ValueError("No shard has a full validation window.")
    target = torch.device(device)
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        stacked = np.stack([np.asarray(s[o : o + seq_len + 1]) for s, o in chunk])
        batch = torch.from_numpy(stacked.astype(np.int64))
        yield batch[:, :-1].to(target), batch[:, 1:].to(target)


def load_split(directory: Path, prefix: str) -> list[np.ndarray]:
    """Open every shard of one split read-only (``list_shards`` + ``read_shard``).

    Args:
        directory: Split directory, e.g. ``data_cache/tokens/fineweb-edu/train``.
        prefix: Shard prefix, normally the split name.

    Returns:
        Memory maps, in file order.

    Raises:
        ValueError: If the directory has no shards.
    """
    paths = list_shards(directory, prefix)
    if not paths:
        raise ValueError(f"No {prefix} shards in {directory}.")
    return [read_shard(path) for path in paths]
