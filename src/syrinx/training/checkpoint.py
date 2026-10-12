"""Checkpoints that resume a run exactly: model, optimizer, step, RNG states, data position.

The LR needs no state (``optim.lr_at`` is a function of the step) and the data position is
the train loader's generator state, so a resumed run draws the same batches, uses the same
learning rates and produces the same losses as an uninterrupted one.

Files are ``<dir>/step_NNNNNNN.pt`` (7-digit step = optimizer steps completed), written
atomically (temp file + ``os.replace``) and never overwritten. Everything stored is plain
tensors, numbers, strings, lists, tuples and dicts, so loading uses
``torch.load(weights_only=True)``, which cannot execute code from the file.
"""

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 1
"""Bump when the payload layout changes; loading another version is an error."""

_CHECKPOINT_NAME = re.compile(r"^step_(\d{7})\.pt$")


@dataclass(frozen=True)
class Checkpoint:
    """What :func:`load_checkpoint` returns besides the state it loads in place.

    Attributes:
        step: Optimizer steps completed when saved; training resumes at this step.
        loader_state: ``RandomWindowLoader.state_dict()`` at save time.
        rng_state: :func:`capture_rng_state` at save time (pass to
            :func:`restore_rng_state`).
        config: The run config dict saved with it (``config_to_dict``).
    """

    step: int
    loader_state: dict[str, Any]
    rng_state: dict[str, Any]
    config: dict[str, Any]


def checkpoint_path(directory: Path, step: int) -> Path:
    """``directory / f"step_{step:07d}.pt"``.

    Raises:
        ValueError: If ``step`` is negative.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")
    return Path(directory) / f"step_{step:07d}.pt"


def capture_rng_state() -> dict[str, Any]:
    """Every global RNG the run uses.

    Keys: ``"python"`` (``random.getstate()``), ``"numpy"`` (``np.random.get_state()`` with
    its key array converted to a list, so the payload has no NumPy objects:
    ``(name, key.tolist(), pos, has_gauss, cached_gaussian)``), ``"torch_cpu"``
    (``torch.get_rng_state()``), ``"torch_mps"`` (``torch.mps.get_rng_state()`` if
    ``torch.backends.mps.is_available()``, else None).
    """
    name, key, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, key.tolist(), pos, has_gauss, cached),
        "torch_cpu": torch.get_rng_state(),
        "torch_mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Inverse of :func:`capture_rng_state` (NumPy key back to a ``uint32`` array). The MPS
    state is restored only when it was saved and MPS is available."""
    random.setstate(state=state["python"])
    name, key, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(key, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loader_state: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Write one checkpoint atomically.

    Payload: ``{"version": CHECKPOINT_VERSION, "step", "model": model.state_dict(),
    "optimizer": optimizer.state_dict(), "loader": loader_state, "rng":
    capture_rng_state(), "config": config}``. Creates the parent directory; writes
    ``torch.save`` to ``<name>.tmp`` beside ``path`` and then ``os.replace``s it, so a
    crash mid-write never leaves a truncated ``.pt``.

    Raises:
        FileExistsError: If ``path`` exists (checked before writing anything).
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint {path} already exists; refusing to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader": loader_state,
        "rng": capture_rng_state(),
        "config": config,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> Checkpoint:
    """Load model and optimizer state in place and return the rest.

    ``torch.load(path, map_location="cpu", weights_only=True)``; check the version;
    ``model.load_state_dict(..., strict=True)`` (copies into the model's existing tensors,
    on its device); ``optimizer.load_state_dict(...)`` (moves its state to each parameter's
    device). Does NOT restore RNG state; call :func:`restore_rng_state` with
    ``checkpoint.rng_state`` once everything else is built.

    Raises:
        ValueError: If the version differs from ``CHECKPOINT_VERSION``.
        RuntimeError: From ``load_state_dict`` if the model's parameters don't match.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"{path} has checkpoint version {version}; expected {CHECKPOINT_VERSION}.")
    model.load_state_dict(payload["model"], strict=True)

    optimizer.load_state_dict(payload["optimizer"])
    return Checkpoint(
        step=payload["step"],
        loader_state=payload["loader"],
        rng_state=payload["rng"],
        config=payload["config"],
    )


def list_checkpoints(directory: Path) -> list[Path]:
    """Files in ``directory`` named exactly ``step_NNNNNNN.pt``, sorted by step (temp files
    and anything else ignored). A missing directory gives ``[]``."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _CHECKPOINT_NAME.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


def latest_checkpoint(directory: Path) -> Path | None:
    """The highest-step checkpoint in ``directory``, or None."""
    found = list_checkpoints(directory=directory)
    return found[-1] if found else None


def prune_checkpoints(directory: Path, keep_last: int, keep_every: int | None = None) -> list[Path]:
    """Delete old checkpoints, keeping the ``keep_last`` newest and every checkpoint whose
    step is a multiple of ``keep_every`` (milestones).

    Returns:
        The deleted paths, oldest first.

    Raises:
        ValueError: If ``keep_last < 1`` (the newest checkpoint is never deleted).
    """
    if keep_last < 1:
        raise ValueError(f"keep_last must be at least 1, got {keep_last}")
    removed: list[Path] = []
    for path in list_checkpoints(directory=directory)[:-keep_last]:
        step = int(path.name[len("step_") : -len(".pt")])
        if keep_every is not None and step % keep_every == 0:
            continue
        path.unlink()
        removed.append(path)

    return removed
