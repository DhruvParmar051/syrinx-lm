"""Device selection, synchronization and memory statistics for Syrinx.

This is the only module in Syrinx that decides which hardware to run on. Everything
else receives a ``torch.device`` and calls ``.to(device)``.

Resolution order: explicit argument > ``SYRINX_DEVICE`` env var > auto (mps > cuda > cpu).
A requested-but-unavailable device raises; there is never a silent CPU fallback.
"""

import logging
import os
from dataclasses import dataclass

import psutil
import torch

logger = logging.getLogger(__name__)

VALID_DEVICE_TYPES: frozenset[str] = frozenset({"cpu", "mps", "cuda"})
"""Device types accepted by :func:`resolve_device`."""

DEVICE_ENV_VAR: str = "SYRINX_DEVICE"
"""Environment variable that overrides automatic device selection."""

MPS_FALLBACK_ENV_VAR: str = "PYTORCH_ENABLE_MPS_FALLBACK"
"""If set to ``"1"``, unsupported MPS ops silently run on CPU. Rejected for MPS."""


@dataclass(frozen=True)
class MemoryStats:
    """Memory snapshot for one device, in bytes.

    Attributes:
        device_type: ``"cpu"``, ``"mps"`` or ``"cuda"``.
        allocated: Bytes held by live tensors (accelerator) or process RSS (cpu).
        peak_allocated: Max ``allocated`` since the last :func:`reset_peak_memory`;
            None on cpu.
        driver_allocated: Total bytes the backend holds, including the caching
            allocator (MPS driver memory / CUDA reserved); None on cpu. Compare THIS,
            not ``allocated``, to ``recommended_max``.
        recommended_max: Backend's recommended working-set ceiling (MPS) or total
            device memory (CUDA); None on cpu.
    """

    device_type: str
    allocated: int
    peak_allocated: int | None
    driver_allocated: int | None
    recommended_max: int | None


def _auto_device_type() -> str:
    """Pick the best available backend type: ``"mps"``, then ``"cuda"``, then ``"cpu"``.

    Returns:
        A member of :data:`VALID_DEVICE_TYPES`.
    """

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _check_backend(device_type: str) -> None:
    """Raise if ``device_type`` cannot be used safely on this machine.

    Checks backend availability, and for ``"mps"`` also rejects
    ``PYTORCH_ENABLE_MPS_FALLBACK=1``. ``"cpu"`` always passes.

    Args:
        device_type: A member of :data:`VALID_DEVICE_TYPES` (already normalized).

    Raises:
        RuntimeError: If the backend is unavailable or the MPS fallback is enabled.
            The message tells the user how to force CPU.
    """
    if device_type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                f"Requested device 'mps' but MPS is not available. "
                f"Set {DEVICE_ENV_VAR}=cpu to force CPU."
            )

        if os.environ.get(MPS_FALLBACK_ENV_VAR) == "1":
            raise RuntimeError(
                f"{MPS_FALLBACK_ENV_VAR}=1 is set: unsupported MPS ops would "
                f"silently run on CPU. Unset it before running Syrinx."
            )

    elif device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device 'cuda' but CUDA is not available. "
            f"Set {DEVICE_ENV_VAR}=cpu to force CPU."
        )


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve the torch device to run on.

    Order: ``requested`` if given, else the ``SYRINX_DEVICE`` env var (read at call
    time, not import time), else auto-detect preferring mps, then cuda, then cpu.

    Args:
        requested: ``"cpu"``, ``"mps"``, ``"cuda"`` (case-insensitive) or None.

    Returns:
        The resolved device.

    Raises:
        ValueError: If the name is not in :data:`VALID_DEVICE_TYPES`.
        RuntimeError: If the requested backend is unavailable, or if MPS is selected
            while ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set.
    """
    if requested is not None:
        name, source = requested, "argument"
    elif os.environ.get(DEVICE_ENV_VAR):
        name, source = os.environ[DEVICE_ENV_VAR], f"env {DEVICE_ENV_VAR}"
    else:
        name, source = _auto_device_type(), "auto-detect"

    device_type = name.strip().lower()
    if device_type not in VALID_DEVICE_TYPES:
        raise ValueError(
            f"Unknown device {name!r} (from {source}); "
            f"expected one of {sorted(VALID_DEVICE_TYPES)}."
        )

    _check_backend(device_type)
    logger.info("Using device %s (from %s)", device_type, source)
    return torch.device(device_type)


def synchronize(device: torch.device) -> None:
    """Block until all work queued on ``device`` has finished.

    GPU backends (MPS, CUDA) run asynchronously: a call returns as soon as the work
    is queued, not when it is done. Call this before starting and before stopping
    any timer. No-op on cpu. Dispatch on ``device.type``, never on ``device``
    equality (``torch.device("mps") != torch.device("mps:0")``).

    Args:
        device: Device to synchronize.

    Raises:
        ValueError: If ``device.type`` is not one of :data:`VALID_DEVICE_TYPES`.
    """
    if device.type == "cpu":
        return
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        raise ValueError(
            f"Cannot synchronize device type {device.type!r}; "
            f"expected one of {sorted(VALID_DEVICE_TYPES)}."
        )


def _cpu_rss_bytes() -> int:
    """Resident set size of the current process in bytes."""
    return int(psutil.Process().memory_info().rss)


def memory_stats(device: torch.device) -> MemoryStats:
    """Take a memory snapshot for ``device``.

    Accelerators report live-tensor bytes and their peak via ``torch.accelerator``,
    plus the backend's driver-level total and ceiling. CPU reports process RSS only.
    The CUDA branch follows the same API but is untested (no CUDA hardware).

    Args:
        device: Device to inspect.

    Returns:
        The snapshot, in bytes.

    Raises:
        ValueError: If ``device.type`` is not one of :data:`VALID_DEVICE_TYPES`.
    """
    if device.type == "cpu":
        return MemoryStats("cpu", _cpu_rss_bytes(), None, None, None)
    if device.type == "mps":
        return MemoryStats(
            device_type="mps",
            allocated=int(torch.accelerator.memory_allocated()),
            peak_allocated=int(torch.accelerator.max_memory_allocated()),
            driver_allocated=int(torch.mps.driver_allocated_memory()),
            recommended_max=int(torch.mps.recommended_max_memory()),
        )
    if device.type == "cuda":
        return MemoryStats(
            device_type="cuda",
            allocated=int(torch.cuda.memory_allocated(device)),
            peak_allocated=int(torch.cuda.max_memory_allocated(device)),
            driver_allocated=int(torch.cuda.memory_reserved(device)),
            recommended_max=int(torch.cuda.get_device_properties(device).total_memory),
        )
    raise ValueError(
        f"No memory stats for device type {device.type!r}; "
        f"expected one of {sorted(VALID_DEVICE_TYPES)}."
    )


def reset_peak_memory(device: torch.device) -> None:
    """Reset the peak-allocated counter for ``device`` (no-op on cpu).

    Call this at the start of each measured region so peaks don't carry over.

    Args:
        device: Device whose peak counter to reset.

    Raises:
        ValueError: If ``device.type`` is not one of :data:`VALID_DEVICE_TYPES`.
    """
    if device.type == "cpu":
        return
    if device.type == "mps":
        torch.accelerator.reset_peak_memory_stats()
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        raise ValueError(
            f"Cannot reset peak memory for device type {device.type!r}; "
            f"expected one of {sorted(VALID_DEVICE_TYPES)}."
        )


def bytes_to_gib(num_bytes: int) -> float:
    """Convert bytes to GiB (2**30 bytes).

    Args:
        num_bytes: Byte count; must be non-negative.

    Returns:
        Size in GiB.

    Raises:
        ValueError: If ``num_bytes`` is negative.
    """
    if num_bytes < 0:
        raise ValueError(f"num_bytes must be non-negative, got {num_bytes}.")
    return num_bytes / 2**30
