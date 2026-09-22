"""Device selection for Syrinx.

This is the only module in Syrinx that decides which hardware to run on. Everything
else receives a ``torch.device`` and calls ``.to(device)``.

Resolution order: explicit argument > ``SYRINX_DEVICE`` env var > auto (mps > cuda > cpu).
A requested-but-unavailable device raises; there is never a silent CPU fallback.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

VALID_DEVICE_TYPES: frozenset[str] = frozenset({"cpu", "mps", "cuda"})
"""Device types accepted by :func:`resolve_device`."""

DEVICE_ENV_VAR: str = "SYRINX_DEVICE"
"""Environment variable that overrides automatic device selection."""

MPS_FALLBACK_ENV_VAR: str = "PYTORCH_ENABLE_MPS_FALLBACK"
"""If set to ``"1"``, unsupported MPS ops silently run on CPU. Rejected for MPS."""


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
