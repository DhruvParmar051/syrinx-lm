"""Tests for ``syrinx.utils.device``."""

import pytest
import torch

from syrinx.utils.device import resolve_device, synchronize


def test_resolve_explicit_cpu(clean_env: pytest.MonkeyPatch) -> None:
    """``resolve_device("cpu")`` returns a device with ``.type == "cpu"``."""
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_resolve_env_override(clean_env: pytest.MonkeyPatch) -> None:
    """With SYRINX_DEVICE=cpu, ``resolve_device()`` returns cpu even if MPS exists."""
    clean_env.setenv("SYRINX_DEVICE", "cpu")
    assert resolve_device().type == "cpu"


def test_resolve_invalid_name_raises(clean_env: pytest.MonkeyPatch) -> None:
    """``"gpu"`` raises ValueError whose message lists every valid device type."""
    with pytest.raises(ValueError, match="expected one of"):
        resolve_device("gpu")


def test_resolve_unavailable_mps_raises(clean_env: pytest.MonkeyPatch) -> None:
    """With MPS availability monkeypatched False, requesting mps raises RuntimeError.

    It must NOT return cpu (no silent fallback).
    """
    clean_env.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("mps")


def test_mps_fallback_env_rejected(clean_env: pytest.MonkeyPatch) -> None:
    """With PYTORCH_ENABLE_MPS_FALLBACK=1, resolving mps raises RuntimeError."""
    clean_env.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="PYTORCH_ENABLE_MPS_FALLBACK"):
        resolve_device("mps")


def test_synchronize_cpu_noop() -> None:
    """``synchronize(torch.device("cpu"))`` returns None without error."""
    assert synchronize(torch.device("cpu")) is None


@pytest.mark.mps
def test_synchronize_mps_runs_after_queued_work(mps_device: torch.device) -> None:
    """On MPS: queue a large matmul, synchronize, and the result is finite.

    This proves synchronize runs on MPS; that it actually waits is covered by the
    Timer tests in test_timing.py.
    """
    a = torch.randn(2048, 2048, device=mps_device)
    b = a @ a
    synchronize(mps_device)
    assert torch.isfinite(b).all().item()


def test_synchronize_accepts_indexed_device() -> None:
    """Dispatch is by ``.type``: an indexed device such as ``cpu:0`` is accepted."""
    assert synchronize(torch.device("cpu", 0)) is None


def test_synchronize_unknown_type_raises() -> None:
    """``synchronize(torch.device("meta"))`` raises ValueError naming the type."""
    with pytest.raises(ValueError, match="meta"):
        synchronize(torch.device("meta"))
