"""Tests for ``syrinx.utils.device``."""

import pytest
import torch

from syrinx.utils.device import (
    bytes_to_gib,
    memory_stats,
    reset_peak_memory,
    resolve_device,
    synchronize,
)

MIB = 2**20


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


def test_memory_stats_cpu() -> None:
    """On cpu: allocated is process RSS (> 0); the accelerator-only fields are None."""
    stats = memory_stats(torch.device("cpu"))
    assert stats.device_type == "cpu"
    assert stats.allocated > 0
    assert stats.peak_allocated is None
    assert stats.driver_allocated is None
    assert stats.recommended_max is None


@pytest.mark.mps
def test_memory_stats_mps_invariants(mps_device: torch.device) -> None:
    """On mps: peak >= allocated, driver >= allocated, recommended_max > 0."""
    keep = torch.empty(16 * MIB // 4, device=mps_device)
    synchronize(mps_device)
    stats = memory_stats(mps_device)
    assert stats.device_type == "mps"
    assert stats.allocated >= keep.numel() * 4
    assert stats.peak_allocated is not None and stats.peak_allocated >= stats.allocated
    assert stats.driver_allocated is not None and stats.driver_allocated >= stats.allocated
    assert stats.recommended_max is not None and stats.recommended_max > 0


@pytest.mark.mps
def test_memory_allocation_visible(mps_device: torch.device) -> None:
    """Allocating a 64 MiB tensor raises ``allocated`` by at least 64 MiB."""
    before = memory_stats(mps_device).allocated
    block = torch.empty(64 * MIB // 4, device=mps_device)
    synchronize(mps_device)
    after = memory_stats(mps_device).allocated
    assert after - before >= 64 * MIB
    del block


@pytest.mark.mps
def test_reset_peak(mps_device: torch.device) -> None:
    """A freed 256 MiB temporary raises the peak; after reset, peak == allocated."""
    reset_peak_memory(mps_device)
    temp = torch.empty(256 * MIB // 4, device=mps_device)
    synchronize(mps_device)
    del temp
    stats = memory_stats(mps_device)
    assert stats.peak_allocated is not None
    assert stats.peak_allocated - stats.allocated >= 256 * MIB
    reset_peak_memory(mps_device)
    stats = memory_stats(mps_device)
    assert stats.peak_allocated == stats.allocated


def test_reset_peak_cpu_noop() -> None:
    """``reset_peak_memory(cpu)`` is a no-op."""
    assert reset_peak_memory(torch.device("cpu")) is None


def test_memory_unknown_type_raises() -> None:
    """``memory_stats`` and ``reset_peak_memory`` reject unsupported device types."""
    with pytest.raises(ValueError, match="meta"):
        memory_stats(torch.device("meta"))
    with pytest.raises(ValueError, match="meta"):
        reset_peak_memory(torch.device("meta"))


def test_bytes_to_gib() -> None:
    """``2**30`` bytes is 1 GiB, 0 is 0, and a negative count raises."""
    assert bytes_to_gib(2**30) == 1.0
    assert bytes_to_gib(0) == 0.0
    assert bytes_to_gib(3 * 2**29) == 1.5
    with pytest.raises(ValueError, match="non-negative"):
        bytes_to_gib(-1)
