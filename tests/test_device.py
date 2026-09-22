"""Tests for ``syrinx.utils.device``."""

import pytest
import torch

from syrinx.utils.device import resolve_device


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
