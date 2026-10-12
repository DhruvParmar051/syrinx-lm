"""Shared pytest fixtures and MPS test gating.

Tests marked ``@pytest.mark.mps`` are skipped when MPS is unavailable or when
``SYRINX_DEVICE=cpu`` is set, so ``SYRINX_DEVICE=cpu uv run pytest -q`` stays green.
"""

import os

import pytest
import torch

from syrinx.utils.device import DEVICE_ENV_VAR, MPS_FALLBACK_ENV_VAR, resolve_device


def _mps_tests_enabled() -> bool:
    """True if MPS-marked tests should run: MPS exists and CPU is not forced."""
    forced_cpu = os.environ.get(DEVICE_ENV_VAR, "").strip().lower() == "cpu"
    return torch.backends.mps.is_available() and not forced_cpu


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``mps``-marked tests when :func:`_mps_tests_enabled` is False."""
    if _mps_tests_enabled():
        return
    skip = pytest.mark.skip(reason="MPS unavailable or SYRINX_DEVICE=cpu")
    for item in items:
        if item.get_closest_marker("mps") is not None:
            item.add_marker(skip)


@pytest.fixture
def device() -> torch.device:
    """The device tests should run on, via ``resolve_device()`` (honours SYRINX_DEVICE)."""
    return resolve_device()


@pytest.fixture
def mps_device() -> torch.device:
    """``torch.device("mps")``; only use in ``@pytest.mark.mps`` tests."""
    return torch.device("mps")


@pytest.fixture(params=["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def each_device(request: pytest.FixtureRequest) -> torch.device:
    """Runs the test once per backend: CPU always, MPS when enabled (``mps`` mark)."""
    return torch.device(request.param)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove SYRINX_DEVICE and PYTORCH_ENABLE_MPS_FALLBACK for the test's duration.

    Returns:
        The monkeypatch object, so the test can set variables explicitly.
    """
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(MPS_FALLBACK_ENV_VAR, raising=False)
    return monkeypatch
