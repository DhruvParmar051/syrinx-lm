"""Shared pytest fixtures and accelerator test gating.

Tests marked ``@pytest.mark.mps`` or ``@pytest.mark.cuda`` are skipped when that backend is
unavailable or when ``SYRINX_DEVICE=cpu`` is set, so ``SYRINX_DEVICE=cpu uv run pytest -q``
stays green on any machine.
"""

import os

import pytest
import torch

from syrinx.utils.device import DEVICE_ENV_VAR, MPS_FALLBACK_ENV_VAR, resolve_device


def _forced_cpu() -> bool:
    return os.environ.get(DEVICE_ENV_VAR, "").strip().lower() == "cpu"


def _mps_tests_enabled() -> bool:
    """True if MPS-marked tests should run: MPS exists and CPU is not forced."""
    return torch.backends.mps.is_available() and not _forced_cpu()


def _cuda_tests_enabled() -> bool:
    """True if CUDA-marked tests should run: CUDA exists and CPU is not forced."""
    return torch.cuda.is_available() and not _forced_cpu()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``mps``/``cuda``-marked tests whose backend is not enabled."""
    gates = {
        "mps": (_mps_tests_enabled(), "MPS unavailable or SYRINX_DEVICE=cpu"),
        "cuda": (_cuda_tests_enabled(), "CUDA unavailable or SYRINX_DEVICE=cpu"),
    }
    for marker, (enabled, reason) in gates.items():
        if enabled:
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)


@pytest.fixture
def device() -> torch.device:
    """The device tests should run on, via ``resolve_device()`` (honours SYRINX_DEVICE)."""
    return resolve_device()


@pytest.fixture
def mps_device() -> torch.device:
    """``torch.device("mps")``; only use in ``@pytest.mark.mps`` tests."""
    return torch.device("mps")


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("mps", marks=pytest.mark.mps),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ]
)
def each_device(request: pytest.FixtureRequest) -> torch.device:
    """Runs the test once per backend: CPU always, MPS and CUDA when enabled."""
    return torch.device(request.param)


@pytest.fixture(
    params=[
        pytest.param("mps", marks=pytest.mark.mps),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ]
)
def accelerator(request: pytest.FixtureRequest) -> torch.device:
    """Runs the test once per GPU backend (MPS, CUDA); each is skipped when unavailable."""
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
