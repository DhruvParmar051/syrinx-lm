"""Shared pytest fixtures."""

import pytest

from syrinx.utils.device import DEVICE_ENV_VAR, MPS_FALLBACK_ENV_VAR


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove SYRINX_DEVICE and PYTORCH_ENABLE_MPS_FALLBACK for the test's duration.

    Returns:
        The monkeypatch object, so the test can set variables explicitly.
    """
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    monkeypatch.delenv(MPS_FALLBACK_ENV_VAR, raising=False)
    return monkeypatch
