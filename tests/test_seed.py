"""Tests for ``syrinx.utils.seed``."""

import random

import numpy as np
import pytest
import torch

from syrinx.utils.seed import seed_everything


def _draw() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_same_seed_same_draws() -> None:
    """Re-seeding with the same value reproduces python, numpy and torch draws."""
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    second = _draw()
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_different_seed_different_draws() -> None:
    """Different seeds give different torch draws."""
    seed_everything(1)
    a = torch.rand(8)
    seed_everything(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)


@pytest.mark.mps
def test_seed_covers_mps(mps_device: torch.device) -> None:
    """``torch.manual_seed`` also seeds the MPS generator."""
    seed_everything(7)
    a = torch.rand(8, device=mps_device).cpu()
    seed_everything(7)
    b = torch.rand(8, device=mps_device).cpu()
    assert torch.equal(a, b)


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_seed_out_of_range(bad: int) -> None:
    """Seeds outside [0, 2**32 - 1] raise ValueError."""
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(bad)


@pytest.mark.parametrize("bad", [True, 1.5, "1"])
def test_seed_wrong_type(bad: object) -> None:
    """Non-int seeds (including bool) raise TypeError."""
    with pytest.raises(TypeError, match="seed must be an int"):
        seed_everything(bad)  # type: ignore[arg-type]
