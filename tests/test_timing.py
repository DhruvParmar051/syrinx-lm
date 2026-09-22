"""Tests for ``syrinx.utils.timing``."""

import math
import time

import pytest
import torch

from syrinx.utils.timing import Timer, measure_matmul_tflops

# Far above any Apple GPU (the M5 Pro measured ~31 bf16 TFLOPS). A timer that didn't
# synchronize would only measure the enqueue time and report numbers above this.
IMPOSSIBLE_TFLOPS = 500.0


def test_timer_measures_sleep() -> None:
    """A 50 ms sleep on cpu is measured as >= 50 ms and well under 1 s."""
    with Timer(torch.device("cpu")) as t:
        time.sleep(0.05)
    assert t.elapsed is not None
    assert 0.05 <= t.elapsed < 1.0


def test_timer_elapsed_none_before_exit() -> None:
    """``elapsed`` is None until the block exits."""
    timer = Timer(torch.device("cpu"))
    assert timer.elapsed is None
    with timer:
        assert timer.elapsed is None
    assert timer.elapsed is not None


def test_timer_records_even_on_exception() -> None:
    """If the block raises, the exception propagates and elapsed is still set."""
    timer = Timer(torch.device("cpu"))
    with pytest.raises(KeyError), timer:
        raise KeyError("boom")
    assert timer.elapsed is not None


@pytest.mark.mps
def test_timer_waits_for_gpu_work(mps_device: torch.device) -> None:
    """Timing 20 queued 4096^2 fp32 matmuls implies a physically possible TFLOPS.

    Without synchronization the timer would only see the enqueue time and the
    implied throughput would exceed IMPOSSIBLE_TFLOPS.
    """
    n, iters = 4096, 20
    a = torch.randn(n, n, device=mps_device)
    a @ a  # warm-up
    with Timer(mps_device) as t:
        for _ in range(iters):
            a @ a
    assert t.elapsed is not None
    implied_tflops = 2 * n**3 * iters / t.elapsed / 1e12
    assert implied_tflops < IMPOSSIBLE_TFLOPS


def test_matmul_tflops_cpu_positive() -> None:
    """A small cpu fp32 benchmark returns a finite, positive number."""
    tflops = measure_matmul_tflops(torch.device("cpu"), torch.float32, n=256, iters=3, warmup=1)
    assert math.isfinite(tflops) and tflops > 0


@pytest.mark.mps
def test_matmul_tflops_mps_plausible(mps_device: torch.device) -> None:
    """MPS fp32 at n=2048 is positive and below the impossible bound."""
    tflops = measure_matmul_tflops(mps_device, torch.float32, n=2048, iters=10, warmup=2)
    assert 0 < tflops < IMPOSSIBLE_TFLOPS


@pytest.mark.parametrize(
    ("n", "iters", "warmup"),
    [(0, 1, 0), (-4, 1, 0), (16, 0, 0), (16, 1, -1)],
)
def test_matmul_tflops_rejects_bad_args(n: int, iters: int, warmup: int) -> None:
    """Non-positive n or iters, or negative warmup, raises ValueError."""
    with pytest.raises(ValueError):
        measure_matmul_tflops(torch.device("cpu"), torch.float32, n=n, iters=iters, warmup=warmup)


def test_matmul_tflops_leaves_global_rng_untouched() -> None:
    """The benchmark uses a local generator, so the global torch RNG state is unchanged."""
    before = torch.get_rng_state()
    measure_matmul_tflops(torch.device("cpu"), torch.float32, n=64, iters=1, warmup=0)
    assert torch.equal(before, torch.get_rng_state())
