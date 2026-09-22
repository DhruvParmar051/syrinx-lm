"""Wall-clock timing that is correct on asynchronous GPU backends.

MPS and CUDA queue work and return immediately, so a timer must synchronize the
device before it starts (so earlier queued work isn't counted) and before it stops
(so the timed work has actually finished).
"""

import logging
import time
from types import TracebackType
from typing import Self

import torch

from syrinx.utils.device import synchronize

logger = logging.getLogger(__name__)


class Timer:
    """Context manager measuring the wall-clock seconds of a block on ``device``.

    Example::

        with Timer(device) as t:
            run_step()
        logger.info("step took %.3f s", t.elapsed)

    Attributes:
        device: Device synchronized at both ends of the block.
        elapsed: Seconds taken by the block; None until the block exits.
    """

    def __init__(self, device: torch.device) -> None:
        """Create a timer for ``device``.

        Args:
            device: Device whose queued work must be included in the measurement.
        """
        self.device = device
        self.elapsed: float | None = None
        self._start: float | None = None

    def __enter__(self) -> Self:
        """Synchronize, then start the clock."""
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Synchronize, then stop the clock and store :attr:`elapsed`."""
        synchronize(self.device)
        if self._start is None:
            raise RuntimeError("Timer exited without being entered.")
        self.elapsed = time.perf_counter() - self._start


def measure_matmul_tflops(
    device: torch.device,
    dtype: torch.dtype,
    n: int = 4096,
    iters: int = 30,
    warmup: int = 5,
    seed: int = 0,
) -> float:
    """Measure square-matmul throughput in TFLOPS.

    Each (n x n) @ (n x n) matmul costs 2 * n**3 FLOPs. Inputs are drawn from a
    local seeded generator on the CPU and then moved, so the global RNG is untouched.

    Args:
        device: Device to benchmark on.
        dtype: Element dtype (e.g. ``torch.float32``, ``torch.bfloat16``).
        n: Matrix side length.
        iters: Timed iterations.
        warmup: Untimed iterations run first (kernel selection, allocator warm-up).
        seed: Seed for the input matrices.

    Returns:
        Achieved TFLOPS.

    Raises:
        ValueError: If ``n`` or ``iters`` is not positive, or ``warmup`` is negative.
    """
    if n <= 0 or iters <= 0:
        raise ValueError(f"n and iters must be positive, got n={n}, iters={iters}.")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")

    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=generator).to(device=device, dtype=dtype)
    b = torch.randn(n, n, generator=generator).to(device=device, dtype=dtype)

    for _ in range(warmup):
        a @ b
    with Timer(device) as timer:
        for _ in range(iters):
            a @ b

    assert timer.elapsed is not None
    tflops = 2 * n**3 * iters / timer.elapsed / 1e12
    logger.debug("matmul n=%d dtype=%s on %s: %.2f TFLOPS", n, dtype, device.type, tflops)
    return tflops
