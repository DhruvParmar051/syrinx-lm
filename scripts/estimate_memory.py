"""Print analytic memory and compute estimates for a config.

Usage:
    uv run python scripts/estimate_memory.py configs/syrinx-125m.yaml \\
        [--micro-batch 8] [--precision bf16-autocast] [--tokens 2.5e9] [--tok-per-sec 6674]

Defaults come from the config: micro-batch and precision from ``train``, and
tokens from ``max_steps * tokens_per_step``. ``--tok-per-sec`` must be a measured
number (e.g. from Gate 6a), never a guess.
"""

import argparse
import logging
from pathlib import Path

from syrinx.config import VALID_PRECISIONS, load_config, num_parameters
from syrinx.estimate import (
    MemoryEstimate,
    estimate_train_days,
    estimate_training_memory,
    flops_per_token,
)
from syrinx.utils.device import bytes_to_gib
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``config``, ``micro_batch``, ``precision``, ``tokens`` and
        ``tok_per_sec`` (the last four may be None).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--micro-batch", type=int, help="override train.micro_batch_size")
    parser.add_argument("--precision", choices=sorted(VALID_PRECISIONS), help="override precision")
    parser.add_argument("--tokens", type=float, help="training tokens (default: from config)")
    parser.add_argument("--tok-per-sec", type=float, help="MEASURED throughput for a time estimate")
    return parser.parse_args(argv)


def format_estimate(name: str, estimate: MemoryEstimate, num_params: int) -> str:
    """Render the per-component breakdown in GiB plus the total.

    Args:
        name: Config name.
        estimate: Memory breakdown.
        num_params: Total parameter count, shown in the header.

    Returns:
        The table as text.
    """
    lines = [f"{name}: {num_params:,} parameters", f"{'component':<16}{'GiB':>9}"]
    for component, size in estimate.components.items():
        lines.append(f"{component:<16}{bytes_to_gib(size):>9.2f}")
    subtotal = sum(estimate.components.values())
    lines.append(f"{'subtotal':<16}{bytes_to_gib(subtotal):>9.2f}")
    lines.append(
        f"{'total x' + format(estimate.overhead_factor, 'g'):<16}{bytes_to_gib(estimate.total):>9.2f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point: load the config, estimate, print.

    Args:
        argv: Argument list.

    Returns:
        Process exit code (0 on success).
    """
    setup_logging()
    args = parse_args(argv)
    config = load_config(args.config)
    micro_batch = args.micro_batch or config.train.micro_batch_size
    precision = args.precision or config.train.precision
    seq_len = config.data.seq_len
    tokens = (
        args.tokens if args.tokens is not None else config.train.max_steps * config.tokens_per_step
    )
    n_params = num_parameters(config.model)

    estimate = estimate_training_memory(config.model, micro_batch, seq_len, precision)
    print(f"micro-batch {micro_batch}, seq_len {seq_len}, precision {precision}")
    print(format_estimate(config.name, estimate, n_params))

    per_token = flops_per_token(config.model, seq_len)
    print(f"FLOPs/token {per_token:.3e}; tokens {tokens:.3e}; train FLOPs {per_token * tokens:.3e}")
    if args.tok_per_sec is not None:
        days = estimate_train_days(tokens, args.tok_per_sec)
        print(f"at {args.tok_per_sec:,.0f} tok/s (measured): {days:.2f} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
