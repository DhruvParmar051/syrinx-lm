"""Report sustained training throughput from a run's metrics file (PLAN.md, Gate 6a).

Usage:
    uv run python scripts/throughput_report.py runs/125m-bench/metrics.jsonl \\
        [--window-minutes 15] [--proxy-tok-s 6674] [--out experiments/gate6a.json]

Prints one JSON line: the median tokens/sec over the last ``--window-minutes`` of training, the
ratio to the proxy, whether the shortfall needs investigating, and the Gate 6a decision
(``full-budget`` >= 4,000 tok/s, ``choose`` 2,000-4,000, ``optimize-first`` < 2,000).
"""

import argparse
import json
import logging
from pathlib import Path

from syrinx.utils.log import setup_logging
from syrinx.utils.metrics import read_jsonl
from syrinx.utils.throughput import PROXY_125M_TOK_S, throughput_report

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``metrics``, ``window_minutes``, ``proxy_tok_s`` and ``out`` (may be None).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", type=Path, help="metrics.jsonl written by scripts/train.py")
    parser.add_argument("--window-minutes", type=float, default=15.0)
    parser.add_argument("--proxy-tok-s", type=float, default=PROXY_125M_TOK_S)
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the report. Returns 0; raises ``FileExistsError`` rather than overwrite ``--out``."""
    setup_logging()
    args = parse_args(argv)
    report = throughput_report(
        read_jsonl(args.metrics), args.window_minutes * 60.0, args.proxy_tok_s
    )
    result = {"metrics": str(args.metrics), **report.to_dict()}
    line = json.dumps(result)
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as fh:
            fh.write(line + "\n")
    logger.info(
        "median %.0f tok/s over steps %d-%d (%.2fx proxy): %s",
        report.median_tok_s,
        report.first_step,
        report.last_step,
        report.ratio_to_proxy,
        report.decision,
    )
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
