"""Pass/fail checks for the CUDA dry run (docs/cuda-dry-run.md, steps 4 and 5).

Usage:
    uv run python scripts/check_dryrun.py start runs/cuda-dryrun/metrics.jsonl
    uv run python scripts/check_dryrun.py resume runs/cuda-dryrun/metrics.jsonl \\
        runs/cuda-dryrun-resume/metrics.jsonl --noise runs/cuda-dryrun-repeat/metrics.jsonl

``start``: the first logged loss is within 0.15 of ln(vocab) (the ``test_initial_loss_near_log_vocab``
tolerance), the loss falls from the first to the last logged step, and every logged loss and grad
norm is finite. ``resume``: the largest train-loss difference between a straight run and a resumed
one, next to the same number for two straight runs (the noise floor, as in EXP-011); it fails if the
resumed run is further from the straight run than ``--tolerance`` (default 1e-3, loose enough for
bf16 run-to-run noise, tight enough to catch lost optimizer or data-loader state).

Prints one JSON line; exits 0 on pass, 1 on fail.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from syrinx.utils.log import setup_logging
from syrinx.utils.metrics import read_jsonl

logger = logging.getLogger(__name__)

INITIAL_LOSS_TOLERANCE = 0.15
RESUME_TOLERANCE = 1e-3


def train_losses(path: Path) -> dict[int, dict[str, Any]]:
    """Train records by step; a step logged twice (a resume) keeps its last record.

    Raises:
        ValueError: If the file has no train records.
    """
    records = {r["step"]: r for r in read_jsonl(path) if r["kind"] == "train"}
    if not records:
        raise ValueError(f"{path}: no train records")
    return dict(sorted(records.items()))


def check_start(path: Path, vocab_size: int) -> dict[str, Any]:
    """Step-4 checks on one run: initial loss near ln(vocab), loss falls, all values finite."""
    records = train_losses(path)
    first, last = next(iter(records.values())), list(records.values())[-1]
    expected = math.log(vocab_size)
    finite = all(
        isinstance(r["loss"], float) and math.isfinite(r["loss"])
        and isinstance(r["grad_norm"], float) and math.isfinite(r["grad_norm"])
        for r in records.values()
    )  # fmt: skip
    checks = {
        "initial_loss_ok": abs(first["loss"] - expected) <= INITIAL_LOSS_TOLERANCE,
        "loss_falls": last["loss"] < first["loss"],
        "all_finite": finite,
    }
    return {
        "metrics": str(path),
        "first_step": first["step"],
        "first_loss": first["loss"],
        "expected_initial_loss": expected,
        "last_step": last["step"],
        "last_loss": last["loss"],
        "max_mem_peak_gib": max(r.get("mem_peak") or 0 for r in records.values()) / 2**30,
        **checks,
        "passed": all(checks.values()),
    }


def max_loss_diff(a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]]) -> float:
    """Largest |loss difference| over the steps both runs logged.

    Raises:
        ValueError: If the runs share no step.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two runs have no logged step in common")
    return max(abs(a[s]["loss"] - b[s]["loss"]) for s in shared)


def check_resume(
    straight: Path, resumed: Path, noise: Path | None, tolerance: float = RESUME_TOLERANCE
) -> dict[str, Any]:
    """Step-5 check: the resumed run tracks the straight run (and, if given, the noise floor)."""
    a, b = train_losses(straight), train_losses(resumed)
    if list(a) != list(b):
        raise ValueError(f"steps differ: {straight} has {list(a)}, {resumed} has {list(b)}")
    diff = max_loss_diff(a, b)
    floor = max_loss_diff(a, train_losses(noise)) if noise is not None else None
    return {
        "straight": str(straight),
        "resumed": str(resumed),
        "steps": len(a),
        "max_abs_loss_diff": diff,
        "noise_floor": floor,
        "tolerance": tolerance,
        "passed": diff <= tolerance,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Subcommands ``start METRICS [--vocab-size 32768]`` and
    ``resume STRAIGHT RESUMED [--noise REPEAT] [--tolerance 1e-3]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("metrics", type=Path)
    start.add_argument("--vocab-size", type=int, default=32768)
    resume = sub.add_parser("resume")
    resume.add_argument("straight", type=Path)
    resume.add_argument("resumed", type=Path)
    resume.add_argument("--noise", type=Path, default=None)
    resume.add_argument("--tolerance", type=float, default=RESUME_TOLERANCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one check, print its JSON line, return 0 if it passed and 1 if not."""
    setup_logging()
    args = parse_args(argv)
    if args.command == "start":
        result = check_start(args.metrics, args.vocab_size)
    else:
        result = check_resume(args.straight, args.resumed, args.noise, args.tolerance)
    print(json.dumps(result))
    if not result["passed"]:
        logger.error("%s check FAILED", args.command)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
