"""Phase 5 learning-rate sweep: build the sweep configs and pick the LR.

Each sweep run is the full-run config with only ``lr`` (and ``min_lr = lr / 10``) changed,
so it follows the real warmup + cosine schedule and is stopped early with
``--stop-at-step``. The pick rule (docs/decisions.md, 2026-10-12):

1. A run is *stable* if every logged train loss and grad norm is finite, its val loss at
   the compare step is finite, and no train loss after warmup exceeds the running minimum
   by more than ``SPIKE_NATS``.
2. Among stable runs, take the lowest val loss at the compare step; if lower-LR stable runs
   are within ``TOLERANCE`` of it, take the lowest such LR (short runs favour high LRs).

Usage:
    uv run python scripts/lr_sweep.py make-configs --base configs/syrinx-50m.yaml \\
        --lrs 3e-4 6e-4 1.2e-3 --out-dir runs/50m-sweep
    uv run python scripts/lr_sweep.py pick --runs runs/50m-sweep/lr-* --step 300 \\
        --base configs/syrinx-50m.yaml --config-out runs/50m-a.yaml --summary-out x.json
"""

import argparse
import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from syrinx.config import SyrinxConfig, load_config, save_config
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

SPIKE_NATS = 1.0
"""A train loss this far above the running minimum (after warmup) marks a run unstable."""
TOLERANCE = 0.01
"""Val-loss differences up to this are treated as noise when choosing the LR."""
MIN_LR_RATIO = 0.1


def lr_tag(lr: float) -> str:
    """Directory-safe name for a learning rate, e.g. ``6e-4`` -> ``lr-6.0e-04``."""
    return f"lr-{lr:.1e}"


def with_lr(config: SyrinxConfig, lr: float) -> SyrinxConfig:
    """``config`` with ``lr`` and ``min_lr = MIN_LR_RATIO * lr``; everything else unchanged.

    Raises:
        ValueError: If ``lr`` is not positive.
    """
    if not lr > 0:
        raise ValueError(f"lr must be positive, got {lr}.")
    min_lr = float(f"{lr * MIN_LR_RATIO:.12g}")  # 1.2e-4, not 0.00011999999999999999
    return replace(config, train=replace(config.train, lr=lr, min_lr=min_lr))


@dataclass(frozen=True)
class RunSummary:
    """What the pick rule needs from one sweep run."""

    run: str
    lr: float
    val_loss: float | None
    stable: bool
    reason: str


def summarize_run(
    run: str, lr: float, records: Iterable[dict[str, Any]], step: int, warmup_steps: int
) -> RunSummary:
    """Apply the stability rule to one run's ``metrics.jsonl`` records."""
    val_loss: float | None = None
    running_min = math.inf
    for record in records:
        if record.get("kind") == "val" and record.get("step") == step:
            val_loss = float(record["val_loss"])
        if record.get("kind") != "train":
            continue
        loss, grad_norm = float(record["loss"]), float(record["grad_norm"])
        if not (math.isfinite(loss) and math.isfinite(grad_norm)):
            return RunSummary(run, lr, val_loss, False, f"non-finite at step {record['step']}")
        if record["step"] > warmup_steps and loss > running_min + SPIKE_NATS:
            reason = f"loss spike at step {record['step']}: {loss:.3f} vs min {running_min:.3f}"
            return RunSummary(run, lr, val_loss, False, reason)
        running_min = min(running_min, loss)
    if val_loss is None:
        return RunSummary(run, lr, None, False, f"no val loss at step {step}")
    if not math.isfinite(val_loss):
        return RunSummary(run, lr, val_loss, False, "non-finite val loss")
    return RunSummary(run, lr, val_loss, True, "ok")


def choose_lr(summaries: Sequence[RunSummary], tolerance: float = TOLERANCE) -> RunSummary:
    """The run picked by the rule in the module docstring.

    Raises:
        ValueError: If no run is stable.
    """
    stable = [s for s in summaries if s.stable and s.val_loss is not None]
    if not stable:
        raise ValueError("no stable sweep run; not choosing an LR.")
    best = min(s.val_loss for s in stable if s.val_loss is not None)
    close = [s for s in stable if s.val_loss is not None and s.val_loss <= best + tolerance]
    return min(close, key=lambda s: s.lr)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """All records of a JSONL file."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Subcommands ``make-configs`` and ``pick`` (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-configs")
    make.add_argument("--base", type=Path, required=True)
    make.add_argument("--lrs", type=float, nargs="+", required=True)
    make.add_argument("--out-dir", type=Path, required=True)
    pick = sub.add_parser("pick")
    pick.add_argument("--runs", type=Path, nargs="+", required=True)
    pick.add_argument("--step", type=int, required=True)
    pick.add_argument("--base", type=Path, required=True)
    pick.add_argument("--config-out", type=Path, required=True)
    pick.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. ``pick`` returns 1 (and writes no config) if no run is stable."""
    setup_logging()
    args = parse_args(argv)
    if args.command == "make-configs":
        base = load_config(args.base)
        for lr in args.lrs:
            path = args.out_dir / f"{lr_tag(lr)}.yaml"
            if path.exists():
                logger.info("keeping existing %s", path)
                continue
            save_config(with_lr(base, lr), path)
            print(path)
        return 0

    base = load_config(args.base)
    summaries = []
    for run in args.runs:
        config = load_config(run / "config.yaml")
        if not (run / "metrics.jsonl").exists():
            summaries.append(RunSummary(str(run), config.train.lr, None, False, "no metrics"))
            continue
        records = read_jsonl(run / "metrics.jsonl")
        summaries.append(
            summarize_run(str(run), config.train.lr, records, args.step, config.train.warmup_steps)
        )
    result: dict[str, Any] = {
        "step": args.step,
        "spike_nats": SPIKE_NATS,
        "tolerance": TOLERANCE,
        "runs": [asdict(s) for s in sorted(summaries, key=lambda s: s.lr)],
    }
    try:
        chosen = choose_lr(summaries)
    except ValueError as err:
        result["chosen"] = None
        result["error"] = str(err)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.error("%s", err)
        return 1
    result["chosen"] = asdict(chosen)
    save_config(with_lr(base, chosen.lr), args.config_out)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
