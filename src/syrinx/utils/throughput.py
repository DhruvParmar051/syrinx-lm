"""Sustained throughput from a training run's metrics file (PLAN.md, Gate 6a).

``scripts/train.py`` logs one ``train`` record every ``log_interval`` steps; each carries the
synchronized wall time of that one step (``step_time_s``) and its ``tokens_per_sec``. A record
stands for the ``step - previous_step`` steps since the previous record, so a run's elapsed
training time is estimated as the sum of ``(step - previous_step) * step_time_s``. Evaluation
and checkpoint time are not included: this measures training throughput only.

Gate 6a asks for the median tokens/sec over the last 15 minutes of a 30-45 minute run, compared
with the EXP-000 proxy, and maps it to a decision.
"""

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

PROXY_125M_TOK_S = 6674.0
"""EXP-000 proxy throughput for 125M, bf16 autocast, micro-batch 8 (PLAN.md section 1)."""
FULL_BUDGET_TOK_S = 4000.0
"""At or above this, Gate 6a trains the full 2.5B-token budget."""
OPTIMIZE_BELOW_TOK_S = 2000.0
"""Below this, Gate 6a runs Phase 7 optimizations before the long run."""
PROXY_SHORTFALL = 0.30
"""More than this fraction below the proxy must be explained before the long run."""


def train_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The ``train`` records sorted by step, keeping the LAST record for each step.

    A resumed run appends to the same file and replays the steps after its checkpoint, so a step
    can appear twice; the later record is the one from the run that continued.
    """
    by_step: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("kind") == "train":
            by_step[int(record["step"])] = dict(record)
    return [by_step[step] for step in sorted(by_step)]


def estimated_seconds(train: list[Mapping[str, Any]]) -> list[float]:
    """Seconds each record stands for: ``(step - previous step) * step_time_s``. The first
    record's previous step is 0."""
    out: list[float] = []
    previous = 0
    for record in train:
        out.append((int(record["step"]) - previous) * float(record["step_time_s"]))
        previous = int(record["step"])
    return out


def last_window(train: list[Mapping[str, Any]], seconds: float) -> list[Mapping[str, Any]]:
    """The trailing records that together stand for at least ``seconds`` of training.

    Raises:
        ValueError: If ``seconds`` is not positive, or the whole run is shorter than ``seconds``
            (the window would silently include the warm-up, which Gate 6a must exclude).
    """
    if seconds <= 0:
        raise ValueError(f"window must be positive, got {seconds} s")
    durations = estimated_seconds(train)
    total = 0.0
    for i in range(len(train) - 1, -1, -1):
        total += durations[i]
        if total >= seconds:
            return train[i:]
    raise ValueError(
        f"run covers only {total:.0f} s of training, shorter than the {seconds:.0f} s window"
    )


@dataclass(frozen=True)
class ThroughputReport:
    """Gate 6a numbers for one run."""

    window_seconds: float
    """Requested window length."""
    covered_seconds: float
    """Training time the window's records stand for (>= ``window_seconds``)."""
    run_seconds: float
    """Estimated training time of the whole run."""
    first_step: int
    last_step: int
    samples: int
    """Number of logged records in the window."""
    median_tok_s: float
    min_tok_s: float
    max_tok_s: float
    proxy_tok_s: float
    ratio_to_proxy: float
    investigate: bool
    """True if the median is more than :data:`PROXY_SHORTFALL` below the proxy."""
    decision: str
    """``"full-budget"``, ``"choose"`` or ``"optimize-first"`` (PLAN.md Gate 6a)."""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON output."""
        return asdict(self)


def gate_decision(
    median_tok_s: float,
    full_budget: float = FULL_BUDGET_TOK_S,
    optimize_below: float = OPTIMIZE_BELOW_TOK_S,
) -> str:
    """``"full-budget"`` if ``median >= full_budget``, ``"optimize-first"`` if
    ``median < optimize_below``, else ``"choose"``."""
    if median_tok_s >= full_budget:
        return "full-budget"
    if median_tok_s < optimize_below:
        return "optimize-first"
    return "choose"


def throughput_report(
    records: Iterable[Mapping[str, Any]],
    window_seconds: float = 900.0,
    proxy_tok_s: float = PROXY_125M_TOK_S,
) -> ThroughputReport:
    """Median tokens/sec over the last ``window_seconds`` of training, with the Gate 6a decision.

    Raises:
        ValueError: If there are no train records or the run is shorter than the window.
    """
    train = train_records(records)
    if not train:
        raise ValueError("no train records")
    window = last_window(train, window_seconds)
    covered = sum(estimated_seconds(train)[len(train) - len(window) :])
    rates = [float(r["tokens_per_sec"]) for r in window]
    median = statistics.median(rates)
    ratio = median / proxy_tok_s
    return ThroughputReport(
        window_seconds=window_seconds,
        covered_seconds=covered,
        run_seconds=sum(estimated_seconds(train)),
        first_step=int(window[0]["step"]),
        last_step=int(window[-1]["step"]),
        samples=len(window),
        median_tok_s=median,
        min_tok_s=min(rates),
        max_tok_s=max(rates),
        proxy_tok_s=proxy_tok_s,
        ratio_to_proxy=ratio,
        investigate=ratio < 1.0 - PROXY_SHORTFALL,
        decision=gate_decision(median),
    )
