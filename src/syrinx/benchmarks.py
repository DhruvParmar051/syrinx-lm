"""Benchmark scores split by contamination status (Phase 8).

lm-evaluation-harness scores every example; this module labels each scored example with its
contamination status from EXP-016 and recomputes the metrics per subset. Examples are matched
by content (the same normalized text the contamination check hashed), never by position, because
the harness may reorder or preprocess documents. Statuses:

- ``contaminated``: listed in the contamination report;
- ``unchecked``: fewer than ``n`` words, so the check could not test it;
- ``clean``: checked and not contaminated;
- ``unmatched``: its text is not in our benchmark file (never counted as clean).
"""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from syrinx.contamination import example_text, normalize_words

SUBSETS = ("all", "clean", "contaminated", "unchecked", "unmatched")


@dataclass(frozen=True)
class TaskSpec:
    """How one benchmark is run and reported."""

    task: str
    """Local lm-eval task name (``configs/lm_eval``)."""
    metrics: tuple[str, ...]
    """Per-example metrics in the harness samples."""
    headline: str
    """The metric reported first."""


TASKS: dict[str, TaskSpec] = {
    "hellaswag": TaskSpec("syrinx_hellaswag", ("acc", "acc_norm"), "acc_norm"),
    "arc_easy": TaskSpec("syrinx_arc_easy", ("acc", "acc_norm"), "acc"),
    "piqa": TaskSpec("syrinx_piqa", ("acc", "acc_norm"), "acc_norm"),
    "lambada": TaskSpec("syrinx_lambada", ("acc", "perplexity"), "acc"),
}


def content_key(benchmark: str, doc: Mapping[str, Any]) -> str:
    """``" ".join(normalize_words(example_text(benchmark, doc)))``: the text the contamination
    check hashed, so a harness document and our benchmark row compare equal."""
    return " ".join(normalize_words(example_text(benchmark, doc)))


def benchmark_keys(name: str, path: Path) -> dict[str, list[str]]:
    """``{content_key: [example ids]}`` for every JSONL row of ``path``.

    Ids follow ``scripts/check_contamination.py``: ``f"{name}/{row['id']}"`` if the row has an
    ``id``, else ``f"{name}/{line_index}"``. Rows with identical text share a key.
    """
    keys: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            example_id = f"{name}/{row['id']}" if "id" in row else f"{name}/{i}"
            keys.setdefault(content_key(name, row), []).append(example_id)
    return keys


def load_contaminated(path: Path) -> tuple[set[str], int]:
    """Contaminated example ids (all benchmarks) and the n-gram size from a contamination
    report written by ``scripts/check_contamination.py``."""
    report = json.loads(path.read_text(encoding="utf-8"))
    ids = {i for bench in report["benchmarks"].values() for i in bench["contaminated_ids"]}
    return ids, int(report["n"])


def status(key: str, keys: Mapping[str, Sequence[str]], contaminated: set[str], n: int) -> str:
    """Contamination status of one scored example (see module docstring).

    ``unmatched`` if ``key`` is not in ``keys``; else ``unchecked`` if the key has fewer than
    ``n`` words; else ``contaminated`` if any of its ids is in ``contaminated``; else ``clean``.
    """
    if key not in keys:
        return "unmatched"
    if len(key.split()) < n:
        return "unchecked"
    if any(i in contaminated for i in keys[key]):
        return "contaminated"
    return "clean"


@dataclass(frozen=True)
class Score:
    """One metric over one subset."""

    n: int
    value: float | None
    """Mean (accuracy) or ``exp(-mean log-likelihood)`` (perplexity); None if ``n == 0``."""
    stderr: float | None
    """Binomial ``sqrt(p (1 - p) / n)`` for accuracies; None for perplexity or ``n == 0``."""


def score(metric: str, values: Sequence[float]) -> Score:
    """Aggregate per-example ``values`` of ``metric``.

    ``perplexity`` samples are per-example log-likelihoods, aggregated as ``exp(-mean)`` (the
    harness's own perplexity aggregation), with no standard error. Every other metric is a 0/1
    accuracy: mean and binomial standard error.
    """
    n = len(values)
    if n == 0:
        return Score(0, None, None)
    mean = sum(values) / n
    if metric == "perplexity":
        return Score(n, math.exp(-mean), None)
    return Score(n, mean, math.sqrt(mean * (1.0 - mean) / n))


def score_by_subset(
    benchmark: str,
    samples: Iterable[Mapping[str, Any]],
    keys: Mapping[str, Sequence[str]],
    contaminated: set[str],
    n: int,
) -> dict[str, dict[str, Score]]:
    """``{subset: {metric: Score}}`` for every subset in :data:`SUBSETS` and every metric of
    ``TASKS[benchmark]``. Each sample (a harness record with ``doc`` and per-example metric
    fields) goes to ``all`` and to exactly one status subset."""
    spec = TASKS[benchmark]
    values: dict[str, dict[str, list[float]]] = {s: {m: [] for m in spec.metrics} for s in SUBSETS}
    for sample in samples:
        subset = status(
            content_key(benchmark=benchmark, doc=sample["doc"]),
            keys=keys,
            contaminated=contaminated,
            n=n,
        )
        for metric in spec.metrics:
            value = float(sample[metric])
            values["all"][metric].append(value)
            values[subset][metric].append(value)
    return {s: {m: score(m, v) for m, v in per.items()} for s, per in values.items()}
