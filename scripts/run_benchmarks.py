"""Score an exported model on HellaSwag, ARC-Easy, PIQA and LAMBADA, overall and on clean examples.

Runs lm-evaluation-harness 0-shot on the local task configs in ``configs/lm_eval`` (the stock
prompts and metrics, on the pinned benchmark files), offline, with ``add_bos_token=True`` and fp32
weights. Each scored example is labelled with its contamination status (EXP-016) and the metrics
are recomputed per subset (``syrinx.benchmarks``).

Usage:
    uv run python scripts/run_benchmarks.py --model-dir exports/syrinx-10m \\
        --out experiments/benchmarks-10m-2026-10-14.json
    uv run python scripts/run_benchmarks.py --model-dir exports/syrinx-10m --limit 20   # smoke test
"""

import argparse
import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import transformers

from syrinx.benchmarks import TASKS, benchmark_keys, load_contaminated, score_by_subset
from syrinx.tokenizer import file_sha256
from syrinx.utils.device import resolve_device
from syrinx.utils.log import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONTAMINATION = Path("experiments/contamination-2026-10-14.json")
DEFAULT_BENCHMARK_DIR = Path("data_cache/benchmarks")
DEFAULT_TASK_DIR = Path("configs/lm_eval")
OFFLINE_ENV = {"HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options and defaults: ``--model-dir`` (required, ``Path``), ``--benchmark`` (repeatable,
    choices ``TASKS``; none given = all, in ``TASKS`` order), ``--contamination``
    (DEFAULT_CONTAMINATION), ``--benchmark-dir`` (DEFAULT_BENCHMARK_DIR), ``--task-dir``
    (DEFAULT_TASK_DIR), ``--device`` (None), ``--batch-size`` (32), ``--limit`` (int, None =
    every example), ``--out`` (``Path``, None)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=list(TASKS), action="append", default=None)
    parser.add_argument("--contamination", type=Path, default=DEFAULT_CONTAMINATION)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.benchmark is None:
        args.benchmark = list(TASKS)
    return args


def run_harness(
    model_dir: Path,
    tasks: list[str],
    task_dir: Path,
    device: str,
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    """``lm_eval.simple_evaluate`` with ``model="hf"``, ``model_args = f"pretrained={model_dir},
    dtype=float32,add_bos_token=True"``, the local tasks (``TaskManager(include_path=task_dir)``),
    ``num_fewshot=0``, ``log_samples=True`` and ``bootstrap_iters=0`` (no process pool; standard
    errors come from :func:`syrinx.benchmarks.score`). Module-level so tests can replace it."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    return lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_dir},dtype=float32,add_bos_token=True",
        tasks=tasks,
        task_manager=TaskManager(include_path=str(task_dir)),
        num_fewshot=0,
        device=device,
        batch_size=batch_size,
        limit=limit,
        log_samples=True,
        bootstrap_iters=0,
    )


def main(argv: list[str] | None = None) -> int:
    """``setup_logging()``; parse args; set :data:`OFFLINE_ENV` in ``os.environ``;
    ``device = str(resolve_device(args.device))``; load contaminated ids and ``n``
    (``load_contaminated``) and, per benchmark, ``benchmark_keys(name, benchmark_dir /
    f"{name}.jsonl")``; ``run_harness`` once for all selected tasks (timed); for each benchmark,
    ``score_by_subset`` on ``output["samples"][task]``.

    Record: ``{"model_dir" (str), "model_sha256" (of ``model.safetensors``), "config_sha256"
    (of ``config.json``), "lm_eval_version", "transformers_version", "torch_version", "device",
    "batch_size", "limit", "num_fewshot" (0), "add_bos_token" (True), "dtype" ("float32"),
    "contamination" (str path), "contamination_sha256", "seconds", "benchmarks": {name: {"task",
    "headline", "task_config_sha256" (of ``task_dir / f"{task}.yaml"``), "harness": {metric:
    output["results"][task][f"{metric},none"]}, "subsets": {subset: {metric: Score as dict}}}}}``.
    If ``--out`` is given, write it as indented JSON (parents created; mode ``"x"``). Print it as
    ONE JSON line. Returns 0.
    """
    setup_logging()
    args = parse_args(argv)
    os.environ.update(OFFLINE_ENV)
    device = str(resolve_device(args.device))
    contaminated, n = load_contaminated(args.contamination)
    keys = {
        name: benchmark_keys(name, args.benchmark_dir / f"{name}.jsonl") for name in args.benchmark
    }
    tasks = [TASKS[name].task for name in args.benchmark]
    start = time.perf_counter()
    output = run_harness(args.model_dir, tasks, args.task_dir, device, args.batch_size, args.limit)
    seconds = time.perf_counter() - start
    import lm_eval

    benchmarks: dict[str, Any] = {}
    for name in args.benchmark:
        spec = TASKS[name]
        subsets = score_by_subset(name, output["samples"][spec.task], keys[name], contaminated, n)
        benchmarks[name] = {
            "task": spec.task,
            "headline": spec.headline,
            "task_config_sha256": file_sha256(args.task_dir / f"{spec.task}.yaml"),
            "harness": {m: output["results"][spec.task][f"{m},none"] for m in spec.metrics},
            "subsets": {
                s: {m: dataclasses.asdict(sc) for m, sc in per.items()}
                for s, per in subsets.items()
            },
        }
    record: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "model_sha256": file_sha256(args.model_dir / "model.safetensors"),
        "config_sha256": file_sha256(args.model_dir / "config.json"),
        "lm_eval_version": lm_eval.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "num_fewshot": 0,
        "add_bos_token": True,
        "dtype": "float32",
        "contamination": str(args.contamination),
        "contamination_sha256": file_sha256(args.contamination),
        "seconds": seconds,
        "benchmarks": benchmarks,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
