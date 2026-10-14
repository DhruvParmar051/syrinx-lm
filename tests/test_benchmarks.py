"""Tests for ``syrinx.benchmarks``: content keys, statuses and per-subset scores."""

import json
import math
from pathlib import Path

import pytest

from syrinx.benchmarks import (
    SUBSETS,
    TASKS,
    Score,
    benchmark_keys,
    content_key,
    load_contaminated,
    score,
    score_by_subset,
    status,
)
from syrinx.contamination import build_index, example_text, normalize_words

LONG = "a girl stands at the edge of the diving board and looks down at the blue water below"
OTHER = (
    "photosynthesis converts light energy into chemical energy that plants store as sugar in leaves"
)


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# content_key / benchmark_keys -----------------------------------------------------------------


def test_content_key_is_normalized_example_text() -> None:
    row = {"ctx": "A Girl, stands!", "endings": ["She JUMPS.", "waves"], "label": "0"}
    assert content_key("hellaswag", row) == "a girl stands she jumps waves"
    assert content_key("hellaswag", row) == " ".join(
        normalize_words(example_text("hellaswag", row))
    )


def test_content_key_ignores_harness_added_fields() -> None:
    """lm-eval's HellaSwag processing adds query/choices/gold but keeps ctx and endings."""
    raw = {
        "ctx": "A man is sitting on a roof. he",
        "endings": ["is using wrap", "starts pulling"],
        "label": "3",
    }
    processed = {
        **raw,
        "query": "Roof shingle removal: A man is sitting on a roof. He",
        "choices": ["x"],
        "gold": 3,
    }
    assert content_key("hellaswag", processed) == content_key("hellaswag", raw)


def test_content_key_matches_contamination_index_words() -> None:
    row = {"text": LONG}
    index = build_index({"lambada/0": example_text("lambada", row)})
    words = content_key("lambada", row).split()
    assert tuple(words[:13]) in index.owners


def test_benchmark_keys_ids_and_duplicates(tmp_path: Path) -> None:
    arc = _jsonl(
        tmp_path / "arc.jsonl",
        [
            {"id": "Q1", "question": "Why?", "choices": {"text": ["A"]}},
            {"question": "How?", "choices": {"text": ["B"]}},
            {"id": "Q3", "question": "Why?", "choices": {"text": ["A"]}},
        ],
    )
    keys = benchmark_keys("arc_easy", arc)
    assert keys == {"why a": ["arc_easy/Q1", "arc_easy/Q3"], "how b": ["arc_easy/1"]}


def test_load_contaminated(tmp_path: Path) -> None:
    report = {
        "n": 13,
        "benchmarks": {
            "a": {"contaminated_ids": ["a/1"]},
            "b": {"contaminated_ids": ["b/x", "b/y"]},
        },
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_contaminated(path) == ({"a/1", "b/x", "b/y"}, 13)
    report["n"] = 8
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_contaminated(path)[1] == 8


# status ------------------------------------------------------------------------------------------


def test_status_all_four() -> None:
    keys = {LONG: ["x/0"], OTHER: ["x/1"], "too short": ["x/2"]}
    assert status(LONG, keys, {"x/0"}, 13) == "contaminated"
    assert status(OTHER, keys, {"x/0"}, 13) == "clean"
    assert status("too short", keys, set(), 13) == "unchecked"
    assert (
        status(
            "not in the file at all but long enough to be checked by thirteen words here",
            keys,
            set(),
            13,
        )
        == "unmatched"
    )


def test_status_unmatched_even_if_short_and_unchecked_even_if_listed() -> None:
    keys = {"tiny": ["x/0"]}
    assert status("short", keys, set(), 13) == "unmatched"
    assert status("tiny", keys, {"x/0"}, 13) == "unchecked"


def test_status_exactly_n_words_is_checked() -> None:
    thirteen = " ".join(LONG.split()[:13])
    assert status(thirteen, {thirteen: ["x/0"]}, set(), 13) == "clean"
    twelve = " ".join(LONG.split()[:12])
    assert status(twelve, {twelve: ["x/0"]}, set(), 13) == "unchecked"


def test_status_any_duplicate_id_contaminated() -> None:
    assert status(LONG, {LONG: ["x/0", "x/5"]}, {"x/5"}, 13) == "contaminated"


# score -------------------------------------------------------------------------------------------


def test_score_accuracy_mean_and_binomial_stderr() -> None:
    s = score("acc", [1.0, 0.0, 1.0, 1.0])
    assert s.n == 4 and s.value == pytest.approx(0.75)
    assert s.stderr == pytest.approx(math.sqrt(0.75 * 0.25 / 4))


def test_score_perplexity_from_loglikelihoods() -> None:
    s = score("perplexity", [-2.0, -4.0])
    assert s.n == 2 and s.value == pytest.approx(math.exp(3.0)) and s.stderr is None


def test_score_empty() -> None:
    assert score("acc", []) == Score(0, None, None) == score("perplexity", [])


# score_by_subset ----------------------------------------------------------------------------------


def _sample(text: str, acc: float, acc_norm: float) -> dict:
    return {"doc": {"text": text}, "acc": acc, "acc_norm": acc_norm, "perplexity": -1.0}


def test_score_by_subset_partitions_and_values() -> None:
    keys = {LONG: ["lambada/0"], OTHER: ["lambada/1"], "short one": ["lambada/2"]}
    samples = [
        {"doc": {"text": LONG}, "acc": 1, "perplexity": -1.0},
        {"doc": {"text": OTHER}, "acc": 0, "perplexity": -3.0},
        {
            "doc": {"text": OTHER.upper()},
            "acc": 1,
            "perplexity": -2.0,
        },  # same key after normalization
        {"doc": {"text": "short one"}, "acc": 1, "perplexity": -1.0},
        {"doc": {"text": LONG + " extra words make it unmatched"}, "acc": 0, "perplexity": -5.0},
    ]
    out = score_by_subset("lambada", samples, keys, {"lambada/0"}, 13)
    assert set(out) == set(SUBSETS) and all(set(v) == {"acc", "perplexity"} for v in out.values())
    counts = {s: out[s]["acc"].n for s in SUBSETS}
    assert counts == {"all": 5, "clean": 2, "contaminated": 1, "unchecked": 1, "unmatched": 1}
    assert sum(counts[s] for s in SUBSETS if s != "all") == counts["all"]
    assert out["clean"]["acc"].value == pytest.approx(0.5)
    assert out["all"]["acc"].value == pytest.approx(0.6)
    assert out["clean"]["perplexity"].value == pytest.approx(math.exp(2.5))
    assert out["unmatched"]["acc"].value == 0.0 and out["contaminated"]["acc"].value == 1.0


def test_score_by_subset_uses_task_metrics() -> None:
    samples = [{"doc": {"goal": "g", "sol1": "a", "sol2": "b"}, "acc": 1, "acc_norm": 0}]
    out = score_by_subset("piqa", samples, {}, set(), 13)
    assert set(out["all"]) == set(TASKS["piqa"].metrics) == {"acc", "acc_norm"}
    assert out["unmatched"]["acc"].n == 1 and out["clean"]["acc"].n == 0


def test_tasks_table() -> None:
    assert set(TASKS) == {"hellaswag", "arc_easy", "piqa", "lambada"}
    assert {k: v.headline for k, v in TASKS.items()} == {
        "hellaswag": "acc_norm",
        "arc_easy": "acc",
        "piqa": "acc_norm",
        "lambada": "acc",
    }
    assert all(v.headline in v.metrics and v.task == f"syrinx_{k}" for k, v in TASKS.items())
