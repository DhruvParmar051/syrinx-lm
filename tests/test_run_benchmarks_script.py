"""Tests for ``scripts/run_benchmarks.py`` and the local lm-eval task configs."""

import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import ModuleType

import pytest
import torch
import yaml

from syrinx.benchmarks import TASKS
from syrinx.config import config_from_dict, config_to_dict, load_config
from syrinx.export import export_checkpoint
from syrinx.model import SyrinxLM
from syrinx.tokenizer import file_sha256
from syrinx.training.checkpoint import checkpoint_path, save_checkpoint
from syrinx.training.optim import build_optimizer

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "configs" / "lm_eval"
TOKENIZER = ROOT / "tokenizer" / "syrinx-bpe-32k" / "tokenizer.json"
LONG = "a girl stands at the edge of the diving board and looks down at the blue water below her"
STOCK = {
    "hellaswag": "hellaswag/hellaswag.yaml",
    "arc_easy": "arc/arc_easy.yaml",
    "piqa": "piqa/piqa.yaml",
    "lambada": "lambada/lambada_openai.yaml",
}
COPIED_FIELDS = ("output_type", "doc_to_text", "doc_to_target", "doc_to_choice", "metric_list")


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor(
    "!function", lambda loader, node: f"!function {loader.construct_scalar(node)}"
)


def _yaml(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_benchmarks_script", ROOT / "scripts" / "run_benchmarks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# local task configs -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TASKS))
def test_local_task_config_matches_stock(name: str) -> None:
    import lm_eval

    ours = _yaml(TASK_DIR / f"{TASKS[name].task}.yaml")
    stock = _yaml(Path(lm_eval.__file__).parent / "tasks" / STOCK[name])
    assert ours["task"] == TASKS[name].task
    for field in COPIED_FIELDS:
        assert ours.get(field) == stock.get(field), field
    assert ours.get("process_docs") == stock.get("process_docs")
    assert ours["dataset_path"] == "json"
    files = ours["dataset_kwargs"]["data_files"]
    split = next(iter(files))
    assert files == {split: f"data_cache/benchmarks/{name}.jsonl"}
    assert ours.get(f"{split}_split") == split
    assert "fewshot_split" not in ours and "num_fewshot" not in ours


def test_hellaswag_utils_reexports_stock_processing() -> None:
    from lm_eval.tasks.hellaswag import utils as stock

    spec = importlib.util.spec_from_file_location("local_utils", TASK_DIR / "utils.py")
    assert spec is not None and spec.loader is not None
    local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local)
    assert local.process_docs is stock.process_docs


# parse_args / run_harness ---------------------------------------------------------------------


def test_parse_args_defaults(script: ModuleType) -> None:
    args = script.parse_args(["--model-dir", "m"])
    assert args.model_dir == Path("m") and args.benchmark == list(TASKS)
    assert args.contamination == script.DEFAULT_CONTAMINATION
    assert (args.benchmark_dir, args.task_dir) == (
        script.DEFAULT_BENCHMARK_DIR,
        script.DEFAULT_TASK_DIR,
    )
    assert (args.device, args.batch_size, args.limit, args.out) == (None, 32, None, None)
    assert script.parse_args(["--model-dir", "m", "--benchmark", "piqa"]).benchmark == ["piqa"]
    with pytest.raises(SystemExit):
        script.parse_args(["--model-dir", "m", "--benchmark", "mmlu"])


def test_run_harness_arguments(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lm_eval

    seen: dict = {}
    monkeypatch.setattr(lm_eval, "simple_evaluate", lambda **kw: seen.update(kw) or {"ok": True})
    assert script.run_harness(Path("exports/m"), ["syrinx_piqa"], TASK_DIR, "cpu", 8, 3) == {
        "ok": True
    }
    assert seen["model"] == "hf"
    assert seen["model_args"] == "pretrained=exports/m,dtype=float32,add_bos_token=True"
    assert seen["tasks"] == ["syrinx_piqa"] and seen["num_fewshot"] == 0
    assert (seen["device"], seen["batch_size"], seen["limit"]) == ("cpu", 8, 3)
    assert seen["log_samples"] is True and seen["bootstrap_iters"] == 0
    assert "syrinx_piqa" in seen["task_manager"].all_tasks


# main with a fake harness ------------------------------------------------------------------------


@pytest.fixture
def fake_setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    bench = tmp_path / "bench"
    bench.mkdir()
    _jsonl(
        bench / "lambada.jsonl",
        [{"text": LONG}, {"text": "short text"}, {"text": LONG + " again and again"}],
    )
    _jsonl(bench / "piqa.jsonl", [{"goal": LONG, "sol1": "a", "sol2": "b"}])
    report = {
        "n": 13,
        "benchmarks": {
            "lambada": {"contaminated_ids": ["lambada/2"]},
            "piqa": {"contaminated_ids": []},
        },
    }
    contamination = tmp_path / "contamination.json"
    contamination.write_text(json.dumps(report), encoding="utf-8")
    outputs = {
        "results": {
            "syrinx_lambada": {"acc,none": 0.5, "perplexity,none": 10.0},
            "syrinx_piqa": {"acc,none": 1.0, "acc_norm,none": 0.0},
        },
        "samples": {
            "syrinx_lambada": [
                {"doc": {"text": LONG}, "acc": 1, "perplexity": -1.0},
                {"doc": {"text": "short text"}, "acc": 0, "perplexity": -2.0},
                {"doc": {"text": LONG + " again and again"}, "acc": 1, "perplexity": -3.0},
                {
                    "doc": {"text": "something else entirely that is not in the benchmark file"},
                    "acc": 0,
                    "perplexity": -4.0,
                },
            ],
            "syrinx_piqa": [
                {"doc": {"goal": LONG, "sol1": "a", "sol2": "b"}, "acc": 1, "acc_norm": 0}
            ],
        },
    }
    argv = [
        "--model-dir",
        str(model),
        "--benchmark-dir",
        str(bench),
        "--contamination",
        str(contamination),
    ]
    argv += ["--benchmark", "lambada", "--benchmark", "piqa", "--device", "cpu"]
    return argv, model, contamination, outputs


def _record(script: ModuleType, capsys, argv: list[str]) -> dict:  # type: ignore[no-untyped-def]
    assert script.main(argv) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(lines) == 1
    return json.loads(lines[0])


def test_main_record(
    script: ModuleType, fake_setup, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    argv, model, contamination, outputs = fake_setup
    calls: list = []
    monkeypatch.setattr(script, "run_harness", lambda *a: calls.append(a) or outputs)
    for key in script.OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    record = _record(script, capsys, argv)
    assert calls == [
        (model, ["syrinx_lambada", "syrinx_piqa"], script.DEFAULT_TASK_DIR, "cpu", 32, None)
    ]
    assert all(os.environ[k] == v for k, v in script.OFFLINE_ENV.items())
    assert record["model_sha256"] == file_sha256(model / "model.safetensors")
    assert record["config_sha256"] == file_sha256(model / "config.json")
    assert record["contamination_sha256"] == file_sha256(contamination)
    assert (record["num_fewshot"], record["add_bos_token"], record["dtype"]) == (0, True, "float32")
    assert record["device"] == "cpu" and record["limit"] is None and record["seconds"] >= 0
    assert record["lm_eval_version"] and record["transformers_version"] and record["torch_version"]
    lam = record["benchmarks"]["lambada"]
    assert (lam["task"], lam["headline"]) == ("syrinx_lambada", "acc")
    assert lam["task_config_sha256"] == file_sha256(TASK_DIR / "syrinx_lambada.yaml")
    assert lam["harness"] == {"acc": 0.5, "perplexity": 10.0}
    counts = {s: lam["subsets"][s]["acc"]["n"] for s in lam["subsets"]}
    assert counts == {"all": 4, "clean": 1, "contaminated": 1, "unchecked": 1, "unmatched": 1}
    assert lam["subsets"]["clean"]["acc"] == {"n": 1, "value": 1.0, "stderr": 0.0}
    assert record["benchmarks"]["piqa"]["subsets"]["clean"]["acc_norm"]["value"] == 0.0


def test_main_never_overwrites_out(
    script: ModuleType, fake_setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    argv, _, _, outputs = fake_setup
    monkeypatch.setattr(script, "run_harness", lambda *a: outputs)
    out = tmp_path / "res" / "bench.json"
    printed = _record(script, capsys, [*argv, "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == printed
    with pytest.raises(FileExistsError):
        script.main([*argv, "--out", str(out)])


# end to end with the real harness (tiny model, tiny local files) -------------------------------------


def test_end_to_end_real_harness(script: ModuleType, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    raw = config_to_dict(load_config(ROOT / "configs" / "syrinx-10m.yaml"))
    raw["model"].update(
        {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "max_seq_len": 256}
    )
    raw["data"]["seq_len"] = 256
    config = config_from_dict(raw)
    torch.manual_seed(0)
    model = SyrinxLM(config.model)
    ckpt = checkpoint_path(tmp_path / "ckpt", 1)
    save_checkpoint(
        ckpt, model, build_optimizer(model, config.train), 1, {}, config_to_dict(config)
    )
    export_checkpoint(ckpt, TOKENIZER, tmp_path / "export")

    bench, tasks = tmp_path / "bench", tmp_path / "tasks"
    bench.mkdir()
    tasks.mkdir()
    shutil.copy(TASK_DIR / "utils.py", tasks / "utils.py")
    for name, spec in TASKS.items():
        text = (TASK_DIR / f"{spec.task}.yaml").read_text(encoding="utf-8")
        (tasks / f"{spec.task}.yaml").write_text(
            text.replace("data_cache/benchmarks", str(bench)), encoding="utf-8"
        )
    _jsonl(
        bench / "hellaswag.jsonl",
        [
            {
                "ind": i,
                "activity_label": "Roof",
                "ctx_a": f"A man {i} climbs",
                "ctx_b": "the ladder and",
                "ctx": f"A man {i} climbs the ladder and",
                "endings": ["falls.", "waves.", "sits.", "runs."],
                "source_id": "s",
                "split": "val",
                "split_type": "indomain",
                "label": "1",
            }
            for i in range(3)
        ],
    )
    _jsonl(
        bench / "arc_easy.jsonl",
        [
            {
                "id": f"Q{i}",
                "question": f"What is {i}?",
                "choices": {"text": ["one", "two"], "label": ["A", "B"]},
                "answerKey": "A",
            }
            for i in range(3)
        ],
    )
    _jsonl(
        bench / "piqa.jsonl",
        [
            {"goal": f"Open a jar {i}", "sol1": "twist the lid", "sol2": "hit it", "label": 0}
            for i in range(3)
        ],
    )
    _jsonl(
        bench / "lambada.jsonl", [{"text": f"the cat sat on the mat number {i}"} for i in range(3)]
    )
    report = {"n": 13, "benchmarks": {n: {"contaminated_ids": []} for n in TASKS}}
    contamination = tmp_path / "c.json"
    contamination.write_text(json.dumps(report), encoding="utf-8")

    argv = [
        "--model-dir",
        str(tmp_path / "export"),
        "--benchmark-dir",
        str(bench),
        "--task-dir",
        str(tasks),
    ]
    argv += ["--contamination", str(contamination), "--device", "cpu", "--batch-size", "4"]
    record = _record(script, capsys, argv)
    for name, spec in TASKS.items():
        b = record["benchmarks"][name]
        assert b["subsets"]["all"][spec.headline]["n"] == 3, name
        assert b["subsets"]["unmatched"][spec.headline]["n"] == 0, name  # every harness doc matched
        assert b["subsets"]["all"]["acc"]["value"] == pytest.approx(b["harness"]["acc"]), name


def test_main_passes_limit_and_batch_size(
    script: ModuleType, fake_setup, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    argv, _, _, outputs = fake_setup
    calls: list = []
    monkeypatch.setattr(script, "run_harness", lambda *a: calls.append(a) or outputs)
    record = _record(script, capsys, [*argv, "--limit", "2", "--batch-size", "7"])
    assert calls[0][4:] == (7, 2)
    assert (record["limit"], record["batch_size"]) == (2, 7)
