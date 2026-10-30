"""Tests for ``syrinx.model_card``: manifest, evidence, consistency checks and rendering."""

import dataclasses
import json
import re
from pathlib import Path

import pytest
import yaml

from syrinx.config import load_config, num_parameters
from syrinx.model_card import (
    DATASETS,
    Evidence,
    Manifest,
    _contaminated_note,
    chance_levels,
    check_consistency,
    file_sha256,
    fmt_int,
    fmt_loss,
    fmt_pct,
    fmt_ppl,
    fmt_wall,
    load_evidence,
    load_manifest,
    parse_train_log,
    render_card,
)

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [
    "## Model details",
    "## Training",
    "## Our metrics",
    "## Benchmarks",
    "## Comparisons",
    "## How to use",
    "## Intended use and limitations",
    "## Evidence",
    "## License and attribution",
]


def _bench(value: float, n: int, stderr: float | None) -> dict:
    return {"n": n, "value": value, "stderr": stderr}


def _subsets(metric: str, all_: float, clean: float, cont: float | None) -> dict:
    empty = {"n": 0, "value": None, "stderr": None}
    out = {
        "all": {metric: _bench(all_, 1234, 0.0137)},
        "clean": {metric: _bench(clean, 1201, 0.0139)},
        "contaminated": {metric: empty if cont is None else _bench(cont, 33, 0.0712)},
        "unchecked": {metric: empty},
        "unmatched": {metric: empty},
    }
    if metric != "acc":  # a second metric that must NOT be reported as the headline
        for sub in out.values():
            sub["acc"] = _bench(0.7777, 4321, 0.0101)
    return out


def make_evidence(root: Path, **override) -> Path:  # type: ignore[no-untyped-def]
    """Write a full, consistent set of evidence files with distinctive numbers; return the manifest."""
    export = root / "export"
    export.mkdir(parents=True)
    (export / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "intermediate_size": 768,
                "vocab_size": 32768,
                "max_position_embeddings": 1024,
            }
        ),
        encoding="utf-8",
    )
    (export / "model.safetensors").write_bytes(b"fake weights")
    log = root / "train.txt"
    log.write_text(
        '# header\n{"end_step": 4321, "train_loss": 1.111111, "val_loss": 1.555555}\n'
        '{"end_step": 9000, "train_loss": 1.432109, "val_loss": 1.876543}\nreal 7777.25\nuser 1.0\n',
        encoding="utf-8",
    )
    ev = {
        "checkpoint": "runs/x/step_0009000.pt",
        "step": 9000,
        "seq_len": 1024,
        "precision": "bf16-autocast",
        "tokens": 2072001,
        "bytes": 8113339,
        "loss": 1.234567,
        "perplexity": 3.437891,
        "bpb": 0.611223,
    }
    (root / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    parity = {
        "checkpoint": "runs/x/step_0009000.pt",
        "step": 9000,
        "out_dir": str(export),
        "max_abs_logit_diff": 3.5e-06,
        "tolerance": 1e-4,
        "passed": True,
    }
    (root / "parity.json").write_text(json.dumps(parity), encoding="utf-8")
    contamination = {"n": 13, "benchmarks": {}}
    (root / "contamination.json").write_text(json.dumps(contamination), encoding="utf-8")
    bench_dir = root / "bench"
    bench_dir.mkdir()
    (bench_dir / "hellaswag.jsonl").write_text(
        json.dumps({"endings": ["a", "b", "c", "d"]}) + "\n", encoding="utf-8"
    )
    (bench_dir / "arc_easy.jsonl").write_text(
        json.dumps({"choices": {"text": ["a", "b", "c"]}})
        + "\n"
        + json.dumps({"choices": {"text": ["a", "b", "c", "d", "e"]}})
        + "\n",
        encoding="utf-8",
    )
    (bench_dir / "piqa.jsonl").write_text(json.dumps({"goal": "g"}) + "\n", encoding="utf-8")
    benchmarks = {
        "model_sha256": file_sha256(export / "model.safetensors"),
        "contamination_sha256": file_sha256(root / "contamination.json"),
        "lm_eval_version": "9.8.7",
        "num_fewshot": 0,
        "dtype": "float32",
        "benchmarks": {
            "hellaswag": {
                "headline": "acc_norm",
                "subsets": _subsets("acc_norm", 0.2873, 0.2911, 0.4242),
            },
            "arc_easy": {"headline": "acc", "subsets": _subsets("acc", 0.3317, 0.3333, None)},
            "piqa": {
                "headline": "acc_norm",
                "subsets": _subsets("acc_norm", 0.5671, 0.5689, 0.6061),
            },
            "lambada": {"headline": "acc", "subsets": _subsets("acc", 0.0913, 0.0913, None)},
        },
    }
    (root / "benchmarks.json").write_text(json.dumps(benchmarks), encoding="utf-8")
    manifest = {
        "name": "syrinx-test",
        "repo_id": "someone/syrinx-test",
        "export_dir": str(export),
        "train_config": str(ROOT / "configs" / "syrinx-10m.yaml"),
        "train_log": str(log),
        "eval": str(root / "eval.json"),
        "parity": str(root / "parity.json"),
        "contamination": str(root / "contamination.json"),
        "benchmarks": str(root / "benchmarks.json"),
        "benchmark_dir": str(bench_dir),
        "experiments": {
            "training": "EXP-A",
            "eval": "EXP-B",
            "parity": "EXP-C",
            "contamination": "EXP-D",
            "benchmarks": "EXP-E",
        },
        "datasets": ["tinystories"],
        "hardware": "a test laptop",
        "summary": "A test model summary.",
        "intended_use": "Testing the card.",
        "limitations": ["First limitation.", "Second limitation."],
    }
    manifest.update(override)
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def _load(root: Path, **override) -> tuple[Manifest, Evidence]:  # type: ignore[no-untyped-def]
    manifest = load_manifest(make_evidence(root, **override))
    return manifest, load_evidence(manifest)


# manifest -------------------------------------------------------------------------------------------


def test_load_manifest_fields(tmp_path: Path) -> None:
    m = load_manifest(make_evidence(tmp_path))
    assert m.name == "syrinx-test" and m.repo_id == "someone/syrinx-test"
    assert m.export_dir == tmp_path / "export" and m.benchmarks == tmp_path / "benchmarks.json"
    assert m.experiments["eval"] == "EXP-B" and m.datasets == ["tinystories"]
    assert m.limitations == ["First limitation.", "Second limitation."]


def test_load_manifest_repo_id_defaults_to_name(tmp_path: Path) -> None:
    path = make_evidence(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["repo_id"]
    path.write_text(yaml.safe_dump(raw))
    assert load_manifest(path).repo_id == "syrinx-test"


@pytest.mark.parametrize(
    "field", ["summary", "hardware", "eval", "benchmarks", "experiments", "datasets"]
)
def test_load_manifest_missing_field_named(tmp_path: Path, field: str) -> None:
    path = make_evidence(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw[field]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=field):
        load_manifest(path)


def test_load_manifest_rejects_bad_experiments_and_datasets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parity"):
        load_manifest(
            make_evidence(
                tmp_path / "a",
                experiments={"training": "x", "eval": "x", "contamination": "x", "benchmarks": "x"},
            )
        )
    with pytest.raises(ValueError, match="unknown datasets"):
        load_manifest(make_evidence(tmp_path / "b", datasets=["wikipedia"]))


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    path = make_evidence(tmp_path)
    (tmp_path / "eval.json").unlink()
    with pytest.raises(FileNotFoundError, match="eval"):
        load_manifest(path)


# evidence -------------------------------------------------------------------------------------------


def test_parse_train_log_uses_last_summary(tmp_path: Path) -> None:
    make_evidence(tmp_path)
    assert parse_train_log(tmp_path / "train.txt") == {
        "end_step": 9000,
        "train_loss": 1.432109,
        "val_loss": 1.876543,
        "wall_seconds": 7777.25,
        "segments": 1,
    }


SUMMARY = '{"end_step": 30, "train_loss": 2.0, "val_loss": 2.5}'


def test_parse_train_log_sums_resumed_segments(tmp_path: Path) -> None:
    """A run stopped twice by request (no "real" line) and finished once: every segment counts,
    not only the last one."""
    path = tmp_path / "train.txt"
    path.write_text(
        "# 2026-10-14 10:00:00 | git abc | AC | train attempt 1: --out-dir runs/x\n"
        "2026-10-14 10:00:01 INFO    __main__: step 10 loss 5.0\n"
        "2026-10-14 11:00:00 INFO    __main__: step 20 loss 4.0\n"
        "# 2026-10-15 09:00:00 | git abc | AC | train attempt 1: --resume\n"
        "2026-10-15 09:30:00 INFO    __main__: step 20 loss 4.0\n"
        "# 2026-10-16 09:00:00 | git abc | AC | train attempt 1: --resume\n"
        "2026-10-16 09:10:00 INFO    __main__: step 30 loss 2.0\n"
        f"{SUMMARY}\n"
        "real 700.50\n"
        "exit 0\n",
        encoding="utf-8",
    )
    train = parse_train_log(path)
    assert train["segments"] == 3
    assert train["wall_seconds"] == pytest.approx(3600 + 1800 + 700.5)


def test_parse_train_log_single_segment_prefers_real_line(tmp_path: Path) -> None:
    """The 10m log layout: one header, a "real" line, then "# end <timestamp>"."""
    path = tmp_path / "train.txt"
    path.write_text(
        "# 2026-10-12 21:59:12 | git 6c8739d | AC\n# command: train.py\n"
        f"2026-10-12 22:00:00 INFO step 10\n{SUMMARY}\nreal 8940.55\n"
        "# end 2026-10-13 00:28:13 | AC\n",
        encoding="utf-8",
    )
    assert parse_train_log(path)["wall_seconds"] == 8940.55
    assert parse_train_log(path)["segments"] == 1


def test_resumed_wall_time_is_labelled_in_card() -> None:
    assert fmt_wall({"wall_seconds": 7200.0, "segments": 1}) == "2.00 h"
    assert fmt_wall({"wall_seconds": 75504.0, "segments": 8}).startswith("20.97 h in 8 segments")


def test_parse_train_log_missing_lines(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text('{"end_step": 1, "train_loss": 1, "val_loss": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="real"):
        parse_train_log(path)
    path.write_text("2026-10-12 22:00:00 INFO step 10\nreal 5.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        parse_train_log(path)


def test_chance_levels(tmp_path: Path) -> None:
    make_evidence(tmp_path)
    chance = chance_levels(tmp_path / "bench")
    assert chance["hellaswag"] == pytest.approx(0.25) and chance["piqa"] == pytest.approx(0.5)
    assert chance["arc_easy"] == pytest.approx((1 / 3 + 1 / 5) / 2) and chance["lambada"] is None


def test_load_evidence(tmp_path: Path) -> None:
    m, e = _load(tmp_path)
    assert e.model_sha256 == file_sha256(tmp_path / "export" / "model.safetensors")
    assert e.train["end_step"] == 9000 and e.eval["loss"] == 1.234567
    assert set(e.file_hashes) == {
        "train_config",
        "train_log",
        "eval",
        "parity",
        "contamination",
        "benchmarks",
    }
    assert e.file_hashes["eval"] == file_sha256(m.eval)


# consistency ------------------------------------------------------------------------------------------


def test_consistent_evidence_passes(tmp_path: Path) -> None:
    check_consistency(*_load(tmp_path))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.parity.update(passed=False), "parity check did not pass"),
        (lambda e: e.parity.update(out_dir="exports/other"), "out_dir"),
        (lambda e: e.parity.update(step=8000), "checkpoint/step"),
        (lambda e: e.parity.update(checkpoint="runs/y/step_0009000.pt"), "checkpoint/step"),
        (lambda e: e.train.update(end_step=8999), "end_step"),
        (lambda e: e.benchmarks.update(model_sha256="0" * 64), "model_sha256"),
        (lambda e: e.benchmarks.update(contamination_sha256="0" * 64), "contamination"),
        (lambda e: e.export_config.update(hidden_size=512), "shape"),
        (lambda e: e.export_config.update(max_position_embeddings=2048), "shape"),
    ],
)
def test_inconsistent_evidence_rejected(tmp_path: Path, change, message: str) -> None:  # type: ignore[no-untyped-def]
    m, e = _load(tmp_path)
    change(e)
    with pytest.raises(ValueError, match=message):
        check_consistency(m, e)


# formatting -------------------------------------------------------------------------------------------


def test_formatters() -> None:
    assert fmt_loss(1.23456) == "1.2346" and fmt_int(1234567) == "1,234,567"
    assert fmt_pct(0.2586) == "25.9%" and fmt_pct(0.2586, 0.0044) == "25.9% ± 0.4"
    assert fmt_ppl(75330.4) == "75,330" and fmt_ppl(4.7782) == "4.78"
    assert (
        fmt_ppl(123.456) == "123.46" and fmt_ppl(999.994) == "999.99" and fmt_ppl(1000.4) == "1,000"
    )


# rendering --------------------------------------------------------------------------------------------


def test_card_sections_in_order(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path))
    positions = [card.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    assert card.count("## Comparisons") == 1


def test_front_matter(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path))
    assert card.startswith("---\n")
    front = yaml.safe_load(card.split("---")[1])
    assert front["license"] == "apache-2.0" and front["library_name"] == "transformers"
    assert front["pipeline_tag"] == "text-generation" and front["datasets"] == [
        "roneneldan/TinyStories"
    ]


def test_our_metrics_use_full_split_eval_not_training_val(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path))
    ours = card[card.index("## Our metrics") : card.index("## Benchmarks")]
    assert "| 1.2346 | 3.44 | 0.6112 |" in ours
    assert (
        "1.8765" not in ours
        and "1.8765" in card[card.index("## Training") : card.index("## Our metrics")]
    )
    assert "2,072,001" in ours and "8,113,339" in ours


def test_benchmark_rows(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path))
    bench = card[card.index("## Benchmarks") : card.index("## Comparisons")]
    assert (
        "| HellaSwag | acc_norm | 28.7% ± 1.4 (n=1,234) | 29.1% ± 1.4 (n=1,201) | 42.4% ± 7.1 (n=33) | 25.0% |"
        in bench
    )
    assert (
        "| ARC-Easy | acc | 33.2% ± 1.4 (n=1,234) | 33.3% ± 1.4 (n=1,201) | n/a | 26.7% |" in bench
    )
    assert (
        "| LAMBADA (OpenAI) | acc | 9.1% ± 1.4 (n=1,234) | 9.1% ± 1.4 (n=1,201) | n/a | n/a |"
        in bench
    )
    assert "77.8%" not in bench and "4,321" not in bench  # the non-headline acc is not shown
    assert (
        "9.8.7" in bench
        and "0-shot" in bench
        and "13-gram" in bench
        and "EXP-E" in bench
        and "EXP-D" in bench
    )


def test_comparisons_default_text(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path))
    comp = card[card.index("## Comparisons") : card.index("## How to use")]
    assert "No comparisons with other models are made" in comp
    assert "%" not in comp and not re.search(r"\d", comp)


def test_usage_snippet(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path))
    use = card[card.index("## How to use") : card.index("## Intended use")]
    assert 'from_pretrained("someone/syrinx-test")' in use and "tok.bos_token +" in use
    assert not re.search(r"\d", use.replace("float32", "").replace("out[0]", ""))


def test_training_and_details(tmp_path: Path) -> None:
    m, e = _load(tmp_path)
    card = render_card(m, e)
    cfg = load_config(ROOT / "configs" / "syrinx-10m.yaml")
    assert fmt_int(num_parameters(cfg.model)) in card
    assert fmt_int(num_parameters(cfg.model, include_embedding=False)) in card
    assert (
        f"{fmt_int(9000 * cfg.tokens_per_step)} (9,000 steps x {fmt_int(cfg.tokens_per_step)} tokens)"
        in card
    )
    assert "2.16 h" in card and "1.4321 / 1.8765" in card and "a test laptop" in card


def test_limitations_evidence_and_attribution(tmp_path: Path) -> None:
    m, e = _load(tmp_path)
    card = render_card(m, e)
    assert (
        "- First limitation." in card
        and "- Second limitation." in card
        and "Testing the card." in card
    )
    for key in ("train_log", "eval", "parity", "contamination", "benchmarks"):
        assert f"`{e.file_hashes[key]}`" in card and f"`{getattr(m, key)}`" in card
    assert f"`{e.model_sha256}`" in card and "3.5e-06" in card
    assert DATASETS["tinystories"].attribution in card and "ODC-By" not in card


def test_fineweb_attribution(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path, datasets=["fineweb-edu"]))
    lic = card[card.index("## License and attribution") :]
    assert DATASETS["fineweb-edu"].attribution in lic
    assert "ODC-By" in lic and "Common Crawl" in lic and "HuggingFaceFW/fineweb-edu" in card


def _numbers(text: str) -> set[str]:
    text = re.sub(r"\b[0-9a-f]{64}\b", " ", text)  # SHA-256 digests are not numbers
    return set(re.findall(r"(?<![A-Za-z_\d.])\d[\d,]*(?:\.\d+)?(?:e-?\d+)?", text))


def _scaled(subsets: dict, f: float) -> dict:
    out = {}
    for name, metrics in subsets.items():
        out[name] = {}
        for metric, sc in metrics.items():
            if sc["value"] is None:
                out[name][metric] = sc
            else:
                out[name][metric] = {
                    "n": sc["n"] * 3 + 1,
                    "value": sc["value"] * f,
                    "stderr": sc["stderr"] * f,
                }
    return out


def test_every_number_comes_from_the_evidence(tmp_path: Path) -> None:
    """Provenance: render twice, the second time with every evidence number changed. Any number in
    both cards must come from a static input (training config, manifest text, license text);
    otherwise it is hard-coded in the template."""
    m1, e1 = _load(tmp_path / "a")
    m2, e2 = _load(tmp_path / "b")
    bench = {**e2.benchmarks, "lm_eval_version": "5.4.3", "num_fewshot": 2}
    bench["benchmarks"] = {
        k: {**v, "subsets": _scaled(v["subsets"], 0.83)} for k, v in bench["benchmarks"].items()
    }
    e2 = dataclasses.replace(
        e2,
        train={k: v * 1.5 for k, v in e2.train.items()},
        eval={
            **e2.eval,
            **{k: e2.eval[k] * 1.7 for k in ("loss", "perplexity", "bpb", "tokens", "bytes")},
            "seq_len": 512,
        },
        benchmarks=bench,
        chance={k: None if v is None else v * 0.9 for k, v in e2.chance.items()},
        contamination={**e2.contamination, "n": 17},
        parity={**e2.parity, "max_abs_logit_diff": 7.7e-07, "tolerance": 3e-05},
    )
    card1, card2 = render_card(m1, e1), render_card(m2, e2)
    static = "\n".join(
        [
            (tmp_path / "a" / "manifest.yaml").read_text(encoding="utf-8"),
            (ROOT / "configs" / "syrinx-10m.yaml").read_text(encoding="utf-8"),
            (m1.export_dir / "config.json").read_text(encoding="utf-8"),
            *(d.attribution for d in DATASETS.values()),
        ]
    )
    cfg = load_config(ROOT / "configs" / "syrinx-10m.yaml")
    allowed = _numbers(static) | {"2.0"}  # "Apache-2.0"
    allowed |= {
        fmt_int(num_parameters(cfg.model)),
        fmt_int(num_parameters(cfg.model, include_embedding=False)),
    }
    allowed |= {fmt_int(cfg.tokens_per_step), fmt_int(cfg.model.vocab_size)}
    allowed |= {
        f"{x:g}"
        for x in (
            cfg.train.lr,
            cfg.train.min_lr,
            cfg.train.weight_decay,
            cfg.model.rope_theta,
            cfg.model.norm_eps,
        )
    }
    unexplained = (_numbers(card1) & _numbers(card2)) - allowed
    assert unexplained == set()
    assert _numbers(card1) != _numbers(card2)


def test_real_10m_evidence_is_consistent() -> None:
    manifest_path = ROOT / "configs" / "cards" / "syrinx-10m.yaml"
    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError as err:
        pytest.skip(f"local evidence not available: {err}")
    evidence = load_evidence(manifest)
    check_consistency(manifest, evidence)
    card = render_card(manifest, evidence)
    assert "| 1.5641 | 4.78 | 0.5769 |" in card  # EXP-014
    assert "11,798,784" in card


def test_usage_prompt_defaults_and_can_be_set(tmp_path: Path) -> None:
    """The snippet's prompt comes from the manifest, so each model can show text in its domain."""
    assert 'tok.bos_token + "Once upon a time"' in render_card(*_load(tmp_path / "a"))
    card = render_card(*_load(tmp_path / "b", usage_prompt='The "water" cycle is'))
    assert 'tok.bos_token + "The \\"water\\" cycle is", return_tensors="pt")' in card


def test_small_contaminated_subsets_get_a_note(tmp_path: Path) -> None:
    card = render_card(*_load(tmp_path))
    assert "The contaminated subsets are small" in card and "at most 33 per benchmark" in card


def _contaminated(n: int) -> dict:
    return {"headline": "acc", "subsets": {"contaminated": {"acc": {"n": n}}}}


@pytest.mark.parametrize(("sizes", "noted"), [((0, 0), False), ((3, 40), True), ((40, 100), False)])
def test_contaminated_note_threshold(sizes: tuple[int, int], noted: bool) -> None:
    benchmarks = {f"b{i}": _contaminated(n) for i, n in enumerate(sizes)}
    assert bool(_contaminated_note(benchmarks)) is noted
