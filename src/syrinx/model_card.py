"""Generate a Hugging Face model card from logged evidence only (Phase 10).

Every number in the card is read from an evidence file named in a manifest
(``configs/cards/<name>.yaml``): the training log, the training config, the full-split evaluation,
the export parity check, the contamination report and the benchmark results. Before rendering,
:func:`check_consistency` makes sure all of them describe the same model; the card is never
rendered from mismatched evidence.

PLAN.md keeps three kinds of numbers apart, and so does the card: "Our metrics" (loss, bits per
byte), "Benchmarks" (all and clean subsets, with chance levels) and "Comparisons" (none by default).
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from syrinx.config import SyrinxConfig, load_config, num_parameters

REQUIRED_FILES = ("train_config", "train_log", "eval", "parity", "contamination", "benchmarks")
REQUIRED_TEXT = ("name", "summary", "intended_use", "hardware")
DEFAULT_USAGE_PROMPT = "Once upon a time"
SMALL_SUBSET = 100
"""Contaminated subsets smaller than this (per benchmark) get a note that they are too small to read."""
EXPERIMENT_KEYS = ("training", "eval", "parity", "contamination", "benchmarks")
BENCHMARK_NAMES = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "piqa": "PIQA",
    "lambada": "LAMBADA (OpenAI)",
}


@dataclass(frozen=True)
class DatasetCredit:
    """A training dataset and what its license requires the card to say."""

    name: str
    hf_id: str
    license: str
    attribution: str


DATASETS: dict[str, DatasetCredit] = {
    "tinystories": DatasetCredit(
        "TinyStories",
        "roneneldan/TinyStories",
        "CDLA-Sharing-1.0",
        "TinyStories (Eldan and Li) is used under CDLA-Sharing-1.0; the data itself is not redistributed.",
    ),
    "fineweb-edu": DatasetCredit(
        "FineWeb-Edu",
        "HuggingFaceFW/fineweb-edu",
        "ODC-By-1.0",
        "FineWeb-Edu (Hugging Face, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is used "
        "under ODC-By v1.0 (https://opendatacommons.org/licenses/by/1-0/); it is derived from Common Crawl "
        "and subject to the Common Crawl Terms of Use.",
    ),
}
"""License text from docs/data.md ("License obligations")."""


@dataclass(frozen=True)
class Manifest:
    """Paths to the evidence and the descriptive text for one model's card."""

    name: str
    repo_id: str
    """Hub repository id used in the usage snippet (manifest ``repo_id``, default ``name``)."""
    export_dir: Path
    train_config: Path
    train_log: Path
    eval: Path
    parity: Path
    contamination: Path
    benchmarks: Path
    benchmark_dir: Path
    experiments: dict[str, str]
    datasets: list[str]
    hardware: str
    summary: str
    intended_use: str
    limitations: list[str]
    usage_prompt: str = DEFAULT_USAGE_PROMPT
    """Prompt in the usage snippet (manifest ``usage_prompt``); pick one in the training domain."""


def load_manifest(path: Path) -> Manifest:
    """Read and validate a card manifest.

    Raises:
        ValueError: If a required field is missing or empty (the message names it), an
            ``experiments`` key is missing, or a dataset key is not in :data:`DATASETS`.
        FileNotFoundError: If ``export_dir``, ``benchmark_dir`` or an evidence file is missing.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        *REQUIRED_TEXT,
        *REQUIRED_FILES,
        "export_dir",
        "benchmark_dir",
        "experiments",
        "datasets",
    ):
        if not raw.get(key):
            raise ValueError(f"manifest {path}: missing or empty field {key!r}")
    missing = [k for k in EXPERIMENT_KEYS if k not in raw["experiments"]]

    if missing:
        raise ValueError(f"manifest {path}: experiments is missing {missing}")
    unknown = [d for d in raw["datasets"] if d not in DATASETS]
    if unknown:
        raise ValueError(
            f"manifest {path}: unknown datasets {unknown}; expected {sorted(DATASETS)}"
        )

    paths = {k: Path(raw[k]) for k in (*REQUIRED_FILES, "export_dir", "benchmark_dir")}
    for key, value in paths.items():
        if not value.exists():
            raise FileNotFoundError(f"manifest {path}: {key} {value} does not exist.")

    return Manifest(
        name=raw["name"],
        repo_id=str(raw.get("repo_id") or raw["name"]),
        experiments={k: str(raw["experiments"][k]) for k in EXPERIMENT_KEYS},
        datasets=list(raw["datasets"]),
        hardware=raw["hardware"],
        summary=raw["summary"],
        intended_use=raw["intended_use"],
        limitations=list(raw.get("limitations") or []),
        usage_prompt=str(raw.get("usage_prompt") or DEFAULT_USAGE_PROMPT),
        **paths,
    )


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


SEGMENT_HEADER = re.compile(r"^# (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \|")
TIMESTAMP = re.compile(r"^(?:# end )?(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def _segment_seconds(start: str, end: str) -> float:
    # Log timestamps are local wall-clock time; both ends share a zone, so any fixed one gives
    # the right difference (a DST change inside a segment would be off by an hour).
    def parse(stamp: str) -> datetime:
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)

    return (parse(end) - parse(start)).total_seconds()


def parse_train_log(path: Path) -> dict[str, float]:
    """``{"end_step", "train_loss", "val_loss", "wall_seconds", "segments"}`` from a training log.

    The summary is the last line starting with ``{"end_step"`` (train.py's). A run that was stopped
    and resumed has one segment per ``# <timestamp> |`` header line, and its wall time is the sum
    over segments: the segment's ``real <seconds>`` line (``/usr/bin/time -p``) when it has one,
    otherwise header to last timestamped line (a segment stopped by request has no ``real`` line).
    A log without header lines is one segment timed by its last ``real`` line.

    Raises:
        ValueError: If the summary is missing, or a log without headers has no ``real`` line.
    """
    summary: dict[str, Any] | None = None
    segments: list[dict[str, Any]] = []
    unsegmented_real: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith('{"end_step"'):
            summary = json.loads(line)
        elif line.startswith("real "):
            if segments:
                segments[-1]["real"] = float(line.split()[1])
            else:
                unsegmented_real = float(line.split()[1])
        elif header := SEGMENT_HEADER.match(line):
            segments.append({"start": header[1], "end": header[1], "real": None})
        elif segments and (stamp := TIMESTAMP.match(line)):
            segments[-1]["end"] = stamp[1]

    if summary is None:
        raise ValueError(f"{path}: no train summary line")
    if segments:
        wall = sum(
            s["real"] if s["real"] is not None else _segment_seconds(s["start"], s["end"])
            for s in segments
        )
    elif unsegmented_real is not None:
        wall = unsegmented_real
    else:
        raise ValueError(f'{path}: no "# <timestamp> |" segment headers and no "real" timing line')

    return {
        "end_step": summary["end_step"],
        "train_loss": summary["train_loss"],
        "val_loss": summary["val_loss"],
        "wall_seconds": wall,
        "segments": max(len(segments), 1),
    }


def chance_levels(benchmark_dir: Path) -> dict[str, float | None]:
    """Expected accuracy of uniform guessing, from the pinned benchmark files: the mean of
    ``1 / number of choices`` (HellaSwag ``endings``, ARC ``choices.text``, PIQA 2); LAMBADA None."""
    out: dict[str, float | None] = {"lambada": None}
    for name, choices in (
        ("hellaswag", lambda r: len(r["endings"])),
        ("arc_easy", lambda r: len(r["choices"]["text"])),
        ("piqa", lambda r: 2),
    ):
        rows = [
            json.loads(line) for line in (benchmark_dir / f"{name}.jsonl").open(encoding="utf-8")
        ]
        out[name] = sum(1 / choices(r) for r in rows) / len(rows)
    return out


@dataclass(frozen=True)
class Evidence:
    """Everything the card states, loaded from the manifest's files."""

    export_config: dict[str, Any]
    model_sha256: str
    train_config: SyrinxConfig
    train: dict[str, float]
    eval: dict[str, Any]
    parity: dict[str, Any]
    contamination: dict[str, Any]
    benchmarks: dict[str, Any]
    chance: dict[str, float | None]
    file_hashes: dict[str, str]
    """SHA-256 of every evidence file, keyed like the manifest fields."""


def load_evidence(manifest: Manifest) -> Evidence:
    """Load every evidence file named in ``manifest``."""

    def read(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    return Evidence(
        export_config=read(manifest.export_dir / "config.json"),
        model_sha256=file_sha256(manifest.export_dir / "model.safetensors"),
        train_config=load_config(manifest.train_config),
        train=parse_train_log(manifest.train_log),
        eval=read(manifest.eval),
        parity=read(manifest.parity),
        contamination=read(manifest.contamination),
        benchmarks=read(manifest.benchmarks),
        chance=chance_levels(manifest.benchmark_dir),
        file_hashes={k: file_sha256(getattr(manifest, k)) for k in REQUIRED_FILES},
    )


def check_consistency(manifest: Manifest, evidence: Evidence) -> None:
    """Refuse evidence that does not describe one model.

    Checks, each with its own message: parity passed; parity ``out_dir`` is the export dir;
    parity and eval name the same ``checkpoint`` and ``step``; the training log's ``end_step`` is
    that step; benchmarks ``model_sha256`` is the export's ``model.safetensors``; benchmarks
    ``contamination_sha256`` is the manifest's contamination file; the export config has the
    training config's shape (hidden size, layers, heads, intermediate size, vocab, context).

    Raises:
        ValueError: On the first failed check.
    """
    p, e, b = evidence.parity, evidence.eval, evidence.benchmarks
    if not p.get("passed"):
        raise ValueError("parity check did not pass; refusing to write a card")
    if Path(p["out_dir"]) != manifest.export_dir:
        raise ValueError(
            f"parity out_dir {p['out_dir']} is not the export dir {manifest.export_dir}"
        )
    if (p["checkpoint"], p["step"]) != (e["checkpoint"], e["step"]):
        raise ValueError(
            f"parity checkpoint/step {p['checkpoint']}@{p['step']} != eval {e['checkpoint']}@{e['step']}"
        )
    if evidence.train["end_step"] != e["step"]:
        raise ValueError(
            f"training log end_step {evidence.train['end_step']} != evaluated step {e['step']}"
        )
    if b["model_sha256"] != evidence.model_sha256:
        raise ValueError("benchmarks model_sha256 does not match the export's model.safetensors")
    if b["contamination_sha256"] != evidence.file_hashes["contamination"]:
        raise ValueError("benchmarks were scored with a different contamination report")
    m, c = evidence.train_config.model, evidence.export_config
    shape = (m.d_model, m.n_layers, m.n_heads, m.d_ff, m.vocab_size, m.max_seq_len)
    exported = (
        c["hidden_size"],
        c["num_hidden_layers"],
        c["num_attention_heads"],
        c["intermediate_size"],
        c["vocab_size"],
        c["max_position_embeddings"],
    )
    if shape != exported:
        raise ValueError(f"export config shape {exported} != training config shape {shape}")


def fmt_loss(x: float) -> str:
    """4 decimals."""
    return f"{x:.4f}"


def fmt_int(x: float) -> str:
    """Thousands separators."""
    return f"{int(x):,}"


def fmt_wall(train: dict[str, float]) -> str:
    hours = f"{train['wall_seconds'] / 3600:.2f} h"
    n = int(train["segments"])
    if n == 1:
        return hours
    return f"{hours} in {n} segments (stopped and resumed; includes steps re-run after each resume)"


def fmt_pct(value: float, stderr: float | None = None) -> str:
    """Percent with 1 decimal, and ``± stderr`` in percentage points if given."""
    text = f"{100 * value:.1f}%"
    return text if stderr is None else f"{text} ± {100 * stderr:.1f}"


def fmt_ppl(x: float) -> str:
    """Integer with separators above 1,000, else 2 decimals."""
    return fmt_int(round(x)) if x >= 1000 else f"{x:.2f}"


def _cell(subsets: dict[str, Any], metric: str, subset: str) -> str:
    """One benchmark table cell: ``pct ± se (n=N)``, or ``n/a`` for an empty subset."""
    score = subsets[subset][metric]
    if score["value"] is None:
        return "n/a"
    return f"{fmt_pct(score['value'], score['stderr'])} (n={fmt_int(score['n'])})"


def _contaminated_note(benchmarks: dict[str, Any]) -> list[str]:
    """A note under the benchmark table when the contaminated subsets are too small to read
    (largest below :data:`SMALL_SUBSET`); empty when there are none or they are large enough."""
    sizes = [
        bench["subsets"]["contaminated"][bench["headline"]]["n"] for bench in benchmarks.values()
    ]
    largest = max(sizes, default=0)
    if largest == 0 or largest >= SMALL_SUBSET:
        return []
    note = (
        f"The contaminated subsets are small ({fmt_int(sum(sizes))} examples in total, at most "
        f"{fmt_int(largest)} per benchmark), so their scores have wide error bars and are not "
        "evidence of leakage either way. Compare All with Clean instead."
    )
    return [note, ""]


def render_card(manifest: Manifest, evidence: Evidence) -> str:
    """The card as Markdown with YAML front matter. Sections, in order: front matter, title and
    summary, Model details, Training, Our metrics, Benchmarks, Comparisons, How to use,
    Limitations, Evidence, License and attribution."""
    cfg, t, ev, x = evidence.train_config, evidence.train, evidence.eval, manifest.experiments
    m = cfg.model
    credits = [DATASETS[d] for d in manifest.datasets]
    front = {
        "license": "apache-2.0",
        "language": ["en"],
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "datasets": [d.hf_id for d in credits],
        "tags": ["syrinx", "llama", "from-scratch"],
    }
    lines = [
        "---",
        yaml.safe_dump(front, sort_keys=False).strip(),
        "---",
        "",
        f"# {manifest.name}",
        "",
    ]
    lines += [manifest.summary, ""]

    lines += ["## Model details", "", "| | |", "|---|---|"]
    lines += [
        "| Architecture | decoder-only transformer, `LlamaForCausalLM` (RMSNorm, RoPE, SwiGLU, tied embeddings) |",
        f"| Parameters | {fmt_int(num_parameters(m))} ({fmt_int(num_parameters(m, include_embedding=False))} non-embedding) |",
        f"| Layers / d_model / heads / d_ff | {m.n_layers} / {m.d_model} / {m.n_heads} / {m.d_ff} |",
        f"| Context length | {m.max_seq_len} tokens |",
        f"| Vocabulary | {fmt_int(m.vocab_size)} (byte-level BPE) |",
        f"| RoPE theta / norm eps | {m.rope_theta:g} / {m.norm_eps:g} |",
        "| Weights | fp32 safetensors |",
        "",
    ]

    tokens = t["end_step"] * cfg.tokens_per_step
    lines += ["## Training", "", "| | |", "|---|---|"]
    lines += [
        f"| Data | {', '.join(f'{d.name} ({d.license})' for d in credits)} |",
        f"| Tokens seen | {fmt_int(tokens)} ({fmt_int(t['end_step'])} steps x {fmt_int(cfg.tokens_per_step)} tokens) |",
        f"| Optimizer | AdamW, peak LR {cfg.train.lr:g}, min LR {cfg.train.min_lr:g}, {cfg.train.warmup_steps} warmup steps, cosine decay, weight decay {cfg.train.weight_decay:g} |",
        f"| Precision | {cfg.train.precision} |",
        f"| Hardware | {manifest.hardware} |",
        f"| Wall time | {fmt_wall(t)} |",
        f"| Final train / val loss | {fmt_loss(t['train_loss'])} / {fmt_loss(t['val_loss'])} (val: training-time estimate) |",
        "",
        f"Source: {x['training']}.",
        "",
    ]

    lines += ["## Our metrics", ""]
    lines += [
        (
            f"Full validation split, non-overlapping {ev['seq_len']}-token windows ({fmt_int(ev['tokens'])} target "
            f"tokens, {fmt_int(ev['bytes'])} bytes), {ev['precision']}:"
        ),
        "",
        "| Loss (nats/token) | Perplexity | Bits per byte |",
        "|---|---|---|",
        f"| {fmt_loss(ev['loss'])} | {fmt_ppl(ev['perplexity'])} | {fmt_loss(ev['bpb'])} |",
        "",
        f"Bits per byte is the only one of these comparable across tokenizers. Source: {x['eval']}.",
        "",
    ]

    b = evidence.benchmarks
    lines += ["## Benchmarks", ""]
    lines += [
        (
            f"lm-evaluation-harness {b['lm_eval_version']}, {b['num_fewshot']}-shot, {b['dtype']}, `<|endoftext|>` "
            'prepended, on the pinned benchmark files. ± is the binomial standard error. "Clean" excludes examples '
            f"that share a {evidence.contamination['n']}-gram with the training data and examples too short to check."
        ),
        "",
        "| Benchmark | Metric | All | Clean | Contaminated | Chance |",
        "|---|---|---|---|---|---|",
    ]
    for key, bench in b["benchmarks"].items():
        metric = bench["headline"]
        sub = bench["subsets"]
        chance = evidence.chance.get(key)
        lines.append(
            f"| {BENCHMARK_NAMES[key]} | {metric} | {_cell(sub, metric, 'all')} | {_cell(sub, metric, 'clean')} | {_cell(sub, metric, 'contaminated')} | "
            f"{'n/a' if chance is None else fmt_pct(chance)} |"
        )
    lines += ["", *_contaminated_note(b["benchmarks"])]
    lines += [f"Sources: {x['benchmarks']} (scores), {x['contamination']} (contamination).", ""]

    lines += ["## Comparisons", ""]
    lines += [
        (
            "No comparisons with other models are made: published scores use different training data, token "
            "counts, tokenizers and prompts, so they would not be like for like."
        ),
        "",
    ]

    lines += ["## How to use", "", "```python"]
    lines += [
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'tok = AutoTokenizer.from_pretrained("{manifest.repo_id}")',
        f'model = AutoModelForCausalLM.from_pretrained("{manifest.repo_id}", dtype=torch.float32)',
        "# Training documents always followed <|endoftext|>, so start prompts with it.",
        (
            f"ids = tok(tok.bos_token + {json.dumps(manifest.usage_prompt, ensure_ascii=False)}, "
            'return_tensors="pt")'
        ),
        "out = model.generate(**ids)  # uses the shipped generation_config",
        "print(tok.decode(out[0], skip_special_tokens=True))",
        "```",
        "",
    ]

    lines += ["## Intended use and limitations", "", manifest.intended_use, ""]
    lines += [f"- {item}" for item in manifest.limitations]
    lines += [""]

    lines += ["## Evidence", "", "| Item | Experiment | File | SHA-256 |", "|---|---|---|---|"]
    rows = (
        ("Training", "training", "train_log"),
        ("Evaluation", "eval", "eval"),
        ("Export parity", "parity", "parity"),
        ("Contamination", "contamination", "contamination"),
        ("Benchmarks", "benchmarks", "benchmarks"),
    )
    for label, exp_key, file_key in rows:
        lines.append(
            f"| {label} | {x[exp_key]} | `{getattr(manifest, file_key)}` | `{evidence.file_hashes[file_key]}` |"
        )
    lines += [f"| Weights | | `model.safetensors` | `{evidence.model_sha256}` |", ""]
    lines += [
        (
            f"Export parity: max absolute logit difference {evidence.parity['max_abs_logit_diff']:g} against the "
            f"Syrinx implementation (tolerance {evidence.parity['tolerance']:g})."
        ),
        "",
    ]

    lines += ["## License and attribution", "", "Code and weights: Apache-2.0.", ""]
    lines += [f"- {d.attribution}" for d in credits]
    lines += [""]
    return "\n".join(lines)
