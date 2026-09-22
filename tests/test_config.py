"""Tests for ``syrinx.config``. Spec: docs/reference/config-spec.md."""

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from syrinx.config import (
    ConfigError,
    SyrinxConfig,
    config_from_dict,
    config_to_dict,
    load_config,
    num_parameters,
    save_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

PINNED_PARAMS: dict[str, int] = {
    "syrinx-10m": 11_798_784,
    "syrinx-50m": 50_866_688,
    "syrinx-125m": 124_275_456,
}
"""Pinned total parameter counts (PLAN.md Gate 0). Never edit to make a test pass."""

PLANNED_TOKENS: dict[str, float] = {"syrinx-50m": 1.0e9, "syrinx-125m": 2.5e9}
"""Token budgets from PLAN.md §1."""


def _raw(name: str = "syrinx-10m") -> dict[str, Any]:
    """A fresh, mutable copy of a shipped config as parsed YAML."""
    with (CONFIG_DIR / f"{name}.yaml").open(encoding="utf-8") as fh:
        return copy.deepcopy(yaml.safe_load(fh))


# --- shipped configs and parameter counts ---------------------------------------------


@pytest.mark.parametrize("name", sorted(PINNED_PARAMS))
def test_param_count_pinned(name: str) -> None:
    """Each shipped config's analytic parameter count equals the pinned number."""
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert config.name == name
    assert num_parameters(config.model) == PINNED_PARAMS[name]


@pytest.mark.parametrize("name", sorted(PINNED_PARAMS))
def test_non_embedding_count(name: str) -> None:
    """``include_embedding=False`` subtracts exactly V*d for tied embeddings."""
    model = load_config(CONFIG_DIR / f"{name}.yaml").model
    embedding = model.vocab_size * model.d_model
    assert num_parameters(model, include_embedding=False) == PINNED_PARAMS[name] - embedding


def test_untied_adds_output_projection() -> None:
    """Untying adds exactly V*d to the total; the non-embedding count is unchanged."""
    model = load_config(CONFIG_DIR / "syrinx-10m.yaml").model
    untied = dataclasses.replace(model, tie_embeddings=False)
    extra = model.vocab_size * model.d_model
    assert num_parameters(untied) == num_parameters(model) + extra
    assert num_parameters(untied, False) == num_parameters(model, False)


def test_param_count_formula_by_hand() -> None:
    """A tiny config matches the formula written out term by term."""
    raw = _raw()
    raw["model"].update(vocab_size=10, d_model=4, n_layers=2, n_heads=2, d_ff=8)
    model = config_from_dict(raw).model
    per_block = 4 * 16 + 3 * 4 * 8 + 2 * 4  # attention + SwiGLU + two norms = 168
    assert num_parameters(model) == 10 * 4 + 2 * per_block + 4  # emb + blocks + final norm


@pytest.mark.parametrize("name", sorted(PLANNED_TOKENS))
def test_token_budget_matches_plan(name: str) -> None:
    """max_steps * tokens_per_step reaches the PLAN.md budget (within one step)."""
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    total = config.train.max_steps * config.tokens_per_step
    assert PLANNED_TOKENS[name] <= total < PLANNED_TOKENS[name] + config.tokens_per_step


def test_derived_values() -> None:
    """head_dim and tokens_per_step for syrinx-125m."""
    config = load_config(CONFIG_DIR / "syrinx-125m.yaml")
    assert config.model.head_dim == 64
    assert config.tokens_per_step == 8 * 64 * 1024


# --- key and type validation ----------------------------------------------------------


@pytest.mark.parametrize("section", ["model", "train", "data"])
def test_unknown_key_rejected(section: str) -> None:
    """An extra key raises ConfigError naming the key and the section."""
    raw = _raw()
    raw[section]["dropuot"] = 0.1
    with pytest.raises(ConfigError, match=rf"\[{section}\].*dropuot"):
        config_from_dict(raw)


def test_unknown_top_level_key_rejected() -> None:
    """An extra top-level key raises ConfigError."""
    raw = _raw()
    raw["optimizer"] = {}
    with pytest.raises(ConfigError, match="optimizer"):
        config_from_dict(raw)


def test_all_unknown_keys_listed_sorted() -> None:
    """Two unknown keys are both named, in sorted order."""
    raw = _raw()
    raw["model"]["zeta"] = 1
    raw["model"]["alpha"] = 1
    with pytest.raises(ConfigError, match=r"\['alpha', 'zeta'\]"):
        config_from_dict(raw)


def test_missing_key_rejected() -> None:
    """Removing a required key raises ConfigError naming it."""
    raw = _raw()
    del raw["train"]["lr"]
    with pytest.raises(ConfigError, match=r"Missing.*\[train\].*lr"):
        config_from_dict(raw)


def test_default_field_may_be_omitted() -> None:
    """``tie_embeddings`` has a default, so omitting it is allowed (defaults to True)."""
    raw = _raw()
    del raw["model"]["tie_embeddings"]
    assert config_from_dict(raw).model.tie_embeddings is True


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "d_model", "256"),
        ("model", "n_layers", 4.0),
        ("model", "d_model", True),
        ("model", "tie_embeddings", 1),
        ("train", "precision", 16),
        ("data", "train_path", ""),
        ("train", "betas", [0.9]),
        ("train", "betas", [0.9, "0.95"]),
    ],
)
def test_wrong_type_rejected(section: str, key: str, value: object) -> None:
    """Wrong types raise ConfigError: no silent casts, bools are not ints."""
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ConfigError, match=key):
        config_from_dict(raw)


def test_yaml_exponent_without_dot_is_rejected(tmp_path: Path) -> None:
    """PyYAML parses ``6e-4`` as a string; it must fail loudly with a hint."""
    text = (CONFIG_DIR / "syrinx-10m.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("lr: 1.0e-3", "lr: 1e-3"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"train\.lr.*6\.0e-4"):
        load_config(bad)


def test_int_accepted_for_float_field() -> None:
    """``rope_theta: 10000`` (an int in YAML) is accepted and stored as a float."""
    raw = _raw()
    raw["model"]["rope_theta"] = 10000
    value = config_from_dict(raw).model.rope_theta
    assert value == 10000.0 and isinstance(value, float)


def test_section_must_be_mapping() -> None:
    """A section that is not a mapping raises ConfigError."""
    raw = _raw()
    raw["data"] = ["not", "a", "mapping"]
    with pytest.raises(ConfigError, match=r"\[data\] must be a mapping"):
        config_from_dict(raw)


# --- value validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "updates", "match"),
    [
        ("model", {"n_heads": 3}, "divisible"),
        ("model", {"d_model": 12, "n_heads": 4}, "even"),
        ("model", {"vocab_size": 65536}, "uint16"),
        ("model", {"n_layers": 0}, "n_layers must be positive"),
        ("model", {"norm_eps": 0.0}, "norm_eps"),
        ("train", {"min_lr": 1.0}, "min_lr"),
        ("train", {"min_lr": 0.0}, "min_lr"),
        ("train", {"warmup_steps": 9000}, "warmup_steps"),
        ("train", {"betas": [0.9, 1.0]}, "betas"),
        ("train", {"grad_clip": 0.0}, "grad_clip"),
        ("train", {"precision": "fp16"}, "precision"),
        ("train", {"seed": -1}, "seed"),
        ("data", {"seq_len": 0}, "seq_len"),
    ],
)
def test_invalid_values_rejected(section: str, updates: dict[str, Any], match: str) -> None:
    """Out-of-range values raise ConfigError naming the field."""
    raw = _raw()
    raw[section].update(updates)
    with pytest.raises(ConfigError, match=match):
        config_from_dict(raw)


def test_seq_len_le_max_seq_len() -> None:
    """``data.seq_len > model.max_seq_len`` raises ConfigError."""
    raw = _raw()
    raw["data"]["seq_len"] = 2048
    with pytest.raises(ConfigError, match="max_seq_len"):
        config_from_dict(raw)


def test_train_and_val_paths_must_differ() -> None:
    """Identical train and val paths raise ConfigError (leakage guard)."""
    raw = _raw()
    raw["data"]["val_path"] = raw["data"]["train_path"]
    with pytest.raises(ConfigError, match="leakage"):
        config_from_dict(raw)


# --- serialization --------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PINNED_PARAMS))
def test_roundtrip_dict(name: str) -> None:
    """``config_from_dict(config_to_dict(c)) == c`` for every shipped config."""
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert config_from_dict(config_to_dict(config)) == config


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """save_config then load_config gives an equal config; saving twice refuses."""
    config = load_config(CONFIG_DIR / "syrinx-50m.yaml")
    out = tmp_path / "run" / "config.yaml"
    save_config(config, out)
    assert load_config(out) == config
    with pytest.raises(FileExistsError):
        save_config(config, out)


def test_config_is_immutable() -> None:
    """Assigning to a field raises FrozenInstanceError."""
    config = load_config(CONFIG_DIR / "syrinx-10m.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model.d_model = 512  # type: ignore[misc]


def test_load_missing_file(tmp_path: Path) -> None:
    """A nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_empty_file(tmp_path: Path) -> None:
    """An empty YAML file raises ConfigError mentioning the file."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty.yaml"):
        load_config(empty)


def test_returns_syrinx_config() -> None:
    """load_config returns a SyrinxConfig with Path-typed data paths."""
    config = load_config(CONFIG_DIR / "syrinx-10m.yaml")
    assert isinstance(config, SyrinxConfig)
    assert isinstance(config.data.train_path, Path)
