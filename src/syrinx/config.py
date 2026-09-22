"""Typed, immutable run configuration loaded from YAML.

Three sections (model / train / data) plus a top-level name. Unknown keys, missing
keys and wrong types are errors, never silently ignored or cast. The one accepted
conversion is int -> float for float fields (YAML writes ``10000`` for a float
``rope_theta``). Spec: ``docs/reference/config-spec.md``.
"""

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

logger = logging.getLogger(__name__)

MAX_UINT16_VOCAB = 65536
"""Token ids are stored as uint16, so ``vocab_size`` must be strictly below this."""

VALID_PRECISIONS: frozenset[str] = frozenset({"fp32", "bf16-autocast"})
"""Allowed values for :attr:`TrainConfig.precision`."""

_MAX_SEED = 2**32 - 1
_TOP_LEVEL_KEYS = frozenset({"name", "model", "train", "data"})

_SectionT = TypeVar("_SectionT")


class ConfigError(ValueError):
    """Any invalid configuration: unknown/missing key, wrong type or bad value."""


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ConfigError` with ``message`` if ``condition`` is False."""
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class ModelConfig:
    """Transformer architecture hyperparameters.

    Attributes:
        vocab_size: Tokenizer vocabulary size V (32768). Must be < 65536.
        d_model: Residual stream width C.
        n_layers: Number of transformer blocks L.
        n_heads: Attention heads H; ``d_model % n_heads == 0``.
        d_ff: SwiGLU hidden width.
        max_seq_len: Maximum context length (1024).
        rope_theta: RoPE base; must equal the exported ``LlamaConfig.rope_theta``.
        norm_eps: RMSNorm epsilon; must equal ``LlamaConfig.rms_norm_eps``.
        tie_embeddings: Share the input embedding and output projection weights.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    rope_theta: float
    norm_eps: float
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate values.

        Raises:
            ConfigError: On any invalid value, naming the field.
        """
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"):
            value = getattr(self, name)
            _require(value > 0, f"model.{name} must be positive, got {value}.")
        _require(
            self.vocab_size < MAX_UINT16_VOCAB,
            f"model.vocab_size must be < {MAX_UINT16_VOCAB} (uint16 token shards), "
            f"got {self.vocab_size}.",
        )
        _require(
            self.d_model % self.n_heads == 0,
            f"model.d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).",
        )
        _require(
            self.head_dim % 2 == 0,
            f"model head_dim ({self.head_dim}) must be even: RoPE rotates pairs.",
        )
        _require(self.rope_theta > 0, f"model.rope_theta must be positive, got {self.rope_theta}.")
        _require(self.norm_eps > 0, f"model.norm_eps must be positive, got {self.norm_eps}.")

    @property
    def head_dim(self) -> int:
        """Per-head width D = d_model // n_heads."""
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TrainConfig:
    """Optimization and run-control hyperparameters.

    Attributes:
        micro_batch_size: Sequences per forward/backward pass.
        grad_accum_steps: Micro-batches per optimizer step.
        lr: Peak learning rate.
        min_lr: Final learning rate after decay; ``0 < min_lr <= lr``.
        warmup_steps: Linear warmup length in optimizer steps; ``< max_steps``.
        max_steps: Total optimizer steps.
        weight_decay: AdamW decoupled weight decay (not applied to norms/embeddings).
        betas: AdamW (beta1, beta2), each in (0, 1).
        grad_clip: Max global grad norm; > 0.
        precision: One of :data:`VALID_PRECISIONS`.
        seed: Global RNG seed in ``[0, 2**32 - 1]``.
    """

    micro_batch_size: int
    grad_accum_steps: int
    lr: float
    min_lr: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    betas: tuple[float, float]
    grad_clip: float
    precision: str
    seed: int

    def __post_init__(self) -> None:
        """Validate ranges and cross-field constraints.

        Raises:
            ConfigError: On any invalid value, naming the field.
        """
        _require(self.micro_batch_size > 0, "train.micro_batch_size must be positive.")
        _require(self.grad_accum_steps > 0, "train.grad_accum_steps must be positive.")
        _require(self.lr > 0, f"train.lr must be positive, got {self.lr}.")
        _require(
            0 < self.min_lr <= self.lr,
            f"train.min_lr must satisfy 0 < min_lr <= lr, got min_lr={self.min_lr}, lr={self.lr}.",
        )
        _require(self.max_steps > 0, f"train.max_steps must be positive, got {self.max_steps}.")
        _require(
            0 <= self.warmup_steps < self.max_steps,
            f"train.warmup_steps must be in [0, max_steps), got {self.warmup_steps} "
            f"with max_steps={self.max_steps}.",
        )
        _require(self.weight_decay >= 0, "train.weight_decay must be non-negative.")
        _require(
            len(self.betas) == 2 and all(0 < b < 1 for b in self.betas),
            f"train.betas must be two values in (0, 1), got {self.betas}.",
        )
        _require(self.grad_clip > 0, f"train.grad_clip must be positive, got {self.grad_clip}.")
        _require(
            self.precision in VALID_PRECISIONS,
            f"train.precision must be one of {sorted(VALID_PRECISIONS)}, got {self.precision!r}.",
        )
        _require(
            0 <= self.seed <= _MAX_SEED, f"train.seed must be in [0, {_MAX_SEED}], got {self.seed}."
        )


@dataclass(frozen=True)
class DataConfig:
    """Dataset locations and sequence length.

    Attributes:
        train_path: Train token shards (file or directory).
        val_path: Validation token shards; must differ from ``train_path``.
        seq_len: Tokens per training sequence; ``<= model.max_seq_len``.
    """

    train_path: Path
    val_path: Path
    seq_len: int

    def __post_init__(self) -> None:
        """Validate values.

        Raises:
            ConfigError: On any invalid value.
        """
        _require(self.seq_len > 0, f"data.seq_len must be positive, got {self.seq_len}.")
        _require(
            self.train_path != self.val_path,
            f"data.train_path and data.val_path must differ (leakage), both are {self.train_path}.",
        )


@dataclass(frozen=True)
class SyrinxConfig:
    """Top-level run configuration.

    Attributes:
        name: Config name, e.g. ``"syrinx-125m"``.
        model: Architecture section.
        train: Optimization section.
        data: Data section.
    """

    name: str
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

    def __post_init__(self) -> None:
        """Validate cross-section constraints.

        Raises:
            ConfigError: If ``name`` is empty or ``data.seq_len > model.max_seq_len``.
        """
        _require(bool(self.name.strip()), "name must be a non-empty string.")
        _require(
            self.data.seq_len <= self.model.max_seq_len,
            f"data.seq_len ({self.data.seq_len}) exceeds model.max_seq_len "
            f"({self.model.max_seq_len}).",
        )

    @property
    def tokens_per_step(self) -> int:
        """Tokens per optimizer step = micro_batch_size * grad_accum_steps * seq_len."""
        return self.train.micro_batch_size * self.train.grad_accum_steps * self.data.seq_len


def _check_keys(
    section: str, raw: Mapping[str, Any], allowed: set[str], required: set[str]
) -> None:
    """Reject unknown and missing keys in one section.

    Raises:
        ConfigError: Listing every unknown (or missing) key, sorted, with the section.
    """
    unknown = sorted(set(raw) - allowed)
    _require(not unknown, f"Unknown key(s) in [{section}]: {unknown}. Allowed: {sorted(allowed)}.")
    missing = sorted(required - set(raw))
    _require(not missing, f"Missing key(s) in [{section}]: {missing}.")


def _coerce(section: str, name: str, value: Any, annotation: Any) -> Any:
    """Type-check ``value`` against ``annotation`` and convert where safe.

    Rules: bools are never accepted as numbers; ints are accepted for float fields;
    ``Path`` fields take a non-empty string; ``tuple[...]`` fields take a YAML list
    of the right length. No other conversion (``"256"`` is not an int).

    Raises:
        ConfigError: Naming the section, field, expected and actual type.
    """
    where = f"{section}.{name}"
    actual = type(value).__name__
    if annotation is bool:
        _require(isinstance(value, bool), f"{where} must be a bool, got {actual} {value!r}.")
        return value
    if annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        _require(ok, f"{where} must be an int, got {actual} {value!r}.")
        return value
    if annotation is float:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        hint = " (YAML reads 6e-4 as a string; write 6.0e-4)" if isinstance(value, str) else ""
        _require(ok, f"{where} must be a number, got {actual} {value!r}{hint}.")
        return float(value)
    if annotation is str:
        _require(isinstance(value, str), f"{where} must be a string, got {actual} {value!r}.")
        return value
    if annotation is Path:
        ok = isinstance(value, str) and bool(value.strip())
        _require(ok, f"{where} must be a non-empty path string, got {actual} {value!r}.")
        return Path(value)
    if get_origin(annotation) is tuple:
        item_types = get_args(annotation)
        ok = isinstance(value, list | tuple) and len(value) == len(item_types)
        _require(ok, f"{where} must be a list of {len(item_types)} values, got {value!r}.")
        return tuple(
            _coerce(section, f"{name}[{i}]", item, item_type)
            for i, (item, item_type) in enumerate(zip(value, item_types, strict=True))
        )
    raise TypeError(f"Unsupported config annotation {annotation!r} for {where}.")


def _section_from_dict(cls: type[_SectionT], section: str, raw: Any) -> _SectionT:
    """Build one section dataclass from its mapping, validating keys and types.

    Raises:
        ConfigError: If ``raw`` is not a mapping or any check fails.
    """
    _require(isinstance(raw, Mapping), f"[{section}] must be a mapping, got {type(raw).__name__}.")
    fields = dataclasses.fields(cls)  # type: ignore[arg-type]
    hints = get_type_hints(cls)
    allowed = {f.name for f in fields}
    required = {
        f.name
        for f in fields
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    _check_keys(section, raw, allowed, required)
    kwargs = {key: _coerce(section, key, value, hints[key]) for key, value in raw.items()}
    return cls(**kwargs)


def config_from_dict(raw: Mapping[str, Any]) -> SyrinxConfig:
    """Build a validated :class:`SyrinxConfig` from a parsed YAML mapping.

    Args:
        raw: Mapping with top-level keys ``name``, ``model``, ``train``, ``data``.

    Returns:
        The validated config.

    Raises:
        ConfigError: On unknown/missing keys, wrong types or invalid values.
    """
    _require(isinstance(raw, Mapping), f"Config must be a mapping, got {type(raw).__name__}.")
    _check_keys("top-level", raw, set(_TOP_LEVEL_KEYS), set(_TOP_LEVEL_KEYS))
    name = _coerce("top-level", "name", raw["name"], str)
    return SyrinxConfig(
        name=name,
        model=_section_from_dict(ModelConfig, "model", raw["model"]),
        train=_section_from_dict(TrainConfig, "train", raw["train"]),
        data=_section_from_dict(DataConfig, "data", raw["data"]),
    )


def _to_plain(value: Any) -> Any:
    """Convert Paths to str and tuples to lists, recursively."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(v) for v in value]
    return value


def config_to_dict(config: SyrinxConfig) -> dict[str, Any]:
    """Convert a config to a plain, YAML-serializable mapping.

    Round-trips: ``config_from_dict(config_to_dict(c)) == c``.

    Args:
        config: Config to serialize.

    Returns:
        Plain mapping (Paths as str, tuples as lists).
    """
    return _to_plain(dataclasses.asdict(config))


def load_config(path: str | Path) -> SyrinxConfig:
    """Load and validate a YAML config file.

    Args:
        path: Path to a ``.yaml`` file.

    Returns:
        The validated config.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigError: If the contents are invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    try:
        return config_from_dict(raw)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def save_config(config: SyrinxConfig, path: str | Path) -> None:
    """Write ``config`` as YAML to a new file (e.g. a snapshot in a run directory).

    Args:
        config: Config to write.
        path: Destination; must not exist (parents are created).

    Raises:
        FileExistsError: If ``path`` exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(config), fh, sort_keys=False)


def num_parameters(model: ModelConfig, include_embedding: bool = True) -> int:
    """Analytic parameter count (no model instantiation).

    Tied embeddings, no biases, two RMSNorms per block plus a final norm::

        per_block = 4*d**2 + 3*d*d_ff + 2*d
        N_total   = V*d + L*per_block + d      (+ V*d if untied)

    Pinned: 11,798,784 (10m) / 50,866,688 (50m) / 124,275,456 (125m).

    Args:
        model: Architecture config.
        include_embedding: If False, exclude every V*d matrix (input embedding and,
            when untied, the output projection): the "non-embedding" count used for
            scaling-law fits.

    Returns:
        Parameter count.
    """
    d, ff, vocab = model.d_model, model.d_ff, model.vocab_size
    per_block = 4 * d * d + 3 * d * ff + 2 * d
    body = model.n_layers * per_block + d
    embedding = vocab * d * (1 if model.tie_embeddings else 2)
    return body + embedding if include_embedding else body
