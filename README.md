# Syrinx

A GPT-style decoder-only language model built from scratch in PyTorch: RMSNorm, RoPE, SwiGLU,
causal multi-head attention, tied embeddings, a 32K byte-level BPE tokenizer, and context 1024.
It is scaled 10M → 50M → 125M parameters on a single laptop (Apple M5 Pro, PyTorch MPS) and
released as a checkpoint that loads with `LlamaForCausalLM`, no custom code needed.

**Status:** Phase 0 (environment + foundations). No results yet.

## Setup
```bash
uv sync --all-extras          # core + export (transformers) + eval (lm-eval) + dev
uv run pytest -q              # run tests (MPS if available)
SYRINX_DEVICE=cpu uv run pytest -q   # force CPU
```

## Layout
- [`PLAN.md`](PLAN.md): phases, gates, compute budget, runbook
- [`docs/decisions.md`](docs/decisions.md): design decisions and why
- [`docs/data.md`](docs/data.md): data sources, licenses, pipeline rules
- [`docs/reference/config-spec.md`](docs/reference/config-spec.md): config and parameter-count spec
- [`experiments/log.md`](experiments/log.md): every run, with numbers traced to log files

## License
Apache-2.0. See [LICENSE](LICENSE).
