# Config spec (reference — no code)

Use this when writing `src/syrinx/config.py`. It says *what* the config must do, not how.
Write your own version first and use this page to check it.

## Structure
Three typed, immutable dataclasses, plus one top-level wrapper that holds them:

| Section | Field | Type | Notes |
|---|---|---|---|
| model | `vocab_size` | int | 32768. Must be < 65536 (uint16 shards) |
| model | `d_model` | int | |
| model | `n_layers` | int | |
| model | `n_heads` | int | `d_model % n_heads == 0` |
| model | `d_ff` | int | SwiGLU hidden width |
| model | `max_seq_len` | int | 1024 |
| model | `rope_theta` | float | Must match the exported `LlamaConfig.rope_theta` |
| model | `norm_eps` | float | Must match `LlamaConfig.rms_norm_eps` |
| model | `tie_embeddings` | bool | True |
| train | `micro_batch_size` | int | > 0 |
| train | `grad_accum_steps` | int | > 0 |
| train | `lr`, `min_lr` | float | `0 < min_lr <= lr` |
| train | `warmup_steps`, `max_steps` | int | `warmup_steps < max_steps` |
| train | `weight_decay` | float | ≥ 0 |
| train | `betas` | tuple[float, float] | each in (0, 1) |
| train | `grad_clip` | float | > 0 |
| train | `precision` | str | one of a fixed set (e.g. `fp32`, `bf16-autocast`) |
| train | `seed` | int | |
| data | `train_path`, `val_path` | str/Path | |
| data | `seq_len` | int | ≤ `model.max_seq_len` |

Derived values (computed, never stored in YAML): `head_dim = d_model // n_heads`,
`tokens_per_step = micro_batch_size · grad_accum_steps · seq_len`.

## Validation rules (each gets an explicit error message)
- An unknown key in any YAML section → error that names the key and the section. **Never ignore it silently.**
- A missing required key → error.
- Wrong type (e.g. `"256"` for an int) → error, not a silent cast.
- `d_model % n_heads != 0` → error.
- `head_dim` odd → error (RoPE rotates pairs).
- `vocab_size >= 65536` → error (it would overflow uint16 shards).
- `data.seq_len > model.max_seq_len` → error.

## Analytic parameter count
Tied embeddings, no biases, 2 RMSNorms per block plus one final RMSNorm:

```
per_block = 4·d²            (Wq, Wk, Wv, Wo)
          + 3·d·d_ff        (W_gate, W_up, W_down)
          + 2·d             (attn_norm, mlp_norm weights)
N_total   = V·d + L·per_block + d
N_non_emb = N_total − V·d
```

Worked checks (these must be pinned in tests):

| Config | V·d | L·per_block | +d | N_total |
|---|---|---|---|---|
| 10m  | 8,388,608  | 4 · 852,480 = 3,409,920     | 256 | **11,798,784** |
| 50m  | 16,777,216 | 10 · 3,408,896 = 34,088,960 | 512 | **50,866,688** |
| 125m | 25,165,824 | 12 · 8,259,072 = 99,108,864 | 768 | **124,275,456** |

If untied, add another V·d. RoPE has no parameters.

## Tests to write (descriptions only)
- `test_param_count_pinned[10m|50m|125m]`: the analytic count equals the pinned number for each shipped config.
- `test_param_count_matches_model`: (Phase 3) `sum(p.numel())` of the built model equals the analytic count.
- `test_unknown_key_rejected`: an extra key in each section raises, and the message names the key.
- `test_missing_key_rejected`
- `test_wrong_type_rejected`
- `test_head_divisibility` and `test_head_dim_even`
- `test_vocab_fits_uint16`
- `test_seq_len_le_max`
- `test_roundtrip_yaml`: load → dump → load gives an equal config.
- `test_config_is_immutable`: assigning to a field raises.
