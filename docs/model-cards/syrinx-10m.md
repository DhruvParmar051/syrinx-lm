---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
datasets:
- roneneldan/TinyStories
tags:
- syrinx
- llama
- from-scratch
---

# syrinx-10m

The smallest Syrinx model: a decoder-only transformer trained from scratch on TinyStories, a corpus of short synthetic children's stories. It is the first of three models (10M, 50M, 125M) in a scaling series and exists to validate the training, evaluation and export pipeline end to end.

## Model details

| | |
|---|---|
| Architecture | decoder-only transformer, `LlamaForCausalLM` (RMSNorm, RoPE, SwiGLU, tied embeddings) |
| Parameters | 11,798,784 (3,410,176 non-embedding) |
| Layers / d_model / heads / d_ff | 4 / 256 / 4 / 768 |
| Context length | 1024 tokens |
| Vocabulary | 32,768 (byte-level BPE) |
| RoPE theta / norm eps | 10000 / 1e-05 |
| Weights | fp32 safetensors |

## Training

| | |
|---|---|
| Data | TinyStories (CDLA-Sharing-1.0) |
| Tokens seen | 294,912,000 (9,000 steps x 32,768 tokens) |
| Optimizer | AdamW, peak LR 0.001, min LR 0.0001, 200 warmup steps, cosine decay, weight decay 0.1 |
| Precision | bf16-autocast |
| Hardware | Apple M5 Pro laptop (20-core GPU, 48 GB unified memory), PyTorch MPS backend |
| Wall time | 2.48 h |
| Final train / val loss | 1.5620 / 1.5725 (val: training-time estimate) |

Source: EXP-012.

## Our metrics

Full validation split, non-overlapping 1024-token windows (2,073,600 target tokens, 8,110,848 bytes), bf16-autocast:

| Loss (nats/token) | Perplexity | Bits per byte |
|---|---|---|
| 1.5641 | 4.78 | 0.5769 |

Bits per byte is the only one of these comparable across tokenizers. Source: EXP-014.

## Benchmarks

lm-evaluation-harness 0.4.13, 0-shot, float32, `<|endoftext|>` prepended, on the pinned benchmark files. ± is the binomial standard error. "Clean" excludes examples that share a 13-gram with the training data and examples too short to check.

| Benchmark | Metric | All | Clean | Contaminated | Chance |
|---|---|---|---|---|---|
| HellaSwag | acc_norm | 25.9% ± 0.4 (n=10,042) | 25.9% ± 0.4 (n=10,002) | 22.5% ± 6.6 (n=40) | 25.0% |
| ARC-Easy | acc | 27.0% ± 0.9 (n=2,376) | 27.2% ± 0.9 (n=2,278) | 4.5% ± 4.4 (n=22) | 25.0% |
| PIQA | acc_norm | 53.0% ± 1.2 (n=1,838) | 53.5% ± 1.2 (n=1,682) | 33.3% ± 27.2 (n=3) | 50.0% |
| LAMBADA (OpenAI) | acc | 7.8% ± 0.4 (n=5,153) | 7.8% ± 0.4 (n=5,153) | n/a | n/a |

The contaminated subsets are small (65 examples in total, at most 40 per benchmark), so their scores have wide error bars and are not evidence of leakage either way. Compare All with Clean instead.

Sources: EXP-017 (scores), EXP-016 (contamination).

## Comparisons

No comparisons with other models are made: published scores use different training data, token counts, tokenizers and prompts, so they would not be like for like.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("DhruvParmar051/syrinx-10m")
model = AutoModelForCausalLM.from_pretrained("DhruvParmar051/syrinx-10m", dtype=torch.float32)
# Training documents always followed <|endoftext|>, so start prompts with it.
ids = tok(tok.bos_token + "Once upon a time", return_tensors="pt")
out = model.generate(**ids)  # uses the shipped generation_config
print(tok.decode(out[0], skip_special_tokens=True))
```

## Intended use and limitations

Generating short, simple children's stories, and studying a small language model trained end to end on one laptop. It is a base model: it continues text and does not follow instructions.

- Trained only on TinyStories, so it knows almost nothing outside simple children's stories; it does not answer questions or state facts reliably.
- Base model, not instruction-tuned or safety-tuned; outputs can be incoherent, repetitive or wrong.
- Greedy decoding tends to loop; sampling with the shipped generation settings works better.
- Benchmark scores are close to chance and are reported for completeness, not as evidence of capability.

## Evidence

| Item | Experiment | File | SHA-256 |
|---|---|---|---|
| Training | EXP-012 | `experiments/10m-a-train-2026-10-12.txt` | `830997730463c63d95385a5db7a6d4cecb7c6044ed61c863aaed8524282a4808` |
| Evaluation | EXP-014 | `experiments/eval-10m-a-step9000-bf16-autocast-2026-10-13.json` | `36de4dffede0a71036517d4c34f964b6fd9931d247fb597aff6559b969ba03e4` |
| Export parity | EXP-015 | `experiments/export-10m-a-step9000-parity-2026-10-13.json` | `e78bcefc5b370ab4b22090031901274f41c88ae027844e38e2721249a76fdc9f` |
| Contamination | EXP-016 | `experiments/contamination-2026-10-14.json` | `839e2d86e90a4d68cb1bb6c1213ea6e61252c1ea7c6d76cb8df644c2fc64bda6` |
| Benchmarks | EXP-017 | `experiments/benchmarks-10m-a-2026-10-14.json` | `e3eb81340e6b3c35cf4ef58f6c5dd0dc235e385118d913dccbe086f78faa3ef4` |
| Weights | | `model.safetensors` | `5e50d4d295bc69a052ba3d024413946fa5e801bdefc725a8634f4fe44bd50f8a` |

Export parity: max absolute logit difference 0 against the Syrinx implementation (tolerance 0.0001).

## License and attribution

Code and weights: Apache-2.0.

- TinyStories (Eldan and Li) is used under CDLA-Sharing-1.0; the data itself is not redistributed.
