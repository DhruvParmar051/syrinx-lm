---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- syrinx
- llama
- from-scratch
---

# syrinx-50m

The middle Syrinx model: a decoder-only transformer trained from scratch on 1.0B tokens of FineWeb-Edu, a corpus of educational web pages. It is the second of three models (10M, 50M, 125M) in a scaling series, trained on one laptop.

## Model details

| | |
|---|---|
| Architecture | decoder-only transformer, `LlamaForCausalLM` (RMSNorm, RoPE, SwiGLU, tied embeddings) |
| Parameters | 50,866,688 (34,089,472 non-embedding) |
| Layers / d_model / heads / d_ff | 10 / 512 / 8 / 1536 |
| Context length | 1024 tokens |
| Vocabulary | 32,768 (byte-level BPE) |
| RoPE theta / norm eps | 10000 / 1e-05 |
| Weights | fp32 safetensors |

## Training

| | |
|---|---|
| Data | FineWeb-Edu (ODC-By-1.0) |
| Tokens seen | 1,000,079,360 (3,815 steps x 262,144 tokens) |
| Optimizer | AdamW, peak LR 0.0012, min LR 0.00012, 200 warmup steps, cosine decay, weight decay 0.1 |
| Precision | bf16-autocast |
| Hardware | Apple M5 Pro laptop (20-core GPU, 48 GB unified memory), PyTorch MPS backend |
| Wall time | 20.97 h in 8 segments (stopped and resumed; includes steps re-run after each resume) |
| Final train / val loss | 3.3581 / 3.3217 (val: training-time estimate) |

Source: EXP-018.

## Our metrics

Full validation split, non-overlapping 1024-token windows (12,215,296 target tokens, 54,310,934 bytes), bf16-autocast:

| Loss (nats/token) | Perplexity | Bits per byte |
|---|---|---|
| 3.3206 | 27.68 | 1.0775 |

Bits per byte is the only one of these comparable across tokenizers. Source: EXP-019.

## Benchmarks

lm-evaluation-harness 0.4.13, 0-shot, float32, `<|endoftext|>` prepended, on the pinned benchmark files. ± is the binomial standard error. "Clean" excludes examples that share a 13-gram with the training data and examples too short to check.

| Benchmark | Metric | All | Clean | Contaminated | Chance |
|---|---|---|---|---|---|
| HellaSwag | acc_norm | 28.1% ± 0.4 (n=10,042) | 28.1% ± 0.4 (n=10,002) | 35.0% ± 7.5 (n=40) | 25.0% |
| ARC-Easy | acc | 47.2% ± 1.0 (n=2,376) | 47.0% ± 1.0 (n=2,278) | 59.1% ± 10.5 (n=22) | 25.0% |
| PIQA | acc_norm | 59.2% ± 1.1 (n=1,838) | 59.8% ± 1.2 (n=1,682) | 100.0% ± 0.0 (n=3) | 50.0% |
| LAMBADA (OpenAI) | acc | 18.6% ± 0.5 (n=5,153) | 18.6% ± 0.5 (n=5,153) | n/a | n/a |

The contaminated subsets are small (65 examples in total, at most 40 per benchmark), so their scores have wide error bars and are not evidence of leakage either way. Compare All with Clean instead.

Sources: EXP-021 (scores), EXP-016 (contamination).

## Comparisons

No comparisons with other models are made: published scores use different training data, token counts, tokenizers and prompts, so they would not be like for like.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("DhruvParmar051/syrinx-50m")
model = AutoModelForCausalLM.from_pretrained("DhruvParmar051/syrinx-50m", dtype=torch.float32)
# Training documents always followed <|endoftext|>, so start prompts with it.
ids = tok(tok.bos_token + "The water cycle is", return_tensors="pt")
out = model.generate(**ids)  # uses the shipped generation_config
print(tok.decode(out[0], skip_special_tokens=True))
```

## Intended use and limitations

Studying a small language model trained end to end on one laptop, and continuing English text in an educational register. It is a base model: it continues text and does not follow instructions.

- Base model, not instruction-tuned or safety-tuned; it does not answer questions reliably and its continuations can be fluent but factually wrong or repetitive.
- Trained on 1.0B tokens of English educational web text only; other languages, code and specialised domains are poorly covered.
- Inherits the biases and errors of web text; FineWeb-Edu's quality filter does not remove them.
- Benchmark scores are above chance but low, as expected at this size.

## Evidence

| Item | Experiment | File | SHA-256 |
|---|---|---|---|
| Training | EXP-018 | `experiments/50m-a-train.txt` | `7aefc5fed1bc0240d5cfb7f4d2473c0eb5b9be5c456e7b3a970168fce572da53` |
| Evaluation | EXP-019 | `experiments/eval-50m-a-step3815-bf16-autocast.json` | `8a9362282e93525bee691565330f7db3a416456fa2aec543c3f9c9d6b7a2a9d2` |
| Export parity | EXP-020 | `experiments/export-50m-a-step3815-parity.json` | `21e0516e62d0cbe7f435e9014be4ccfecd8804763d8adb4a2ec1e7dbf4ae2f21` |
| Contamination | EXP-016 | `experiments/contamination-2026-10-14.json` | `839e2d86e90a4d68cb1bb6c1213ea6e61252c1ea7c6d76cb8df644c2fc64bda6` |
| Benchmarks | EXP-021 | `experiments/benchmarks-50m-a.json` | `93dc32e9c0c8f279b5d4970bcdcaa30bb1f27287927c60d731a24a0891f62929` |
| Weights | | `model.safetensors` | `13e7981df6adc109bf0ad6c8817dbcde54be2b378b5518b531668b6b95020aa0` |

Export parity: max absolute logit difference 0 against the Syrinx implementation (tolerance 0.0001).

## License and attribution

Code and weights: Apache-2.0.

- FineWeb-Edu (Hugging Face, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is used under ODC-By v1.0 (https://opendatacommons.org/licenses/by/1-0/); it is derived from Common Crawl and subject to the Common Crawl Terms of Use.
