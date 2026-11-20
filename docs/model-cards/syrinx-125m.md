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

# syrinx-125m

The largest Syrinx model: a decoder-only transformer trained from scratch on 2.5B tokens of FineWeb-Edu, a corpus of educational web pages. It is the third of three models (10M, 50M, 125M) in a scaling series; the 10M and 50M models were trained on a laptop, this one on a single GPU of a university cluster, with the same code.

## Model details

| | |
|---|---|
| Architecture | decoder-only transformer, `LlamaForCausalLM` (RMSNorm, RoPE, SwiGLU, tied embeddings) |
| Parameters | 124,275,456 (99,109,632 non-embedding) |
| Layers / d_model / heads / d_ff | 12 / 768 / 12 / 2560 |
| Context length | 1024 tokens |
| Vocabulary | 32,768 (byte-level BPE) |
| RoPE theta / norm eps | 10000 / 1e-05 |
| Weights | fp32 safetensors |

## Training

| | |
|---|---|
| Data | FineWeb-Edu (ODC-By-1.0) |
| Tokens seen | 2,500,329,472 (4,769 steps x 524,288 tokens) |
| Optimizer | AdamW, peak LR 0.0024, min LR 0.00024, 300 warmup steps, cosine decay, weight decay 0.1 |
| Precision | bf16-autocast |
| Hardware | one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) GPU of a university cluster, PyTorch CUDA backend |
| Wall time | 4.75 h |
| Final train / val loss | 3.0406 / 3.0263 (val: training-time estimate) |

Source: EXP-025.

## Our metrics

Full validation split, non-overlapping 1024-token windows (12,215,296 target tokens, 54,310,934 bytes), bf16-autocast:

| Loss (nats/token) | Perplexity | Bits per byte |
|---|---|---|
| 3.0280 | 20.66 | 0.9825 |

Bits per byte is the only one of these comparable across tokenizers. Source: EXP-026.

## Benchmarks

lm-evaluation-harness 0.4.13, 0-shot, float32, `<|endoftext|>` prepended, on the pinned benchmark files. ± is the binomial standard error. "Clean" excludes examples that share a 13-gram with the training data and examples too short to check.

| Benchmark | Metric | All | Clean | Contaminated | Chance |
|---|---|---|---|---|---|
| HellaSwag | acc_norm | 31.4% ± 0.5 (n=10,042) | 31.3% ± 0.5 (n=10,002) | 47.5% ± 7.9 (n=40) | 25.0% |
| ARC-Easy | acc | 50.7% ± 1.0 (n=2,376) | 50.4% ± 1.0 (n=2,278) | 63.6% ± 10.3 (n=22) | 25.0% |
| PIQA | acc_norm | 61.3% ± 1.1 (n=1,838) | 62.0% ± 1.2 (n=1,682) | 100.0% ± 0.0 (n=3) | 50.0% |
| LAMBADA (OpenAI) | acc | 22.6% ± 0.6 (n=5,153) | 22.6% ± 0.6 (n=5,153) | n/a | n/a |

The contaminated subsets are small (65 examples in total, at most 40 per benchmark), so their scores have wide error bars and are not evidence of leakage either way. Compare All with Clean instead.

Sources: EXP-024 (scores), EXP-016 (contamination).

## Comparisons

No comparisons with other models are made: published scores use different training data, token counts, tokenizers and prompts, so they would not be like for like.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("DhruvParmar051/syrinx-125m")
model = AutoModelForCausalLM.from_pretrained("DhruvParmar051/syrinx-125m", dtype=torch.float32)
# Training documents always followed <|endoftext|>, so start prompts with it.
ids = tok(tok.bos_token + "The water cycle is", return_tensors="pt")
out = model.generate(**ids)  # uses the shipped generation_config
print(tok.decode(out[0], skip_special_tokens=True))
```

## Intended use and limitations

Studying a small language model trained end to end from scratch, and continuing English text in an educational register. It is a base model: it continues text and does not follow instructions.

- Base model, not instruction-tuned or safety-tuned; it does not answer questions reliably and its continuations can be fluent but factually wrong or repetitive.
- Trained on 2.5B tokens of English educational web text only; other languages, code and specialised domains are poorly covered.
- Inherits the biases and errors of web text; FineWeb-Edu's quality filter does not remove them.
- Benchmark scores are above chance but low, as expected at this size.
- On Apple MPS, run it in float32. Under `torch.autocast("mps", dtype=torch.bfloat16)`, transformers 5.17.0 computes the rotary angles in bf16 and the loss gets clearly worse (EXP-023).

## Evidence

| Item | Experiment | File | SHA-256 |
|---|---|---|---|
| Training | EXP-025 | `experiments/125m-a-train.txt` | `847409ef2baa3d465a3f55c325d7464805643908100f0a2e666f8449e818371b` |
| Evaluation | EXP-026 | `experiments/eval-125m-a-step4769-bf16-autocast.json` | `327c8a32871e2774ce12fb74c1eb75f0bcd212f8100424079c297c2d162ee6ac` |
| Export parity | EXP-023 | `experiments/export-125m-a-step4769-parity-mac.json` | `fac11f49c60858680ae8e03e0d8766988ffc3f8f55910af476e591c3b3c3f1b3` |
| Contamination | EXP-016 | `experiments/contamination-2026-10-14.json` | `839e2d86e90a4d68cb1bb6c1213ea6e61252c1ea7c6d76cb8df644c2fc64bda6` |
| Benchmarks | EXP-024 | `experiments/benchmarks-125m-a.json` | `31f9d33ce56a7b011a4a5d2040f9ccef050ee8bb317d31f381e9af3447451b73` |
| Weights | | `model.safetensors` | `29dfaf45d1c5132c770a8b598108f11612044ccd569e2a3db9f43c6240ad495d` |

Export parity: max absolute logit difference 0 against the Syrinx implementation (tolerance 0.0001).

## License and attribution

Code and weights: Apache-2.0.

- FineWeb-Edu (Hugging Face, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is used under ODC-By v1.0 (https://opendatacommons.org/licenses/by/1-0/); it is derived from Common Crawl and subject to the Common Crawl Terms of Use.
