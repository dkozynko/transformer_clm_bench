# Transformer CLM Bench

Compact PyTorch benchmarks for a recent Transformer architecture variation and strong decoder-only baselines.

## Overview

This repository benchmarks a recent Transformer architecture variation for compact causal language modeling in pure PyTorch.

Implemented models:

- `vanilla`: GPT-style causal Transformer with learned positional embeddings
- `llama`: RoPE + RMSNorm + SwiGLU decoder-only Transformer
- `differential`: Differential Transformer attention inside the same decoder-style scaffold

Primary paper:

- Tianzhu Ye et al., `Differential Transformer`, arXiv:2410.05258, submitted October 7, 2024 and revised April 7, 2025

The benchmark is intentionally local-machine-friendly. It is designed for relative comparison under shared constraints, not paper-scale reproduction.

## Project Layout

- `src/transformer_clm_bench/`: models, data pipeline, training, and benchmark logic
- `scripts/run_benchmark.py`: benchmark entrypoint
- `tests/`: unit and regression tests
- `results/`: benchmark outputs
- `docs/superpowers/specs/`: design spec
- `docs/superpowers/plans/`: implementation plan

## Dataset And Tokenization

Both presets use `WikiText-2`.

- `compact`: word-level vocabulary capped at `5000` tokens for quick smoke testing
- `meaningful`: byte-level tokenization for more interpretable samples and a less degenerate benchmark

The benchmark expects these cached files:

- `.cache/wikitext-2/train.txt`
- `.cache/wikitext-2/validation.txt`
- `.cache/wikitext-2/test.txt`

If the Python downloader is blocked by a sandboxed environment, populate the cache directly with:

```bash
mkdir -p .cache/wikitext-2
curl -fsSL https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt -o .cache/wikitext-2/train.txt
curl -fsSL https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt -o .cache/wikitext-2/validation.txt
curl -fsSL https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt -o .cache/wikitext-2/test.txt
```

## Setup

The repo currently relies on the packages already available in the local environment:

- `torch`
- `pytest`
- `tqdm`

### Apple Silicon Acceleration

This benchmark is optimized for Apple Silicon (M1/M2/M3/M4) using the Metal Performance Shaders (MPS) backend. On a MacBook M4 Pro, the `meaningful` benchmark takes under 5 minutes on the CPU, and significantly faster on the GPU.

To run with MPS acceleration:

```bash
python3 scripts/run_benchmark.py --preset meaningful --device mps
```

## Running Tests

```bash
pytest -v
```

## Benchmark Presets

### Compact

Purpose:

- quick smoke test
- fast architecture wiring check

Command:

```bash
python3 scripts/run_benchmark.py --preset compact
```

### Meaningful

Purpose:

- roughly 10-minute CPU benchmark
- readable generation samples
- stronger relative comparison across architectures

Command:

```bash
python3 scripts/run_benchmark.py --preset meaningful
```

Outputs:

- `results/benchmark_summary_meaningful.json`
- `results/benchmark_report_meaningful.md`

Current recorded result:

| Model | Params | Val PPL | Test PPL | Tokens/sec |
| --- | ---: | ---: | ---: | ---: |
| vanilla | 372,864 | 13.94 | 13.82 | 16397.47 |
| llama | 471,648 | 12.36 | 12.24 | 13075.33 |
| differential | 528,960 | 10.70 | 10.56 | 7964.19 |


## Notes And Limitations

- The `compact` preset is only for smoke-testing architecture behavior, not for claiming absolute quality.
- The `meaningful` preset is the benchmark to care about for relative comparison in this repo.
- The `advanced` preset demonstrates scaling and stability over 3,000 training steps.
- Differential Transformer uses a dual-attention mechanism with learnable noise cancellation. Recent improvements to the initialization of the $\lambda$ parameter have significantly improved its performance at small scales.
- Byte-level perplexity from `meaningful` is not numerically comparable to word-level perplexity from `compact`.
- Differential Transformer is implemented from its core differential-attention idea in this repo, but this is still a compact adaptation rather than a full large-scale paper reproduction.
- The generation samples are still sanity checks, not polished text generation demos. In the current meaningful run the models often terminate immediately after the prompt, which is still readable but not yet a strong continuation benchmark.
