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

The benchmark is intentionally compact and local-machine-friendly. It is designed for relative comparison under shared constraints, not paper-scale reproduction.

## Project Layout

- `src/transformer_clm_bench/`: models, data pipeline, training, and benchmark logic
- `scripts/run_benchmark.py`: benchmark entrypoint
- `tests/`: unit and regression tests
- `results/`: benchmark outputs
- `docs/superpowers/specs/`: design spec
- `docs/superpowers/plans/`: implementation plan

## Dataset

The compact preset uses `WikiText-2` with a word-level vocabulary capped at `5000` tokens.

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

If you want to install the package in a fresh environment:

```bash
pip install -e .
```

## Running Tests

```bash
pytest -v
```

## Running The Compact Benchmark

```bash
python3 scripts/run_benchmark.py --preset compact
```

Outputs:

- `results/benchmark_summary.json`
- `results/benchmark_report.md`

## Current Compact Benchmark Result

These numbers were produced locally on CPU with the current compact preset:

| Model | Params | Val PPL | Test PPL | Tokens/sec |
| --- | ---: | ---: | ---: | ---: |
| vanilla | 424,192 | 2923.40 | 2880.96 | 8414.14 |
| llama | 453,056 | 2917.95 | 2888.86 | 3209.21 |
| differential | 470,240 | 2969.49 | 2933.16 | 2803.41 |

## Notes And Limitations

- The current preset is only `20` optimization steps, so the benchmark is useful for smoke-testing architecture behavior, not for claiming absolute quality.
- Differential Transformer is implemented from its core differential-attention idea in this repo, but this is still a compact adaptation rather than a full large-scale paper reproduction.
- The generation samples are sanity checks only and are expected to be poor under such a small training budget.
