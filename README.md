# Transformer CLM Bench

Compact PyTorch benchmarks for a recent Transformer architecture variation and strong decoder-only baselines.

## Overview

This repository benchmarks a recent Transformer architecture variation for compact causal language modeling in pure PyTorch.

Implemented models:

- `vanilla`: GPT-style causal Transformer with learned positional embeddings
- `llama`: RoPE + RMSNorm + SwiGLU decoder-only Transformer
- `differential`: Differential Transformer attention inside the same decoder-style scaffold
- `fix`: Fine-grained Forgetting Transformer (FiX) attention with RoPE, RMSNorm, and SwiGLU

Primary paper:

- Tianzhu Ye et al., `Differential Transformer`, arXiv:2410.05258, submitted October 7, 2024 and revised April 7, 2025
- Runzhong Li et al., `FiX: Introducing Fine-grained Forget Gate into Softmax Attention`, ICML 2026

FiX retains ordinary causal softmax attention but applies a learned, per-feature cumulative forget gate to the value-output path. The first decoder block derives gates from token IDs; later blocks use a low-rank, RMS-normalized projection of hidden states. The portable reference backend is pure PyTorch; an optional CUDA backend delegates to the official FiX custom-autograd Triton kernel.

The benchmark is intentionally local-machine-friendly. It is designed for relative comparison under shared constraints, not paper-scale reproduction.

## Project Layout

- `src/transformer_clm_bench/`: models, data pipeline, training, and benchmark logic
- `scripts/run_benchmark.py`: benchmark entrypoint
- `tests/`: unit and regression tests
- `results/`: benchmark outputs

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

The baseline environment requires:

- `torch`
- `tqdm`

Install test tooling with:

```bash
uv sync --extra test
```

### Apple Silicon Acceleration

This benchmark supports Apple Silicon (M1/M2/M3/M4) through the Metal Performance Shaders (MPS) backend. Runtime depends strongly on the selected model set: FiX materializes a feature-wise decay tensor in this pure-PyTorch implementation, so the four-model meaningful run should be treated as a longer quality benchmark rather than a quick smoke test.

To run with MPS acceleration:

```bash
python3 scripts/run_benchmark.py --preset meaningful --device mps
```

### Exact FiX CUDA Backend

The fused backend uses the official FiX repository at commit `e761e25ff9feb95f8a10882950a950a880884c8c`. It requires a CUDA-capable NVIDIA GPU, BF16 autocast, zero attention dropout, and an attention head dimension of `16`, `32`, `64`, or `128`. It cannot run on MPS or CPU.

Install the optional dependency in a CUDA environment:

```bash
uv sync --extra fix-cuda --extra test
```

Run the CUDA-fused smoke profile:

```bash
python3 scripts/run_benchmark.py --preset cuda-fused --device cuda
```

`cuda-fused` uses 32-dimensional heads and BF16 autocast so it meets the kernel contract. It validates the integration and is not a paper-scale reproduction. To reproduce paper-scale metrics, additionally match the official training dataset, tokenizer, model size, sequence length, optimizer, schedule, token budget, seed, and GPU setup.

### Fair CUDA Comparison

The CUDA profiles compare FiX with hardware-accelerated baselines rather than with their portable PyTorch reference paths:

```bash
uv run python scripts/run_benchmark.py --preset cuda-throughput --device cuda
uv run python scripts/run_benchmark.py --preset cuda-quality --device cuda
uv run python scripts/run_benchmark.py --preset cuda-throughput-v2 --device cuda
uv run python scripts/run_benchmark.py --preset cuda-quality-v2 --device cuda
```

`cuda-throughput` is a fixed-shape, single-seed measurement: batch 4, sequence length 1024, 200 training steps, BF16, zero dropout, and 20 excluded warm-up steps. Vanilla, Llama-style, and Differential Transformer use strict PyTorch Flash SDPA; FiX uses the official FLA fused kernel. The timer synchronizes CUDA and excludes evaluation and checkpoint work. It measures execution speed, not quality at equal parameter count.

`cuda-quality` runs 500 training steps for each of three seeds (`2026`, `2027`, `2028`) under the same data, token order, initialization seed, optimizer, precision, sequence length, and batch size. It uses near-parameter-matched model sizes: 957568 parameters for four-layer vanilla, 825472 for three-layer Llama-style, 926464 for three-layer Differential Transformer, and 863344 for three-layer FiX. Its report gives mean plus sample standard deviation for validation/test perplexity and throughput; this is the profile for an accuracy claim at comparable compact scale.

`cuda-quality-v2` is the main compact GPU quality profile. It trains one complete shuffled WikiText-2 byte epoch for every model and each of the three seeds. It records the loader-resolved step and token budgets, token-weighted perplexity, source revision, corpus SHA-256 hashes, PyTorch/CUDA/GPU information, and TF32 settings. Llama-style, Differential, and FiX share the same RoPE/RMSNorm/SwiGLU decoder scaffold and form the controlled attention comparison; vanilla is reported separately as a broader scaffold baseline.

`cuda-throughput-v2` is the main compact GPU performance profile. It runs five BF16 training measurements with cyclic model order, strict Flash SDPA for baseline attention, FiX's fused FLA kernel, CUDA-synchronized timing, and excluded warm-up. It reports individual measurements plus mean, sample standard deviation, median, minimum, and maximum tokens/sec. It does not claim fixed-clock or paper-scale performance.

Outputs:

- `results/benchmark_summary_cuda-throughput.json`
- `results/benchmark_report_cuda-throughput.md`
- `results/benchmark_summary_cuda-quality.json`
- `results/benchmark_report_cuda-quality.md`
- `results/benchmark_summary_cuda-throughput-v2.json`
- `results/benchmark_report_cuda-throughput-v2.md`
- `results/benchmark_summary_cuda-quality-v2.json`
- `results/benchmark_report_cuda-quality-v2.md`

## Running Tests

```bash
uv run pytest -v
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

- longer local quality benchmark
- readable generation samples
- stronger relative comparison across architectures

Command:

```bash
python3 scripts/run_benchmark.py --preset meaningful
```

Outputs:

- `results/benchmark_summary_meaningful.json`
- `results/benchmark_report_meaningful.md`

## Notes And Limitations

- The `compact` preset is only for smoke-testing architecture behavior, not for claiming absolute quality.
- The `meaningful` preset is the benchmark to care about for relative comparison in this repo.
- The `advanced` preset demonstrates scaling and stability over 3,000 training steps.
- `cuda-throughput` and `cuda-quality` are separate measurements: do not use the throughput result as an equal-parameter quality comparison.
- `cuda-quality-v2` and `cuda-throughput-v2` improve experimental controls, but they do not include architecture-specific hyperparameter tuning, long-context retrieval, or fixed GPU-clock control.
- Differential Transformer uses a dual-attention mechanism with learnable noise cancellation. Recent improvements to the initialization of the $\lambda$ parameter have significantly improved its performance at small scales.
- FiX uses `fix_backend=auto` by default. It selects the official fused kernel only when CUDA, BF16 autocast, zero attention dropout, supported head dimensions, and the optional dependency are all available; otherwise it uses the reference backend.
- FiX materializes its feature-wise causal value-decay tensor in the reference backend. Its compact and meaningful throughput is not comparable to the CUDA-fused kernel.
- The CUDA parity test compares fused and reference logits, loss, and a representative gradient when CUDA and the optional dependency are available. It is skipped on CPU and MPS machines.
- Byte-level perplexity from `meaningful` is not numerically comparable to word-level perplexity from `compact`.
- Differential Transformer is implemented from its core differential-attention idea in this repo, but this is still a compact adaptation rather than a full large-scale paper reproduction.
- The generation samples are still sanity checks, not polished text generation demos. In the current meaningful run the models often terminate immediately after the prompt, which is still readable but not yet a strong continuation benchmark.