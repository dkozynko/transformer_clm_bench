# Fair CUDA Comparison: FiX vs Decoder-Only Baselines

## Verdict

This benchmark does not show an overall FiX win against the compact Llama-style baseline. Under the three-seed, near-parameter-matched quality protocol, Llama-style reaches lower test perplexity than FiX: `6.3040 +/- 0.0608` versus `6.5996 +/- 0.0500`. FiX is substantially better than the vanilla and Differential Transformer configurations used here, but the Llama-style model is the strongest of the four.

The completed single-run throughput profile shows FiX faster than Llama-style, but that speed result is not stable enough to claim a general fused-kernel advantage. The multi-seed quality runs report the opposite ordering for tokens/sec. Treat FiX's speed advantage as inconclusive until an additional interleaved, repeated throughput-only experiment is run with fixed GPU clocks or equivalent clock telemetry.

## Environment

- GPU: NVIDIA RTX A5000, 24564 MiB
- NVIDIA driver: 580.159.04 (CUDA 13.0 driver support)
- Dataset: WikiText-2 with byte-level tokenization (259-token vocabulary)
- Precision: BF16 autocast
- FiX kernel: official FLA fused backend pinned at commit `e761e25ff9feb95f8a10882950a950a880884c8c`
- Baseline kernels: PyTorch SDPA locked to `FLASH_ATTENTION`; CUDA tests verify no math or memory-efficient fallback

## Quality Protocol

Every model used batch size 4, sequence length 1024, width 128, four heads, zero dropout, AdamW learning rate `2e-4`, weight decay `0.01`, and 500 training steps. Each seed reset both model initialization and the shuffled training-data generator, so every architecture received the same initial random-number origin and token order for that seed.

The three independent seeds were `2026`, `2027`, and `2028`. Vanilla used four layers to bring its parameter count close to the three-layer alternatives. Counts range from 825472 to 957568 parameters (within about 8 percent of their 893212-parameter mean).

| Model | Parameters | Test PPL, mean +/- sample std | Training tokens/s, mean +/- sample std |
| --- | ---: | ---: | ---: |
| Vanilla | 957568 | 10.8052 +/- 0.0378 | 332880 +/- 48078 |
| Llama-style | 825472 | 6.3040 +/- 0.0608 | 359226 +/- 33531 |
| Differential | 926464 | 7.6052 +/- 0.0228 | 248500 +/- 9844 |
| FiX | 863344 | 6.5996 +/- 0.0500 | 237812 +/- 14528 |

FiX's mean test perplexity is 4.69 percent higher than Llama-style (`+0.2956` PPL), 13.22 percent lower than Differential Transformer, and 38.92 percent lower than vanilla. The Llama-FiX gap is much larger than either model's observed three-run variation, but three compact runs are evidence for this configuration rather than a paper-scale statistical claim.

## Throughput Protocol

The shape-matched throughput profile used three layers for all four models, batch size 4, sequence length 1024, 200 training steps, and a 20-step excluded warm-up. CUDA synchronization brackets the timed work; data transfer, evaluation, and checkpoint copies are excluded. Baselines use strict Flash SDPA and FiX uses its fused FLA kernel.

| Model | Parameters | Training tokens/s |
| --- | ---: | ---: |
| Vanilla | 759296 | 342792 |
| Llama-style | 825472 | 227768 |
| Differential | 926464 | 241747 |
| FiX | 863344 | 255461 |

On this one run, FiX is 12.16 percent faster than Llama-style and 5.67 percent faster than Differential Transformer, while vanilla is 34.19 percent faster than FiX. The quality profile's three-run throughput means instead put FiX 33.80 percent below Llama-style. That discrepancy indicates that the present performance evidence is insufficient to separate kernel behavior from host scheduling, GPU clocking, and run-order effects.

## Reproduce

```bash
uv sync --extra fix-cuda --extra test
uv run pytest -v
uv run python scripts/run_benchmark.py --preset cuda-throughput --device cuda
uv run python scripts/run_benchmark.py --preset cuda-quality --device cuda
```

The raw, machine-readable measurements are in `benchmark_summary_cuda-throughput.json` and `benchmark_summary_cuda-quality.json`; their matching Markdown reports contain the same individual and aggregate values.
