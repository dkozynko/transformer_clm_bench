# Benchmark Report

- Preset: `cuda-quality`
- Tokenizer: `byte`
- Device: `cuda`
- Mixed precision: `bfloat16`
- Attention backend: `sdpa-flash`
- FiX backend: `fused`
- Workload: batch 4, sequence 1024, 500 steps
- Seeds: 2026, 2027, 2028

## Models

### vanilla
- Test perplexity (mean +/- sample std): 10.8052 +/- 0.0378
- Validation perplexity (mean +/- sample std): 10.9458 +/- 0.0345
- Steady-state training tokens/sec (mean +/- sample std): 332879.69 +/- 48077.88
- Parameters: 957568
- Layers: 4
- Attention backend: `sdpa-flash`

### llama
- Test perplexity (mean +/- sample std): 6.3040 +/- 0.0608
- Validation perplexity (mean +/- sample std): 6.3740 +/- 0.0530
- Steady-state training tokens/sec (mean +/- sample std): 359226.35 +/- 33531.10
- Parameters: 825472
- Layers: 3
- Attention backend: `sdpa-flash`

### differential
- Test perplexity (mean +/- sample std): 7.6052 +/- 0.0228
- Validation perplexity (mean +/- sample std): 7.6871 +/- 0.0126
- Steady-state training tokens/sec (mean +/- sample std): 248499.93 +/- 9844.10
- Parameters: 926464
- Layers: 3
- Attention backend: `sdpa-flash`

### fix
- Test perplexity (mean +/- sample std): 6.5996 +/- 0.0500
- Validation perplexity (mean +/- sample std): 6.6693 +/- 0.0459
- Steady-state training tokens/sec (mean +/- sample std): 237811.74 +/- 14528.33
- Parameters: 863344
- Layers: 3
- Attention backend: `fla-fused`
