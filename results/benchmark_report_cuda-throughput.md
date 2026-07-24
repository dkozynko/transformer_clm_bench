# Benchmark Report

- Preset: `cuda-throughput`
- Tokenizer: `byte`
- Device: `cuda`
- Mixed precision: `bfloat16`
- Attention backend: `sdpa-flash`
- FiX backend: `fused`
- Workload: batch 4, sequence 1024, 200 steps
- Seeds: single run

## Models

### vanilla
- Test perplexity: 12.4096
- Validation perplexity: 12.5157
- Parameters: 759296
- Tokens/sec: 342791.83
- Sample: `The meaning of life is`

### llama
- Test perplexity: 10.0196
- Validation perplexity: 10.1339
- Parameters: 825472
- Tokens/sec: 227768.16
- Sample: `The meaning of life is`

### differential
- Test perplexity: 11.7736
- Validation perplexity: 11.8811
- Parameters: 926464
- Tokens/sec: 241747.34
- Sample: `The meaning of life is`

### fix
- Test perplexity: 9.5767
- Validation perplexity: 9.6910
- Parameters: 863344
- Tokens/sec: 255460.72
- Sample: `The meaning of life is`
