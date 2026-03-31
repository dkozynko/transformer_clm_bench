# Transformer CLM Bench Design

## Summary

This project will be a standalone repository for implementing and benchmarking a recent Transformer-architecture variation on a compact causal language modeling task.

The primary architecture will be **Differential Transformer** from arXiv:2410.05258, "Differential Transformer," submitted on October 7, 2024 and revised on April 7, 2025. It was accepted as an oral presentation at ICLR 2025. We will implement the paper's core differential-attention idea directly in PyTorch rather than wrapping a third-party pretrained model.

The benchmark will target **compact causal language modeling** on **WikiText-2** using a **word-level vocabulary** and a shared training harness. The initial comparison set will include:

- a GPT-style causal Transformer
- a LLaMA-style causal Transformer with RoPE, RMSNorm, and SwiGLU
- a Differential Transformer using the same overall decoder layout

The goal is a clean, reproducible relative comparison under local resource constraints, not a paper-scale reproduction.

## Goals

- Create a separate Git repository at `/Users/dmko/IdeaProjects/eeg/transformer-clm-bench`
- Implement Differential Transformer in PyTorch for causal language modeling
- Implement at least two strong Transformer baselines in the same codebase
- Train and evaluate all models under the same compact benchmark configuration
- Produce reproducible metrics and a short benchmark report

## Non-Goals

- Reproducing the original paper's large-scale training regime
- Matching state-of-the-art absolute perplexity on modern large corpora
- Supporting distributed training in the first version
- Building a general-purpose framework for arbitrary tokenizers and datasets
- Adding unrelated retrieval, instruction-tuning, or RLHF evaluation tasks

## Repository Boundary

This repository will be self-contained and will not import from the existing EEG project directories. It will have its own:

- Git history
- dependencies
- tests
- configs
- scripts
- results and artifact conventions

Planned top-level layout:

- `src/transformer_clm_bench/`
- `configs/`
- `scripts/`
- `tests/`
- `results/`
- `artifacts/` (gitignored)
- `.cache/` (gitignored)

## Architecture Choice

### Primary paper

We considered two recent primary-source candidates:

- **Differential Transformer** (arXiv:2410.05258; revised April 7, 2025)
- **Forgetting Transformer: Softmax Attention with a Forget Gate** (arXiv:2503.02130; revised March 31, 2025)

We will implement **Differential Transformer** first because it is a better fit for a compact, fair decoder-only benchmark. Its core architectural change is local to the attention module and can be compared cleanly against strong decoder baselines while keeping the rest of the model stack aligned.

### Rationale

Differential Transformer is recent, credible, and architecturally distinctive without forcing a large amount of benchmark-specific engineering. It also supports a fair "same harness, different attention" comparison, which is important for a compact local benchmark.

## Model Design

All models will share a common decoder language-model interface:

- token embedding layer
- stack of decoder blocks
- final normalization
- vocabulary projection head

Only the block internals will vary between model families.

### Model 1: Vanilla Transformer LM

This baseline will use:

- causal masked multi-head self-attention
- standard feed-forward network
- learned positional embeddings
- LayerNorm or a deliberately simple normalization scheme consistent with a GPT-style baseline

### Model 2: LLaMA-Style Transformer LM

This baseline will use:

- causal masked multi-head self-attention
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward block

This gives the benchmark a stronger modern Transformer baseline than the vanilla GPT-style model alone.

### Model 3: Differential Transformer LM

This model will preserve the same macro decoder layout as the stronger baseline, but replace standard attention with **differential attention**.

Core attention behavior:

- split queries and keys into two groups
- compute two separate causal attention maps
- subtract one softmax attention map from the other, scaled by a learnable lambda term
- multiply the resulting differential attention weights by the value tensor
- apply the paper-inspired per-head normalization and output scaling choices in a compact form

The implementation goal is to capture the paper's core mechanism faithfully enough for a compact benchmark, while avoiding unnecessary complexity that would obscure comparison.

## Data Pipeline

Dataset choice:

- **WikiText-2**

Tokenization choice:

- **word-level vocabulary**

Pipeline:

1. Load raw `train`, `validation`, and `test` text splits
2. Normalize and tokenize text into word-level tokens
3. Build a vocabulary from training data with special tokens
4. Encode each split into contiguous token streams
5. Slice streams into fixed-length causal language modeling sequences
6. Form autoregressive inputs and targets by shifting by one token

The word-level approach is intentionally simple. It reduces external dependencies and keeps the benchmark compact and easy to inspect.

## Training Flow

Shared training behavior:

- same dataset preprocessing
- same sequence length
- same optimizer family
- same scheduler family
- same evaluation cadence
- same logging format
- same early-stopping or checkpoint-selection policy

Training loop:

1. Build batches of token IDs
2. Run forward pass
3. Compute cross-entropy loss for next-token prediction
4. Backpropagate and update parameters
5. Evaluate periodically on validation data
6. Save the best checkpoint under the shared selection rule
7. Run final test evaluation with the selected checkpoint

## Benchmarking Plan

The first benchmark preset will be intentionally compact:

- small decoder sizes
- matched depth and width budgets
- one local-machine-friendly sequence length
- one reproducible training preset

Reported metrics:

- validation perplexity over time
- final test perplexity
- total parameter count
- training throughput in tokens per second
- a short generation sample for qualitative sanity checking

Outputs:

- machine-readable summary file in `results/`
- human-readable markdown report in `results/`

## Parameter and Fairness Strategy

To keep the comparison meaningful:

- all models will use the same training harness
- all models will target a similar parameter budget
- all models will share the same dataset, vocabulary, split handling, and sequence length
- architectural differences will be isolated to the smallest reasonable surface area

If exact parameter equality is impractical, the benchmark will document small unavoidable differences and the reason for them.

## Error Handling and Operational Constraints

Expected constraints:

- local-only training
- no guaranteed GPU availability
- dataset download may fail or be unavailable in some environments

Planned handling:

- cache dataset downloads locally
- provide a fallback path for reruns without re-downloading
- keep configs small enough to run on CPU, though slowly
- fail clearly when required dependencies or dataset access are missing
- write partial metrics and logs in a structured way when runs are interrupted

## Testing Strategy

Testing will focus first on correctness, then on small-scope execution safety.

### Unit tests

- causal masking correctness
- RoPE application sanity
- differential attention tensor shape and masking behavior
- vocabulary and batch-building correctness
- optional tied-weights behavior if enabled

### Behavioral tests

- tiny overfit test on a miniature corpus for each model family
- smoke training test for a few optimization steps
- evaluation smoke test producing perplexity and result files

### Integration test

- one benchmark command that runs a tiny preset and verifies result artifacts are produced with the expected schema

## Success Criteria

This project is complete when all of the following are true:

- the standalone repo is created and runnable from a fresh environment
- Differential Transformer is implemented directly in PyTorch
- two strong baselines are implemented in the same codebase
- one compact benchmark command trains and evaluates all three models on WikiText-2
- results are exported in both machine-readable and human-readable form
- README documents the paper choice, exact commands, assumptions, and limitations

## Risks and Tradeoffs

### Benchmark realism

This setup prioritizes clarity and reproducibility over modern large-scale realism. Word-level WikiText-2 is a compact benchmark, not a frontier-language-model training setup.

### Paper fidelity

A compact implementation may omit some large-scale training choices from the paper. Any simplification must be documented explicitly so the benchmark does not overclaim fidelity.

### Compute variance

Throughput and wall-clock measurements will depend heavily on local hardware. Relative perplexity comparisons are more robust than raw speed comparisons across machines.

## Planned Documentation

The repository will include:

- a top-level `README.md` with setup and benchmark commands
- configuration files documenting the compact preset
- generated result summaries
- this design document

## Implementation Handoff

The next step after spec review is to write an implementation plan that decomposes the work into small, test-first tasks for creating the repository, implementing the models, adding the training harness, and running the benchmark.
