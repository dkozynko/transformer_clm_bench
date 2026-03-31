# Transformer CLM Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone PyTorch repository that implements Differential Transformer plus two strong causal Transformer baselines and benchmarks them on compact WikiText-2 causal language modeling.

**Architecture:** The codebase will use one shared data pipeline, one shared training and evaluation harness, and three swappable decoder language-model implementations: a GPT-style baseline, a LLaMA-style baseline, and a Differential Transformer variant. The benchmark will use a compact, word-level WikiText-2 setup and produce consistent machine-readable and markdown reports.

**Tech Stack:** Python 3.13, PyTorch, pytest, tqdm, plain-file dataset download via curl, word-level tokenization, git

---

### Task 1: Bootstrap The Repository

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/transformer_clm_bench/__init__.py`
- Create: `src/transformer_clm_bench/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
from transformer_clm_bench.config import BenchmarkConfig


def test_default_config_has_expected_model_names():
    config = BenchmarkConfig.default_compact()
    assert config.model_names == ["vanilla", "llama", "differential"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL because the package and config module do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the package scaffold, default compact config, top-level project metadata, and ignored artifact directories.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore README.md src/transformer_clm_bench/__init__.py src/transformer_clm_bench/config.py tests/test_config.py
git commit -m "feat: bootstrap benchmark package"
```

### Task 2: Build The WikiText-2 Data Pipeline

**Files:**
- Create: `src/transformer_clm_bench/data.py`
- Modify: `src/transformer_clm_bench/config.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
from transformer_clm_bench.data import build_vocabulary, encode_tokens


def test_build_vocabulary_and_encoding_cover_special_tokens():
    vocab = build_vocabulary([["hello", "world"], ["hello"]], min_freq=1)
    ids = encode_tokens(["<bos>", "hello", "<eos>"], vocab)
    assert len(vocab) >= 4
    assert len(ids) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL because the data module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement raw text loading, word tokenization, vocabulary building, split encoding, fixed-length batch sampling, and a download helper for WikiText-2 raw files.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transformer_clm_bench/data.py src/transformer_clm_bench/config.py tests/test_data.py
git commit -m "feat: add wikitext data pipeline"
```

### Task 3: Implement Baseline Transformer Language Models

**Files:**
- Create: `src/transformer_clm_bench/modeling.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from transformer_clm_bench.modeling import build_model


def test_vanilla_and_llama_models_return_vocab_logits():
    x = torch.randint(0, 32, (2, 16))
    for name in ("vanilla", "llama"):
        model = build_model(name=name, vocab_size=32, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
        y = model(x)
        assert y.shape == (2, 16, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_vanilla_and_llama_models_return_vocab_logits -v`
Expected: FAIL because the modeling module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement shared decoder machinery plus the GPT-style and LLaMA-style model variants with causal masking.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_vanilla_and_llama_models_return_vocab_logits -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transformer_clm_bench/modeling.py tests/test_models.py
git commit -m "feat: add baseline transformer language models"
```

### Task 4: Implement Differential Attention And The Differential Transformer

**Files:**
- Modify: `src/transformer_clm_bench/modeling.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from transformer_clm_bench.modeling import build_model


def test_differential_model_returns_vocab_logits():
    x = torch.randint(0, 32, (2, 16))
    model = build_model(name="differential", vocab_size=32, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    y = model(x)
    assert y.shape == (2, 16, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_differential_model_returns_vocab_logits -v`
Expected: FAIL because the differential model path is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Implement differential attention, connect it to the decoder block, and expose the Differential Transformer via the shared model builder.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_differential_model_returns_vocab_logits -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transformer_clm_bench/modeling.py tests/test_models.py
git commit -m "feat: add differential transformer model"
```

### Task 5: Add Training, Evaluation, And Benchmark Reporting

**Files:**
- Create: `src/transformer_clm_bench/training.py`
- Create: `src/transformer_clm_bench/benchmark.py`
- Create: `scripts/run_benchmark.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from transformer_clm_bench.benchmark import write_benchmark_report


def test_write_benchmark_report_creates_json_and_markdown(tmp_path: Path):
    summary = {"models": [{"name": "vanilla", "test_perplexity": 12.3}]}
    paths = write_benchmark_report(summary, output_dir=tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py -v`
Expected: FAIL because the training and benchmark modules do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement the train loop, evaluation helpers, checkpoint and metric tracking, report writing, and a benchmark entrypoint that runs all configured models.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_training.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transformer_clm_bench/training.py src/transformer_clm_bench/benchmark.py scripts/run_benchmark.py tests/test_training.py
git commit -m "feat: add training and benchmark pipeline"
```

### Task 6: Run Verification And Document The Benchmark

**Files:**
- Modify: `README.md`
- Modify: `results/` (generated)

- [ ] **Step 1: Write the failing test or check**

Document the expected benchmark command and expected result files in `README.md` before the final run.

- [ ] **Step 2: Run verification to confirm the integrated benchmark works**

Run: `pytest -v`
Expected: PASS

Then run the compact benchmark command:

```bash
python3 scripts/run_benchmark.py --preset compact
```

Expected: exit code 0, benchmark summary JSON written, markdown report written, and all three models present in the output.

- [ ] **Step 3: Write minimal implementation or fixes**

Patch any integration gaps discovered by the full test run or compact benchmark run.

- [ ] **Step 4: Re-run verification**

Run:

```bash
pytest -v
python3 scripts/run_benchmark.py --preset compact
```

Expected: both commands succeed.

- [ ] **Step 5: Commit**

```bash
git add README.md results
git commit -m "docs: document and record compact benchmark"
```
