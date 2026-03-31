# Meaningful Benchmark Preset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a more meaningful byte-level, roughly 10-minute benchmark preset while preserving the current compact smoke-test preset.

**Architecture:** Keep one shared benchmark harness, but make tokenization, preset sizing, output naming, and generation behavior preset-aware. The compact preset remains a fast word-level smoke test, while the meaningful preset switches to byte-level tokenization and a larger training budget for more informative comparison.

**Tech Stack:** Python 3.13, PyTorch, pytest, tqdm, plain-file dataset cache, git

---

### Task 1: Add Preset-Aware Configuration

**Files:**
- Modify: `src/transformer_clm_bench/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add a test proving `BenchmarkConfig.default_meaningful()` exists and sets `tokenizer_name == "byte"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL because the meaningful preset does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add preset-aware config fields and factory methods for `compact` and `meaningful`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transformer_clm_bench/config.py tests/test_config.py
git commit -m "feat: add meaningful benchmark preset config"
```

### Task 2: Add Byte-Level Tokenization Support

**Files:**
- Modify: `src/transformer_clm_bench/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Add a test proving byte tokenization round-trips readable text and does not depend on a word vocabulary.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL because byte tokenization helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement preset-aware corpus loading, byte encoding/decoding helpers, and tokenizer metadata.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transformer_clm_bench/data.py tests/test_data.py
git commit -m "feat: add byte-level tokenization"
```

### Task 3: Make Benchmarking Preset-Aware

**Files:**
- Modify: `src/transformer_clm_bench/benchmark.py`
- Modify: `scripts/run_benchmark.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Write the failing test**

Add a test proving benchmark reports use preset-specific output names and include tokenizer metadata.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py -v`
Expected: FAIL because report naming and metadata are not preset-aware yet.

- [ ] **Step 3: Write minimal implementation**

Make benchmark loading, sample generation, summary metadata, and output file names work for both presets.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_training.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transformer_clm_bench/benchmark.py scripts/run_benchmark.py tests/test_training.py
git commit -m "feat: support compact and meaningful benchmark presets"
```

### Task 4: Verify Both Presets And Record The Meaningful Run

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Modify: `results/` (generated)

- [ ] **Step 1: Update docs before final verification**

Document both presets, their commands, and the output file layout.

- [ ] **Step 2: Run full verification**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 3: Run the compact preset**

Run:

```bash
python3 scripts/run_benchmark.py --preset compact
```

Expected: exit code 0 and compact report files written.

- [ ] **Step 4: Run the meaningful preset**

Run:

```bash
python3 scripts/run_benchmark.py --preset meaningful
```

Expected: exit code 0, meaningful report files written, readable byte-decoded samples, and non-degenerate perplexity values.

- [ ] **Step 5: Commit**

```bash
git add README.md .gitignore results
git commit -m "docs: record meaningful benchmark preset"
```
