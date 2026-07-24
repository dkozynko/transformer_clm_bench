from hashlib import sha256

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformer_clm_bench.benchmark import (
    aggregate_throughput_summaries,
    collect_provenance,
    cyclic_model_orders,
    run_benchmark,
    run_repeated_throughput_benchmark,
    resolve_training_budget,
)
from transformer_clm_bench.config import BenchmarkConfig


def _write_tiny_corpus(data_dir):
    data_dir.mkdir()
    for split in ("train", "validation", "test"):
        (data_dir / f"{split}.txt").write_text("abcdefgh", encoding="utf-8")


def test_resolve_training_budget_uses_the_actual_loader_length_for_epochs():
    x = torch.arange(10).view(5, 2)
    loader = DataLoader(TensorDataset(x, x), batch_size=2)
    config = BenchmarkConfig(max_steps=99, train_epochs=1.5)

    budget = resolve_training_budget(config, loader)

    assert budget["batches_per_epoch"] == 3
    assert budget["resolved_steps"] == 5
    assert budget["requested_epochs"] == 1.5


def test_cyclic_model_orders_rotate_the_first_model_each_repeat():
    orders = cyclic_model_orders(["vanilla", "llama", "differential", "fix"], repeats=5)

    assert orders == [
        ["vanilla", "llama", "differential", "fix"],
        ["llama", "differential", "fix", "vanilla"],
        ["differential", "fix", "vanilla", "llama"],
        ["fix", "vanilla", "llama", "differential"],
        ["vanilla", "llama", "differential", "fix"],
    ]


def test_aggregate_throughput_summaries_is_independent_of_each_run_order():
    summaries = [
        {
            "models": [
                {"name": "llama", "tokens_per_second": 20.0},
                {"name": "fix", "tokens_per_second": 10.0},
            ]
        },
        {
            "models": [
                {"name": "fix", "tokens_per_second": 14.0},
                {"name": "llama", "tokens_per_second": 24.0},
            ]
        },
    ]

    aggregate = aggregate_throughput_summaries(summaries, model_names=["llama", "fix"])

    llama, fix = aggregate["models"]
    assert llama["tokens_per_second_mean"] == 22.0
    assert llama["tokens_per_second_median"] == 22.0
    assert llama["tokens_per_second_min"] == 20.0
    assert llama["tokens_per_second_max"] == 24.0
    assert fix["tokens_per_second_std"] == pytest.approx(2.8284271247461903)
    assert [run["tokens_per_second"] for run in fix["runs"]] == [10.0, 14.0]


def test_collect_provenance_hashes_data_without_cuda_or_a_git_checkout(tmp_path):
    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    train_path.write_text("train", encoding="utf-8")
    valid_path.write_text("valid", encoding="utf-8")
    test_path.write_text("test", encoding="utf-8")

    provenance = collect_provenance(
        {"train": train_path, "validation": valid_path, "test": test_path},
        repo_dir=tmp_path,
    )

    assert provenance["git"]["revision"] is None
    assert provenance["cuda"]["available"] is False
    assert provenance["data_sha256"]["train"] == sha256(b"train").hexdigest()
    assert provenance["torch_version"] == torch.__version__


def test_run_benchmark_resolves_epoch_budget_and_records_provenance(tmp_path):
    data_dir = tmp_path / "data"
    _write_tiny_corpus(data_dir)
    config = BenchmarkConfig(
        preset_name="quality-v2-test",
        tokenizer_name="byte",
        model_names=["llama"],
        data_dir=data_dir,
        seq_len=2,
        batch_size=2,
        d_model=8,
        n_layers=1,
        n_heads=2,
        eval_interval=1,
        max_steps=0,
        train_epochs=1,
    )

    summary = run_benchmark(config)

    assert summary["training_budget"]["resolved_steps"] == 2
    assert summary["models"][0]["training_tokens"] == 8
    assert summary["provenance"]["data_sha256"]["train"]


def test_repeated_throughput_benchmark_records_cyclic_repeat_metadata(tmp_path):
    data_dir = tmp_path / "data"
    _write_tiny_corpus(data_dir)
    config = BenchmarkConfig(
        preset_name="throughput-v2-test",
        tokenizer_name="byte",
        model_names=["llama", "fix"],
        data_dir=data_dir,
        seq_len=2,
        batch_size=2,
        d_model=8,
        n_layers=1,
        n_heads=2,
        eval_interval=2,
        max_steps=2,
        throughput_repeats=2,
    )

    summary = run_repeated_throughput_benchmark(config)

    assert summary["repeat_orders"] == [["llama", "fix"], ["fix", "llama"]]
    assert summary["throughput_repeats"] == 2
    assert [run["repeat_index"] for run in summary["models"][0]["runs"]] == [0, 1]
