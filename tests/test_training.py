import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformer_clm_bench.benchmark import aggregate_seed_summaries, write_benchmark_report
from transformer_clm_bench.training import evaluate_model, train_model


class _LogitsFromInput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*x.shape, 2)
        logits[..., 0] = x.float() * 4.0
        return logits


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 4)
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(x))


def test_evaluate_model_weights_loss_by_target_tokens():
    x = torch.tensor([[0], [0], [1]])
    y = torch.zeros_like(x)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    metrics = evaluate_model(_LogitsFromInput(), loader, torch.device("cpu"))
    expected = F.cross_entropy(_LogitsFromInput()(x).reshape(-1, 2), y.reshape(-1)).item()

    assert metrics["tokens"] == 3
    assert metrics["loss"] == pytest.approx(expected)


def test_train_model_reports_processed_and_timed_token_counts():
    x = torch.tensor([[0], [1], [2], [3]])
    y = torch.tensor([[1], [2], [3], [0]])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    result = train_model(
        _TinyLanguageModel(),
        loader,
        loader,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=3,
        eval_interval=1,
        timing_warmup_steps=1,
    )

    assert result.training_tokens == 6
    assert result.timed_tokens == 4


def test_write_benchmark_report_creates_preset_named_files_and_preserves_metadata(tmp_path: Path):
    summary = {
        "config": {"preset_name": "meaningful", "tokenizer_name": "byte"},
        "models": [{"name": "vanilla", "test_perplexity": 12.3}],
    }
    paths = write_benchmark_report(summary, output_dir=tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["json"].name == "benchmark_summary_meaningful.json"
    assert paths["markdown"].name == "benchmark_report_meaningful.md"
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["config"]["tokenizer_name"] == "byte"


def test_aggregate_seed_summaries_reports_mean_and_sample_standard_deviation():
    summaries = [
        {
            "config": {"preset_name": "cuda-quality", "seed": 2026},
            "vocab_size": 259,
            "models": [
                {
                    "name": "fix",
                    "parameters": 100,
                    "comparison_role": "controlled_attention",
                    "validation_perplexity": 2.0,
                    "test_perplexity": 3.0,
                    "tokens_per_second": 10.0,
                    "steps_ran": 5,
                    "training_tokens": 20,
                    "timed_tokens": 16,
                    "test_tokens": 8,
                }
            ],
        },
        {
            "config": {"preset_name": "cuda-quality", "seed": 2027},
            "vocab_size": 259,
            "models": [
                {
                    "name": "fix",
                    "parameters": 100,
                    "comparison_role": "controlled_attention",
                    "validation_perplexity": 4.0,
                    "test_perplexity": 7.0,
                    "tokens_per_second": 14.0,
                    "steps_ran": 5,
                    "training_tokens": 20,
                    "timed_tokens": 16,
                    "test_tokens": 8,
                }
            ],
        },
    ]

    summary = aggregate_seed_summaries(summaries)
    model = summary["models"][0]

    assert summary["seeds"] == [2026, 2027]
    assert model["test_perplexity_mean"] == 5.0
    assert model["test_perplexity_std"] == pytest.approx(2.8284271247461903)
    assert model["tokens_per_second_mean"] == 12.0
    assert model["comparison_role"] == "controlled_attention"
    assert model["training_tokens"] == 20
    assert model["timed_tokens"] == 16
    assert model["test_tokens"] == 8
    assert len(model["runs"]) == 2


def test_v2_report_labels_scaffold_baselines_and_repeated_throughput(tmp_path: Path):
    summary = {
        "config": {
            "preset_name": "cuda-throughput-v2",
            "tokenizer_name": "byte",
            "device": "cuda",
            "mixed_precision": "bfloat16",
            "attention_backend": "sdpa-flash",
            "fix_backend": "fused",
            "batch_size": 4,
            "seq_len": 1024,
            "max_steps": 500,
        },
        "training_budget": {"resolved_steps": 500, "requested_epochs": None},
        "throughput_repeats": 2,
        "models": [
            {
                "name": "vanilla",
                "comparison_role": "scaffold_baseline",
                "runs": [{"tokens_per_second": 10.0}, {"tokens_per_second": 14.0}],
                "tokens_per_second_mean": 12.0,
                "tokens_per_second_std": 2.828,
                "tokens_per_second_median": 12.0,
                "tokens_per_second_min": 10.0,
                "tokens_per_second_max": 14.0,
            }
        ],
    }

    report_path = write_benchmark_report(summary, output_dir=tmp_path)["markdown"]
    report = report_path.read_text(encoding="utf-8")

    assert "Throughput repetitions: 2" in report
    assert "Median training tokens/sec: 12.00" in report
    assert "Comparison role: scaffold baseline" in report
