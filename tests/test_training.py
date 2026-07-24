import json
from pathlib import Path

import pytest

from transformer_clm_bench.benchmark import aggregate_seed_summaries, write_benchmark_report


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
                    "validation_perplexity": 2.0,
                    "test_perplexity": 3.0,
                    "tokens_per_second": 10.0,
                    "steps_ran": 5,
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
                    "validation_perplexity": 4.0,
                    "test_perplexity": 7.0,
                    "tokens_per_second": 14.0,
                    "steps_ran": 5,
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
    assert len(model["runs"]) == 2
