from pathlib import Path

from transformer_clm_bench.benchmark import write_benchmark_report


def test_write_benchmark_report_creates_json_and_markdown(tmp_path: Path):
    summary = {"models": [{"name": "vanilla", "test_perplexity": 12.3}]}
    paths = write_benchmark_report(summary, output_dir=tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
