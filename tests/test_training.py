import json
from pathlib import Path

from transformer_clm_bench.benchmark import write_benchmark_report


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
