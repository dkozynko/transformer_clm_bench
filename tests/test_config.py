from transformer_clm_bench.config import BenchmarkConfig


def test_default_config_has_expected_model_names():
    config = BenchmarkConfig.default_compact()
    assert config.model_names == ["vanilla", "llama", "differential"]
