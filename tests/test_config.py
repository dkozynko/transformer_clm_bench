from transformer_clm_bench.config import BenchmarkConfig


def test_default_config_has_expected_model_names():
    config = BenchmarkConfig.default_compact()
    assert config.model_names == ["vanilla", "llama", "differential", "fix"]


def test_meaningful_config_uses_byte_tokenization():
    config = BenchmarkConfig.default_meaningful()
    assert config.tokenizer_name == "byte"
    assert config.preset_name == "meaningful"
    assert "fix" in config.model_names


def test_cuda_fused_config_uses_supported_fix_kernel_dimensions():
    config = BenchmarkConfig.default_cuda_fused()
    assert config.fix_backend == "fused"
    assert config.mixed_precision == "bfloat16"
    assert config.model_names == ["fix"]
    assert config.d_model // config.n_heads == 32
