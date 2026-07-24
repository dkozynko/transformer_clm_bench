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


def test_cuda_throughput_config_uses_common_shapes_and_strict_kernels():
    config = BenchmarkConfig.default_cuda_throughput()

    assert config.model_names == ["vanilla", "llama", "differential", "fix"]
    assert config.attention_backend == "sdpa-flash"
    assert config.fix_backend == "fused"
    assert config.mixed_precision == "bfloat16"
    assert config.d_model == 128
    assert config.n_layers == 3
    assert config.d_model // config.n_heads == 32
    assert config.seq_len * config.batch_size == 4096
    assert config.timing_warmup_steps > 0


def test_cuda_quality_config_has_three_seeds_and_near_parameter_layers():
    config = BenchmarkConfig.default_cuda_quality()

    assert config.seeds == [2026, 2027, 2028]
    assert config.max_steps == 500
    assert config.layers_for("vanilla") == 4
    assert config.layers_for("llama") == 3
    assert config.layers_for("differential") == 3
    assert config.layers_for("fix") == 3
    assert config.differential_learning_rate_multiplier == 1.0


def test_cuda_quality_v2_config_uses_one_epoch_and_strict_cuda_kernels():
    config = BenchmarkConfig.default_cuda_quality_v2()

    assert config.preset_name == "cuda-quality-v2"
    assert config.train_epochs == 1
    assert config.seeds == [2026, 2027, 2028]
    assert config.attention_backend == "sdpa-flash"
    assert config.fix_backend == "fused"
    assert config.mixed_precision == "bfloat16"
    assert config.layers_for("vanilla") == 4
    assert config.layers_for("llama") == 3
    assert config.layers_for("differential") == 3
    assert config.layers_for("fix") == 3


def test_cuda_throughput_v2_config_uses_repeated_shape_matched_measurements():
    config = BenchmarkConfig.default_cuda_throughput_v2()

    assert config.preset_name == "cuda-throughput-v2"
    assert config.throughput_repeats == 5
    assert config.n_layers == 3
    assert config.model_layers == {}
    assert config.attention_backend == "sdpa-flash"
    assert config.fix_backend == "fused"
    assert config.mixed_precision == "bfloat16"
    assert config.timing_warmup_steps > 0
