from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BenchmarkConfig:
    preset_name: str = "compact"
    tokenizer_name: str = "word"
    model_names: list[str] = field(default_factory=list)
    data_dir: Path = Path(".cache/wikitext-2")
    output_dir: Path = Path("results")
    seq_len: int = 64
    batch_size: int = 16
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    eval_interval: int = 10
    max_steps: int = 20
    max_vocab_size: int | None = 5000
    min_freq: int = 1
    seed: int = 2026
    device: str | None = None
    sample_prompt: str = "the meaning of life"
    max_new_tokens: int = 16
    fix_backend: str = "auto"
    attention_backend: str = "reference"
    mixed_precision: str = "none"
    timing_warmup_steps: int = 0
    model_layers: dict[str, int] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    differential_learning_rate_multiplier: float = 2.0
    train_epochs: float | None = None
    throughput_repeats: int = 1

    def layers_for(self, model_name: str) -> int:
        return self.model_layers.get(model_name, self.n_layers)

    @classmethod
    def default_advanced(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="advanced",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential", "fix"],
            seq_len=256,
            batch_size=16,
            d_model=256,
            n_layers=6,
            n_heads=8,
            learning_rate=1e-4,
            eval_interval=100,
            max_steps=3000,
            max_vocab_size=None,
            min_freq=1,
            sample_prompt="The meaning of life is a question that has",
            max_new_tokens=64,
        )

    @classmethod
    def default_compact(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="compact",
            tokenizer_name="word",
            model_names=["vanilla", "llama", "differential", "fix"],
        )

    @classmethod
    def default_meaningful(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="meaningful",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential", "fix"],
            seq_len=128,
            batch_size=24,
            d_model=96,
            n_layers=3,
            n_heads=4,
            learning_rate=2e-4,
            eval_interval=25,
            max_steps=200,
            max_vocab_size=None,
            min_freq=1,
            sample_prompt="The meaning of life is",
            max_new_tokens=48,
        )

    @classmethod
    def default_cuda_fused(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="cuda-fused",
            tokenizer_name="byte",
            model_names=["fix"],
            seq_len=1024,
            batch_size=4,
            d_model=128,
            n_layers=3,
            n_heads=4,
            learning_rate=2e-4,
            eval_interval=25,
            max_steps=200,
            max_vocab_size=None,
            min_freq=1,
            sample_prompt="The meaning of life is",
            max_new_tokens=48,
            fix_backend="fused",
            mixed_precision="bfloat16",
        )

    @classmethod
    def default_cuda_throughput(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="cuda-throughput",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential", "fix"],
            seq_len=1024,
            batch_size=4,
            d_model=128,
            n_layers=3,
            n_heads=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_interval=100,
            max_steps=200,
            max_vocab_size=None,
            min_freq=1,
            sample_prompt="The meaning of life is",
            max_new_tokens=48,
            fix_backend="fused",
            attention_backend="sdpa-flash",
            mixed_precision="bfloat16",
            timing_warmup_steps=20,
            differential_learning_rate_multiplier=1.0,
        )

    @classmethod
    def default_cuda_quality(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="cuda-quality",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential", "fix"],
            seq_len=1024,
            batch_size=4,
            d_model=128,
            n_layers=3,
            n_heads=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_interval=100,
            max_steps=500,
            max_vocab_size=None,
            min_freq=1,
            sample_prompt="The meaning of life is",
            max_new_tokens=48,
            fix_backend="fused",
            attention_backend="sdpa-flash",
            mixed_precision="bfloat16",
            timing_warmup_steps=20,
            model_layers={"vanilla": 4},
            seeds=[2026, 2027, 2028],
            differential_learning_rate_multiplier=1.0,
        )

    @classmethod
    def default_cuda_quality_v2(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="cuda-quality-v2",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential", "fix"],
            seq_len=1024,
            batch_size=4,
            d_model=128,
            n_layers=3,
            n_heads=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_interval=500,
            max_steps=0,
            max_vocab_size=None,
            min_freq=1,
            sample_prompt="The meaning of life is",
            max_new_tokens=48,
            fix_backend="fused",
            attention_backend="sdpa-flash",
            mixed_precision="bfloat16",
            timing_warmup_steps=20,
            model_layers={"vanilla": 4},
            seeds=[2026, 2027, 2028],
            differential_learning_rate_multiplier=1.0,
            train_epochs=1,
        )

    @classmethod
    def default_cuda_throughput_v2(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="cuda-throughput-v2",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential", "fix"],
            seq_len=1024,
            batch_size=4,
            d_model=128,
            n_layers=3,
            n_heads=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_interval=500,
            max_steps=500,
            max_vocab_size=None,
            min_freq=1,
            sample_prompt="The meaning of life is",
            max_new_tokens=48,
            fix_backend="fused",
            attention_backend="sdpa-flash",
            mixed_precision="bfloat16",
            timing_warmup_steps=50,
            differential_learning_rate_multiplier=1.0,
            throughput_repeats=5,
        )
