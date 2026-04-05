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

    @classmethod
    def default_advanced(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="advanced",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential"],
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
            model_names=["vanilla", "llama", "differential"],
        )

    @classmethod
    def default_meaningful(cls) -> "BenchmarkConfig":
        return cls(
            preset_name="meaningful",
            tokenizer_name="byte",
            model_names=["vanilla", "llama", "differential"],
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
