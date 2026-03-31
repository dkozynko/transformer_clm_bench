from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BenchmarkConfig:
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

    @classmethod
    def default_compact(cls) -> "BenchmarkConfig":
        return cls(model_names=["vanilla", "llama", "differential"])
