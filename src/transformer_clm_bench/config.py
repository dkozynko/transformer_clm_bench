from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BenchmarkConfig:
    model_names: list[str] = field(default_factory=list)
    data_dir: Path = Path(".cache/wikitext-2")
    seq_len: int = 64
    batch_size: int = 16
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_steps: int = 200
    min_freq: int = 1

    @classmethod
    def default_compact(cls) -> "BenchmarkConfig":
        return cls(model_names=["vanilla", "llama", "differential"])
