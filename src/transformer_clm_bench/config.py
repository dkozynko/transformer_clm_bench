from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BenchmarkConfig:
    model_names: list[str] = field(default_factory=list)
    seq_len: int = 64
    batch_size: int = 16
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_steps: int = 200

    @classmethod
    def default_compact(cls) -> "BenchmarkConfig":
        return cls(model_names=["vanilla", "llama", "differential"])
