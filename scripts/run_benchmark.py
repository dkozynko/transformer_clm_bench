from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformer_clm_bench.benchmark import run_benchmark, write_benchmark_report
from transformer_clm_bench.config import BenchmarkConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Transformer CLM benchmark presets.")
    parser.add_argument("--preset", default="compact", choices=["compact", "meaningful", "advanced", "cuda-fused"])
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fix-backend", default=None, choices=["auto", "reference", "fused"])
    parser.add_argument("--mixed-precision", default=None, choices=["none", "bfloat16"])
    args = parser.parse_args()

    if args.preset == "compact":
        config = BenchmarkConfig.default_compact()
    elif args.preset == "meaningful":
        config = BenchmarkConfig.default_meaningful()
    elif args.preset == "advanced":
        config = BenchmarkConfig.default_advanced()
    elif args.preset == "cuda-fused":
        config = BenchmarkConfig.default_cuda_fused()
    else:
        raise ValueError(f"Unsupported preset: {args.preset}")
    config.device = args.device
    if args.fix_backend is not None:
        config.fix_backend = args.fix_backend
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    summary = run_benchmark(config)
    paths = write_benchmark_report(summary, config.output_dir)
    print(f"Wrote benchmark summary to {paths['json']}")
    print(f"Wrote benchmark report to {paths['markdown']}")


if __name__ == "__main__":
    main()
