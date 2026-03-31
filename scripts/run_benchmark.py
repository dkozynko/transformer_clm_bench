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
    parser = argparse.ArgumentParser(description="Run the compact Transformer CLM benchmark.")
    parser.add_argument("--preset", default="compact", choices=["compact"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.preset != "compact":
        raise ValueError(f"Unsupported preset: {args.preset}")

    config = BenchmarkConfig.default_compact()
    config.device = args.device
    summary = run_benchmark(config)
    paths = write_benchmark_report(summary, config.output_dir)
    print(f"Wrote benchmark summary to {paths['json']}")
    print(f"Wrote benchmark report to {paths['markdown']}")


if __name__ == "__main__":
    main()
