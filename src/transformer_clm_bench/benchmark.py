from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean, median, stdev

import torch
from torch.utils.data import DataLoader

from .config import BenchmarkConfig
from .data import LanguageModelingDataset, decode_token_ids, encode_text, ensure_wikitext2_dataset, load_corpus_bundle
from .modeling import build_model
from .training import autocast_context, evaluate_model, resolve_autocast_dtype, resolve_device, set_seed, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_training_budget(config: BenchmarkConfig, train_loader) -> dict[str, int | float | None]:
    batches_per_epoch = len(train_loader)
    if config.train_epochs is None:
        resolved_steps = config.max_steps
    else:
        if config.train_epochs <= 0:
            raise ValueError("train_epochs must be positive when provided.")
        resolved_steps = math.ceil(batches_per_epoch * config.train_epochs)
    if resolved_steps <= config.timing_warmup_steps:
        raise ValueError("Resolved training steps must exceed timing_warmup_steps.")
    return {
        "requested_epochs": config.train_epochs,
        "batches_per_epoch": batches_per_epoch,
        "resolved_steps": resolved_steps,
    }


def cyclic_model_orders(model_names: list[str], *, repeats: int) -> list[list[str]]:
    if not model_names:
        raise ValueError("At least one model is required for cyclic ordering.")
    if repeats <= 0:
        raise ValueError("throughput repeats must be positive.")
    return [model_names[repeat % len(model_names) :] + model_names[: repeat % len(model_names)] for repeat in range(repeats)]


def aggregate_throughput_summaries(seed_summaries: list[dict], *, model_names: list[str]) -> dict:
    if not seed_summaries:
        raise ValueError("At least one throughput summary is required for aggregation.")
    models = []
    for name in model_names:
        runs = [next(model for model in summary["models"] if model["name"] == name) for summary in seed_summaries]
        values = [run["tokens_per_second"] for run in runs]
        model = {
            key: runs[0][key]
            for key in ("name", "parameters", "n_layers", "attention_backend", "comparison_role")
            if key in runs[0]
        }
        model.update(
            {
                "runs": runs,
                "tokens_per_second_mean": fmean(values),
                "tokens_per_second_std": stdev(values) if len(values) > 1 else 0.0,
                "tokens_per_second_median": median(values),
                "tokens_per_second_min": min(values),
                "tokens_per_second_max": max(values),
            }
        )
        models.append(model)
    return {"repeats": len(seed_summaries), "models": models}


def _run_optional_command(command: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(data_paths: dict[str, Path], *, repo_dir: Path = PROJECT_ROOT) -> dict:
    revision = _run_optional_command(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    status = _run_optional_command(["git", "status", "--porcelain"], cwd=repo_dir)
    cuda_available = torch.cuda.is_available()
    cuda: dict[str, object] = {
        "available": cuda_available,
        "runtime_version": torch.version.cuda,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "device_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git": {"revision": revision, "dirty": None if status is None else bool(status)},
        "cuda": cuda,
        "data_sha256": {name: _sha256(path) for name, path in data_paths.items()},
    }


def generate_sample(
    model: torch.nn.Module,
    *,
    tokenizer_name: str,
    vocab: dict[str, int] | None,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 16,
    autocast_dtype: torch.dtype | None = None,
) -> str:
    model.eval()
    token_buffer = encode_text(prompt, tokenizer_name=tokenizer_name, vocab=vocab)
    x = torch.tensor([token_buffer], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with autocast_context(device, autocast_dtype):
                logits = model(x[:, -model.max_seq_len :])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            if tokenizer_name == "word" and vocab is not None and next_token.item() == vocab["<eos>"]:
                break
            if tokenizer_name == "byte" and next_token.item() == 258:
                break
    return decode_token_ids(x[0], tokenizer_name=tokenizer_name, vocab=vocab)


def run_benchmark(config: BenchmarkConfig) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    autocast_dtype = resolve_autocast_dtype(config.mixed_precision, device)
    data_paths = ensure_wikitext2_dataset(config.data_dir)
    corpus = load_corpus_bundle(
        config.data_dir,
        tokenizer_name=config.tokenizer_name,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    train_dataset = LanguageModelingDataset(corpus.train_ids, config.seq_len)
    budget_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    training_budget = resolve_training_budget(config, budget_loader)
    valid_loader = DataLoader(LanguageModelingDataset(corpus.valid_ids, config.seq_len), batch_size=config.batch_size)
    test_loader = DataLoader(LanguageModelingDataset(corpus.test_ids, config.seq_len), batch_size=config.batch_size)

    summary = {
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
            "device": str(device),
        },
        "vocab_size": corpus.vocab_size,
        "training_budget": training_budget,
        "provenance": collect_provenance(data_paths),
        "models": [],
    }

    for model_name in config.model_names:
        # Reuse the same initialization and shuffled-token order for every architecture in a seed.
        set_seed(config.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        lr = config.learning_rate * (
            config.differential_learning_rate_multiplier if model_name == "differential" else 1.0
        )
        model = build_model(
            name=model_name,
            vocab_size=corpus.vocab_size,
            d_model=config.d_model,
            n_layers=config.layers_for(model_name),
            n_heads=config.n_heads,
            max_seq_len=config.seq_len,
            dropout=config.dropout,
            fix_backend=config.fix_backend,
            attention_backend=config.attention_backend,
        )
        train_result = train_model(
            model,
            train_loader,
            valid_loader,
            device=device,
            learning_rate=lr,
            weight_decay=config.weight_decay,
            max_steps=training_budget["resolved_steps"],
            eval_interval=config.eval_interval,
            autocast_dtype=autocast_dtype,
            timing_warmup_steps=config.timing_warmup_steps,
        )
        test_metrics = evaluate_model(model, test_loader, device, autocast_dtype=autocast_dtype)
        sample = generate_sample(
            model,
            tokenizer_name=corpus.tokenizer_name,
            vocab=corpus.vocab,
            prompt=config.sample_prompt,
            device=device,
            max_new_tokens=config.max_new_tokens,
            autocast_dtype=autocast_dtype,
        )
        model_summary = {
            "name": model_name,
            "parameters": sum(param.numel() for param in model.parameters()),
            "n_layers": config.layers_for(model_name),
            "attention_backend": "fla-fused" if model_name == "fix" and config.fix_backend == "fused" else config.attention_backend,
            "validation_perplexity": train_result.best_validation_perplexity,
            "test_perplexity": test_metrics["perplexity"],
            "test_tokens": test_metrics["tokens"],
            "tokens_per_second": train_result.tokens_per_second,
            "steps_ran": train_result.steps_ran,
            "training_tokens": train_result.training_tokens,
            "timed_tokens": train_result.timed_tokens,
            "sample": sample,
        }
        if config.preset_name.endswith("-v2"):
            model_summary["comparison_role"] = "scaffold_baseline" if model_name == "vanilla" else "controlled_attention"
        summary["models"].append(model_summary)
    return summary


def aggregate_seed_summaries(seed_summaries: list[dict]) -> dict:
    if not seed_summaries:
        raise ValueError("At least one seed summary is required for aggregation.")

    reference = seed_summaries[0]
    seeds = [summary["config"]["seed"] for summary in seed_summaries]
    aggregate = {
        "config": reference["config"],
        "vocab_size": reference["vocab_size"],
        "seeds": seeds,
        "models": [],
    }
    for reference_model in reference["models"]:
        name = reference_model["name"]
        runs = [next(model for model in summary["models"] if model["name"] == name) for summary in seed_summaries]
        model = {
            key: reference_model[key]
            for key in (
                "name",
                "parameters",
                "n_layers",
                "attention_backend",
                "steps_ran",
                "comparison_role",
                "training_tokens",
                "timed_tokens",
                "test_tokens",
            )
            if key in reference_model
        }
        model["runs"] = runs
        for metric in ("validation_perplexity", "test_perplexity", "tokens_per_second"):
            values = [run[metric] for run in runs]
            model[f"{metric}_mean"] = fmean(values)
            model[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate["models"].append(model)
    return aggregate


def run_seeded_benchmark(config: BenchmarkConfig) -> dict:
    if not config.seeds:
        return run_benchmark(config)

    seed_summaries = [run_benchmark(replace(config, seed=seed, seeds=[])) for seed in config.seeds]
    aggregate = aggregate_seed_summaries(seed_summaries)
    aggregate["config"] = {
        **aggregate["config"],
        "seeds": config.seeds,
    }
    return aggregate


def run_repeated_throughput_benchmark(config: BenchmarkConfig) -> dict:
    if config.throughput_repeats <= 0:
        raise ValueError("throughput_repeats must be positive.")
    orders = cyclic_model_orders(config.model_names, repeats=config.throughput_repeats)
    repeat_summaries = []
    for repeat_index, order in enumerate(orders):
        summary = run_benchmark(replace(config, model_names=order, seeds=[]))
        summary["repeat_index"] = repeat_index
        summary["model_order"] = order
        repeat_summaries.append(summary)

    aggregate = aggregate_throughput_summaries(repeat_summaries, model_names=config.model_names)
    for model in aggregate["models"]:
        for run, summary in zip(model["runs"], repeat_summaries, strict=True):
            run["repeat_index"] = summary["repeat_index"]
            run["execution_position"] = summary["model_order"].index(model["name"])
    reference = repeat_summaries[0]
    return {
        "config": {**reference["config"], "model_names": config.model_names},
        "vocab_size": reference["vocab_size"],
        "training_budget": reference["training_budget"],
        "provenance": reference["provenance"],
        "throughput_repeats": config.throughput_repeats,
        "repeat_orders": orders,
        "models": aggregate["models"],
    }


def write_benchmark_report(summary: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = summary.get("config", {})
    preset_name = config.get("preset_name", "benchmark")
    tokenizer_name = config.get("tokenizer_name", "unknown")
    json_path = output_dir / f"benchmark_summary_{preset_name}.json"
    markdown_path = output_dir / f"benchmark_report_{preset_name}.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Benchmark Report",
        "",
        f"- Preset: `{preset_name}`",
        f"- Tokenizer: `{tokenizer_name}`",
        f"- Device: `{config['device']}`" if "device" in config else "- Device: unknown",
        f"- Mixed precision: `{config['mixed_precision']}`" if "mixed_precision" in config else "- Mixed precision: unknown",
        f"- Attention backend: `{config['attention_backend']}`" if "attention_backend" in config else "- Attention backend: unknown",
        f"- FiX backend: `{config['fix_backend']}`" if "fix_backend" in config else "- FiX backend: unknown",
        f"- Workload: batch {config['batch_size']}, sequence {config['seq_len']}, {summary.get('training_budget', {}).get('resolved_steps', config['max_steps'])} steps" if {"batch_size", "seq_len", "max_steps"} <= config.keys() else "- Workload: unknown",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}" if "seeds" in summary else "- Seeds: single run",
    ]
    training_budget = summary.get("training_budget")
    if training_budget is not None:
        lines.append(f"- Resolved training steps: {training_budget['resolved_steps']}")
        if training_budget["requested_epochs"] is not None:
            lines.append(f"- Requested training epochs: {training_budget['requested_epochs']}")
    if "throughput_repeats" in summary:
        lines.append(f"- Throughput repetitions: {summary['throughput_repeats']}")
    provenance = summary.get("provenance")
    if provenance is not None:
        git = provenance["git"]
        lines.extend(
            [
                f"- Git revision: `{git['revision'] or 'unavailable'}`",
                f"- Git dirty: `{git['dirty']}`",
                f"- PyTorch: `{provenance['torch_version']}`",
                f"- CUDA runtime: `{provenance['cuda']['runtime_version'] or 'unavailable'}`",
            ]
        )
    lines.extend(["", "## Models", ""])
    for model in summary.get("models", []):
        comparison_role = model.get("comparison_role", "").replace("_", " ")
        if "tokens_per_second_median" in model:
            individual_runs = ", ".join(f"{run['tokens_per_second']:.2f}" for run in model["runs"])
            lines.extend(
                [
                    f"### {model['name']}",
                    f"- Mean training tokens/sec: {model['tokens_per_second_mean']:.2f}",
                    f"- Sample std training tokens/sec: {model['tokens_per_second_std']:.2f}",
                    f"- Median training tokens/sec: {model['tokens_per_second_median']:.2f}",
                    f"- Minimum training tokens/sec: {model['tokens_per_second_min']:.2f}",
                    f"- Maximum training tokens/sec: {model['tokens_per_second_max']:.2f}",
                    f"- Individual training tokens/sec: {individual_runs}",
                    f"- Comparison role: {comparison_role}" if comparison_role else "- Comparison role: n/a",
                    "",
                ]
            )
            continue
        if "runs" in model:
            lines.extend(
                [
                    f"### {model['name']}",
                    f"- Test perplexity (mean +/- sample std): {model['test_perplexity_mean']:.4f} +/- {model['test_perplexity_std']:.4f}",
                    f"- Validation perplexity (mean +/- sample std): {model['validation_perplexity_mean']:.4f} +/- {model['validation_perplexity_std']:.4f}",
                    f"- Steady-state training tokens/sec (mean +/- sample std): {model['tokens_per_second_mean']:.2f} +/- {model['tokens_per_second_std']:.2f}",
                    f"- Parameters: {model['parameters']}" if "parameters" in model else "- Parameters: n/a",
                    f"- Layers: {model['n_layers']}" if "n_layers" in model else "- Layers: n/a",
                    f"- Attention backend: `{model['attention_backend']}`" if "attention_backend" in model else "- Attention backend: n/a",
                    f"- Comparison role: {comparison_role}" if comparison_role else "- Comparison role: n/a",
                    "",
                ]
            )
            continue
        validation_perplexity = model.get("validation_perplexity")
        parameters = model.get("parameters")
        tokens_per_second = model.get("tokens_per_second")
        sample = model.get("sample", "")
        lines.extend(
            [
                f"### {model['name']}",
                f"- Test perplexity: {model['test_perplexity']:.4f}",
                f"- Validation perplexity: {validation_perplexity:.4f}" if validation_perplexity is not None else "- Validation perplexity: n/a",
                f"- Parameters: {parameters}" if parameters is not None else "- Parameters: n/a",
                f"- Tokens/sec: {tokens_per_second:.2f}" if tokens_per_second is not None else "- Tokens/sec: n/a",
                f"- Comparison role: {comparison_role}" if comparison_role else "- Comparison role: n/a",
                f"- Sample: `{sample}`" if sample else "- Sample: n/a",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
