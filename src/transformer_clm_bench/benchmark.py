from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import BenchmarkConfig
from .data import LanguageModelingDataset, decode_ids, encode_tokens, load_corpus_bundle
from .modeling import build_model
from .training import evaluate_model, resolve_device, set_seed, train_model


def generate_sample(
    model: torch.nn.Module,
    vocab: dict[str, int],
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 16,
) -> str:
    model.eval()
    token_buffer = encode_tokens(["<bos>", *prompt.split()], vocab)
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    x = torch.tensor([token_buffer], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x[:, -model.max_seq_len :])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            if next_token.item() == vocab["<eos>"]:
                break
    return " ".join(reverse_vocab.get(token_id, "<unk>") for token_id in x[0].tolist())


def run_benchmark(config: BenchmarkConfig) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    corpus = load_corpus_bundle(
        config.data_dir,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    train_loader = DataLoader(
        LanguageModelingDataset(corpus.train_ids, config.seq_len),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(LanguageModelingDataset(corpus.valid_ids, config.seq_len), batch_size=config.batch_size)
    test_loader = DataLoader(LanguageModelingDataset(corpus.test_ids, config.seq_len), batch_size=config.batch_size)

    summary = {
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
            "device": str(device),
        },
        "vocab_size": len(corpus.vocab),
        "models": [],
    }

    for model_name in config.model_names:
        model = build_model(
            name=model_name,
            vocab_size=len(corpus.vocab),
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            max_seq_len=config.seq_len,
            dropout=config.dropout,
        )
        train_result = train_model(
            model,
            train_loader,
            valid_loader,
            device=device,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_steps=config.max_steps,
            eval_interval=config.eval_interval,
        )
        test_metrics = evaluate_model(model, test_loader, device)
        sample = generate_sample(model, corpus.vocab, config.sample_prompt, device=device)
        summary["models"].append(
            {
                "name": model_name,
                "parameters": sum(param.numel() for param in model.parameters()),
                "validation_perplexity": train_result.best_validation_perplexity,
                "test_perplexity": test_metrics["perplexity"],
                "tokens_per_second": train_result.tokens_per_second,
                "steps_ran": train_result.steps_ran,
                "sample": sample,
            }
        )
    return summary


def write_benchmark_report(summary: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_summary.json"
    markdown_path = output_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Benchmark Report", "", "## Models", ""]
    for model in summary.get("models", []):
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
                f"- Sample: `{sample}`" if sample else "- Sample: n/a",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
