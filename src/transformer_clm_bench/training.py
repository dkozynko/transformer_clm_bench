from __future__ import annotations

import copy
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class TrainResult:
    best_validation_loss: float
    best_validation_perplexity: float
    steps_ran: int
    tokens_per_second: float
    training_tokens: int
    timed_tokens: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def resolve_autocast_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if requested == "none":
        return None
    if requested == "bfloat16" and device.type == "cuda":
        return torch.bfloat16
    if requested == "bfloat16":
        raise ValueError("bfloat16 mixed precision is only supported for CUDA benchmarks.")
    raise ValueError(f"Unknown mixed precision setting: {requested}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_idx, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device)
        y = y.to(device)
        with autocast_context(device, autocast_dtype):
            logits = model(x)
            loss = compute_loss(logits, y)
        token_count = y.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count
        if max_batches is not None and batch_idx >= max_batches:
            break
    avg_loss = total_loss / max(total_tokens, 1)
    return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20)), "tokens": total_tokens}


def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_steps: int,
    eval_interval: int,
    autocast_dtype: torch.dtype | None = None,
    timing_warmup_steps: int = 0,
) -> TrainResult:
    if not 0 <= timing_warmup_steps < max_steps:
        raise ValueError("timing_warmup_steps must be non-negative and smaller than max_steps.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    train_iterator = iter(train_loader)
    training_tokens_seen = 0
    timed_tokens_seen = 0
    timed_elapsed = 0.0
    timing_started_at: float | None = None

    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(device)
        y = y.to(device)
        training_tokens_seen += x.numel()
        if step > timing_warmup_steps and timing_started_at is None:
            synchronize_device(device)
            timing_started_at = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, autocast_dtype):
            logits = model(x)
            loss = compute_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step > timing_warmup_steps:
            timed_tokens_seen += x.numel()

        if step % eval_interval == 0 or step == max_steps:
            if timing_started_at is not None:
                synchronize_device(device)
                timed_elapsed += time.perf_counter() - timing_started_at
                timing_started_at = None
            valid_metrics = evaluate_model(model, valid_loader, device, autocast_dtype=autocast_dtype)
            if valid_metrics["loss"] < best_validation_loss:
                best_validation_loss = valid_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())

    if timing_started_at is not None:
        synchronize_device(device)
        timed_elapsed += time.perf_counter() - timing_started_at
    elapsed = max(timed_elapsed, 1e-6)
    model.load_state_dict(best_state)
    return TrainResult(
        best_validation_loss=best_validation_loss,
        best_validation_perplexity=math.exp(min(best_validation_loss, 20)),
        steps_ran=max_steps,
        tokens_per_second=timed_tokens_seen / elapsed,
        training_tokens=training_tokens_seen,
        timed_tokens=timed_tokens_seen,
    )
