from __future__ import annotations

import copy
import math
import random
import time
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


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_idx, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = compute_loss(logits, y)
        total_loss += loss.item()
        total_batches += 1
        if max_batches is not None and batch_idx >= max_batches:
            break
    avg_loss = total_loss / max(total_batches, 1)
    return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20))}


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
) -> TrainResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    train_iterator = iter(train_loader)
    tokens_seen = 0
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(device)
        y = y.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = compute_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tokens_seen += x.numel()

        if step % eval_interval == 0 or step == max_steps:
            valid_metrics = evaluate_model(model, valid_loader, device)
            if valid_metrics["loss"] < best_validation_loss:
                best_validation_loss = valid_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())

    elapsed = max(time.perf_counter() - started, 1e-6)
    model.load_state_dict(best_state)
    return TrainResult(
        best_validation_loss=best_validation_loss,
        best_validation_perplexity=math.exp(min(best_validation_loss, 20)),
        steps_ran=max_steps,
        tokens_per_second=tokens_seen / elapsed,
    )
