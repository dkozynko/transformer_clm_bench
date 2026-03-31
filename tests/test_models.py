import torch

from transformer_clm_bench.modeling import build_model
from transformer_clm_bench.training import compute_loss


def test_vanilla_and_llama_models_return_vocab_logits():
    x = torch.randint(0, 32, (2, 16))
    for name in ("vanilla", "llama"):
        model = build_model(
            name=name,
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=16,
        )
        y = model(x)
        assert y.shape == (2, 16, 32)


def test_differential_model_returns_vocab_logits():
    x = torch.randint(0, 32, (2, 16))
    model = build_model(
        name="differential",
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
    )
    y = model(x)
    assert y.shape == (2, 16, 32)


def test_model_initialization_keeps_initial_loss_in_reasonable_range():
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    for name in ("vanilla", "llama", "differential"):
        model = build_model(
            name=name,
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=16,
        )
        logits = model(x)
        loss = compute_loss(logits, y)
        assert loss.item() < 10.0
