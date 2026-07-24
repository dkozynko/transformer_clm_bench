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


def test_fix_model_returns_vocab_logits():
    x = torch.randint(0, 32, (2, 16))
    model = build_model(
        name="fix",
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
    )
    y = model(x)
    assert y.shape == (2, 16, 32)


def test_fix_logits_do_not_depend_on_future_token_ids():
    torch.manual_seed(5)
    model = build_model(
        name="fix",
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
    ).eval()
    x = torch.randint(0, 32, (1, 16))
    changed_future = x.clone()
    changed_future[:, 8:] = torch.randint(0, 32, (1, 8))

    with torch.no_grad():
        original_logits = model(x)
        changed_logits = model(changed_future)

    torch.testing.assert_close(original_logits[:, :8], changed_logits[:, :8])


def test_fix_backward_pass_remains_finite():
    torch.manual_seed(11)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    model = build_model(
        name="fix",
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
    )
    loss = compute_loss(model(x), y)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())


def test_model_initialization_keeps_initial_loss_in_reasonable_range():
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    for name in ("vanilla", "llama", "differential", "fix"):
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
