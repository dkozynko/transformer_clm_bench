import torch
import pytest

from transformer_clm_bench.modeling import FiXSelfAttention, build_model
from transformer_clm_bench.training import compute_loss


def _cuda_fused_fix_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from fla.ops.fix_attn import parallel_fix_attn  # noqa: F401
    except ImportError:
        return False
    return True


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


def test_fix_auto_backend_uses_reference_for_cpu_tensors():
    attention = FiXSelfAttention(
        d_model=32,
        n_heads=4,
        use_rope=True,
        vocab_size=32,
        backend="auto",
    )
    assert attention._select_backend(torch.zeros(1, 16, 32)) == "reference"


def test_fix_fused_backend_requires_cuda():
    model = build_model(
        name="fix",
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
        fix_backend="fused",
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        model(torch.randint(0, 32, (1, 16)))


@pytest.mark.skipif(not _cuda_fused_fix_available(), reason="requires CUDA and transformer-clm-bench[fix-cuda]")
def test_fix_fused_backend_matches_reference_forward_and_backward():
    torch.manual_seed(17)
    device = torch.device("cuda")
    reference = build_model(
        name="fix",
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
        fix_backend="reference",
    ).to(device)
    fused = build_model(
        name="fix",
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
        fix_backend="fused",
    ).to(device)
    fused.load_state_dict(reference.state_dict())
    x = torch.randint(0, 32, (2, 16), device=device)
    y = torch.randint(0, 32, (2, 16), device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_logits = reference(x)
        fused_logits = fused(x)
        reference_loss = compute_loss(reference_logits, y)
        fused_loss = compute_loss(fused_logits, y)
    reference_loss.backward()
    fused_loss.backward()

    torch.testing.assert_close(reference_logits.float(), fused_logits.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(reference_loss.float(), fused_loss.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(
        reference.blocks[0].attn.qkv_proj.weight.grad.float(),
        fused.blocks[0].attn.qkv_proj.weight.grad.float(),
        rtol=1e-1,
        atol=1e-1,
    )


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
