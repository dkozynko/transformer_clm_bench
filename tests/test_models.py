import torch

from transformer_clm_bench.modeling import build_model


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
