from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(rms + self.eps)
        return normed * self.weight


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / max(half_dim, 1)))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(dtype=dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=dtype)[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim)
        self.value = nn.Linear(d_model, hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.gate(x)) * self.value(x))


class StandardSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        use_rope: bool,
        dropout: float = 0.0,
        layer_idx: int = 0,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.use_rope:
            cos, sin = build_rope_cache(seq_len, self.head_dim, x.device, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(out))


class DifferentialSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        use_rope: bool,
        dropout: float = 0.0,
        layer_idx: int = 0,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.q_proj = nn.Linear(d_model, 2 * d_model)
        self.k_proj = nn.Linear(d_model, 2 * d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.head_norm_weight = nn.Parameter(torch.ones(n_heads, self.head_dim))
        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.eps = 1e-6

    def _lambda(self) -> torch.Tensor:
        lam_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1))
        lam_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1))
        return self.lambda_init + lam_1 - lam_2

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, 2 * self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, 2 * self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        if self.use_rope:
            cos, sin = build_rope_cache(seq_len, self.head_dim, x.device, q1.dtype)
            q1 = apply_rope(q1, cos, sin)
            q2 = apply_rope(q2, cos, sin)
            k1 = apply_rope(k1, cos, sin)
            k2 = apply_rope(k2, cos, sin)

        scores_1 = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_2 = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores_1 = scores_1.masked_fill(causal_mask, torch.finfo(scores_1.dtype).min)
        scores_2 = scores_2.masked_fill(causal_mask, torch.finfo(scores_2.dtype).min)
        attn_1 = torch.softmax(scores_1, dim=-1)
        attn_2 = torch.softmax(scores_2, dim=-1)
        lambdas = self._lambda().view(1, self.n_heads, 1, 1).to(dtype=attn_1.dtype, device=x.device)
        attn = self.attn_dropout(attn_1 - lambdas * attn_2)
        out = torch.matmul(attn, v)
        # Sub-layer norm (RMSNorm style but per-head)
        rms = out.pow(2).mean(dim=-1, keepdim=True)
        out = out * torch.rsqrt(rms + self.eps)
        out = out * self.head_norm_weight.view(1, self.n_heads, 1, self.head_dim)
        out = out * (1.0 - self.lambda_init)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(out))


class FiXSelfAttention(nn.Module):
    """Fine-grained Forgetting Attention from Li et al. (ICML 2026).

    The reference backend materializes feature-wise decay for portable,
    short-context verification. The optional fused backend delegates to the
    official CUDA/Triton custom-autograd implementation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        use_rope: bool,
        dropout: float = 0.0,
        layer_idx: int = 0,
        vocab_size: int | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if vocab_size is None:
            raise ValueError("FiX requires vocab_size for the first-layer forget-gate embedding.")
        if backend not in {"auto", "reference", "fused"}:
            raise ValueError("FiX backend must be one of: auto, reference, fused.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.layer_idx = layer_idx
        self.backend = backend
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # The official fused operator normalizes each head before the output
        # projection, with one affine vector shared across heads.
        self.head_norm = RMSNorm(self.head_dim, eps=1e-30)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if layer_idx == 0:
            # The paper avoids a hidden-state projection for discrete input
            # embeddings because its first-layer gate gradients are unstable.
            self.token_gate_embedding = nn.Embedding(vocab_size, d_model)
            self.gate_down = None
            self.gate_norm = None
            self.gate_up = None
            self.log_gate_strength = None
        else:
            gate_rank = max(1, math.ceil(d_model / 16))
            self.token_gate_embedding = None
            self.gate_down = nn.Linear(d_model, gate_rank, bias=False)
            self.gate_norm = RMSNorm(gate_rank)
            self.gate_up = nn.Linear(gate_rank, d_model)
            self.log_gate_strength = nn.Parameter(torch.empty(d_model))

    def reset_gate_parameters(self) -> None:
        """Initialize FiX decay scales with the paper's Mamba-style recipe."""
        if self.log_gate_strength is None or self.gate_up is None:
            return
        with torch.no_grad():
            strength = torch.empty_like(self.log_gate_strength).uniform_(0.0, 16.0).clamp_min_(1e-4)
            self.log_gate_strength.copy_(strength.log())
            time_step = torch.empty_like(self.gate_up.bias).uniform_(math.log(1e-3), math.log(1e-1)).exp_()
            time_step.clamp_min_(1e-4)
            # softplus^{-1}(x) = x + log(1 - exp(-x)), computed stably.
            self.gate_up.bias.copy_(time_step + torch.log(-torch.expm1(-time_step)))

    def _log_forget_gates(self, x: torch.Tensor, token_ids: torch.Tensor | None) -> torch.Tensor:
        if self.token_gate_embedding is not None:
            if token_ids is None:
                raise ValueError("FiX first-layer attention requires token IDs.")
            return F.logsigmoid(-self.token_gate_embedding(token_ids).float())

        assert self.gate_down is not None
        assert self.gate_norm is not None
        assert self.gate_up is not None
        assert self.log_gate_strength is not None
        gate_logits = self.gate_up(self.gate_norm(self.gate_down(x)))
        # sigmoid(-x) ** a = exp(-softplus(x) * a). The log form is both
        # faithful to FiX and stable when decays span many timesteps.
        return (-F.softplus(gate_logits.float()) * self.log_gate_strength.float().exp()).float()

    def _fused_unavailable_reason(self, x: torch.Tensor) -> str | None:
        if x.device.type != "cuda":
            return "requires CUDA tensors"
        if x.dtype != torch.bfloat16:
            return "requires torch.bfloat16 activations and parameters"
        if self.head_dim not in {16, 32, 64, 128}:
            return "requires a head dimension in {16, 32, 64, 128}"
        if self.attn_dropout.p != 0.0:
            return "does not support attention dropout"
        try:
            from fla.ops.fix_attn import parallel_fix_attn  # noqa: F401
        except ImportError:
            return "requires the pinned optional `fix-cuda` dependency"
        return None

    def _select_backend(self, q: torch.Tensor) -> str:
        if self.backend == "reference":
            return "reference"
        unavailable_reason = self._fused_unavailable_reason(q)
        if unavailable_reason is None:
            return "fused"
        if self.backend == "fused":
            raise RuntimeError(f"FiX fused backend {unavailable_reason}.")
        return "reference"

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        if self.use_rope:
            cos, sin = build_rope_cache(seq_len, self.head_dim, x.device, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        return q, k, v

    def _forward_reference(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))

        log_gates = self._log_forget_gates(x, token_ids)
        log_gates = log_gates.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        cumulative_log_gates = log_gates.cumsum(dim=2)
        log_decay = cumulative_log_gates.unsqueeze(3) - cumulative_log_gates.unsqueeze(2)
        log_decay = log_decay.masked_fill(causal_mask.view(1, 1, seq_len, seq_len, 1), 0.0)
        value_decay = log_decay.exp().to(dtype=v.dtype)

        out = torch.einsum("bhts,bhtsd,bhsd->bhtd", attn, value_decay, v)
        out = self.head_norm(out)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(out))

    def _forward_fused(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        from fla.ops.fix_attn import parallel_fix_attn

        batch_size, seq_len, d_model = x.shape
        log_gates = self._log_forget_gates(x, token_ids)
        log_gates = log_gates.view(batch_size, seq_len, self.n_heads, self.head_dim).contiguous()
        out = parallel_fix_attn(
            q=q.transpose(1, 2).contiguous(),
            k=k.transpose(1, 2).contiguous(),
            v=v.transpose(1, 2).contiguous(),
            g=log_gates,
            scale=self.head_dim**-0.5,
            o_norm=True,
            o_norm_weight=self.head_norm.weight.to(dtype=q.dtype),
            o_norm_eps=self.head_norm.eps,
        )
        out = out.contiguous().view(batch_size, seq_len, d_model)
        return self.resid_dropout(self.out_proj(out))

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        if self._select_backend(q) == "fused":
            return self._forward_fused(x, token_ids, q, k, v)
        return self._forward_reference(x, token_ids, q, k, v)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        attention_cls: type[nn.Module],
        norm_cls: type[nn.Module],
        ff_cls: type[nn.Module],
        use_rope: bool,
        dropout: float = 0.0,
        layer_idx: int = 0,
        vocab_size: int | None = None,
        fix_backend: str = "auto",
    ) -> None:
        super().__init__()
        hidden_dim = 4 * d_model
        self.norm_1 = norm_cls(d_model)
        attention_kwargs = {
            "use_rope": use_rope,
            "dropout": dropout,
            "layer_idx": layer_idx,
            "vocab_size": vocab_size,
        }
        if attention_cls is FiXSelfAttention:
            attention_kwargs["backend"] = fix_backend
        self.attn = attention_cls(
            d_model,
            n_heads,
            **attention_kwargs,
        )
        self.norm_2 = norm_cls(d_model)
        self.ff = ff_cls(d_model, hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), token_ids)
        x = x + self.ff(self.norm_2(x))
        return x


@dataclass(slots=True)
class ModelSpec:
    attention_cls: type[nn.Module]
    norm_cls: type[nn.Module]
    ff_cls: type[nn.Module]
    use_rope: bool
    learned_positions: bool


MODEL_SPECS = {
    "vanilla": ModelSpec(
        attention_cls=StandardSelfAttention,
        norm_cls=nn.LayerNorm,
        ff_cls=FeedForward,
        use_rope=False,
        learned_positions=True,
    ),
    "llama": ModelSpec(
        attention_cls=StandardSelfAttention,
        norm_cls=RMSNorm,
        ff_cls=SwiGLUFeedForward,
        use_rope=True,
        learned_positions=False,
    ),
    "differential": ModelSpec(
        attention_cls=DifferentialSelfAttention,
        norm_cls=RMSNorm,
        ff_cls=SwiGLUFeedForward,
        use_rope=True,
        learned_positions=False,
    ),
    "fix": ModelSpec(
        attention_cls=FiXSelfAttention,
        norm_cls=RMSNorm,
        ff_cls=SwiGLUFeedForward,
        use_rope=True,
        learned_positions=False,
    ),
}


class TransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        spec: ModelSpec,
        dropout: float = 0.0,
        fix_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.position_emb = nn.Embedding(max_seq_len, d_model) if spec.learned_positions else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    attention_cls=spec.attention_cls,
                    norm_cls=spec.norm_cls,
                    ff_cls=spec.ff_cls,
                    use_rope=spec.use_rope,
                    dropout=dropout,
                    layer_idx=layer_idx,
                    vocab_size=vocab_size,
                    fix_backend=fix_backend,
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.norm = spec.norm_cls(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            if isinstance(block.attn, FiXSelfAttention):
                block.attn.reset_gate_parameters()
        self.lm_head.weight = self.token_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")
        x = self.token_emb(token_ids)
        if self.position_emb is not None:
            positions = torch.arange(seq_len, device=token_ids.device)
            x = x + self.position_emb(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, token_ids)
        x = self.norm(x)
        return self.lm_head(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)


def build_model(
    *,
    name: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
    dropout: float = 0.0,
    fix_backend: str = "auto",
) -> nn.Module:
    try:
        spec = MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {name}") from exc
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        spec=spec,
        dropout=dropout,
        fix_backend=fix_backend,
    )
