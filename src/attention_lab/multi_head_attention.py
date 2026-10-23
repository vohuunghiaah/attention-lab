"""Multi-head self-attention with inspectable intermediate computations."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from attention_lab.torch_attention import scaled_dot_product_attention


@dataclass(frozen=True)
class MultiHeadResult:
    """Q/K/V and head outputs (B,H,T,Dh), weights (B,H,T,T), outputs (B,T,D)."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    weights: torch.Tensor
    head_outputs: torch.Tensor
    concatenated: torch.Tensor
    output: torch.Tensor


class MultiHeadAttention(nn.Module):
    """Bias-free Q/K/V and output projections, following the standard equations."""

    def __init__(self, d_model: int, num_heads: int, *, seed: int | None = None) -> None:
        """Initialize equal-sized heads; optional seed preserves the global RNG."""
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.d_model, self.num_heads = d_model, num_heads
        self.d_head = d_model // num_heads
        with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext():
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def inspect(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, *, causal: bool = False
    ) -> MultiHeadResult:
        """Compute all stages from input (B,T,D); mask broadcasts to (B,H,T,T)."""
        if x.ndim != 3 or x.shape[-1] != self.d_model or 0 in x.shape:
            raise ValueError("Input must have nonempty shape (batch, tokens, d_model)")
        if x.dtype != self.q_proj.weight.dtype or x.device != self.q_proj.weight.device:
            raise ValueError("Input dtype/device must match module parameters")
        batch_size, seq_len, _ = x.shape

        def split(projection: nn.Linear) -> torch.Tensor:
            """Project (B,T,D), reshape, transpose to (B,H,T,Dh)."""
            return (
                projection(x)
                .reshape(batch_size, seq_len, self.num_heads, self.d_head)
                .transpose(1, 2)
            )

        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        heads, weights = scaled_dot_product_attention(q, k, v, mask, causal=causal)
        concatenated = (
            heads.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        )  # (B,T,D)
        return MultiHeadResult(q, k, v, weights, heads, concatenated, self.out_proj(concatenated))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, *, causal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected output (B,T,D) and per-head weights (B,H,T,T)."""
        result = self.inspect(x, mask, causal=causal)
        return result.output, result.weights
