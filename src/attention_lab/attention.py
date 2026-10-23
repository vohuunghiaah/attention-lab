"""Functional and module interfaces for single-head attention."""

import torch
from torch import nn

from attention_lab.torch_attention import scaled_dot_product_attention

__all__ = ["ScaledDotProductAttention", "scaled_dot_product_attention"]


class ScaledDotProductAttention(nn.Module):
    """Parameter-free attention module for inputs (..., T, features)."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention output and weights; see functional API for shapes."""
        return scaled_dot_product_attention(q, k, v, mask, causal=causal)
