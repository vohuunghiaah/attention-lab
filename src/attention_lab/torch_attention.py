"""Manual scaled dot-product attention with safe fully masked rows."""

import math

import torch

from attention_lab.masking import causal_mask


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return output (..., Tq, dv), weights (..., Tq, Tk).

    Inputs have matching leading dimensions; Q/K end in dk and K/V share Tk.
    All inputs must have the same floating dtype and device. A boolean mask
    broadcasts to scores, with True meaning allowed. Fully masked rows return
    zero output and weights, with finite gradients. No dropout is applied.
    """
    if any(x.ndim < 2 or 0 in x.shape for x in (q, k, v)):
        raise ValueError("Q, K, V require nonempty matrices")
    if any(not x.is_floating_point() for x in (q, k, v)):
        raise TypeError("Q, K, V must be floating tensors")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, v)):
        raise ValueError("Q, K, V must share dtype and device")
    if any(not torch.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("Q, K, V must be finite")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("Leading batch dimensions must match")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K feature dimensions and K/V lengths must match")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # (..., Tq, Tk)
    if not torch.isfinite(scores).all():
        raise ValueError("Attention scores overflowed; reduce input magnitudes")
    allowed = torch.ones_like(scores, dtype=torch.bool)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("Mask must be boolean (True = allowed)")
        if mask.device != q.device:
            raise ValueError("Mask must be on the input device")
        try:
            allowed = allowed & torch.broadcast_to(mask, scores.shape)
        except RuntimeError as exc:
            raise ValueError("Mask cannot broadcast to attention scores") from exc
    if causal:
        allowed = allowed & causal_mask(q.shape[-2], k.shape[-2], device=q.device)
    masked = scores.masked_fill(~allowed, -torch.inf)
    valid = allowed.any(dim=-1, keepdim=True)
    safe = torch.where(valid, masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=-1).masked_fill(~allowed, 0)
    return weights @ v, weights  # (..., Tq, dv), (..., Tq, Tk)
