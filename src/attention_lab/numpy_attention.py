"""Visible, stable NumPy attention computations; masks use True for allowed keys."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


@dataclass(frozen=True)
class AttentionResult:
    """Scores/weights (..., Tq, Tk) and output (..., Tq, dv)."""

    raw_scores: Array
    scaled_scores: Array
    masked_scores: Array
    weights: Array
    output: Array


def stable_softmax(scores: Array) -> Array:
    """Normalize the last axis; all -inf rows return zeros, not NaNs."""
    scores = np.asarray(scores)
    if scores.ndim < 1 or scores.shape[-1] == 0:
        raise ValueError("Softmax requires a nonempty final dimension")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Scores may contain finite values or -inf only")
    maximum = np.max(scores, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    exponentials = np.exp(scores - maximum)
    denominator = exponentials.sum(axis=-1, keepdims=True)
    return np.divide(
        exponentials, denominator, out=np.zeros_like(exponentials), where=denominator != 0
    )


def numpy_attention(
    q: Array,
    k: Array,
    v: Array,
    mask: NDArray[np.bool_] | None = None,
    *,
    causal: bool = False,
    scale: bool = True,
) -> AttentionResult:
    """Compute attention for Q (..., Tq, dk), K (..., Tk, dk), V (..., Tk, dv).

    Leading dimensions must match exactly. Boolean masks broadcast to scores.
    Rectangular causal masks allow key index j <= query index i (upper-left).
    Fully masked rows produce zero weights and zero outputs.
    """
    q, k, v = (np.asarray(x) for x in (q, k, v))
    if any(x.ndim < 2 or 0 in x.shape for x in (q, k, v)):
        raise ValueError("Q, K, V require nonempty matrices")
    if any(not np.issubdtype(x.dtype, np.floating) for x in (q, k, v)):
        raise TypeError("Q, K, V must have floating dtype")
    if any(not np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("Q, K, V must be finite")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("Leading batch dimensions must match")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K feature dimensions and K/V lengths must match")
    raw = q @ k.swapaxes(-1, -2)  # (..., Tq, Tk)
    scaled = raw / np.sqrt(q.shape[-1]) if scale else raw.copy()
    if not np.isfinite(scaled).all():
        raise ValueError("Attention scores overflowed; reduce input magnitudes")
    allowed = np.ones(raw.shape, dtype=bool)
    if mask is not None:
        if np.asarray(mask).dtype != np.bool_:
            raise TypeError("Mask must be boolean (True = allowed)")
        try:
            allowed &= np.broadcast_to(mask, raw.shape)
        except ValueError as exc:
            raise ValueError("Mask cannot broadcast to attention scores") from exc
    if causal:
        allowed &= np.arange(k.shape[-2])[None, :] <= np.arange(q.shape[-2])[:, None]
    masked = np.where(allowed, scaled, -np.inf)
    weights = stable_softmax(masked)
    return AttentionResult(raw, scaled, masked, weights, weights @ v)


def mathematical_example() -> tuple[Array, Array, Array]:
    """Return fresh Q, K, V matrices, each (3, 2), for the hand-worked example."""
    return (
        np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 2.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
