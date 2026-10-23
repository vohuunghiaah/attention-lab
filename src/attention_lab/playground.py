"""Deterministic, untrained embeddings for the input playground."""

import hashlib

import numpy as np

from attention_lab.numpy_attention import Array


def token_projections(tokens: list[str], d_model: int, seed: int) -> tuple[Array, ...]:
    """Return X, Wq, Wk, Wv, Q, K, V for tokens (T,) and width D.

    SHA-256 seeds each token independently of Python's randomized hash.
    Repeated tokens have identical embeddings; no positional encoding is added.
    """
    if not tokens or len(tokens) > 24 or d_model <= 0:
        raise ValueError("Use 1–24 tokens and a positive embedding dimension")
    embeddings = []
    for token in tokens:
        digest = hashlib.sha256(f"{seed}:{token}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        embeddings.append(rng.standard_normal(d_model))
    x = np.stack(embeddings)  # (T,D)
    rng = np.random.default_rng(seed)
    matrices = tuple(rng.standard_normal((d_model, d_model)) / np.sqrt(d_model) for _ in range(3))
    return (x, *matrices, *(x @ matrix for matrix in matrices))
