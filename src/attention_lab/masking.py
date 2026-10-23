"""Boolean attention masks: True means the key is allowed."""

import torch


def causal_mask(
    seq_len_q: int, seq_len_k: int | None = None, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return (Tq, Tk) with j <= i; rectangular masks are upper-left aligned."""
    seq_len_k = seq_len_q if seq_len_k is None else seq_len_k
    if seq_len_q <= 0 or seq_len_k <= 0:
        raise ValueError("Sequence lengths must be positive")
    return (
        torch.arange(seq_len_k, device=device)[None, :]
        <= torch.arange(seq_len_q, device=device)[:, None]
    )


def padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Convert integer lengths (B,) to allowed-key mask (B, 1, 1, Tk)."""
    if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("Lengths must be a one-dimensional integer tensor")
    if max_length <= 0 or torch.any((lengths < 0) | (lengths > max_length)):
        raise ValueError("Lengths must be between zero and max_length")
    return (torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None])[
        :, None, None, :
    ]
