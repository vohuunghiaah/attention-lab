"""Educational NumPy and PyTorch attention engines."""

from attention_lab.attention import ScaledDotProductAttention
from attention_lab.multi_head_attention import MultiHeadAttention
from attention_lab.numpy_attention import mathematical_example, numpy_attention
from attention_lab.torch_attention import scaled_dot_product_attention

__all__ = [
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    "mathematical_example",
    "numpy_attention",
    "scaled_dot_product_attention",
]
