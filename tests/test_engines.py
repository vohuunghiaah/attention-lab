"""Independent mathematical expectations and reference implementation checks."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from attention_lab import MultiHeadAttention, numpy_attention, scaled_dot_product_attention
from attention_lab.masking import causal_mask, padding_mask
from attention_lab.numpy_attention import mathematical_example, stable_softmax


def test_known_example() -> None:
    q, k, v = mathematical_example()
    result = numpy_attention(q, k, v)
    raw = np.array([[1, 1, 2], [0, 1, 1], [1, 2, 3]])
    np.testing.assert_array_equal(result.raw_scores, raw)
    np.testing.assert_allclose(result.scaled_scores, raw / np.sqrt(2))
    expected = np.exp(raw / np.sqrt(2))
    expected /= expected.sum(-1, keepdims=True)
    np.testing.assert_allclose(result.weights, expected)
    np.testing.assert_allclose(result.output, expected @ v)
    assert (result.weights > 0).all()  # noncausal permits future positions


def test_stable_softmax() -> None:
    np.testing.assert_allclose(stable_softmax(np.array([[10000.0, 10000.0]])), [[0.5, 0.5]])
    np.testing.assert_array_equal(stable_softmax(np.array([[-np.inf, -np.inf]])), [[0, 0]])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backends_and_reference(causal: bool, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(17)
    q, k, v = [
        torch.randn(s, generator=generator, dtype=dtype)
        for s in [(2, 3, 4, 5), (2, 3, 6, 5), (2, 3, 6, 7)]
    ]
    output, weights = scaled_dot_product_attention(q, k, v, causal=causal)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(output, reference)
    result = numpy_attention(q.numpy(), k.numpy(), v.numpy(), causal=causal)
    np.testing.assert_allclose(output.numpy(), result.output, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(weights.numpy(), result.weights, atol=2e-6, rtol=2e-5)
    assert output.shape == (2, 3, 4, 7)
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    if causal:
        assert torch.count_nonzero(weights[..., ~causal_mask(4, 6)]) == 0


def test_mask_broadcast_fully_masked_and_gradients() -> None:
    q = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    mask = padding_mask(torch.tensor([0, 2]), 3)
    output, weights = scaled_dot_product_attention(q, k, v, mask)
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(weights[..., 2]) == 0
    output.square().sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad[1].abs().sum() > 0
    result = numpy_attention(
        q.detach().numpy(), k.detach().numpy(), v.detach().numpy(), mask.numpy()
    )
    np.testing.assert_allclose(result.output, output.detach().numpy(), atol=1e-12)


def test_gradcheck() -> None:
    inputs = tuple(torch.randn(1, 2, 2, dtype=torch.double, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(
        lambda q, k, v: scaled_dot_product_attention(q, k, v)[0], inputs
    )


@pytest.mark.parametrize("bad", ["shape", "batch", "empty", "integer", "nan", "mask"])
def test_invalid_inputs(bad: str) -> None:
    q, k, v = mathematical_example()
    mask = None
    if bad == "shape":
        k = k[:, :1]
    elif bad == "batch":
        q = q[None]
    elif bad == "empty":
        q = q[:0]
    elif bad == "integer":
        q = q.astype(int)
    elif bad == "nan":
        q[0, 0] = np.nan
    else:
        mask = np.ones((4, 4), dtype=bool)
    with pytest.raises((ValueError, TypeError)):
        numpy_attention(q, k, v, mask)
    with pytest.raises((ValueError, TypeError)):
        scaled_dot_product_attention(
            *(torch.from_numpy(x) for x in (q, k, v)),
            None if mask is None else torch.from_numpy(mask),
        )


def test_mask_and_dtype_validation() -> None:
    q = torch.ones(2, 3)
    with pytest.raises(ValueError, match="dtype"):
        scaled_dot_product_attention(q, q.double(), q)
    with pytest.raises(TypeError, match="boolean"):
        scaled_dot_product_attention(q, q, q, torch.ones(2, 2))
    with pytest.raises(TypeError, match="boolean"):
        numpy_attention(q.numpy(), q.numpy(), q.numpy(), np.ones((2, 2)))
    torch.testing.assert_close(
        causal_mask(3),
        torch.tensor([[True, False, False], [True, True, False], [True, True, True]]),
    )
    with pytest.raises(ValueError):
        causal_mask(0)
    with pytest.raises(ValueError):
        padding_mask(torch.tensor([4]), 3)


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_multihead_against_torch(heads: int) -> None:
    model = MultiHeadAttention(8, heads, seed=42).double()
    reference = torch.nn.MultiheadAttention(8, heads, bias=False, batch_first=True).double()
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat([model.q_proj.weight, model.k_proj.weight, model.v_proj.weight])
        )
        reference.out_proj.weight.copy_(model.out_proj.weight)
    x = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)
    details = model.inspect(x, causal=True)
    expected, weights = reference(x, x, x, attn_mask=~causal_mask(5), average_attn_weights=False)
    torch.testing.assert_close(details.output, expected)
    torch.testing.assert_close(details.weights, weights)
    assert details.head_outputs.shape == (2, heads, 5, 8 // heads)
    assert details.concatenated.shape == (2, 5, 8)
    details.output.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_multihead_reproducibility_and_errors() -> None:
    state = torch.random.get_rng_state()
    a, b = MultiHeadAttention(8, 2, seed=7), MultiHeadAttention(8, 2, seed=7)
    assert torch.equal(state, torch.random.get_rng_state())
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q)
    with pytest.raises(ValueError):
        MultiHeadAttention(7, 2)
    with pytest.raises(ValueError):
        a(torch.ones(3, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; CPU is default")
def test_cuda() -> None:
    q = torch.randn(1, 3, 4, device="cuda", requires_grad=True)
    output, _ = scaled_dot_product_attention(q, q, q, causal=True)
    output.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
