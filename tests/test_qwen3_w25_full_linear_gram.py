import pytest
import torch
from torch.nn import functional as F

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.full_linear_gram import (
    FullLinearGram,
    build_full_linear_gram,
    full_linear_gram_mse,
)
from litebsq.protected_channel_quant import (
    decode_protected_channel_weight,
    encode_protected_channel_weight,
)


def _tolerances(dtype):
    return {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 2e-12, "atol": 2e-13}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_gram_loss_and_error_gradient_match_direct_output(dtype):
    generator = torch.Generator().manual_seed(3101)
    # Uneven token chunks and a noncontiguous input view exercise the same N.
    inputs = torch.randn(47, 34, generator=generator).bfloat16()[:, ::2]
    error = torch.randn(9, 17, generator=generator, dtype=dtype).requires_grad_()
    gram = build_full_linear_gram(inputs, dtype=dtype)
    actual = full_linear_gram_mse(error, gram)
    expected = F.linear(inputs.to(dtype), error).square().mean()
    actual_grad = torch.autograd.grad(actual, error)[0]
    expected_grad = torch.autograd.grad(expected, error)[0]
    torch.testing.assert_close(actual, expected, **_tolerances(dtype))
    torch.testing.assert_close(actual_grad, expected_grad, **_tolerances(dtype))
    assert gram.token_count == 47 and gram.matrix.shape == (17, 17)
    assert gram.matrix.dtype == dtype and actual.dtype == dtype
    if dtype == torch.float32:
        streamed = linear_output_mse(error, inputs, chunk_size=13, recompute=True)
        streamed_grad = torch.autograd.grad(streamed, error)[0]
        torch.testing.assert_close(actual, streamed, **_tolerances(dtype))
        torch.testing.assert_close(actual_grad, streamed_grad, **_tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_correlated_input_int8_protection_retains_trainable_cross_gradient(dtype):
    generator = torch.Generator().manual_seed(3102)
    protected_indices = torch.tensor([13, 1, 8, 5])
    keep = torch.ones(16, dtype=torch.bool)
    keep[protected_indices] = False
    unprotected_indices = torch.arange(16)[keep]
    teacher = torch.randn(8, 16, generator=generator)
    teacher[:, protected_indices] *= 10
    qvalues, scales = encode_protected_channel_weight(
        teacher[:, protected_indices].T, quant_format="int8")
    protected_weight = decode_protected_channel_weight(
        qvalues, scales, quant_format="int8", dtype=dtype).T
    before = (qvalues.clone(), scales.clone(), protected_indices.clone())
    inputs = torch.randn(61, 16, generator=generator).to(dtype)
    # Every protected column correlates with an unprotected column. Its small
    # quantization error must contribute to the latter's trainable gradient.
    inputs[:, protected_indices] = inputs[:, unprotected_indices[:4]] + 0.03 * inputs[:, protected_indices]
    trainable = (teacher[:, keep].to(dtype) + 0.02 * torch.randn(
        8, 12, generator=generator, dtype=dtype)).requires_grad_()
    full_weight = torch.zeros(8, 16, dtype=dtype).index_copy(
        1, protected_indices, protected_weight).index_copy(1, unprotected_indices, trainable)
    error = full_weight - teacher.to(dtype)
    assert torch.count_nonzero(error[:, protected_indices]) > 0
    gram = build_full_linear_gram(inputs, dtype=dtype)
    actual = full_linear_gram_mse(error, gram)
    expected = F.linear(inputs, error).square().mean()
    actual_grad = torch.autograd.grad(actual, trainable, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, trainable, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected, **_tolerances(dtype))
    torch.testing.assert_close(actual_grad, expected_grad, **_tolerances(dtype))
    cropped = F.linear(inputs[:, keep], error[:, keep]).square().mean()
    cropped_grad = torch.autograd.grad(cropped, trainable)[0]
    cross_gradient = 2 * error[:, protected_indices].detach() @ gram.matrix[
        protected_indices][:, unprotected_indices] / (len(inputs) * len(teacher))
    torch.testing.assert_close(actual_grad - cropped_grad, cross_gradient, **_tolerances(dtype))
    assert torch.linalg.vector_norm(cross_gradient) > 0.001
    assert not torch.allclose(actual_grad, cropped_grad, rtol=1e-3, atol=1e-5)
    for actual_tensor, original in zip((qvalues, scales, protected_indices), before):
        assert torch.equal(actual_tensor, original) and not actual_tensor.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gram_is_temporary_detached_and_explicit_dtype_ignores_autocast(dtype):
    generator = torch.Generator().manual_seed(3103)
    inputs = torch.randn(23, 7, generator=generator, requires_grad=True)
    # A float32 E leaf also exercises the differentiable cast in FP64 mode.
    error = torch.randn(5, 7, generator=generator, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        gram = build_full_linear_gram(inputs, dtype=dtype)
        loss = full_linear_gram_mse(error, gram)
    expected = F.linear(inputs.detach().to(dtype), error.to(dtype)).square().mean()
    expected_grad = torch.autograd.grad(expected, error)[0]
    loss.backward()
    assert inputs.grad is None
    assert not gram.matrix.requires_grad and gram.matrix.grad_fn is None
    assert gram.matrix.dtype == dtype and loss.dtype == dtype
    assert not isinstance(gram, torch.nn.Module) and not hasattr(gram, "state_dict")
    torch.testing.assert_close(loss, expected, **_tolerances(dtype))
    torch.testing.assert_close(error.grad, expected_grad, rtol=2e-5, atol=2e-6)


def test_invalid_inputs_and_nondetached_gram_fail_explicitly():
    with pytest.raises(ValueError, match="float32 or float64"):
        build_full_linear_gram(torch.ones(3, 4), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="nonempty inputs"):
        build_full_linear_gram(torch.empty(0, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="floating point"):
        build_full_linear_gram(torch.ones(3, 4, dtype=torch.long), dtype=torch.float32)
    gram = build_full_linear_gram(torch.ones(3, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="matching full square Gram"):
        full_linear_gram_mse(torch.ones(2, 3), gram)
    with pytest.raises(ValueError, match="positive integer"):
        full_linear_gram_mse(torch.ones(2, 4), FullLinearGram(gram.matrix, 0))
    with pytest.raises(ValueError, match="detached"):
        full_linear_gram_mse(torch.ones(2, 4), FullLinearGram(gram.matrix.clone().requires_grad_(), 3))
