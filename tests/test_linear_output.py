import torch
import torch.nn.functional as F

from experiments.linear_output.config import stage_steps
from experiments.linear_output.objectives import linear_output_mse
from experiments.linear_output.output_kernel import output_error_reference, output_mse


def test_linear_output_mse_matches_dense_reference_and_gradient():
    torch.manual_seed(31)
    delta = torch.randn(5, 11, requires_grad=True)
    inputs = torch.randn(23, 11)
    reference = F.linear(inputs, delta).square().mean()

    actual = linear_output_mse(delta, inputs, chunk_size=4, recompute=False)
    torch.testing.assert_close(actual, reference)

    actual.backward()
    grad_delta = delta.grad.detach().clone()

    delta_ref = delta.detach().clone().requires_grad_()
    F.linear(inputs, delta_ref).square().mean().backward()
    torch.testing.assert_close(grad_delta, delta_ref.grad)


def test_linear_output_mse_checkpoint_chunking_is_exact():
    torch.manual_seed(7)
    delta = torch.randn(3, 9, requires_grad=True)
    inputs = torch.randn(17, 9)
    reference = F.linear(inputs, delta).square().mean()
    actual = linear_output_mse(delta, inputs, chunk_size=3, recompute=True)
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert torch.isfinite(delta.grad).all()


def test_stage_steps_preserves_total_and_minimum():
    assert stage_steps(5000, 2) == [2500, 2500]
    assert stage_steps(7, 3) == [3, 2, 2]


def test_full_block_output_kernel_matches_reference_and_packed_layout():
    torch.manual_seed(11)
    decoded = torch.randn(4, 3, 32, requires_grad=True)
    target = torch.randn(4, 3, 32)
    inputs = torch.randn(13, 96)
    reference = output_error_reference(decoded, target, inputs).square().mean()
    actual = output_mse(decoded, target, inputs, use_triton=False)
    torch.testing.assert_close(actual, reference)
    packed = decoded.detach().reshape(-1, 1, 32).requires_grad_()
    packed_target = target.reshape(-1, 1, 32)
    packed_actual = output_mse(packed, packed_target, inputs, use_triton=False)
    torch.testing.assert_close(packed_actual, reference.detach())
    actual.backward()
    assert torch.isfinite(decoded.grad).all()
