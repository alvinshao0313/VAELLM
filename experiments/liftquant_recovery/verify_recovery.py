"""Numerical contract for scaled, saturated STE and native packed arithmetic."""
import torch

from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
from experiments.liftquant_recovery.all_bits import _AllBitsLinear
from experiments.liftquant_recovery.proxy_coordinates import dense_ste, hard_bits


def check_ste(device):
    torch.manual_seed(7)
    shape, scale = (32, 1, 64), 0.004
    scores = (torch.randint(0, 2, shape, device=device).float()-0.5)*scale
    # Cover both saturation edges and the round-to-even bit threshold.
    scores[0, 0, :7] = torch.tensor([-1.1, -1., -.5, 0., .5, .999, 1.], device=device)*scale
    scores.requires_grad_()
    bits = hard_bits(scores.detach(), scale)
    packed = pack_bool_tensor_to_uint8(bits, logical_shape=shape)
    weight = torch.randn(1, 128, 64, device=device, requires_grad=True)
    bias = torch.randn(1, 128, device=device, requires_grad=True)
    gradient = torch.randn(32, 1, 128, device=device, dtype=torch.bfloat16)
    output = _AllBitsLinear.apply(packed, weight, bias, scores, torch.bfloat16, scale)
    output.backward(gradient)
    ref_scores = scores.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    with torch.autocast('cuda', enabled=False):
        codes = dense_ste(ref_scores, scale)
        rounded_w = ref_weight + (ref_weight.to(torch.bfloat16).float()-ref_weight).detach()
        reference = (torch.bmm(codes.transpose(0, 1), rounded_w.transpose(1, 2)).transpose(0, 1)
                     + ref_bias.unsqueeze(0)).to(torch.bfloat16)
    reference.backward(gradient)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(scores.grad, ref_scores.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(bias.grad, ref_bias.grad, rtol=0, atol=0)
    edited = scores.detach().clone()
    edited[2, 0, 2] = -edited[2, 0, 2]
    roundtrip = unpack_uint8_tensor_to_bool(
        pack_bool_tensor_to_uint8(hard_bits(edited, scale), logical_shape=shape), logical_shape=shape)
    assert (roundtrip != bits).sum().item() == 1
    assert scores.grad.dtype == torch.float32
    return dict(status='PASS', scaled_ste='independent dense autograd, including 1/s and saturation',
                packed_projection_boundary_crossing=True)
