"""Compare fused execution against the actual packed decoder operators."""
import pytest
import torch
from torch.nn import functional as F

from litebsq.autoencoder import Decoder
from litebsq.packed_bit_linear import packed_u8_parallel_linear
import litebsq.fused_multistage_decoder as fused

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

def _case(in_dim, storage_dtype, *, hidden_dim=128, out_dim=32):
    torch.manual_seed(31)
    decoder = Decoder(
        in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, num_res_blocks=0,
        norm_type="layer", decoder_type="symmetric", use_checkpoint=False,
        num_models=2,
    ).to(device="cuda", dtype=storage_dtype).eval()
    binary = torch.randint(0, 2, (256, 2, in_dim), device="cuda", dtype=torch.uint8)
    shifts = torch.arange(8, device="cuda", dtype=torch.uint8)
    packed = (binary.view(256, 2, in_dim // 8, 8) << shifts).sum(-1).to(torch.uint8)
    return decoder, binary.to(torch.bfloat16), packed

def _native(decoder, packed):
    hidden = packed_u8_parallel_linear(
        packed, decoder.linear_in, logical_in_dim=decoder.in_dim,
        activation_dtype=torch.bfloat16,
    )
    output = decoder.linear_out(F.silu(decoder.norm_out(hidden)))
    # VAELinear explicitly casts after summing residual stages.
    return output.sum(dim=1).to(torch.bfloat16)

def _assert_packed_parity(in_dim, storage_dtype, *, hidden_dim=128, out_dim=32, group_size=2):
    decoder, dense, packed = _case(in_dim, storage_dtype, hidden_dim=hidden_dim, out_dim=out_dim)
    parameters = tuple(decoder.parameters())
    torch.manual_seed(47)
    gradient = torch.randn(256, out_dim, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _native(decoder, packed)
        expected_grads = torch.autograd.grad(expected, parameters, gradient)
        actual = fused.fused_decode_packed_symmetric_decoder(decoder, dense)
        actual_grads = torch.autograd.grad(actual, parameters, gradient)
    epsilon = torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(actual, expected, atol=2 * epsilon, rtol=2 * epsilon)
    for (name, parameter), expected_grad, actual_grad in zip(
        decoder.named_parameters(), expected_grads, actual_grads,
    ):
        assert actual_grad.dtype == parameter.dtype, name
        relative = (actual_grad.float() - expected_grad.float()).norm() / expected_grad.float().norm().clamp_min(1e-12)
        # Packed first-linear weights accumulate gradients in their FP32 storage
        # precision; autocast must not silently round them through BF16.
        limit = 1e-5 if storage_dtype == torch.float32 and name == "linear_in.conv.weight" else 2 * epsilon
        assert relative <= limit, (name, float(relative))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _native(decoder, packed)
        actual = fused.fused_decode_packed_symmetric_decoder(decoder, dense)
        batched = fused.fused_decode_batched_same_shape_decoders(
            [decoder] * group_size, [dense] * group_size,
        )
    torch.testing.assert_close(actual, expected, atol=2 * epsilon, rtol=2 * epsilon)
    assert len(batched) == group_size
    for item in batched:
        torch.testing.assert_close(item, expected, atol=2 * epsilon, rtol=2 * epsilon)


@pytest.mark.parametrize("in_dim", [32, 48, 64])
@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_fused_matches_actual_packed_decoder_and_fp32_gradients(in_dim, storage_dtype):
    _assert_packed_parity(in_dim, storage_dtype)


def test_h256_k48_fp32_bf16_native_and_group8_keep_gradients():
    # M128 inference/group8 must launch on A800 without changing tile/stage settings;
    # keep first-linear FP32 gradients while the launch uses BF16 matmul operands.
    _assert_packed_parity(48, torch.float32, hidden_dim=256, group_size=8)


@pytest.mark.parametrize("in_dim", [16, 24])
def test_d16_h128_bf16_native_and_group8_keep_gradients(in_dim):
    # K24 packs each stage into three bytes; OUT16 exercises the smaller output tile.
    _assert_packed_parity(in_dim, torch.bfloat16, out_dim=16, group_size=8)
