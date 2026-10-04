"""Four-stage packed decoder numerics and the existing v6 persistence contract."""

import pytest
import torch

from litebsq.autoencoder import Decoder
import litebsq.fused_multistage_decoder as fused
from test_cat_packed_decoder import (
    test_cat_conversion_packed_roundtrip as _cat_conversion_packed_roundtrip,
)
from test_fused_multistage_decoder_parity import _native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("in_dim", [16, 24])
def test_four_stage_bf16_native_fused_and_group8_parity(in_dim):
    torch.manual_seed(31)
    decoder = Decoder(
        in_dim=in_dim, out_dim=32, hidden_dim=128, num_res_blocks=0,
        norm_type="layer", decoder_type="symmetric", use_checkpoint=False,
        num_models=4,
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    binary = torch.randint(0, 2, (256, 4, in_dim), device="cuda", dtype=torch.uint8)
    shifts = torch.arange(8, device="cuda", dtype=torch.uint8)
    packed = (binary.view(256, 4, in_dim // 8, 8) << shifts).sum(-1).to(torch.uint8)
    dense = binary.to(torch.bfloat16)
    parameters = tuple(decoder.parameters())
    torch.manual_seed(47)
    gradient = torch.randn(256, 32, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _native(decoder, packed)
        expected_grads = torch.autograd.grad(expected, parameters, gradient)
        actual = fused.fused_decode_packed_symmetric_decoder(decoder, dense)
        actual_grads = torch.autograd.grad(actual, parameters, gradient)

    tolerance = 2 * torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for (name, parameter), expected_grad, actual_grad in zip(
        decoder.named_parameters(), expected_grads, actual_grads,
    ):
        assert actual_grad.dtype == parameter.dtype, name
        relative = (
            (actual_grad.float() - expected_grad.float()).norm()
            / expected_grad.float().norm().clamp_min(1e-12)
        )
        assert relative <= tolerance, (name, float(relative))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _native(decoder, packed)
        actual = fused.fused_decode_packed_symmetric_decoder(decoder, dense)
        batched = fused.fused_decode_batched_same_shape_decoders(
            [decoder] * 8, [dense] * 8,
        )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert len(batched) == 8
    for item in batched:
        torch.testing.assert_close(item, expected, atol=tolerance, rtol=tolerance)


def test_four_stage_cat_packed_v6_roundtrip(tmp_path):
    # This fixture is FP32 with small geometry; GPU cases cover the BF16 recipe.
    _cat_conversion_packed_roundtrip(
        tmp_path, kind="final_model", decoder_type="symmetric", stages=4, parts=1,
    )
