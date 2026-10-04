from copy import deepcopy
from io import BytesIO

import pytest
import torch
from torch.nn import functional as F

from experiments.qwen3_w25.local_bits import _changed_bits, calibrate_local_bits_cpu, optimize_local_bits
from litebsq.autoencoder import Decoder
from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real SparseBit CUDA/Triton path required")


def _fixture(device="cpu", dtype=torch.bfloat16, transpose=False, protected=False):
    decoders = []
    for gain in (0.5, 0.125):
        decoder = Decoder(in_dim=9, out_dim=4, hidden_dim=16, num_res_blocks=0,
                          norm_type="layer", decoder_type="linear")
        with torch.no_grad():
            decoder.linear.linear.weight.zero_()
            decoder.linear.linear.weight[:, :4].copy_(torch.eye(4) * gain)
            decoder.linear.linear.bias.zero_()
        decoders.append(decoder.to(dtype))
    protection = {}
    if protected:
        source = torch.linspace(-0.7, 0.9, 32).reshape(4, 8)
        qvalues, scales = encode_protected_channel_weight(source, quant_format="int8")
        protection = dict(compressed_in_features=12, protected_input_indices=torch.tensor([13, 1, 8, 5]),
                          protected_input_qvalues=qvalues, protected_input_scales=scales,
                          protected_channel_quant_format="int8")
    module = VAELinear(
        in_features=16, out_features=8, bias=None, codebook_dim=4, transpose=transpose,
        stage_vq_weights=[torch.zeros(24 if protected else 32, 1, 9, dtype=torch.bool) for _ in range(2)],
        stage_decoders=decoders,
        weight_rotation_spec=None if protected else rotation_spec("two_sided", 0, 31, "local_bits_test"),
        **protection,
    ).to(device)
    module.pack_parallel_stage_decoder_(trainable=False)
    module.eval()
    teacher = deepcopy(module)
    with torch.no_grad():
        for stage in range(2):
            spec = teacher.get_stage_part_vq_spec(stage, 0)
            logical = torch.ones(spec["logical_shape"], dtype=torch.bool, device=device)
            teacher.get_stage_part_vq_storage(stage, 0).copy_(
                pack_bool_tensor_to_uint8(logical, logical_shape=spec["logical_shape"]))
        teacher._build_parallel_stage_decode_plan()
        reference = teacher._decode_weight(dtype=torch.float32).detach()
    train_x = torch.eye(16, device=device, dtype=dtype).repeat(2, 1)
    heldout_x = torch.randn(23, 16, generator=torch.Generator().manual_seed(317)).to(device, dtype)
    return module, reference, train_x, heldout_x


def _snapshot(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _assert_restored(module, before, *, training=False):
    assert module.training == training
    assert not module.trainable_decode and module.cache_decoded_weight and module.parallel_stage_decode
    assert not hasattr(module, "_sparse_bit_binding")
    assert module._cached_weight is None
    assert all(parameter.grad is None for parameter in module.parameters())
    for name, tensor in module.state_dict().items():
        assert tensor.dtype == before[name].dtype and torch.equal(tensor, before[name]), name


def test_invalid_configuration_and_cpu_rejected_without_mutation():
    module, reference, train_x, heldout_x = _fixture()
    before = _snapshot(module)
    for kwargs, expected in (({"round_steps": 0}, "round_steps"),
                             ({"round_steps": 4, "optimizer": "unknown"}, "bit_optimizer"),
                             ({"round_steps": 4}, "CUDA")):
        with pytest.raises((ValueError, RuntimeError), match=expected):
            optimize_local_bits(module, reference, train_x, heldout_x, steps=4, lr=0.25, **kwargs)
        _assert_restored(module, before)


def test_cpu_scale_estimate_uses_real_decoder_and_preserves_state_rng():
    module, _, _, _ = _fixture()
    before = _snapshot(module)
    rng = torch.get_rng_state().clone()
    result = calibrate_local_bits_cpu(module)
    assert [stage["scale"] for stage in result["stages"]] == pytest.approx([0.5 / 3, 0.125 / 3])
    assert all(stage["rows"] == 32 for stage in result["stages"])
    assert torch.equal(rng, torch.get_rng_state())
    _assert_restored(module, before)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype,transpose,protected", [
    (torch.bfloat16, True, False), (torch.float32, False, False),
    (torch.bfloat16, True, True), (torch.bfloat16, False, True),
], ids=["rotation-bf16-transpose", "rotation-fp32", "int8-bf16-transpose", "int8-bf16"])
def test_real_two_stage_bits_change_decoder_frozen_and_native_forward(dtype, transpose, protected):
    module, reference, train_x, heldout_x = _fixture("cuda:0", dtype, transpose, protected)
    before = _snapshot(module)
    ids = {name: id(parameter) for name, parameter in module.named_parameters()}
    flags = {name: parameter.requires_grad for name, parameter in module.named_parameters()}
    result = optimize_local_bits(module, reference, train_x, heldout_x,
                                 steps=10, lr=0.025, round_steps=6, chunk_size=7)
    assert result["after_native"]["heldout_mse"] < result["before_native"]["heldout_mse"]
    evidence = result["evidence"]
    assert all(value > 0 for value in evidence["stage_changed_bits"])
    assert all(value > 0 for value in evidence["stage_gradient_l2_max"])
    assert evidence["rounds_ended"] >= 1
    assert evidence["proxy_parameter_dtypes"] == evidence["proxy_gradient_dtypes"] == ["torch.float32"]
    assert evidence["optimizer_moment_dtypes"] == ["torch.float32"]
    assert evidence["coordinate_scales_fixed"]
    assert [stage["scale"] for stage in evidence["stage_coordinates"]] == pytest.approx([0.5 / 3, 0.125 / 3])
    assert all(stage["lr_over_radius"] == pytest.approx(0.025 / stage["radius"])
               for stage in evidence["stage_coordinates"])
    assert all(stage["nonzero_fraction"] > 0 for stage in evidence["first_step_gradients"])
    assert evidence["frozen_sha256_before"] == evidence["frozen_sha256_after"]
    assert evidence["persistent_storage_bits_before"] == evidence["persistent_storage_bits_after"]
    assert ids == {name: id(parameter) for name, parameter in module.named_parameters()}
    assert flags == {name: parameter.requires_grad for name, parameter in module.named_parameters()}
    assert not hasattr(module, "_sparse_bit_binding") and not hasattr(module, "_decoder_compute_dtype")
    assert not module.training and not module.trainable_decode and module.cache_decoded_weight
    assert module._cached_weight is None
    for name, tensor in module.state_dict().items():
        if name not in evidence["code_state_keys"]:
            assert tensor.dtype == before[name].dtype and torch.equal(tensor, before[name]), name
    with torch.no_grad():
        weight = module._decode_weight(dtype=dtype)
        expected_mse = F.linear(heldout_x.float(), weight.float() - reference).square().mean()
        assert result["after_native"]["heldout_mse"] == pytest.approx(float(expected_mse), rel=2e-6)
        torch.testing.assert_close(module(heldout_x), F.linear(heldout_x, weight), atol=0, rtol=0)
    saved = BytesIO()
    torch.save(module.state_dict(), saved)
    saved.seek(0)
    restored, _, _, _ = _fixture("cuda:0", dtype, transpose, protected)
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    restored._build_parallel_stage_decode_plan()
    restored.clear_decoded_weight_cache()
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored(heldout_x), module(heldout_x), rtol=0, atol=0)


@CUDA_REQUIRED
def test_real_flip_then_exception_rolls_back_and_noop_is_rejected():
    module, reference, train_x, heldout_x = _fixture("cuda:0")
    before = _snapshot(module)
    module.train()
    module._decoder_compute_dtype = torch.bfloat16

    def cancel_after_flip(report):
        assert report["step_flip_count"] > 0
        raise RuntimeError("caller cancellation after actual packed update")

    with pytest.raises(RuntimeError, match="caller cancellation"):
        optimize_local_bits(module, reference, train_x, heldout_x, steps=3, lr=2.0,
                            round_steps=2, progress_callback=cancel_after_flip)
    _assert_restored(module, before, training=True)
    assert module._decoder_compute_dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="No permanent packed code changed"):
        optimize_local_bits(module, reference, train_x, heldout_x, steps=1, lr=1e-9, round_steps=2)
    _assert_restored(module, before, training=True)


@pytest.mark.parametrize("transpose", [False, True])
def test_cpu_int8_protected_packed_flip_and_serialized_scatter(transpose):
    module, _, _, _ = _fixture(transpose=transpose, protected=True)
    before = _snapshot(module)
    calibration = calibrate_local_bits_cpu(module)
    assert [stage["scale"] for stage in calibration["stages"]] == pytest.approx([0.5 / 3, 0.125 / 3])
    assert all(stage["rows"] == 24 for stage in calibration["stages"])
    _assert_restored(module, before)
    with torch.no_grad():
        initial_weight = module._decode_weight(dtype=torch.float32).clone()
        for stage in range(2):
            storage = module.get_stage_part_vq_storage(stage, 0)
            logical_shape = module.get_stage_part_vq_spec(stage, 0)["logical_shape"]
            logical = unpack_uint8_tensor_to_bool(storage, logical_shape=logical_shape)
            logical[5, 0, 2] = ~logical[5, 0, 2]
            prior = storage.clone()
            storage.copy_(pack_bool_tensor_to_uint8(logical, logical_shape=logical_shape))
            assert _changed_bits(prior, storage, logical_shape) == 1
        module._build_parallel_stage_decode_plan()
        module.clear_decoded_weight_cache()
        changed_weight = module._decode_weight(dtype=torch.float32)
        torch.testing.assert_close(changed_weight[:, module.protected_input_indices],
                                   initial_weight[:, module.protected_input_indices], rtol=0, atol=0)
        assert torch.count_nonzero(changed_weight - initial_weight) == 1
        expected_row, expected_column = (6, 3) if transpose else (1, 14)
        assert float((changed_weight - initial_weight)[expected_row, expected_column]) == 0.625
    for name, tensor in module.state_dict().items():
        if name not in ("vq_weight", "vq_weight_s1"):
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    saved = BytesIO()
    torch.save(module.state_dict(), saved)
    saved.seek(0)
    restored, _, _, _ = _fixture(transpose=transpose, protected=True)
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    restored._build_parallel_stage_decode_plan()
    restored.clear_decoded_weight_cache()
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored._decode_weight(dtype=torch.float32), changed_weight, rtol=0, atol=0)
