from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.local_decoder import _validate_payload_profile, optimize_local_decoder
from litebsq.autoencoder import Decoder
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec


def _fixture(dtype=torch.float32, transpose=False, protected=False):
    torch.manual_seed(3107)
    decoders = [Decoder(in_dim=8, out_dim=4, hidden_dim=8, num_res_blocks=1,
                        norm_type="layer", decoder_type="asymmetric").to(dtype) for _ in range(2)]
    protection = {}
    if protected:
        qvalues, scales = encode_protected_channel_weight(torch.randn(4, 8), quant_format="int8")
        protection = dict(compressed_in_features=12, protected_input_indices=torch.tensor([13, 1, 8, 5]),
                          protected_input_qvalues=qvalues, protected_input_scales=scales,
                          protected_channel_quant_format="int8")
    module = VAELinear(
        in_features=16, out_features=8, bias=nn.Parameter(torch.randn(8, dtype=dtype)),
        codebook_dim=4, transpose=transpose,
        stage_vq_weights=[torch.randint(0, 2, (24 if protected else 32, 1, 8), dtype=torch.bool) for _ in range(2)],
        stage_decoders=decoders,
        weight_rotation_spec=None if protected else rotation_spec("two_sided", 0, 31, "proj"),
        **protection,
    )
    module.pack_parallel_stage_decoder_(trainable=False)
    module.eval()
    teacher = deepcopy(module)
    with torch.no_grad():
        teacher._parallel_stage_decoder.linear_out.conv.bias.add_(0.08)
        reference = teacher._decode_weight(dtype=torch.float32).detach()
        if protected:
            reference[:, module.protected_input_indices] += 0.015
    train_x, heldout_x = torch.randn(47, 16), torch.randn(29, 16)
    # Correlated channels exercise the full output objective's cross terms.
    train_x[:, 1] = train_x[:, 0] + 0.1 * train_x[:, 1]
    heldout_x[:, 1] = heldout_x[:, 0] + 0.1 * heldout_x[:, 1]
    return module, reference, train_x, heldout_x


@pytest.mark.parametrize("dtype,activation_dtype,transpose", [
    (torch.float32, torch.float32, False),
    (torch.bfloat16, torch.bfloat16, True),
    (torch.float32, torch.bfloat16, False),
])
def test_real_packed_two_stage_rotation_update_and_cast(dtype, activation_dtype, transpose):
    module, reference, train_x, heldout_x = _fixture(dtype, transpose)
    train_x, heldout_x = train_x.to(activation_dtype), heldout_x.to(activation_dtype)
    before = {name: t.clone() for name, t in module.state_dict().items()}
    ids = {name: id(p) for name, p in module.named_parameters()}
    grad_flags = {name: p.requires_grad for name, p in module.named_parameters()}
    module.prime_decoded_weight_cache(dtype=dtype)
    assert module._cached_weight is not None
    result = optimize_local_decoder(module, reference, train_x, heldout_x,
                                    steps=24, lr=0.005, chunk_size=13)
    assert result["settings"]["native_decode_dtype"] == str(activation_dtype)
    assert result["after_fp32"]["train_mse"] < result["before_fp32"]["train_mse"]
    assert result["after_cast"]["heldout_mse"] < result["before_native"]["heldout_mse"]
    evidence = result["evidence"]
    assert all(v > 0 for v in evidence["stage_gradient_l2_max"])
    assert all(v > 0 for v in evidence["stage_parameter_delta_l2_saved"])
    assert evidence["optimizer_moment_dtypes"] == ["torch.float32"]
    assert evidence["frozen_sha256_before"] == evidence["frozen_sha256_after"]
    assert evidence["persistent_storage_bits_before"] == evidence["persistent_storage_bits_after"]
    assert ids == {name: id(p) for name, p in module.named_parameters()}
    assert grad_flags == {name: p.requires_grad for name, p in module.named_parameters()}
    assert all(p.dtype == dtype and p.grad is None for p in module.parameters())
    assert not module.training and not module.trainable_decode and module.cache_decoded_weight
    assert module.parallel_stage_decode and module._cached_weight is None
    assert not hasattr(module, "_decoder_compute_dtype")
    for name, tensor in module.state_dict().items():
        if not name.startswith("_parallel_stage_decoder."):
            assert torch.equal(tensor, before[name]), name
    with torch.no_grad():
        weight = module._decode_weight(dtype=activation_dtype)
        delta_outputs = F.linear(heldout_x.float(), weight.float() - reference)
        assert result["after_cast"]["heldout_mse"] == pytest.approx(float(delta_outputs.square().mean()), rel=2e-6)
        expected = F.linear(heldout_x, weight, module.bias.to(activation_dtype))
        torch.testing.assert_close(module(heldout_x), expected, atol=0, rtol=0)


def test_exception_rolls_back_values_dtype_runtime_and_preserves_other_modules():
    module, reference, train_x, heldout_x = _fixture(torch.bfloat16)
    outside = nn.Linear(16, 8)
    parent = nn.ModuleList([module, outside])
    before = {name: t.clone() for name, t in parent.state_dict().items()}
    module._decoder_compute_dtype = torch.bfloat16
    module.train()

    def fail_after_update(_):
        raise RuntimeError("intentional caller cancellation")

    with pytest.raises(RuntimeError, match="caller cancellation"):
        optimize_local_decoder(module, reference, train_x, heldout_x, steps=3,
                               lr=0.005, progress_callback=fail_after_update)
    assert module.training and not module.trainable_decode and module.cache_decoded_weight
    assert module._decoder_compute_dtype == torch.bfloat16
    assert module._cached_weight is None
    for name, tensor in parent.state_dict().items():
        assert tensor.dtype == before[name].dtype and torch.equal(tensor, before[name]), name


def test_unpacked_or_lora_modules_rejected_before_mutation():
    module, reference, train_x, heldout_x = _fixture()
    module.low_rank_a = nn.Parameter(torch.ones(8, 2))
    module.low_rank_b = nn.Parameter(torch.ones(2, 16))
    with pytest.raises(ValueError, match="LoRA"):
        optimize_local_decoder(module, reference, train_x, heldout_x, steps=1, lr=0.01)
    module.low_rank_a = module.low_rank_b = None
    module.parallel_stage_decode = False
    with pytest.raises(ValueError, match="packed"):
        optimize_local_decoder(module, reference, train_x, heldout_x, steps=1, lr=0.01)


def test_mixed_activation_dtypes_rejected():
    module, reference, train_x, heldout_x = _fixture()
    with pytest.raises(ValueError, match="same activation dtype"):
        optimize_local_decoder(module, reference, train_x, heldout_x.bfloat16(), steps=1, lr=0.01)


def test_zero_gradient_is_explicit_failure_and_rolls_back():
    module, reference, train_x, heldout_x = _fixture(torch.bfloat16)
    before = {name: t.clone() for name, t in module.state_dict().items()}
    with pytest.raises(RuntimeError, match="Both packed stages"):
        optimize_local_decoder(module, reference, torch.zeros_like(train_x), heldout_x,
                               steps=1, lr=0.01)
    for name, tensor in module.state_dict().items():
        assert tensor.dtype == before[name].dtype and torch.equal(tensor, before[name]), name


@pytest.mark.parametrize("transpose", [False, True])
def test_int8_protection_full_output_gradient_update_and_frozen_payload(transpose):
    module, reference, train_x, heldout_x = _fixture(torch.bfloat16, transpose, protected=True)
    before = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    parameters = list(module._parallel_stage_decoder.parameters())
    module.enable_trainable_decode(parallel_stage_decode=True)
    module._decoder_compute_dtype = torch.float32
    weight = module._decode_weight(dtype=torch.float32)
    full_loss = linear_output_mse(weight - reference, train_x, chunk_size=13, recompute=False)
    direct_loss = F.linear(train_x, weight - reference).square().mean()
    full_grad = torch.autograd.grad(full_loss, parameters, retain_graph=True)
    direct_grad = torch.autograd.grad(direct_loss, parameters, retain_graph=True)
    for actual, expected in zip(full_grad, direct_grad):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    keep = torch.ones(16, dtype=torch.bool)
    keep[module.protected_input_indices] = False
    cropped_loss = F.linear(train_x[:, keep], (weight - reference)[:, keep]).square().mean()
    cropped_grad = torch.autograd.grad(cropped_loss, parameters)
    assert any(not torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(full_grad, cropped_grad))
    module.disable_trainable_decode()
    delattr(module, "_decoder_compute_dtype")
    result = optimize_local_decoder(module, reference, train_x, heldout_x, steps=12, lr=0.005, chunk_size=13)
    assert result["settings"]["source_profile"] == "no_rotation_int8_input"
    assert result["after_cast"]["heldout_mse"] < result["before_native"]["heldout_mse"]
    assert all(value > 0 for value in result["evidence"]["stage_gradient_l2_max"])
    assert all(value > 0 for value in result["evidence"]["stage_parameter_delta_l2_saved"])
    for name, tensor in module.state_dict().items():
        if not name.startswith("_parallel_stage_decoder."):
            assert torch.equal(tensor, before[name]), name
    with torch.no_grad():
        saved_weight = module._decode_weight(dtype=torch.float32)
        assert result["after_cast"]["heldout_mse"] == pytest.approx(
            float(F.linear(heldout_x, saved_weight - reference).square().mean()), rel=2e-6)
        protected_rows = module._decode_protected_input_weight_rows(dtype=torch.float32, device=torch.device("cpu"))
        torch.testing.assert_close(saved_weight[:, module.protected_input_indices], protected_rows.T, rtol=0, atol=0)


def test_int8_protection_profile_rejects_wrong_storage_and_nonmultiple_count():
    module, _, _, _ = _fixture(protected=True)
    assert _validate_payload_profile(module) == "no_rotation_int8_input"
    module.protected_input_indices = module.protected_input_indices[:3]
    with pytest.raises(ValueError, match="codebook_dim multiple"):
        _validate_payload_profile(module)
    module, _, _, _ = _fixture(protected=True)
    module.protected_channel_quant_format = "fp8_e4m3"
    with pytest.raises(ValueError, match="INT8 profile"):
        _validate_payload_profile(module)
    module.protected_channel_quant_format = "int8"
    module.protected_input_scales = module.protected_input_scales.float()
    with pytest.raises(ValueError, match="qvalues/scales"):
        _validate_payload_profile(module)
