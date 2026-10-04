"""Single-stage K48/D16 packed numerics and the CAT VAE full batch/module shape."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
from transformers import get_scheduler

from litebsq.autoencoder import Decoder
import litebsq.fused_multistage_decoder as fused
from litebsq.llm_vae import MultiLayerVAE
from test_cat_packed_decoder import _Model, _inputs, _save, _serial
from test_fused_multistage_decoder_parity import _native
from train_utils.cat_train_data import apply_stage_norm, compute_stage_norm_stats
from train_utils.cat_train_pipeline import (
    _build_vae_linear_from_stage_payload,
    _fuse_norm_into_decoder,
    _fuse_q_scale_into_decoder,
)
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
from train_utils.train_args import create_optimizer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_single_stage_d16_bf16_packed_native_fused_and_group8(record_property):
    torch.manual_seed(31)
    decoder = Decoder(
        in_dim=48, out_dim=16, hidden_dim=128, num_res_blocks=0,
        norm_type="layer", activation_type="swish", decoder_type="symmetric",
        use_checkpoint=False, num_models=1,
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    assert decoder.num_models == decoder.linear_in.num_models == 1
    assert fused.packed_symmetric_decoder_supports_fuse(decoder)
    binary = torch.randint(0, 2, (256, 1, 48), device="cuda", dtype=torch.uint8)
    shifts = torch.arange(8, device="cuda", dtype=torch.uint8)
    packed = (binary.view(256, 1, 6, 8) << shifts).sum(-1).to(torch.uint8)
    dense = binary.to(torch.bfloat16)
    parameters = tuple(decoder.parameters())
    torch.manual_seed(47)
    gradient = torch.randn(256, 16, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _native(decoder, packed)
        expected_grads = torch.autograd.grad(expected, parameters, gradient)
        actual = fused.fused_decode_packed_symmetric_decoder(decoder, dense)
        actual_grads = torch.autograd.grad(actual, parameters, gradient)
    tolerance = 2 * torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    gradient_errors = {}
    for (name, parameter), reference, value in zip(
        decoder.named_parameters(), expected_grads, actual_grads, strict=True,
    ):
        assert value.dtype == parameter.dtype == torch.bfloat16, name
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        relative = (value.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
        assert relative <= tolerance, (name, float(relative))
        gradient_errors[name] = float(relative)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _native(decoder, packed)
        actual = fused.fused_decode_packed_symmetric_decoder(decoder, dense)
        batched = fused.fused_decode_batched_same_shape_decoders([decoder] * 8, [dense] * 8)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert len(batched) == 8
    for item in batched:
        torch.testing.assert_close(item, expected, atol=tolerance, rtol=tolerance)
    record_property("parity", json.dumps({
        "bits": 48, "dim": 16, "stages": 1, "hidden": 128, "residual_blocks": 0,
        "deployment_num_models": 1, "rows": 256, "group_size": 8,
        "storage_dtype": "bfloat16", "output_atol_rtol": tolerance,
        "all_parameter_gradient_relative_l2_limit": tolerance,
        "parameter_gradient_relative_l2": gradient_errors,
    }))


def test_single_stage_d16_cat_batch8192_models36_and_packed_v6(tmp_path, record_property):
    # Synthetic weight blocks exercise the real CAT VAE core, without the 8B
    # model, calibration data, or a complete category dataset resident in memory.
    torch.manual_seed(31)
    args = SimpleNamespace(
        parallel_layers=36, codebook_bits=48, codebook_dim=16,
        base_ch=128, num_res_blocks=0, decoder_type="symmetric",
        norm_type="layer", activation_type="swish", vae_decoder_checkpoint=True,
        vae_weight_dtype="bf16", vae_autocast_dtype="bf16",
        recon_loss_type="mse", quantizer_type="BSQ", new_quant=True,
        gamma0=1.0, gamma=1.0, zeta=1.0, inv_temperature=100.0,
        l1_weight=1.0, lfq_weight=2.5, commitment_loss_weight=0.25,
        entropy_loss_weight=0.01, optimizer="adamw", beta1=0.9, beta2=0.95,
        weight_decay=0.0, lr=3e-3,
    )
    blocks = torch.randn(8192, 36, 16)
    mean, scale = compute_stage_norm_stats(blocks)
    x = apply_stage_norm(blocks, mean=mean, scale=scale).to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    vae = MultiLayerVAE(args).cuda().train()
    assert vae.num_models == 36 and vae.model.decoder.use_checkpoint
    assert vae.model.encoder.num_res_blocks == vae.model.decoder.num_res_blocks == 0
    assert all(p.dtype == torch.bfloat16 for p in vae.parameters())
    initial = {name: p.detach().clone() for name, p in vae.named_parameters()}
    optimizer = create_optimizer(vae.parameters(), args, args.lr)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=20000)
    steps = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        recon, losses = vae(x, is_train=True)
        assert recon.shape == x.shape and recon.dtype == torch.bfloat16
        loss = losses["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in vae.named_parameters():
            assert parameter.grad is not None, name
            assert parameter.grad.dtype == parameter.dtype, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
        scheduler.step()
        assert all(torch.isfinite(p).all() for p in vae.parameters())
        for state in optimizer.state.values():
            assert all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all()
                       for value in state.values())
        torch.cuda.synchronize()
        steps.append({
            "stage": 1, "step": step + 1, "loss": float(loss.detach()),
            "learning_rate": scheduler.get_last_lr()[0],
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
    for prefix in ("model.encoder.", "model.decoder."):
        assert any(not torch.equal(initial[name], p) for name, p in vae.named_parameters()
                   if name.startswith(prefix)), prefix
    vae.eval()
    with torch.no_grad():
        recon, bits = vae(x, is_train=False)
    assert bits.shape == (8192, 36, 48)
    assert ((bits == 0) | (bits == 1)).all()
    decoder = vae.model.decoder.get_sub_decoder(0)
    _fuse_q_scale_into_decoder(decoder, q_scale=1 / math.sqrt(48))
    _fuse_norm_into_decoder(decoder, mean=float(mean[0]), std=float(scale[0]))
    payload_decoder = decoder.cpu()
    payload_bits = bits[:1, :1].to(device="cpu", dtype=torch.bool)
    del vae, optimizer, scheduler, initial, x, recon, bits, loss, losses
    torch.cuda.empty_cache()

    # The tiny carrier persists this trained single-stage K48/D16 payload. CAT's
    # actual conversion chooses its single-stage constructor and then packs it.
    inputs = _inputs(stages=1, dims=[16])
    inputs["stage_part_bits_payload"] = [payload_bits]
    inputs["stage_part_decoders_payload"] = [payload_decoder]
    serial = _serial(inputs).cuda()
    layer = _build_vae_linear_from_stage_payload(**inputs).cuda()
    assert layer.parallel_stage_decode and not layer.trainable_decode
    assert layer.residual_stages == layer._parallel_stage_decoder.num_models == 1
    assert layer._parallel_stage_decoder.in_dim == 48
    assert layer._parallel_stage_decoder.out_dim == 16
    assert all(p.dtype == torch.bfloat16 for p in layer._parallel_stage_decoder.parameters())
    storage = layer.get_stage_part_vq_storage(0, 0)
    assert storage.dtype == torch.uint8 and storage.shape == (1, 1, 6)
    probe = torch.randn(3, 4, device="cuda", dtype=torch.bfloat16)
    tolerance = 2 * torch.finfo(torch.bfloat16).eps
    with torch.no_grad():
        expected = serial(probe)
        packed_output = layer(probe)
    torch.testing.assert_close(packed_output, expected, atol=tolerance, rtol=tolerance)
    model = _Model()
    model.model.layers[0].down_proj = layer.cpu()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    saved = _save(model, tmp_path / "final_model", "final_model")
    spec = saved["meta_payload"]["converted_modules"][0]
    assert spec["parallel_stage_decode"] and spec["residual_stages"] == 1
    loaded, _, result = load_v6_full_checkpoint_into_model(
        _Model(), str(tmp_path / "final_model"), strict=True,
    )
    assert not result.missing_keys and not result.unexpected_keys
    loaded_layer = loaded.model.layers[0].down_proj
    assert loaded_layer.residual_stages == loaded_layer._parallel_stage_decoder.num_models == 1
    loaded_state = loaded.state_dict()
    assert state.keys() == loaded_state.keys()
    for key, value in loaded_state.items():
        assert value.shape == state[key].shape and value.dtype == state[key].dtype, key
        assert torch.equal(value.contiguous().view(-1).view(torch.uint8),
                           state[key].contiguous().view(-1).view(torch.uint8)), key
    with torch.no_grad():
        reloaded_output = loaded.cuda()(probe)
    torch.testing.assert_close(reloaded_output, packed_output, atol=tolerance, rtol=tolerance)
    record_property("training_and_persistence", json.dumps({
        "args": vars(args), "input_shape": [8192, 36, 16],
        "sequential_stages": 1, "steps_per_stage": 3, "scheduler_horizon": 20000,
        "steps": steps, "encoder_and_decoder_updated": True,
        "gradients_and_optimizer_finite": True, "deployment_num_models": 1,
        "strict_reloaded_persistent_tensors": len(state), "persistent_bytes_exact": True,
        "persistence_carrier_linear_shape": [4, 4],
        "boundary": "Real CAT VAE core; synthetic blocks; no 8B model/calibration/dataset residency",
    }))
