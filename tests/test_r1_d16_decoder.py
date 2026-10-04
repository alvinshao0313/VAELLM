"""R1/D16 native packed numerics and the CAT VAE full batch/module shape."""

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from transformers import get_scheduler

from litebsq.autoencoder import Decoder
from litebsq.fused_multistage_decoder import packed_symmetric_decoder_supports_fuse
from litebsq.llm_vae import MultiLayerVAE
from litebsq.vae_linear import VAELinear, prime_named_vae_linear_cache
from test_cat_packed_decoder import _Model, _inputs, _save, _serial
from train_utils.cat_train_data import (
    apply_stage_norm,
    compute_stage_norm_stats,
    restore_stage_norm,
)
from train_utils.cat_train_pipeline import (
    _build_vae_linear_from_stage_payload,
    _fuse_norm_into_decoder,
    _fuse_q_scale_into_decoder,
)
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
from train_utils.train_args import create_optimizer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_r1_d16_bf16_native_packed_and_group8(record_property):
    torch.manual_seed(31)
    bits = [torch.randint(0, 2, (256, 1, 24), dtype=torch.bool) for _ in range(2)]
    decoders = [Decoder(
        in_dim=24, out_dim=16, hidden_dim=128, num_res_blocks=1,
        norm_type="layer", activation_type="swish", decoder_type="symmetric",
        use_checkpoint=True,
    ).to(dtype=torch.bfloat16) for _ in range(2)]
    layer = VAELinear(
        in_features=128, out_features=32, bias=None, codebook_dim=16,
        stage_codebook_dims=[16, 16], stage_vq_weights=bits,
        stage_decoders=decoders, transpose=False,
    ).cuda()
    layer.enable_trainable_decode(parallel_stage_decode=True)
    decoder = layer._parallel_stage_decoder
    assert decoder.num_models == 2 and decoder.num_res_blocks == 1
    # R1 is deliberately outside the whole-decoder Triton kernel contract.
    assert not packed_symmetric_decoder_supports_fuse(decoder)
    assert all(p.dtype == torch.bfloat16 for p in decoder.parameters())
    dense = layer._get_parallel_stage_grouped_vq(dtype=torch.bfloat16, device=torch.device("cuda:0"))
    parameters = tuple(decoder.parameters())
    gradient = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    layer.reset_fuse_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = decoder(dense).sum(dim=1).to(torch.bfloat16).reshape(32, 128)
        expected_grads = torch.autograd.grad(expected, parameters, gradient)
        actual = layer._decode_split_weight_parallel_stages(dtype=torch.bfloat16)
        actual_grads = torch.autograd.grad(actual, parameters, gradient)
    tolerance = 2 * torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    gradient_errors = {}
    for (name, parameter), reference, value in zip(
        decoder.named_parameters(), expected_grads, actual_grads, strict=True,
    ):
        assert value.dtype == parameter.dtype and torch.isfinite(value).all(), name
        assert torch.isfinite(reference).all(), name
        relative = (value.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
        assert relative <= tolerance, (name, float(relative))
        gradient_errors[name] = float(relative)
    packed_stats = layer.get_fuse_stats()
    assert packed_stats["packed_u8_linear_hit"] > 0

    layer.disable_trainable_decode()
    layer.eval().requires_grad_(False)
    modules = [copy.deepcopy(layer) for _ in range(8)]
    snapshots = [{name: value.detach().cpu().clone() for name, value in m.state_dict().items()}
                 for m in modules]
    probe = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cold_output = modules[0](probe)
        cold_weight = modules[0]._cached_weight.clone()
        torch.testing.assert_close(cold_weight, expected, atol=tolerance, rtol=tolerance)
        for module in modules:
            module.clear_decoded_weight_cache()
        layer.reset_fuse_stats()
        warmed = prime_named_vae_linear_cache(
            [(f"model.layers.{index}.mlp.down_proj", module) for index, module in enumerate(modules)],
            dtype=torch.bfloat16, group_size=8, compute_device="cuda:0",
        )
        assert warmed == {"total": 8, "warmed": 8, "skipped": 0, "failed": 0}
        group_stats = layer.get_fuse_stats()
        assert group_stats["packed_u8_linear_hit"] == 8
        for module, snapshot in zip(modules, snapshots, strict=True):
            torch.testing.assert_close(module._cached_weight, cold_weight, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(module(probe), cold_output, atol=tolerance, rtol=tolerance)
            state = module.state_dict()
            assert state.keys() == snapshot.keys()
            for name, value in state.items():
                before = snapshot[name]
                assert value.shape == before.shape and value.dtype == before.dtype, name
                assert torch.equal(value.detach().cpu().contiguous().view(-1).view(torch.uint8),
                                   before.contiguous().view(-1).view(torch.uint8)), name
    record_property("parity", json.dumps({
        "bits": 24, "dim": 16, "stages": 2, "hidden": 128, "residual_blocks": 1,
        "rows": 256, "linear_shape": [32, 128], "storage_dtype": "bfloat16",
        "group_size": 8, "output_atol_rtol": tolerance,
        "all_parameter_gradient_relative_l2_limit": tolerance,
        "gradient_relative_l2": gradient_errors,
        "wholefused_supported": False, "packed_stats": packed_stats,
        "group8_stats": group_stats, "group8_warmed": warmed,
        "persistent_state_unchanged": True,
    }))


def test_r1_d16_cat_batch8192_models36_and_packed_v6(tmp_path, record_property):
    # This is the real CAT VAE core, with synthetic weight blocks. It does not
    # measure the resident 8B model, calibration, or complete category dataset.
    torch.manual_seed(31)
    args = SimpleNamespace(
        parallel_layers=36, codebook_bits=24, codebook_dim=16,
        base_ch=128, num_res_blocks=1, decoder_type="symmetric",
        norm_type="layer", activation_type="swish", vae_decoder_checkpoint=True,
        vae_weight_dtype="bf16", vae_autocast_dtype="bf16",
        recon_loss_type="mse", quantizer_type="BSQ", new_quant=True,
        gamma0=1.0, gamma=1.0, zeta=1.0, inv_temperature=100.0,
        l1_weight=1.0, lfq_weight=2.5, commitment_loss_weight=0.25,
        entropy_loss_weight=0.01, optimizer="adamw", beta1=0.9, beta2=0.95,
        weight_decay=0.0, lr=3e-3,
    )
    residual = torch.randn(8192, 36, 16)
    payload_bits, payload_decoders, steps = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for stage in range(2):
        mean, scale = compute_stage_norm_stats(residual)
        x = apply_stage_norm(residual, mean=mean, scale=scale).to(
            device="cuda", dtype=torch.bfloat16,
        )
        vae = MultiLayerVAE(args).cuda().train()
        assert vae.num_models == 36 and vae.model.decoder.use_checkpoint
        assert vae.model.encoder.num_res_blocks == vae.model.decoder.num_res_blocks == 1
        assert all(p.dtype == torch.bfloat16 for p in vae.parameters())
        initial = {name: p.detach().clone() for name, p in vae.named_parameters()}
        optimizer = create_optimizer(vae.parameters(), args, args.lr)
        scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=10000)
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
                "stage": stage + 1, "step": step + 1, "loss": float(loss.detach()),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(),
            })
        for prefix in ("model.encoder.", "model.decoder.",
                       "model.encoder.blocks.0.", "model.decoder.blocks.0."):
            assert any(not torch.equal(initial[name], p) for name, p in vae.named_parameters()
                       if name.startswith(prefix)), prefix
        vae.eval()
        with torch.no_grad():
            recon, bits = vae(x, is_train=False)
        assert bits.shape == (8192, 36, 24)
        assert ((bits == 0) | (bits == 1)).all()
        decoder = vae.model.decoder.get_sub_decoder(0)
        _fuse_q_scale_into_decoder(decoder, q_scale=1 / math.sqrt(24))
        _fuse_norm_into_decoder(decoder, mean=float(mean[0]), std=float(scale[0]))
        payload_decoders.append(decoder.cpu())
        payload_bits.append(bits[:1, :1].to(device="cpu", dtype=torch.bool))
        residual -= restore_stage_norm(recon.float().cpu(), mean=mean, scale=scale)
        del vae, optimizer, scheduler, initial, x, recon, bits, loss, losses
        torch.cuda.empty_cache()

    # Reuse the existing tiny v6 carrier; its actual payload is the trained
    # K24/D16/H128/R1 BF16 decoder pair and binary codes from the core above.
    inputs = _inputs(stages=2, dims=[16, 16])
    inputs["stage_part_bits_payload"] = payload_bits
    inputs["stage_part_decoders_payload"] = payload_decoders
    serial = _serial(inputs).cuda()
    layer = _build_vae_linear_from_stage_payload(**inputs).cuda()
    assert layer.parallel_stage_decode and not layer.trainable_decode
    assert layer._parallel_stage_decoder.num_models == 2
    assert layer._parallel_stage_decoder.num_res_blocks == 1
    assert not packed_symmetric_decoder_supports_fuse(layer._parallel_stage_decoder)
    assert all(p.dtype == torch.bfloat16 for p in layer._parallel_stage_decoder.parameters())
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
    assert saved["meta_payload"]["converted_modules"][0]["parallel_stage_decode"]
    loaded, _, result = load_v6_full_checkpoint_into_model(
        _Model(), str(tmp_path / "final_model"), strict=True,
    )
    assert not result.missing_keys and not result.unexpected_keys
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
        "sequential_stages": 2, "steps_per_stage": 3, "steps": steps,
        "encoder_and_decoder_updated": True, "both_residual_blocks_updated": True,
        "gradients_and_optimizer_finite": True, "wholefused_supported": False,
        "strict_reloaded_persistent_tensors": len(state), "persistent_bytes_exact": True,
        "persistence_carrier_linear_shape": [4, 4],
        "boundary": "Real CAT VAE core; synthetic blocks; no 8B model/calibration/dataset residency",
    }))
