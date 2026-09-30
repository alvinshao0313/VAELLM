import copy

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _decoder(latent_dim=9, codebook_dim=4):
    dec = Decoder(
        in_dim=latent_dim,
        out_dim=codebook_dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )
    return dec


def _bits(offset=0):
    base = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )
    return ~base if offset else base


def _root_with_layer(layer):
    root = nn.Module()
    root.add_module("layer", layer)
    return root


def _force_flip_all_current_scores(manager):
    for score in manager.score_module.score_chunks:
        assert score.grad is not None
        score.grad.copy_(torch.where(score.detach() >= 0, 1.0, -1.0).to(torch.float16))


def test_serial_bit_graph_updates_packed_and_next_forward():
    device = torch.device("cuda:0")
    layer = VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=_bits(),
        decoder=_decoder(),
        codebook_dim=4,
        transpose=False,
    ).to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    root = _root_with_layer(layer)
    manager = SparseBitTuningManager(
        root_model=root,
        targets=[("layer", layer)],
        target_devices={"layer": device},
        training_seed=5,
        config=SparseBitTuningConfig(
            enabled=True, active_ratio=0.5, optimizer="rms_sgd", bit_lr=2.0, round_steps=1
        ),
        streaming=False,
    )
    manager.configure_schedule(total_optimizer_steps=2)
    manager.initialize_scores()
    before_packed = layer.get_stage_part_vq_storage(stage_idx=0, part_idx=0).detach().clone()
    before_weight = layer._decode_weight(dtype=torch.bfloat16).detach().clone()
    loss = layer._decode_weight(dtype=torch.bfloat16).float().sum()
    loss.backward()
    assert all(param.grad is None for dec in [layer.get_stage_part_decoder(0, 0)] for param in dec.parameters())
    assert all(score.grad is not None for score in manager.score_module.score_chunks)
    _force_flip_all_current_scores(manager)
    telemetry = manager.optimizer_step()
    assert telemetry.round_ended
    assert telemetry.step_flip_count == sum(spec.n_active for spec in manager.bank_specs)
    after_packed = layer.get_stage_part_vq_storage(stage_idx=0, part_idx=0).detach().clone()
    assert not torch.equal(before_packed, after_packed)
    with torch.no_grad():
        after_weight = layer._decode_weight(dtype=torch.bfloat16)
    assert not torch.equal(before_weight, after_weight)
    assert manager.global_bit_round == 1
    assert next(iter(manager.sampler_states.values())).cursor > 0
    manager.final_commit()
    manager.detach_runtime()
    assert not hasattr(root, "sparse_bit_tuning")
    assert not hasattr(layer, "_sparse_bit_binding")


@pytest.mark.parametrize("proxy_coordinates", ["unit", "decoder_sensitivity"])
def test_grouped_multistage_bit_graph_uses_actual_layout_and_commits(proxy_coordinates):
    device = torch.device("cuda:0")
    dec0, dec1 = _decoder(), _decoder()
    layer = VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=None,
        decoder=None,
        stage_vq_weights=[_bits(), _bits(offset=1)],
        stage_decoders=[dec0, dec1],
        codebook_dim=4,
        stage_codebook_dims=[4, 4],
        transpose=False,
        parallel_parts=1,
    ).to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
    assert layer._parallel_stage_model_indices is not None
    root = _root_with_layer(layer)
    manager = SparseBitTuningManager(
        root_model=root,
        targets=[("layer", layer)],
        target_devices={"layer": device},
        training_seed=7,
        config=SparseBitTuningConfig(
            enabled=True, proxy_coordinates=proxy_coordinates, active_ratio=0.5, optimizer="rms_sgd", bit_lr=2.0, round_steps=1
        ),
        streaming=False,
    )
    manager.configure_schedule(total_optimizer_steps=2)
    manager.initialize_scores()
    before = [
        layer.get_stage_part_vq_storage(stage_idx=s, part_idx=0).detach().clone() for s in range(2)
    ]
    loss = layer._decode_weight(dtype=torch.bfloat16).float().square().mean()
    loss.backward()
    _force_flip_all_current_scores(manager)
    telemetry = manager.optimizer_step()
    assert telemetry.round_ended
    assert telemetry.step_flip_count == sum(spec.n_active for spec in manager.bank_specs)
    after = [layer.get_stage_part_vq_storage(stage_idx=s, part_idx=0).detach().clone() for s in range(2)]
    assert all(not torch.equal(a, b) for a, b in zip(before, after))
    assert all(not p.requires_grad for p in layer._parallel_stage_decoder.parameters())


@pytest.mark.parametrize("decoder_type", ["linear", "symmetric"])
def test_sensitivity_calibration_preserves_rng_model_and_initial_bits(decoder_type):
    device = torch.device("cuda:0")
    decoder = Decoder(
        in_dim=9, out_dim=4, hidden_dim=8, num_res_blocks=1,
        norm_type="layer", decoder_type=decoder_type,
        use_checkpoint=False, num_models=1,
    )
    layer = VAELinear(
        in_features=4, out_features=4, bias=None, original_weight=None,
        vq_weight=_bits(), decoder=decoder, codebook_dim=4, transpose=False,
    ).to(device=device, dtype=torch.float32)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    root = _root_with_layer(layer)
    before = {key: value.detach().clone() for key, value in layer.state_dict().items()}
    manager = SparseBitTuningManager(
        root_model=root, targets=[("layer", layer)], target_devices={"layer": device},
        training_seed=61, streaming=False,
        config=SparseBitTuningConfig(
            enabled=True, active_ratio=0.5, bit_lr=2e-5,
            proxy_coordinates="decoder_sensitivity",
        ),
    )
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    manager.initialize_scores()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
    for key, value in before.items():
        assert torch.equal(value, layer.state_dict()[key])
    spec = manager.bank_specs[0]
    scale = manager.score_module.coordinate_scales[spec.canonical_key]
    # Independent full-decoder finite differences, using dense logical input.
    decoder = layer.get_stage_part_decoder(0, 0)
    bits = _bits().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        reference = decoder(bits)
        mse = []
        for bit in range(9):
            trial = bits.clone()
            trial[..., bit] = 1 - trial[..., bit]
            mse.append((decoder(trial) - reference).square().mean())
        expected = torch.stack(mse).mean().sqrt()
    # Packed FP32 tl.dot uses TF32 (10 mantissa bits); dense PyTorch may use IEEE.
    # Two TF32 ulps allow that documented rounding, not a different scale formula.
    torch.testing.assert_close(torch.tensor(scale, device=device), expected, atol=1e-6, rtol=2e-3)
    score = manager.score_module.score_view(spec)
    assert score.dtype == torch.float32
    assert torch.equal(score.abs(), torch.full_like(score, scale / 2))
    snapshot = manager.checkpoint_packed_snapshot()[spec.canonical_key]
    assert torch.equal(snapshot, layer.get_stage_part_vq_storage(0, 0).cpu())


def test_sensitivity_calibration_uses_explicit_input_dtype_for_fp32_decoder():
    from sparse_bit_tuning.coordinates import calibrate_bank_scale
    device = torch.device("cuda:0")
    layer = VAELinear(
        in_features=4, out_features=4, bias=None, original_weight=None,
        vq_weight=_bits(), decoder=_decoder(), codebook_dim=4, transpose=False,
    ).to(device=device, dtype=torch.float32)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    root = _root_with_layer(layer)
    manager = SparseBitTuningManager(
        root_model=root, targets=[("layer", layer)], target_devices={"layer": device},
        training_seed=67, streaming=False, calibration_dtype=torch.bfloat16,
        config=SparseBitTuningConfig(
            enabled=True, active_ratio=0.5, bit_lr=2e-5,
            proxy_coordinates="decoder_sensitivity",
        ),
    )
    manager.initialize_scores()
    spec = manager.bank_specs[0]
    explicit = manager.score_module.coordinate_scales[spec.canonical_key]
    layer._decoder_compute_dtype = torch.bfloat16
    assert calibrate_bank_scale(layer, spec) == explicit
    assert all(p.dtype == torch.float32 for p in layer.get_stage_part_decoder(0, 0).parameters())


@pytest.mark.parametrize("proxy_coordinates", ["unit", "decoder_sensitivity"])
def test_low_lr_real_backward_changes_bits_only_with_reachable_coordinates(proxy_coordinates):
    # Orthogonal decoder columns isolate the effect of coordinate distance.
    # Real packed forward/backward and Adam are used, with no injected gradients.
    from litebsq.packed_bit_linear import resolve_parallel_linear_weight_bias
    device = torch.device("cuda:0")
    decoder = _decoder(latent_dim=9, codebook_dim=9).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        weight, bias = resolve_parallel_linear_weight_bias(decoder.linear)
        weight.copy_(torch.eye(9, device=device, dtype=torch.bfloat16).unsqueeze(0) * 1e-3)
        bias.zero_()
    bits = _bits()[:1]
    layer = VAELinear(
        in_features=3, out_features=3, bias=None, original_weight=None,
        vq_weight=bits, decoder=decoder, codebook_dim=9, transpose=False,
    ).to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    root = _root_with_layer(layer)
    manager = SparseBitTuningManager(
        root_model=root, targets=[("layer", layer)], target_devices={"layer": device},
        training_seed=71, streaming=False,
        config=SparseBitTuningConfig(
            enabled=True, active_ratio=0.5, optimizer="adam", bit_lr=2e-5,
            round_steps=100, proxy_coordinates=proxy_coordinates,
        ),
    )
    manager.configure_schedule(total_optimizer_steps=100)
    manager.initialize_scores()
    before = layer.get_stage_part_vq_storage(0, 0).clone()
    initial = layer._decode_weight(dtype=torch.bfloat16).detach().clone()
    target = torch.full_like(initial, 1e-3) - initial
    initial_loss = (initial.float() - target.float()).square().mean().item()
    first_flip = None
    for step in range(1, 13):
        root.zero_grad(set_to_none=True)
        loss = (layer._decode_weight(dtype=torch.bfloat16).float() - target.float()).square().mean()
        loss.backward()
        telemetry = manager.optimizer_step()
        if telemetry.step_flip_count:
            first_flip = step
            break
    final_loss = (layer._decode_weight(dtype=torch.bfloat16).float() - target.float()).square().mean().item()
    changed = not torch.equal(before, layer.get_stage_part_vq_storage(0, 0))
    if proxy_coordinates == "decoder_sensitivity":
        assert first_flip is not None and changed
        assert final_loss < initial_loss * 0.75
    else:
        assert first_flip is None and not changed
        assert final_loss == initial_loss
    assert all(p.grad is None for p in layer.get_stage_part_decoder(0, 0).parameters())
    print(f"{proxy_coordinates}: first_flip={first_flip}, mse={initial_loss:.8g}->{final_loss:.8g}")
