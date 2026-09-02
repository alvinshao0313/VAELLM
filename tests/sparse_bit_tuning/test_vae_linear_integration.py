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


def test_grouped_multistage_bit_graph_uses_actual_layout_and_commits():
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
            enabled=True, active_ratio=0.5, optimizer="rms_sgd", bit_lr=2.0, round_steps=1
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
