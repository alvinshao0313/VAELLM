import tempfile

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.checkpoint import load_sidecar, save_sidecar, sidecar_complete
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _decoder(latent_dim=9, codebook_dim=4):
    return Decoder(
        in_dim=latent_dim,
        out_dim=codebook_dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )


def _layer():
    bits = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=_decoder(),
        codebook_dim=4,
        transpose=False,
    )


def _manager(*, streaming=False, round_steps=5, seed=17):
    device = torch.device("cuda:0")
    layer = _layer().to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=False)
    root = nn.Module()
    root.add_module("layer", layer)
    manager = SparseBitTuningManager(
        root_model=root,
        targets=[("layer", layer)],
        target_devices={"layer": device},
        training_seed=seed,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="rms_sgd",
            bit_lr=2.0,
            round_steps=round_steps,
        ),
        streaming=streaming,
    )
    manager.configure_schedule(total_optimizer_steps=20)
    manager.initialize_scores()
    return root, layer, manager


def _invert_score_signs(manager):
    with torch.no_grad():
        for score in manager.score_module.score_chunks:
            score.copy_(torch.where(score >= 0, -torch.ones_like(score), torch.ones_like(score)))


def test_checkpoint_snapshot_is_read_only_and_sidecar_round_trip():
    _root, layer, manager = _manager(streaming=False)
    persistent_before = layer.get_stage_part_vq_storage(0, 0).detach().clone()
    state_before = manager.sampler_states[next(iter(manager.sampler_states))]
    _invert_score_signs(manager)
    snapshot = manager.checkpoint_packed_snapshot()
    coverage = manager.coverage_metadata()
    assert torch.equal(layer.get_stage_part_vq_storage(0, 0), persistent_before)
    assert manager.sampler_states[state_before.canonical_key] == state_before
    assert manager.bit_round_step == 0
    assert not torch.equal(snapshot[state_before.canonical_key], persistent_before.cpu())

    with tempfile.TemporaryDirectory() as tmp:
        save_sidecar(tmp, packed_banks=snapshot, coverage=coverage)
        assert sidecar_complete(tmp)
        packed2, coverage2 = load_sidecar(tmp)
        assert torch.equal(packed2[state_before.canonical_key], snapshot[state_before.canonical_key])
        assert coverage2 == coverage


def test_streaming_round_end_snapshot_serializes_next_sampler_without_mutating_live_pending():
    _root, layer, manager = _manager(streaming=True, round_steps=1)
    old_state = manager.sampler_states[next(iter(manager.sampler_states))]
    persistent_before = layer.get_stage_part_vq_storage(0, 0).detach().clone()
    for score in manager.score_module.score_chunks:
        score.grad = torch.where(score.detach() >= 0, torch.ones_like(score), -torch.ones_like(score))
    telemetry = manager.optimizer_step()
    assert telemetry.round_ended
    assert manager.pending_next_states
    pending_before = dict(manager.pending_next_states)
    snapshot = manager.checkpoint_packed_snapshot()
    coverage = manager.coverage_metadata()
    assert manager.pending_next_states == pending_before
    assert manager.sampler_states[old_state.canonical_key] == old_state
    assert torch.equal(layer.get_stage_part_vq_storage(0, 0), persistent_before)
    assert coverage["global_bit_round"] == 1
    bank_meta = {item["canonical_key"]: item for item in coverage["banks"]}[old_state.canonical_key]
    next_state = pending_before[old_state.canonical_key]
    assert bank_meta["coverage_id"] == next_state.coverage_id
    assert bank_meta["cursor"] == next_state.cursor
    assert not torch.equal(snapshot[old_state.canonical_key], persistent_before.cpu())


def test_resume_restores_packed_and_post_step_sampler_then_reinitializes_score():
    _root, _layer0, manager0 = _manager(streaming=True, round_steps=1, seed=23)
    for score in manager0.score_module.score_chunks:
        score.grad = torch.where(score.detach() >= 0, torch.ones_like(score), -torch.ones_like(score))
    manager0.optimizer_step()
    snapshot = manager0.checkpoint_packed_snapshot()
    coverage = manager0.coverage_metadata()

    _root1, layer1, manager1 = _manager(streaming=True, round_steps=7, seed=23)
    manager1.restore_checkpoint_packed(snapshot)
    manager1.restore_coverage_metadata(coverage)
    assert manager1.global_bit_round == coverage["global_bit_round"]
    assert manager1.bit_round_step == 0
    assert manager1.stable_counter == 0
    assert not manager1.pending_next_states
    manager1.initialize_scores()
    spec = manager1.bank_specs[0]
    restored_storage = layer1.get_stage_part_vq_storage(0, 0).detach().cpu()
    assert torch.equal(restored_storage, snapshot[spec.canonical_key])
    expected_meta = {item["canonical_key"]: item for item in coverage["banks"]}[spec.canonical_key]
    state = manager1.sampler_states[spec.canonical_key]
    assert state.coverage_id == expected_meta["coverage_id"]
    assert state.cursor == expected_meta["cursor"]
    active = state.active_indices()
    logical = torch.empty(spec.n_bits, dtype=torch.bool)
    packed = restored_storage.reshape(-1)
    for idx in range(spec.n_bits):
        block = idx // spec.latent_dim
        inner = idx % spec.latent_dim
        byte = block * ((spec.latent_dim + 7) // 8) + inner // 8
        bit = inner % 8
        logical[idx] = bool((int(packed[byte]) >> bit) & 1)
    score = manager1.score_module.score_view(spec).detach().cpu()
    expected_score = torch.tensor([1.0 if logical[i] else -1.0 for i in active], dtype=torch.float16)
    assert torch.equal(score, expected_score)


def test_resume_rejects_sampling_config_mismatch():
    _root, _layer0, manager0 = _manager(seed=31)
    coverage = manager0.coverage_metadata()
    _root1, _layer1, manager1 = _manager(seed=32)
    with pytest.raises(ValueError, match="training seed mismatch"):
        manager1.restore_coverage_metadata(coverage)
