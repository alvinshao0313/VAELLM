import tempfile

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.checkpoint import load_sidecar, save_sidecar, sidecar_complete
from sparse_bit_tuning.exact_checkpoint import (
    exact_sidecar_complete,
    load_exact_sidecar,
    restore_exact_sidecar,
    save_exact_sidecar,
)
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


def _manager(*, streaming=False, round_steps=5, seed=17, optimizer="rms_sgd",
             proxy_coordinates="unit", initialize=True):
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
            proxy_coordinates=proxy_coordinates,
            active_ratio=0.5,
            optimizer=str(optimizer),
            bit_lr=2.0,
            round_steps=round_steps,
        ),
        streaming=streaming,
    )
    manager.configure_schedule(total_optimizer_steps=20)
    if initialize:
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


def _assert_tensor_tree_equal(lhs, rhs):
    if torch.is_tensor(lhs) or torch.is_tensor(rhs):
        assert torch.is_tensor(lhs) and torch.is_tensor(rhs)
        assert torch.equal(lhs, rhs)
        return
    if isinstance(lhs, dict) or isinstance(rhs, dict):
        assert isinstance(lhs, dict) and isinstance(rhs, dict)
        assert set(lhs) == set(rhs)
        for key in lhs:
            _assert_tensor_tree_equal(lhs[key], rhs[key])
        return
    if isinstance(lhs, (list, tuple)) or isinstance(rhs, (list, tuple)):
        assert isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple))
        assert len(lhs) == len(rhs)
        for a, b in zip(lhs, rhs):
            _assert_tensor_tree_equal(a, b)
        return
    assert lhs == rhs


def test_exact_state_round_trip_preserves_live_scores_counters_and_adam_state():
    _root0, _layer0, manager0 = _manager(streaming=False, round_steps=5, seed=41, optimizer="adam")
    for score in manager0.score_module.score_chunks:
        score.grad = torch.where(score.detach() >= 0, torch.ones_like(score), -torch.ones_like(score))
    manager0.optimizer_step()
    assert manager0.bit_round_step == 1
    exact0 = manager0.exact_state_dict()
    assert any(
        payload["exp_avg"] is not None
        for payload in exact0["bit_optimizer"]["chunks"].values()
    )

    _root1, _layer1, manager1 = _manager(streaming=False, round_steps=5, seed=41, optimizer="adam")
    manager1.load_exact_state_dict(exact0)
    exact1 = manager1.exact_state_dict()
    _assert_tensor_tree_equal(exact0, exact1)
    # configure_schedule is called again by Trainer.create_scheduler on resume;
    # it must validate, not reset, the restored schedule.
    manager1.configure_schedule(total_optimizer_steps=20)
    assert manager1.bit_round_step == 1


def test_exact_state_round_trip_preserves_streaming_pending_transition():
    _root0, _layer0, manager0 = _manager(streaming=True, round_steps=1, seed=43)
    for score in manager0.score_module.score_chunks:
        score.grad = torch.where(score.detach() >= 0, torch.ones_like(score), -torch.ones_like(score))
    telemetry = manager0.optimizer_step()
    assert telemetry.round_ended
    assert manager0.pending_next_states
    exact0 = manager0.exact_state_dict()

    _root1, _layer1, manager1 = _manager(streaming=True, round_steps=1, seed=43)
    manager1.load_exact_state_dict(exact0)
    assert manager1.pending_next_states == manager0.pending_next_states
    assert manager1.sampler_states == manager0.sampler_states
    assert manager1.global_bit_round == manager0.global_bit_round == 1
    assert manager1.bit_round_step == manager0.bit_round_step == 0
    _assert_tensor_tree_equal(exact0, manager1.exact_state_dict())


def test_exact_sidecar_file_round_trip_uses_exact_restore_api():
    _root0, _layer0, manager0 = _manager(streaming=False, round_steps=5, seed=47, optimizer="adam")
    for score in manager0.score_module.score_chunks:
        score.grad = torch.where(score.detach() >= 0, torch.ones_like(score), -torch.ones_like(score))
    manager0.optimizer_step()
    expected = manager0.exact_state_dict()

    with tempfile.TemporaryDirectory() as tmp:
        save_exact_sidecar(tmp, manager0)
        assert exact_sidecar_complete(tmp)
        _assert_tensor_tree_equal(expected, load_exact_sidecar(tmp))

        _root1, _layer1, manager1 = _manager(streaming=False, round_steps=5, seed=47, optimizer="adam")
        restore_exact_sidecar(tmp, manager1)
        _assert_tensor_tree_equal(expected, manager1.exact_state_dict())


@pytest.mark.parametrize("streaming", [False, True])
def test_sensitivity_coordinates_survive_exact_and_coverage_resume(tmp_path, streaming):
    root, layer, manager = _manager(
        streaming=streaming, round_steps=1, seed=53, optimizer="adam",
        proxy_coordinates="decoder_sensitivity",
    )
    scales = dict(manager.score_module.coordinate_scales)
    for score in manager.score_module.score_chunks:
        assert score.dtype == torch.float32
        score.grad = torch.sign(score.detach())
    telemetry = manager.optimizer_step()
    assert telemetry.step_flip_count > 0
    assert telemetry.round_ended
    assert manager.score_module.coordinate_scales == scales
    exact = manager.exact_state_dict()
    assert exact["version"] == 2
    assert exact["coordinate_scales"] == scales
    save_exact_sidecar(str(tmp_path), manager)

    _root2, layer2, manager2 = _manager(
        streaming=streaming, round_steps=1, seed=53, optimizer="adam",
        proxy_coordinates="decoder_sensitivity", initialize=False,
    )
    # A resumed decoder may have changed; its initial coordinate scale must not.
    with torch.no_grad():
        for parameter in layer2.get_stage_part_decoder(0, 0).parameters():
            parameter.mul_(3)
    restore_exact_sidecar(str(tmp_path), manager2)
    manager2.initialize_scores()
    _assert_tensor_tree_equal(exact, manager2.exact_state_dict())

    snapshot, coverage = manager.checkpoint_packed_snapshot(), manager.coverage_metadata()
    _root3, _layer3, manager3 = _manager(
        streaming=streaming, round_steps=1, seed=53, optimizer="adam",
        proxy_coordinates="decoder_sensitivity", initialize=False,
    )
    manager3.restore_checkpoint_packed(snapshot)
    manager3.restore_coverage_metadata(coverage)
    manager3.initialize_scores()
    assert manager3.score_module.coordinate_scales == scales
    for spec in manager3.bank_specs:
        scores = manager3.score_module.score_view(spec)
        assert torch.equal(scores.abs(), torch.full_like(scores, scales[spec.canonical_key] / 2))
    for key, packed in manager3.checkpoint_packed_snapshot().items():
        assert torch.equal(packed, snapshot[key])

    manager.final_commit()
    manager.detach_runtime()
    assert not hasattr(root, "sparse_bit_tuning")
    assert not any("coordinate" in key or "score" in key for key in root.state_dict())
    assert layer.get_stage_part_vq_storage(0, 0).dtype == torch.uint8


def test_sensitivity_resume_rejects_missing_or_invalid_scales():
    import copy
    _, _, manager = _manager(proxy_coordinates="decoder_sensitivity")
    state = manager.exact_state_dict()
    for scales in ({}, {key: float("nan") for key in state["coordinate_scales"]}):
        damaged = copy.deepcopy(state)
        damaged["coordinate_scales"] = scales
        with pytest.raises(ValueError, match="scale"):
            manager.load_exact_state_dict(damaged)
    _, _, unit_manager = _manager()
    with pytest.raises(ValueError, match="format/version"):
        unit_manager.load_exact_state_dict(state)
