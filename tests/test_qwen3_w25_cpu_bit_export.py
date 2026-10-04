"""CPU export checks with real VAELinear banks and exact-state schema fixtures.

Production snapshot projection runs Triton. These CPU fixtures use its existing
independent packed reference with a real manager's layout, sampler and optimizer
metadata; they do not replace any runtime dependency.
"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.qwen3_w25.cpu_bit_export import restore_packed_bits_cpu
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.exact_checkpoint import EXACT_FORMAT, EXACT_VERSION, exact_state_path
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.reference import apply_bank_scores_reference, read_bank_hard_bits, scores_from_hard_bits


def _model():
    torch.manual_seed(37)
    model = nn.Module()
    model.frozen = nn.Linear(4, 4, bias=False)
    for name in ("left", "right"):
        model.add_module(name, VAELinear(in_features=4, out_features=4, bias=None,
            codebook_dim=4, transpose=False,
            stage_vq_weights=[torch.randint(0, 2, (4, 1, 9), dtype=torch.bool) for _ in range(2)],
            stage_decoders=[Decoder(in_dim=9, out_dim=4, hidden_dim=8, num_res_blocks=0,
                norm_type="layer", decoder_type="linear") for _ in range(2)]))
    return model


def _fixture(*, ratio=1.0, coordinates="decoder_sensitivity", global_round=0, pending=False,
             split_layout=False):
    model = _model()
    selected = [(name, model.get_submodule(name)) for name in ("left", "right")]
    config = SparseBitTuningConfig(enabled=True, active_ratio=ratio, optimizer="adam", bit_lr=4e-5,
                                   round_steps=128, proxy_coordinates=coordinates)
    # CPU indices let the real layout function exercise distinct/reversed device
    # groups without allocating GPU tensors or emulating GPU kernels.
    devices = {"left": torch.device("cpu:1"), "right": torch.device("cpu:0")} if split_layout else {
        name: torch.device("cpu") for name, _ in selected}
    manager = SparseBitTuningManager(root_model=model, targets=selected, target_devices=devices,
                                    training_seed=31, config=config, streaming=pending)
    manager.configure_schedule(total_optimizer_steps=2000)
    scales = {spec.canonical_key: 0.0125 + i * 0.001 for i, spec in enumerate(manager.bank_specs)}
    if coordinates == "decoder_sensitivity":
        manager._set_coordinate_scales(scales)
    packed = {}
    for spec in manager.bank_specs:
        key = spec.canonical_key
        sampler = manager.sampler_states[key]
        for _ in range(global_round - int(pending)):
            sampler = sampler.advance()
        manager.sampler_states[key] = sampler
        if pending:
            manager.pending_next_states[key] = sampler.advance()
        storage = model.get_submodule(spec.module_path).get_stage_part_vq_storage(spec.stage_idx, spec.part_idx)
        original_bits = read_bank_hard_bits(storage, state=sampler, model_idx=0, logical_in_dim=spec.latent_dim)
        score = scores_from_hard_bits(original_bits, proxy_scale=scales[key] if coordinates == "decoder_sensitivity" else None)
        score[::3] *= -0.5
        score[1] = 0.0  # Zero is a hard one, including negative zero.
        with torch.no_grad():
            manager.score_module.score_view(spec).copy_(score)
        packed[key], _ = apply_bank_scores_reference(storage, state=sampler, scores=score,
                                                      model_idx=0, logical_in_dim=spec.latent_dim)
    state = {"format": "sparse_bit_tuning_exact_state", "version": manager.coordinate_state_version,
        **manager._coordinate_metadata(), "training_seed": 31, "streaming": pending,
        "bit_active_ratio": ratio, "target_modules": ["left", "right"],
        "banks": [{field: getattr(spec, field) for field in (
            "canonical_key", "module_path", "stage_idx", "part_idx", "n_bits", "n_active", "chunk_id",
            "score_start", "score_end")} for spec in manager.bank_specs],
        "packed_banks": packed,
        "score_chunks": [score.detach().clone() for score in manager.score_module.score_chunks],
        "sampler_states": {key: value.to_metadata() for key, value in manager.sampler_states.items()},
        "pending_next_states": {key: value.to_metadata() for key, value in manager.pending_next_states.items()},
        "bit_round_steps": manager.bit_round_steps, "stable_steps": manager.stable_steps,
        "global_bit_round": global_round, "bit_round_step": 0 if pending else 5,
        "stable_counter": 0, "cumulative_flip_count": 0 if pending else 7, "had_flip": not pending,
        "bit_optimizer": manager.bit_optimizer.state_dict()}
    contract = {"sparse_bit": {name: getattr(config, name) for name in (
        "active_ratio", "optimizer", "bit_lr", "weight_decay", "round_steps", "proxy_coordinates")},
        "data": {"seed": 31}, "optimization": {"steps": 2000},
        "runtime": {"offload_mode": "layerwise" if pending else "none"}}
    manager.detach_runtime()
    return state, contract


def _write(tmp_path, state):
    from pathlib import Path
    path = Path(exact_state_path(str(tmp_path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": EXACT_FORMAT, "version": EXACT_VERSION, "state": state}, path)


@pytest.mark.parametrize("ratio,coordinates,global_round,pending,split_layout", [
    (1.0, "decoder_sensitivity", 0, False, False),
    (1.0, "decoder_sensitivity", 3, False, True),
    (0.4, "decoder_sensitivity", 2, False, True),  # Tail subset with earlier-prefix fill.
    (1.0, "decoder_sensitivity", 3, True, True),
    (0.4, "unit", 3, True, False),
])
def test_current_hard_snapshot_matches_reference_and_preserves_other_state(
        tmp_path, ratio, coordinates, global_round, pending, split_layout):
    state, contract = _fixture(ratio=ratio, coordinates=coordinates, global_round=global_round,
                               pending=pending, split_layout=split_layout)
    _write(tmp_path, state)
    model = _model()
    before = deepcopy(model.state_dict())
    selected = [(name, model.get_submodule(name)) for name in ("left", "right")]
    audit = restore_packed_bits_cpu(selected, checkpoint=tmp_path, contract=contract)
    assert audit["packed_banks_compared"] == 4
    assert audit["exact_sidecar_hard_codes_restored"] is True
    assert audit["exact_training_state_restored"] is False
    assert audit["internal_bit_optimizer_restored"] is False
    assert audit["source_score_chunks"] == (2 if split_layout else 1)
    assert audit["sidecar_read_mode"] == "cpu_mmap"
    assert not hasattr(model, "sparse_bit_tuning")
    expected_state = deepcopy(before)
    changed_storages = []
    for row in state["banks"]:
        module = model.get_submodule(row["module_path"])
        target = module.get_stage_part_vq_storage(row["stage_idx"], row["part_idx"])
        assert torch.equal(target, state["packed_banks"][row["canonical_key"]])
        assert getattr(module, "_sparse_bit_binding", None) is None
        changed_storages.append(target.data_ptr())
    for key, tensor in model.state_dict().items():
        if tensor.data_ptr() not in changed_storages:
            assert torch.equal(tensor, expected_state[key]), key
    assert any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize("corruption", ["sign", "nan", "range", "chunk_overlap", "key", "sampler_seed",
                                      "sampler_round", "scale", "schedule", "partial_pending", "wrong_pending"])
def test_corrupt_snapshot_is_rejected_before_any_model_mutation(tmp_path, corruption):
    state, contract = _fixture(ratio=0.4, global_round=3, pending=True, split_layout=True)
    row = state["banks"][0]
    key = row["canonical_key"]
    scores = state["score_chunks"][row["chunk_id"]]
    if corruption == "sign":
        scores[row["score_start"]] *= -1
    elif corruption == "nan":
        scores[row["score_start"]] = float("nan")
    elif corruption == "range":
        scores[row["score_start"]] = 1.0
    elif corruption == "chunk_overlap":
        state["banks"][1]["score_start"] = 0
    elif corruption == "key":
        row["canonical_key"] = "wrong|stage=0|part=0"
    elif corruption == "sampler_seed":
        state["sampler_states"][key]["training_seed"] += 1
    elif corruption == "sampler_round":
        state["global_bit_round"] += 1
    elif corruption == "scale":
        state["coordinate_scales"][key] = 0.0
    elif corruption == "schedule":
        state["bit_round_steps"] = 129
    elif corruption == "partial_pending":
        del state["pending_next_states"][key]
    elif corruption == "wrong_pending":
        state["pending_next_states"][key] = state["sampler_states"][key]
    _write(tmp_path, state)
    model = _model()
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        restore_packed_bits_cpu([(name, model.get_submodule(name)) for name in ("left", "right")],
                                checkpoint=tmp_path, contract=contract)
    assert all(torch.equal(tensor, before[key]) for key, tensor in model.state_dict().items())
