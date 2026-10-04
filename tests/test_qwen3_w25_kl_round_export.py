from copy import deepcopy
import hashlib

import pytest
import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID
from experiments.qwen3_w25.kl_round_export import (
    CODE_KEYS, CODE_SHAPE, compare_replay, save_code_payload, validate_codes)


def _report():
    return {"status": "complete", "source_checkpoint_id": SOURCE_ID, "source_fingerprint": {},
            "steps": 256, "round_steps": 128, "target": "model.layers.0.mlp.down_proj",
            "configuration": {}, "registration": {}, "data_archive": {"sha256": "archive"},
            "final_bits": {}, "heldout": {}, "round_reset_events": [], "frozen_state": {},
            "cumulative_flip_events_all_steps": 0,
            "steps_observed": [{"step": i + 1, "batch_manifest_index": i, "loss": .25,
                                "telemetry": {"bit/step_flip_count": 0}, "training_step_seconds": 1.0}
                               for i in range(256)]}


def test_packed_payload_roundtrip_preserves_two_real_geometry_uint8_banks(tmp_path):
    codes = {name: torch.full(CODE_SHAPE, 85 + index, dtype=torch.uint8)
             for index, name in enumerate(CODE_KEYS)}
    path = tmp_path / "codes.pt"
    evidence = save_code_payload(path, codes, _report())
    actual = torch.load(path, weights_only=True)["codes"]
    assert evidence["tensor_count"] == 2
    assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(torch.equal(actual[name], value) for name, value in codes.items())
    with pytest.raises(ValueError, match="overwrite"):
        save_code_payload(path, codes, _report())


@pytest.mark.parametrize("bad", ["missing", "extra", "shape", "dtype"])
def test_code_inventory_and_packed_layout_reject_invalid_payload(bad):
    codes = {name: torch.zeros(CODE_SHAPE, dtype=torch.uint8) for name in CODE_KEYS}
    if bad == "missing":
        codes.pop(CODE_KEYS[1])
    elif bad == "extra":
        codes["model.norm.weight"] = torch.zeros(1)
    elif bad == "shape":
        codes[CODE_KEYS[0]] = torch.zeros((1, 1, 6), dtype=torch.uint8)
    else:
        codes[CODE_KEYS[0]] = codes[CODE_KEYS[0]].bool()
    with pytest.raises(ValueError):
        validate_codes(codes)


def test_replay_comparison_ignores_runtime_but_detects_actual_update_changes():
    reference = _report()
    replay = deepcopy(reference)
    replay["worker_pid"] = 12
    replay["steps_observed"][0]["training_step_seconds"] = 5.0
    assert compare_replay(replay, reference)["recorded_trajectory_exact"]
    replay["steps_observed"][128]["telemetry"]["bit/step_flip_count"] = 1
    result = compare_replay(replay, reference)
    assert not result["recorded_trajectory_exact"]
    assert result["different_step_fields"] == [{"step": 129, "field": "telemetry"}]
    replay = deepcopy(reference)
    replay["final_bits"] = {"changed": 1}
    assert compare_replay(replay, reference)["different_summary_fields"] == ["final_bits"]
