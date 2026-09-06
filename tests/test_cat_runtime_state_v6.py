from __future__ import annotations

import pytest
import torch

from train_utils.activation_utils import ActivationCalibrationCache
from train_utils.cat_runtime_state_v6 import (
    build_cat_runtime_state,
    restore_cat_runtime_state,
    serialize_activation_runtime,
    validate_cat_runtime_identity,
)
from train_utils.channel_protection import AdaptiveChannelPlan


def _plan() -> AdaptiveChannelPlan:
    return AdaptiveChannelPlan(
        scope="global",
        axis="input",
        score_metric="weight_abs",
        raw_budget=4,
        used_channels=4,
        counts={"model.layers.0.k_proj": 4},
        selected_indices={"model.layers.0.k_proj": [0, 1, 2, 3]},
        groups=[["model.layers.0.k_proj"]],
        signatures=[(4, 4, False, 1, 1, 4, 4)],
        group_seed_offsets=[0],
        artifact={"raw_budget": 4},
        groups_by_category={"k_proj": [["model.layers.0.k_proj"]]},
        signatures_by_category={"k_proj": [(4, 4, False, 1, 1, 4, 4)]},
        group_seed_offsets_by_category={"k_proj": [0]},
    )


def _activation_runtime():
    cache = ActivationCalibrationCache(
        dataset="wiki=1.0",
        model_path="tiny",
        nsamples=2,
        seqlen=4,
        seed=11,
        input_ids=[torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]])],
    )
    return {
        "dataset": "wiki=1.0",
        "nsamples": 2,
        "seqlen": 4,
        "seed": 11,
        "device": "cpu",
        "log_every": 0,
        "model_path": "tiny",
        "access_token": "x",
        "cache": cache,
        "stats_by_linear": {
            "model.layers.0.k_proj": {
                "absmax": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "abs_mean": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            }
        },
        "mlp_channel_plan_by_linear": {
            "model.layers.0.up_proj": torch.tensor([1, 3], dtype=torch.long)
        },
    }


def test_cat_runtime_state_roundtrip_preserves_planning_state_and_excludes_token():
    runtime = _activation_runtime()
    identity = {
        "compression_categories": ["q_proj", "k_proj"],
        "channel_scope": "global",
        "activation_seed": 11,
    }
    serialized_activation = serialize_activation_runtime(runtime)
    assert "access_token" not in serialized_activation

    payload = build_cat_runtime_state(
        activation_runtime=runtime,
        global_adaptive_plan=_plan(),
        runtime_identity=identity,
    )
    restored_runtime, restored_plan, restored_identity = restore_cat_runtime_state(
        payload,
        access_token="y",
    )
    assert restored_identity == identity
    assert restored_runtime["access_token"] == "y"
    assert restored_runtime["cache"].dataset == "wiki=1.0"
    assert torch.equal(restored_runtime["cache"].input_ids[1], torch.tensor([[5, 6, 7, 8]]))
    assert torch.equal(
        restored_runtime["stats_by_linear"]["model.layers.0.k_proj"]["absmax"],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert restored_plan.scope == "global"
    assert restored_plan.selected_indices == _plan().selected_indices


def test_cat_runtime_identity_is_strict():
    saved = {"channel_scope": "global", "seed": 11}
    validate_cat_runtime_identity(saved, dict(saved))
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        validate_cat_runtime_identity(saved, {"channel_scope": "global", "seed": 12})
