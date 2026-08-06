from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from mix_bit.model_adapter import get_model_adapter
from mix_bit.model_inventory import (
    ModelInventory,
    TargetLinearSpec,
    compute_inventory_fingerprint,
    inventory_from_targets,
)
from mix_bit.schema import (
    CandidateTrainingSpec,
    CategorySpec,
    ModelProfile,
    load_model_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class _Block(nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, k_out: int | None = None):
        super().__init__()
        k_dim = hidden if k_out is None else k_out
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden, hidden),
                "k_proj": nn.Linear(hidden, k_dim),
                "v_proj": nn.Linear(hidden, hidden),
                "o_proj": nn.Linear(hidden, hidden),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden, intermediate),
                "up_proj": nn.Linear(hidden, intermediate),
                "down_proj": nn.Linear(intermediate, hidden),
            }
        )


class _Model(nn.Module):
    def __init__(self, layer_specs: list[tuple[int, int, int | None]]):
        super().__init__()
        layers = []
        for hidden, intermediate, k_out in layer_specs:
            layers.append(_Block(hidden, intermediate, k_out=k_out))
        self.model = nn.ModuleDict({"layers": nn.ModuleList(layers)})
        self.config = type("Cfg", (), {"model_type": "toy"})()


def _profile(categories: list[CategorySpec] | None = None) -> ModelProfile:
    if categories is None:
        categories = [
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("k_proj", "k_proj", False),
            CategorySpec("v_proj", "v_proj", True),
            CategorySpec("o_proj", "o_proj", True),
            CategorySpec("gate_proj", "gate_proj", False),
            CategorySpec("up_proj", "up_proj", False),
            CategorySpec("down_proj", "down_proj", True),
        ]
    return ModelProfile(
        model_id="toy",
        model_path="toy",
        adapter="generic_decoder",
        only_decoder_projections=True,
        candidate_training=CandidateTrainingSpec(linear_group_size="all", allow_tail_group=True),
        layer_index_patterns=(r"(?:^|\.)model\.layers\.(\d+)\.",),
        categories=tuple(categories),
        regression_expectations={},
    )


def _build_inventory(model: nn.Module, profile: ModelProfile) -> ModelInventory:
    adapter = get_model_adapter("generic_decoder")
    targets = adapter.discover_target_linears(model, profile)
    return inventory_from_targets(
        profile=profile,
        model=model,
        targets=targets,
        model_profile_sha256="a" * 64,
    )


def test_inventory_supports_different_layer_counts():
    profile = _profile()
    inv_2 = _build_inventory(_Model([(8, 16, None), (8, 16, None)]), profile)
    inv_3 = _build_inventory(
        _Model([(8, 16, None), (8, 16, None), (8, 16, None)]),
        profile,
    )
    assert inv_2.block_count == 2
    assert inv_3.block_count == 3
    assert len(inv_2.targets) == 14
    assert len(inv_3.targets) == 21


def test_inventory_supports_nonuniform_linear_shapes():
    profile = _profile()
    inv = _build_inventory(
        _Model([(8, 16, 4), (12, 24, 6)]),
        profile,
    )
    k0 = next(t for t in inv.targets if t.block_index == 0 and t.category == "k_proj")
    k1 = next(t for t in inv.targets if t.block_index == 1 and t.category == "k_proj")
    assert k0.out_features == 4
    assert k1.out_features == 6
    assert k0.in_features == 8
    assert k1.in_features == 12


def test_inventory_fingerprint_changes_when_shape_changes():
    profile = _profile()
    inv_a = _build_inventory(_Model([(8, 16, None)]), profile)
    inv_b = _build_inventory(_Model([(8, 32, None)]), profile)
    assert inv_a.fingerprint_sha256 != inv_b.fingerprint_sha256


def test_inventory_fingerprint_changes_when_transpose_changes():
    profile_a = _profile()
    profile_b = _profile(
        [
            CategorySpec("q_proj", "q_proj", False),
            CategorySpec("k_proj", "k_proj", False),
            CategorySpec("v_proj", "v_proj", True),
            CategorySpec("o_proj", "o_proj", True),
            CategorySpec("gate_proj", "gate_proj", False),
            CategorySpec("up_proj", "up_proj", False),
            CategorySpec("down_proj", "down_proj", True),
        ]
    )
    model = _Model([(8, 16, None)])
    inv_a = _build_inventory(model, profile_a)
    inv_b = _build_inventory(model, profile_b)
    assert inv_a.fingerprint_sha256 != inv_b.fingerprint_sha256


def test_inventory_rejects_shared_target_module_objects():
    adapter = get_model_adapter("generic_decoder")
    model = _Model([(8, 16, None)])
    shared = nn.Linear(8, 8)
    model.model["layers"][0].self_attn["q_proj"] = shared
    model.model["layers"][0].self_attn["v_proj"] = shared
    profile = _profile(
        [
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("v_proj", "v_proj", True),
        ]
    )
    try:
        adapter.discover_target_linears(model, profile)
        raised = False
    except ValueError as exc:
        raised = True
        assert "shared" in str(exc).lower()
    assert raised


def test_qwen_profile_regression_expectations_are_profile_only():
    profile = load_model_profile(str(REPO_ROOT / "mix_bit/configs/models/qwen3_8b.json"))
    assert profile.regression_expectations == {
        "block_count": 36,
        "target_linear_count": 252,
        "category_count": 7,
    }
    # Algorithm helpers must not hardcode these Qwen counts.
    source = (REPO_ROOT / "mix_bit/model_inventory.py").read_text(encoding="utf-8")
    assert "252" not in source
    assert "36" not in source
    adapter_source = (REPO_ROOT / "mix_bit/model_adapter.py").read_text(encoding="utf-8")
    assert "252" not in adapter_source
    assert "q_proj,k_proj,v_proj" not in adapter_source


def test_compute_inventory_fingerprint_is_stable():
    target = TargetLinearSpec(
        module_name="model.layers.0.q_proj",
        category="q_proj",
        module_suffix="q_proj",
        block_index=0,
        in_features=4,
        out_features=4,
        has_bias=False,
        param_count=16,
        transpose=True,
    )
    inv = ModelInventory(
        model_id="toy",
        model_path="toy",
        transformers_model_type="toy",
        resolved_model_class="Module",
        adapter_name="generic_decoder",
        model_profile_sha256="b" * 64,
        category_order=("q_proj",),
        block_count=1,
        targets=(target,),
        total_target_parameters=16,
        fingerprint_sha256="",
    )
    fp1 = compute_inventory_fingerprint(inv)
    fp2 = compute_inventory_fingerprint(inv)
    assert fp1 == fp2
    assert len(fp1) == 64
