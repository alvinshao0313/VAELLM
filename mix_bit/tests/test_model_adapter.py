from __future__ import annotations

import torch
from torch import nn

from mix_bit.model_adapter import get_model_adapter
from mix_bit.schema import (
    CandidateTrainingSpec,
    CategorySpec,
    ModelProfile,
)


def _profile(
    *,
    categories: list[CategorySpec],
    layer_index_patterns: tuple[str, ...] = (r"(?:^|\.)model\.layers\.(\d+)\.",),
) -> ModelProfile:
    return ModelProfile(
        model_id="toy",
        model_path="toy",
        adapter="generic_decoder",
        only_decoder_projections=True,
        candidate_training=CandidateTrainingSpec(linear_group_size="all", allow_tail_group=True),
        layer_index_patterns=layer_index_patterns,
        categories=tuple(categories),
        regression_expectations={},
    )


class _QwenLikeBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden, hidden, bias=True),
                "k_proj": nn.Linear(hidden, hidden // 2, bias=False),
                "v_proj": nn.Linear(hidden, hidden, bias=True),
                "o_proj": nn.Linear(hidden, hidden, bias=False),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden, intermediate, bias=False),
                "up_proj": nn.Linear(hidden, intermediate, bias=False),
                "down_proj": nn.Linear(intermediate, hidden, bias=False),
            }
        )


class _QwenLikeModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = 8, intermediate: int = 16):
        super().__init__()
        self.model = nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [_QwenLikeBlock(hidden, intermediate) for _ in range(n_layers)]
                )
            }
        )


class _AltBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.wq = nn.Linear(hidden, hidden)
        self.wk = nn.Linear(hidden, hidden)
        self.wv = nn.Linear(hidden, hidden)
        self.wo = nn.Linear(hidden, hidden)
        self.w1 = nn.Linear(hidden, hidden * 2)
        self.w3 = nn.Linear(hidden, hidden * 2)
        self.w2 = nn.Linear(hidden * 2, hidden)


class _AltModel(nn.Module):
    def __init__(self, n_layers: int = 3, hidden: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([_AltBlock(hidden) for _ in range(n_layers)])


def test_generic_adapter_discovers_qwen_like_names():
    adapter = get_model_adapter("generic_decoder")
    model = _QwenLikeModel(n_layers=2)
    profile = _profile(
        categories=[
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("k_proj", "k_proj", False),
            CategorySpec("v_proj", "v_proj", True),
            CategorySpec("o_proj", "o_proj", True),
            CategorySpec("gate_proj", "gate_proj", False),
            CategorySpec("up_proj", "up_proj", False),
            CategorySpec("down_proj", "down_proj", True),
        ]
    )
    targets = adapter.discover_target_linears(model, profile)
    assert len(targets) == 14
    assert [t.category for t in targets[:7]] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert targets[0].module_name == "model.layers.0.self_attn.q_proj"
    assert targets[0].block_index == 0
    assert targets[0].transpose is True
    assert targets[1].has_bias is False
    assert targets[1].out_features == 4


def test_generic_adapter_discovers_alternate_wq_wk_wv_names_from_profile():
    adapter = get_model_adapter("generic_decoder")
    model = _AltModel(n_layers=2)
    profile = _profile(
        categories=[
            CategorySpec("attn_q", "wq", True),
            CategorySpec("attn_k", "wk", False),
            CategorySpec("attn_v", "wv", True),
            CategorySpec("attn_o", "wo", True),
            CategorySpec("ffn_up", "w1", False),
            CategorySpec("ffn_gate", "w3", False),
            CategorySpec("ffn_down", "w2", True),
        ],
        layer_index_patterns=(r"(?:^|\.)blocks\.(\d+)\.",),
    )
    targets = adapter.discover_target_linears(model, profile)
    assert len(targets) == 14
    assert targets[0].module_suffix == "wq"
    assert targets[0].category == "attn_q"
    assert targets[0].module_name == "blocks.0.wq"


def test_inventory_rejects_shared_target_module_objects():
    adapter = get_model_adapter("generic_decoder")
    model = _QwenLikeModel(n_layers=1)
    shared = nn.Linear(8, 8)
    model.model["layers"][0].self_attn["q_proj"] = shared
    model.model["layers"][0].self_attn["k_proj"] = shared
    profile = _profile(
        categories=[
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("k_proj", "k_proj", False),
        ]
    )
    try:
        adapter.discover_target_linears(model, profile)
        raised = False
    except ValueError as exc:
        raised = True
        assert "shared" in str(exc).lower()
    assert raised
