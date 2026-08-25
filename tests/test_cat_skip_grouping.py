from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_utils.activation_utils as activation_utils
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.activation_utils import ActivationCalibrationCache, collect_mlp_block_activation_stats
from train_utils.cat_train_pipeline import _filter_eligible_vae_refs
from train_utils.cat_train_runtime import normalize_cat_runtime_vae_original_state
from train_utils.mlp_channel_selection import (
    MLP_CATEGORIES,
    MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS,
    build_mlp_aligned_plans_all_layers,
    compute_mlp_intermediate_scores,
    select_mlp_aligned_activation_weighted_channels,
)
from train_utils.utils import LinearRef


def _make_refs(count: int, category: str = "q_proj"):
    return [
        (
            layer_idx,
            LinearRef(
                name=f"model.layers.{layer_idx}.self_attn.{category}",
                module=nn.Linear(2, 2),
                category=category,
                transpose=False,
            ),
        )
        for layer_idx in range(count)
    ]


def _planned_group_sizes(refs_sorted, skip_layer_keys, *, group_size: int, allow_tail: bool):
    eligible = _filter_eligible_vae_refs(refs_sorted, skip_layer_keys)
    if allow_tail:
        planned = eligible
    else:
        planned_count = (len(eligible) // int(group_size)) * int(group_size)
        planned = eligible[:planned_count]
    return [
        len(planned[start:start + int(group_size)])
        for start in range(0, len(planned), int(group_size))
    ]


def test_skip_filtering_precedes_tail_grouping():
    refs36 = _make_refs(36)
    skip4 = {(0, "q_proj"), (1, "q_proj"), (2, "q_proj"), (3, "q_proj")}
    assert len(_filter_eligible_vae_refs(refs36, skip4)) == 32
    assert _planned_group_sizes(refs36, skip4, group_size=36, allow_tail=True) == [32]
    assert _planned_group_sizes(refs36, skip4, group_size=36, allow_tail=False) == []

    refs40 = _make_refs(40)
    assert len(_filter_eligible_vae_refs(refs40, skip4)) == 36
    assert _planned_group_sizes(refs40, skip4, group_size=16, allow_tail=True) == [16, 16, 4]
    assert _planned_group_sizes(refs40, skip4, group_size=16, allow_tail=False) == [16, 16]


def test_skip_refs_are_not_eligible_for_replacement():
    refs = _make_refs(5)
    skip = {(1, "q_proj"), (3, "q_proj")}
    eligible = _filter_eligible_vae_refs(refs, skip)

    assert [layer_idx for layer_idx, _ in eligible] == [0, 2, 4]
    assert isinstance(refs[1][1].module, nn.Linear)
    assert isinstance(refs[3][1].module, nn.Linear)


class _TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 5, bias=False)
        self.up_proj = nn.Linear(4, 5, bias=False)
        self.down_proj = nn.Linear(5, 4, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _TinyMlp()

    def forward(self, x):
        return self.mlp(x)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([_TinyLayer()]))
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids):
        x = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).to(dtype=torch.float32)
        return self.model.layers[0](x)


def _make_mlp_model():
    model = _TinyModel()
    mlp = model.model.layers[0].mlp
    with torch.no_grad():
        mlp.up_proj.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
        ]))
        mlp.gate_proj.weight.copy_(torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [2.0, 2.0, 0.0, 0.0],
        ]))
        mlp.down_proj.weight.copy_(torch.tensor([
            [1.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 5.0],
        ]))
    return model


def _block_stats():
    return {
        "abs_mean_in": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "sq_mean_in": torch.tensor([1.0, 4.0, 9.0, 16.0]),
        "a_in": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "abs_mean_mid": torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0]),
        "sq_mean_mid": torch.tensor([1.0, 9.0, 4.0, 16.0, 25.0]),
        "a_mid": torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0]),
        "num_tokens": torch.tensor(8),
    }


def _expected_subset_indices(detail, categories, fuse_weights, k):
    score_by_cat = {
        "up_proj": detail.score_up / (detail.score_up.mean() + 1e-8),
        "gate_proj": detail.score_gate / (detail.score_gate.mean() + 1e-8),
        "down_proj": detail.score_down / (detail.score_down.mean() + 1e-8),
    }
    weight_by_cat = {
        "up_proj": float(fuse_weights[0]),
        "gate_proj": float(fuse_weights[1]),
        "down_proj": float(fuse_weights[2]),
    }
    denom = sum(weight_by_cat[cat] for cat in categories)
    fused = sum((weight_by_cat[cat] / denom) * score_by_cat[cat] for cat in categories)
    _, idx = torch.topk(fused, k=int(k), largest=True, sorted=False)
    return torch.sort(idx.to(dtype=torch.long)).values


def test_mlp_aligned_all_eligible_matches_existing_formula():
    model = _make_mlp_model()
    stats = _block_stats()
    fuse_weights = (2.0, 3.0, 5.0)
    expected, _ = select_mlp_aligned_activation_weighted_channels(
        model.model.layers[0].mlp.up_proj.weight,
        model.model.layers[0].mlp.gate_proj.weight,
        model.model.layers[0].mlp.down_proj.weight,
        stats,
        rank_metric=MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS,
        protect_count=2,
        fuse_weights=fuse_weights,
    )

    plan, summary = build_mlp_aligned_plans_all_layers(
        model=model,
        stats_by_mlp_block={0: stats},
        protect_count=2,
        fuse_weights=fuse_weights,
        rank_metric=MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS,
        skip_layer_keys=set(),
    )

    assert set(plan) == {f"model.layers.0.mlp.{cat}" for cat in MLP_CATEGORIES}
    assert all(torch.equal(indices, expected) for indices in plan.values())
    assert summary[0]["protected_count"] == 2


@pytest.mark.parametrize(
    ("skip", "eligible"),
    [
        ({"gate_proj"}, ["up_proj", "down_proj"]),
        ({"up_proj"}, ["gate_proj", "down_proj"]),
        ({"down_proj"}, ["gate_proj", "up_proj"]),
        ({"gate_proj", "up_proj"}, ["down_proj"]),
    ],
)
def test_mlp_aligned_partial_skip_only_scores_eligible_branches(skip, eligible):
    model = _make_mlp_model()
    stats = _block_stats()
    fuse_weights = (2.0, 3.0, 5.0)
    mlp = model.model.layers[0].mlp
    detail = compute_mlp_intermediate_scores(
        mlp.up_proj.weight,
        mlp.gate_proj.weight,
        mlp.down_proj.weight,
        stats,
        rank_metric=MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS,
        fuse_weights=fuse_weights,
    )
    expected = _expected_subset_indices(detail, eligible, fuse_weights, k=2)

    plan, _summary = build_mlp_aligned_plans_all_layers(
        model=model,
        stats_by_mlp_block={0: stats},
        protect_count=2,
        fuse_weights=fuse_weights,
        rank_metric=MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS,
        skip_layer_keys={(0, cat) for cat in skip},
    )

    assert set(plan) == {f"model.layers.0.mlp.{cat}" for cat in eligible}
    assert all(torch.equal(indices, expected) for indices in plan.values())


def test_mlp_aligned_all_skipped_generates_no_plan():
    model = _make_mlp_model()

    plan, summary = build_mlp_aligned_plans_all_layers(
        model=model,
        stats_by_mlp_block={0: _block_stats()},
        protect_count=2,
        fuse_weights=(1.0, 1.0, 1.0),
        rank_metric=MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS,
        skip_layer_keys={(0, cat) for cat in MLP_CATEGORIES},
    )

    assert plan == {}
    assert summary == {}


def test_mlp_block_activation_stats_only_skip_full_mlp_layer(monkeypatch):
    model = _make_mlp_model()
    monkeypatch.setattr(activation_utils, "_cache_matches", lambda *args, **kwargs: True)
    cache = ActivationCalibrationCache(
        dataset="openorca=1.0",
        model_path="toy",
        nsamples=1,
        seqlen=4,
        seed=7,
        input_ids=[torch.tensor([[0, 1, 2, 3]], dtype=torch.long)],
    )

    partial_stats, _ = collect_mlp_block_activation_stats(
        model=model,
        layer_indices=[0],
        model_path="toy",
        access_token=None,
        dataset="openorca=1.0",
        nsamples=1,
        seqlen=4,
        seed=7,
        device="cpu",
        cache=cache,
        skip_layer_keys={(0, "gate_proj")},
    )
    full_skip_stats, _ = collect_mlp_block_activation_stats(
        model=model,
        layer_indices=[0],
        model_path="toy",
        access_token=None,
        dataset="openorca=1.0",
        nsamples=1,
        seqlen=4,
        seed=7,
        device="cpu",
        cache=cache,
        skip_layer_keys={(0, cat) for cat in MLP_CATEGORIES},
    )

    assert set(partial_stats) == {0}
    assert full_skip_stats == {}


def test_missing_eligible_mlp_plan_is_fail_fast():
    plan = {"model.layers.0.mlp.up_proj": torch.tensor([1, 3], dtype=torch.long)}

    with pytest.raises(KeyError):
        _ = plan["model.layers.0.mlp.down_proj"]


def _make_decoder():
    return Decoder(
        in_dim=9,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )


def _make_vae_linear(*, original_weight=True, always_use_original=False, protect_original_weight=False):
    bits = torch.tensor(
        [
            [[True, False, True, False, True, False, True, False, True]],
            [[False, True, False, True, False, True, False, True, False]],
            [[True, True, False, False, True, True, False, False, True]],
            [[False, False, True, True, False, False, True, True, False]],
        ],
        dtype=torch.bool,
    )
    weight = torch.randn(4, 4) if original_weight else None
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=weight,
        vq_weight=bits,
        decoder=_make_decoder(),
        codebook_dim=4,
        transpose=False,
        always_use_original=always_use_original,
        protect_original_weight=protect_original_weight,
    )


def test_normalize_cat_runtime_vae_original_state_strips_inert_original_weight():
    model = nn.Sequential(_make_vae_linear(original_weight=True))

    stripped = normalize_cat_runtime_vae_original_state(model)

    layer = model[0]
    assert stripped == 1
    assert layer.original_weight is None
    assert layer.always_use_original is False
    assert layer.protect_original_weight is False
    assert layer.temporary is True


def test_normalize_cat_runtime_vae_original_state_rejects_legacy_skip_vae():
    model = nn.Sequential(
        _make_vae_linear(
            original_weight=True,
            always_use_original=True,
            protect_original_weight=True,
        )
    )

    with pytest.raises(ValueError, match="Legacy skip-as-VAELinear"):
        normalize_cat_runtime_vae_original_state(model)
