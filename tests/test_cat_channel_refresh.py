import logging
from types import SimpleNamespace

import torch
from torch import nn

import train_utils.cat_train_pipeline as pipeline
from train_utils.channel_protection import AdaptiveChannelPlan


def _plan(*, raw_budget, counts, groups_by_category):
    return AdaptiveChannelPlan(
        scope="global",
        axis="input",
        score_metric="channel_weight_abs",
        raw_budget=int(raw_budget),
        used_channels=sum(int(v) for v in counts.values()),
        counts=dict(counts),
        selected_indices={name: list(range(int(count))) for name, count in counts.items()},
        groups=[group for groups in groups_by_category.values() for group in groups],
        signatures=[],
        group_seed_offsets=[],
        artifact={},
        groups_by_category={key: [list(group) for group in groups] for key, groups in groups_by_category.items()},
        signatures_by_category={key: [] for key in groups_by_category},
        group_seed_offsets_by_category={key: [] for key in groups_by_category},
    )


def test_channel_refresh_stats_override_one_shot_activation_stats():
    ref = SimpleNamespace(name="model.layers.0.self_attn.q_proj")
    base = {
        ref.name: {
            "max": torch.tensor([1.0, 1.0]),
            "abs_mean": torch.tensor([1.0, 1.0]),
            "sq_mean": torch.tensor([1.0, 1.0]),
        }
    }
    refreshed = {
        ref.name: {
            "max": torch.tensor([3.0, 4.0]),
            "abs_mean": torch.tensor([5.0, 6.0]),
            "sq_mean": torch.tensor([7.0, 8.0]),
        }
    }
    weight_view, mean_view = pipeline._activation_views_for_refs(
        [ref],
        {"stats_by_linear": base, "channel_refresh_stats": refreshed},
        category="q_proj",
        rank_metric="channel_weight_actmean_abs",
    )
    assert torch.equal(weight_view[ref.name], torch.tensor([3.0, 4.0]))
    assert torch.equal(mean_view[ref.name], torch.tensor([5.0, 6.0]))


def test_dynamic_global_refresh_preserves_remaining_raw_budget(monkeypatch):
    captured = {}

    def fake_rebuild(**kwargs):
        captured["raw_budget"] = int(kwargs["raw_budget"])
        captured["future_categories"] = tuple(kwargs["future_categories"])
        return None, kwargs["activation_runtime"]

    monkeypatch.setattr(pipeline, "_build_dynamic_global_channel_plan", fake_rebuild)
    current_plan = _plan(
        raw_budget=20,
        counts={"q0": 4, "k0": 8},
        groups_by_category={"q_proj": [["q0"]], "k_proj": [["k0"]]},
    )
    cat_args = SimpleNamespace(
        channel_refresh_after_category=True,
        channel_scope="global",
        channel_min_per_layer=0,
        allow_tail_group=True,
        channel_mlp_fuse_weights=(1.0, 1.0, 1.0),
    )
    _runtime, next_plan, remaining_budget = pipeline._refresh_channel_protection_after_category(
        model=nn.Linear(2, 2),
        current_category="q_proj",
        current_category_idx=0,
        active_categories=("q_proj", "k_proj"),
        cat_args=cat_args,
        resolved_category_cfgs={},
        transpose_modules=(),
        only_decoder_projections=True,
        compression_categories=("q_proj", "k_proj"),
        target_layers="all",
        skip_layer_keys=set(),
        activation_runtime=None,
        global_adaptive_plan=current_plan,
        global_raw_budget_value=20,
        resolved_channel_mode="channel",
        resolved_channel_rank_metric="channel_weight_abs",
        resolved_channel_mlp_rank_metric="none",
        channel_axis="input",
        category_channel_protect_count={"q_proj": 0, "k_proj": 0},
        linear_group_size=32,
        plan_is_main=True,
        plan_world_size=1,
        run_output_dir=".",
        logger=logging.getLogger("test_channel_refresh"),
    )
    assert next_plan is None
    assert remaining_budget == 16
    assert captured["raw_budget"] == 16
    assert captured["future_categories"] == ("k_proj",)


def test_channel_refresh_flag_false_is_strict_noop():
    plan = _plan(
        raw_budget=20,
        counts={"q0": 4, "k0": 8},
        groups_by_category={"q_proj": [["q0"]], "k_proj": [["k0"]]},
    )
    runtime = {"sentinel": object()}
    cat_args = SimpleNamespace(channel_refresh_after_category=False)
    out_runtime, out_plan, out_budget = pipeline._refresh_channel_protection_after_category(
        model=nn.Linear(2, 2),
        current_category="q_proj",
        current_category_idx=0,
        active_categories=("q_proj", "k_proj"),
        cat_args=cat_args,
        resolved_category_cfgs={},
        transpose_modules=(),
        only_decoder_projections=True,
        compression_categories=("q_proj", "k_proj"),
        target_layers="all",
        skip_layer_keys=set(),
        activation_runtime=runtime,
        global_adaptive_plan=plan,
        global_raw_budget_value=20,
        resolved_channel_mode="channel",
        resolved_channel_rank_metric="channel_weight_abs",
        resolved_channel_mlp_rank_metric="none",
        channel_axis="input",
        category_channel_protect_count={},
        linear_group_size=32,
        plan_is_main=True,
        plan_world_size=1,
        run_output_dir=".",
        logger=logging.getLogger("test_channel_refresh"),
    )
    assert out_runtime is runtime
    assert out_plan is plan
    assert out_budget == 20
