import json

import pytest
import torch

from train_utils.cat_data_prep import (
    LinearPrepRef,
    materialize_prepared_group_data,
    prepare_group_linear_entries,
)
from train_utils.channel_protection import (
    CHANNEL_ALLOCATION_ARTIFACT_FILENAME,
    ChannelLinearSpec,
    allocate_codebook_aligned_counts,
    check_vae_part_legality,
    compressed_features_after_protection,
    group_adaptive_inventory,
    group_layer_scope_inventory,
    is_legal_protect_count,
    layer_scope_group_seed_offsets,
    adaptive_scope_group_seed_offsets,
    legal_protect_counts,
    resolve_adaptive_channel_plan,
    select_channel_indices,
    stack_compatible_signature,
    validate_adaptive_channel_tail_policy,
    validate_global_channel_runtime,
    vae_group_shuffle_seed,
    vae_layout_rows_cols,
)
from train_utils.config import parse_cat_cli
from train_utils.config.configs import VAECoreConfig


def _spec(
    name,
    *,
    in_features=16,
    out_features=8,
    codebook_dim=4,
    transpose=False,
    intra_parallel=(1, 1),
    ref_position=0,
    scores=None,
    axis="input",
):
    return ChannelLinearSpec(
        name=name,
        in_features=int(in_features),
        out_features=int(out_features),
        codebook_dim=int(codebook_dim),
        transpose=bool(transpose),
        intra_parallel=tuple(intra_parallel),
        ref_position=int(ref_position),
        scores=scores,
        axis=str(axis),
    )


def _legacy_layer_groups(names, group_size, allow_tail_group):
    planned = list(names)
    if not allow_tail_group:
        planned = planned[: (len(planned) // int(group_size)) * int(group_size)]
    return [planned[i : i + int(group_size)] for i in range(0, len(planned), int(group_size))]


def _materialize_one(weight, *, name, protect_count, codebook_dim, intra_parallel, transpose=False, axis="input"):
    ref = LinearPrepRef(
        name=name,
        weight=weight,
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
        transpose=bool(transpose),
    )
    plan = None
    if int(protect_count) > 0:
        plan = {name: torch.arange(int(protect_count), dtype=torch.long)}
    entries = prepare_group_linear_entries(
        group_refs=[ref],
        activation_weight_by_linear=None,
        channel_protect_count=int(protect_count),
        channel_axis=axis,
        recon_loss_type="mse",
        channel_plan=plan,
        apply_outlier_channel_removal=True,
    )
    return materialize_prepared_group_data(
        prepared_entries=entries,
        intra_parallel=intra_parallel,
        codebook_dim=int(codebook_dim),
        batch_size=8,
        normalize_weight=False,
        recon_loss_type="mse",
        train_device="cpu",
    )


def test_vae_core_config_accepts_intra_parallel_default_1x1():
    core = VAECoreConfig()
    core.validate()
    assert core.intra_parallel == (1, 1)


def test_cat_cli_resolves_intra_parallel_category_override():
    cfg = parse_cat_cli(
        [
            "--model_path",
            "dummy-model",
            "--compression_categories",
            "q_proj,k_proj",
            "--target_layers",
            "all",
            "--intra_parallel",
            "default=1x1,cat:k_proj=2x1",
        ]
    )
    q_vae, _ = cfg.resolve_category_config("q_proj")
    k_vae, _ = cfg.resolve_category_config("k_proj")
    assert q_vae.core.intra_parallel == (1, 1)
    assert k_vae.core.intra_parallel == (2, 1)


def test_part_legality_matches_materializer_for_legal_and_illegal_k():
    weight = torch.randn(6, 16)
    legal_kwargs = dict(
        in_features=16,
        out_features=6,
        protect_count=4,
        axis="input",
        transpose=False,
        intra_parallel=(1, 1),
        codebook_dim=8,
    )
    check_vae_part_legality(**legal_kwargs)
    assert is_legal_protect_count(**legal_kwargs)
    result = _materialize_one(
        weight,
        name="q.0",
        protect_count=4,
        codebook_dim=8,
        intra_parallel=(1, 1),
    )
    assert int(result.split_metas[0].compressed_in_features) == 12

    illegal_kwargs = dict(legal_kwargs)
    illegal_kwargs["protect_count"] = 1
    assert not is_legal_protect_count(**illegal_kwargs)
    with pytest.raises(ValueError):
        check_vae_part_legality(**illegal_kwargs)


def test_intra_parallel_2x1_rejects_row_not_divisible_and_n_part():
    # axis=output changes rows. K=0 leaves rows=15, 15 % 2 != 0.
    row_kwargs = dict(
        in_features=10,
        out_features=15,
        protect_count=0,
        axis="output",
        transpose=False,
        intra_parallel=(2, 1),
        codebook_dim=5,
    )
    assert not is_legal_protect_count(**row_kwargs)
    with pytest.raises(ValueError):
        check_vae_part_legality(**row_kwargs)
    with pytest.raises(ValueError):
        _materialize_one(
            torch.randn(15, 10),
            name="row_bad",
            protect_count=0,
            codebook_dim=5,
            intra_parallel=(2, 1),
            axis="output",
        )

    # K=5 is a codebook multiple and makes rows=10 divisible by 2; N_part=5*10=50, 50%5==0.
    ok_kwargs = dict(row_kwargs)
    ok_kwargs["protect_count"] = 5
    assert is_legal_protect_count(**ok_kwargs)
    check_vae_part_legality(**ok_kwargs)
    _materialize_one(
        torch.randn(15, 10),
        name="row_ok",
        protect_count=5,
        codebook_dim=5,
        intra_parallel=(2, 1),
        axis="output",
    )

    npart_kwargs = dict(
        in_features=16,
        out_features=6,
        protect_count=0,
        axis="input",
        transpose=False,
        intra_parallel=(2, 1),
        codebook_dim=10,
    )
    assert not is_legal_protect_count(**npart_kwargs)
    with pytest.raises(ValueError):
        check_vae_part_legality(**npart_kwargs)
    with pytest.raises(ValueError):
        _materialize_one(
            torch.randn(6, 16),
            name="npart_bad",
            protect_count=0,
            codebook_dim=10,
            intra_parallel=(2, 1),
        )


def test_intra_parallel_1x2_rejects_col_not_divisible_and_n_part():
    col_kwargs = dict(
        in_features=15,
        out_features=10,
        protect_count=0,
        axis="input",
        transpose=False,
        intra_parallel=(1, 2),
        codebook_dim=5,
    )
    assert not is_legal_protect_count(**col_kwargs)
    with pytest.raises(ValueError):
        check_vae_part_legality(**col_kwargs)
    with pytest.raises(ValueError):
        _materialize_one(
            torch.randn(10, 15),
            name="col_bad",
            protect_count=0,
            codebook_dim=5,
            intra_parallel=(1, 2),
        )

    ok_kwargs = dict(col_kwargs)
    ok_kwargs["protect_count"] = 5
    assert is_legal_protect_count(**ok_kwargs)
    _materialize_one(
        torch.randn(10, 15),
        name="col_ok",
        protect_count=5,
        codebook_dim=5,
        intra_parallel=(1, 2),
    )

    npart_kwargs = dict(
        in_features=20,
        out_features=4,
        protect_count=0,
        axis="input",
        transpose=False,
        intra_parallel=(1, 2),
        codebook_dim=12,
    )
    assert not is_legal_protect_count(**npart_kwargs)
    with pytest.raises(ValueError):
        check_vae_part_legality(**npart_kwargs)


def test_different_legal_k_same_original_shape_go_to_different_buckets_and_materialize():
    scores_a = torch.tensor([10.0, 9.0, 8.0, 7.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scores_b = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    specs = [
        _spec("layer0.q_proj", scores=scores_a, ref_position=0),
        _spec("layer1.q_proj", scores=scores_b, ref_position=1),
    ]
    counts = allocate_codebook_aligned_counts(specs, raw_budget=12, min_per_layer=0)
    assert counts["layer0.q_proj"] != counts["layer1.q_proj"]
    assert counts["layer0.q_proj"] % 4 == 0
    assert counts["layer1.q_proj"] % 4 == 0

    groups = group_adaptive_inventory(specs, counts, linear_group_size=36)
    assert len(groups) == 2
    assert {name for group in groups for name in group} == {"layer0.q_proj", "layer1.q_proj"}
    assert groups[0] != groups[1]

    for spec in specs:
        _materialize_one(
            torch.randn(spec.out_features, spec.in_features),
            name=spec.name,
            protect_count=counts[spec.name],
            codebook_dim=spec.codebook_dim,
            intra_parallel=spec.intra_parallel,
            transpose=spec.transpose,
        )


def test_different_k_same_signature_share_bucket_and_different_signature_never_share():
    # in=16 K=4 and in=20 K=8 both give compressed_in=12, so same signature.
    spec_a = _spec("a", in_features=16, out_features=8, codebook_dim=4, ref_position=0)
    spec_b = _spec("b", in_features=20, out_features=8, codebook_dim=4, ref_position=1)
    spec_c = _spec("c", in_features=16, out_features=8, codebook_dim=4, ref_position=2)
    counts = {"a": 4, "b": 8, "c": 8}
    sig_a = stack_compatible_signature(spec_a, protect_count=4)
    sig_b = stack_compatible_signature(spec_b, protect_count=8)
    sig_c = stack_compatible_signature(spec_c, protect_count=8)
    assert sig_a == sig_b
    assert sig_a != sig_c

    groups = group_adaptive_inventory([spec_a, spec_b, spec_c], counts, linear_group_size=36)
    grouped = {frozenset(group) for group in groups}
    assert frozenset({"a", "b"}) in grouped
    assert frozenset({"c"}) in grouped
    for group in groups:
        signatures = {
            stack_compatible_signature(
                next(spec for spec in (spec_a, spec_b, spec_c) if spec.name == name),
                protect_count=counts[name],
            )
            for name in group
        }
        assert len(signatures) == 1


def test_bucket_order_is_first_appearance_and_within_bucket_keeps_layer_order():
    specs = [
        _spec("l0", in_features=16, ref_position=0),
        _spec("l1", in_features=16, ref_position=1),
        _spec("l2", in_features=16, ref_position=2),
        _spec("l3", in_features=16, ref_position=3),
    ]
    counts = {"l0": 4, "l1": 8, "l2": 4, "l3": 8}
    groups = group_adaptive_inventory(specs, counts, linear_group_size=36)
    assert groups == [["l0", "l2"], ["l1", "l3"]]
    again = group_adaptive_inventory(specs, counts, linear_group_size=36)
    assert again == groups


def test_allocator_and_grouping_are_deterministic():
    scores = [
        torch.linspace(16.0, 1.0, 16),
        torch.linspace(8.0, 0.5, 16),
        torch.arange(16.0, 0.0, -1.0),
    ]
    specs = [
        _spec(f"l{i}", scores=score, ref_position=i)
        for i, score in enumerate(scores)
    ]
    first = allocate_codebook_aligned_counts(specs, raw_budget=12, min_per_layer=0)
    second = allocate_codebook_aligned_counts(specs, raw_budget=12, min_per_layer=0)
    assert first == second
    groups_a = group_adaptive_inventory(specs, first, linear_group_size=2)
    groups_b = group_adaptive_inventory(specs, second, linear_group_size=2)
    assert groups_a == groups_b


def test_linear_group_size_is_max_chunk_and_keeps_short_bucket_tails():
    bucket_a = [_spec(f"a{i}", in_features=16, ref_position=i) for i in range(20)]
    bucket_b = [_spec(f"b{i}", in_features=20, ref_position=100 + i) for i in range(16)]
    counts = {spec.name: 4 for spec in bucket_a}
    counts.update({spec.name: 4 for spec in bucket_b})
    groups = group_adaptive_inventory(bucket_a + bucket_b, counts, linear_group_size=36)
    assert len(groups) == 2
    assert [spec.name for spec in bucket_a] in groups
    assert [spec.name for spec in bucket_b] in groups
    assert all(len(group) <= 36 for group in groups)


def test_adaptive_scope_requires_allow_tail_group_when_budget_positive():
    with pytest.raises(ValueError, match="allow_tail_group"):
        validate_adaptive_channel_tail_policy(
            scope="category",
            budget=8,
            allow_tail_group=False,
        )
    with pytest.raises(ValueError, match="allow_tail_group"):
        validate_adaptive_channel_tail_policy(
            scope="global",
            budget=1,
            allow_tail_group=False,
        )
    validate_adaptive_channel_tail_policy(scope="category", budget=0, allow_tail_group=False)
    validate_adaptive_channel_tail_policy(scope="global", budget=0, allow_tail_group=False)
    validate_adaptive_channel_tail_policy(scope="layer", budget=32, allow_tail_group=False)


def test_layer_scope_grouping_matches_pre_refactor_recipe():
    names = [f"layers.{i}.q_proj" for i in range(40)]
    actual = group_layer_scope_inventory(
        names,
        linear_group_size=36,
        allow_tail_group=True,
    )
    assert actual == _legacy_layer_groups(names, 36, True)
    dropped = group_layer_scope_inventory(
        names,
        linear_group_size=36,
        allow_tail_group=False,
    )
    assert dropped == _legacy_layer_groups(names, 36, False)


def test_layer_scope_group_seeds_use_inventory_start_not_group_idx():
    names = [f"layers.{i}.q_proj" for i in range(80)]
    groups = group_layer_scope_inventory(
        names,
        linear_group_size=36,
        allow_tail_group=True,
    )
    offsets = layer_scope_group_seed_offsets(
        names,
        linear_group_size=36,
        allow_tail_group=True,
    )
    assert [len(group) for group in groups] == [36, 36, 8]
    assert offsets == [0, 36, 72]
    assert offsets != list(range(len(groups)))
    assert vae_group_shuffle_seed(42, 1, offsets[1]) == 42 + 100000 + 36
    dropped_offsets = layer_scope_group_seed_offsets(
        names,
        linear_group_size=36,
        allow_tail_group=False,
    )
    assert dropped_offsets == [0, 36]


def test_adaptive_scope_group_seeds_are_stable_ordinals():
    first = adaptive_scope_group_seed_offsets(3)
    second = adaptive_scope_group_seed_offsets(3)
    assert first == [0, 1, 2]
    assert first == second
    assert vae_group_shuffle_seed(7, 0, first[2]) == 7 + 2


def test_select_channel_indices_stable_tie_prefers_smaller_index():
    scores = torch.tensor([1.0, 5.0, 5.0, 3.0], dtype=torch.float32)
    idx = select_channel_indices(scores, 2)
    assert idx.tolist() == [1, 2]


def test_legal_counts_are_codebook_aligned_and_part_legal():
    spec = _spec("x", in_features=16, out_features=8, codebook_dim=4, intra_parallel=(1, 1))
    assert legal_protect_counts(spec) == (0, 4, 8, 12)


def test_old_cat_parser_rejects_removed_residual_flags():
    from train_utils.config.cli import parse_cat_cli

    with pytest.raises(SystemExit):
        parse_cat_cli(["--outlier_residual_top_p", "default=0.01"])


def test_compressed_layout_uses_transpose_before_parts():
    cout, cin = compressed_features_after_protection(
        in_features=16,
        out_features=8,
        protect_count=4,
        axis="input",
    )
    assert (cout, cin) == (8, 12)
    assert vae_layout_rows_cols(compressed_out=8, compressed_in=12, transpose=False) == (8, 12)
    assert vae_layout_rows_cols(compressed_out=8, compressed_in=12, transpose=True) == (12, 8)


def test_cat_parser_rejects_global_with_output_axis():
    from train_utils.config.cli import parse_cat_cli

    with pytest.raises(SystemExit):
        parse_cat_cli(
            [
                "--model_path",
                "dummy-model",
                "--compression_categories",
                "q_proj",
                "--channel_scope",
                "global",
                "--channel_protect_mode",
                "channel",
                "--channel_axis",
                "output",
                "--channel_protect_count",
                "0.001",
            ]
        )


def test_global_axis_output_hard_errors_before_planner():
    scored = [
        _spec(
            "a",
            in_features=16,
            scores=torch.arange(16, 0, -1, dtype=torch.float32),
            ref_position=0,
        )
    ]
    score_calls = {"n": 0}

    def _score(specs, _weight, _mean):
        score_calls["n"] += 1
        return list(specs)

    with pytest.raises(ValueError, match="channel_scope=global"):
        validate_global_channel_runtime(
            channel_scope="global",
            channel_protect_mode="channel",
            channel_axis="output",
        )
    with pytest.raises(ValueError, match="channel_scope=global"):
        resolve_adaptive_channel_plan(
            scored,
            raw_budget=4,
            min_per_layer=0,
            linear_group_size=2,
            metric="channel_weight_abs",
            axis="output",
            scope="global",
            is_main=True,
            world_size=1,
            broadcast_fn=lambda payload: payload,
            activation_view_fn=lambda _specs: (None, None),
            score_fn=_score,
        )
    assert score_calls["n"] == 0


def _scored_specs():
    return [
        _spec(
            "layers.0.q_proj",
            in_features=16,
            scores=torch.arange(16, 0, -1, dtype=torch.float32),
            ref_position=0,
            axis="input",
        ),
        _spec(
            "layers.1.q_proj",
            in_features=16,
            scores=torch.arange(1, 17, dtype=torch.float32),
            ref_position=1,
            axis="input",
        ),
    ]


def test_adaptive_plan_single_rank_skips_broadcast(tmp_path):
    def _boom(_payload):
        raise AssertionError("single-card must not broadcast")

    plan = resolve_adaptive_channel_plan(
        _scored_specs(),
        raw_budget=8,
        min_per_layer=0,
        linear_group_size=2,
        metric="channel_weight_abs",
        axis="input",
        scope="category",
        category="q_proj",
        is_main=True,
        world_size=1,
        broadcast_fn=_boom,
        activation_view_fn=lambda _specs: (None, None),
        score_fn=lambda specs, _w, _m: list(specs),
        run_output_dir=str(tmp_path),
    )
    assert plan.groups
    assert plan.artifact["used_channels"] > 0


def test_adaptive_plan_only_rank0_scores_and_groups_match():
    score_calls = {"n": 0}
    view_calls = {"n": 0}
    stored = {}

    def _views(_specs):
        view_calls["n"] += 1
        return None, None

    def _score(specs, _weight, _mean):
        score_calls["n"] += 1
        return list(specs)

    def _broadcast(payload):
        if payload is not None:
            stored["payload"] = payload
            return payload
        return stored["payload"]

    specs = _scored_specs()
    plan0 = resolve_adaptive_channel_plan(
        specs,
        raw_budget=8,
        min_per_layer=0,
        linear_group_size=2,
        metric="channel_weight_abs",
        axis="input",
        scope="category",
        category="q_proj",
        is_main=True,
        world_size=2,
        broadcast_fn=_broadcast,
        activation_view_fn=_views,
        score_fn=_score,
    )
    plan1 = resolve_adaptive_channel_plan(
        specs,
        raw_budget=8,
        min_per_layer=0,
        linear_group_size=2,
        metric="channel_weight_abs",
        axis="input",
        scope="category",
        category="q_proj",
        is_main=False,
        world_size=2,
        broadcast_fn=_broadcast,
        activation_view_fn=_views,
        score_fn=_score,
    )
    assert view_calls["n"] == 1
    assert score_calls["n"] == 1
    assert plan0.groups == plan1.groups
    assert plan0.counts == plan1.counts
    assert plan0.selected_indices == plan1.selected_indices
    assert plan0.signatures == plan1.signatures
    assert plan0.raw_budget == plan1.raw_budget
    assert plan0.used_channels == plan1.used_channels
    assert plan0.artifact == plan1.artifact
    assert plan0.group_seed_offsets == plan1.group_seed_offsets == [0]


def test_resolve_adaptive_plan_writes_main_chain_artifact(tmp_path):
    plan = resolve_adaptive_channel_plan(
        _scored_specs(),
        raw_budget=8,
        min_per_layer=0,
        linear_group_size=2,
        metric="channel_weight_abs",
        axis="input",
        scope="category",
        category="q_proj",
        is_main=True,
        world_size=1,
        broadcast_fn=lambda payload: payload,
        activation_view_fn=lambda _specs: (None, None),
        score_fn=lambda specs, _w, _m: list(specs),
        run_output_dir=str(tmp_path),
    )
    artifact_path = tmp_path / CHANNEL_ALLOCATION_ARTIFACT_FILENAME
    assert artifact_path.is_file()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    recorded = payload["plans"]["q_proj"]
    assert recorded == plan.artifact
    assert recorded["raw_budget"] == 8
    assert recorded["used_channels"] == plan.used_channels
    assert recorded["score_metric"] == "channel_weight_abs"
    assert "achieved_channel_ratio" in recorded
    assert "achieved_parameter_ratio" in recorded
    linear = recorded["per_linear"]["layers.0.q_proj"]
    assert "count" in linear
    assert "indices" in linear
    assert "codebook_dim" in linear
    assert "compressed_out" in linear
    assert "compressed_in" in linear
    assert "intra_parallel" in linear
    assert "n_part" in linear
    assert "group_signature" in linear
    assert "group_id" in linear
