from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from e2e_common.full_lora import finalize_model_level_lora, iter_named_peft_lora_layers
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
import train_utils.cat_after_category_distill as cat_after_distill
from train_utils.cat_after_category_common import (
    get_or_build_cat_projection_name_inventory,
    resolve_canonical_after_category_mode,
    resolve_exact_current_compressed_targets,
    select_compressed_decoder_targets_from_inventory,
    select_remaining_dense_names_from_inventory,
)
from train_utils.config.configs import AuxTrainableConfig
from train_utils.distill_decoder import NamedMainDecoderTarget, finalize_main_decoder_targets
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, build_model_level_param_groups
from train_utils.model_level_trainables import build_model_level_trainable_selection


def _vae_linear(dim: int = 4) -> VAELinear:
    latent_dim = 9
    bits = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )[:dim]
    decoder = Decoder(
        in_dim=latent_dim,
        out_dim=dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    ).to(dtype=torch.float32)
    return VAELinear(
        in_features=dim,
        out_features=dim,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=decoder,
        codebook_dim=dim,
        transpose=False,
    )


class _TinyCatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _vae_linear()
        self.k_proj = _vae_linear()
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.q_proj(x)))


def _clone_tensors(params):
    return [param.detach().clone() for param in params]


def _any_changed(before, params) -> bool:
    return any(not torch.equal(old, param.detach()) for old, param in zip(before, params))


def test_after_category_mode_resolution_uses_canonical_value_directly():
    assert resolve_canonical_after_category_mode(SimpleNamespace(after_category_mode="current_decoder")) == "current_decoder"
    assert resolve_canonical_after_category_mode(SimpleNamespace()) == "none"


def test_exact_current_decoder_target_inventory_rejects_wrong_category_and_non_vae():
    model = _TinyCatModel()
    targets = resolve_exact_current_compressed_targets(
        model,
        category="q_proj",
        target_names=["q_proj"],
    )
    assert targets == (("q_proj", model.q_proj),)

    with pytest.raises(ValueError, match="does not belong"):
        resolve_exact_current_compressed_targets(
            model,
            category="q_proj",
            target_names=["k_proj"],
        )
    with pytest.raises(TypeError, match="must resolve to VAELinear"):
        resolve_exact_current_compressed_targets(
            model,
            category="norm",
            target_names=["norm"],
        )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_exact_current_compressed_targets(
            model,
            category="q_proj",
            target_names=["q_proj", "q_proj"],
        )


def test_current_decoder_selection_has_no_lora_and_optimizer_groups_are_canonical():
    model = _TinyCatModel()
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(
            norm_train_mode="all",
            norm_lr=3e-5,
            lm_head_train_mode="linear",
            lm_head_lr=4e-5,
        ),
        compressed_modules=[("q_proj", model.q_proj)],
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=True,
        train_lora=False,
        freeze=True,
    )
    assert selection.decoder_parameters
    assert selection.lora_parameters == {}
    assert selection.norm_parameters
    assert selection.lm_head_parameters

    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-4,
            weight_decay=0.01,
            decoder_lr=2e-5,
            norm_lr=3e-5,
            lm_head_lr=4e-5,
        ),
        model=selection.peft_model,
    )
    by_name = {group["group_name"]: group for group in groups}
    assert set(by_name) == {"decoder", "norm", "lm_head"}
    assert by_name["decoder"]["lr"] == pytest.approx(2e-5)
    assert by_name["decoder"]["weight_decay"] == 0.0
    assert by_name["norm"]["lr"] == pytest.approx(3e-5)
    assert by_name["norm"]["weight_decay"] == 0.0
    assert by_name["lm_head"]["lr"] == pytest.approx(4e-5)
    assert by_name["lm_head"]["weight_decay"] == 0.0

    selected_ids = {
        id(param)
        for inv in (
            selection.decoder_parameters,
            selection.norm_parameters,
            selection.lm_head_parameters,
        )
        for param in inv.values()
    }
    all_trainable_ids = {id(param) for param in selection.peft_model.parameters() if param.requires_grad}
    assert selected_ids == all_trainable_ids
    assert all(not param.requires_grad for param in model.k_proj.parameters())


def test_current_decoder_one_step_updates_only_selected_decoder_and_aux():
    torch.manual_seed(7)
    model = _TinyCatModel()
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(norm_train_mode="all", lm_head_train_mode="none"),
        compressed_modules=[("q_proj", model.q_proj)],
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=True,
        train_lora=False,
        freeze=True,
    )
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.2,
            decoder_lr=2e-3,
            norm_lr=3e-3,
        ),
        model=selection.peft_model,
    )
    optimizer = torch.optim.AdamW(groups)

    decoder_params = list(selection.decoder_parameters.values())
    norm_params = list(selection.norm_parameters.values())
    frozen_k_params = list(model.k_proj.parameters())
    decoder_before = _clone_tensors(decoder_params)
    norm_before = _clone_tensors(norm_params)
    frozen_before = _clone_tensors(frozen_k_params)

    x = torch.randn(6, 4)
    target = torch.randn(6, 4)
    loss = torch.nn.functional.mse_loss(selection.peft_model(x), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    assert _any_changed(decoder_before, decoder_params)
    assert _any_changed(norm_before, norm_params)
    assert all(torch.equal(old, param.detach()) for old, param in zip(frozen_before, frozen_k_params))

    finalized = finalize_main_decoder_targets(
        [NamedMainDecoderTarget(name="q_proj", base_layer=model.q_proj)]
    )
    assert finalized == 1
    assert model.q_proj.trainable_decode is False
    assert all(not param.requires_grad for param in model.q_proj.parameters())


def test_current_lora_uses_full_space_proxy_only_and_finalizes_back_to_vae_low_rank():
    torch.manual_seed(11)
    model = _TinyCatModel()
    q_base = model.q_proj
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(norm_train_mode="all", norm_lr=3e-4),
        compressed_modules=[("q_proj", q_base)],
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=False,
        train_lora=True,
        freeze=True,
    )
    assert selection.lora_parameters
    assert selection.decoder_parameters == {}
    assert selection.norm_parameters
    assert all(not param.requires_grad for param in q_base.decoder.parameters())

    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.02,
            decoder_lr=9e-3,
            norm_lr=3e-4,
        ),
        model=selection.peft_model,
    )
    by_name = {group["group_name"]: group for group in groups}
    assert set(by_name) == {"lora", "norm"}
    assert by_name["lora"]["lr"] == pytest.approx(1e-3)
    assert by_name["lora"]["weight_decay"] == pytest.approx(0.02)
    assert by_name["norm"]["lr"] == pytest.approx(3e-4)
    assert by_name["norm"]["weight_decay"] == 0.0

    for param in selection.lora_parameters.values():
        param.data.normal_(mean=0.0, std=0.03)
    selection.peft_model.eval()
    x = torch.randn(5, 4)
    with torch.no_grad():
        before = selection.peft_model(x)
    finalized_model = finalize_model_level_lora(
        selection.peft_model,
        compressed_proxy_names=["q_proj"],
    )
    assert isinstance(finalized_model.q_proj, VAELinear)
    assert finalized_model.q_proj.low_rank_a is not None
    assert finalized_model.q_proj.low_rank_b is not None
    assert not list(iter_named_peft_lora_layers(finalized_model))
    with torch.no_grad():
        after = finalized_model(x)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


def test_current_lora_decoder_one_step_updates_lora_and_decoder_then_finalizes_parity():
    torch.manual_seed(13)
    model = _TinyCatModel()
    q_base = model.q_proj
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(norm_train_mode="none", lm_head_train_mode="none"),
        compressed_modules=[("q_proj", q_base)],
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=True,
        train_lora=True,
        freeze=True,
    )
    assert selection.lora_parameters
    assert selection.decoder_parameters
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
            decoder_lr=2e-3,
        ),
        model=selection.peft_model,
    )
    by_name = {group["group_name"]: group for group in groups}
    assert set(by_name) == {"lora", "decoder"}
    assert by_name["lora"]["lr"] == pytest.approx(1e-3)
    assert by_name["lora"]["weight_decay"] == pytest.approx(0.01)
    assert by_name["decoder"]["lr"] == pytest.approx(2e-3)
    assert by_name["decoder"]["weight_decay"] == 0.0

    optimizer = torch.optim.AdamW(groups)
    lora_params = list(selection.lora_parameters.values())
    decoder_params = list(selection.decoder_parameters.values())
    lora_before = _clone_tensors(lora_params)
    decoder_before = _clone_tensors(decoder_params)
    x = torch.randn(6, 4)
    target = torch.randn(6, 4)
    loss = torch.nn.functional.mse_loss(selection.peft_model(x), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert _any_changed(lora_before, lora_params)
    assert _any_changed(decoder_before, decoder_params)

    selection.peft_model.eval()
    with torch.no_grad():
        before_finalize = selection.peft_model(x)
    assert finalize_main_decoder_targets(
        [NamedMainDecoderTarget(name="q_proj", base_layer=q_base)]
    ) == 1
    finalized_model = finalize_model_level_lora(
        selection.peft_model,
        compressed_proxy_names=["q_proj"],
    )
    assert finalized_model.q_proj.trainable_decode is False
    assert finalized_model.q_proj.low_rank_a is not None
    assert finalized_model.q_proj.low_rank_b is not None
    assert not list(iter_named_peft_lora_layers(finalized_model))
    with torch.no_grad():
        after_finalize = finalized_model(x)
    torch.testing.assert_close(after_finalize, before_finalize, rtol=1e-5, atol=1e-5)


class _SequentialProjectionLayer(nn.Module):
    def __init__(self, *, q_kind: str, k_kind: str, v_kind: str):
        super().__init__()
        self.q_proj = _vae_linear() if q_kind == "vae" else nn.Linear(4, 4, bias=False)
        self.k_proj = _vae_linear() if k_kind == "vae" else nn.Linear(4, 4, bias=False)
        self.v_proj = _vae_linear() if v_kind == "vae" else nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(self.k_proj(self.q_proj(x)))


class _RemainingModel(nn.Module):
    def __init__(self, *, q_kind="vae", k_kind="linear", v_kind="linear"):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            _SequentialProjectionLayer(q_kind=q_kind, k_kind=k_kind, v_kind=v_kind)
        ])
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.model.layers[0](x)))


def _remaining_inventory(model):
    return get_or_build_cat_projection_name_inventory(
        model,
        vae_args=SimpleNamespace(),
        compression_categories=("q_proj", "k_proj", "v_proj"),
    )


def test_remaining_selector_uses_cat_inventory_and_respects_target_layers_and_skip():
    model = _RemainingModel(q_kind="vae", k_kind="linear", v_kind="linear")
    inventory = _remaining_inventory(model)
    assert list(inventory.items()) == [
        ((0, "q_proj"), "model.layers.0.q_proj"),
        ((0, "k_proj"), "model.layers.0.k_proj"),
        ((0, "v_proj"), "model.layers.0.v_proj"),
    ]
    names = select_remaining_dense_names_from_inventory(
        model,
        inventory=inventory,
        remaining_categories=("k_proj", "v_proj"),
        target_layers="all",
        skip_layers="0.v_proj",
    )
    assert names == ("model.layers.0.k_proj",)
    assert select_remaining_dense_names_from_inventory(
        model,
        inventory=inventory,
        remaining_categories=("k_proj", "v_proj"),
        target_layers=(1,),
        skip_layers="",
    ) == ()


def test_remaining_selector_hard_errors_if_future_target_is_already_compressed():
    model = _RemainingModel(q_kind="vae", k_kind="vae", v_kind="linear")
    inventory = _remaining_inventory(model)
    with pytest.raises(TypeError, match="must still be ordinary nn.Linear"):
        select_remaining_dense_names_from_inventory(
            model,
            inventory=inventory,
            remaining_categories=("k_proj", "v_proj"),
            target_layers="all",
            skip_layers="",
        )


def test_remaining_lora_selection_targets_future_dense_only_and_aux():
    model = _RemainingModel(q_kind="vae", k_kind="linear", v_kind="linear")
    inventory = _remaining_inventory(model)
    remaining = select_remaining_dense_names_from_inventory(
        model,
        inventory=inventory,
        remaining_categories=("k_proj", "v_proj"),
        target_layers="all",
        skip_layers="",
    )
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(norm_train_mode="all", norm_lr=3e-4),
        compressed_modules=(),
        dense_target_modules=remaining,
        decoder_modules=(),
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=False,
        train_lora=True,
        freeze=True,
    )
    assert selection.compressed_lora_targets == []
    assert selection.lora_parameters
    assert selection.decoder_parameters == {}
    assert selection.norm_parameters
    assert all("k_proj" in key or "v_proj" in key for key in selection.lora_parameters)
    assert all(not param.requires_grad for param in model.model.layers[0].q_proj.parameters())


def test_remaining_lora_current_decoder_separates_dense_lora_from_current_decoder():
    model = _RemainingModel(q_kind="vae", k_kind="linear", v_kind="linear")
    inventory = _remaining_inventory(model)
    remaining = select_remaining_dense_names_from_inventory(
        model,
        inventory=inventory,
        remaining_categories=("k_proj", "v_proj"),
        target_layers="all",
        skip_layers="",
    )
    decoder_targets = select_compressed_decoder_targets_from_inventory(
        model,
        inventory=inventory,
        decoder_categories=("q_proj",),
        target_layers="all",
        skip_layers="",
    )
    assert tuple(name for name, _ in decoder_targets) == ("model.layers.0.q_proj",)
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(lm_head_train_mode="linear", lm_head_lr=4e-4),
        compressed_modules=(),
        dense_target_modules=remaining,
        decoder_modules=decoder_targets,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=True,
        train_lora=True,
        freeze=True,
    )
    assert selection.compressed_lora_targets == []
    assert selection.lora_parameters
    assert selection.decoder_parameters
    assert selection.lm_head_parameters
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.03,
            decoder_lr=2e-3,
            lm_head_lr=4e-4,
        ),
        model=selection.peft_model,
    )
    by_name = {group["group_name"]: group for group in groups}
    assert set(by_name) == {"lora", "decoder", "lm_head"}
    assert by_name["lora"]["lr"] == pytest.approx(1e-3)
    assert by_name["lora"]["weight_decay"] == pytest.approx(0.03)
    assert by_name["decoder"]["lr"] == pytest.approx(2e-3)
    assert by_name["decoder"]["weight_decay"] == 0.0
    assert by_name["lm_head"]["lr"] == pytest.approx(4e-4)
    assert by_name["lm_head"]["weight_decay"] == 0.0


def test_remaining_lora_prefix_decoder_one_step_updates_future_lora_and_all_prefix_decoders():
    torch.manual_seed(17)
    model = _RemainingModel(q_kind="vae", k_kind="vae", v_kind="linear")
    inventory = _remaining_inventory(model)
    remaining = select_remaining_dense_names_from_inventory(
        model,
        inventory=inventory,
        remaining_categories=("v_proj",),
        target_layers="all",
        skip_layers="",
    )
    decoder_targets = select_compressed_decoder_targets_from_inventory(
        model,
        inventory=inventory,
        decoder_categories=("q_proj", "k_proj"),
        target_layers="all",
        skip_layers="",
    )
    assert tuple(name for name, _ in decoder_targets) == (
        "model.layers.0.q_proj",
        "model.layers.0.k_proj",
    )
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(norm_train_mode="all", norm_lr=4e-4),
        compressed_modules=(),
        dense_target_modules=remaining,
        decoder_modules=decoder_targets,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=True,
        train_lora=True,
        freeze=True,
    )
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.02,
            decoder_lr=2e-3,
            norm_lr=4e-4,
        ),
        model=selection.peft_model,
    )
    optimizer = torch.optim.AdamW(groups)
    lora_params = list(selection.lora_parameters.values())
    decoder_params = list(selection.decoder_parameters.values())
    norm_params = list(selection.norm_parameters.values())
    lora_before = _clone_tensors(lora_params)
    decoder_before = _clone_tensors(decoder_params)
    norm_before = _clone_tensors(norm_params)

    x = torch.randn(8, 4)
    target = torch.randn(8, 4)
    loss = torch.nn.functional.mse_loss(selection.peft_model(x), target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert _any_changed(lora_before, lora_params)
    assert _any_changed(decoder_before, decoder_params)
    assert _any_changed(norm_before, norm_params)

    selection.peft_model.eval()
    with torch.no_grad():
        before_finalize = selection.peft_model(x)
    finalized_decoder_count = finalize_main_decoder_targets(
        [NamedMainDecoderTarget(name=name, base_layer=module) for name, module in decoder_targets]
    )
    assert finalized_decoder_count == 2
    finalized_model = finalize_model_level_lora(selection.peft_model, compressed_proxy_names=None)
    assert isinstance(finalized_model.model.layers[0].v_proj, nn.Linear)
    assert not isinstance(finalized_model.model.layers[0].v_proj, VAELinear)
    assert not list(iter_named_peft_lora_layers(finalized_model))
    with torch.no_grad():
        after_finalize = finalized_model(x)
    torch.testing.assert_close(after_finalize, before_finalize, rtol=1e-5, atol=1e-5)


def test_all_six_after_category_modes_accept_aux_trainables():
    aux = AuxTrainableConfig(
        norm_train_mode="all",
        norm_lr=3e-4,
        lm_head_train_mode="linear",
        lm_head_lr=4e-4,
    )

    for train_lora, train_decoder in ((False, True), (True, False), (True, True)):
        model = _TinyCatModel()
        selection = build_model_level_trainable_selection(
            model,
            aux=aux,
            compressed_modules=[("q_proj", model.q_proj)],
            rank=2,
            alpha=4.0,
            dropout=0.0,
            train_decoder=train_decoder,
            train_lora=train_lora,
            freeze=True,
        )
        assert selection.norm_parameters
        assert selection.lm_head_parameters

    for mode in ("remaining_lora", "remaining_lora_current_decoder", "remaining_lora_prefix_decoder"):
        model = _RemainingModel(
            q_kind="vae",
            k_kind=("vae" if mode == "remaining_lora_prefix_decoder" else "linear"),
            v_kind="linear",
        )
        inventory = _remaining_inventory(model)
        current = "k_proj" if mode == "remaining_lora_prefix_decoder" else "q_proj"
        categories = ("q_proj", "k_proj", "v_proj")
        current_idx = categories.index(current)
        remaining = select_remaining_dense_names_from_inventory(
            model,
            inventory=inventory,
            remaining_categories=categories[current_idx + 1 :],
            target_layers="all",
            skip_layers="",
        )
        decoder_categories = ()
        if mode == "remaining_lora_current_decoder":
            decoder_categories = (current,)
        elif mode == "remaining_lora_prefix_decoder":
            decoder_categories = categories[: current_idx + 1]
        decoder_targets = select_compressed_decoder_targets_from_inventory(
            model,
            inventory=inventory,
            decoder_categories=decoder_categories,
            target_layers="all",
            skip_layers="",
        )
        selection = build_model_level_trainable_selection(
            model,
            aux=aux,
            compressed_modules=(),
            dense_target_modules=remaining,
            decoder_modules=decoder_targets,
            rank=2,
            alpha=4.0,
            dropout=0.0,
            train_decoder=bool(decoder_targets),
            train_lora=bool(remaining),
            freeze=True,
        )
        assert selection.norm_parameters, mode
        assert selection.lm_head_parameters, mode


@pytest.mark.parametrize(
    ("canonical_mode", "runner_name"),
    [
        ("current_decoder", "run_canonical_current_decoder"),
        ("current_lora", "run_canonical_current_lora"),
        ("current_lora_decoder", "run_canonical_current_lora_decoder"),
        ("remaining_lora", "run_canonical_remaining_lora"),
        (
            "remaining_lora_current_decoder",
            "run_canonical_remaining_lora_current_decoder",
        ),
        (
            "remaining_lora_prefix_decoder",
            "run_canonical_remaining_lora_prefix_decoder",
        ),
    ],
)
def test_online_run_after_category_routes_all_six_modes_to_canonical_runner(
    canonical_mode,
    runner_name,
):
    model = _RemainingModel(q_kind="vae", k_kind="linear", v_kind="linear")
    cat_args = SimpleNamespace(
        after_category_mode=canonical_mode,
        target_layers="all",
        skip_layers="",
    )
    vae_args = SimpleNamespace()
    training_args = SimpleNamespace()
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    stage = SimpleNamespace(mode=canonical_mode)
    canonical_result = SimpleNamespace(
        model=model,
        did_train=True,
        distill_meta={"mode": canonical_mode, "did_train": True},
    )

    with mock.patch.object(cat_after_distill, "resolve_cat_after_category_stage", return_value=stage), mock.patch.object(
        cat_after_distill,
        runner_name,
        return_value=canonical_result,
    ) as canonical_runner:
        result = cat_after_distill.run_after_category_distill(
            model=model,
            category="q_proj",
            cat_args=cat_args,
            vae_args=vae_args,
            training_args=training_args,
            logger=logger,
            lora_round_idx=0,
            transpose_modules=("q_proj",),
            only_decoder_projections=True,
            compression_categories=("q_proj", "k_proj", "v_proj"),
            teacher_runtime=None,
            newly_compressed_target_count=1,
            online_cat=True,
        )

    canonical_runner.assert_called_once()
    assert result.did_train is True
    assert result.next_lora_round_idx == 1
    assert result.distill_meta["mode"] == canonical_mode


def test_remaining_family_metadata_carries_exact_target_inventories():
    model = _RemainingModel(q_kind="vae", k_kind="linear", v_kind="linear")
    inventory = _remaining_inventory(model)
    remaining = select_remaining_dense_names_from_inventory(
        model,
        inventory=inventory,
        remaining_categories=("k_proj", "v_proj"),
        target_layers="all",
        skip_layers="0.v_proj",
    )
    decoders = select_compressed_decoder_targets_from_inventory(
        model,
        inventory=inventory,
        decoder_categories=("q_proj",),
        target_layers="all",
        skip_layers="",
    )
    assert remaining == ("model.layers.0.k_proj",)
    assert [name for name, _module in decoders] == ["model.layers.0.q_proj"]
