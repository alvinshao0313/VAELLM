import argparse
import json

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.cat_train_pipeline import _collect_current_trainable_linears
from train_utils.cat_train_runtime import _to_jsonable, normalize_cat_runtime_vae_original_state
from train_utils.config.targets import (
    collect_e2e_compressed_targets,
    discover_cat_projection_inventory,
    parse_compression_categories,
    parse_skip_layers,
    parse_target_layers,
    parse_target_modules,
    select_remaining_dense_refs,
    validate_skip_layers_against_inventory,
)
from train_utils.utils import collect_linears


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


def _make_vae_linear(*, always_use_original=False, protect_original_weight=False, original_weight=True):
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


class _Layer(nn.Module):
    def __init__(self, *, q="compressed", k="compressed", v="linear"):
        super().__init__()
        self.q_proj = self._make("q", q)
        self.k_proj = self._make("k", k)
        self.v_proj = self._make("v", v)

    def _make(self, _name, kind):
        if kind == "compressed":
            return _make_vae_linear()
        if kind == "original_only":
            return _make_vae_linear(always_use_original=True, protect_original_weight=True)
        if kind == "protected_compressed":
            return _make_vae_linear(always_use_original=False, protect_original_weight=True)
        if kind == "linear":
            return nn.Linear(4, 4, bias=False)
        raise AssertionError(kind)


class _TinyModel(nn.Module):
    def __init__(self, layer_kinds=None):
        super().__init__()
        kinds = layer_kinds or [{"q": "compressed", "k": "compressed", "v": "linear"}] * 2
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(**spec) for spec in kinds])


def test_e2e_rejects_target_module_alias_q():
    with pytest.raises((ValueError, Exception), match="q"):
        parse_target_modules("q")


def test_e2e_ordinary_skip_linear_is_not_a_compressed_target():
    model = _TinyModel()
    selected = collect_e2e_compressed_targets(
        model,
        target_layers=parse_target_layers("all"),
        target_modules=parse_target_modules("all"),
        num_layers=2,
    )
    names = [name for name, _module in selected]
    assert any(name.endswith("q_proj") for name in names)
    assert any(name.endswith("k_proj") for name in names)
    assert not any(name.endswith("v_proj") for name in names)


def test_e2e_always_use_original_vaelinear_is_excluded():
    model = _TinyModel(layer_kinds=[{"q": "original_only", "k": "compressed", "v": "linear"}] * 2)
    selected = collect_e2e_compressed_targets(
        model,
        target_layers=parse_target_layers("all"),
        target_modules=parse_target_modules("all"),
        num_layers=2,
    )
    names = [name for name, _module in selected]
    assert not any(name.endswith("q_proj") for name in names)
    assert any(name.endswith("k_proj") for name in names)


def test_e2e_protect_original_weight_without_always_use_original_is_legal():
    model = _TinyModel(layer_kinds=[{"q": "protected_compressed", "k": "compressed", "v": "linear"}] * 2)
    selected = collect_e2e_compressed_targets(
        model,
        target_layers=parse_target_layers("all"),
        target_modules=parse_target_modules("q_proj"),
        num_layers=2,
    )
    assert len(selected) == 2
    for _name, module in selected:
        assert isinstance(module, VAELinear)
        assert module.always_use_original is False
        assert module.protect_original_weight is True


def test_e2e_explicit_target_module_without_legal_vaelinear_errors():
    model = _TinyModel()
    with pytest.raises(ValueError, match="v_proj"):
        collect_e2e_compressed_targets(
            model,
            target_layers=parse_target_layers("all"),
            target_modules=parse_target_modules("v_proj"),
            num_layers=2,
        )


def test_cat_skip_requires_inventory_membership():
    model = _TinyModel()
    categories = parse_compression_categories("q_proj,k_proj,v_proj")
    target_layers = parse_target_layers("0-1")
    inventory = discover_cat_projection_inventory(model, compression_categories=categories)
    validate_skip_layers_against_inventory(
        parse_skip_layers("1.v_proj"),
        target_layers=target_layers,
        compression_categories=categories,
        inventory=inventory,
    )
    with pytest.raises(ValueError, match="target_layers"):
        validate_skip_layers_against_inventory(
            parse_skip_layers("7.q_proj"),
            target_layers=target_layers,
            compression_categories=categories,
            inventory=inventory,
        )
    with pytest.raises(ValueError, match="compression_categories"):
        validate_skip_layers_against_inventory(
            parse_skip_layers("1.down_proj"),
            target_layers=target_layers,
            compression_categories=categories,
            inventory=inventory,
        )


def test_remaining_lora_never_selects_skip_linears():
    model = _TinyModel()
    categories = ("q_proj", "k_proj", "v_proj")
    skip = parse_skip_layers("0.v_proj")
    refs = collect_linears(
        model,
        transpose_modules=(),
        only_decoder_projections=True,
        categories=categories,
    )
    remaining = select_remaining_dense_refs(
        refs,
        remaining_categories=("v_proj",),
        skip_layers=skip,
        target_layers=parse_target_layers("all"),
    )
    names = [ref.name for ref in remaining]
    assert "model.layers.1.v_proj" in names
    assert "model.layers.0.v_proj" not in names
    assert all(isinstance(ref.module, nn.Linear) for ref in remaining)
    assert all(ref.category == "v_proj" for ref in remaining)


def test_cat_pipeline_collects_current_linears_with_canonical_category_argument():
    refs = _collect_current_trainable_linears(
        _TinyModel(layer_kinds=[{"q": "linear", "k": "linear", "v": "linear"}] * 2),
        transpose_modules=("q_proj",),
        only_decoder_projections=True,
        compression_categories=("q_proj", "k_proj"),
    )
    assert [ref.category for ref in refs] == ["q_proj", "k_proj", "q_proj", "k_proj"]
    assert [ref.transpose for ref in refs] == [True, False, True, False]


def test_cat_runtime_snapshot_serializes_common_frozen_sets():
    payload = _to_jsonable(
        argparse.Namespace(
            skip_layers=frozenset({(1, "q_proj"), (0, "k_proj")}),
            resolve_after_category_config=lambda _category: None,
            _v6_cat_runtime_state_payload={"tensor": torch.ones(1)},
        )
    )
    assert payload == {"skip_layers": [[0, "k_proj"], [1, "q_proj"]]}
    json.dumps(payload)


def test_normalize_runtime_only_treats_always_use_original_as_legacy_skip():
    protected = _make_vae_linear(always_use_original=False, protect_original_weight=True)
    model = nn.Sequential(protected)
    stripped = normalize_cat_runtime_vae_original_state(model)
    assert stripped == 0
    assert protected.always_use_original is False
    assert protected.protect_original_weight is True
    assert protected.original_weight is not None

    legacy = nn.Sequential(_make_vae_linear(always_use_original=True, protect_original_weight=True))
    with pytest.raises(ValueError, match="Legacy skip-as-VAELinear"):
        normalize_cat_runtime_vae_original_state(legacy)
