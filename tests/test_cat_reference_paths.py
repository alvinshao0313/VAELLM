from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train_utils.base_reference import (
    clone_frozen_linear_from_reference,
    get_reference_module,
    load_frozen_base_reference_model,
)
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.cat_residual_from_base import (
    _ResidualFromBaseResidency,
    _ResidualTarget,
    _apply_residual_from_base_residency,
    _reference_weight,
    _restore_all_residual_from_base_vae,
)
from train_utils.eval_utils import evaluate_vae_linear_mse


class TinyReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {
                                "q_proj": nn.Linear(3, 2),
                            }
                        )
                    }
                )
            ]
        )


def test_load_frozen_base_reference_model_freezes_eval_and_disables_cache(monkeypatch):
    loaded = TinyReferenceModel()
    calls = []

    def fake_get_model(model_path, access_token):
        calls.append((model_path, access_token))
        return loaded

    monkeypatch.setattr("train_utils.base_reference.model_utils.get_model", fake_get_model)

    model = load_frozen_base_reference_model(
        "tiny-path",
        access_token="token",
        device="cpu",
        dtype=torch.float64,
    )

    assert model is loaded
    assert calls == [("tiny-path", "token")]
    assert model.training is False
    assert model.config.use_cache is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float64}


def test_get_reference_module_supports_numeric_module_paths():
    model = TinyReferenceModel()

    module = get_reference_module(model, "model.layers.0.self_attn.q_proj")

    assert module is model.model.layers[0]["self_attn"]["q_proj"]


def test_get_reference_module_missing_path_mentions_full_name():
    model = TinyReferenceModel()

    with pytest.raises(ValueError, match="model.layers.0.mlp.gate_proj"):
        get_reference_module(model, "model.layers.0.mlp.gate_proj")


def test_clone_frozen_linear_from_reference_copies_values_without_sharing_storage():
    model = TinyReferenceModel()
    source = model.model.layers[0]["self_attn"]["q_proj"]
    with torch.no_grad():
        source.weight.copy_(torch.arange(source.weight.numel()).view_as(source.weight))
        source.bias.copy_(torch.arange(source.bias.numel()))

    cloned = clone_frozen_linear_from_reference(
        model,
        "model.layers.0.self_attn.q_proj",
        device="cpu",
        dtype=torch.float64,
    )

    assert cloned.training is False
    assert cloned.in_features == source.in_features
    assert cloned.out_features == source.out_features
    assert cloned.weight.requires_grad is False
    assert cloned.bias is not None
    assert cloned.bias.requires_grad is False
    assert cloned.weight is not source.weight
    assert cloned.bias is not source.bias
    assert cloned.weight.data_ptr() != source.weight.data_ptr()
    assert cloned.bias.data_ptr() != source.bias.data_ptr()
    torch.testing.assert_close(cloned.weight, source.weight.to(dtype=torch.float64))
    torch.testing.assert_close(cloned.bias, source.bias.to(dtype=torch.float64))


def test_clone_frozen_linear_rejects_non_linear_target():
    model = TinyReferenceModel()

    with pytest.raises(ValueError, match="not nn.Linear"):
        clone_frozen_linear_from_reference(model, "model.layers.0.self_attn", device="cpu")


def _make_residual_decoder() -> Decoder:
    return Decoder(
        in_dim=9,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    ).to(dtype=torch.float32)


def _make_residual_vae_linear() -> VAELinear:
    bits = torch.tensor(
        [
            [[True, False, True, False, True, False, True, False, True]],
            [[False, True, False, True, False, True, False, True, False]],
            [[True, True, False, False, True, True, False, False, True]],
            [[False, False, True, True, False, False, True, True, False]],
        ],
        dtype=torch.bool,
    )
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=_make_residual_decoder(),
        codebook_dim=4,
        transpose=False,
    )


class ResidualReferenceModel(nn.Module):
    def __init__(self, *, vae: bool, offset: float = 0.0):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for idx, category in enumerate(("q_proj", "k_proj")):
            if vae:
                module = _make_residual_vae_linear()
            else:
                module = nn.Linear(4, 4, bias=True)
                with torch.no_grad():
                    module.weight.copy_(torch.arange(16, dtype=torch.float32).view(4, 4) + offset + idx)
                    module.bias.copy_(torch.arange(4, dtype=torch.float32) + offset + idx + 0.5)
            setattr(layer.self_attn, category, module)
        self.model.layers = nn.ModuleList([layer])


def test_residual_reference_weight_comes_from_independent_reference_model():
    compressed = ResidualReferenceModel(vae=True)
    reference = ResidualReferenceModel(vae=False, offset=100.0)
    target = _ResidualTarget(
        name="model.layers.0.self_attn.q_proj",
        category="q_proj",
        module=compressed.model.layers[0].self_attn.q_proj,
        transpose=False,
    )

    weight = _reference_weight(reference, target)

    assert torch.equal(weight, reference.model.layers[0].self_attn.q_proj.weight)
    assert target.module.original_weight is None


def test_residual_from_base_residency_uses_reference_clone_for_inactive_category():
    compressed = ResidualReferenceModel(vae=True)
    reference = ResidualReferenceModel(vae=False, offset=200.0)
    residency = _ResidualFromBaseResidency()

    prewarm = _apply_residual_from_base_residency(
        model=compressed,
        reference_model=reference,
        residency=residency,
        active_categories=["q_proj"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    q_live = compressed.model.layers[0].self_attn.q_proj
    k_live = compressed.model.layers[0].self_attn.k_proj
    k_ref = reference.model.layers[0].self_attn.k_proj
    assert isinstance(q_live, VAELinear)
    assert isinstance(k_live, nn.Linear)
    assert [target.name for target in prewarm] == ["model.layers.0.self_attn.q_proj"]
    assert torch.equal(k_live.weight, k_ref.weight)
    assert k_live.weight.data_ptr() != k_ref.weight.data_ptr()
    assert k_live.weight.requires_grad is False
    assert "model.layers.0.self_attn.k_proj" in residency.stashed_vae_modules


def test_residual_from_base_final_restore_returns_all_vae():
    compressed = ResidualReferenceModel(vae=True)
    reference = ResidualReferenceModel(vae=False, offset=300.0)
    residency = _ResidualFromBaseResidency()

    _apply_residual_from_base_residency(
        model=compressed,
        reference_model=reference,
        residency=residency,
        active_categories=["q_proj"],
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    _restore_all_residual_from_base_vae(
        model=compressed,
        residency=residency,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert isinstance(compressed.model.layers[0].self_attn.q_proj, VAELinear)
    assert isinstance(compressed.model.layers[0].self_attn.k_proj, VAELinear)
    assert compressed.model.layers[0].self_attn.q_proj.original_weight is None
    assert compressed.model.layers[0].self_attn.k_proj.original_weight is None
    assert residency.stashed_vae_modules == {}


def test_linear_mse_uses_ref_model_with_original_none():
    compressed = ResidualReferenceModel(vae=True)
    reference = ResidualReferenceModel(vae=False, offset=400.0)

    result = evaluate_vae_linear_mse(
        compressed,
        ref_model=reference,
        topk=2,
        log=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
    )

    assert result["num_vae_linear"] == 2
    assert result["num_compared"] == 2
    assert result["num_skipped"] == 0
    assert {item["source"] for item in result["worst_by_mse"]} == {"ref_model"}
    assert compressed.model.layers[0].self_attn.q_proj.original_weight is None


def test_linear_mse_requires_ref_model():
    compressed = ResidualReferenceModel(vae=True)

    with pytest.raises(ValueError, match="ref_model is required"):
        evaluate_vae_linear_mse(
            compressed,
            ref_model=None,
            topk=2,
            log=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        )


def test_linear_mse_missing_reference_module_errors():
    compressed = ResidualReferenceModel(vae=True)
    reference = ResidualReferenceModel(vae=False, offset=500.0)
    del reference.model.layers[0].self_attn.k_proj

    with pytest.raises((AttributeError, ValueError), match="k_proj"):
        evaluate_vae_linear_mse(
            compressed,
            ref_model=reference,
            topk=2,
            log=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        )
