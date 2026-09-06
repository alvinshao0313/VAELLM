from __future__ import annotations

import pytest
import torch
from torch import nn

from e2e_common.full_lora import (
    FullCompressedPeftProxy,
    assert_exact_adapter_target_set,
    build_full_compressed_peft_model,
    collect_exact_peft_lora_config,
    collect_logical_adapter_target_names,
    finalize_model_level_lora,
    iter_named_full_compressed_peft_proxies,
)
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.config.configs import AuxTrainableConfig
from train_utils.model_level_trainables import (
    NORM_TYPE_REGISTRY,
    apply_aux_trainables,
    assert_disjoint_component_inventories,
    build_model_level_trainable_selection,
    classify_peft_lora_parameters,
    collect_lm_head_lora_parameters,
    collect_lora_parameters,
    enable_decoder_trainables,
    freeze_all_parameters,
    is_backbone_norm_module,
    setup_lm_head_trainables,
)


def _vae_linear(dim: int = 4):
    latent_dim = 9
    codebook_dim = dim
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
        out_dim=codebook_dim,
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
        codebook_dim=codebook_dim,
        transpose=False,
    )


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _vae_linear(4)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 6, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.layer(x)))


def test_norm_type_registry_contains_layernorm_and_rmsnorm():
    assert nn.LayerNorm in NORM_TYPE_REGISTRY
    assert nn.RMSNorm in NORM_TYPE_REGISTRY
    assert is_backbone_norm_module(nn.LayerNorm(3))


def test_exact_adapter_target_set_for_compressed_only():
    model = _TinyModel()
    peft_model = build_full_compressed_peft_model(
        model,
        selected_modules=[("layer", model.layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    assert collect_logical_adapter_target_names(peft_model) == {"layer"}
    assert_exact_adapter_target_set(peft_model, compressed_proxy_names=["layer"])


def test_one_adapter_union_compressed_and_lm_head():
    model = _TinyModel()
    peft_model = build_full_compressed_peft_model(
        model,
        selected_modules=[("layer", model.layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        include_lm_head=True,
    )
    assert collect_logical_adapter_target_names(peft_model) == {"layer", "lm_head"}
    assert_exact_adapter_target_set(
        peft_model,
        compressed_proxy_names=["layer"],
        include_lm_head=True,
    )


def test_heterogeneous_existing_ranks_use_one_exact_rank_pattern_with_step_zero_parity():
    class TwoLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _vae_linear(4)
            self.b = _vae_linear(4)

        def forward(self, x):
            return self.b(self.a(x))

    torch.manual_seed(11)
    model = TwoLayer()
    payloads = {
        "a": (torch.randn(4, 2), torch.randn(2, 4)),
        "b": (torch.randn(4, 4), torch.randn(4, 4)),
    }
    model.a.low_rank_a = nn.Parameter(payloads["a"][0].clone(), requires_grad=False)
    model.a.low_rank_b = nn.Parameter(payloads["a"][1].clone(), requires_grad=False)
    model.b.low_rank_a = nn.Parameter(payloads["b"][0].clone(), requires_grad=False)
    model.b.low_rank_b = nn.Parameter(payloads["b"][1].clone(), requires_grad=False)
    x = torch.randn(3, 4)
    before = model(x).detach()

    peft_model = build_full_compressed_peft_model(
        model,
        selected_modules=[("a", model.a), ("b", model.b)],
        initial_low_rank_payloads=payloads,
        rank=3,
        alpha=6.0,
        dropout=0.0,
    )
    exact = collect_exact_peft_lora_config(
        peft_model,
        default_rank=3,
        alpha=6.0,
        dropout=0.0,
    )
    assert exact == {
        "rank": 3,
        "alpha": 6.0,
        "dropout": 0.0,
        "rank_pattern": {"a": 2, "b": 4},
        "target_modules": ["a", "b"],
    }
    torch.testing.assert_close(peft_model(x), before, rtol=1e-5, atol=1e-5)


def test_explicit_rank_rejects_existing_payload_rank_conflict():
    model = _TinyModel()
    payloads = {"layer": (torch.randn(4, 2), torch.randn(2, 4))}
    model.layer.low_rank_a = nn.Parameter(payloads["layer"][0].clone(), requires_grad=False)
    model.layer.low_rank_b = nn.Parameter(payloads["layer"][1].clone(), requires_grad=False)
    with pytest.raises(ValueError, match="explicit value 3.*payload rank 2"):
        build_full_compressed_peft_model(
            model,
            selected_modules=[("layer", model.layer)],
            initial_low_rank_payloads=payloads,
            rank=3,
            alpha=6.0,
            dropout=0.0,
            rank_explicit=True,
        )
def test_finalize_compressed_plus_lm_head_preserves_base_and_matches_forward():
    torch.manual_seed(0)
    model = _TinyModel()
    peft_model = build_full_compressed_peft_model(
        model,
        selected_modules=[("layer", model.layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        include_lm_head=True,
    )
    # Force non-zero adapter weights for a meaningful parity check.
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data.normal_(mean=0.0, std=0.05)

    x = torch.randn(3, 4)
    with torch.no_grad():
        before = peft_model(x)

    proxy = dict(iter_named_full_compressed_peft_proxies(peft_model))["layer"]
    from e2e_common.full_lora import _vae_linear_base_fingerprint

    base_before = _vae_linear_base_fingerprint(proxy.base_layer)

    finalized = finalize_model_level_lora(peft_model, compressed_proxy_names=["layer"])
    assert isinstance(finalized.layer, VAELinear)
    assert not list(iter_named_full_compressed_peft_proxies(finalized))
    after_fp = _vae_linear_base_fingerprint(finalized.layer)
    assert set(after_fp) == set(base_before)
    for key in base_before:
        assert torch.equal(after_fp[key], base_before[key])
    assert finalized.layer.low_rank_a is not None
    assert finalized.layer.low_rank_b is not None

    with torch.no_grad():
        after = finalized(x)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


def test_finalize_compressed_lora_preserves_payload_dtype_and_bf16_forward():
    torch.manual_seed(7)
    layer = _vae_linear().to(dtype=torch.bfloat16)
    model = nn.Module()
    model.add_module("layer", layer)
    peft_model = build_full_compressed_peft_model(
        model,
        selected_modules=[("layer", layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            assert param.dtype == torch.bfloat16
            param.data.normal_(mean=0.0, std=0.05)

    x = torch.randn(3, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        before = model.layer(x)
    finalized = finalize_model_level_lora(
        peft_model,
        compressed_proxy_names=["layer"],
    )
    assert finalized.layer.low_rank_a.dtype == torch.bfloat16
    assert finalized.layer.low_rank_b.dtype == torch.bfloat16

    with torch.no_grad():
        after = finalized.layer(x)
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)

    finalized.layer.prime_decoded_weight_cache(dtype=torch.bfloat16)
    with torch.no_grad():
        after_cached = finalized.layer(x)
    torch.testing.assert_close(after_cached, before, rtol=0.0, atol=0.0)


def test_aux_norm_final_and_lm_head_linear():
    model = _TinyModel()
    freeze_all_parameters(model)
    aux = AuxTrainableConfig(norm_train_mode="final", lm_head_train_mode="linear")
    # TinyModel is not a HF model; final-norm path needs get_model_type.
    # Exercise all-mode + linear head instead for this scaffold.
    aux = AuxTrainableConfig(norm_train_mode="all", lm_head_train_mode="linear")
    norm_params, head_params = apply_aux_trainables(model, aux)
    assert norm_params
    assert any("norm" in key for key in norm_params)
    assert head_params
    assert isinstance(model.lm_head, type(model.lm_head))  # still module
    from e2e_common.post_norm_head import LMHeadWithPostNormLinear

    assert isinstance(model.lm_head, LMHeadWithPostNormLinear)


def test_lm_head_full_unties_shared_embedding():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(6, 4)
            self.lm_head = nn.Linear(4, 6, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
            self.config = type("C", (), {"tie_word_embeddings": True})()

    model = Tied()
    freeze_all_parameters(model)
    params = setup_lm_head_trainables(model, lm_head_train_mode="full")
    assert params
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
    assert model.config.tie_word_embeddings is False
    assert torch.equal(model.lm_head.weight.detach(), model.embed_tokens.weight.detach())


def _assert_unique_param_ids(*inventories):
    seen = {}
    for inv_name, params in inventories:
        for key, param in params.items():
            pid = id(param)
            assert pid not in seen, f"duplicate Parameter id: {inv_name}[{key}] vs {seen[pid]}"
            seen[pid] = f"{inv_name}[{key}]"


def test_compressed_plus_lm_head_lora_parameter_grouping():
    model = _TinyModel()
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(lm_head_train_mode="lora"),
        compressed_modules=[("layer", model.layer)],
        dense_target_modules=(),
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=False,
    )
    assert selection.include_lm_head_lora is True
    assert selection.lora_parameters
    assert selection.lm_head_parameters
    assert all(key.startswith("lora::layer.") for key in selection.lora_parameters)
    assert all(key.startswith("lm_head_lora::") for key in selection.lm_head_parameters)
    # Blanket "lora_" dump would put lm_head into backbone inventory; classify must not.
    backbone, lm_head = classify_peft_lora_parameters(selection.peft_model)
    assert backbone == selection.lora_parameters
    assert lm_head == selection.lm_head_parameters
    assert collect_lora_parameters(selection.peft_model) == backbone
    assert collect_lm_head_lora_parameters(selection.peft_model) == lm_head
    _assert_unique_param_ids(
        ("lora", selection.lora_parameters),
        ("lm_head", selection.lm_head_parameters),
    )
    assert_disjoint_component_inventories(
        decoder_parameters=selection.decoder_parameters,
        lora_parameters=selection.lora_parameters,
        norm_parameters=selection.norm_parameters,
        lm_head_parameters=selection.lm_head_parameters,
    )


def test_remaining_plus_lm_head_lora_parameter_grouping():
    class DenseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self.lm_head = nn.Linear(4, 6, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.proj(x))

    model = DenseModel()
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(lm_head_train_mode="lora"),
        compressed_modules=(),
        dense_target_modules=["proj"],
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    assert selection.lora_parameters
    assert selection.lm_head_parameters
    assert all(key.startswith("lora::proj.") for key in selection.lora_parameters)
    assert all(key.startswith("lm_head_lora::") for key in selection.lm_head_parameters)
    _assert_unique_param_ids(
        ("lora", selection.lora_parameters),
        ("lm_head", selection.lm_head_parameters),
    )


def test_lm_head_only_lora_parameter_grouping():
    class HeadOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(4, 6, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lm_head(x)

    model = HeadOnly()
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(lm_head_train_mode="lora"),
        compressed_modules=(),
        dense_target_modules=(),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    assert selection.lora_parameters == {}
    assert selection.lm_head_parameters
    assert all(key.startswith("lm_head_lora::") for key in selection.lm_head_parameters)
    assert collect_lora_parameters(selection.peft_model) == {}
    assert collect_lm_head_lora_parameters(selection.peft_model) == selection.lm_head_parameters


def test_parameter_id_appears_once_across_inventories_and_decoder_dedupe():
    model = _TinyModel()
    # Same VAELinear listed twice under different logical names must not duplicate Parameter ids.
    decoder_params = enable_decoder_trainables(
        model,
        selected_modules=[("layer", model.layer), ("layer_alias", model.layer)],
    )
    ids = [id(p) for p in decoder_params.values()]
    assert len(ids) == len(set(ids))
    assert all(key.startswith("decoder::layer.") for key in decoder_params)
    assert not any(key.startswith("decoder::layer_alias.") for key in decoder_params)

    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(norm_train_mode="all", lm_head_train_mode="lora"),
        compressed_modules=[("layer", model.layer)],
        dense_target_modules=(),
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=True,
        freeze=True,
    )
    assert selection.decoder_parameters
    assert selection.lora_parameters
    assert selection.norm_parameters
    assert selection.lm_head_parameters
    _assert_unique_param_ids(
        ("decoder", selection.decoder_parameters),
        ("lora", selection.lora_parameters),
        ("norm", selection.norm_parameters),
        ("lm_head", selection.lm_head_parameters),
    )
    # Cross-component conflict must hard-error.
    conflicted = dict(selection.lm_head_parameters)
    conflicted["conflict"] = next(iter(selection.lora_parameters.values()))
    with pytest.raises(RuntimeError, match="Parameter id conflict"):
        assert_disjoint_component_inventories(
            decoder_parameters=selection.decoder_parameters,
            lora_parameters=selection.lora_parameters,
            norm_parameters=selection.norm_parameters,
            lm_head_parameters=conflicted,
        )


def test_lm_head_full_unties_via_get_input_embeddings_not_decoy():
    class DecoyTied(nn.Module):
        def __init__(self):
            super().__init__()
            # Decoy would win a naive named_modules scan for embed_tokens.
            self.embed_tokens = nn.Embedding(6, 4)
            self.true_embed = nn.Embedding(6, 4)
            self.lm_head = nn.Linear(4, 6, bias=False)
            self.lm_head.weight = self.true_embed.weight
            self.config = type("C", (), {"tie_word_embeddings": True})()
            with torch.no_grad():
                self.embed_tokens.weight.add_(1.0)

        def get_input_embeddings(self):
            return self.true_embed

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.true_embed(token_ids))

    model = DecoyTied()
    assert model.lm_head.weight.data_ptr() == model.true_embed.weight.data_ptr()
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()

    token_ids = torch.randint(0, 6, (2, 3))
    with torch.no_grad():
        before = model(token_ids)

    freeze_all_parameters(model)
    params = setup_lm_head_trainables(model, lm_head_train_mode="full")
    assert params
    assert model.lm_head.weight.data_ptr() != model.true_embed.weight.data_ptr()
    assert model.config.tie_word_embeddings is False
    assert torch.equal(model.lm_head.weight.detach(), model.true_embed.weight.detach())
    assert not torch.equal(model.lm_head.weight.detach(), model.embed_tokens.weight.detach())

    with torch.no_grad():
        after = model(token_ids)
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
