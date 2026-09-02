import logging
import tempfile

import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.aux_trainables import (
    apply_auxiliary_payload_to_compressed_model,
    enable_compressed_lora_auxiliary_trainables,
    load_auxiliary_sidecar,
    restore_auxiliary_trainables,
    save_auxiliary_sidecar,
    snapshot_auxiliary_trainables,
)
from compressed_e2e_fintuning.runtime import _build_subspace_low_rank_peft_model
from litebsq.autoencoder import Decoder
from litebsq.low_rank_scope import LOW_RANK_SCOPE_COMPRESSED_SUBSPACE, LOW_RANK_SCOPE_FULL
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.full_lora_proxy import build_full_compressed_peft_model


def _layer_with_bias():
    bits = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )
    decoder = Decoder(
        in_dim=9,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=nn.Parameter(torch.arange(4, dtype=torch.float32)),
        original_weight=None,
        vq_weight=bits,
        decoder=decoder,
        codebook_dim=4,
        transpose=False,
    )


def _root(layer):
    root = nn.Module()
    root.add_module("layer", layer)
    root.prepare_inputs_for_generation = lambda *args, **kwargs: kwargs
    return root


def test_full_lora_bit_aux_bias_is_trainable_and_sidecar_round_trips():
    layer = _layer_with_bias()
    root = _root(layer)
    peft_model = build_full_compressed_peft_model(
        root,
        selected_modules=[("layer", layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    selection = enable_compressed_lora_auxiliary_trainables(
        peft_model,
        selected_vae_modules=[("layer", layer)],
        low_rank_scope=LOW_RANK_SCOPE_FULL,
        sparse_bit_tuning=True,
        vae_tune_bias=True,
        tune_final_norm=False,
        use_post_norm_head_linear=False,
    )
    assert set(selection.parameters) == {"vae_bias::layer"}
    assert layer.bias.requires_grad
    assert any("lora_" in name and param.requires_grad for name, param in peft_model.named_parameters())

    with torch.no_grad():
        layer.bias.add_(3.0)
    expected = layer.bias.detach().clone()
    payload = snapshot_auxiliary_trainables(selection.parameters)
    with tempfile.TemporaryDirectory() as tmp:
        save_auxiliary_sidecar(tmp, payload)
        loaded = load_auxiliary_sidecar(tmp)
        with torch.no_grad():
            layer.bias.zero_()
        restore_auxiliary_trainables(selection.parameters, loaded)
        assert torch.equal(layer.bias, expected)

        fresh_layer = _layer_with_bias()
        fresh_root = _root(fresh_layer)
        written = apply_auxiliary_payload_to_compressed_model(fresh_root, loaded)
        assert written == 1
        assert torch.equal(fresh_layer.bias, expected)


def test_subspace_lora_bit_aux_bias_reenables_base_bias_after_peft_freeze():
    layer = _layer_with_bias()
    root = _root(layer)
    peft_model, _ = _build_subspace_low_rank_peft_model(
        root,
        selected_modules=[("layer", layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        decoder_layer_ids=[0],
        target_module_suffixes=["layer"],
        parallel_stage_decode=False,
        log=logging.getLogger("sbt-test"),
    )
    assert not layer.bias.requires_grad
    selection = enable_compressed_lora_auxiliary_trainables(
        peft_model,
        selected_vae_modules=[("layer", layer)],
        low_rank_scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        sparse_bit_tuning=True,
        vae_tune_bias=True,
        tune_final_norm=False,
        use_post_norm_head_linear=False,
    )
    assert layer.bias.requires_grad
    assert set(selection.parameters) == {"vae_bias::layer"}
    assert any("lora_" in name and param.requires_grad for name, param in peft_model.named_parameters())
