import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.aux_trainables import (
    apply_auxiliary_payload_to_compressed_model,
    enable_compressed_lora_auxiliary_trainables,
)
from litebsq.autoencoder import Decoder
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


def test_vae_bias_sidecar_apply_hard_error():
    """Current contract: legacy vae_bias::* sidecar payload must hard-error."""
    layer = _layer_with_bias()
    root = _root(layer)
    with pytest.raises(ValueError, match="vae_bias auxiliary payload is no longer supported"):
        apply_auxiliary_payload_to_compressed_model(
            root,
            {"vae_bias::layer": torch.zeros_like(layer.bias)},
        )
