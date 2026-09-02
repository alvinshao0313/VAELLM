import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.low_rank_scope import LOW_RANK_SCOPE_FULL
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.full_lora_proxy import (
    FullCompressedPeftProxy,
    build_full_compressed_peft_model,
    extract_full_proxy_low_rank_payloads,
    iter_named_full_compressed_peft_proxies,
)


def _vae_linear():
    latent_dim = 9
    codebook_dim = 4
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
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=decoder,
        codebook_dim=codebook_dim,
        transpose=False,
    )


def _root(layer):
    root = nn.Module()
    root.add_module("layer", layer)
    return root


def test_full_proxy_keeps_vae_base_and_only_lora_is_trainable():
    layer = _vae_linear()
    root = _root(layer)
    out = build_full_compressed_peft_model(
        root,
        selected_modules=[("layer", layer)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    assert out is not root
    assert hasattr(out, "get_base_model")
    refs = list(iter_named_full_compressed_peft_proxies(root))
    assert len(refs) == 1
    name, proxy = refs[0]
    assert name == "layer"
    assert isinstance(proxy, FullCompressedPeftProxy)
    assert proxy.base_layer is layer
    assert isinstance(root.layer, FullCompressedPeftProxy)
    trainable = [(name, param) for name, param in root.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora_" in name for name, _ in trainable)
    assert all(not param.requires_grad for param in proxy.base_layer.parameters())

    x = torch.randn(3, 4)
    y = root.layer(x)
    y.sum().backward()
    assert any(param.grad is not None for _name, param in trainable)


def test_full_proxy_existing_payload_round_trip():
    layer = _vae_linear()
    rank = 2
    low_rank_a = torch.randn(4, rank)
    low_rank_b = torch.randn(rank, 4)
    layer.low_rank_scope = LOW_RANK_SCOPE_FULL
    layer.register_parameter("low_rank_a", nn.Parameter(low_rank_a.clone(), requires_grad=False))
    layer.register_parameter("low_rank_b", nn.Parameter(low_rank_b.clone(), requires_grad=False))
    root = _root(layer)
    build_full_compressed_peft_model(
        root,
        selected_modules=[("layer", layer)],
        initial_low_rank_payloads={"layer": (low_rank_a.clone(), low_rank_b.clone())},
        rank=rank,
        alpha=4.0,
        dropout=0.0,
    )
    assert layer.low_rank_a is None and layer.low_rank_b is None
    payloads = extract_full_proxy_low_rank_payloads(root, module_names=["layer"])
    out_a, out_b = payloads["layer"]
    assert torch.allclose(out_a, low_rank_a)
    assert torch.allclose(out_b, low_rank_b)
