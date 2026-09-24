"""Exact local RHT, VAELinear coordinates, gradients, cache and v6 state."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from rotation.hadamard_utils import get_hadK
from rotation.weight_preprocess import build_weight_rotation, rotation_spec
from train_utils import checkpoint_v6 as v6
from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity
from train_utils.config.cli import parse_cat_cli


def _dense_hadamard(n: int) -> torch.Tensor:
    small, k = get_hadK(n)
    sylvester = torch.ones(1, 1, dtype=torch.float64)
    while sylvester.shape[0] < n // k:
        sylvester = torch.cat((torch.cat((sylvester, sylvester), 1),
                               torch.cat((sylvester, -sylvester), 1)), 0)
    small = torch.ones(1, 1, dtype=torch.float64) if small is None else small.double()
    return torch.kron(small, sylvester) / n**0.5


def _dense_rotation(rotation, axis: str) -> torch.Tensor:
    n = rotation.in_features if axis == "input" else rotation.out_features
    size = rotation.block_size or n
    had = torch.block_diag(*([_dense_hadamard(size)] * (n // size)))
    signs = getattr(rotation, f"{axis}_signs").detach().cpu().double()
    return had * signs[None, :]


@pytest.mark.parametrize("shape,block", [((8, 16), 4), ((8, 16), 0), ((12, 24), 0), ((24, 48), 12)])
def test_transform_matches_dense_and_preserves_norm(shape, block):
    torch.manual_seed(31)
    state = torch.random.get_rng_state().clone()
    r = build_weight_rotation(*shape, rotation_spec("two_sided", block, 31, "proj"))
    assert torch.equal(state, torch.random.get_rng_state())
    w = torch.randn(*shape, dtype=torch.float64)
    u, v = _dense_rotation(r, "output"), _dense_rotation(r, "input")
    torch.testing.assert_close(r(w), u @ w @ v.T, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(r(r(w), inverse=True), w, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(r(w).norm(), w.norm(), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(u.T @ u, torch.eye(shape[0], dtype=w.dtype), atol=2e-12, rtol=0)


def test_inverse_gradcheck_and_invalid_shape():
    r = build_weight_rotation(12, 8, rotation_spec("two_sided", 0, 37, "proj"))
    w = torch.randn(12, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: r(x, inverse=True), (w,), fast_mode=True)
    with pytest.raises(ValueError, match="shape mismatch"):
        r(torch.randn(8, 12))
    with pytest.raises(ValueError, match="No exact Hadamard"):
        build_weight_rotation(32, 4064, rotation_spec("two_sided", 0, 31, "proj"))
    with pytest.raises(ValueError, match="must divide"):
        build_weight_rotation(8, 12, rotation_spec("two_sided", 8, 31, "proj"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape,block", [((32, 64), 32), ((24, 32), 0)])
def test_cuda_forward_backward_and_bf16(shape, block):
    torch.manual_seed(31)
    r = build_weight_rotation(*shape, rotation_spec("two_sided", block, 31, "proj")).cuda()
    w = torch.randn(*shape, device="cuda", requires_grad=True)
    restored = r(r(w), inverse=True)
    torch.testing.assert_close(restored, w, atol=2e-6, rtol=2e-6)
    restored.square().sum().backward()
    torch.testing.assert_close(w.grad, 2 * w, atol=1e-5, rtol=1e-5)
    wb = w.detach().bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rb = r(r(wb), inverse=True)
    relative = (rb.float() - wb.float()).norm() / wb.float().norm()
    assert float(relative) < 0.007  # Two BF16 materializations, FP32 transform accumulation.
    reference = _dense_rotation(r.cpu(), "output") @ w.detach().cpu().double() @ _dense_rotation(r, "input").T
    torch.testing.assert_close(r.cuda()(w).cpu().double(), reference, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_qwen_mlp_dimension_roundtrip():
    # Full production dimensions, not a power-of-two cropped approximation.
    r = build_weight_rotation(4096, 12288, rotation_spec("two_sided", 0, 31, "down_proj")).cuda()
    with torch.no_grad():
        w = torch.randn(4096, 12288, device="cuda")
        restored = r(r(w), inverse=True)
        relative = (restored - w).norm() / w.norm()
        assert float(relative) < 2e-6
    h, k = get_hadK(12288)
    torch.testing.assert_close(h.double().T @ h.double(),
                               torch.eye(k, dtype=torch.float64) * k, atol=0, rtol=0)


def _make_layer(*, transpose=False, axis="input", mode="two_sided"):
    torch.manual_seed(31)
    out_features = 10 if axis == "output" else 8
    in_features = 10 if axis == "input" else 8
    extras = {}
    if axis == "input":
        extras.update(protected_input_indices=torch.tensor([1, 7]),
                      protected_input_weight=torch.randn(2, out_features))
    elif axis == "output":
        extras.update(protected_output_indices=torch.tensor([1, 7]),
                      protected_output_weight=torch.randn(2, in_features))
    decoders = [[Decoder(in_dim=8, out_dim=4, decoder_type="linear", num_models=1)
                 for _ in range(2)] for _ in range(2)]
    bits = [[torch.randint(0, 2, (8, 1, 8)).bool() for _ in range(2)] for _ in range(2)]
    return VAELinear(
        in_features=in_features, out_features=out_features, bias=nn.Parameter(torch.randn(out_features)),
        original_weight=torch.randn(out_features, in_features),
        stage_vq_weights=bits, stage_decoders=decoders, stage_codebook_dims=[4, 4],
        codebook_dim=4, transpose=transpose, parallel_parts=2, parallel_rows=1, parallel_cols=2,
        compressed_out_features=8, compressed_in_features=8,
        low_rank_a=torch.randn(out_features, 3), low_rank_b=torch.randn(3, in_features),
        weight_rotation_spec=rotation_spec(mode, 4, 31, "proj"), **extras,
    )


def _expected_weight(layer):
    compressed = layer._decode_compressed_weight(dtype=torch.float32)
    if layer.weight_rotation is not None:
        u = _dense_rotation(layer.weight_rotation, "output").float()
        v = _dense_rotation(layer.weight_rotation, "input").float()
        compressed = u.T @ compressed @ v
    full = torch.empty(layer.out_features, layer.in_features)
    if layer.protected_input_indices is not None:
        keep = torch.ones(layer.in_features, dtype=torch.bool)
        keep[layer.protected_input_indices.long()] = False
        full[:, keep] = compressed
        full[:, ~keep] = layer.protected_input_weight.T
    elif layer.protected_output_indices is not None:
        keep = torch.ones(layer.out_features, dtype=torch.bool)
        keep[layer.protected_output_indices.long()] = False
        full[keep] = compressed
        full[~keep] = layer.protected_output_weight
    else:
        full.copy_(compressed)
    return full + layer.low_rank_a @ layer.low_rank_b


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("axis", ["none", "input", "output"])
def test_vae_decode_protection_lora_cache_original_and_gradient(transpose, axis):
    layer = _make_layer(transpose=transpose, axis=axis)
    x = torch.randn(3, layer.in_features)
    with torch.no_grad():
        expected = _expected_weight(layer)
        torch.testing.assert_close(layer._decode_weight(dtype=torch.float32), expected, atol=3e-6, rtol=3e-6)
        layer.prime_decoded_weight_cache(dtype=torch.float32)
        torch.testing.assert_close(layer(x), F.linear(x, expected, layer.bias), atol=5e-6, rtol=5e-6)
        layer.clear_decoded_weight_cache()
        layer.pack_parallel_stage_decoder_(trainable=False)
        torch.testing.assert_close(layer._decode_weight(dtype=torch.float32), expected, atol=5e-6, rtol=5e-6)
        layer.set_temporary(False)
        torch.testing.assert_close(layer(x), F.linear(x, layer.original_weight, layer.bias), atol=0, rtol=0)
        layer.set_temporary(True)
    layer.cache_decoded_weight = False
    layer.trainable_decode = True
    layer.requires_grad_(True)
    x.requires_grad_(True)
    loss = layer(x).square().mean()
    loss.backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    grads = [p.grad for name, p in layer.named_parameters() if "decoder" in name]
    assert grads and any(g is not None and float(g.norm()) > 0 for g in grads)
    assert all(bool(torch.isfinite(g).all()) for g in grads if g is not None)
    assert all(not buffer.requires_grad for buffer in layer.weight_rotation.buffers())


@pytest.mark.parametrize("mode", ["none", "two_sided"])
def test_v6_full_checkpoint_roundtrip(tmp_path, mode):
    host = nn.Module()
    host.proj = _make_layer(transpose=True, axis="input", mode=mode)
    host.proj.pack_parallel_stage_decoder_(trainable=False)
    out = str(tmp_path / mode)
    v6.save_v6_full_checkpoint(
        host, out, checkpoint_kind="final_model", compressed_targets=["proj"],
        pending_dense_targets=[], skip_targets=[], train_mode="decoder", completed_categories=["q_proj"],
    )
    skeleton = nn.Module()
    skeleton.proj = nn.Linear(host.proj.in_features, host.proj.out_features, bias=True)
    loaded, _, _ = v6.load_v6_full_checkpoint_into_model(skeleton, out, expected_kind="final_model")
    torch.testing.assert_close(host.proj._decode_weight(dtype=torch.float32),
                               loaded.proj._decode_weight(dtype=torch.float32), atol=0, rtol=0)
    if mode == "two_sided":
        assert loaded.proj.weight_rotation.to_spec() == host.proj.weight_rotation.to_spec()
        assert torch.equal(loaded.proj.weight_rotation.input_signs, host.proj.weight_rotation.input_signs)
        loaded.proj.weight_rotation.input_signs[0] = 0
        with pytest.raises(ValueError, match="other than"):
            v6.refresh_vae_linear_runtime_after_state_load(loaded)
    else:
        assert loaded.proj.weight_rotation is None
        assert not any("weight_rotation" in key for key in host.state_dict())


def test_cli_defaults_modes_and_weighted_loss_rejection():
    base = ["--model_path", "Qwen/Qwen3-8B", "--compression_categories", "q_proj,down_proj"]
    cfg = parse_cat_cli(base)
    assert cfg.core_template.weight_rotation == "none"
    cfg = parse_cat_cli(base + ["--weight_rotation", "two_sided", "--weight_rotation_block_size", "0"])
    assert cfg.core_template.weight_rotation_block_size == 0
    compression, _ = cfg.resolve_category_config(cfg.compression_categories[0])
    assert compression.core.weight_rotation == "two_sided"
    for extra in (["--rot_llm"], ["--recon_loss_type", "default=amse"],
                  ["--recon_loss_type", "default=mse,cat:down_proj=wa_mse"]):
        with pytest.raises(SystemExit):
            parse_cat_cli(base + ["--weight_rotation", "two_sided"] + extra)


def test_disabled_rotation_preserves_legacy_runtime_identity():
    kwargs = dict(cat_args=SimpleNamespace(seed=31), resolved_category_cfgs={},
                  compression_categories=[], target_layers="all", skip_layers=[], transpose_modules=[])
    old = build_cat_cross_category_runtime_identity(vae_args=SimpleNamespace(normalize_weight=True), **kwargs)
    args = SimpleNamespace(normalize_weight=True, weight_rotation="none", weight_rotation_block_size=32)
    assert build_cat_cross_category_runtime_identity(vae_args=args, **kwargs) == old
    args.weight_rotation = "two_sided"
    assert build_cat_cross_category_runtime_identity(vae_args=args, **kwargs) != old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("axis", ["input", "output"])
def test_production_cat_two_stage_protected_roundtrip_and_gradients(tmp_path, axis):
    from experiments.weight_rotation_ab import _host_for_weight, train_patch
    from litebsq.vae_linear_prewarm import prime_model_vae_linear_cache

    torch.manual_seed(41)
    weight = torch.randn(64, 128) * 0.03
    name = "model.layers.0.self_attn.q_proj"
    host, layer, metrics = train_patch(
        weight, module_name=name, category="q_proj", mode="two_sided_32", seed=31,
        steps=2, batch_size=128, protected_count=32, protected_axis=axis,
    )
    assert layer.weight_rotation is not None and layer.residual_stages == 2
    assert layer.weight_rotation.to_spec() == rotation_spec("two_sided", 32, 31, name)
    assert layer.protected_channel_quant_format == "int8"
    assert metrics["rotation_state_bytes"] > 0
    with torch.no_grad():
        expected = layer._decode_weight(dtype=torch.float32)
        # Protection is still in the original channel basis, outside inversion.
        if axis == "input":
            indices = layer.protected_input_indices.long()
            protected = layer._decode_protected_input_weight_rows(dtype=torch.float32, device=expected.device).T
            torch.testing.assert_close(expected[:, indices], protected, atol=0, rtol=0)
        else:
            indices = layer.protected_output_indices.long()
            protected = layer._decode_protected_output_weight_rows(dtype=torch.float32, device=expected.device)
            torch.testing.assert_close(expected[indices], protected, atol=0, rtol=0)
        stats = prime_model_vae_linear_cache(host, dtype=torch.float32, group_size=2, clear_existing=True)
        assert stats["warmed"] == 1
        torch.testing.assert_close(layer._cached_weight, expected, atol=3e-5, rtol=3e-5)
    checkpoint = str(tmp_path / axis)
    v6.save_v6_full_checkpoint(host, checkpoint, checkpoint_kind="final_model", compressed_targets=[name],
                               pending_dense_targets=[], skip_targets=[], train_mode="decoder", completed_categories=["q_proj"])
    skeleton, _ = _host_for_weight(weight, name)
    loaded, _, _ = v6.load_v6_full_checkpoint_into_model(skeleton, checkpoint, expected_kind="final_model")
    loaded_layer = loaded.get_submodule(name).cuda()
    with torch.no_grad():
        torch.testing.assert_close(loaded_layer._decode_weight(dtype=torch.float32), expected, atol=0, rtol=0)
    loaded_layer.cache_decoded_weight = False
    loaded_layer.trainable_decode = True
    loaded_layer.requires_grad_(True)
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    target = torch.randn(3, 64, device="cuda")
    loss = (loaded_layer(x) - target).square().mean()
    loss.backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    gradients = [p.grad for n, p in loaded_layer.named_parameters() if "decoder" in n]
    assert gradients and any(g is not None and float(g.norm()) > 0 for g in gradients)
    assert all(bool(torch.isfinite(g).all()) for g in gradients if g is not None)
