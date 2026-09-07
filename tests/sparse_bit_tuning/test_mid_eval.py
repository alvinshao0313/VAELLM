import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.mid_eval import temporary_inference_decode_mode
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.optimizer import SparseBitCompositeOptimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _decoder():
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


def _bits(inv=False):
    x = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )
    return ~x if inv else x


def _parallel_layer():
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=None,
        decoder=None,
        stage_vq_weights=[_bits(False), _bits(True)],
        stage_decoders=[_decoder(), _decoder()],
        codebook_dim=4,
        stage_codebook_dims=[4, 4],
        transpose=False,
        parallel_parts=1,
    )


def test_mid_eval_restores_pure_bit_parallel_decode_without_unfreezing_decoder():
    device = torch.device("cuda:0")
    layer = _parallel_layer().to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
    root = nn.Module()
    root.add_module("layer", layer)
    manager = SparseBitTuningManager(
        root_model=root,
        targets=[("layer", layer)],
        target_devices={"layer": device},
        training_seed=4,
        config=SparseBitTuningConfig(enabled=True, active_ratio=0.5, optimizer="rms_sgd", bit_lr=0.1),
        streaming=False,
    )
    manager.configure_schedule(total_optimizer_steps=2)
    manager.initialize_scores()
    assert layer._parallel_stage_decoder is not None
    assert all(not p.requires_grad for p in layer._parallel_stage_decoder.parameters())

    with temporary_inference_decode_mode(root, cache_decoded_weight=False):
        assert not layer.trainable_decode
        assert all(not p.requires_grad for p in layer._parallel_stage_decoder.parameters())

    assert layer.trainable_decode
    assert layer.parallel_stage_decode
    assert all(not p.requires_grad for p in layer._parallel_stage_decoder.parameters())
    assert getattr(layer, "_sparse_bit_binding", None) is not None


def test_composite_eval_state_offload_restore_preserves_original_devices():
    device = torch.device("cuda:0")
    main_param = nn.Parameter(torch.tensor([1.0], device=device))
    main_opt = torch.optim.AdamW([main_param], lr=1e-3)
    main_param.grad = torch.tensor([0.25], device=device)
    main_opt.step()
    main_opt.zero_grad(set_to_none=True)

    layer = _parallel_layer().to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
    root = nn.Module()
    root.add_module("layer", layer)
    manager = SparseBitTuningManager(
        root_model=root,
        targets=[("layer", layer)],
        target_devices={"layer": device},
        training_seed=5,
        config=SparseBitTuningConfig(enabled=True, active_ratio=0.5, optimizer="adam", bit_lr=0.02),
        streaming=False,
    )
    manager.configure_schedule(total_optimizer_steps=2)
    manager.initialize_scores()
    for score in manager.score_module.score_chunks:
        score.grad = torch.ones_like(score)
    manager.bit_optimizer.step_scores(optimizer_step_in_round=1)

    composite = SparseBitCompositeOptimizer(
        main_optimizer=main_opt,
        bit_manager=manager.bit_optimizer,
        step_callback=lambda: None,
    )
    original_bit_devices = [tensor.device for tensor in manager.bit_optimizer.state_tensors()]
    assert original_bit_devices and all(dev.type == "cuda" for dev in original_bit_devices)
    assert any(
        torch.is_tensor(v) and v.device.type == "cuda"
        for state in main_opt.state.values()
        for v in state.values()
    )

    restore_ticket = composite.offload_training_state_for_eval()
    assert all(t.device.type == "cpu" for t in manager.bit_optimizer.state_tensors())
    assert all(
        not torch.is_tensor(v) or v.device.type == "cpu"
        for state in main_opt.state.values()
        for v in state.values()
    )
    composite.restore_training_state_after_eval(restore_ticket)
    assert [tensor.device for tensor in manager.bit_optimizer.state_tensors()] == original_bit_devices
    assert any(
        torch.is_tensor(v) and v.device == device
        for state in main_opt.state.values()
        for v in state.values()
    )


def _prepare_no_grad_parallel_layer(*, cache_decoded_weight: bool):
    device = torch.device("cuda:0")
    layer = _parallel_layer().to(device=device, dtype=torch.bfloat16)
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
    layer.disable_trainable_decode()
    layer.cache_decoded_weight = bool(cache_decoded_weight)
    return layer


def test_no_cache_no_grad_eval_prefers_packed_before_dense_grouped_vq(monkeypatch):
    import litebsq.vae_linear as vae_linear_mod

    layer = _prepare_no_grad_parallel_layer(cache_decoded_weight=False)
    monkeypatch.setattr(vae_linear_mod, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(vae_linear_mod, "packed_symmetric_decoder_supports_fuse", lambda _decoder: True)

    packed_called = False
    dense_materialized = False
    logical_shape = layer._parallel_stage_grouped_vq_logical_shape
    expected_models = int(layer.residual_stages) * int(layer.parallel_parts)
    expected_rows = int(logical_shape[0])
    synthetic_stage_out = torch.zeros(
        (expected_rows, expected_models, 4),
        device=next(layer.parameters()).device,
        dtype=torch.bfloat16,
    )
    restored = torch.zeros(
        (int(layer.out_features), int(layer.in_features)),
        device=synthetic_stage_out.device,
        dtype=torch.bfloat16,
    )

    def _spy_grouped_vq(*_args, **_kwargs):
        nonlocal dense_materialized
        dense_materialized = True
        raise AssertionError("dense grouped VQ should not be materialized on packed-first eval")

    def _spy_packed(*_args, **_kwargs):
        nonlocal packed_called
        packed_called = True
        return synthetic_stage_out

    def _spy_restore(*_args, **_kwargs):
        return restored

    monkeypatch.setattr(layer, "_get_parallel_stage_grouped_vq", _spy_grouped_vq)
    monkeypatch.setattr(layer, "_try_packed_u8_parallel_stage_decode", _spy_packed)
    monkeypatch.setattr(layer, "_restore_stage_part_flats_fast", _spy_restore)

    with torch.no_grad():
        out = layer._decode_split_weight_parallel_stages(dtype=torch.bfloat16)

    assert packed_called is True
    assert dense_materialized is False
    assert out.shape == (int(layer.out_features), int(layer.in_features))


def test_no_cache_no_grad_forward_clears_dense_fallback_runtime_scratch(monkeypatch):
    device = torch.device("cuda:0")
    layer = _prepare_no_grad_parallel_layer(cache_decoded_weight=False)
    decoded = torch.zeros(
        (int(layer.out_features), int(layer.in_features)),
        device=device,
        dtype=torch.bfloat16,
    )

    def _fake_decode_weight(*, dtype, include_low_rank=False):
        del include_low_rank
        layer._parallel_stage_grouped_vq_weight = torch.ones(1, device=device, dtype=dtype)
        layer._parallel_stage_grouped_vq_runtime = torch.ones(2, device=device, dtype=dtype)
        layer._parallel_stage_grouped_vq_packed_runtime = torch.ones(3, device=device, dtype=torch.uint8)
        layer._protected_residual_parallel_grouped_vq_runtime = torch.ones(4, device=device, dtype=dtype)
        layer._cached_sparse_residual_values = torch.ones(5, device=device, dtype=dtype)
        return decoded

    monkeypatch.setattr(layer, "_decode_weight", _fake_decode_weight)
    x = torch.randn(2, int(layer.in_features), device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = layer(x)

    assert out.shape == (2, int(layer.out_features))
    assert layer._cached_weight is None
    assert layer._parallel_stage_grouped_vq_weight is None
    assert layer._parallel_stage_grouped_vq_runtime is None
    assert layer._parallel_stage_grouped_vq_packed_runtime is None
    assert layer._protected_residual_parallel_grouped_vq_runtime is None
    assert layer._cached_sparse_residual_values is None


def test_cached_no_grad_inference_keeps_dense_fused_first_contract(monkeypatch):
    import litebsq.vae_linear as vae_linear_mod

    layer = _prepare_no_grad_parallel_layer(cache_decoded_weight=True)
    monkeypatch.setattr(vae_linear_mod, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(vae_linear_mod, "packed_symmetric_decoder_supports_fuse", lambda _decoder: True)

    call_order = []
    reconstructed = torch.zeros(
        (int(layer.out_features), int(layer.in_features)),
        device=next(layer.parameters()).device,
        dtype=torch.bfloat16,
    )

    def _spy_grouped_vq(*_args, **_kwargs):
        call_order.append("_get_parallel_stage_grouped_vq")
        return torch.zeros(1, device=reconstructed.device, dtype=torch.bfloat16)

    def _spy_fused(*_args, **_kwargs):
        call_order.append("_try_fused_parallel_stage_decode")
        return reconstructed

    def _spy_packed(*_args, **_kwargs):
        call_order.append("_try_packed_u8_parallel_stage_decode")
        raise AssertionError("cached no-grad inference must not fall through to packed decode")

    monkeypatch.setattr(layer, "_get_parallel_stage_grouped_vq", _spy_grouped_vq)
    monkeypatch.setattr(layer, "_try_fused_parallel_stage_decode", _spy_fused)
    monkeypatch.setattr(layer, "_try_packed_u8_parallel_stage_decode", _spy_packed)

    with torch.no_grad():
        out = layer._decode_split_weight_parallel_stages(dtype=torch.bfloat16)

    assert call_order == [
        "_get_parallel_stage_grouped_vq",
        "_try_fused_parallel_stage_decode",
    ]
    assert out is reconstructed
