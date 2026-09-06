"""Task 7 decoder execution / runtime contract tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.config.configs import DistillLossConfig, DistillRuntimeConfig, teacher_required
from train_utils.decoder_execution import (
    DecodeCapacityError,
    apply_decoder_execution_plan,
    enable_vae_linear_by_execution_plan,
    iter_decode_group_size_candidates,
    is_retryable_decode_capacity_error,
    resolve_decoder_execution_plan,
    run_with_decode_group_size_fallback,
)
from train_utils.distill_teacher import distill_loss_requires_teacher, resolve_distill_teacher_required
from e2e_common.e2e_args import needs_teacher


def _make_decoder(*, latent_dim: int = 9, codebook_dim: int = 4) -> Decoder:
    return Decoder(
        in_dim=latent_dim,
        out_dim=codebook_dim,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    ).to(dtype=torch.float32)


def _make_single_stage_vae_linear() -> VAELinear:
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
        decoder=_make_decoder(),
        codebook_dim=4,
        transpose=False,
    )


def _make_compatible_multi_stage() -> VAELinear:
    part0 = torch.tensor(
        [
            [[True, False, True, False, True, False, True, False, True]],
            [[False, True, False, True, False, True, False, True, False]],
        ],
        dtype=torch.bool,
    )
    part1 = torch.tensor(
        [
            [[True, True, False, False, True, True, False, False, True]],
            [[False, False, True, True, False, False, True, True, False]],
        ],
        dtype=torch.bool,
    )
    stage_decoders = [
        [_make_decoder(), _make_decoder()],
        [_make_decoder(), _make_decoder()],
    ]
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=None,
        decoder=None,
        stage_vq_weights=[[part0, part1], [~part0, ~part1]],
        stage_decoders=stage_decoders,
        codebook_dim=4,
        stage_codebook_dims=[4, 4],
        transpose=False,
        parallel_parts=2,
        parallel_rows=1,
        parallel_cols=2,
    )


def _make_mismatched_codebook_multi_stage() -> VAELinear:
    """Compatible payload with intentionally incompatible stage_codebook_dims metadata."""
    layer = _make_compatible_multi_stage()
    # Force pack incompatibility without rebuilding VQ storage (contract is metadata-level).
    layer.stage_codebook_dims = [4, 8]
    return layer


def test_single_decoder_plan_is_serial_and_does_not_pack():
    layer = _make_single_stage_vae_linear()
    plan = resolve_decoder_execution_plan(layer, mode="trainable_decoder")
    assert plan.use_packed is False
    assert plan.reason == "single_decoder_serial"
    apply_decoder_execution_plan(layer, plan)
    assert getattr(layer, "_parallel_stage_decoder", None) is None
    assert layer.parallel_stage_decode is False


def test_compatible_multi_stage_plan_packs():
    layer = _make_compatible_multi_stage()
    plan = enable_vae_linear_by_execution_plan(layer, mode="trainable_decoder")
    assert plan.use_packed is True
    assert plan.reason == "compatible_multi_stage_pack"
    assert getattr(layer, "_parallel_stage_decoder", None) is not None
    assert layer.parallel_stage_decode is True


def test_mismatched_codebook_normal_fallback_serial():
    layer = _make_mismatched_codebook_multi_stage()
    plan = enable_vae_linear_by_execution_plan(layer, mode="trainable_decoder")
    assert plan.use_packed is False
    assert plan.pack_compatible is False
    assert plan.reason.startswith("fallback_serial:")
    assert getattr(layer, "_parallel_stage_decoder", None) is None
    assert layer.trainable_decode is True
    assert layer.parallel_stage_decode is False


def test_mismatched_codebook_sparse_bit_hard_error():
    layer = _make_mismatched_codebook_multi_stage()
    with pytest.raises(RuntimeError, match="Sparse Bit requires packed decode"):
        resolve_decoder_execution_plan(layer, mode="sparse_bit")


def test_decoder_sparse_bit_single_stage_keeps_decoder_trainable():
    layer = _make_single_stage_vae_linear()
    plan = enable_vae_linear_by_execution_plan(layer, mode="decoder_sparse_bit")
    assert plan.use_packed is False
    assert layer.trainable_decode is True
    assert layer.parallel_stage_decode is False
    decoder = layer.get_stage_part_decoder(stage_idx=0, part_idx=0)
    assert all(param.requires_grad for param in decoder.parameters())


def test_decoder_sparse_bit_compatible_multi_stage_packs_and_keeps_decoder_trainable():
    layer = _make_compatible_multi_stage()
    plan = enable_vae_linear_by_execution_plan(layer, mode="decoder_sparse_bit")
    assert plan.use_packed is True
    assert plan.reason == "compatible_multi_stage_pack"
    packed = getattr(layer, "_parallel_stage_decoder", None)
    assert packed is not None
    assert layer.parallel_stage_decode is True
    assert all(param.requires_grad for param in packed.parameters())


def test_decoder_sparse_bit_mismatched_multi_stage_hard_error():
    layer = _make_mismatched_codebook_multi_stage()
    with pytest.raises(RuntimeError, match="Sparse Bit requires packed decode"):
        resolve_decoder_execution_plan(layer, mode="decoder_sparse_bit")


def test_group_size_candidates_start_from_min_8_or_n():
    assert iter_decode_group_size_candidates(16) == (8, 4, 2, 1)
    assert iter_decode_group_size_candidates(8) == (8, 4, 2, 1)
    assert iter_decode_group_size_candidates(6) == (6, 4, 2, 1)
    assert iter_decode_group_size_candidates(3) == (3, 2, 1)
    assert iter_decode_group_size_candidates(1) == (1,)
    assert iter_decode_group_size_candidates(16, initial_group_size=4) == (4, 2, 1)
    assert iter_decode_group_size_candidates(3, initial_group_size=8) == (3, 2, 1)
    with pytest.raises(ValueError, match="initial_group_size"):
        iter_decode_group_size_candidates(8, initial_group_size=0)


def test_group_size_fallback_retries_only_capacity_errors():
    seen = []

    def flaky(group_size: int):
        seen.append(int(group_size))
        if group_size > 2:
            raise DecodeCapacityError(f"capacity at {group_size}")
        return f"ok-{group_size}"

    result, resolved = run_with_decode_group_size_fallback(flaky, num_targets=10)
    assert result == "ok-2"
    assert resolved.group_size == 2
    assert resolved.fallback_reason == "capacity_oom_fallback"
    assert resolved.attempted == (8, 4, 2)
    assert seen == [8, 4, 2]


def test_group_size_fallback_does_not_retry_metadata_errors():
    def boom(_group_size: int):
        raise ValueError("stage codebook dims are not identical")

    with pytest.raises(ValueError, match="stage codebook"):
        run_with_decode_group_size_fallback(boom, num_targets=10)


def test_is_retryable_capacity_error_contract():
    class OutOfMemoryError(RuntimeError):
        pass

    assert is_retryable_decode_capacity_error(DecodeCapacityError("capacity"))
    assert is_retryable_decode_capacity_error(OutOfMemoryError("CUDA out of memory"))
    assert is_retryable_decode_capacity_error(RuntimeError("CUDA error: out of memory"))
    assert not is_retryable_decode_capacity_error(OutOfMemoryError("business failure"))
    assert not is_retryable_decode_capacity_error(ValueError("illegal codebook"))
    assert not is_retryable_decode_capacity_error(RuntimeError("shape metadata mismatch"))


def test_group_fallback_wrappers_materialize_generator_inputs(monkeypatch):
    decode_seen = []
    prime_seen = []

    def fake_decode(named_targets, **_kwargs):
        items = list(named_targets)
        decode_seen.append(len(items))
        return [f"decoded-{idx}" for idx, _ in enumerate(items)]

    def fake_prime(named_targets, **_kwargs):
        items = list(named_targets)
        prime_seen.append(len(items))
        return {"total": len(items), "warmed": len(items), "skipped": 0, "failed": 0}

    monkeypatch.setattr(
        "litebsq.vae_linear_prewarm.decode_named_vae_linear_weights",
        fake_decode,
    )
    monkeypatch.setattr(
        "litebsq.vae_linear_prewarm.prime_named_vae_linear_cache",
        fake_prime,
    )

    def decode_gen():
        yield object()
        yield object()
        yield object()

    def prime_gen():
        yield object()
        yield object()

    from train_utils.decoder_execution import (
        decode_named_vae_linear_weights_with_group_fallback,
        prime_named_vae_linear_cache_with_group_fallback,
    )

    decoded, decode_resolved = decode_named_vae_linear_weights_with_group_fallback(decode_gen())
    primed, prime_resolved = prime_named_vae_linear_cache_with_group_fallback(prime_gen())

    assert decode_seen == [3]
    assert prime_seen == [2]
    assert len(decoded) == 3
    assert primed["warmed"] == 2
    assert decode_resolved.num_targets == 3
    assert prime_resolved.num_targets == 2
    assert decode_resolved.group_size == 3
    assert prime_resolved.group_size == 2


def test_distill_runtime_config_dp_rejects_non_auto_layer_map():
    cfg = DistillRuntimeConfig(parallel_mode="dp", layer_device_map="cuda:0=0-15")
    with pytest.raises(ValueError, match="parallel_mode=dp requires layer_device_map=auto"):
        cfg.validate()


def test_distill_runtime_config_dp_accepts_auto_layer_map():
    cfg = DistillRuntimeConfig(parallel_mode="dp", layer_device_map="auto")
    cfg.validate()
    assert cfg.layer_device_map == "auto"


@pytest.mark.parametrize("loss_type", ["sft", "kl", "kl_top", "kd", "kd_top"])
@pytest.mark.parametrize("hidden", [0.0, 0.1])
@pytest.mark.parametrize("pre_mlp", [0.0, 0.2])
def test_teacher_required_helpers_parity(loss_type, hidden, pre_mlp):
    canonical = resolve_distill_teacher_required(
        loss_type=loss_type,
        hidden_loss_weight=hidden,
        pre_mlp_hidden_loss_weight=pre_mlp,
    )
    loss_cfg = DistillLossConfig(
        loss_type=loss_type,
        hidden_loss_weight=hidden,
        pre_mlp_hidden_loss_weight=pre_mlp,
    )
    assert teacher_required(loss_cfg) is canonical
    # needs_teacher / distill_loss_requires_teacher are loss-type-only thin adapters.
    loss_only = distill_loss_requires_teacher(loss_type)
    assert needs_teacher(loss_type) is loss_only
    if hidden == 0.0 and pre_mlp == 0.0:
        assert loss_only is canonical
    else:
        assert canonical is True
