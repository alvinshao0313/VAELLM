from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from litebsq.llm_vae import Decoder
from litebsq.vae_linear import VAELinear
from mix_bit.checkpoint_pool import CheckpointSource, ModuleCandidate
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    STATE_DICT_FILENAME,
    save_model_checkpoint,
)


MODULE_NAME = "model.layers.0.q_proj"


def _make_decoder(cdim: int) -> Decoder:
    return Decoder(
        in_dim=cdim,
        out_dim=cdim,
        hidden_dim=16,
        num_res_blocks=0,
        decoder_type="linear",
        norm_type="group",
        activation_type="swish",
    )


def _make_vae_linear(
    *,
    in_features: int = 8,
    out_features: int = 8,
    cdim: int = 4,
    residual_stages: int = 2,
    with_bias: bool = True,
    with_original: bool = False,
) -> VAELinear:
    n_blocks = (in_features * out_features) // cdim
    logical = (n_blocks, 1, cdim)
    stages = []
    decoders = []
    for _ in range(residual_stages):
        stages.append(torch.randint(0, 2, logical, dtype=torch.bool))
        decoders.append(_make_decoder(cdim))
    bias = nn.Parameter(torch.zeros(out_features)) if with_bias else None
    original = nn.Parameter(torch.randn(out_features, in_features)) if with_original else None
    return VAELinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        original_weight=original,
        stage_vq_weights=stages,
        stage_decoders=decoders,
        codebook_dim=cdim,
        stage_codebook_dims=[cdim] * residual_stages,
        transpose=False,
        parallel_parts=1,
    )


def _make_nested_host(module: VAELinear) -> nn.Module:
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = nn.ModuleList([nn.Module()])
    host.model.layers[0].q_proj = module
    host.model.layers[0].k_proj = nn.Linear(8, 8, bias=False)
    return host


def _checkpoint_fixture(tmp_path: Path) -> tuple[ModuleCandidate, dict[str, torch.Tensor], nn.Module]:
    source_module = _make_vae_linear(with_bias=True, with_original=False)
    host = _make_nested_host(source_module)
    ckpt_dir = tmp_path / "tiny_ckpt"
    save_model_checkpoint(host, str(ckpt_dir), save_config=False, unload_vae_original_weights=True)

    meta = json.loads((ckpt_dir / META_FILENAME).read_text(encoding="utf-8"))
    full_state = torch.load(ckpt_dir / STATE_DICT_FILENAME, map_location="cpu", weights_only=True)
    specs = [s for s in meta["converted_modules"] if s["name"] == MODULE_NAME]
    assert len(specs) == 1
    spec = specs[0]
    assert spec["has_original_weight"] is False

    prefix = f"{MODULE_NAME}."
    compact_state = {k: v for k, v in full_state.items() if k.startswith(prefix)}
    assert compact_state

    source = CheckpointSource(
        category="q_proj",
        module_suffix="q_proj",
        mode_name="b16d4s2",
        trial_root=str(tmp_path / "trial"),
        candidate_meta_path=str(tmp_path / "candidate_meta.json"),
        compact_state_path=str(ckpt_dir / STATE_DICT_FILENAME),
        candidate_meta_sha256="a" * 64,
        compact_state_sha256="b" * 64,
    )
    candidate = ModuleCandidate(
        module_name=MODULE_NAME,
        category="q_proj",
        module_suffix="q_proj",
        block_index=0,
        mode_name="b16d4s2",
        nominal_bit=1.0,
        in_features=int(spec["in_features"]),
        out_features=int(spec["out_features"]),
        has_bias=bool(spec["has_bias"]),
        param_count=int(spec["in_features"]) * int(spec["out_features"]),
        source=source,
        module_spec=copy.deepcopy(spec),
    )
    return candidate, compact_state, host


def test_build_candidate_module_loads_all_keys_strictly(tmp_path: Path):
    from mix_bit.module_swap import build_candidate_module

    candidate, compact_state, _ = _checkpoint_fixture(tmp_path)
    rebuilt = build_candidate_module(candidate, compact_state, device="cpu")
    assert isinstance(rebuilt, VAELinear)
    local = {k[len(MODULE_NAME) + 1 :]: v for k, v in compact_state.items()}
    rebuilt_state = rebuilt.state_dict()
    assert set(rebuilt_state) == set(local)
    for key, value in local.items():
        torch.testing.assert_close(rebuilt_state[key].cpu(), value.cpu())
    assert rebuilt.in_features == candidate.in_features
    assert rebuilt.out_features == candidate.out_features
    assert (rebuilt.bias is not None) == candidate.has_bias
    assert int(rebuilt.residual_stages) == int(candidate.module_spec["residual_stages"])
    assert int(rebuilt.codebook_dim) == int(candidate.module_spec["codebook_dim"])


def test_wrong_state_prefix_fails(tmp_path: Path):
    from mix_bit.module_swap import build_candidate_module

    candidate, compact_state, _ = _checkpoint_fixture(tmp_path)
    wrong = {k.replace(MODULE_NAME, "model.layers.0.k_proj", 1): v for k, v in compact_state.items()}
    with pytest.raises(ValueError, match="prefix|does not use prefix|not consumed"):
        build_candidate_module(candidate, wrong, device="cpu")


def test_temporary_swap_restores_original_object_on_success():
    from mix_bit.module_swap import temporary_module_swap

    host = _make_nested_host(_make_vae_linear())
    original = host.model.layers[0].q_proj
    replacement = _make_vae_linear()
    with temporary_module_swap(host, MODULE_NAME, replacement) as current:
        assert current is replacement
        assert host.model.layers[0].q_proj is replacement
    assert host.model.layers[0].q_proj is original


def test_temporary_swap_restores_original_object_on_exception():
    from mix_bit.module_swap import temporary_module_swap

    host = _make_nested_host(_make_vae_linear())
    original = host.model.layers[0].q_proj
    replacement = _make_vae_linear()
    with pytest.raises(RuntimeError, match="boom"):
        with temporary_module_swap(host, MODULE_NAME, replacement):
            assert host.model.layers[0].q_proj is replacement
            raise RuntimeError("boom")
    assert host.model.layers[0].q_proj is original


def test_candidate_construction_does_not_require_teacher_model(tmp_path: Path):
    from mix_bit.module_swap import build_candidate_module

    candidate, compact_state, _ = _checkpoint_fixture(tmp_path)
    # Only ModuleCandidate + compact tensors — no teacher/host required.
    rebuilt = build_candidate_module(candidate, compact_state, device="cpu")
    x = torch.randn(2, candidate.in_features)
    with torch.no_grad():
        y = rebuilt(x)
    assert y.shape == (2, candidate.out_features)


def test_shape_only_placeholder_does_not_allocate_full_dense_weight(tmp_path: Path, monkeypatch):
    from mix_bit import module_swap
    from mix_bit.module_swap import build_candidate_module
    from train_utils import model_checkpoint_io

    candidate, compact_state, _ = _checkpoint_fixture(tmp_path)
    seen_weight_numels: list[int] = []
    original_rebuild = model_checkpoint_io._rebuild_converted_modules

    def _spy(holder, specs, **kwargs):
        target = holder.target
        weight = getattr(target, "weight", None)
        assert weight is not None
        seen_weight_numels.append(int(weight.numel()))
        in_f = int(candidate.in_features)
        out_f = int(candidate.out_features)
        assert int(weight.numel()) != in_f * out_f
        assert int(weight.numel()) == 0
        return original_rebuild(holder, specs, **kwargs)

    monkeypatch.setattr(module_swap, "_rebuild_converted_modules", _spy)
    build_candidate_module(candidate, compact_state, device="cpu")
    assert seen_weight_numels == [0]


def test_candidate_rejects_has_original_weight(tmp_path: Path):
    from mix_bit.module_swap import build_candidate_module

    candidate, compact_state, _ = _checkpoint_fixture(tmp_path)
    bad_spec = copy.deepcopy(candidate.module_spec)
    bad_spec["has_original_weight"] = True
    bad = ModuleCandidate(
        module_name=candidate.module_name,
        category=candidate.category,
        module_suffix=candidate.module_suffix,
        block_index=candidate.block_index,
        mode_name=candidate.mode_name,
        nominal_bit=candidate.nominal_bit,
        in_features=candidate.in_features,
        out_features=candidate.out_features,
        has_bias=candidate.has_bias,
        param_count=candidate.param_count,
        source=candidate.source,
        module_spec=bad_spec,
    )
    with pytest.raises(ValueError, match="has_original_weight"):
        build_candidate_module(bad, compact_state, device="cpu")


def test_same_candidate_swap_preserves_logits(tmp_path: Path):
    from mix_bit.module_swap import build_candidate_module, temporary_module_swap

    candidate, compact_state, _host = _checkpoint_fixture(tmp_path)
    rebuilt = build_candidate_module(candidate, compact_state, device="cpu")
    rebuilt2 = build_candidate_module(candidate, compact_state, device="cpu")
    x = torch.randn(4, candidate.in_features)
    with torch.no_grad():
        y0 = rebuilt(x)
        y1 = rebuilt2(x)
        torch.testing.assert_close(y1, y0)
        swap_host = _make_nested_host(_make_vae_linear(with_bias=True, with_original=False))
        with temporary_module_swap(swap_host, MODULE_NAME, rebuilt):
            y_swap = swap_host.model.layers[0].q_proj(x)
        torch.testing.assert_close(y_swap, y0)


def test_decoded_cache_is_cleared_after_state_load(tmp_path: Path):
    from mix_bit.module_swap import build_candidate_module, refresh_vae_runtime

    candidate, compact_state, _ = _checkpoint_fixture(tmp_path)
    rebuilt = build_candidate_module(candidate, compact_state, device="cpu")
    assert rebuilt._cached_weight is None
    _ = rebuilt(torch.randn(2, candidate.in_features))
    assert rebuilt._cached_weight is not None
    refresh_vae_runtime(rebuilt)
    assert rebuilt._cached_weight is None
