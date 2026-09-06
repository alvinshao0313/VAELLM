from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from tools.migrate_checkpoint_v6 import build_parser, migrate_checkpoint_v6
from train_utils.legacy_checkpoint_io import inspect_legacy_checkpoint, normalize_legacy_model_for_v6


def _legacy_dir(tmp_path, extra=None):
    path = tmp_path / "legacy"
    path.mkdir()
    meta = {
        "format": "vaellm_state_dict_with_meta",
        "version": 7,
        "base_model_path": "tiny",
    }
    meta.update(extra or {})
    (path / "checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_migration_cli_defaults_to_dry_run_and_does_not_write(tmp_path):
    source = _legacy_dir(tmp_path)
    output = tmp_path / "v6"
    args = build_parser().parse_args(["--source", str(source), "--output_dir", str(output)])
    assert args.dry_run is True
    report = migrate_checkpoint_v6(source=str(source), output_dir=str(output), dry_run=True)
    assert report["status"] == "validated"
    assert not output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        {"compressed_lora_scope": "compressed_subspace"},
        {"lora_use_dora": True},
        {"use_rslora": True},
        {"vae_lora_variant": "adalora"},
        {"outlier_protect_mode": "channel_residual_vae"},
        {"outlier_protect_mode": "residual_sparse"},
        {"checkpoint_kind": "block_vae_lora_layer"},
    ],
)
def test_migration_rejects_forbidden_legacy_topologies(tmp_path, extra):
    source = _legacy_dir(tmp_path, extra)
    with pytest.raises(ValueError, match="unsupported"):
        inspect_legacy_checkpoint(str(source))


def _vae(*, always_use_original: bool, protect_original_weight: bool, original: bool):
    bits = torch.ones((4, 1, 9), dtype=torch.bool)
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
        original_weight=torch.arange(16, dtype=torch.float32).view(4, 4) if original else None,
        vq_weight=bits,
        decoder=decoder,
        codebook_dim=4,
        transpose=False,
        always_use_original=always_use_original,
        protect_original_weight=protect_original_weight,
    )


def test_original_only_becomes_frozen_linear_but_protected_compressed_stays_vae():
    model = nn.Module()
    model.original_only = _vae(
        always_use_original=True, protect_original_weight=True, original=True
    )
    model.protected = _vae(
        always_use_original=False, protect_original_weight=True, original=False
    )
    expected = model.original_only.original_weight.detach().clone()
    compressed, original_only = normalize_legacy_model_for_v6(model)
    assert isinstance(model.original_only, nn.Linear)
    assert torch.equal(model.original_only.weight, expected)
    assert not model.original_only.weight.requires_grad
    assert isinstance(model.protected, VAELinear)
    assert compressed == ("protected",)
    assert original_only == ("original_only",)


def test_original_only_without_original_weight_is_rejected():
    model = nn.Module()
    model.proj = _vae(always_use_original=True, protect_original_weight=False, original=False)
    with pytest.raises(ValueError, match="missing original_weight"):
        normalize_legacy_model_for_v6(model)
