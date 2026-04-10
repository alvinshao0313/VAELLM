import json

import pytest
import torch
from torch import nn

from e2e_fintuning.checkpoint_io import (
    load_e2e_checkpoint_into_model,
    save_e2e_model_checkpoint,
)
from litebsq.llm_vae import Decoder
from litebsq.vae_linear import VAELinear
from tools.cat_train import _build_sparse_residual_coo_patch
from train_utils.cat_train_args import process_cat_train_args
from train_utils.model_checkpoint_io import _rebuild_converted_modules, save_model_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)


def _build_zero_decoder() -> Decoder:
    decoder = Decoder(
        in_dim=1,
        out_dim=1,
        hidden_dim=4,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )
    with torch.no_grad():
        for param in decoder.parameters():
            param.zero_()
    return decoder


def _build_sparse_vae_linear() -> VAELinear:
    return VAELinear(
        in_features=4,
        out_features=3,
        bias=None,
        original_weight=None,
        vq_weight=torch.zeros((12, 1, 1), dtype=torch.float32),
        decoder=_build_zero_decoder(),
        codebook_dim=1,
        transpose=False,
        parallel_parts=1,
        sparse_residual_row_indices=torch.tensor([0, 2], dtype=torch.uint16),
        sparse_residual_col_indices=torch.tensor([1, 3], dtype=torch.uint16),
        sparse_residual_values=torch.tensor([1.5, -0.5], dtype=torch.float16),
    )


def test_process_cat_train_args_accepts_residual_sparse_mode():
    cat_args, _hf_args, _training_args, _vae_args = process_cat_train_args(
        [
            "--outlier_protect_mode",
            "residual_sparse",
            "--outlier_residual_top_p",
            "0.25",
            "--outlier_residual_score",
            "input_act_weighted_abs",
        ]
    )
    assert cat_args.outlier_protect_mode == "residual_sparse"
    assert cat_args.outlier_residual_top_p == pytest.approx(0.25)
    assert cat_args.outlier_residual_score == "input_act_weighted_abs"


def test_process_cat_train_args_rejects_invalid_residual_sparse_configs():
    with pytest.raises(ValueError):
        process_cat_train_args(
            [
                "--outlier_protect_mode",
                "residual_sparse",
                "--outlier_residual_top_p",
                "0.0",
            ]
        )
    with pytest.raises(ValueError):
        process_cat_train_args(
            [
                "--outlier_protect_mode",
                "channel",
                "--outlier_residual_top_p",
                "0.1",
            ]
        )
    with pytest.raises(ValueError):
        process_cat_train_args(
            [
                "--outlier_protect_mode",
                "residual_sparse",
                "--outlier_residual_top_p",
                "0.1",
                "--outlier_protect_count",
                "default=1",
            ]
        )


def test_build_sparse_residual_coo_patch_supports_abs_and_input_weighted_scores():
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    reconstructed = torch.zeros_like(original)

    row_idx, col_idx, values = _build_sparse_residual_coo_patch(
        linear_name="toy_abs",
        original_weight=original,
        reconstructed_weight=reconstructed,
        activation_weight=None,
        score_mode="abs",
        top_p=0.5,
    )
    assert row_idx.dtype == torch.uint16
    assert col_idx.dtype == torch.uint16
    assert values.dtype == torch.float16
    assert row_idx.tolist() == [1, 1]
    assert col_idx.tolist() == [0, 1]
    assert values.tolist() == [3.0, 4.0]

    row_idx, col_idx, values = _build_sparse_residual_coo_patch(
        linear_name="toy_weighted",
        original_weight=original,
        reconstructed_weight=reconstructed,
        activation_weight=torch.tensor([10.0, 1.0], dtype=torch.float32),
        score_mode="input_act_weighted_abs",
        top_p=0.25,
    )
    assert row_idx.tolist() == [1]
    assert col_idx.tolist() == [0]
    assert values.tolist() == [3.0]


def test_vae_linear_decode_weight_applies_sparse_residual_patch():
    vae_linear = _build_sparse_vae_linear()
    decoded = vae_linear._decode_weight(dtype=torch.float32)
    expected = torch.tensor(
        [
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)


def test_save_model_checkpoint_round_trip_preserves_sparse_residual(tmp_path):
    model = TinyModel()
    model.linear = _build_sparse_vae_linear()
    expected = model.linear._decode_weight(dtype=torch.float32)

    save_dir = tmp_path / "cat_ckpt"
    save_model_checkpoint(model, str(save_dir), save_config=False)

    meta = json.loads((save_dir / "checkpoint_meta.json").read_text(encoding="utf-8"))
    spec = meta["converted_modules"][0]
    assert spec["sparse_residual_row_indices"]["dtype"] == "uint16"
    assert spec["sparse_residual_col_indices"]["dtype"] == "uint16"
    assert spec["sparse_residual_values"]["dtype"] == "float16"

    restored = TinyModel()
    _rebuild_converted_modules(restored, meta["converted_modules"])
    state_dict = torch.load(save_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    restored.load_state_dict(state_dict, strict=True)

    assert isinstance(restored.linear, VAELinear)
    torch.testing.assert_close(restored.linear._decode_weight(dtype=torch.float32), expected, atol=0, rtol=0)


def test_save_e2e_checkpoint_round_trip_preserves_sparse_residual(tmp_path):
    model = TinyModel()
    model.linear = _build_sparse_vae_linear()
    expected = model.linear._decode_weight(dtype=torch.float32)

    save_dir = tmp_path / "e2e_ckpt"
    save_e2e_model_checkpoint(
        model,
        str(save_dir),
        save_config=False,
        compact_unload_vae_original_weights=True,
    )

    restored = TinyModel()
    load_e2e_checkpoint_into_model(restored, str(save_dir), map_location="cpu", strict=True)

    assert isinstance(restored.linear, VAELinear)
    torch.testing.assert_close(restored.linear._decode_weight(dtype=torch.float32), expected, atol=0, rtol=0)
