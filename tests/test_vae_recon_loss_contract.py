from __future__ import annotations

import pytest
import torch

from litebsq.autoencoder import AutoEncoder
from litebsq.bsq import BSQ
from train_utils.block_vae_lora_args import _RECON_LOSS_TYPE_CHOICES
from train_utils.cat_train_args import _CAT_RECON_LOSS_CHOICES
from train_utils.cat_train_pipeline import (
    _compute_recon_loss as compute_category_recon_loss,
    _compute_reconstruction_eval_metrics,
)


def test_bsq_current_constructor_quantizes_and_backpropagates() -> None:
    torch.manual_seed(43)
    quantizer = BSQ(
        dim=8,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        num_codebooks=1,
        keep_num_codebooks_dim=False,
        codebook_scale=1.0,
        has_projections=False,
        projection_has_bias=True,
        channel_first=False,
        spherical=True,
        force_quantization_f32=True,
        inv_temperature=100.0,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        new_quant=False,
    )
    z = torch.randn(2, 4, 8, requires_grad=True)
    result = quantizer(z)

    assert result.quantized.shape == z.shape
    assert torch.isfinite(result.quantized).all()
    assert result.entropy_aux_loss.ndim == 0
    assert torch.isfinite(result.entropy_aux_loss)

    objective = result.quantized.float().pow(2).mean()
    objective = objective + result.entropy_aux_loss.float()
    objective.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


_CATEGORY_ONLY_RECON_LOSSES = frozenset({"wa_mse", "amse"})


def test_recon_loss_choice_sets_match_dispatchers_and_execute() -> None:
    block_only_in_cat = set(_CAT_RECON_LOSS_CHOICES) - set(_RECON_LOSS_TYPE_CHOICES)
    assert block_only_in_cat == set(_CATEGORY_ONLY_RECON_LOSSES)

    torch.manual_seed(41)
    x = torch.randn(2, 3, 8)
    weights = torch.rand(2, 3, 8).add_(0.1)
    autoencoder = object.__new__(AutoEncoder)

    for loss_type in _RECON_LOSS_TYPE_CHOICES:
        x_recon = torch.randn(2, 3, 8, requires_grad=True)
        autoencoder.recon_loss_type = loss_type
        loss = AutoEncoder._compute_recon_loss(autoencoder, x_recon, x)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()

    for loss_type in _CAT_RECON_LOSS_CHOICES:
        x_recon = torch.randn(2, 3, 8, requires_grad=True)
        act_max = weights if loss_type in _CATEGORY_ONLY_RECON_LOSSES else None
        loss = compute_category_recon_loss(
            recon_loss_type=loss_type,
            x_recon=x_recon,
            x=x,
            act_max=act_max,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()


def test_recon_loss_choice_contracts_are_exact() -> None:
    assert _CAT_RECON_LOSS_CHOICES == (
        "mse",
        "l1",
        "huber",
        "relative_l1",
        "w_mse",
        "w2_mse",
        "wa_mse",
        "amse",
    )
    assert _RECON_LOSS_TYPE_CHOICES == (
        "mse",
        "l1",
        "huber",
        "relative_l1",
        "w_mse",
        "w2_mse",
    )


def test_autoencoder_unknown_recon_loss_raises_generic_value_error() -> None:
    autoencoder = object.__new__(AutoEncoder)
    autoencoder.recon_loss_type = "not_a_real_loss"
    x = torch.randn(2, 3, 4)
    x_recon = torch.randn(2, 3, 4)

    with pytest.raises(
        ValueError,
        match=r"Unsupported recon_loss_type='not_a_real_loss'\.",
    ):
        AutoEncoder._compute_recon_loss(
            autoencoder,
            x_recon,
            x,
        )


def test_category_unknown_recon_loss_raises_generic_value_error() -> None:
    x = torch.randn(2, 3, 4)
    x_recon = torch.randn(2, 3, 4)

    with pytest.raises(
        ValueError,
        match=r"Unsupported recon_loss_type='not_a_real_loss'\.",
    ):
        compute_category_recon_loss(
            recon_loss_type="not_a_real_loss",
            x_recon=x_recon,
            x=x,
        )


@pytest.mark.parametrize(
    "loss_type",
    (
        "mse",
        "l1",
        "huber",
        "relative_l1",
        "w_mse",
        "w2_mse",
    ),
)
def test_shared_recon_losses_are_finite_and_backpropagate(
    loss_type: str,
) -> None:
    torch.manual_seed(31)
    x = torch.randn(2, 3, 8)
    auto_recon = torch.randn(2, 3, 8, requires_grad=True)
    cat_recon = auto_recon.detach().clone().requires_grad_(True)

    autoencoder = object.__new__(AutoEncoder)
    autoencoder.recon_loss_type = loss_type

    auto_loss = AutoEncoder._compute_recon_loss(
        autoencoder,
        auto_recon,
        x,
    )
    cat_loss = compute_category_recon_loss(
        recon_loss_type=loss_type,
        x_recon=cat_recon,
        x=x,
    )

    assert auto_loss.ndim == 0
    assert cat_loss.ndim == 0
    assert torch.isfinite(auto_loss)
    assert torch.isfinite(cat_loss)
    assert torch.allclose(auto_loss, cat_loss, rtol=1e-6, atol=1e-7)

    auto_loss.backward()
    cat_loss.backward()
    assert auto_recon.grad is not None
    assert cat_recon.grad is not None
    assert torch.isfinite(auto_recon.grad).all()
    assert torch.isfinite(cat_recon.grad).all()


@pytest.mark.parametrize("loss_type", ("wa_mse", "amse"))
def test_weighted_recon_losses_are_finite_and_backpropagate(
    loss_type: str,
) -> None:
    torch.manual_seed(37)
    x = torch.randn(2, 3, 8)
    weights = torch.rand(2, 3, 8).add_(0.1)
    auto_recon = torch.randn(2, 3, 8, requires_grad=True)
    cat_recon = auto_recon.detach().clone().requires_grad_(True)

    autoencoder = object.__new__(AutoEncoder)
    autoencoder.recon_loss_type = loss_type

    auto_loss = AutoEncoder._compute_recon_loss(
        autoencoder,
        auto_recon,
        x,
        act_max=weights,
    )
    cat_loss = compute_category_recon_loss(
        recon_loss_type=loss_type,
        x_recon=cat_recon,
        x=x,
        act_max=weights,
    )

    assert torch.isfinite(auto_loss)
    assert torch.isfinite(cat_loss)
    assert torch.allclose(auto_loss, cat_loss, rtol=1e-6, atol=1e-7)

    auto_loss.backward()
    cat_loss.backward()
    assert torch.isfinite(auto_recon.grad).all()
    assert torch.isfinite(cat_recon.grad).all()


def test_reconstruction_eval_metrics_select_topk_per_parallel_model() -> None:
    # Two parallel models (P=2): top-k must be chosen independently per model.
    # Model 0 largest abs refs are at cols 0,1; model 1 largest abs refs at cols 2,3.
    x_eval = torch.tensor(
        [
            [
                [10.0, 9.0, 0.1, 0.2],
                [0.1, 0.2, 8.0, 7.0],
            ]
        ],
        dtype=torch.float32,
    )
    x_recon = torch.tensor(
        [
            [
                [11.0, 7.0, 5.0, 5.0],
                [5.0, 5.0, 10.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )

    overall_sum, selected_sum, overall_numel, selected_numel = (
        _compute_reconstruction_eval_metrics(x_eval, x_recon, top_k=2)
    )

    squared = (x_recon - x_eval).pow(2)
    assert overall_numel == 8
    assert selected_numel == 4
    assert torch.allclose(overall_sum, squared.sum())

    # Per-model selected squared errors:
    # model0: (11-10)^2 + (7-9)^2 = 1 + 4 = 5
    # model1: (10-8)^2 + (4-7)^2 = 4 + 9 = 13
    assert torch.allclose(selected_sum, torch.tensor(18.0))
