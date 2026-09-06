from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.cat_checkpoint_distill_v6 import (
    CheckpointDistillV6Source,
    load_checkpoint_distill_progress,
    resolve_checkpoint_distill_mode,
    save_checkpoint_distill_v6_model,
)
from train_utils.checkpoint_v6 import load_v6_meta
from train_utils.config.configs import (
    AfterCategoryResolvedConfig,
    AuxTrainableConfig,
    DistillDataConfig,
    DistillLossConfig,
    DistillOptimizationConfig,
    DistillRuntimeConfig,
    LoRAConfig,
)


@pytest.mark.parametrize("mode", ["current_lora", "current_decoder", "current_lora_decoder"])
def test_checkpoint_distill_accepts_only_current_modes(mode):
    assert resolve_checkpoint_distill_mode(SimpleNamespace(after_category_mode=mode)) == mode


@pytest.mark.parametrize(
    "mode", ["remaining_lora", "remaining_lora_current_decoder", "remaining_lora_prefix_decoder"]
)
def test_checkpoint_distill_rejects_remaining_modes(mode):
    with pytest.raises(ValueError, match="does not support remaining"):
        resolve_checkpoint_distill_mode(SimpleNamespace(after_category_mode=mode))


def test_checkpoint_distill_progress_is_separate_and_counts_only_trained_rounds():
    source = CheckpointDistillV6Source(
        requested_checkpoint_dir="/tmp/source",
        checkpoint_dir="/tmp/source",
        checkpoint_kind="category_boundary",
        meta={
            "completed_categories": ["q_proj"],
            "extra_meta": {
                "checkpoint_distill_completed_categories": ["q_proj", "v_proj"],
                "checkpoint_distill_stage_history": [
                    {"category": "q_proj", "mode": "current_lora", "did_train": True},
                    {"category": "k_proj", "mode": "current_lora", "did_train": False},
                    {"category": "v_proj", "mode": "current_lora", "did_train": True},
                ],
            },
        },
        cat_runtime_state=None,
    )
    progress = load_checkpoint_distill_progress(source)
    assert progress.completed_categories == ("q_proj", "v_proj")
    assert progress.lora_round_idx == 2
    assert source.meta["completed_categories"] == ["q_proj"]


def _vae() -> VAELinear:
    bits = torch.tensor(
        [[[1, 0, 1, 0, 1, 0, 1, 0, 1]] for _ in range(4)], dtype=torch.bool
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
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=decoder,
        codebook_dim=4,
        transpose=False,
    )


def test_checkpoint_distill_stable_save_keeps_online_progress_and_null_lora(tmp_path):
    model = nn.Module()
    model.proj = _vae()
    source = CheckpointDistillV6Source(
        requested_checkpoint_dir="/tmp/source",
        checkpoint_dir="/tmp/source",
        checkpoint_kind="category_boundary",
        meta={
            "checkpoint_id": "source-id",
            "base_model_path": "tiny",
            "compressed_targets": ["proj"],
            "pending_dense_targets": [],
            "skip_targets": [],
            "legacy_original_only_sources": [],
            "completed_categories": ["q_proj"],
            "compression_categories": ["q_proj", "k_proj"],
            "target_layers": [],
            "target_modules": [],
        },
        cat_runtime_state=None,
    )
    resolved = AfterCategoryResolvedConfig(
        data=DistillDataConfig(dataset_mix="openorca"),
        loss=DistillLossConfig(loss_type="kl_top"),
        opt=DistillOptimizationConfig(),
        lora=LoRAConfig(rank=3, alpha=6.0, dropout=0.1),
        aux=AuxTrainableConfig(),
        runtime=DistillRuntimeConfig(),
    )
    cat_args = SimpleNamespace(
        after_category_mode="current_lora",
        resolve_after_category_config=lambda _category: resolved,
    )
    output = tmp_path / "final_model"
    save_checkpoint_distill_v6_model(
        model,
        str(output),
        checkpoint_kind="final_model",
        category=None,
        mode="current_lora",
        source=source,
        checkpoint_distill_completed_categories=["k_proj"],
        checkpoint_distill_stage_history=[
            {"category": "k_proj", "mode": "current_lora", "did_train": True}
        ],
        cat_args=cat_args,
        training_args=SimpleNamespace(),
        vae_args=SimpleNamespace(model_path="tiny"),
        tokenizer=None,
        round_idx=0,
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )
    meta = load_v6_meta(str(output))
    assert meta["checkpoint_kind"] == "final_model"
    assert meta["lora_config"] is None
    assert meta["completed_categories"] == ["q_proj"]
    assert meta["extra_meta"]["checkpoint_distill_completed_categories"] == ["k_proj"]
    assert meta["runtime_audit"]["recovery_lora_config"]["rank"] == 3
