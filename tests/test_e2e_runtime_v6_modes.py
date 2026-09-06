from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from compressed_e2e_fintuning.runtime_v6 import _finalize_decoders, _resolve_train_components
from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager
from train_utils.config.configs import AuxTrainableConfig
from train_utils.decoder_execution import enable_vae_linear_by_execution_plan
from train_utils.model_level_optimizer import (
    GROUP_DECODER,
    GROUP_LORA,
    GROUP_NORM,
    ModelLevelOptimizerLRConfig,
    attach_model_level_optimizer_contract,
    build_model_level_param_groups,
)
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.train_args import TrainingArguments


MODES = (
    "none",
    "decoder",
    "lora",
    "sparse_bit",
    "decoder_lora",
    "decoder_sparse_bit",
    "lora_sparse_bit",
    "decoder_lora_sparse_bit",
)


def _vae_linear() -> VAELinear:
    latent_dim = 9
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
        out_dim=4,
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
        codebook_dim=4,
        transpose=False,
    )


def _multi_stage_vae_linear() -> VAELinear:
    latent_dim = 9
    bits = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )

    def decoder():
        return Decoder(
            in_dim=latent_dim,
            out_dim=4,
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
        vq_weight=None,
        decoder=None,
        stage_vq_weights=[bits, ~bits],
        stage_decoders=[decoder(), decoder()],
        codebook_dim=4,
        stage_codebook_dims=[4, 4],
        transpose=False,
    )


class _TinyModeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _vae_linear()
        self.skip = nn.Linear(4, 4, bias=False)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 6, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.layer(x) + self.skip(x)))


class _TinyRegressionDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        self.x = torch.randn(4, 4, generator=generator)
        self.target = torch.randn(4, 6, generator=generator)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {
            "x": self.x[int(index)].clone(),
            "target": self.target[int(index)].clone(),
        }


class _RegressionTrainer(VAEDecoderE2ETrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        output = model(inputs["x"])
        target = inputs["target"].to(device=output.device, dtype=output.dtype)
        loss = (output - target).float().square().mean()
        return (loss, {"output": output}) if return_outputs else loss


def _decoder_parameters(layer: VAELinear):
    packed = getattr(layer, "_parallel_stage_decoder", None)
    decoder = packed if isinstance(packed, nn.Module) else layer.get_stage_part_decoder(stage_idx=0, part_idx=0)
    return tuple(decoder.parameters())


def test_e2e_decoder_finalization_preserves_packed_execution_payload():
    layer = _multi_stage_vae_linear()
    enable_vae_linear_by_execution_plan(layer, mode="decoder_sparse_bit")
    packed = layer._parallel_stage_decoder
    layer.eval()
    with torch.no_grad():
        before = layer._decode_compressed_weight(dtype=torch.float32)

    assert _finalize_decoders([("layer", layer)]) == 1

    assert layer._parallel_stage_decoder is packed
    assert layer.parallel_stage_decode is True
    assert layer.trainable_decode is False
    assert all(not parameter.requires_grad for parameter in packed.parameters())
    with torch.no_grad():
        after = layer._decode_compressed_weight(dtype=torch.float32)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("train_mode", MODES)
def test_all_canonical_e2e_modes_have_exact_trainable_and_optimizer_components(train_mode: str):
    model = _TinyModeModel()
    vae = model.layer
    train_decoder, train_lora, train_sparse = _resolve_train_components(train_mode)
    aux = AuxTrainableConfig(norm_train_mode="all" if train_mode == "none" else "none")
    execution_mode = "decoder_sparse_bit" if train_decoder and train_sparse else "trainable_decoder"

    selection = build_model_level_trainable_selection(
        model,
        aux=aux,
        compressed_modules=[("layer", vae)],
        dense_target_modules=(),
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=train_decoder,
        train_lora=train_lora,
        decoder_execution_mode=execution_mode,
        freeze=True,
    )
    train_model = selection.peft_model or model
    if train_sparse and not train_decoder:
        enable_vae_linear_by_execution_plan(vae, mode="sparse_bit")

    assert bool(selection.decoder_parameters) is train_decoder
    assert bool(selection.lora_parameters) is train_lora
    assert bool(selection.norm_parameters) is (train_mode == "none")
    assert all(not parameter.requires_grad for parameter in model.skip.parameters())
    assert all(parameter.requires_grad is train_decoder for parameter in _decoder_parameters(vae))

    continuous_groups = []
    if any(
        (
            selection.decoder_parameters,
            selection.lora_parameters,
            selection.norm_parameters,
            selection.lm_head_parameters,
        )
    ):
        continuous_groups = build_model_level_param_groups(
            selection,
            lr_config=ModelLevelOptimizerLRConfig(
                learning_rate=1e-4,
                weight_decay=0.01,
                decoder_lr=2e-5,
                norm_lr=3e-5,
            ),
            model=train_model,
        )
    group_by_name = {str(group["group_name"]): group for group in continuous_groups}
    assert (GROUP_DECODER in group_by_name) is train_decoder
    assert (GROUP_LORA in group_by_name) is train_lora
    assert (GROUP_NORM in group_by_name) is (train_mode == "none")
    if train_decoder:
        assert group_by_name[GROUP_DECODER]["lr"] == pytest.approx(2e-5)
        assert group_by_name[GROUP_DECODER]["weight_decay"] == 0.0
    if train_lora:
        assert group_by_name[GROUP_LORA]["lr"] == pytest.approx(1e-4)
        assert group_by_name[GROUP_LORA]["weight_decay"] == pytest.approx(0.01)
    if train_mode == "none":
        assert group_by_name[GROUP_NORM]["lr"] == pytest.approx(3e-5)
        assert group_by_name[GROUP_NORM]["weight_decay"] == 0.0

    manager = None
    if train_sparse:
        manager = SparseBitTuningManager(
            root_model=train_model,
            targets=[("layer", vae)],
            target_devices={"layer": torch.device("cpu")},
            training_seed=7,
            config=SparseBitTuningConfig(
                enabled=True,
                active_ratio=0.25,
                optimizer="adam",
                bit_lr=0.02,
                round_steps=4,
            ),
            streaming=False,
        )
        bit_parameters = tuple(manager.score_module.bit_parameters())
        assert bit_parameters
        assert all(parameter.requires_grad for parameter in bit_parameters)
    else:
        assert not hasattr(train_model, "sparse_bit_tuning")

    if manager is not None:
        manager.detach_runtime()


@pytest.mark.parametrize("train_mode", ["decoder", "lora"])
def test_decoder_and_lora_modes_complete_one_real_optimizer_step(train_mode: str, tmp_path):
    torch.manual_seed(314)
    model = _TinyModeModel()
    train_decoder, train_lora, train_sparse = _resolve_train_components(train_mode)
    assert train_sparse is False
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(),
        compressed_modules=[("layer", model.layer)],
        dense_target_modules=(),
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=train_decoder,
        train_lora=train_lora,
        freeze=True,
    )
    train_model = selection.peft_model or model
    selected_parameters = (
        selection.decoder_parameters if train_decoder else selection.lora_parameters
    )
    assert selected_parameters
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in selected_parameters.items()
    }
    skip_before = model.skip.weight.detach().cpu().clone()

    args = TrainingArguments(
        output_dir=str(tmp_path / train_mode),
        per_device_train_batch_size=1,
        max_steps=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        use_cpu=True,
        seed=314,
        data_seed=314,
    )
    trainer = _RegressionTrainer(
        model=train_model,
        args=args,
        train_dataset=_TinyRegressionDataset(),
        loss_type="sft",
    )
    attach_model_level_optimizer_contract(
        trainer,
        selection=selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-2,
            weight_decay=0.0,
            decoder_lr=1e-2,
        ),
    )
    trainer.train()

    assert int(trainer.state.global_step) == 1
    changed = [
        name
        for name, parameter in selected_parameters.items()
        if not torch.equal(before[name], parameter.detach().cpu())
    ]
    assert changed, f"{train_mode} selected parameters did not update"
    assert torch.equal(skip_before, model.skip.weight.detach().cpu())
