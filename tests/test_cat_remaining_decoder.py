import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.lora_training import _DistillOptimizerGroupingMixin
from train_utils.lora_utils import (
    _LoraTrainerLogCallback,
)
from train_utils.distill_decoder import (
    NamedMainDecoderTarget,
    enable_main_decoder_targets,
    finalize_main_decoder_targets,
    iter_main_decoder_modules,
)


class _OptimizerToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = nn.Linear(3, 3)
        self.extra_norm = nn.LayerNorm(3)
        self.decoder = nn.Linear(3, 3)


class _BaseDummyTrainer:
    def __init__(self, *, model, args):
        self.model = model
        self.args = args
        self.optimizer = None
        self.optimizer_cls_and_kwargs = None
        self.legacy_create_optimizer_called = False

    def create_optimizer(self):
        self.legacy_create_optimizer_called = True
        return "legacy"

    def get_decay_parameter_names(self, _model):
        return {"lora.weight", "decoder.weight"}

    def get_optimizer_cls_and_kwargs(self, args, _model):
        return torch.optim.SGD, {"lr": float(args.learning_rate)}


class _DummyGroupedTrainer(_DistillOptimizerGroupingMixin, _BaseDummyTrainer):
    pass


def _make_decoder(latent_dim: int = 9, codebook_dim: int = 4) -> Decoder:
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
    return decoder


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


def _make_two_stage_parallel_vae_linear() -> VAELinear:
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


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class _TwoVaeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.historical = _make_single_stage_vae_linear()
        self.current = _make_single_stage_vae_linear()
        _freeze(self.historical)
        _freeze(self.current)


def test_decoder_helper_single_stage_main_decoder_trainable_and_finalized():
    layer = _make_single_stage_vae_linear()
    _freeze(layer)
    target = NamedMainDecoderTarget(name="layer.q_proj", base_layer=layer)

    params = enable_main_decoder_targets([target])

    assert params
    assert len({id(param) for param in params}) == len(params)
    assert getattr(layer, "_parallel_stage_decoder", None) is None
    assert layer.trainable_decode is True
    assert layer.cache_decoded_weight is False
    assert all(param.requires_grad for param in params)
    assert all(param.requires_grad for decoder in iter_main_decoder_modules(layer) for param in decoder.parameters())

    finalized = finalize_main_decoder_targets([target])

    assert finalized == 1
    assert layer.trainable_decode is False
    assert layer.cache_decoded_weight is True
    assert all(not param.requires_grad for decoder in iter_main_decoder_modules(layer) for param in decoder.parameters())


def test_decoder_helper_two_stage_packed_decoder_trainable_and_retained():
    layer = _make_two_stage_parallel_vae_linear()
    _freeze(layer)
    target = NamedMainDecoderTarget(name="layer.o_proj", base_layer=layer)

    params = enable_main_decoder_targets([target])
    packed = getattr(layer, "_parallel_stage_decoder", None)

    assert packed is not None
    assert iter_main_decoder_modules(layer) == (packed,)
    assert params
    assert len({id(param) for param in params}) == len(params)
    assert all(param.requires_grad for param in packed.parameters())

    finalized = finalize_main_decoder_targets([target])

    assert finalized == 1
    assert getattr(layer, "_parallel_stage_decoder", None) is packed
    assert layer.trainable_decode is False
    assert layer.cache_decoded_weight is True
    assert all(not param.requires_grad for param in packed.parameters())


def test_decoder_helper_ignores_protected_residual_decoder():
    layer = _make_two_stage_parallel_vae_linear()
    protected = nn.Linear(3, 3)
    _freeze(protected)
    layer._protected_residual_parallel_decoder = protected
    _freeze(layer)
    target = NamedMainDecoderTarget(name="layer.down_proj", base_layer=layer)

    params = enable_main_decoder_targets([target])

    assert params
    assert all(not param.requires_grad for param in protected.parameters())
    assert protected not in iter_main_decoder_modules(layer)

    finalize_main_decoder_targets([target])

    assert all(not param.requires_grad for param in protected.parameters())


def test_optimizer_no_decoder_uses_legacy_create_optimizer_path():
    trainer = _DummyGroupedTrainer(
        model=_OptimizerToyModel(),
        args=SimpleNamespace(weight_decay=0.1, learning_rate=1e-3),
    )

    optimizer = trainer.create_optimizer()

    assert optimizer == "legacy"
    assert trainer.legacy_create_optimizer_called is True


def test_optimizer_decoder_group_has_independent_lr_and_zero_weight_decay():
    from train_utils.model_level_optimizer import (
        GROUP_DECODER,
        GROUP_LORA,
        GROUP_NORM,
        ModelLevelOptimizerLRConfig,
        attach_model_level_optimizer_contract,
        selection_from_component_parameters,
    )

    model = _OptimizerToyModel()
    trainer = _DummyGroupedTrainer(
        model=model,
        args=SimpleNamespace(weight_decay=0.1, learning_rate=1e-3),
        decoder_param_ids=[id(param) for param in model.decoder.parameters()],
        decoder_lr=5e-5,
    )
    selection = selection_from_component_parameters(
        lora_parameters={"lora": model.lora.weight},
        decoder_parameters={
            "decoder::w": model.decoder.weight,
            "decoder::b": model.decoder.bias,
        },
        norm_parameters={"norm::w": model.extra_norm.weight, "norm::b": model.extra_norm.bias},
    )
    # Toy Linear bias may be None depending on defaults; Linear(3,3) has bias.
    if model.lora.bias is not None:
        model.lora.bias.requires_grad_(False)
    attach_model_level_optimizer_contract(
        trainer,
        selection=selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.1,
            decoder_lr=5e-5,
        ),
    )

    optimizer = trainer.create_optimizer()

    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {GROUP_LORA, GROUP_DECODER, GROUP_NORM}
    assert groups[GROUP_LORA]["lr"] == pytest.approx(1e-3)
    assert groups[GROUP_LORA]["weight_decay"] == pytest.approx(0.1)
    assert groups[GROUP_DECODER]["lr"] == pytest.approx(5e-5)
    assert groups[GROUP_DECODER]["weight_decay"] == pytest.approx(0.0)
    assert groups[GROUP_NORM]["weight_decay"] == pytest.approx(0.0)

    grouped_param_ids = [
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    ]
    trainable_param_ids = [id(param) for param in model.parameters() if param.requires_grad]
    assert sorted(grouped_param_ids) == sorted(trainable_param_ids)
    assert len(grouped_param_ids) == len(set(grouped_param_ids))


def test_optimizer_telemetry_reads_actual_group_lrs(tmp_path):
    from train_utils.model_level_optimizer import (
        ModelLevelOptimizerLRConfig,
        attach_model_level_optimizer_contract,
        selection_from_component_parameters,
    )

    model = _OptimizerToyModel()
    if model.lora.bias is not None:
        model.lora.bias.requires_grad_(False)
    trainer = _DummyGroupedTrainer(
        model=model,
        args=SimpleNamespace(weight_decay=0.1, learning_rate=1e-3),
        decoder_param_ids=[id(param) for param in model.decoder.parameters()],
        decoder_lr=5e-5,
    )
    attach_model_level_optimizer_contract(
        trainer,
        selection=selection_from_component_parameters(
            lora_parameters={"lora": model.lora.weight},
            decoder_parameters={
                "decoder::w": model.decoder.weight,
                "decoder::b": model.decoder.bias,
            },
            norm_parameters={"norm::w": model.extra_norm.weight, "norm::b": model.extra_norm.bias},
        ),
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.1,
            decoder_lr=5e-5,
        ),
    )
    optimizer = trainer.create_optimizer()
    # group order: lora, decoder, norm
    by_name = {g["group_name"]: g for g in optimizer.param_groups}
    by_name["lora"]["lr"] = 5e-4
    by_name["decoder"]["lr"] = 2.5e-5
    by_name["norm"]["lr"] = 5e-4

    log_path = tmp_path / "train.log"
    logger = logging.getLogger(f"decoder_optimizer_telemetry_{id(tmp_path)}")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    callback = _LoraTrainerLogCallback(logger=logger)

    callback.on_log(
        None,
        SimpleNamespace(is_world_process_zero=True, global_step=3),
        None,
        logs={"loss": 1.0, "learning_rate": 5e-4},
        optimizer=optimizer,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "lr_lora=0.0005" in text
    assert "lr_decoder=2.5e-05" in text
