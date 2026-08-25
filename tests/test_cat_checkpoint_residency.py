import copy
from unittest import mock

import torch
from torch import nn

from e2e_common.temporary_switch_linear import TemporarySwitchLinear
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.cat_checkpoint_distill import (
    _CheckpointDistillResidency,
    _apply_checkpoint_distill_residency,
    _restore_checkpoint_distill_residency,
)


_CATEGORIES = ("q_proj", "k_proj", "v_proj")


def _make_decoder() -> Decoder:
    decoder = Decoder(
        in_dim=9,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    ).to(dtype=torch.float32)
    with torch.no_grad():
        for idx, param in enumerate(decoder.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype).view_as(param)
            param.copy_(values / float(param.numel() + 1) + float(idx + 1) * 0.01)
    return decoder


def _make_vae_linear() -> VAELinear:
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
        decoder=copy.deepcopy(_make_decoder()),
        codebook_dim=4,
        transpose=False,
    )


class _Layer(nn.Module):
    def __init__(self, *, vae: bool, offset: float):
        super().__init__()
        for idx, category in enumerate(_CATEGORIES):
            if vae:
                module = _make_vae_linear()
            else:
                module = nn.Linear(4, 4, bias=True)
                with torch.no_grad():
                    module.weight.copy_(torch.arange(16, dtype=torch.float32).view(4, 4) + offset + idx)
                    module.bias.copy_(torch.arange(4, dtype=torch.float32) + offset + idx + 0.25)
            setattr(self, category, module)


class _Model(nn.Module):
    def __init__(self, *, vae: bool, offset: float = 0.0):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(vae=vae, offset=offset)])


class _StaticTeacherRuntime:
    def __init__(self, model):
        self.model = model
        self.load_count = 0

    def get_or_load(self):
        self.load_count += 1
        return self.model


def _module(model, category):
    return getattr(model.model.layers[0], category)


def _name(category):
    return f"model.layers.0.{category}"


def _assert_reference_clone_matches_teacher(clone, teacher_linear):
    assert isinstance(clone, nn.Linear)
    assert torch.equal(clone.weight, teacher_linear.weight)
    assert clone.weight.data_ptr() != teacher_linear.weight.data_ptr()
    assert clone.weight.requires_grad is False
    assert clone.bias is not None
    assert teacher_linear.bias is not None
    assert torch.equal(clone.bias, teacher_linear.bias)
    assert clone.bias.data_ptr() != teacher_linear.bias.data_ptr()


def test_cumulative_q_round_uses_q_vae_and_future_reference_clones():
    model = _Model(vae=True)
    teacher = _Model(vae=False, offset=10.0)
    residency = _CheckpointDistillResidency()

    _apply_checkpoint_distill_residency(
        model=model,
        active_categories=["q_proj"],
        residency=residency,
        teacher_runtime=_StaticTeacherRuntime(teacher),
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=mock.Mock(),
    )

    assert isinstance(_module(model, "q_proj"), VAELinear)
    assert isinstance(_module(model, "k_proj"), nn.Linear)
    assert isinstance(_module(model, "v_proj"), nn.Linear)
    _assert_reference_clone_matches_teacher(_module(model, "k_proj"), _module(teacher, "k_proj"))
    _assert_reference_clone_matches_teacher(_module(model, "v_proj"), _module(teacher, "v_proj"))
    assert set(residency.stashed_vae_modules) == {_name("k_proj"), _name("v_proj")}
    assert _module(model, "q_proj").original_weight is None


def test_cumulative_k_round_restores_k_vae_and_keeps_clone_cache_identity():
    model = _Model(vae=True)
    teacher = _Model(vae=False, offset=20.0)
    runtime = _StaticTeacherRuntime(teacher)
    residency = _CheckpointDistillResidency()

    _apply_checkpoint_distill_residency(
        model=model,
        active_categories=["q_proj"],
        residency=residency,
        teacher_runtime=runtime,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=mock.Mock(),
    )
    v_clone_id = id(residency.reference_dense_linears[_name("v_proj")])
    _apply_checkpoint_distill_residency(
        model=model,
        active_categories=["q_proj", "k_proj"],
        residency=residency,
        teacher_runtime=runtime,
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=mock.Mock(),
    )

    assert isinstance(_module(model, "q_proj"), VAELinear)
    assert isinstance(_module(model, "k_proj"), VAELinear)
    assert isinstance(_module(model, "v_proj"), nn.Linear)
    assert id(residency.reference_dense_linears[_name("v_proj")]) == v_clone_id
    assert _name("k_proj") not in residency.stashed_vae_modules
    assert runtime.load_count == 2


def test_independent_k_round_only_k_is_vae():
    model = _Model(vae=True)
    teacher = _Model(vae=False, offset=30.0)
    residency = _CheckpointDistillResidency()

    _apply_checkpoint_distill_residency(
        model=model,
        active_categories=["k_proj"],
        residency=residency,
        teacher_runtime=_StaticTeacherRuntime(teacher),
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=mock.Mock(),
    )

    assert isinstance(_module(model, "q_proj"), nn.Linear)
    assert isinstance(_module(model, "k_proj"), VAELinear)
    assert isinstance(_module(model, "v_proj"), nn.Linear)
    assert set(residency.stashed_vae_modules) == {_name("q_proj"), _name("v_proj")}


def test_restore_all_returns_graph_to_checkpoint_vae_modules_without_switches():
    model = _Model(vae=True)
    teacher = _Model(vae=False, offset=40.0)
    residency = _CheckpointDistillResidency()

    _apply_checkpoint_distill_residency(
        model=model,
        active_categories=["k_proj"],
        residency=residency,
        teacher_runtime=_StaticTeacherRuntime(teacher),
        device=torch.device("cpu"),
        dtype=torch.float32,
        logger=mock.Mock(),
    )
    _restore_checkpoint_distill_residency(model=model, residency=residency, logger=mock.Mock())

    assert all(isinstance(_module(model, category), VAELinear) for category in _CATEGORIES)
    assert not any(isinstance(module, TemporarySwitchLinear) for module in model.modules())
    assert all(_module(model, category).original_weight is None for category in _CATEGORIES)
    assert residency.stashed_vae_modules == {}
    state_keys = set(model.state_dict())
    assert "model.layers.0.q_proj.weight" not in state_keys
    assert "model.layers.0.k_proj.weight" not in state_keys
    assert "model.layers.0.v_proj.weight" not in state_keys
