from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils.cat_checkpoint_v6 import (
    build_cat_v6_target_inventory,
    save_cat_v6_full_checkpoint,
)
from train_utils.cat_train_pipeline import _load_checkpoint_tokenizer
from train_utils.checkpoint_v6 import (
    CAT_RUNTIME_STATE_FILENAME,
    load_v6_cat_runtime_state,
    load_v6_full_checkpoint_into_model,
    load_v6_meta,
    save_v6_training_step_payload,
)
from train_utils.cat_train_runtime import (
    load_cat_resume_distill_progress,
    resolve_cat_resume_source,
)


CATEGORIES = ("q_proj", "k_proj", "v_proj")


def _decoder(scale: float) -> Decoder:
    module = Decoder(
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
        for idx, param in enumerate(module.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype).view_as(param)
            param.copy_(values.mul_(0.01).add_(float(scale) + idx * 0.001))
    return module


def _vae(scale: float) -> VAELinear:
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
        decoder=_decoder(scale),
        codebook_dim=4,
        transpose=False,
        always_use_original=False,
        protect_original_weight=False,
    )


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.v_proj(torch.tanh(self.k_proj(torch.tanh(self.q_proj(x)))))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(self, x):
        for layer in self.model.layers:
            x = x + layer(x)
        return x


def _baseline() -> _Model:
    torch.manual_seed(20260905)
    return _Model()


def _vae_name(layer_idx: int, category: str) -> str:
    return f"model.layers.{layer_idx}.{category}"


def _replace_category(model: _Model, category: str) -> None:
    cat_scale = {"q_proj": 0.1, "k_proj": 0.2, "v_proj": 0.3}[category]
    for layer_idx, layer in enumerate(model.model.layers):
        setattr(layer, category, _vae(cat_scale + layer_idx * 0.01))


def _vae_args():
    return SimpleNamespace(model_path="tiny")


def _cat_args(*, skip_layers="", target_layers="all"):
    return SimpleNamespace(
        after_category_mode="none",
        target_layers=target_layers,
        skip_layers=skip_layers,
    )


def test_cat_v6_inventory_exactly_classifies_compressed_pending_and_skip():
    model = _baseline()
    _replace_category(model, "q_proj")
    inventory = build_cat_v6_target_inventory(
        model,
        vae_args=_vae_args(),
        compression_categories=CATEGORIES,
        completed_categories=("q_proj",),
        target_layers="all",
        skip_layers="1.k_proj",
    )

    assert inventory.compressed_targets == (
        _vae_name(0, "q_proj"),
        _vae_name(1, "q_proj"),
    )
    assert inventory.pending_dense_targets == (
        _vae_name(0, "k_proj"),
        _vae_name(0, "v_proj"),
        _vae_name(1, "v_proj"),
    )
    assert inventory.skip_targets == (_vae_name(1, "k_proj"),)
    assert inventory.implicit_tail_skip_targets == ()
    assert set(inventory.compressed_targets).isdisjoint(inventory.pending_dense_targets)
    assert set(inventory.compressed_targets).isdisjoint(inventory.skip_targets)
    assert set(inventory.pending_dense_targets).isdisjoint(inventory.skip_targets)


def test_cat_v6_inventory_records_completed_dense_tail_as_permanent_skip():
    model = _baseline()
    model.model.layers[0].q_proj = _vae(0.1)
    inventory = build_cat_v6_target_inventory(
        model,
        vae_args=_vae_args(),
        compression_categories=CATEGORIES,
        completed_categories=("q_proj",),
        target_layers="all",
        skip_layers="",
    )
    tail_name = _vae_name(1, "q_proj")
    assert tail_name in inventory.skip_targets
    assert inventory.implicit_tail_skip_targets == (tail_name,)
    assert tail_name not in inventory.pending_dense_targets


def test_cat_v6_inventory_rejects_future_category_already_compressed():
    model = _baseline()
    _replace_category(model, "q_proj")
    model.model.layers[0].k_proj = _vae(0.2)
    with pytest.raises(ValueError, match="outside the completed prefix/active round"):
        build_cat_v6_target_inventory(
            model,
            vae_args=_vae_args(),
            compression_categories=CATEGORIES,
            completed_categories=("q_proj",),
            target_layers="all",
            skip_layers="",
        )


def test_cat_round_base_inventory_allows_only_active_category_after_completed_prefix():
    model = _baseline()
    _replace_category(model, "q_proj")
    inventory = build_cat_v6_target_inventory(
        model,
        vae_args=_vae_args(),
        compression_categories=CATEGORIES,
        completed_categories=(),
        target_layers="all",
        skip_layers="",
        active_category="q_proj",
    )
    assert inventory.compressed_targets == (
        _vae_name(0, "q_proj"),
        _vae_name(1, "q_proj"),
    )
    assert set(inventory.pending_dense_targets) == {
        _vae_name(0, "k_proj"),
        _vae_name(1, "k_proj"),
        _vae_name(0, "v_proj"),
        _vae_name(1, "v_proj"),
    }

    with pytest.raises(ValueError, match="first category after completed prefix"):
        build_cat_v6_target_inventory(
            model,
            vae_args=_vae_args(),
            compression_categories=CATEGORIES,
            completed_categories=(),
            target_layers="all",
            skip_layers="",
            active_category="k_proj",
        )


def test_cat_round_base_saves_runtime_sidecar_atomically(tmp_path):
    model = _baseline()
    _replace_category(model, "q_proj")
    output_dir = tmp_path / "q_proj_round" / "round_base"
    runtime_state = {
        "format": "test_cat_runtime",
        "stats": {"model.layers.0.k_proj": torch.tensor([1.0, 2.0])},
    }
    save_cat_v6_full_checkpoint(
        model,
        str(output_dir),
        checkpoint_kind="round_base",
        category="q_proj",
        completed_categories=(),
        compression_categories=CATEGORIES,
        cat_args=_cat_args(),
        vae_args=_vae_args(),
        training_args=SimpleNamespace(),
        base_model_path="tiny",
        cat_runtime_state=runtime_state,
    )
    meta = load_v6_meta(str(output_dir))
    assert meta["checkpoint_kind"] == "round_base"
    assert meta["completed_categories"] == []
    assert meta["extra_meta"]["active_category"] == "q_proj"
    assert meta["extra_meta"]["cat_runtime_state_file"] == CAT_RUNTIME_STATE_FILENAME
    assert (output_dir / CAT_RUNTIME_STATE_FILENAME).is_file()
    loaded = load_v6_cat_runtime_state(str(output_dir), required=True)
    assert loaded["format"] == "test_cat_runtime"
    assert torch.equal(loaded["stats"]["model.layers.0.k_proj"], torch.tensor([1.0, 2.0]))


def test_cat_training_step_resume_source_resolves_round_base_and_progress(tmp_path):
    model = _baseline()
    _replace_category(model, "q_proj")
    round_base = tmp_path / "q_proj_round" / "round_base"
    save_cat_v6_full_checkpoint(
        model,
        str(round_base),
        checkpoint_kind="round_base",
        category="q_proj",
        completed_categories=(),
        compression_categories=CATEGORIES,
        cat_args=_cat_args(),
        vae_args=_vae_args(),
        training_args=SimpleNamespace(),
        base_model_path="tiny",
        distill_stage_history=(),
        cat_runtime_state={"format": "test_cat_runtime"},
    )
    round_meta = load_v6_meta(str(round_base))
    step_dir = tmp_path / "q_proj_round" / "trainer" / "checkpoint-1"
    step_dir.mkdir(parents=True)
    save_v6_training_step_payload(
        str(step_dir),
        round_base_ref=os.path.relpath(str(round_base), start=str(step_dir)),
        round_base_checkpoint_id=str(round_meta["checkpoint_id"]),
        mutable_state={},
        mutable_state_manifest=[],
        train_mode="none",
        after_category_mode="current_decoder",
        compressed_targets=round_meta["compressed_targets"],
        pending_dense_targets=round_meta["pending_dense_targets"],
        skip_targets=round_meta["skip_targets"],
        completed_categories=(),
        compression_categories=CATEGORIES,
        target_layers=None,
        target_modules=CATEGORIES,
        immutable_resume_contract={"version": 1},
        extra_meta={
            "active_category": "q_proj",
            "distill_stage_history": [],
        },
    )

    source = resolve_cat_resume_source(str(step_dir))
    assert source.source_kind == "training_step"
    assert source.model_checkpoint_kind == "round_base"
    assert source.model_checkpoint_dir == os.path.abspath(str(round_base))
    assert source.active_category == "q_proj"
    progress = load_cat_resume_distill_progress(str(step_dir))
    assert progress.completed_categories == ()
    assert progress.active_category == "q_proj"
    assert progress.lora_round_idx == 0
    assert progress.training_step_checkpoint == os.path.abspath(str(step_dir))


def test_cat_v6_inventory_requires_completed_categories_to_be_prefix():
    model = _baseline()
    with pytest.raises(ValueError, match="exact prefix"):
        build_cat_v6_target_inventory(
            model,
            vae_args=_vae_args(),
            compression_categories=CATEGORIES,
            completed_categories=("k_proj",),
            target_layers="all",
            skip_layers="",
        )


def test_cat_v6_category_boundary_save_has_root_progress_and_exact_inventory(tmp_path):
    model = _baseline()
    _replace_category(model, "q_proj")
    output_dir = tmp_path / "after_q_proj"
    result = save_cat_v6_full_checkpoint(
        model,
        str(output_dir),
        checkpoint_kind="category_boundary",
        category="q_proj",
        completed_categories=("q_proj",),
        compression_categories=CATEGORIES,
        cat_args=_cat_args(skip_layers="1.k_proj"),
        vae_args=_vae_args(),
        training_args=SimpleNamespace(),
        base_model_path="tiny",
        distill_stage_meta={"category": "q_proj", "did_train": False},
        distill_stage_history=({"category": "q_proj", "did_train": False},),
    )
    meta = load_v6_meta(str(output_dir))

    assert result["meta_payload"]["checkpoint_kind"] == "category_boundary"
    assert meta["completed_categories"] == ["q_proj"]
    assert meta["compressed_targets"] == [
        _vae_name(0, "q_proj"),
        _vae_name(1, "q_proj"),
    ]
    assert meta["pending_dense_targets"] == [
        _vae_name(0, "k_proj"),
        _vae_name(0, "v_proj"),
        _vae_name(1, "v_proj"),
    ]
    assert meta["skip_targets"] == [_vae_name(1, "k_proj")]
    assert meta["extra_meta"]["distill_stage_history"] == [
        {"category": "q_proj", "did_train": False}
    ]
    assert meta["finalized_status"]["stable_category_boundary"] is True


def test_cat_category_boundary_resume_matches_uninterrupted_q_k_v(tmp_path):
    base = _baseline()
    uninterrupted = copy.deepcopy(base)
    interrupted = copy.deepcopy(base)

    for category in CATEGORIES:
        _replace_category(uninterrupted, category)

    _replace_category(interrupted, "q_proj")
    checkpoint_dir = tmp_path / "after_q_proj"
    save_cat_v6_full_checkpoint(
        interrupted,
        str(checkpoint_dir),
        checkpoint_kind="category_boundary",
        category="q_proj",
        completed_categories=("q_proj",),
        compression_categories=CATEGORIES,
        cat_args=_cat_args(),
        vae_args=_vae_args(),
        training_args=SimpleNamespace(),
        base_model_path="tiny",
        distill_stage_meta={"category": "q_proj", "did_train": True},
        distill_stage_history=({"category": "q_proj", "did_train": True},),
    )

    resumed, meta, _ = load_v6_full_checkpoint_into_model(
        copy.deepcopy(base),
        str(checkpoint_dir),
        expected_kind="category_boundary",
        strict=True,
    )
    assert meta["completed_categories"] == ["q_proj"]
    for category in CATEGORIES[1:]:
        _replace_category(resumed, category)

    uninterrupted.eval()
    resumed.eval()
    state_a = uninterrupted.state_dict()
    state_b = resumed.state_dict()
    assert tuple(state_a) == tuple(state_b)
    for name in state_a:
        assert torch.equal(state_a[name].cpu(), state_b[name].cpu()), name
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(uninterrupted(x), resumed(x), atol=0, rtol=0)
    assert all(isinstance(getattr(layer, category), VAELinear) for layer in resumed.model.layers for category in CATEGORIES)


def test_cat_round_checkpoint_tokenizer_receives_runtime_access_token(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_pretrained(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_from_pretrained)
    assert _load_checkpoint_tokenizer("real-model", "runtime-token") is sentinel
    assert captured == {
        "model_path": "real-model",
        "use_fast": True,
        "token": "runtime-token",
    }
