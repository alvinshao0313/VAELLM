"""Task 8 focused tests for train_utils.checkpoint_v6."""

from __future__ import annotations

import ast
import copy
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from train_utils.shared_protected_residual import register_shared_protected_residual_decoder


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
    with torch.no_grad():
        for idx, param in enumerate(decoder.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype).view_as(param)
            param.copy_(values + float(idx + 1))
    return decoder


def _single_stage_vae(*, protect_original: bool = False) -> VAELinear:
    bits = torch.tensor(
        [
            [[True, False, True, False, True, False, True, False, True]],
            [[False, True, False, True, False, True, False, True, False]],
            [[True, True, False, False, True, True, False, False, True]],
            [[False, False, True, True, False, False, True, True, False]],
        ],
        dtype=torch.bool,
    )
    original = torch.randn(4, 4) if protect_original else None
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=original,
        vq_weight=bits,
        decoder=_make_decoder(),
        codebook_dim=4,
        transpose=False,
        always_use_original=False,
        protect_original_weight=bool(protect_original),
    )


def _two_stage_vae() -> VAELinear:
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
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=None,
        decoder=None,
        stage_vq_weights=[[part0, part1], [~part0, ~part1]],
        stage_decoders=[
            [_make_decoder(), _make_decoder()],
            [_make_decoder(), _make_decoder()],
        ],
        codebook_dim=4,
        stage_codebook_dims=[4, 4],
        transpose=False,
        parallel_parts=2,
        parallel_rows=1,
        parallel_cols=2,
    )


def _protected_stage_bits() -> torch.Tensor:
    return torch.tensor(
        [
            [[True, False, True, False, True, False, True, False, True]],
            [[False, True, False, True, False, True, False, True, False]],
        ],
        dtype=torch.bool,
    )


def _single_stage_vae_with_sparse_coo() -> VAELinear:
    base = _single_stage_vae()
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=base.get_stage_part_vq_weight(stage_idx=0, part_idx=0),
        decoder=_make_decoder(),
        codebook_dim=4,
        transpose=False,
        sparse_residual_format="coo_fp16",
        sparse_residual_row_indices=torch.tensor([0, 3], dtype=torch.int64),
        sparse_residual_col_indices=torch.tensor([1, 2], dtype=torch.int64),
        sparse_residual_values=torch.tensor([0.25, -0.5], dtype=torch.float16),
    )


def _single_stage_vae_with_private_protected_residual() -> VAELinear:
    base = _single_stage_vae()
    bits = _protected_stage_bits()
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=base.get_stage_part_vq_weight(stage_idx=0, part_idx=0),
        decoder=_make_decoder(),
        codebook_dim=4,
        transpose=False,
        protected_residual_axis="input",
        protected_residual_indices=torch.tensor([1, 3], dtype=torch.int64),
        protected_residual_stage_vq_weights=[bits, ~bits],
        protected_residual_stage_decoders=[_make_decoder(), _make_decoder()],
        protected_residual_stage_codebook_dims=[4, 4],
    )


def _single_stage_vae_with_shared_protected_residual(host: nn.Module) -> VAELinear:
    base = _single_stage_vae()
    bits = _protected_stage_bits()
    shared0 = _make_decoder()
    shared1 = _make_decoder()
    register_shared_protected_residual_decoder(host, "protected_s0", shared0)
    register_shared_protected_residual_decoder(host, "protected_s1", shared1)
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=base.get_stage_part_vq_weight(stage_idx=0, part_idx=0),
        decoder=_make_decoder(),
        codebook_dim=4,
        transpose=False,
        protected_residual_axis="output",
        protected_residual_indices=torch.tensor([0, 2], dtype=torch.int64),
        protected_residual_stage_vq_weights=[bits, ~bits],
        protected_residual_shared_decoder_refs=["protected_s0", "protected_s1"],
        protected_residual_shared_stage_decoders=[shared0, shared1],
        protected_residual_stage_codebook_dims=[4, 4],
    )


class _Host(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.proj = layer
        self.pending = nn.Linear(4, 4, bias=False)
        self.skip = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


def _skeleton_like(saved: _Host) -> _Host:
    return _Host(nn.Linear(4, 4, bias=False))


def _decoded(layer: VAELinear) -> torch.Tensor:
    return layer._decode_weight(dtype=torch.float32).detach().cpu()


def test_checkpoint_v6_module_does_not_import_legacy_io():
    path = Path(v6.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    assert "train_utils.model_checkpoint_io" not in imported
    assert "e2e_common.checkpoint_io" not in imported
    assert not any(name.startswith("train_utils.model_checkpoint_io") for name in imported)


@pytest.mark.parametrize("kind", ["round_base", "category_boundary", "final_model"])
def test_full_checkpoint_kinds_roundtrip(kind):
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, kind)
        result = v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind=kind,
            compressed_targets=["proj"],
            pending_dense_targets=["pending"],
            skip_targets=["skip"],
            train_mode="decoder",
            completed_categories=[] if kind == "round_base" else ["q_proj"],
            resolved_learning_rates={"learning_rate": 1e-4, "decoder_lr": 1e-5},
        )
        meta = result["meta_payload"]
        assert meta["format"] == v6.FORMAT_V6
        assert meta["schema_version"] == 6
        assert meta["checkpoint_kind"] == kind
        assert meta["checkpoint_id"] == result["checkpoint_id"]
        assert meta["compressed_targets"] == ["proj"]
        assert meta["pending_dense_targets"] == ["pending"]
        assert meta["skip_targets"] == ["skip"]
        assert meta["lora_config"] is None
        if kind == "round_base":
            assert meta["completed_categories"] == []

        before = _decoded(model.proj)
        host = _skeleton_like(model)
        loaded, loaded_meta, _ = v6.load_v6_full_checkpoint_into_model(
            host, out, expected_kind=kind
        )
        assert isinstance(loaded.proj, VAELinear)
        assert isinstance(loaded.pending, nn.Linear)
        assert loaded_meta["checkpoint_kind"] == kind
        assert torch.allclose(before, _decoded(loaded.proj), atol=0, rtol=0)
        x = torch.randn(2, 4)
        assert torch.allclose(model(x), loaded(x), atol=1e-5, rtol=1e-5)


def _assert_extended_full_roundtrip(model: _Host, tmp: str) -> tuple[_Host, dict]:
    out = os.path.join(tmp, "extended")
    before_weight = _decoded(model.proj)
    x = torch.randn(2, 4)
    before_forward = model(x).detach().cpu()
    result = v6.save_v6_full_checkpoint(
        model,
        out,
        checkpoint_kind="category_boundary",
        compressed_targets=["proj"],
        pending_dense_targets=["pending"],
        skip_targets=["skip"],
        completed_categories=["q_proj"],
    )
    loaded, loaded_meta, _ = v6.load_v6_full_checkpoint_into_model(
        _skeleton_like(model),
        out,
        expected_kind="category_boundary",
    )
    assert result["meta_payload"]["checkpoint_id"] == loaded_meta["checkpoint_id"]
    assert torch.allclose(before_weight, _decoded(loaded.proj), atol=0, rtol=0)
    assert torch.allclose(before_forward, loaded(x).detach().cpu(), atol=1e-5, rtol=1e-5)
    return loaded, loaded_meta


def test_v6_full_roundtrip_preserves_sparse_coo_residual():
    model = _Host(_single_stage_vae_with_sparse_coo())
    with tempfile.TemporaryDirectory() as tmp:
        loaded, meta = _assert_extended_full_roundtrip(model, tmp)
    assert meta["converted_modules"][0]["sparse_residual_format"] == "coo_fp16"
    assert torch.equal(loaded.proj.sparse_residual_row_indices, model.proj.sparse_residual_row_indices)
    assert torch.equal(loaded.proj.sparse_residual_col_indices, model.proj.sparse_residual_col_indices)
    assert torch.equal(loaded.proj.sparse_residual_values, model.proj.sparse_residual_values)


def test_v6_full_roundtrip_preserves_private_protected_residual():
    model = _Host(_single_stage_vae_with_private_protected_residual())
    with tempfile.TemporaryDirectory() as tmp:
        loaded, meta = _assert_extended_full_roundtrip(model, tmp)
    converted = meta["converted_modules"][0]
    assert converted["protected_residual_axis"] == "input"
    assert converted["protected_residual_stages"] == 2
    assert converted["protected_residual_shared_decoder_refs"] is None
    assert loaded.proj.protected_residual_axis == "input"
    assert loaded.proj.protected_residual_stages == 2
    assert torch.equal(loaded.proj.protected_residual_indices, model.proj.protected_residual_indices)
    for stage_idx in range(2):
        assert torch.equal(
            loaded.proj.get_protected_residual_stage_vq_storage(stage_idx),
            model.proj.get_protected_residual_stage_vq_storage(stage_idx),
        )


def test_v6_full_roundtrip_preserves_shared_protected_residual_registry():
    model = _Host(nn.Linear(4, 4, bias=False))
    model.proj = _single_stage_vae_with_shared_protected_residual(model)
    with tempfile.TemporaryDirectory() as tmp:
        loaded, meta = _assert_extended_full_roundtrip(model, tmp)
    shared_specs = meta["shared_protected_residual_decoders"]
    assert [item["ref"] for item in shared_specs] == ["protected_s0", "protected_s1"]
    assert loaded.proj.protected_residual_shared_decoder_refs == ["protected_s0", "protected_s1"]
    registry = getattr(loaded, "_vaellm_shared_protected_residual_decoders")
    assert set(registry.keys()) == {"protected_s0", "protected_s1"}
    shared_runtime = loaded.proj.__dict__["_protected_residual_shared_stage_decoders"]
    assert shared_runtime[0] is registry["protected_s0"]
    assert shared_runtime[1] is registry["protected_s1"]


def test_round_base_does_not_default_completed_categories_advancement():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "round_base")
        result = v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="round_base",
            compressed_targets=["proj"],
            # omit completed_categories
        )
        assert result["meta_payload"]["completed_categories"] == []


def test_protect_original_weight_remains_compressed():
    layer = _single_stage_vae(protect_original=True)
    assert layer.always_use_original is False
    assert layer.protect_original_weight is True
    model = _Host(layer)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
        )
        host = _skeleton_like(model)
        loaded, meta, _ = v6.load_v6_full_checkpoint_into_model(host, out)
        assert isinstance(loaded.proj, VAELinear)
        assert loaded.proj.always_use_original is False
        assert loaded.proj.protect_original_weight is True
        assert "proj" in meta["compressed_targets"]
        assert "proj" not in meta["skip_targets"]
        assert torch.allclose(_decoded(model.proj), _decoded(loaded.proj), atol=0, rtol=0)


def test_low_rank_payload_parity():
    layer = _single_stage_vae()
    with torch.no_grad():
        layer.low_rank_a = nn.Parameter(torch.randn(4, 2))
        layer.low_rank_b = nn.Parameter(torch.randn(2, 4))
    model = _Host(layer)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
        )
        host = _skeleton_like(model)
        loaded, _, _ = v6.load_v6_full_checkpoint_into_model(host, out)
        assert torch.equal(model.proj.low_rank_a, loaded.proj.low_rank_a)
        assert torch.equal(model.proj.low_rank_b, loaded.proj.low_rank_b)


def test_stable_full_checkpoint_rejects_training_step_lora_config():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="lora_config=None"):
            v6.save_v6_full_checkpoint(
                model,
                os.path.join(tmp, "final_model"),
                checkpoint_kind="final_model",
                compressed_targets=["proj"],
                lora_config={"rank": 2, "alpha": 4.0, "dropout": 0.0},
            )


def test_stable_full_checkpoint_rejects_live_peft_wrapper():
    from e2e_common.full_lora import build_full_compressed_peft_model

    model = _Host(_single_stage_vae())
    peft_model = build_full_compressed_peft_model(
        model,
        selected_modules=[("proj", model.proj)],
        initial_low_rank_payloads=None,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="live PeftModel"):
            v6.save_v6_full_checkpoint(
                peft_model,
                os.path.join(tmp, "final_model"),
                checkpoint_kind="final_model",
                compressed_targets=["proj"],
            )


def test_stable_heterogeneous_low_rank_payload_roundtrip():
    class TwoHost(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a = a
            self.b = b

        def forward(self, x):
            return self.b(self.a(x))

    torch.manual_seed(29)
    model = TwoHost(_single_stage_vae(), _single_stage_vae())
    model.a.low_rank_a = nn.Parameter(torch.randn(4, 2), requires_grad=False)
    model.a.low_rank_b = nn.Parameter(torch.randn(2, 4), requires_grad=False)
    model.b.low_rank_a = nn.Parameter(torch.randn(4, 3), requires_grad=False)
    model.b.low_rank_b = nn.Parameter(torch.randn(3, 4), requires_grad=False)
    x = torch.randn(2, 4)
    expected = model(x).detach()

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        saved = v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["a", "b"],
        )
        assert saved["meta_payload"]["lora_config"] is None
        loaded, meta, _ = v6.load_v6_full_checkpoint_into_model(
            TwoHost(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)),
            out,
        )
        assert meta["lora_config"] is None
        assert int(loaded.a.low_rank_a.shape[1]) == 2
        assert int(loaded.b.low_rank_a.shape[1]) == 3
        torch.testing.assert_close(loaded(x), expected, rtol=1e-5, atol=1e-5)


def test_multi_stage_module_type_and_forward_parity():
    model = _Host(_two_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "category_boundary")
        v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="category_boundary",
            compressed_targets=["proj"],
            completed_categories=["q_proj"],
        )
        host = _skeleton_like(model)
        loaded, _, _ = v6.load_v6_full_checkpoint_into_model(
            host, out, expected_kind="category_boundary"
        )
        assert isinstance(loaded.proj, VAELinear)
        assert int(loaded.proj.residual_stages) == 2
        assert torch.allclose(_decoded(model.proj), _decoded(loaded.proj), atol=0, rtol=0)


def test_reject_wrong_checkpoint_kind_and_format():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
        )
        with pytest.raises(ValueError, match="Expected checkpoint_kind"):
            v6.load_v6_full_checkpoint_into_model(
                _skeleton_like(model), out, expected_kind="round_base"
            )

        meta_path = os.path.join(out, v6.META_FILENAME)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        meta["format"] = "vaellm_state_dict_with_meta"
        meta["schema_version"] = 5
        Path(meta_path).write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported checkpoint format"):
            v6.load_v6_meta(out)


def test_reject_target_inventory_overlap():
    with pytest.raises(ValueError, match="overlap"):
        v6.validate_target_inventories(["a"], ["a"], [])
    with pytest.raises(ValueError, match="overlap"):
        v6.validate_target_inventories(["a"], [], ["a"])
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="overlap"):
            v6.save_v6_full_checkpoint(
                model,
                os.path.join(tmp, "bad"),
                checkpoint_kind="final_model",
                compressed_targets=["proj"],
                skip_targets=["proj"],
            )


def test_reject_always_use_original_on_save():
    layer = _single_stage_vae(protect_original=True)
    layer.always_use_original = True
    model = _Host(layer)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="always_use_original=True"):
            v6.save_v6_full_checkpoint(
                model,
                os.path.join(tmp, "bad"),
                checkpoint_kind="final_model",
                compressed_targets=["proj"],
            )


def test_reject_incomplete_and_tmp_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_ckpt = os.path.join(tmp, ".final_model.tmp-deadbeef")
        os.makedirs(tmp_ckpt)
        Path(os.path.join(tmp_ckpt, v6.META_FILENAME)).write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="temp checkpoint|incomplete"):
            v6.resolve_v6_checkpoint_dir(tmp_ckpt)

        incomplete = os.path.join(tmp, "final_model")
        os.makedirs(incomplete)
        with pytest.raises((FileNotFoundError, ValueError)):
            v6.resolve_v6_checkpoint_dir(incomplete)


def test_training_step_round_base_id_mismatch_and_mutable_parity():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = os.path.join(tmp, "round_base")
        base = v6.save_v6_full_checkpoint(
            model,
            base_dir,
            checkpoint_kind="round_base",
            compressed_targets=["proj"],
        )
        step_dir = os.path.join(tmp, "checkpoint-10")
        os.makedirs(step_dir)
        sample_name, sample_param = next(iter(model.named_parameters()))
        mutable = {sample_name: sample_param.detach().cpu().clone()}
        manifest = v6.build_uniform_mutable_state_manifest(mutable, component_class="decoder")
        v6.save_v6_training_step_payload(
            step_dir,
            round_base_ref=base_dir,
            round_base_checkpoint_id=base["checkpoint_id"],
            mutable_state=mutable,
            mutable_state_manifest=manifest,
            train_mode="decoder",
            compressed_targets=["proj"],
            hf_artifact_refs={"optimizer": "optimizer.pt"},
        )
        step_meta = v6.load_v6_training_step_meta(step_dir)
        assert step_meta["checkpoint_kind"] == "training_step"
        assert step_meta["round_base_checkpoint_id"] == base["checkpoint_id"]
        assert "pytorch_model.bin" not in os.listdir(step_dir)

        base_meta = v6.load_v6_meta(base_dir)
        v6.validate_training_step_round_base(step_meta, base_meta)

        bad_base = dict(base_meta)
        bad_base["checkpoint_id"] = "not-the-same"
        with pytest.raises(ValueError, match="round_base_checkpoint_id"):
            v6.validate_training_step_round_base(step_meta, bad_base)

        state, loaded_manifest = v6.load_v6_training_model_state(step_dir)
        v6.validate_mutable_state_manifest(loaded_manifest, state)
        assert set(state.keys()) == set(mutable.keys())
        for key in mutable:
            assert torch.equal(state[key], mutable[key])


def test_training_step_lora_config_roundtrips_exact_rank_pattern_and_targets():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = os.path.join(tmp, "round_base")
        base = v6.save_v6_full_checkpoint(
            model,
            base_dir,
            checkpoint_kind="round_base",
            compressed_targets=["proj"],
        )
        step_dir = os.path.join(tmp, "checkpoint-1")
        os.makedirs(step_dir)
        sample_name, sample_param = next(iter(model.named_parameters()))
        mutable = {sample_name: sample_param.detach().cpu().clone()}
        manifest = v6.build_uniform_mutable_state_manifest(mutable, component_class="lora")
        exact_lora = {
            "rank": 3,
            "alpha": 6.0,
            "dropout": 0.0,
            "rank_pattern": {"proj": 2},
            "target_modules": ["proj"],
        }
        v6.save_v6_training_step_payload(
            step_dir,
            round_base_ref=base_dir,
            round_base_checkpoint_id=base["checkpoint_id"],
            mutable_state=mutable,
            mutable_state_manifest=manifest,
            train_mode="lora",
            compressed_targets=["proj"],
            lora_config=exact_lora,
        )
        assert v6.load_v6_training_step_meta(step_dir)["lora_config"] == exact_lora


def test_mutable_manifest_mismatch_hard_error():
    tensors = {"a": torch.zeros(2, 3)}
    manifest = v6.build_mutable_state_manifest(
        tensors, component_classes={"a": "lora"}
    )
    with pytest.raises(ValueError):
        v6.validate_mutable_state_manifest(manifest, {"a": torch.zeros(3, 2)})
    with pytest.raises(ValueError):
        v6.validate_mutable_state_manifest(manifest, {"a": torch.zeros(2, 3), "extra": torch.zeros(1)})


def test_compressed_inventory_under_and_over_report_hard_error():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="compressed_targets must exactly match"):
            v6.save_v6_full_checkpoint(
                model,
                os.path.join(tmp, "under"),
                checkpoint_kind="final_model",
                compressed_targets=[],
            )
        with pytest.raises(ValueError, match="compressed_targets must exactly match"):
            v6.save_v6_full_checkpoint(
                model,
                os.path.join(tmp, "over"),
                checkpoint_kind="final_model",
                compressed_targets=["proj", "ghost"],
            )


def test_pending_and_skip_wrong_module_type_hard_error():
    class _NonLinearPendingHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _single_stage_vae()
            self.pending = nn.Embedding(4, 4)
            self.skip = nn.Linear(4, 4, bias=False)

    model = _NonLinearPendingHost()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(TypeError, match="pending_dense_targets.*nn.Linear"):
            v6.save_v6_full_checkpoint(
                model,
                os.path.join(tmp, "bad_pending"),
                checkpoint_kind="final_model",
                compressed_targets=["proj"],
                pending_dense_targets=["pending"],
            )

    # Direct type gate: pending/skip must not resolve to VAELinear.
    good = _Host(_single_stage_vae())
    with pytest.raises(ValueError, match="pending_dense_targets.*VAELinear"):
        v6._require_ordinary_linear_target(
            good, "proj", inventory_name="pending_dense_targets"
        )
    with pytest.raises(ValueError, match="skip_targets.*VAELinear"):
        v6._require_ordinary_linear_target(good, "proj", inventory_name="skip_targets")

    with pytest.raises(ValueError):
        v6.validate_model_target_inventories(
            good,
            compressed=["proj"],
            pending=["pending"],
            skip=["proj"],
        )
    with pytest.raises(ValueError, match="legacy_original_only_sources must be a subset"):
        v6.validate_model_target_inventories(
            good,
            compressed=["proj"],
            pending=["pending"],
            skip=["skip"],
            legacy_original_only_sources=["not_in_skip"],
        )


def _pack_incompatible_multi_stage_vae() -> VAELinear:
    """Multi-stage layer with unequal stage codebook dims (serial-only; cannot pack)."""
    # stage0 cd=4: 2 parts * 2 blocks * 4 = 16
    s0p0 = torch.randint(0, 2, (2, 1, 9), dtype=torch.bool)
    s0p1 = torch.randint(0, 2, (2, 1, 9), dtype=torch.bool)
    # stage1 cd=8: 2 parts * 1 block * 8 = 16
    s1p0 = torch.randint(0, 2, (1, 1, 9), dtype=torch.bool)
    s1p1 = torch.randint(0, 2, (1, 1, 9), dtype=torch.bool)
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=None,
        decoder=None,
        stage_vq_weights=[[s0p0, s0p1], [s1p0, s1p1]],
        stage_decoders=[
            [_make_decoder(codebook_dim=4), _make_decoder(codebook_dim=4)],
            [_make_decoder(codebook_dim=8), _make_decoder(codebook_dim=8)],
        ],
        codebook_dim=4,
        stage_codebook_dims=[4, 8],
        transpose=False,
        parallel_parts=2,
        parallel_rows=1,
        parallel_cols=2,
    )


def test_serial_pack_incompatible_multi_stage_full_roundtrip():
    layer = _pack_incompatible_multi_stage_vae()
    assert getattr(layer, "_parallel_stage_decoder", None) is None
    with pytest.raises(ValueError, match="identical stage codebook dims"):
        layer.pack_parallel_stage_decoder_(trainable=False)
    model = _Host(layer)
    before = _decoded(model.proj)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        result = v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
            pending_dense_targets=["pending"],
            skip_targets=["skip"],
        )
        meta = result["meta_payload"]
        assert meta["converted_modules"][0]["parallel_stage_decode"] is False
        assert meta["converted_modules"][0]["stage_codebook_dims"] == [4, 8]
        # Topology must remain serial after save (no force-pack).
        assert getattr(model.proj, "_parallel_stage_decoder", None) is None

        host = _skeleton_like(model)
        loaded, _, _ = v6.load_v6_full_checkpoint_into_model(host, out)
        assert isinstance(loaded.proj, VAELinear)
        assert getattr(loaded.proj, "_parallel_stage_decoder", None) is None
        assert list(loaded.proj.stage_codebook_dims) == [4, 8]
        assert torch.allclose(before, _decoded(loaded.proj), atol=0, rtol=0)


def test_training_step_stale_meta_crash_injection_rejects_loader():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = os.path.join(tmp, "round_base")
        base = v6.save_v6_full_checkpoint(
            model,
            base_dir,
            checkpoint_kind="round_base",
            compressed_targets=["proj"],
        )
        step_dir = os.path.join(tmp, "checkpoint-10")
        os.makedirs(step_dir)
        sample_name, sample_param = next(iter(model.named_parameters()))
        mutable_old = {sample_name: sample_param.detach().cpu().clone()}
        manifest_old = v6.build_uniform_mutable_state_manifest(mutable_old, component_class="decoder")
        v6.save_v6_training_step_payload(
            step_dir,
            round_base_ref="../round_base",
            round_base_checkpoint_id=base["checkpoint_id"],
            mutable_state=mutable_old,
            mutable_state_manifest=manifest_old,
            train_mode="decoder",
            compressed_targets=["proj"],
        )

        # Simulate overwrite crash: invalidate meta, write new state, stop before new meta.
        meta_path = os.path.join(step_dir, v6.META_FILENAME)
        v6._invalidate_existing_meta_marker(meta_path)
        assert not os.path.isfile(meta_path)
        mutable_new = {sample_name: (sample_param.detach().cpu().clone() + 1.0)}
        torch.save(mutable_new, os.path.join(step_dir, v6.TRAINING_MODEL_STATE_FILENAME))

        with pytest.raises(FileNotFoundError, match="Missing checkpoint_meta"):
            v6.load_v6_training_step_meta(step_dir)
        with pytest.raises(FileNotFoundError, match="Missing checkpoint_meta"):
            v6.load_v6_training_model_state(step_dir)

        # Default refuse silent overwrite when valid payload already exists.
        with pytest.raises(FileExistsError, match="allow_overwrite"):
            v6.save_v6_training_step_payload(
                step_dir,
                round_base_ref="../round_base",
                round_base_checkpoint_id=base["checkpoint_id"],
                mutable_state=mutable_new,
                mutable_state_manifest=manifest_old,
                train_mode="decoder",
                compressed_targets=["proj"],
            )


def test_multi_rank_canonical_checkpoint_id():
    model = _Host(_single_stage_vae())
    barriers = []

    def barrier():
        barriers.append("hit")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        main = v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
            pending_dense_targets=["pending"],
            skip_targets=["skip"],
            is_main_process=True,
            distributed_barrier=barrier,
        )
        non_main = v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
            pending_dense_targets=["pending"],
            skip_targets=["skip"],
            is_main_process=False,
            distributed_barrier=barrier,
        )
        assert main["checkpoint_id"] == non_main["checkpoint_id"]
        assert main["meta_payload"]["checkpoint_id"] == non_main["meta_payload"]["checkpoint_id"]
        assert len(barriers) == 2

        step_dir = os.path.join(tmp, "checkpoint-1")
        os.makedirs(step_dir)
        sample_name, sample_param = next(iter(model.named_parameters()))
        mutable = {sample_name: sample_param.detach().cpu().clone()}
        manifest = v6.build_uniform_mutable_state_manifest(mutable, component_class="decoder")
        main_step = v6.save_v6_training_step_payload(
            step_dir,
            round_base_ref=out,
            round_base_checkpoint_id=main["checkpoint_id"],
            mutable_state=mutable,
            mutable_state_manifest=manifest,
            train_mode="decoder",
            compressed_targets=["proj"],
            is_main_process=True,
            distributed_barrier=barrier,
        )
        non_main_step = v6.save_v6_training_step_payload(
            step_dir,
            round_base_ref=out,
            round_base_checkpoint_id=main["checkpoint_id"],
            mutable_state=mutable,
            mutable_state_manifest=manifest,
            train_mode="decoder",
            compressed_targets=["proj"],
            is_main_process=False,
            distributed_barrier=barrier,
        )
        assert main_step["checkpoint_id"] == non_main_step["checkpoint_id"]


def test_mutable_manifest_duplicate_name_class_and_identity():
    a = torch.zeros(2, 2)
    b = torch.ones(2, 2)
    with pytest.raises(ValueError, match="component_class|sparse_bit"):
        v6.build_mutable_state_manifest({"a": a}, component_classes={"a": "sparse_bit"})
    with pytest.raises(ValueError, match="duplicate tensor identity"):
        v6.build_mutable_state_manifest(
            {"a": a, "b": a},
            component_classes={"a": "lora", "b": "decoder"},
        )
    manifest = [
        {"name": "a", "shape": [2, 2], "dtype": "float32", "component_class": "lora"},
        {"name": "a", "shape": [2, 2], "dtype": "float32", "component_class": "lora"},
    ]
    with pytest.raises(ValueError, match="duplicate name in mutable_state_manifest"):
        v6.validate_mutable_state_manifest(manifest, {"a": a})

    good = v6.build_mutable_state_manifest(
        {"a": a, "b": b},
        component_classes={"a": "lora", "b": "decoder"},
    )
    with pytest.raises(ValueError, match="component_class mismatch"):
        v6.validate_mutable_state_manifest(
            good,
            {"a": a, "b": b},
            component_classes={"a": "norm", "b": "decoder"},
        )


def test_relative_round_base_ref_survives_tree_move():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        run_a = os.path.join(tmp, "run_a")
        os.makedirs(run_a)
        base_dir = os.path.join(run_a, "round_base")
        step_dir = os.path.join(run_a, "checkpoint-10")
        os.makedirs(step_dir)
        base = v6.save_v6_full_checkpoint(
            model,
            base_dir,
            checkpoint_kind="round_base",
            compressed_targets=["proj"],
        )
        sample_name, sample_param = next(iter(model.named_parameters()))
        mutable = {sample_name: sample_param.detach().cpu().clone()}
        manifest = v6.build_uniform_mutable_state_manifest(mutable, component_class="decoder")
        v6.save_v6_training_step_payload(
            step_dir,
            round_base_ref="../round_base",
            round_base_checkpoint_id=base["checkpoint_id"],
            mutable_state=mutable,
            mutable_state_manifest=manifest,
            train_mode="decoder",
            compressed_targets=["proj"],
        )

        run_b = os.path.join(tmp, "run_b")
        os.rename(run_a, run_b)
        moved_step = os.path.join(run_b, "checkpoint-10")
        step_meta = v6.load_v6_training_step_meta(moved_step)
        resolved, base_meta = v6.resolve_training_step_round_base_ref(moved_step, step_meta)
        assert resolved == os.path.abspath(os.path.join(run_b, "round_base"))
        assert base_meta["checkpoint_id"] == base["checkpoint_id"]

        step_meta["round_base_checkpoint_id"] = "wrong-id"
        with pytest.raises(ValueError, match="round_base_checkpoint_id"):
            v6.resolve_training_step_round_base_ref(moved_step, step_meta)


def test_converted_module_count_and_name_schema_mismatch():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
            pending_dense_targets=["pending"],
            skip_targets=["skip"],
        )
        meta_path = os.path.join(out, v6.META_FILENAME)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        meta["converted_module_count"] = int(meta["converted_module_count"]) + 1
        Path(meta_path).write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="converted_module_count"):
            v6.load_v6_meta(out)

        meta["converted_module_count"] = len(meta["converted_modules"])
        meta["converted_modules"].append(dict(meta["converted_modules"][0]))
        meta["converted_module_count"] = len(meta["converted_modules"])
        Path(meta_path).write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate converted_modules name"):
            v6.validate_full_converted_modules_meta(meta)


def test_bad_meta_fails_before_model_mutation():
    model = _Host(_single_stage_vae())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "final_model")
        v6.save_v6_full_checkpoint(
            model,
            out,
            checkpoint_kind="final_model",
            compressed_targets=["proj"],
            pending_dense_targets=["pending"],
            skip_targets=["skip"],
        )
        meta_path = os.path.join(out, v6.META_FILENAME)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        meta["converted_module_count"] = 999
        Path(meta_path).write_text(json.dumps(meta), encoding="utf-8")

        host = _skeleton_like(model)
        assert isinstance(host.proj, nn.Linear)
        with pytest.raises(ValueError, match="converted_module_count"):
            v6.load_v6_full_checkpoint_into_model(host, out)
        # Model must not have been half-mutated into VAELinear.
        assert isinstance(host.proj, nn.Linear)
        assert not isinstance(host.proj, VAELinear)
