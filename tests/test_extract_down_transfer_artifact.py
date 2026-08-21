from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from tools.extract_down_transfer_artifact import (
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    SHARED_DECODER_ERROR,
    TRANSFER_META_FILENAME,
    _build_transfer_module_specs,
    _extract_compressed_state,
    _reject_shared_protected_residual_decoders,
    _validate_compressed_state,
    extract_down_transfer_artifact,
)


def _make_down_spec(
    layer_idx: int,
    *,
    residual_stages: int = 2,
    has_original_weight: bool = True,
    protected_input_indices: dict | None = {"shape": [4], "dtype": "int64"},
    protected_input_weight: dict | None = {"shape": [8, 4], "dtype": "bfloat16"},
    shared_decoder_refs: list[str] | None = None,
) -> dict:
    name = f"model.layers.{layer_idx}.mlp.down_proj"
    spec = {
        "name": name,
        "in_features": 16,
        "out_features": 8,
        "compressed_in_features": 12,
        "compressed_out_features": 8,
        "codebook_dim": 4,
        "transpose": True,
        "parallel_parts": 1,
        "parallel_rows": 1,
        "parallel_cols": 1,
        "residual_stages": residual_stages,
        "stage_codebook_dims": [4] * residual_stages,
        "parallel_stage_decode": False,
        "has_bias": False,
        "has_original_weight": has_original_weight,
        "always_use_original": False,
        "protect_original_weight": False,
        "vq_weights": [{"shape": [8, 1, 1], "dtype": "uint8"}],
        "decoders": [{"in_dim": 8, "out_dim": 4, "hidden_dim": 8, "num_res_blocks": 0,
                      "norm_type": "layer", "activation_type": "swish", "decoder_type": "linear",
                      "use_checkpoint": False, "param_dtype": "float32"}],
        "protected_input_indices": protected_input_indices,
        "protected_input_weight": protected_input_weight,
        "protected_residual_shared_decoder_refs": shared_decoder_refs,
    }
    return spec


def _make_down_state(name: str, *, with_original: bool = True, with_stage1: bool = True) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        f"{name}.vq_weight": torch.randint(0, 255, (8, 1, 1), dtype=torch.uint8),
        f"{name}.protected_input_indices": torch.arange(4, dtype=torch.int64),
        f"{name}.protected_input_weight": torch.randn(8, 4, dtype=torch.bfloat16),
        f"{name}.decoder.linear_in.weight": torch.randn(4, 8),
        f"{name}.low_rank_a": torch.randn(2, 16),
        f"{name}.sparse_residual_values": torch.randn(3),
    }
    if with_stage1:
        state[f"{name}.vq_weight_s1"] = torch.randint(0, 255, (8, 1, 1), dtype=torch.uint8)
        state[f"{name}.decoder_s1.linear_in.weight"] = torch.randn(4, 8)
    if with_original:
        state[f"{name}.original_weight"] = torch.randn(8, 16, dtype=torch.bfloat16)
    return state


def _write_source_checkpoint(tmp_path: Path, source_state: dict[str, torch.Tensor], converted_modules: list[dict]) -> Path:
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "final_model"
    ckpt_dir.mkdir(parents=True)
    torch.save(source_state, ckpt_dir / "pytorch_model.bin")
    meta = {
        "format": "vaellm_state_dict_with_meta",
        "version": 5,
        "base_model_path": "Qwen/Qwen3-8B",
        "state_dict_file": "pytorch_model.bin",
        "converted_module_count": len(converted_modules),
        "converted_modules": converted_modules,
    }
    (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return run_dir


def _build_mixed_source_state() -> tuple[dict[str, torch.Tensor], list[dict]]:
    down0 = "model.layers.0.mlp.down_proj"
    down1 = "model.layers.1.mlp.down_proj"
    source_state: dict[str, torch.Tensor] = {}
    source_state.update(_make_down_state(down0))
    source_state.update(_make_down_state(down1))
    source_state.update({
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(8, 16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(8, 16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(32, 16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(32, 16),
        "model.embed_tokens.weight": torch.randn(128, 16),
        "model.norm.weight": torch.randn(16),
        "lm_head.weight": torch.randn(128, 16),
    })
    converted_modules = [
        _make_down_spec(0),
        _make_down_spec(1),
        {"name": "model.layers.0.self_attn.q_proj", "residual_stages": 1},
    ]
    return source_state, converted_modules


def test_artifact_contains_only_down_prefixes(tmp_path: Path):
    source_state, converted_modules = _build_mixed_source_state()
    run_dir = _write_source_checkpoint(tmp_path, source_state, converted_modules)
    out_dir = tmp_path / "artifact"

    result = extract_down_transfer_artifact(
        source_checkpoint=str(run_dir),
        output_dir=str(out_dir),
    )
    artifact_state = result["compressed_state"]
    prefixes = ("model.layers.0.mlp.down_proj.", "model.layers.1.mlp.down_proj.")
    assert artifact_state
    assert all(k.startswith(prefixes) for k in artifact_state)
    forbidden_prefixes = (
        "model.layers.0.self_attn.",
        "model.layers.0.mlp.gate_proj.",
        "model.layers.0.mlp.up_proj.",
        "model.embed_tokens.",
        "model.norm.",
        "lm_head.",
    )
    assert not any(k.startswith(forbidden_prefixes) for k in artifact_state)


def test_artifact_keeps_future_down_prefix_fields(tmp_path: Path):
    down_name = "model.layers.0.mlp.down_proj"
    source_state = _make_down_state(down_name)
    converted_modules = [_make_down_spec(0)]
    run_dir = _write_source_checkpoint(tmp_path, source_state, converted_modules)

    artifact_state = _extract_compressed_state(source_state, [down_name])
    assert f"{down_name}.low_rank_a" in artifact_state
    assert f"{down_name}.sparse_residual_values" in artifact_state


def test_original_weight_is_excluded_but_other_down_state_kept():
    down_name = "model.layers.0.mlp.down_proj"
    source_state = _make_down_state(down_name, with_original=True)
    artifact_state = _extract_compressed_state(source_state, [down_name])

    assert not any(k.endswith(".original_weight") for k in artifact_state)
    assert f"{down_name}.vq_weight" in artifact_state
    assert f"{down_name}.protected_input_indices" in artifact_state


def test_transfer_specs_set_has_original_weight_false_without_mutating_donor():
    donor_specs = [_make_down_spec(0, has_original_weight=True)]
    donor_copy = copy.deepcopy(donor_specs)

    transfer_specs = _build_transfer_module_specs(donor_specs)
    assert transfer_specs[0]["has_original_weight"] is False
    assert donor_specs[0]["has_original_weight"] is True
    assert donor_copy[0]["has_original_weight"] is True


def test_protected_input_state_required_by_metadata():
    down_name = "model.layers.0.mlp.down_proj"
    spec = _make_down_spec(0)
    state = _make_down_state(down_name)
    del state[f"{down_name}.protected_input_indices"]
    compressed = _extract_compressed_state(state, [down_name])

    with pytest.raises(ValueError, match="protected_input_indices"):
        _validate_compressed_state(compressed, [spec])


def test_missing_vq_weight_s1_fails_for_two_stage_module():
    down_name = "model.layers.0.mlp.down_proj"
    spec = _make_down_spec(0, residual_stages=2)
    state = _make_down_state(down_name, with_stage1=False)
    compressed = _extract_compressed_state(state, [down_name])

    with pytest.raises(ValueError, match="vq_weight_s1"):
        _validate_compressed_state(compressed, [spec])


def test_shared_protected_residual_decoder_ref_fails():
    spec = _make_down_spec(0, shared_decoder_refs=["shared_decoder_a"])
    with pytest.raises(ValueError, match=SHARED_DECODER_ERROR):
        _reject_shared_protected_residual_decoders([spec])


def test_saved_artifact_tensors_are_bit_exact(tmp_path: Path):
    source_state, converted_modules = _build_mixed_source_state()
    run_dir = _write_source_checkpoint(tmp_path, source_state, converted_modules)
    out_dir = tmp_path / "artifact"

    result = extract_down_transfer_artifact(
        source_checkpoint=str(run_dir),
        output_dir=str(out_dir),
    )
    loaded = torch.load(out_dir / "down_proj_compressed_state.pt", map_location="cpu", weights_only=False)
    for key, tensor in result["compressed_state"].items():
        assert torch.equal(loaded[key], source_state[key])


def test_transfer_meta_schema_and_non_empty_output_guard(tmp_path: Path):
    source_state, converted_modules = _build_mixed_source_state()
    run_dir = _write_source_checkpoint(tmp_path, source_state, converted_modules)
    out_dir = tmp_path / "artifact"

    extract_down_transfer_artifact(source_checkpoint=str(run_dir), output_dir=str(out_dir))
    meta = json.loads((out_dir / TRANSFER_META_FILENAME).read_text(encoding="utf-8"))

    assert meta["format"] == ARTIFACT_FORMAT
    assert meta["version"] == ARTIFACT_VERSION
    assert meta["module_count"] == 2
    assert meta["module_names"] == [
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    ]
    assert all(spec["has_original_weight"] is False for spec in meta["module_specs"])
    assert meta["original_weight_policy"] == "excluded_for_transfer"

    with pytest.raises(FileExistsError, match="not empty"):
        extract_down_transfer_artifact(source_checkpoint=str(run_dir), output_dir=str(out_dir))


def test_donor_meta_file_not_modified(tmp_path: Path):
    source_state, converted_modules = _build_mixed_source_state()
    run_dir = _write_source_checkpoint(tmp_path, source_state, converted_modules)
    meta_path = run_dir / "final_model" / "checkpoint_meta.json"
    before = meta_path.read_bytes()

    extract_down_transfer_artifact(
        source_checkpoint=str(run_dir),
        output_dir=str(tmp_path / "artifact"),
    )
    after = meta_path.read_bytes()
    assert before == after
