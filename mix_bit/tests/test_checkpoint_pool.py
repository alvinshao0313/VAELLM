from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from mix_bit.candidate_artifact import (
    CANDIDATE_META_FILENAME,
    COMPLETED_FILENAME,
    MODULE_STATE_FILENAME,
)
from mix_bit.candidate_pool import candidate_pool_root, generate_candidate_trials
from mix_bit.model_inventory import (
    ModelInventory,
    TargetLinearSpec,
    inventory_from_targets,
    write_model_inventory,
)
from mix_bit.schema import (
    CalibrationConfig,
    CandidateMode,
    CandidateSpaceConfig,
    CandidateTrainingSpec,
    CategorySpec,
    MixBitRunConfig,
    ModelProfile,
    ResolvedRunConfig,
    TrainingRecipeConfig,
    resolve_run_config,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY_PATH = REPO_ROOT / ".result/mix_bit/qwen3_8b/model_inventory.json"


class _Block(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden, hidden, bias=False),
                "k_proj": nn.Linear(hidden, hidden, bias=False),
            }
        )


class _ToyModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = 8):
        super().__init__()
        self.model = nn.ModuleDict({"layers": nn.ModuleList([_Block(hidden) for _ in range(n_layers)])})
        self.config = type("Cfg", (), {"model_type": "toy", "_name_or_path": "toy"})()


def _toy_profile(
    categories: list[CategorySpec] | None = None,
) -> ModelProfile:
    if categories is None:
        categories = [
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("k_proj", "k_proj", False),
        ]
    return ModelProfile(
        model_id="toy",
        model_path="toy-model",
        adapter="generic_decoder",
        only_decoder_projections=True,
        candidate_training=CandidateTrainingSpec(linear_group_size="all", allow_tail_group=True),
        layer_index_patterns=(r"(?:^|\.)model\.layers\.(\d+)\.",),
        categories=tuple(categories),
        regression_expectations={},
    )


def _toy_modes(n: int = 2) -> tuple[CandidateMode, ...]:
    return tuple(
        CandidateMode(
            name=f"b{16 * (i + 1)}d4s2",
            nominal_bit=float(2 * (16 * (i + 1)) // 4),
            codebook_bits=16 * (i + 1),
            codebook_dim=4,
            residual_stages=2,
        )
        for i in range(n)
    )


def _make_resolved(
    tmp_path: Path,
    *,
    profile: ModelProfile | None = None,
    modes: tuple[CandidateMode, ...] | None = None,
) -> ResolvedRunConfig:
    profile = profile or _toy_profile()
    modes = modes or _toy_modes()
    recipe = TrainingRecipeConfig(
        recipe_id="toy_recipe",
        values={
            "seed": 31,
            "deterministic": True,
            "steps_per_category": 10,
            "batch_size": 8,
            "base_ch": 8,
            "num_res_blocks": 0,
            "decoder_base_ch": 8,
            "decoder_num_res_blocks": 0,
            "norm_type": "layer",
            "activation_type": "swish",
            "decoder_type": "linear",
            "recon_loss_type": "mse",
            "quantizer_type": "BSQ",
            "gamma0": 1.0,
            "gamma": 1.0,
            "zeta": 1.0,
            "inv_temperature": 100.0,
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "lr_scheduler": "linear",
            "lr_warmup_steps": 0,
            "l1_weight": 1.0,
            "lfq_weight": 1.0,
            "commitment_loss_weight": 0.25,
            "entropy_loss_weight": 0.01,
            "normalize_weight": True,
            "vae_decoder_checkpoint": True,
            "new_quant": True,
            "log_every": 1,
            "eval_every": 0,
            "eval_blocks": 8,
            "outlier_protect_mode": "channel",
            "outlier_protect_count": 0,
            "outlier_protect_min_per_layer": 0,
            "distill_after_category": "none",
            "eval_ppl": False,
            "eval_tasks": "",
            "rot_llm": False,
            "fp16": False,
            "bf16": True,
        },
    )
    config = MixBitRunConfig(
        run_id="toy_run",
        model_profile=profile,
        candidate_space=CandidateSpaceConfig(
            candidate_space_id="toy_space",
            baseline_mode=modes[0].name,
            target_average_bit=2.0,
            modes=modes,
        ),
        training_recipe=recipe,
        calibration=CalibrationConfig(
            source_jsonl=str(tmp_path / "calib.jsonl"),
            input_format="text",
            max_samples=1,
            max_length=8,
            seed=0,
            label_mode="all_nonpad",
        ),
    )
    root = tmp_path / "result" / "mix_bit" / profile.model_id
    return ResolvedRunConfig(
        config=config,
        run_config_path=str(tmp_path / "run.json"),
        run_config_sha256="r" * 64,
        model_profile_path=str(tmp_path / "profile.json"),
        model_profile_sha256="p" * 64,
        candidate_space_path=str(tmp_path / "space.json"),
        candidate_space_sha256="c" * 64,
        training_recipe_path=str(tmp_path / "recipe.json"),
        training_recipe_sha256="t" * 64,
        canonical_model_root=str(root),
        canonical_run_root=str(root / "runs" / config.run_id),
    )


def _inventory_for(profile: ModelProfile, model: nn.Module) -> ModelInventory:
    from mix_bit.model_adapter import get_model_adapter

    adapter = get_model_adapter("generic_decoder")
    targets = adapter.discover_target_linears(model, profile)
    return inventory_from_targets(
        profile=profile,
        model=model,
        targets=targets,
        model_profile_sha256="p" * 64,
    )


def _packed_vq_spec(bits: int) -> dict[str, Any]:
    return {
        "storage_format": "bitpack_u8",
        "dtype": "uint8",
        "logical_dtype": "bool",
        "pack_bits": 8,
        "logical_shape": [8, 1, bits],
        "shape": [8, 1, (bits + 7) // 8],
    }


def _decoder_spec(bits: int, dim: int) -> dict[str, Any]:
    return {
        "in_dim": bits,
        "out_dim": dim,
        "hidden_dim": 8,
        "num_res_blocks": 0,
        "norm_type": "layer",
        "activation_type": "swish",
        "decoder_type": "linear",
        "use_checkpoint": False,
        "param_dtype": "float32",
    }


def _module_spec_for(target: TargetLinearSpec, mode: CandidateMode | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": target.module_name,
        "in_features": target.in_features,
        "out_features": target.out_features,
        "has_bias": target.has_bias,
        "transpose": target.transpose,
        "has_original_weight": False,
        "protected_input_indices": None,
        "protected_input_weight": None,
        "protected_output_indices": None,
        "protected_output_weight": None,
        "low_rank_a": None,
        "low_rank_b": None,
        "protected_residual_indices": None,
        "protected_residual_stages": 0,
        "sparse_residual_indices": None,
        "sparse_residual_values": None,
    }
    if mode is not None:
        bits = int(mode.codebook_bits)
        dim = int(mode.codebook_dim)
        stages = int(mode.residual_stages)
        spec.update(
            {
                "codebook_dim": dim,
                "residual_stages": stages,
                "stage_codebook_dims": [dim] * stages,
                "parallel_parts": 1,
                "stage_vq_weights": [_packed_vq_spec(bits) for _ in range(stages)],
                "stage_decoders": [_decoder_spec(bits, dim) for _ in range(stages)],
                "vq_weights": [_packed_vq_spec(bits)],
                "decoders": [_decoder_spec(bits, dim)],
            }
        )
    return spec


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_artifact(
    trial_root: Path,
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    category_name: str,
    mode: CandidateMode,
    module_specs: list[dict[str, Any]],
    expected_module_names: list[str],
    payload_keys: list[str] | None = None,
    state_bytes: bytes | None = None,
    mutate_meta: dict[str, Any] | None = None,
    mutate_completed: dict[str, Any] | None = None,
    corrupt_state_hash_in_meta: bool = False,
) -> None:
    artifact = trial_root / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    if payload_keys is None:
        payload_keys = [f"{name}.vq_weight" for name in expected_module_names]
    if state_bytes is None:
        # Compact opaque bytes; indexing must not depend on tensor contents.
        state_bytes = ("STATE:" + "|".join(payload_keys)).encode("utf-8")
    state_path = artifact / MODULE_STATE_FILENAME
    state_sha = _write_bytes(state_path, state_bytes)
    payload_summaries = {
        key: {"dtype": "uint8", "shape": [1], "nbytes": 1} for key in payload_keys
    }
    meta = {
        "format": "vaellm_candidate_modules_v1",
        "module_specs": module_specs,
        "expected_module_names": expected_module_names,
        "payload_summaries": payload_summaries,
        "run_config_sha256": resolved.run_config_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "mode": {
            "name": mode.name,
            "nominal_bit": mode.nominal_bit,
            "codebook_bits": mode.codebook_bits,
            "codebook_dim": mode.codebook_dim,
            "residual_stages": mode.residual_stages,
        },
        "category_name": category_name,
        "source_run_dir": str(trial_root / "runs" / "fake"),
        "module_state_file": MODULE_STATE_FILENAME,
        "module_state_sha256": ("0" * 64) if corrupt_state_hash_in_meta else state_sha,
    }
    if mutate_meta:
        meta.update(mutate_meta)
    meta_path = artifact / CANDIDATE_META_FILENAME
    _write_json(meta_path, meta)
    meta_sha = sha256_file(meta_path)
    completed = {
        "format": "vaellm_candidate_modules_v1",
        "module_state_sha256": state_sha,
        "candidate_meta_sha256": meta_sha,
        "module_count": len(expected_module_names),
    }
    if mutate_completed:
        completed.update(mutate_completed)
    _write_json(artifact / COMPLETED_FILENAME, completed)


def _populate_valid_pool(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    *,
    output_root: Path | None = None,
) -> Path:
    trials = generate_candidate_trials(resolved, inventory, output_root=str(output_root) if output_root else None)
    targets_by_cat: dict[str, list[TargetLinearSpec]] = {}
    for target in inventory.targets:
        targets_by_cat.setdefault(target.category, []).append(target)
    for trial in trials:
        specs = [_module_spec_for(t, trial.mode) for t in targets_by_cat[trial.category_name]]
        names = [t.module_name for t in targets_by_cat[trial.category_name]]
        _write_artifact(
            Path(trial.trial_root),
            resolved=resolved,
            inventory=inventory,
            category_name=trial.category_name,
            mode=trial.mode,
            module_specs=specs,
            expected_module_names=names,
        )
    return candidate_pool_root(resolved, output_root=str(output_root) if output_root else None)


def test_index_maps_every_module_and_mode_to_one_source(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(2)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    index = build_candidate_pool_index(resolved, inventory)
    assert index.category_count == 2
    assert index.mode_count == 2
    assert index.target_linear_count == 4
    assert index.expected_trial_count == 4
    assert index.dense_module_mode_count == 8
    assert len(index.candidates) == 8
    for target in inventory.targets:
        for mode in modes:
            key = (target.module_name, mode.name)
            cand = index.candidates[key]
            assert cand.source.category == target.category
            assert cand.source.mode_name == mode.name
            assert cand.in_features == target.in_features
            assert cand.out_features == target.out_features
            assert cand.param_count == target.in_features * target.out_features
            assert cand.block_index == target.block_index


def test_category_checkpoint_rejects_wrong_category_module(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    k_targets = [t for t in inventory.targets if t.category == "k_proj"]
    bad_specs = [_module_spec_for(t, modes[0]) for t in q_targets + k_targets[:1]]
    bad_names = [t.module_name for t in q_targets + k_targets[:1]]
    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=bad_specs,
        expected_module_names=bad_names,
    )
    with pytest.raises(ValueError, match="wrong category|exact inventory target|unexpected"):
        build_candidate_pool_index(resolved, inventory)


def test_missing_inventory_target_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    incomplete = q_targets[:-1]
    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in incomplete],
        expected_module_names=[t.module_name for t in incomplete],
    )
    with pytest.raises(ValueError, match="missing|exact inventory target"):
        build_candidate_pool_index(resolved, inventory)


def test_inventory_fingerprint_mismatch_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        mutate_meta={"model_inventory_fingerprint": "f" * 64},
    )
    with pytest.raises(ValueError, match="inventory fingerprint|fingerprint mismatch"):
        build_candidate_pool_index(resolved, inventory)


def test_missing_candidate_space_sha256_in_meta_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        mutate_meta={"candidate_space_sha256": None},
    )
    with pytest.raises(ValueError, match="missing required hash field candidate_space_sha256"):
        build_candidate_pool_index(resolved, inventory)


def test_missing_model_profile_sha256_in_meta_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        mutate_meta={"model_profile_sha256": None},
    )
    with pytest.raises(ValueError, match="missing required hash field model_profile_sha256"):
        build_candidate_pool_index(resolved, inventory)


def test_duplicate_module_spec_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    specs = [_module_spec_for(t, modes[0]) for t in q_targets]
    specs.append(dict(specs[0]))
    names = [t.module_name for t in q_targets]
    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=specs,
        expected_module_names=names,
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_candidate_pool_index(resolved, inventory)


def test_shape_mismatch_across_modes_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(2)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    bad_specs = [_module_spec_for(t, modes[1]) for t in q_targets]
    bad_specs[0] = dict(bad_specs[0])
    bad_specs[0]["in_features"] = int(bad_specs[0]["in_features"]) + 1
    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[1].name
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[1],
        module_specs=bad_specs,
        expected_module_names=[t.module_name for t in q_targets],
    )
    with pytest.raises(ValueError, match="shape|in_features|mismatch"):
        build_candidate_pool_index(resolved, inventory)


def test_compact_state_hash_mismatch_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        corrupt_state_hash_in_meta=True,
    )
    with pytest.raises(ValueError, match="sha256|hash mismatch|module_state"):
        build_candidate_pool_index(resolved, inventory)


def test_compact_state_has_no_non_target_prefixes(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    names = [t.module_name for t in q_targets]
    bad_keys = [f"{names[0]}.vq_weight", "model.layers.0.k_proj.vq_weight", "embed_tokens.weight"]
    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=names,
        payload_keys=bad_keys,
    )
    with pytest.raises(ValueError, match="non-target|prefix|unexpected key|escapes"):
        build_candidate_pool_index(resolved, inventory)


def test_candidate_manifest_is_model_and_run_scoped(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(2)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    pool_root = _populate_valid_pool(resolved, inventory)

    index = build_candidate_pool_index(resolved, inventory)
    manifest_path = pool_root / "candidate_manifest.json"
    assert Path(index.manifest_path) == manifest_path
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == "toy"
    assert manifest["run_id"] == "toy_run"
    assert manifest["C"] == 2
    assert manifest["L"] == 4
    assert manifest["R"] == 2
    assert manifest["expected_trial_count"] == 4
    assert manifest["dense_module_mode_count"] == 8
    assert manifest["run_config_sha256"] == resolved.run_config_sha256
    assert manifest["model_profile_sha256"] == resolved.model_profile_sha256
    assert manifest["candidate_space_sha256"] == resolved.candidate_space_sha256
    assert manifest["training_recipe_sha256"] == resolved.training_recipe_sha256
    assert manifest["model_inventory_fingerprint"] == inventory.fingerprint_sha256
    assert len(manifest["artifacts"]) == 4
    for art in manifest["artifacts"]:
        assert "compact_state_path" in art
        assert "compact_state_sha256" in art
        assert "candidate_meta_path" in art
        assert "candidate_meta_sha256" in art


def test_protected_or_sparse_module_spec_is_rejected(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = [t for t in inventory.targets if t.category == "q_proj"]
    specs = [_module_spec_for(t, modes[0]) for t in q_targets]
    specs[0] = dict(specs[0])
    specs[0]["sparse_residual_values"] = {"shape": [1], "dtype": "float16"}
    trial_root = Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / modes[0].name
    _write_artifact(
        trial_root,
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=specs,
        expected_module_names=[t.module_name for t in q_targets],
    )
    with pytest.raises(ValueError, match="protected/sparse"):
        build_candidate_pool_index(resolved, inventory)


def test_extract_module_state_strips_prefix_and_rejects_empty():
    from mix_bit.checkpoint_pool import extract_module_state

    state = {
        "model.layers.0.q_proj.vq_weight": torch.zeros(2, dtype=torch.uint8),
        "model.layers.0.q_proj.bias": torch.zeros(2),
        "model.layers.0.k_proj.vq_weight": torch.ones(2, dtype=torch.uint8),
    }
    local = extract_module_state(state, "model.layers.0.q_proj")
    assert set(local) == {"vq_weight", "bias"}
    assert "model.layers.0.k_proj.vq_weight" not in local
    with pytest.raises(ValueError, match="empty|no keys"):
        extract_module_state(state, "model.layers.9.q_proj")


def test_load_compact_state_mmap_reads_source(tmp_path: Path):
    from mix_bit.checkpoint_pool import CheckpointSource, load_compact_state_mmap

    state = {"model.layers.0.q_proj.vq_weight": torch.arange(4, dtype=torch.uint8)}
    path = tmp_path / "module_state.pt"
    torch.save(state, path)
    source = CheckpointSource(
        category="q_proj",
        module_suffix="q_proj",
        mode_name="b16d4s2",
        trial_root=str(tmp_path),
        candidate_meta_path=str(tmp_path / "candidate_meta.json"),
        compact_state_path=str(path),
        candidate_meta_sha256="a" * 64,
        compact_state_sha256=sha256_file(path),
    )
    loaded = load_compact_state_mmap(source)
    assert torch.equal(loaded["model.layers.0.q_proj.vq_weight"], state["model.layers.0.q_proj.vq_weight"])


def _q_proj_trial_root(resolved: ResolvedRunConfig, mode: CandidateMode) -> Path:
    return Path(resolved.canonical_run_root) / "candidate_pool" / "q_proj" / mode.name


def _q_targets(inventory: ModelInventory) -> list[TargetLinearSpec]:
    return [t for t in inventory.targets if t.category == "q_proj"]


def test_pool_rejects_same_mode_name_with_wrong_nominal_bit(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = _q_targets(inventory)
    bad_mode = dict(
        name=modes[0].name,
        nominal_bit=float(modes[0].nominal_bit) + 1.0,
        codebook_bits=modes[0].codebook_bits,
        codebook_dim=modes[0].codebook_dim,
        residual_stages=modes[0].residual_stages,
    )
    _write_artifact(
        _q_proj_trial_root(resolved, modes[0]),
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        mutate_meta={"mode": bad_mode},
    )
    with pytest.raises(ValueError, match="nominal_bit"):
        build_candidate_pool_index(resolved, inventory)


def test_pool_rejects_same_mode_name_with_wrong_codebook_bits(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = _q_targets(inventory)
    bad_mode = dict(
        name=modes[0].name,
        nominal_bit=modes[0].nominal_bit,
        codebook_bits=int(modes[0].codebook_bits) + 8,
        codebook_dim=modes[0].codebook_dim,
        residual_stages=modes[0].residual_stages,
    )
    _write_artifact(
        _q_proj_trial_root(resolved, modes[0]),
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        mutate_meta={"mode": bad_mode},
    )
    with pytest.raises(ValueError, match="codebook_bits"):
        build_candidate_pool_index(resolved, inventory)


def test_pool_rejects_same_mode_name_with_wrong_codebook_dim(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = _q_targets(inventory)
    bad_mode = dict(
        name=modes[0].name,
        nominal_bit=modes[0].nominal_bit,
        codebook_bits=modes[0].codebook_bits,
        codebook_dim=int(modes[0].codebook_dim) + 4,
        residual_stages=modes[0].residual_stages,
    )
    _write_artifact(
        _q_proj_trial_root(resolved, modes[0]),
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        mutate_meta={"mode": bad_mode},
    )
    with pytest.raises(ValueError, match="codebook_dim"):
        build_candidate_pool_index(resolved, inventory)


def test_pool_rejects_same_mode_name_with_wrong_residual_stages(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = _q_targets(inventory)
    bad_mode = dict(
        name=modes[0].name,
        nominal_bit=modes[0].nominal_bit,
        codebook_bits=modes[0].codebook_bits,
        codebook_dim=modes[0].codebook_dim,
        residual_stages=int(modes[0].residual_stages) + 1,
    )
    _write_artifact(
        _q_proj_trial_root(resolved, modes[0]),
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=[_module_spec_for(t, modes[0]) for t in q_targets],
        expected_module_names=[t.module_name for t in q_targets],
        mutate_meta={"mode": bad_mode},
    )
    with pytest.raises(ValueError, match="residual_stages"):
        build_candidate_pool_index(resolved, inventory)


def test_pool_rejects_mislabeled_s2_artifact_with_s1_module_spec(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = _q_targets(inventory)
    # Artifact is labeled s2 (residual_stages=2) but the module spec describes an s1
    # single-stage module. The old name-only check accepted this; the contract must reject it.
    s1_specs = [_module_spec_for(t) for t in q_targets]
    for spec in s1_specs:
        spec.update(
            {
                "residual_stages": 1,
                "codebook_dim": int(modes[0].codebook_dim),
                "stage_codebook_dims": [int(modes[0].codebook_dim)],
                "parallel_parts": 1,
                "vq_weights": [_packed_vq_spec(int(modes[0].codebook_bits))],
                "decoders": [_decoder_spec(int(modes[0].codebook_bits), int(modes[0].codebook_dim))],
                "stage_vq_weights": None,
                "stage_decoders": None,
            }
        )
    _write_artifact(
        _q_proj_trial_root(resolved, modes[0]),
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=s1_specs,
        expected_module_names=[t.module_name for t in q_targets],
    )
    with pytest.raises(ValueError, match="residual_stages"):
        build_candidate_pool_index(resolved, inventory)


def test_pool_rejects_mislabeled_artifact_with_wrong_vq_logical_bits(tmp_path: Path):
    from mix_bit.checkpoint_pool import build_candidate_pool_index

    profile = _toy_profile()
    modes = _toy_modes(1)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    _populate_valid_pool(resolved, inventory)

    q_targets = _q_targets(inventory)
    # Mode claims codebook_bits=16 but the module spec packs 8 logical bits per codebook slot.
    wrong_bits = int(modes[0].codebook_bits) // 2
    specs = [_module_spec_for(t, modes[0]) for t in q_targets]
    for spec in specs:
        spec["stage_vq_weights"] = [_packed_vq_spec(wrong_bits) for _ in range(int(modes[0].residual_stages))]
        spec["vq_weights"] = [_packed_vq_spec(wrong_bits)]
    _write_artifact(
        _q_proj_trial_root(resolved, modes[0]),
        resolved=resolved,
        inventory=inventory,
        category_name="q_proj",
        mode=modes[0],
        module_specs=specs,
        expected_module_names=[t.module_name for t in q_targets],
    )
    with pytest.raises(ValueError, match="logical_shape"):
        build_candidate_pool_index(resolved, inventory)
