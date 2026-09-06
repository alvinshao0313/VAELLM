from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from mix_bit.candidate_space import load_candidate_space
from mix_bit.schema import (
    ALLOWED_TRAINING_RECIPE_KEYS,
    load_model_profile,
    load_training_recipe,
    resolve_run_config,
    validate_resolved_run_config_resume,
)


EXPECTED_DEFAULT_RECIPE_KEYS = {
    "recipe_id",
    "seed",
    "deterministic",
    "vae_steps",
    "vae_batch_size",
    "base_ch",
    "num_res_blocks",
    "decoder_base_ch",
    "decoder_num_res_blocks",
    "norm_type",
    "activation_type",
    "decoder_type",
    "recon_loss_type",
    "quantizer_type",
    "gamma0",
    "gamma",
    "zeta",
    "inv_temperature",
    "vae_learning_rate",
    "beta1",
    "beta2",
    "vae_weight_decay",
    "vae_optim",
    "vae_lr_scheduler_type",
    "vae_warmup_ratio",
    "l1_weight",
    "lfq_weight",
    "commitment_loss_weight",
    "entropy_loss_weight",
    "normalize_weight",
    "vae_decoder_checkpoint",
    "new_quant",
    "log_every",
    "eval_every",
    "eval_blocks",
    "channel_protect_mode",
    "channel_protect_count",
    "channel_min_per_layer",
    "after_category_mode",
    "skip_ppl_eval",
    "eval_tasks",
    "rot_llm",
    "fp16",
    "bf16",
}


def test_default_training_recipe_has_exact_current_keys() -> None:
    raw = json.loads(
        TRAINING_RECIPE_PATH.read_text(encoding="utf-8")
    )
    assert set(raw) == EXPECTED_DEFAULT_RECIPE_KEYS

    recipe = load_training_recipe(str(TRAINING_RECIPE_PATH))
    assert recipe.recipe_id == raw["recipe_id"]
    assert set(recipe.values) == (
        EXPECTED_DEFAULT_RECIPE_KEYS - {"recipe_id"}
    )
    assert set(recipe.values) <= ALLOWED_TRAINING_RECIPE_KEYS


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SPACE_PATH = REPO_ROOT / "mix_bit/configs/candidate_spaces/vae_1to3bit.json"
TRAINING_RECIPE_PATH = REPO_ROOT / "mix_bit/configs/training_recipes/vae_bsq_mse_10k.json"
RUN_CONFIG_PATH = REPO_ROOT / "mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"


def test_candidate_space_has_exact_5_modes():
    space = load_candidate_space(str(CANDIDATE_SPACE_PATH))
    assert len(space.modes) == 5


def test_baseline_is_b32d32s2_at_2bit():
    space = load_candidate_space(str(CANDIDATE_SPACE_PATH))
    assert space.baseline_mode == "b32d32s2"
    assert space.target_average_bit == 2.0
    baseline = next(mode for mode in space.modes if mode.name == space.baseline_mode)
    assert baseline.nominal_bit == 2.0
    assert baseline.residual_stages == 2


def test_each_nominal_bit_has_exactly_one_s2_mode():
    space = load_candidate_space(str(CANDIDATE_SPACE_PATH))
    expected = {1.0, 1.5, 2.0, 2.5, 3.0}
    by_bit = {}
    for mode in space.modes:
        assert mode.residual_stages == 2
        by_bit.setdefault(mode.nominal_bit, []).append(mode.name)
    assert set(by_bit) == expected
    assert all(len(names) == 1 for names in by_bit.values())


def test_confirmed_candidate_space_contains_no_s1_mode():
    space = load_candidate_space(str(CANDIDATE_SPACE_PATH))
    assert all(mode.residual_stages != 1 for mode in space.modes)
    assert all("s1" not in mode.name for mode in space.modes)


def test_duplicate_mode_name_is_rejected():
    raw = json.loads(CANDIDATE_SPACE_PATH.read_text(encoding="utf-8"))
    raw["modes"].append(copy.deepcopy(raw["modes"][0]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dup.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            load_candidate_space(str(path))


def test_training_recipe_rejects_unknown_keys():
    raw = json.loads(TRAINING_RECIPE_PATH.read_text(encoding="utf-8"))
    raw["unknown_hyperparam"] = 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_recipe.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown"):
            load_training_recipe(str(path))


def test_training_recipe_disables_distill_eval_rotation_and_full_checkpoint_save():
    recipe = load_training_recipe(str(TRAINING_RECIPE_PATH))
    assert recipe.values["after_category_mode"] == "none"
    assert recipe.values["skip_ppl_eval"] is True
    assert recipe.values["eval_tasks"] == ""
    assert recipe.values["rot_llm"] is False
    assert recipe.values["eval_every"] == 0
    assert "save_model" not in recipe.values


def test_run_config_resolves_all_relative_paths_and_hashes():
    resolved = resolve_run_config(str(RUN_CONFIG_PATH), write=False)
    expected_calib = (REPO_ROOT / "data/edgerazor_qwen3/task_vaellm_eval_instruct.jsonl").resolve()
    assert Path(resolved.run_config_path).is_absolute()
    assert Path(resolved.model_profile_path).is_absolute()
    assert Path(resolved.candidate_space_path).is_absolute()
    assert Path(resolved.training_recipe_path).is_absolute()
    assert Path(resolved.config.calibration.source_jsonl) == expected_calib
    assert Path(resolved.config.calibration.source_jsonl).exists()
    assert len(resolved.run_config_sha256) == 64
    assert len(resolved.model_profile_sha256) == 64
    assert len(resolved.candidate_space_sha256) == 64
    assert len(resolved.training_recipe_sha256) == 64
    assert resolved.canonical_model_root.endswith("qwen3_8b")
    assert resolved.canonical_run_root.endswith("qwen3_8b_vae_1to3bit")


def test_resume_rejects_changed_referenced_config_hash():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        recipe_src = json.loads(TRAINING_RECIPE_PATH.read_text(encoding="utf-8"))
        profile_src = json.loads(
            (REPO_ROOT / "mix_bit/configs/models/qwen3_8b.json").read_text(encoding="utf-8")
        )
        space_src = json.loads(CANDIDATE_SPACE_PATH.read_text(encoding="utf-8"))

        models_dir = tmp_root / "models"
        spaces_dir = tmp_root / "candidate_spaces"
        recipes_dir = tmp_root / "training_recipes"
        runs_dir = tmp_root / "runs"
        data_dir = tmp_root / "data"
        for directory in (models_dir, spaces_dir, recipes_dir, runs_dir, data_dir):
            directory.mkdir(parents=True)

        calib = data_dir / "calib.jsonl"
        calib.write_text("{}\n", encoding="utf-8")
        profile_path = models_dir / "model.json"
        space_path = spaces_dir / "space.json"
        recipe_path = recipes_dir / "recipe.json"
        profile_path.write_text(json.dumps(profile_src), encoding="utf-8")
        space_path.write_text(json.dumps(space_src), encoding="utf-8")
        recipe_path.write_text(json.dumps(recipe_src), encoding="utf-8")

        run_cfg = {
            "run_id": "tmp_run",
            "model_profile": "../models/model.json",
            "candidate_space": "../candidate_spaces/space.json",
            "training_recipe": "../training_recipes/recipe.json",
            "calibration": {
                "source_jsonl": str(calib.resolve()),
                "input_format": "auto",
                "max_samples": 1,
                "max_length": 32,
                "seed": 1,
                "label_mode": "all_nonpad",
            },
        }
        run_path = runs_dir / "run.json"
        run_path.write_text(json.dumps(run_cfg), encoding="utf-8")

        first = resolve_run_config(
            str(run_path),
            repo_root=str(tmp_root),
            result_root=str(tmp_root / ".result"),
            write=True,
        )
        validate_resolved_run_config_resume(first)

        recipe_src["vae_learning_rate"] = 0.001
        recipe_path.write_text(json.dumps(recipe_src), encoding="utf-8")
        second = resolve_run_config(
            str(run_path),
            repo_root=str(tmp_root),
            result_root=str(tmp_root / ".result"),
            write=False,
        )
        with pytest.raises(ValueError, match="hash"):
            validate_resolved_run_config_resume(second)


def test_model_profile_rejects_duplicate_logical_categories():
    raw = json.loads((REPO_ROOT / "mix_bit/configs/models/qwen3_8b.json").read_text(encoding="utf-8"))
    raw["categories"].append(copy.deepcopy(raw["categories"][0]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dup_cat.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            load_model_profile(str(path))


def test_model_profile_rejects_ambiguous_suffix_mapping():
    raw = json.loads((REPO_ROOT / "mix_bit/configs/models/qwen3_8b.json").read_text(encoding="utf-8"))
    raw["categories"][1]["module_suffix"] = raw["categories"][0]["module_suffix"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ambig.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="ambiguous|suffix"):
            load_model_profile(str(path))


def test_model_profile_does_not_require_qwen_category_names():
    raw = {
        "model_id": "alt",
        "model_path": "org/Alt",
        "adapter": "generic_decoder",
        "only_decoder_projections": True,
        "candidate_training": {"linear_group_size": "all", "allow_tail_group": True},
        "layer_index_patterns": [r"(?:^|\.)blocks\.(\d+)\."],
        "categories": [
            {"name": "attn_q", "module_suffix": "wq", "transpose": True},
            {"name": "attn_k", "module_suffix": "wk", "transpose": False},
            {"name": "attn_v", "module_suffix": "wv", "transpose": True},
            {"name": "attn_o", "module_suffix": "wo", "transpose": True},
            {"name": "ffn_up", "module_suffix": "w1", "transpose": False},
            {"name": "ffn_gate", "module_suffix": "w3", "transpose": False},
            {"name": "ffn_down", "module_suffix": "w2", "transpose": True},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "alt.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        profile = load_model_profile(str(path))
    assert [c.name for c in profile.categories] == [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "ffn_up",
        "ffn_gate",
        "ffn_down",
    ]
    assert [c.module_suffix for c in profile.categories] == [
        "wq",
        "wk",
        "wv",
        "wo",
        "w1",
        "w3",
        "w2",
    ]
