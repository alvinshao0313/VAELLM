"""Only the new saved-state, frozen-base and callback boundaries need coverage."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.w25_eval.lora_state import (audit_lora_state, project_lora_state, lora_storage,
    validate_source, lora_selection, saved_lora_selection, validate_configuration)
from experiments.w25_eval.lora_step_callback import stage_commands


def example(rank=4):
    name = "model.layers.0.mlp.down_proj"
    prefix = f"base_model.model.{name}.full_adapter_linear"
    values = {prefix + ".lora_A.default.weight": torch.arange(rank * 5, dtype=torch.float32).reshape(rank, 5) / 13,
              prefix + ".lora_B.default.weight": torch.arange(3 * rank, dtype=torch.float32).reshape(3, rank) / 11}
    manifest = [{"name": key, "component_class": "lora"} for key in values]
    return name, values, manifest, {name: (3, 5)}


@pytest.mark.parametrize("rank,alpha", [(4, 8), (8, 16)])
def test_native_projection_and_entire_frozen_complement(rank, alpha):
    name, values, manifest, shapes = example(rank)
    projected = project_lora_state(values, manifest, shapes, rank=rank, alpha=alpha)
    torch.testing.assert_close(projected[name + ".low_rank_a"], (next(v for k, v in values.items() if ".lora_B." in k) * (alpha / rank)).bfloat16(), rtol=0, atol=0)
    original = {"frozen_decoder": torch.tensor([1, -0.0], dtype=torch.bfloat16),
                "frozen_codes": torch.tensor([0, 255], dtype=torch.uint8),
                "frozen_head": torch.ones(2, dtype=torch.bfloat16),
                "model.layers.0.mlp.up_proj.frozen_decoder": torch.ones(3, dtype=torch.bfloat16)}
    actual = {**{k: v.clone() for k, v in original.items()}, **projected}
    report = audit_lora_state(original, actual, values, manifest, shapes, rank=rank, alpha=alpha)
    assert report["frozen_tensors_compared"] == 4
    assert report["lora_storage_bits"] == (3 + 5) * rank * 16
    actual["frozen_codes"][0] = 1
    with pytest.raises(ValueError, match="tensor bytes differ"):
        audit_lora_state(original, actual, values, manifest, shapes, rank=rank, alpha=alpha)
    actual["frozen_codes"][0] = 0
    actual["model.layers.0.mlp.up_proj.frozen_decoder"][0] = 2
    with pytest.raises(ValueError, match="tensor bytes differ"):
        audit_lora_state(original, actual, values, manifest, shapes, rank=rank, alpha=alpha)


def test_reject_wrong_mutable_dtype_scope_shape_and_no_real_update():
    name, values, manifest, shapes = example()
    key = next(iter(values))
    invalid = dict(values); invalid[key] = invalid[key].bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        project_lora_state(invalid, manifest, shapes)
    invalid_manifest = deepcopy(manifest); invalid_manifest[0]["component_class"] = "decoder"
    with pytest.raises(ValueError, match="only two LoRA"):
        project_lora_state(values, invalid_manifest, shapes)
    invalid = dict(values); invalid[key] = invalid[key][:3]
    with pytest.raises(ValueError, match="shape"):
        project_lora_state(invalid, manifest, shapes)
    with pytest.raises(ValueError, match="shape"):
        project_lora_state(values, manifest, shapes, rank=8, alpha=16)
    missing = {key: value for key, value in values.items() if ".lora_A." in key}
    with pytest.raises(ValueError, match="only two LoRA"):
        project_lora_state(missing, [{"name": key, "component_class": "lora"} for key in missing], shapes)
    extra = {**values, "base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight": torch.ones(4, 5)}
    with pytest.raises(ValueError, match="only two LoRA"):
        project_lora_state(extra, [{"name": key, "component_class": "lora"} for key in extra], shapes)
    for key in values:
        if ".lora_B." in key:
            values[key].zero_()
    projected = project_lora_state(values, manifest, shapes)
    with pytest.raises(ValueError, match="No real LoRA B update"):
        audit_lora_state({}, projected, values, manifest, shapes)


def test_rank4_budget_and_fixed_eval_commands():
    commands = stage_commands(checkpoint="step", source_dir="source", output_dir="out")
    assert [x[0] for x in commands] == ["export", "audit", "mmlu", "seven", "summary"]
    assert commands[0][1][2] == "experiments.w25_eval.lora_export"
    assert commands[1][1][-2:] == ["--training_checkpoint_dir", "step"]
    for item, batch in zip(commands[2:4], ("64", "32")):
        command = item[1]
        assert command[command.index("--lm_batch_size") + 1] == batch and "--lm_limit" not in command
    assert commands[-1][1][2] == "experiments.w25_eval.summarize_lora_results"
    limited = stage_commands(checkpoint="step", source_dir="source", output_dir="out", diagnostic_eval_limit=64)
    assert len(limited) == 4
    assert all(command[-2:] == ["--lm_limit", "64"] for _, command, _ in limited[2:])


@pytest.mark.parametrize("metadata,projections,layers,parameters", [
    ("result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model/checkpoint_meta.json", 252, 36, 10911744),
    ("result/llama31_8b_w25_20261004/validation_remaining_prefix/native_runs/home_star_models_meta-llama_Llama-3.1-8B-Instruct_20261004_102615/final_model/checkpoint_meta.json", 224, 32, 10485760),
])
def test_real_local_metadata_geometry_without_model_weights(metadata, projections, layers, parameters):
    meta = json.loads(Path(metadata).read_text())
    architecture = validate_source(meta)
    assert architecture["projection_count"] == projections
    assert architecture["num_hidden_layers"] == layers
    assert architecture["weights_loaded_for_inventory"] is False
    storage = lora_storage(architecture["projection_shapes"])
    assert storage == {"parameters": parameters, "tensors": 2 * projections, "storage_bits": parameters * 16}
    if architecture["model_type"] == "qwen3":
        assert (17065672704 + storage["storage_bits"]) / architecture["original_linear_weights"] == 2.482124660326087
    selected, exact = lora_selection(architecture, "all", "gate_proj,down_proj", rank=8, alpha=16, dropout=0)
    assert len(selected) == 2 * layers
    selected_storage = lora_storage(selected, 8)
    assert selected_storage["tensors"] == 4 * layers and selected_storage["parameters"] == 9437184
    if architecture["model_type"] == "qwen3":
        assert (17065672704 + selected_storage["storage_bits"]) / architecture["original_linear_weights"] == 2.4787279211956523
    # Actual parser + native contract builder, in memory only: no training state
    # or model weights are manufactured by this CPU configuration check.
    from compressed_e2e_fintuning.args import parse_args
    from compressed_e2e_fintuning.v6_runtime_state import _jsonable, build_e2e_immutable_resume_contract
    from experiments.w25_eval.subset_state import selected_layers, selected_modules
    from transformers import AutoTokenizer
    argv = json.loads(Path("result/qwen3_8b_mmlu65_20261004/best1200_all252_lora4_validate_s4/launch.json").read_text())["training_argv"]
    for flag, value in {"--student_checkpoint_dir": str(Path(metadata).parent),
            "--teacher_model_path": meta["base_model_path"], "--target_modules": "gate_proj,down_proj",
            "--lora_rank": "8", "--lora_alpha": "16"}.items():
        argv[argv.index(flag) + 1] = value
    cfg, _hf, args = parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(meta["base_model_path"], local_files_only=True, use_fast=True)
    contract = build_e2e_immutable_resume_contract(cfg=cfg, training_args=args, tokenizer=tokenizer,
        input_checkpoint_id=meta["checkpoint_id"], resolved_target_layers=selected_layers(cfg.target_layers, architecture),
        resolved_target_modules=selected_modules(cfg.target_modules, architecture), teacher_identity=None)
    contract["lora"] = exact  # Same exact-PEFT topology replacement as native runtime.
    native = {**meta, "train_mode": "lora", "norm_train_mode": "none", "lm_head_train_mode": "none",
              "lora_config": exact, "immutable_resume_contract": contract,
              "target_layers": contract["target_layers"], "target_modules": contract["target_modules"],
              "round_base_checkpoint_id": meta["checkpoint_id"]}
    snapshot = {"canonical_config": _jsonable(cfg), "training_args": _jsonable(args)}
    assert validate_configuration(native, {"global_step": 4, "max_steps": 2000}, snapshot, architecture) == contract
    assert saved_lora_selection(native, architecture) == (selected, exact)
    wrong = deepcopy(native); wrong["immutable_resume_contract"]["lora"] = {**exact, "rank": 4}
    with pytest.raises(ValueError, match="topology"):
        saved_lora_selection(wrong, architecture)
    wrong = deepcopy(native); wrong["immutable_resume_contract"]["lora"]["target_modules"].pop()
    with pytest.raises(ValueError, match="topology"):
        saved_lora_selection(wrong, architecture)
    invalid = deepcopy(meta)
    invalid["converted_modules"][0]["out_features"] += 1
    with pytest.raises(ValueError, match="Invalid compressed source projection"):
        validate_source(invalid, architecture)
    assert not torch.cuda.is_initialized()
