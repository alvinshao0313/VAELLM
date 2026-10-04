"""Only the new saved-state, frozen-base and callback boundaries need coverage."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.w25_eval.lora_state import audit_lora_state, project_lora_state, lora_storage, validate_source
from experiments.w25_eval.lora_step_callback import stage_commands


def example():
    name = "model.layers.0.mlp.down_proj"
    prefix = f"base_model.model.{name}.full_adapter_linear"
    values = {prefix + ".lora_A.default.weight": torch.arange(20, dtype=torch.float32).reshape(4, 5) / 13,
              prefix + ".lora_B.default.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11}
    manifest = [{"name": key, "component_class": "lora"} for key in values]
    return name, values, manifest, {name: (3, 5)}


def test_native_projection_and_entire_frozen_complement():
    name, values, manifest, shapes = example()
    projected = project_lora_state(values, manifest, shapes)
    torch.testing.assert_close(projected[name + ".low_rank_a"], (next(v for k, v in values.items() if ".lora_B." in k) * 2).bfloat16(), rtol=0, atol=0)
    original = {"frozen_decoder": torch.tensor([1, -0.0], dtype=torch.bfloat16),
                "frozen_codes": torch.tensor([0, 255], dtype=torch.uint8),
                "frozen_head": torch.ones(2, dtype=torch.bfloat16)}
    actual = {**{k: v.clone() for k, v in original.items()}, **projected}
    report = audit_lora_state(original, actual, values, manifest, shapes)
    assert report["frozen_tensors_compared"] == 3
    assert report["lora_storage_bits"] == (3 + 5) * 4 * 16
    actual["frozen_codes"][0] = 1
    with pytest.raises(ValueError, match="tensor bytes differ"):
        audit_lora_state(original, actual, values, manifest, shapes)


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
    invalid = deepcopy(meta)
    invalid["converted_modules"][0]["out_features"] += 1
    with pytest.raises(ValueError, match="Invalid compressed source projection"):
        validate_source(invalid, architecture)
    assert not torch.cuda.is_initialized()
