"""Saved-state contract for all-projection rank4 LoRA on native Qwen/Llama."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from e2e_common.full_lora import _logical_adapter_target_name
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.w25_eval.model_inventory import model_inventory
from experiments.qwen3_w25.export_e2e_selected import _read_json
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref


RANK = 4
ALPHA = 8.0


def lora_storage(shapes):
    parameters = RANK * sum(out + inp for out, inp in shapes.values())
    return {"parameters": parameters, "tensors": 2 * len(shapes), "storage_bits": parameters * 16}


def validate_source(meta, architecture=None):
    architecture = architecture or model_inventory(meta["base_model_path"])
    shapes = architecture["projection_shapes"]
    _require(meta["checkpoint_kind"] == "final_model"
             and Path(meta["base_model_path"]).resolve() == Path(architecture["base_model_path"]),
             "Require a final model of the actual local architecture.")
    _require(set(meta["compressed_targets"]) == set(shapes) and not meta["pending_dense_targets"]
             and not meta["skip_targets"], "Expected all projections of the actual architecture to be compressed.")
    _require(meta.get("lora_config") is None and not meta.get("post_norm_head_linear")
             and (meta.get("extra_meta") or {}).get("residual_lora") is None,
             "Source must have no LoRA or auxiliary branch.")
    specs = {s["name"]: s for s in meta["converted_modules"]}
    _require(len(specs) == len(meta["converted_modules"]) and set(specs) == set(shapes), "Source projection inventory differs.")
    for name, spec in specs.items():
        _require([spec["out_features"], spec["in_features"]] == list(shapes[name])
                 and spec.get("low_rank_a") is None and spec.get("low_rank_b") is None
                 and not spec.get("always_use_original") and not spec.get("has_original_weight"),
                 f"Invalid compressed source projection: {name}.")
    return architecture


def validate_configuration(meta, state, snapshot, architecture):
    shapes = architecture["projection_shapes"]
    categories = tuple(dict.fromkeys(name.rsplit(".", 1)[-1] for name in shapes))
    contract = meta["immutable_resume_contract"]
    cfg, training = snapshot["canonical_config"], snapshot["training_args"]
    _require(meta["train_mode"] == contract["train_mode"] == cfg["train_mode"] == "lora",
             "Only ordinary LoRA may be trained.")
    for key in ("norm_train_mode", "lm_head_train_mode"):
        _require(meta[key] == contract["aux"][key] == cfg["aux"][key] == "none", f"Require {key}=none.")
    _require(cfg["aux"]["residual_lora_mode"] == contract["aux"].get("residual_lora_mode", "none") == "none",
             "Residual LoRA is excluded.")
    expected_lora = {"rank": RANK, "alpha": ALPHA, "dropout": 0.0,
                     "rank_pattern": {}, "target_modules": sorted(shapes)}
    _require(meta["lora_config"] == contract["lora"] == expected_lora, "Wrong exact adapter topology.")
    for key in ("rank", "alpha", "dropout"):
        _require(cfg["lora"][key] == expected_lora[key], f"Snapshot LoRA {key} differs.")
    _require(meta["target_layers"] == contract["target_layers"] == list(range(architecture["num_hidden_layers"])), "Require every actual layer.")
    _require(meta["target_modules"] == contract["target_modules"]
             and set(meta["target_modules"]) == set(categories), "Require all seven categories.")
    _require(cfg["target_layers"] == "all" or cfg["target_layers"] == meta["target_layers"], "Layer selection differs.")
    _require(cfg["target_modules"] == "all" or set(cfg["target_modules"]) == set(categories), "Category selection differs.")
    _require(contract["runtime"]["parallel_mode"] == "dp" and contract["runtime"]["offload_mode"] == "none"
             and not contract["runtime"]["distill_hif4_act"] and contract["distributed"]["world_size"] == 1,
             "Require single-process DP without student offload or HiFloat4.")
    _require(contract["precision"]["bf16"] and not contract["precision"]["fp16"]
             and contract["optimization"]["distill_fp32_components"] == ["lora"],
             "Require FP32 LoRA parameters and BF16 compute/export.")
    _require(type(state["global_step"]) is int and 0 < state["global_step"] <= state["max_steps"]
             and state["max_steps"] == training["max_steps"] == contract["optimization"]["steps"] == 2000,
             "Do not rewrite the original 2000-step schedule to export.")
    _require(training["seed"] == contract["data"]["seed"] == 0, "Seed changed.")
    # Preserve runtime-resolved workers/teacher identity, checking every saved
    # mathematical field against the canonical pre-dataset snapshot.
    expected = deepcopy(contract)
    for key in ("train_mode", "data", "loss", "runtime"):
        expected[key] = deepcopy(cfg[key])
    expected["input_checkpoint_id"] = meta["round_base_checkpoint_id"]
    expected["optimization"] = {k: v for k, v in cfg["opt"].items() if k != "logging_steps"}
    expected["aux"] = {k: v for k, v in cfg["aux"].items() if not k.startswith("residual_lora_")}
    expected["precision"] = {k: training[k] for k in contract["precision"]}
    expected["sparse_bit"] = {k: cfg["bit_" + k] for k in ("active_ratio", "optimizer", "weight_decay", "round_steps")}
    expected["sparse_bit"]["bit_lr"] = cfg["bit_lr"]
    for key in contract["evaluation_execution"]:
        value = cfg["runtime"]["evaluation"][key] if key == "eval_after_save" else training[key]
        if key == "eval_strategy":
            value = str(IntervalStrategy(value)).lower()
        elif key == "save_strategy":
            value = str(SaveStrategy(value)).lower()
        expected["evaluation_execution"][key] = value
    validate_e2e_immutable_resume_contract(contract, expected)
    return contract


def inspect_step(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    meta = load_v6_training_step_meta(str(checkpoint))
    source, base_meta = resolve_training_step_round_base_ref(str(checkpoint), meta)
    architecture = validate_source(base_meta)
    _require(checkpoint.parent.name == "trainer_state" and not (checkpoint / "sparse_bit_tuning").exists(),
             "Select an original LoRA-only training step without a SparseBit sidecar.")
    state = _read_json(checkpoint / "trainer_state.json")
    snapshot = _read_json(checkpoint.parent.parent / "normalized_e2e_runtime_args.json")
    validate_configuration(meta, state, snapshot, architecture)
    _require(meta["base_model_path"] == base_meta["base_model_path"]
             and meta["round_base_checkpoint_id"] == base_meta["checkpoint_id"]
             and set(meta["compressed_targets"]) == set(architecture["projection_shapes"])
             and not meta["pending_dense_targets"] and not meta["skip_targets"], "Step/source identity differs.")
    return meta, state, snapshot, Path(source), base_meta, architecture


def project_lora_state(mutable, manifest, shapes):
    """Mirror native finalizer: low_rank_b=A; low_rank_a=(B*alpha/r), then BF16."""
    _require({row["name"]: row["component_class"] for row in manifest}
             == {key: "lora" for key in mutable} and len(manifest) == len(mutable) == 2 * len(shapes),
             "Selected mutable state must contain only two LoRA factors per target.")
    pairs = {}
    for key, value in mutable.items():
        suffixes = {".lora_A.default.weight": "A", ".lora_B.default.weight": "B"}
        suffix = next((suffix for suffix in suffixes if key.endswith(suffix)), None)
        _require(suffix is not None, f"Unexpected LoRA state key: {key}.")
        name = _logical_adapter_target_name(key[:-len(suffix)])
        side = suffixes[suffix]
        _require(name in shapes and side not in pairs.setdefault(name, {})
                 and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                 f"Invalid selected FP32 factor: {key}.")
        out, inp = shapes[name]
        _require(tuple(value.shape) == ((RANK, inp) if side == "A" else (out, RANK)), f"Wrong factor shape: {key}.")
        pairs[name][side] = value
    _require(set(pairs) == set(shapes), "Missing LoRA target.")
    return {name + "." + key: value.contiguous() for name, pair in pairs.items() for key, value in (
        ("low_rank_b", pair["A"].to(torch.bfloat16)),
        ("low_rank_a", pair["B"].float().mul(ALPHA / RANK).to(torch.bfloat16)))}


def audit_lora_state(original, actual, mutable, manifest, shapes):
    projected = project_lora_state(mutable, manifest, shapes)
    _require(not set(original) & set(projected) and set(actual) == set(original) | set(projected),
             "Deployment must add only finalized LoRA factors.")
    frozen = _assert_state_equal({key: actual[key] for key in original}, original, "Every original tensor remains frozen")
    _assert_state_equal({key: actual[key] for key in projected}, projected, "Selected FP32 LoRA to native BF16 factors")
    _require(all(actual[key].untyped_storage().nbytes() == actual[key].numel() * 2 for key in projected),
             "LoRA storage has excess backing bytes or wrong precision.")
    nonzero_b = sum(int(torch.count_nonzero(value)) for key, value in mutable.items() if key.endswith(".lora_B.default.weight"))
    _require(nonzero_b > 0, "No real LoRA B update from the native zero initialization.")
    return {"frozen_tensors_compared": frozen, "original_tensors_unchanged": True,
            "lora_tensor_count": len(projected), "selected_lora_matches_native_bf16_cast_exactly": True,
            "nonzero_fp32_B_elements": nonzero_b,
            "nonzero_bf16_low_rank_a_elements": sum(int(torch.count_nonzero(v)) for k, v in projected.items() if k.endswith("low_rank_a")),
            "lora_storage_bits": sum(actual[k].untyped_storage().nbytes() * 8 for k in projected),
            "changed_code_bits": 0, "optimizer_steps_executed_by_export": 0}
