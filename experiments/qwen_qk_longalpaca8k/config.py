"""Bind QK decoder recovery to the two completed OV8K scientific configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compressed_e2e_fintuning.args import parse_args
from experiments.long_context_recovery.contract import SOURCE, SOURCE_ID, require
from experiments.long_context_recovery.data import registered_config
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result/qwen3_8b_mmlu65_20261004"
CONTROLS = {
    "resource": RESULT / "best1200_ov72_longalpaca8k_validate_s4",
    "formal": RESULT / "best1200_ov72_longalpaca8k_s400",
}
OUTPUTS = {
    "resource": RESULT / "qwen_qk_longalpaca8k_validate_s4",
    "formal": RESULT / "qwen_qk_longalpaca8k_s400",
}


def read(path):
    return json.loads(Path(path).read_text())


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(set(left) | set(right)):
            name = prefix + "." + key if prefix else key
            result.update(differences(left.get(key), right.get(key), name))
        return result
    return {} if left == right else {prefix: {"control": left, "candidate": right}}


def training_arguments(mode, output):
    argv = list(read(CONTROLS[mode] / "launch.json")["training_argv"])
    for flag, value in (("--target_modules", "q_proj,k_proj"),
                        ("--run_root_dir", str(Path(output).resolve()))):
        require(argv.count(flag) == 1, "Ambiguous original option: " + flag)
        argv[argv.index(flag) + 1] = value
    return argv


def configuration(mode, output, argv):
    require(mode in CONTROLS and argv == training_arguments(mode, output),
            "Only target_modules OV to QK and the output path may change.")
    local = argparse.ArgumentParser(add_help=False)
    local.add_argument("--long_context_manifest", required=True)
    options, remaining = local.parse_known_args(argv)
    cfg, hf, args = parse_args(remaining)
    cfg.data = registered_config(cfg.data, options.long_context_manifest)
    control_training = read(CONTROLS[mode] / "training.json")
    require(control_training["status"] == "trained", "The original native training must have succeeded.")
    snapshot = Path(control_training["training_result"]["run_output_dir"]) / "normalized_e2e_runtime_args.json"
    actual = read(snapshot)
    cfg_diff = differences(actual["canonical_config"], to_runtime_jsonable(cfg))
    require(set(cfg_diff) == {"target_modules", "run_root_dir"}
            and cfg_diff["target_modules"] == {"control": ["v_proj", "o_proj"],
                                              "candidate": ["q_proj", "k_proj"]},
            "Scientific configuration differs beyond OV to QK: " + json.dumps(cfg_diff))
    hf_diff = differences(actual["hf_args"], to_runtime_jsonable(hf))
    args_diff = differences(actual["training_args"], to_runtime_jsonable(args))
    require(not hf_diff and set(args_diff) <= {"_n_gpu", "logging_dir", "output_dir"},
            "Original HF/Trainer parameters changed: " + json.dumps([hf_diff, args_diff]))
    stop = 4 if mode == "resource" else 400
    require(Path(cfg.student_checkpoint_dir).resolve() == SOURCE and cfg.train_mode == "decoder"
            and cfg.target_layers == "all" and tuple(cfg.target_modules) == ("q_proj", "k_proj")
            and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none"
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            "Start fresh f927, with only QK decoder updates and no additional branches.")
    require(args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
            and args.save_steps == (2 if stop == 4 else 400)
            and args.save_total_limit == (2 if stop == 4 else 1)
            and args.per_device_train_batch_size == args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 8192 and args.bf16 and not args.fp16,
            "Keep native 2000-step cosine prefix, B1/8K/BF16 and original save boundaries.")
    registration = cfg.data.data_registration
    require(registration["alias"] == "longalpaca"
            and registration["purpose"] == ("resource_diagnostic" if stop == 4 else "full_dataset")
            and registration["rows"] == (4 if stop == 4 else 12000),
            "Use the original four real8K resource samples or unchanged full LongAlpaca stream.")
    meta = read(SOURCE / "checkpoint_meta.json")
    require(meta["checkpoint_id"] == SOURCE_ID, "The real source UUID changed.")
    architecture = model_inventory(meta["base_model_path"])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    inventory = expected_inventory(SOURCE, architecture, names)
    require(inventory["selected_projections"] == 72 and inventory["selected_decoder_tensors"] == 432
            and inventory["frozen_tensors"] == 2991 and inventory["all_code_banks"] == 504
            and inventory["serialized_tensors"] == 3423 and inventory["active_score_elements"] == 0,
            "Actual QK72/432 decoder/2991 frozen/504-bank inventory changed.")
    comparison = {"actual_control_snapshot": str(snapshot), "canonical_differences": cfg_diff,
                  "hf_differences": hf_diff, "trainer_execution_differences": args_diff}
    return cfg, hf, args, names, architecture, inventory, comparison
