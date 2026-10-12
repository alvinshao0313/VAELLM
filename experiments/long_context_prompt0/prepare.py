"""CPU-only preparation of one matched prompt-weight change; never launch a GPU."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.long_context_recovery.contract import SOURCE, SOURCE_ID, require
from experiments.long_context_prompt0.train import CONTROL, ROOT, parse_configuration, read, training_arguments
from train_utils.runtime_snapshot import to_runtime_jsonable


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(set(left) | set(right)):
            path = prefix + "." + key if prefix else key
            if key not in left or key not in right:
                result[path] = {"control": left.get(key), "candidate": right.get(key)}
            else:
                result.update(differences(left[key], right[key], path))
        return result
    return {} if left == right else {prefix: {"control": left, "candidate": right}}


def stages(checkpoint, output):
    from experiments.qwen3_w25.subset_step_export_callback import stage_commands
    result = []
    for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=SOURCE, output_dir=output):
        if name == "summary":
            continue
        if name in ("export", "audit"):
            command[2] = "experiments.w25_eval.subset_" + name
        require(name in ("export", "audit", "mmlu", "seven") and "--lm_limit" not in command,
                "Only independent export/audit and complete fixed-batch eight-task shards are allowed.")
        result.append((name, command, allowed))
    require([x[0] for x in result] == ["export", "audit", "mmlu", "seven"], "Missing required stage.")
    return result


def prepare(output):
    from experiments.long_context_recovery.train_400 import parse_configuration as parse_control
    import torch
    output = Path(output).resolve()
    require(not (output / "launch.json").exists() and not (output / "training.json").exists(),
            "Never overwrite a prepared or started candidate.")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1") == "1",
            "Use CPU-only preparation without an inherited accumulation override.")
    control = read(CONTROL / "launch.json")
    check_hashes(control["source_sha256"])
    check_hashes(control["input_evidence_sha256"])
    for item in control["hf_weight_stats"]:
        check_stat(item)
    recovery_path = CONTROL / "evaluation_recovery/terminal.json"
    recovery = read(recovery_path)
    require(control["training_exit_code"] == 0 and read(CONTROL / "training.json")["status"] == "trained"
            and recovery["status"] == "completed" and recovery["exit_code"] == 0
            and (CONTROL / "evaluation_recovery/run.exit").read_text().strip() == "0",
            "Require the original successful training and its explicit completed evaluation recovery.")
    old_snapshot_path = Path(control["native_run_dir"]) / "normalized_e2e_runtime_args.json"
    old_snapshot = read(old_snapshot_path)
    old_cfg, old_hf, old_args, old_names, architecture, old_inventory = parse_control(control["training_argv"])
    require(to_runtime_jsonable(old_cfg) == old_snapshot["canonical_config"]
            and to_runtime_jsonable(old_hf) == old_snapshot["hf_args"],
            "The unchanged current parser must reproduce the actual control canonical/HF configuration.")
    old_hf_diff = differences(old_snapshot["training_args"], to_runtime_jsonable(old_args))
    require(set(old_hf_diff) <= {"_n_gpu", "logging_dir", "output_dir"},
            "Unexpected control Trainer difference: " + json.dumps(old_hf_diff))
    argv = training_arguments(output)
    cfg, hf, args, names, candidate_architecture, inventory = parse_configuration(argv)
    actual_cfg = to_runtime_jsonable(cfg)
    config_diff = differences(old_snapshot["canonical_config"], actual_cfg)
    require(set(config_diff) == {"loss.prompt_loss_weight", "run_root_dir"}
            and config_diff["loss.prompt_loss_weight"] == {"control": .3, "candidate": 0.0},
            "The only scientific configuration change must be prompt weight .3 to zero.")
    args_diff = differences(to_runtime_jsonable(old_args), to_runtime_jsonable(args))
    require(set(args_diff) <= {"logging_dir"} and to_runtime_jsonable(old_hf) == to_runtime_jsonable(hf)
            and old_names == names and architecture == candidate_architecture and old_inventory == inventory,
            "Trainer/HF/model inventory differs beyond the stated loss weighting.")
    exposure_path = CONTROL / "exposure.json"
    batches = read(exposure_path)["batches"]
    require(len(batches) == 400 and all(x["response_tokens_shifted"] > 0 for x in batches),
            "Every original batch must retain real response supervision at prompt weight zero.")
    resource = Path(control["resource_validation"]["path"])
    proof = read(resource / "verification.json")
    require(proof["status"] == "verified" and len(proof["batches"]) == 4
            and all(x["shape"] == [1, 8192] and x["response_non_special_tokens_shifted"] > 0
                    for x in proof["batches"]), "Reuse the existing genuine four-update 8K resource proof.")
    result_paths = [CONTROL / "step_0400/mmlu.json", CONTROL / "evaluation_recovery/seven.json"]
    require(read(result_paths[0])["checkpoint_id"] == read(result_paths[1])["checkpoint_id"]
            == recovery["checkpoint_id"], "Control evaluation recovery must refer to the same deployed model.")
    sources = dict(control["source_sha256"])
    for path in sorted(Path(__file__).parent.glob("*.py")):
        ast.parse(path.read_text())
        sources[str(path.resolve())] = sha256(path)
    require((Path(__file__).parent / "dispatch.py").is_file(), "The independently reviewed dispatcher must exist.")
    inputs = dict(control["input_evidence_sha256"])
    for path in [CONTROL / "launch.json", CONTROL / "training.json", old_snapshot_path, exposure_path,
                 CONTROL / "step_0400/audit.json", CONTROL / "step_0400/final_model/checkpoint_meta.json",
                 recovery_path, CONTROL / "evaluation_recovery/run.exit",
                 CONTROL / "evaluation_recovery/summary.json", *result_paths]:
        inputs[str(path)] = sha256(path)
    require(not torch.cuda.is_initialized(), "Preparation must never initialize CUDA.")
    record = {"schema": "long_context_ov72_prompt0_400_v1", "status": "prepared_not_scheduled",
        "gpu": None, "queued": False, "automatic_retry": False, "output_dir": str(output),
        "cwd": str(ROOT), "python_executable": sys.executable,
        "source_checkpoint": str(SOURCE), "source_checkpoint_id": SOURCE_ID,
        "control_launch": str(CONTROL / "launch.json"), "control_exposure_path": str(exposure_path),
        "control_result_paths": [str(p) for p in result_paths],
        "training_argv": argv, "argv": [sys.executable, "-u", "-m", "experiments.long_context_prompt0.train", *argv],
        "dispatch_argv": [sys.executable, "-u", "-m", "experiments.long_context_prompt0.dispatch",
                          "--launch_record", str(output / "launch.json")],
        "source_sha256": sources, "input_evidence_sha256": inputs,
        "hf_weight_stats": deepcopy(control["hf_weight_stats"]), "architecture": architecture,
        "selected_targets": list(names), "expected_inventory": inventory,
        "data_registration": cfg.data.data_registration,
        "cpu_configuration": {"canonical_config": actual_cfg, "hf_args": to_runtime_jsonable(hf),
            "training_args": to_runtime_jsonable(args), "actual_control_canonical_exact": True,
            "control_actual_vs_cpu_trainer_differences": old_hf_diff,
            "candidate_vs_control_trainer_differences": args_diff,
            "candidate_vs_actual_control_canonical_differences": config_diff},
        "environment": {"CUDA_VISIBLE_DEVICES": None, "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(ROOT), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "ACCELERATE_GRADIENT_ACCUMULATION_STEPS": "1"},
        "schedule": deepcopy(control["schedule"]),
        "resource_budget_seconds": 7200, "termination_grace_seconds": 20,
        "resource_validation": {**deepcopy(control["resource_validation"]),
            "reuse_reason": "Only token loss weights change; full logits, actual shapes, scope and numerical path remain identical.",
            "repeat_gpu_short": False},
        "estimated_total_seconds": [2400, 3000],
        "window_note": "Root must bind an actually free single A800 and a 7200+20 second window; no queue or retry.",
        "full8_stage_templates": stages(output / "NATIVE_RUN/trainer_state/checkpoint-400", output / "step_0400"),
        "current_goal_evaluation": "Complete eight-task screening only; candidate LongBench21/nine-item target remains not_evaluated.",
        "comparison_boundary": "Fresh f927; only prompt_loss_weight .3 to 0 changes the scientific recipe. Original context/labels, skip rules and data order stay intact. No claim that prompt token weights equal loss or gradient shares.",
        "environment_difference_from_control": "Existing cached datasets use three offline flags to avoid the historical network-only evaluation failure; data versions and scores do not change.",
        "control_exposure": {"batches": 400, "prompt_tokens_shifted": sum(x["prompt_tokens_shifted"] for x in batches),
            "response_tokens_shifted": sum(x["response_tokens_shifted"] for x in batches),
            "completion_requirement": "The full ordered list of input/labels hashes and token counts must match exactly."},
        "source_dependency_policy": "All 169 inherited source pins remain unchanged. Shared native runtime/export/loader files remain protected for other live consumers.",
        "cuda_initialized": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "launch.json", record)
    print(json.dumps({"status": record["status"], "launch_sha256": sha256(output / "launch.json"),
        "sources": len(sources), "inputs": len(inputs), "science_diff": config_diff,
        "inventory": inventory, "cuda_initialized": False}, ensure_ascii=False))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output)
