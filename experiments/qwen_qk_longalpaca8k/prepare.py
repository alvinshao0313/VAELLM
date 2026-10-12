"""CPU preparation only; formal preparation requires the completed QK8K proof."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import SOURCE, SOURCE_ID, require
from experiments.qwen_qk_longalpaca8k.config import CONTROLS, OUTPUTS, ROOT, configuration, read, training_arguments


def prepare(mode, output, resource_run):
    import torch
    output, resource_run = Path(output).resolve(), Path(resource_run).resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "Hide CUDA for CPU-only preparation.")
    require(not output.exists(), "Never overwrite a prepared or started experiment.")
    require(os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1") == "1",
            "No inherited gradient-accumulation override is permitted.")
    controls = {key: read(path / "launch.json") for key, path in CONTROLS.items()}
    old_proof = read(CONTROLS["resource"] / "verification.json")
    require(controls["resource"]["status"] == old_proof["status"] == "verified"
            and controls["resource"]["dispatcher_exit_code"] == 0
            and old_proof["native_vs_independent_tensors_compared"] == 3423,
            "Require the completed native OV8K resource and saved-state evidence.")
    recovery = read(CONTROLS["formal"] / "evaluation_recovery/terminal.json")
    require(controls["formal"]["training_exit_code"] == 0 and recovery["status"] == "completed"
            and recovery["exit_code"] == 0, "The historical OV training and offline full8 recovery must be complete.")
    sources = dict(controls["formal"]["source_sha256"])
    inputs = dict(controls["formal"]["input_evidence_sha256"])
    check_hashes(sources)
    check_hashes(inputs)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        ast.parse(path.read_text())
        sources[str(path.resolve())] = sha256(path)
    for relative in ("train_utils/runtime_snapshot.py", "experiments/long_context_recovery/dispatch.py",
                     "experiments/long_context_recovery/dispatch_400.py", "experiments/long_context_recovery/verify.py"):
        path = ROOT / relative
        sources[str(path)] = sha256(path)
    for name, path in CONTROLS.items():
        trained = read(path / "training.json")
        snapshot = Path(trained["training_result"]["run_output_dir"]) / "normalized_e2e_runtime_args.json"
        for evidence in (path / "launch.json", path / "training.json", snapshot):
            inputs[str(evidence)] = sha256(evidence)
    for relative in ("exposure.json", "evaluation_recovery/terminal.json", "evaluation_recovery/summary.json"):
        path = CONTROLS["formal"] / relative
        inputs[str(path)] = sha256(path)
    exposure = read(CONTROLS["formal"] / "exposure.json")["batches"]
    require(len(exposure) == 400 and all(row["response_non_special_tokens_shifted"] > 0 for row in exposure),
            "The original full400 exposure must retain real assistant supervision.")
    argv = training_arguments(mode, output)
    cfg, hf, args, names, architecture, inventory, comparison = configuration(mode, output, argv)
    formal_argv = training_arguments("formal", OUTPUTS["formal"])
    formal = configuration("formal", OUTPUTS["formal"], formal_argv)
    require(formal[5] == inventory, "Resource and formal runs must select the identical QK decoder inventory.")
    for item in controls["formal"]["hf_weight_stats"]:
        check_stat(item)
    resource = {"path": str(resource_run), "status": "not_run", "verification_sha256": None}
    if mode == "formal":
        bound, proof = read(resource_run / "launch.json"), read(resource_run / "verification.json")
        require(bound["mode"] == "resource" and bound["status"] == proof["status"] == "verified"
                and bound["dispatcher_exit_code"] == 0 and bound["expected_inventory"] == inventory
                and bound["selected_targets"] == list(names) and proof["source_checkpoint_id"] == SOURCE_ID
                and proof["native_vs_independent_tensors_compared"] == 3423,
                "Formal preparation requires actual successful QK8K updates/export/freeze verification.")
        check_hashes(bound["source_sha256"])
        check_hashes(bound["input_evidence_sha256"])
        require(len(proof["batches"]) == 4 and all(row["shape"] == [1, 8192]
                and row["response_non_special_tokens_shifted"] > 0 for row in proof["batches"]),
                "The QK proof must cover four real fully occupied8K sequences.")
        for relative in ("launch.json", "verification.json", "training.json", "step_0004/audit.json"):
            path = resource_run / relative
            inputs[str(path)] = sha256(path)
        resource.update(status="verified", verification_sha256=sha256(resource_run / "verification.json"))
    check_hashes(sources)
    check_hashes(inputs)
    require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA.")
    dispatcher = "experiments.long_context_recovery.dispatch" + ("_400" if mode == "formal" else "")
    record = {"schema": "qwen_qk_longalpaca8k_v1", "mode": mode, "status": "prepared_not_scheduled",
        "gpu": None, "queued": False, "automatic_retry": False, "output_dir": str(output),
        "cwd": str(ROOT), "python_executable": sys.executable,
        "source_checkpoint": str(SOURCE), "source_checkpoint_id": SOURCE_ID,
        "control_launch": str(CONTROLS[mode] / "launch.json"), "training_argv": argv,
        "argv": [sys.executable, "-u", "-m", "experiments.qwen_qk_longalpaca8k.train",
                 "--launch_record", str(output / "launch.json")],
        "dispatch_argv": [sys.executable, "-u", "-m", dispatcher, "--launch_record", str(output / "launch.json")],
        "source_sha256": sources, "input_evidence_sha256": inputs,
        "hf_weight_stats": deepcopy(controls["formal"]["hf_weight_stats"]),
        "source_weight_stat": file_stat(SOURCE / "pytorch_model.bin"),
        "selected_targets": list(names), "expected_inventory": inventory, "architecture": architecture,
        "data_registration": cfg.data.data_registration, "actual_control_comparison": comparison,
        "resource_validation": resource, "schedule": {"max_steps": 2000, "warmup_steps": 20,
            "save_steps": args.save_steps, "save_total_limit": args.save_total_limit,
            "stop_step": 4 if mode == "resource" else 400},
        "environment": {"CUDA_VISIBLE_DEVICES": None, "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(ROOT), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "ACCELERATE_GRADIENT_ACCUMULATION_STEPS": "1"},
        "resource_budget_seconds": 1180 if mode == "resource" else 7200, "termination_grace_seconds": 20,
        "training_timeout_seconds": 600, "export_timeout_seconds": 180,
        "audit_timeout_seconds": 240, "verify_timeout_seconds": 120,
        "formal400_plan": {"status": "cpu_configuration_verified_resource_proof_required",
            "output_dir": str(OUTPUTS["formal"]), "training_argv": formal_argv,
            "actual_control_comparison": formal[6], "reuse_short_state": False,
            "actual400_input_labels_order": "Require exact comparison to historical OV exposure at natural training completion.",
            "evaluation": "Independent native export/audit, full64-leaf24742 eight tasks; official EN/ZH400 scheduled separately by root."},
        "budget_contract": {"source_user_linear_bits": 17065672704, "original_linear_weights": 6945767424,
            "source_user_linear_bpw": 2.4569887907608696, "expected_added_persistent_bits": 0,
            "scope": "Only existing decoder values change; native final audit must recompute all storage components."},
        "reuse_boundary": "Original OV observers/dispatchers verify supplied selected names and the equal72/432/2991 inventory; historical OV wording in errors is not a module restriction. Training itself is unchanged native runtime.",
        "current_goal_evaluation": "No quality claim from four-step validation; complete LongBench21 is required for nine-item acceptance.",
        "cuda_initialized": False}
    output.mkdir(parents=True)
    write_json(output / "launch.prepared.json", record)
    print(json.dumps({"status": record["status"], "mode": mode, "path": str(output / "launch.prepared.json"),
        "sha256": sha256(output / "launch.prepared.json"), "sources": len(sources), "inputs": len(inputs),
        "inventory": inventory, "comparison": comparison, "cuda_initialized": False}, ensure_ascii=False))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("resource", "formal"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resource_run", type=Path, default=OUTPUTS["resource"])
    args = parser.parse_args()
    prepare(args.mode, args.output or OUTPUTS[args.mode], args.resource_run)
