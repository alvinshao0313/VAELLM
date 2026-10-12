"""One native prompt0 run, independent export, and complete eight-task evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.long_context_recovery.contract import require


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record["source_sha256"])
    check_hashes(record["input_evidence_sha256"])
    for item in record["hf_weight_stats"]:
        check_stat(item)


def run(record_path):
    record_path = Path(record_path).resolve()
    record = read(record_path)
    output = Path(record["output_dir"])
    code = 1
    try:
        require(record["status"] == "bound_ready" and type(record["gpu"]) is int
                and os.environ.get("CUDA_VISIBLE_DEVICES") == str(record["gpu"])
                and Path(sys.executable).resolve() == Path(record["python_executable"]).resolve()
                and not record["queued"] and not record["automatic_retry"],
                "Use the root-bound GPU and bitvae interpreter without queue or retry.")
        require(not any(key in os.environ for key in ("FLASH_ATTENTION_DETERMINISTIC", "NVIDIA_TF32_OVERRIDE")),
                "Do not inherit numerical diagnostic overrides.")
        require(all(os.environ.get(key) == "1" for key in
                    ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
                and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
                and os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS") in (None, "1"),
                "Keep offline caches, the existing allocator and actual accumulation1.")
        require(not (output / "training.json").exists() and not (output / "step_0400").exists(),
                "Use a fresh prepared run; do not overwrite native training or evaluation.")
        check(record)

        def require_idle():
            used = subprocess.check_output(["nvidia-smi", "--id=" + str(record["gpu"]),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
            require(not used, "Assigned GPU is occupied: " + used)

        require_idle()
        record.update(status="training", worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
            started_unix=time.time(), stages={},
            actual_environment={key: os.environ.get(key) for key in record["environment"]})
        write_json(record_path, record)
        with (output / "training.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(record["argv"], cwd=record["cwd"], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            record.update(training_pid=child.pid,
                training_start_ticks=int(Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]))
            write_json(record_path, record)
            status = child.wait()
        (output / "training.exit").write_text(str(status) + "\n")
        record.update(training_exit_code=status, training_finished_unix=time.time())
        write_json(record_path, record)
        require(status == 0, "Native training failed; no automatic retry.")
        trained = read(output / "training.json")
        result = trained["training_result"]
        require(trained["status"] == "trained" and trained["prompt_loss_weight"] == 0
                and trained["actual_control_exposure_exact"] and result["global_step"] == 400
                and result["round_base_checkpoint_id"] == record["source_checkpoint_id"], "Wrong native completion.")
        actual, control_exposure = read(output / "exposure.json"), read(record["control_exposure_path"])
        require(actual["sampling"] == control_exposure["sampling"] == "unchanged_native_stream"
                and len(actual["batches"]) == 400 and actual["batches"] == control_exposure["batches"],
                "Actual input/label hashes, document order or token exposure differs from the control.")
        record["actual_control_exposure_exact"] = True
        record["exposure_sha256"] = sha256(output / "exposure.json")
        checkpoint = Path(result["run_output_dir"]) / "trainer_state/checkpoint-400"
        require_idle()
        from experiments.qwen3_w25.subset_step_export_callback import stage_commands
        from experiments.qwen3_w25.step_export_callback import run_logged_stage
        step = output / "step_0400"
        step.mkdir()
        record.update(status="postprocessing", native_run_dir=result["run_output_dir"])
        write_json(record_path, record)
        started = time.monotonic()
        commands = [row for row in stage_commands(checkpoint=checkpoint,
            source_dir=record["source_checkpoint"], output_dir=step) if row[0] != "summary"]
        require([row[0] for row in commands] == ["export", "audit", "mmlu", "seven"], "Unexpected postprocessing stages.")
        for name, command, allowed in commands:
            require(tuple(allowed) == (0,), "Each actual stage must complete successfully.")
            if name in ("export", "audit"):
                command[2] = "experiments.w25_eval.subset_" + name
            require("--limit" not in command and "--lm_limit" not in command, "Do not limit full evaluation.")
            record["stages"][name] = {"argv": command, "started_unix": time.time()}
            write_json(record_path, record)
            status = run_logged_stage(command, name=name, output_dir=step,
                                      env=os.environ.copy(), allowed_exit_codes=allowed)
            record["stages"][name].update(exit_code=status, finished_unix=time.time())
            write_json(record_path, record)
        from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
        model, meta, budget = validate_checkpoint(step / "final_model", record["architecture"])
        audit = read(step / "audit.json")
        require(audit["status"] == "verified" and audit["source_checkpoint_id"] == record["source_checkpoint_id"]
                and audit["checkpoint_id"] == meta["checkpoint_id"]
                and audit["fresh_strict_reload_state_exact"] and audit["frozen_bytes_unchanged"]
                and audit["rank_zero"] and audit["serialized_tensors_compared"] == 3423
                and audit["reloaded_tensors_compared"] == 3423
                and audit["selected_module_count"] == 72 and audit["selected_decoder_tensor_count"] == 432
                and audit["frozen_tensors_compared"] == 2991 and audit["changed_code_bits"] == 0,
                "Saved OV72 updates or the complete frozen-state audit failed.")
        require(budget["budget_passed"] and budget["user_linear_bpw"] <= 2.5,
                "The actual persistent bit budget exceeds the user limit.")
        current_paths = [step / "mmlu.json", step / "seven.json"]
        current = collect_results(current_paths, record["architecture"], checkpoint=(model, meta))
        require(current["leaf_count"] == 64 and current["sample_count"] == 24742, "Incomplete full eight-task evaluation.")
        control_paths = [Path(path) for path in record["control_result_paths"]]
        require(len(control_paths) == 2, "Require the reviewed control MMLU and recovered seven-task shards.")
        control_model = Path(read(control_paths[0])["checkpoint_dir"]).resolve()
        control_meta = read(control_model / "checkpoint_meta.json")
        control = collect_results(control_paths, record["architecture"], checkpoint=(control_model, control_meta))
        for actual_path, control_path in zip(current_paths, control_paths, strict=True):
            actual_result, control_result = (read(path)["evals"]["lm_eval"] for path in (actual_path, control_path))
            for field in ("configs", "versions", "n_shot", "n_samples"):
                require(actual_result[field] == control_result[field], f"Actual/control {actual_path.stem} {field} differs.")
        check(record)
        summary = {"status": "full8_evaluation_verified", "checkpoint_id": meta["checkpoint_id"],
            "checkpoint_dir": str(model), **current, "audit": str(step / "audit.json"),
            "audit_sha256": sha256(step / "audit.json"), "bit_budget": budget,
            "user_linear_bpw": budget["user_linear_bpw"], "budget_passed": budget["budget_passed"],
            "actual_control_exposure_exact": True, "actual_configs_versions_shots_counts_equal_control": True,
            "control_mean_percent": control["mean_percent"],
            "mean_delta_percentage_points": current["mean_percent"] - control["mean_percent"],
            "task_delta_percentage_points": {task: 100 * (value - control["task_metrics"][task])
                                             for task, value in current["task_metrics"].items()},
            "control_results": control["input_results"], "longbench_status": "not_evaluated",
            "current_nine_item_goal_status": "not_evaluated", "goal_acceptance": False}
        write_json(output / "full8_summary.json", summary)
        record.update(status="completed", final_model=str(model), checkpoint_id=meta["checkpoint_id"],
            mean_percent=current["mean_percent"], mmlu_percent=100 * current["task_metrics"]["mmlu"],
            full8_summary=str(output / "full8_summary.json"), full8_summary_sha256=sha256(output / "full8_summary.json"),
            audit_sha256=sha256(step / "audit.json"), postprocessing_seconds=time.monotonic() - started,
            longbench_status="not_evaluated", current_nine_item_goal_status="not_evaluated",
            current_nine_item_goal_passed=False)
        code = 0
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record["traceback"], flush=True)
    finally:
        record.update(dispatcher_exit_code=code, exit_code=code, finished_unix=time.time())
        write_json(record_path, record)
        (output / "dispatcher.exit").write_text(str(code) + "\n")
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.launch_record))
