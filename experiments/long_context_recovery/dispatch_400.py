"""Train once, release the GPU, then export and evaluate the saved step400."""
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


def run(record_path):
    record_path = Path(record_path).resolve()
    record = json.loads(record_path.read_text())
    output = Path(record["output_dir"])
    code = 1
    try:
        require(record["status"] == "bound_ready" and type(record["gpu"]) is int
                and os.environ.get("CUDA_VISIBLE_DEVICES") == str(record["gpu"])
                and Path(sys.executable).resolve() == Path(record["python_executable"]).resolve()
                and not record["queued"] and not record["automatic_retry"], "Use the root-bound GPU and bitvae interpreter.")
        require(not any(key in os.environ for key in ("FLASH_ATTENTION_DETERMINISTIC", "NVIDIA_TF32_OVERRIDE")),
                "Do not inherit numerical diagnostic overrides.")
        check_hashes(record["source_sha256"])
        check_hashes(record["input_evidence_sha256"])
        for item in record["hf_weight_stats"]:
            check_stat(item)
        def require_idle():
            used = subprocess.check_output(["nvidia-smi", "--id=" + str(record["gpu"]),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
            require(not used, "Assigned GPU is occupied: " + used)
        require_idle()
        record.update(status="training", worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
            started_unix=time.time(), stages={})
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
        trained = json.loads((output / "training.json").read_text())
        result = trained["training_result"]
        require(trained["status"] == "trained" and result["global_step"] == 400
                and result["round_base_checkpoint_id"] == record["source_checkpoint_id"], "Wrong native completion.")
        checkpoint = Path(result["run_output_dir"]) / "trainer_state/checkpoint-400"
        require_idle()
        from experiments.qwen3_w25.subset_step_export_callback import stage_commands
        from experiments.qwen3_w25.step_export_callback import run_logged_stage
        step = output / "step_0400"
        step.mkdir()
        record.update(status="postprocessing", native_run_dir=result["run_output_dir"])
        write_json(record_path, record)
        started = time.monotonic()
        for name, command, allowed in stage_commands(checkpoint=checkpoint,
                source_dir=record["source_checkpoint"], output_dir=step):
            if name in ("export", "audit"):
                command[2] = "experiments.w25_eval.subset_" + name
            record["stages"][name] = {"argv": command, "started_unix": time.time()}
            write_json(record_path, record)
            status = run_logged_stage(command, name=name, output_dir=step,
                                      env=os.environ.copy(), allowed_exit_codes=allowed)
            record["stages"][name].update(exit_code=status, finished_unix=time.time())
            write_json(record_path, record)
        audit = json.loads((step / "audit.json").read_text())
        acceptance = json.loads((step / "acceptance.json").read_text())
        require(audit["status"] == "verified" and audit["source_checkpoint_id"] == record["source_checkpoint_id"]
                and audit["checkpoint_id"] == acceptance["checkpoint_id"]
                and audit["fresh_strict_reload_state_exact"] and audit["frozen_bytes_unchanged"]
                and audit["rank_zero"] and audit["serialized_tensors_compared"] == 3423
                and audit["reloaded_tensors_compared"] == 3423
                and audit["selected_module_count"] == 72 and audit["selected_decoder_tensor_count"] == 432
                and audit["frozen_tensors_compared"] == 2991 and audit["changed_code_bits"] == 0,
                "Saved OV72 update or full frozen-state audit failed.")
        require(acceptance["status"] in ("passed", "not_met") and acceptance["budget_passed"]
                and len(acceptance["n_samples"]) == 64 and sum(acceptance["n_samples"].values()) == 24742,
                "Incomplete full eight-task evaluation.")
        check_hashes(record["source_sha256"])
        check_hashes(record["input_evidence_sha256"])
        for item in record["hf_weight_stats"]:
            check_stat(item)
        record.update(status="completed", final_model=str(step / "final_model"),
            mean_percent=acceptance["mean_percent"], mmlu_percent=100 * acceptance["task_metrics"]["mmlu"],
            checkpoint_id=acceptance["checkpoint_id"], acceptance_sha256=sha256(step / "acceptance.json"),
            audit_sha256=sha256(step / "audit.json"), postprocessing_seconds=time.monotonic() - started,
            current_nine_item_goal_status="not_evaluated", current_nine_item_goal_passed=False)
        code = 0
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record["traceback"], flush=True)
    finally:
        record.update(dispatcher_exit_code=code, finished_unix=time.time())
        write_json(record_path, record)
        (output / "dispatcher.exit").write_text(str(code) + "\n")
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.launch_record))
