"""Execute one root-bound native short run, then separately export and audit it."""
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
    exit_code = 1
    deadline = time.monotonic() + record["resource_budget_seconds"]
    try:
        require(record["status"] == "bound_ready" and record["gpu"] is not None
                and os.environ.get("CUDA_VISIBLE_DEVICES") == str(record["gpu"])
                and not record["queued"] and not record["automatic_retry"], "Root must explicitly bind this one-time short run.")
        check_hashes(record["source_sha256"])
        check_hashes(record["input_evidence_sha256"])
        for item in record["hf_weight_stats"]:
            check_stat(item)
        record.update(status="running", worker_pid=os.getpid(),
                      worker_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
                      started_unix=time.time(), stages={})
        write_json(record_path, record)

        def stage(name, argv):
            require(not (output / (name + ".log")).exists(), "Never overwrite a prior stage log.")
            remaining = deadline - time.monotonic()
            require(remaining > 0, "The assigned total resource budget has expired.")
            with (output / (name + ".log")).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                details = {"argv": argv, "pid": process.pid,
                    "start_ticks": int(Path(f"/proc/{process.pid}/stat").read_text().split()[21]), "started_unix": time.time()}
                record["stages"][name] = details
                write_json(record_path, record)
                try:
                    code = process.wait(timeout=min(record[name + "_timeout_seconds"], remaining))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=record["termination_grace_seconds"])
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    details.update(exit_code=process.returncode, status="timeout", finished_unix=time.time())
                    raise RuntimeError(f"The isolated {name} stage exceeded its explicit resource budget.")
            details.update(exit_code=code, finished_unix=time.time(), log_sha256=sha256(output / (name + ".log")))
            (output / (name + ".exit")).write_text(str(code) + "\n")
            write_json(record_path, record)
            require(code == 0, f"The native {name} stage failed with exit {code}; no automatic retry.")

        stage("training", record["argv"])
        trained = json.loads((output / "training.json").read_text())
        native_step = Path(trained["training_result"]["run_output_dir"]) / "trainer_state/checkpoint-4"
        exported = output / "step_0004/final_model"
        exported.parent.mkdir()
        stage("export", [sys.executable, "-u", "-m", "experiments.w25_eval.subset_export",
            "--checkpoint_dir", str(native_step), "--output_dir", str(exported), "--device", "cuda:0"])
        stage("audit", [sys.executable, "-u", "-m", "experiments.w25_eval.subset_audit",
            "--source_dir", record["source_checkpoint"], "--checkpoint_dir", str(exported),
            "--training_checkpoint_dir", str(native_step), "--output", str(exported.parent / "audit.json")])
        stage("verify", [sys.executable, "-u", "-m", "experiments.long_context_recovery.verify", "--output", str(output)])
        check_hashes(record["source_sha256"])
        check_hashes(record["input_evidence_sha256"])
        for item in record["hf_weight_stats"]:
            check_stat(item)
        record.update(status="verified", verification_sha256=sha256(output / "verification.json"))
        exit_code = 0
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record["traceback"], flush=True)
    finally:
        record.update(dispatcher_exit_code=exit_code, finished_unix=time.time())
        write_json(record_path, record)
        (output / "dispatcher.exit").write_text(str(exit_code) + "\n")
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.launch_record))
