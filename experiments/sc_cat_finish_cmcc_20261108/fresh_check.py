"""Recover the actual already exported node250 with an independent native forward."""
from __future__ import annotations

import time
STARTED = time.time()

import argparse
from pathlib import Path
import subprocess
import sys
import traceback

from . import frozen
from .worker import wrapped_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    snapshot = frozen.install(args.snapshot)
    io = frozen.io
    repair = Path(snapshot["document_versions"]["protocol"]["snapshot"]["path"]).parent
    output = repair / "fresh_validation"
    io.require(not output.exists(), "This actual independent-fresh verification is a single new output.")
    output.mkdir()
    io.base.check_resources((2,))
    old = repair / "formal/artifacts/up_proj/third_resume/node_250/export/fresh_request.json"
    request = dict(io.read(old), gpu=2, deadline_epoch=STARTED + 435, output=str(output / "fresh"))
    path = output / "request.json"
    io.write(path, request)
    command = wrapped_command([sys.executable, "-B", "-u", "-m", "experiments.sc_cat_continue_cmcc_20261107.workers",
        "--kind", "recovery_worker", "--request", str(path)], args.snapshot)
    error = None
    try:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL, timeout=max(.01, STARTED + 435 - time.time()))
        proof_path = output / "fresh/fresh_result.json"
        proof = io.read(proof_path)
        report = io.read(io.bound_path(request["source_export_report"]))
        io.require(proof["checkpoint_id"] == report["uuid"] and proof["weight_digest"] == report["digest"]
            and proof["state_exact"] and proof["forward_executed"] and proof["gpu_validation"] and proof["finite_logits"]
            and proof["exit_code"] == 0, "The recovered actual export requires its own real independent native GPU proof.")
        report.update(fresh_evidence=proof, fresh_forward_evidence=proof, gpu_validation=True,
            fresh_proof_reference=io.binding(proof_path), recovered_from_failed_export=request["source_export_report"])
        io.write(output / "recovered_node250_export.json", report)
    except BaseException:
        error = traceback.format_exc()
    ended = time.time()
    io.write(output / "execution.json", dict(status="completed" if error is None else "failed", exit_code=0 if error is None else 1,
        phase="technical_validation", gpu=2, started_epoch=STARTED, ended_epoch=ended, actual_seconds=ended - STARTED,
        actual_GPU_hours=(ended - STARTED) / 3600, snapshot=io.binding(args.snapshot), command=command, error=error))
    if error:
        print(error, flush=True)
        raise SystemExit(1)
    print(dict(status="completed_actual_fresh", output=str(output), actual_GPU_hours=(ended - STARTED) / 3600), flush=True)


if __name__ == "__main__":
    main()
