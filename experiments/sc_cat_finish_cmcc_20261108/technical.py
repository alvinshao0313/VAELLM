"""Two actual resumed updates with native saved-state/RNG/data verification."""
from __future__ import annotations

import time
STARTED = time.time()

import argparse
from pathlib import Path
import os
import signal
import traceback

from experiments.cat_auto_protocol_cmcc_20261104.native_builder import RecoveryExecutionPolicy
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.sc_cat_live_cmcc_20261106 import live_io
from . import frozen, resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    snapshot = frozen.install(args.snapshot)
    io = frozen.io
    root = Path(snapshot["output_root"])
    receipt = root / "technical_resume_v2.json"
    io.require(not receipt.exists() and os.environ.get("CONDA_DEFAULT_ENV") == "bitvae"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Run this one diagnostic exactly once on the actual free GPU0.")
    io.base.check_resources((0,))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("The resume diagnostic exhausted its30minute window.")))
    signal.setitimer(signal.ITIMER_REAL, max(.01, STARTED + 1800 - time.time()))
    request = dict(io.read(io.bound_path(snapshot["training_request"])), gpu=0,
        output=str(root / "technical" / "resume250_v2"), deadline_epoch=STARTED + 1800)
    run, result, error = None, None, None
    try:
        inventory, prepared, sources, tokenizer = live_io.load_inputs(request, with_sources=True)
        raw = io.read(io.bound_path(request["live_raw"]))
        run = resume.build(snapshot, request, technical=True, live_raw=raw, inventory=inventory,
            settings=RecoverySettings(**request["settings"]), prepared_data=prepared, sources=sources,
            tokenizer=tokenizer, output_dir=request["output"],
            execution_policy=RecoveryExecutionPolicy(**request["execution_policy"]), technical_stop_after=None)
        result = run.train()
    except BaseException:
        error = traceback.format_exc()
    finally:
        if run is not None:
            run.close()
    ended = time.time()
    value = dict(status="completed" if error is None else "failed", exit_code=0 if error is None else 1,
        phase="technical_validation", formal_eligible=False, snapshot=io.binding(args.snapshot), gpu=0,
        started_epoch=STARTED, ended_epoch=ended, actual_seconds=ended - STARTED,
        actual_GPU_hours=(ended - STARTED) / 3600, result=result, error=error,
        new_optimizer_updates=2 if error is None else None,
        note="State restoration and two real original-schedule updates; no new formal candidate, checkpoint252, export or full8.")
    io.write(receipt, value)
    print(dict(status=value["status"], exit_code=value["exit_code"], receipt=str(receipt), actual_GPU_hours=value["actual_GPU_hours"]), flush=True)
    if error is not None:
        print(error, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
