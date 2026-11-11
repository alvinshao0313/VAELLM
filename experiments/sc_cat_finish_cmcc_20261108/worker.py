"""Bootstrap original finite workers with their exact old document versions."""
from __future__ import annotations

import argparse
import functools
import importlib
from pathlib import Path
import subprocess
import sys
import types

from . import frozen, resume


PACKAGE = "experiments.sc_cat_finish_cmcc_20261108"
TARGETS = ("experiments.sc_cat_continue_cmcc_20261107.workers",
    "experiments.sc_cat_continue_cmcc_20261107.longbench", "experiments.sc_cat_continue_cmcc_20261107.replay")


def is_resume_train(request):
    return request.get("operation") == "train" and bool(request.get("resume_checkpoint"))


def wrapped_command(argv, snapshot_path):
    index = argv.index("-m")
    target = argv[index + 1]
    frozen.io.require(target in TARGETS, "Only bootstrap the already declared actual continuation workers.")
    return [*argv[:index], "-m", PACKAGE + ".worker", "--snapshot", str(snapshot_path), "--target", target,
        "--", *argv[index + 2:]]


def configure(snapshot_path):
    snapshot = frozen.install(snapshot_path)
    from experiments.sc_cat_continue_cmcc_20261107 import workers
    original_run = subprocess.run

    def fresh_run(argv, **kwargs):
        return original_run(wrapped_command(argv, snapshot_path), **kwargs)

    workers.finish_export = workers._function(workers.finish_export,
        subprocess=types.SimpleNamespace(run=fresh_run, DEVNULL=subprocess.DEVNULL))
    original = workers.recovery

    def recovery(request):
        if not is_resume_train(request):
            return original(request)
        frozen.io.require(request["operation"] == "train" and request["resume_checkpoint"] == snapshot["checkpoint250"],
            "Only the actual interrupted up250 training may use the native resume entry.")
        workers._active_request = request
        request_path = Path(sys.argv[sys.argv.index("--request") + 1]).resolve()
        builder = functools.partial(resume.build, snapshot, request, actual_request_reference=frozen.io.binding(request_path))
        return workers._function(workers.old_recovery.train, validate_request=workers._validate_recovery,
            package_pins=workers._package_pins, build_live_recovery=builder,
            node_receipts=resume.node_receipts)(request)

    workers.recovery = recovery
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    configure(Path(args.snapshot).resolve())
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [args.target, *arguments]
    importlib.import_module(args.target).main()


if __name__ == "__main__":
    main()
