"""Execute the fixed C/D shared-prefix chain, G tail, or verified saved D continuation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_cat import write


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def require_free_gpu(index):
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    devices = {int(row[0]): (row[1], int(row[2]))
               for line in inventory.strip().splitlines() if (row := [v.strip() for v in line.split(",")])}
    if index not in devices:
        raise ValueError(f"GPU {index} does not exist.")
    uuid, memory = devices[index]
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    if any(line.split(",")[0].strip() == uuid for line in active.splitlines()) or memory > 256:
        raise RuntimeError(f"GPU {index} is occupied; no waiting, reservation, or automatic retry is performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--branch", choices=("C", "G", "D"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--down-gpu", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path,
                        help="For saved D only, run full8/full21 on the same GPU after successful native completion.")
    args = parser.parse_args()
    if args.branch == "C" and (args.down_gpu is None or args.down_gpu == args.gpu):
        parser.error("The C/D chain needs two distinct assigned GPUs.")
    if args.branch != "C" and args.down_gpu is not None:
        parser.error("Only C has a shared down branch.")
    if args.evaluation_output is not None and args.branch != "D":
        parser.error("The saved-continuation evaluation option applies only to D.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "chain.json"
    if report_path.exists():
        parser.error("Refuse to overwrite an existing chain.")
    prepared = load(args.preparation)
    if prepared["status"] != "cpu_verified":
        parser.error("CPU preparation has not passed.")
    snapshot = Path(prepared["profiles"][args.branch]["training_cwd"])
    report = dict(status="starting", pid=os.getpid(), start_ticks=start_ticks(os.getpid()), branch=args.branch,
                  command=[sys.executable, *sys.argv], cwd=str(Path.cwd()),
                  gpu=args.gpu, down_gpu=args.down_gpu, started_unix=time.time(), tasks=[])
    write(report_path, report)

    def start(name, command, gpu, cwd):
        if gpu is not None:
            require_free_gpu(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
        logfile = args.output_dir / (name + ".log")
        with logfile.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        item = dict(name=name, pid=process.pid, start_ticks=start_ticks(process.pid), gpu=gpu, command=command, cwd=str(cwd),
                    log=str(logfile), started_unix=time.time(), status="running")
        report["tasks"].append(item)
        write(report_path, report)
        return process, item

    def finish(job):
        process, item = job
        code = process.wait()
        item.update(exit_code=code, finished_unix=time.time(), status="completed" if code == 0 else "failed")
        write(report_path, report)
        return code

    def native_command(profile, preparation, run_report, *extra):
        return [sys.executable, "-B", "-u", str(PACKAGE / "run_cat.py"),
                "--preparation", str(preparation), "--profile", profile,
                "--report", str(run_report), *extra]

    try:
        report["status"] = "running"
        if args.branch == "D":
            native_report = args.output_dir / "D_down.json"
            job = start("D_down", native_command("D", args.preparation, native_report), args.gpu, snapshot)
            if finish(job):
                raise RuntimeError("D native tail failed; inspect D_down.log.")
            if args.evaluation_output is not None:
                evaluation = start("D_full9", [sys.executable, "-B", "-u", str(PACKAGE / "evaluate_endpoint.py"),
                    "--native-report", str(native_report), "--output", str(args.evaluation_output),
                    "--gpu", str(args.gpu)], None, ROOT)
                if finish(evaluation):
                    raise RuntimeError("D full9 evaluation failed; inspect D_full9.log.")
        elif args.branch == "G":
            job = start("G_full", native_command("G", args.preparation, args.output_dir / "G_full.json"),
                        args.gpu, snapshot)
            if finish(job):
                raise RuntimeError("G native tail failed; inspect G_full.log.")
        else:
            common_report = args.output_dir / "C_shared.json"
            common = start("C_shared", native_command("C", args.preparation, common_report, "--stop-after-up"),
                           args.gpu, snapshot)
            if finish(common):
                raise RuntimeError("C shared gate/up failed; no dependent down tasks were started.")
            common_result = load(common_report)
            if common_result["status"] != "category_boundary_complete":
                raise RuntimeError("C shared run did not finish at a real after_up boundary.")
            after_up = common_result["saved_boundaries"][-1]["path"]
            fork_report = args.output_dir / "prepare_D_after_up.json"
            output_root = Path(prepared["profiles"]["C"]["boundary"]).parent.parent
            fork = start("fork_D", [sys.executable, "-B", "-u", str(PACKAGE / "fork_boundary.py"),
                         "--source", after_up, "--profiles", "D", "--output-root", str(output_root),
                         "--report", str(fork_report)], None, ROOT)
            fork_code = finish(fork)
            # C is a valid independent continuation even if D preparation fails.
            c_down = start("C_down", native_command("C", args.preparation, args.output_dir / "C_down.json",
                           "--resume-boundary", after_up, "--source-run-report", str(common_report)), args.gpu, snapshot)
            d_down = None
            launch_error = None
            if fork_code == 0:
                try:
                    d_down = start("D_down", native_command("D", fork_report, args.output_dir / "D_down.json"),
                                   args.down_gpu, snapshot)
                except Exception as error:
                    launch_error = str(error)
                    report["down_launch_error"] = launch_error
                    write(report_path, report)
            c_code = finish(c_down)
            d_code = finish(d_down) if d_down is not None else None
            if fork_code or c_code or d_code != 0 or launch_error:
                raise RuntimeError(f"C/D chain incomplete: fork={fork_code}, C={c_code}, D={d_code}, launch={launch_error}")
        report.update(status="completed", exit_code=0)
    except BaseException as error:
        report.update(status="failed", exit_code=1, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report["started_unix"])
        write(report_path, report)


if __name__ == "__main__":
    main()
