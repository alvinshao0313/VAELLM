"""The five fixed, finite queues authorized for the October 26 C/G follow-up."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_20261026.queue_predecessor import PREDECESSORS, validate_predecessor
from experiments.night_sequence_20261026.queue_wait import (
    free_gpu, process_start, read_json, require, run_stage, sha256, wait_for_process, write_json,
)

HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/star/miniconda3/envs/bitvae/bin/python")
RESULT = ROOT / "result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026"
ARTIFACTS = Path("/data/star/VAELLM/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026")
QUEUE = RESULT / "queue"
MANIFEST = QUEUE / "manifest.json"
PREPARED = RESULT / "recovery/prepared.json"
DONORS = (("C", "ce10"), ("C", "ce50"), ("G", "ce10"), ("G", "ce50"))
WORKER_ORDER = (1, 3, 0, 2, 4)


def source_pins() -> dict:
    return {str(path): sha256(path) for path in sorted(HERE.glob("*.py"))}


def check_pins(manifest: dict) -> None:
    require(source_pins() == manifest["source_sha256"], "The fixed queue sources changed after preparation.")
    require(sha256(PREPARED) == manifest["prepared_sha256"], "Recovery CPU preparation changed.")


def fixed_commands() -> list[dict]:
    lanes = []
    for gpu, (profile, recipe) in enumerate(DONORS):
        lane = {"gpu": gpu, "predecessor": dict(PREDECESSORS[gpu]), "profile": profile, "recipe": recipe,
                "donor_command": [str(PYTHON), "-B", "-u", str(HERE / "recovery.py"), "run",
                                  "--profile", profile, "--recipe", recipe, "--gpu", str(gpu),
                                  "--prepared", str(PREPARED)],
                "donor_report": str(RESULT / "recovery" / profile / recipe / "pipeline.json")}
        if gpu in (0, 2):
            lane.update(peer_gpu=gpu + 1,
                assembly_command=["timeout", "--signal=TERM", "--kill-after=60s", "28800s",
                    str(PYTHON), "-B", "-u", str(HERE / "recovery.py"), "assemble", "--profile", profile,
                    "--gpus", str(gpu), "--prepared", str(PREPARED)],
                assembly_report=str(RESULT / "recovery" / profile / "assembly/pipeline.json"))
        lanes.append(lane)
    lanes.append({"gpu": 4, "predecessor": dict(PREDECESSORS[4]), "sparse_candidates": [
        {"profile": profile, "command": [str(PYTHON), "-B", "-u", str(HERE / "sparse_main.py"), "run",
            "--profile", profile, "--gpu", "4", "--output", str(RESULT / "sparse" / profile),
            "--artifact-root", str(ARTIFACTS / "sparse" / profile)],
         "report": str(RESULT / "sparse" / profile / "pipeline.json")}
        for profile in ("C", "G")]})
    return lanes


def prepare() -> None:
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "Use the caller-activated bitvae Python.")
    require(not MANIFEST.exists(), f"Never overwrite the fixed queue manifest: {MANIFEST}")
    prepared = read_json(PREPARED)
    require(prepared.get("status") == "cpu_verified"
            and prepared.get("schema") == "qwen_CG_decoder_CE10_CE50_2000_followup_v1",
            "The actual C/G recovery source preparation has not passed.")
    for name in ("recovery.py", "sparse_main.py"):
        require((HERE / name).is_file(), f"The fixed implementation is missing: {name}")
    for binding in PREDECESSORS:
        actual = process_start(binding["pid"])
        require(actual in (None, binding["start_ticks"]), "A bound predecessor PID was reused.")
        if actual is None:
            validate_predecessor(binding)
    QUEUE.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "qwen_CG_fixed_five_lane_sequence_20261026_v1", "status": "prepared",
                "prepared_unix": time.time(), "source_sha256": source_pins(),
                "prepared": str(PREPARED), "prepared_sha256": sha256(PREPARED),
                "lanes": fixed_commands(), "worker_order": list(WORKER_ORDER),
                "resource_wait": "one bound process-exit event; one GPU check before each fixed command",
                "automatic_retry": False, "resource_polling": False, "occupancy_workers": False,
                "adaptive_candidate_selection": False,
                "assembly_rule": "same-source CE50 attention + CE10 MLP, terminal2000 only",
                "sparse_order": "C then G; complete_full9 or genuine over_budget permits G; execution failure stops lane4"}
    write_json(MANIFEST, manifest)
    print(MANIFEST, flush=True)


def load_manifest() -> dict:
    manifest = read_json(MANIFEST)
    require(manifest["schema"] == "qwen_CG_fixed_five_lane_sequence_20261026_v1"
            and manifest["lanes"] == fixed_commands() and manifest["worker_order"] == list(WORKER_ORDER),
            "The manifest does not describe the authorized fixed five queues.")
    check_pins(manifest)
    return manifest


def environment() -> dict:
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))


def successful_report(path: str, status: str | tuple[str, ...]) -> dict:
    report = read_json(Path(path))
    allowed = (status,) if isinstance(status, str) else status
    require(report.get("status") in allowed, f"Fixed task has no valid successful terminal report: {path}")
    return {"path": path, "status": report["status"], "sha256": sha256(Path(path))}


def worker(gpu: int, peer_pid: int | None, peer_start: int | None) -> None:
    manifest = load_manifest()
    lane = manifest["lanes"][gpu]
    directory = QUEUE / f"lane_{gpu}"
    directory.mkdir(exist_ok=False)
    report_path = directory / "queue.json"
    report = {"status": "waiting_predecessor", "pid": os.getpid(), "start_ticks": process_start(os.getpid()),
              "gpu": gpu, "started_unix": time.time(), "predecessor": lane["predecessor"], "stages": {},
              "automatic_retry": False, "resource_polling": False}
    if gpu in (0, 2):
        require(peer_pid is not None and peer_start is not None, "The fixed assembly leader requires its same-source peer.")
        report["peer"] = {"gpu": gpu + 1, "pid": peer_pid, "start_ticks": peer_start}
    write_json(report_path, report)

    def execute(name: str, command: list[str]) -> None:
        check_pins(manifest)
        report.setdefault("resource_checks", {})[name] = free_gpu(gpu)
        report["status"] = "running_" + name
        run_stage(command, cwd=ROOT, environment=environment(), log=directory / (name + ".log"),
                  report=report, report_path=report_path, name=name)

    try:
        binding = lane["predecessor"]
        wait_for_process(binding["pid"], binding["start_ticks"])
        report["predecessor_complete"] = validate_predecessor(binding)
        write_json(report_path, report)
        if gpu < 4:
            execute("donor", lane["donor_command"])
            report["donor_complete"] = successful_report(lane["donor_report"], "complete_donor")
            if gpu in (0, 2):
                report["status"] = "waiting_same_source_donor"
                write_json(report_path, report)
                wait_for_process(peer_pid, peer_start)
                peer = read_json(QUEUE / f"lane_{gpu + 1}" / "queue.json")
                require(peer.get("pid") == peer_pid and peer.get("start_ticks") == peer_start
                        and peer.get("status") == "complete",
                        "The same-source peer donor failed; fixed assembly was not started.")
                report["peer_donor_complete"] = successful_report(manifest["lanes"][gpu + 1]["donor_report"], "complete_donor")
                execute("assembly_full9", lane["assembly_command"])
                report["assembly_complete"] = successful_report(lane["assembly_report"], "complete_assembly_full9")
        else:
            report["sparse_results"] = {}
            for candidate in lane["sparse_candidates"]:
                name = "sparse_" + candidate["profile"]
                execute(name, candidate["command"])
                report["sparse_results"][candidate["profile"]] = successful_report(
                    candidate["report"], ("complete_full9", "over_budget"))
        report.update(status="complete", exit_code=0)
    except BaseException as error:
        report.update(status="failed", exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_unix"] = time.time()
        write_json(report_path, report)


def run() -> None:
    manifest = load_manifest()
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "Use the caller-activated bitvae Python.")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "The fixed queue driver must remain CPU-only.")
    report_path = QUEUE / "driver.json"
    require(not report_path.exists(), "Never restart or overwrite this finite sequence.")
    report = {"status": "launching_fixed_queues", "pid": os.getpid(), "start_ticks": process_start(os.getpid()),
              "session_id": os.getsid(0), "started_unix": time.time(), "command": [sys.executable, *sys.argv],
              "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST), "workers": {},
              "automatic_retry": False, "resource_polling": False}
    write_json(report_path, report)
    launched = {}
    try:
        for gpu in WORKER_ORDER:
            command = [str(PYTHON), "-B", "-u", str(HERE / "night_queue.py"), "worker", "--gpu", str(gpu)]
            if gpu in (0, 2):
                peer = report["workers"][str(gpu + 1)]
                command += ["--peer-pid", str(peer["pid"]), "--peer-start", str(peer["start_ticks"])]
            log = QUEUE / f"lane_{gpu}.log"
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
            launched[gpu] = process
            report["workers"][str(gpu)] = {"pid": process.pid, "start_ticks": process_start(process.pid),
                "command": command, "log": str(log), "predecessor": manifest["lanes"][gpu]["predecessor"]}
            write_json(report_path, report)
        report["status"] = "fixed_queues_running"
        write_json(report_path, report)
    except BaseException as error:
        report.update(status="launch_failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for gpu, process in launched.items():
            report["workers"][str(gpu)]["exit_code"] = process.wait()
            write_json(report_path, report)
        report.update(finished_unix=time.time())
        if report["status"] != "launch_failed":
            passed = len(launched) == 5 and all(row["exit_code"] == 0 for row in report["workers"].values())
            report.update(status="complete" if passed else "failed", exit_code=0 if passed else 1)
        else:
            report["exit_code"] = 1
        write_json(report_path, report)
    require(report["exit_code"] == 0, "A fixed queue failed; inspect its recorded evidence. No retry was attempted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("run")
    child = subparsers.add_parser("worker")
    child.add_argument("--gpu", type=int, choices=range(5), required=True)
    child.add_argument("--peer-pid", type=int)
    child.add_argument("--peer-start", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "run":
        run()
    else:
        worker(args.gpu, args.peer_pid, args.peer_start)
