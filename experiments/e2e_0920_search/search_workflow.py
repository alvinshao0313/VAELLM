"""Finite, sequential search after the already running pilot finishes."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import select
import shlex
import shutil
import subprocess
import sys
import time

from experiments.e2e_0920_search import run_search as pilot
from experiments.e2e_0920_search.search_artifacts import (
    atomic_json, collect_candidate, delete_checkpoints, read_json, read_metrics,
    write_leaderboard,
)
from experiments.e2e_0920_search.search_policy import (
    MAX_EXTRA_PROMOTIONS, best_evaluation, coordinate_winner, coordinates,
    estimate_base_additional_steps, select_promotions,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = pilot.SEARCH_ROOT.parent / "e2e_0920_long_search_20260924"
BASE_PARAMS = {
    "learning_rate": 1e-4, "norm_lr": 1e-4, "lm_head_lr": 1e-4,
    "lora_alpha": 16, "lora_dropout": 0.1, "weight_decay": 0.001,
    "warmup_steps": 150, "hidden_loss_weight": 0.1,
    "pre_mlp_hidden_loss_weight": 0.01, "alpha": 0.95,
}
CHILD_ENV = {
    **pilot.RUN_ENV, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline", "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "DISTILL_NCCL_TIMEOUT_SEC": "10800", "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "10800",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_available(point: dict) -> bool:
    path = Path(point["checkpoint"])
    return all((path / filename).is_file() for filename in (
        "training_model_state.pt", "checkpoint_meta.json", "trainer_state.json",
    ))


def replace_flags(command: list[str], updates: dict) -> list[str]:
    """Replace existing CLI flags, with None meaning remove (not the string None)."""
    result = []
    index = 0
    flags = {f"--{key}" for key in updates}
    while index < len(command):
        token = command[index]
        if token in flags:
            if index + 1 >= len(command):
                raise ValueError(f"Missing value for {token}")
            index += 2
        else:
            result.append(token)
            index += 1
    for key, value in updates.items():
        if value is not None:
            result.extend((f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)))
    return result


def predecessor_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def open_process_exit_descriptor(pid: int) -> int:
    # bitvae's Python/libc omit pidfd_open; this host's Linux x86_64 kernel
    # supports syscall 434. The descriptor signals process exit without polling.
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This process handoff is configured for the confirmed Linux x86_64 host.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(descriptor)


def wait_for_predecessor(pid: int, expected_start: str, manifest_path: Path) -> dict:
    """Wait on the kernel process-exit event; do not poll training or hold a GPU."""
    actual_start = predecessor_start(pid)
    if actual_start is not None:
        if actual_start != expected_start:
            raise RuntimeError("Predecessor PID was reused; refusing to attach to a different process.")
        try:
            descriptor = open_process_exit_descriptor(pid)
        except ProcessLookupError:
            descriptor = None
        if descriptor is not None:
            try:
                if predecessor_start(pid) not in (None, expected_start):
                    raise RuntimeError("Predecessor identity changed while attaching.")
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    completed = read_json(manifest_path)
    if completed.get("status") != "screen_complete":
        raise RuntimeError(f"Pilot did not complete successfully: {completed.get('status')}")
    if len(completed.get("trials", [])) != 3 or any(
        item.get("status") != "paused" or item.get("exit_code") != 0
        for item in completed["trials"]
    ):
        raise RuntimeError("All three pilot trials must have completed and paused successfully.")
    return completed


def preview(scope: str, days: int) -> dict:
    return {
        "scope": scope, "budget_days": days, "target_mean_percent": 69.0,
        "initial_checkpoint": str(pilot.CHECKPOINT), "rank": 8,
        "pilot_promotion": "all three: step1000 -> step5000, same original schedule",
        "coordinates": [asdict(item) for item in coordinates(scope, days)],
        "challenger_schedule": [2500, 5000], "save_eval_interval": 500,
        "base_additional_steps": estimate_base_additional_steps(scope, days),
        "extra_promotions_max": MAX_EXTRA_PROMOTIONS, "final_export_reserve_hours": 2,
        "ranking": "eight full zero-shot task scores, equal mean; final selection after strict reload",
    }


class Search:
    def __init__(self, root: Path, scope: str, days: int, pid: int, start: str):
        self.root = root.resolve()
        self.scope, self.days = scope, days
        self.manifest_path = self.root / "long_search_manifest.json"
        if self.manifest_path.exists():
            raise FileExistsError(f"Search already exists; refusing to overwrite: {self.manifest_path}")
        self.root.mkdir(parents=True, exist_ok=True)
        predecessor = read_json(pilot.SEARCH_ROOT / "search_manifest.json")
        self.source_hashes = predecessor["source_sha256"]
        self.data_identity = read_json(pilot.SEARCH_ROOT / "verification/verification_summary.json")["data_sources"]
        self.state = {
            "status": "waiting_for_pilot", "created_at_utc": utc_now(), "pid": os.getpid(),
            "plan": preview(scope, days),
            "environment": CHILD_ENV,
            "predecessor": {"pid": pid, "start_ticks": start, "manifest": str(pilot.SEARCH_ROOT / "search_manifest.json")},
            "source_sha256": self.source_hashes,
            "search_source_sha256": {str(path.relative_to(pilot.REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(HERE.glob("*.py"))},
            "data_identity": self.data_identity, "candidates": [], "coordinate_results": [],
            "deployments": [], "cleanup": [], "extra_promotions_used": 0,
        }
        self.deadline = None
        self.seconds_per_step = 8.1
        self.save()

    def save(self):
        atomic_json(self.manifest_path, self.state)
        write_leaderboard(self.root / "leaderboard.csv", self.state["candidates"])

    def check_identity(self):
        if pilot.source_hashes() != self.source_hashes:
            raise RuntimeError("Pilot training sources changed; refusing to mix code versions.")
        for name, expected in self.state["search_source_sha256"].items():
            if hashlib.sha256((pilot.REPO / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Active search source changed: {name}")
        if read_json(pilot.CHECKPOINT / "checkpoint_meta.json")["checkpoint_id"] != pilot.CHECKPOINT_ID:
            raise RuntimeError("Initial checkpoint identity changed.")
        for source in self.data_identity:
            stat = Path(source["resolved_path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                raise RuntimeError(f"Training data identity changed: {source['resolved_path']}")

    def can_start(self, steps: int) -> bool:
        return self.deadline is None or time.time() + steps * self.seconds_per_step + 2 * 3600 < self.deadline

    def execute(self, command: list[str], log_path: Path, record: dict, *, env_extra=None):
        self.check_identity()
        pilot.free_gpu_snapshot()
        if shutil.disk_usage(self.root).free < 12 * 1024 ** 3:
            raise RuntimeError("Less than 12 GiB free for the next checkpoint/export stage.")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record.update(command=command, log=str(log_path), started_at_utc=utc_now(), status="running")
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write(shlex.join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command, cwd=pilot.REPO, env={**os.environ, **CHILD_ENV, **(env_extra or {})},
                stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
            record["pid"] = process.pid
            self.save()
            code = process.wait()
        record.update(exit_code=code, finished_at_utc=utc_now(), status="completed" if code == 0 else "failed")
        self.save()
        if code:
            raise RuntimeError(f"Stage exited with {code}: {log_path}")

    def train_to(self, candidate: dict, step: int):
        previous = candidate.get("step", 0)
        if step <= previous:
            raise ValueError("A continuation must advance optimizer steps.")
        updates = {
            **candidate["params"], "steps": 5000, "lora_rank": 8,
            "save_steps": 500, "save_total_limit": 12,
            "stop_after_step": step if step < 5000 else None,
            "resume_from_checkpoint": candidate.get("latest_checkpoint") if previous else None,
        }
        command = replace_flags(candidate["base_command"], updates)
        stage = {"from_step": previous, "to_step": step}
        candidate.setdefault("stages", []).append(stage)
        candidate["status"] = "running"
        begun = time.monotonic()
        self.execute(command, self.root / "launchers" / f"{candidate['name']}_to_{step}.log", stage)
        if previous:
            run = Path(candidate["run_output_dir"])
        else:
            matches = list(Path(candidate["trial_root"]).glob("*/normalized_e2e_runtime_args.json"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one training run for {candidate['name']}, got {matches}")
            run = matches[0].parent
        candidate.update(collect_candidate(run, step))
        candidate["step"] = step
        snapshot = read_json(Path(candidate["configuration"]))
        cfg, training = snapshot["canonical_config"], snapshot["training_args"]
        resolved = {
            "learning_rate": cfg["opt"]["learning_rate"],
            "norm_lr": cfg["aux"]["norm_lr"], "lm_head_lr": cfg["aux"]["lm_head_lr"],
            "lora_alpha": cfg["lora"]["alpha"], "lora_dropout": cfg["lora"]["dropout"],
            "weight_decay": cfg["opt"]["weight_decay"], "warmup_steps": training["warmup_steps"],
            "hidden_loss_weight": cfg["loss"]["hidden_loss_weight"],
            "pre_mlp_hidden_loss_weight": cfg["loss"]["pre_mlp_hidden_loss_weight"],
            "alpha": cfg["loss"]["alpha"],
        }
        if resolved != candidate["params"]:
            raise RuntimeError(f"Resolved hyperparameters differ from requested parameters: {candidate['name']}")
        measured = (time.monotonic() - begun) / (step - previous)
        self.seconds_per_step = max(7.0, 0.75 * self.seconds_per_step + 0.25 * measured)
        self.state["estimated_seconds_per_step_including_eval"] = self.seconds_per_step
        self.save()
        best = best_evaluation(candidate["evaluations"], through_step=step)
        print(f"Finished {candidate['name']} step={step} best_mean={best['mean_percent']:.4f}", flush=True)

    def available_points(self):
        points = []
        for candidate in self.state["candidates"]:
            for point in candidate.get("evaluations", []):
                if checkpoint_available(point):
                    points.append((candidate, point))
        return sorted(points, key=lambda pair: (-pair[1]["mean_percent"], pair[1]["step"], pair[0]["name"]))

    def prune(self, resumable: set[str]):
        protected = {(candidate["name"], point["step"]) for candidate, point in self.available_points()[:3]}
        for candidate in self.state["candidates"]:
            if not candidate.get("evaluations") or candidate.get("status") == "running":
                continue
            valid = [point for point in candidate["evaluations"] if checkpoint_available(point)]
            # One best snapshot per trial supports final re-ranking and an auditable comparison.
            best_step = max(valid, key=lambda point: point["mean_percent"])["step"] if valid else None
            keep = {step for name, step in protected if name == candidate["name"]}
            if best_step is not None:
                keep.add(best_step)
            if candidate["name"] in resumable:
                keep.add(candidate["step"])
            deletions = delete_checkpoints(
                Path(candidate["run_output_dir"]), keep,
                (pilot.SEARCH_ROOT, self.root), remove_final=True,
            )
            self.state["cleanup"].extend(deletions)
            for point in candidate["evaluations"]:
                point["available"] = checkpoint_available(point)
            if candidate.get("final_evaluation"):
                candidate["final_evaluation"]["model_available"] = False
            if candidate["name"] not in resumable:
                candidate["latest_checkpoint"] = None
                if candidate["status"] == "paused":
                    candidate["status"] = "screened_out"
        self.save()

    def deploy(self, candidate: dict, point: dict):
        key = f"{candidate['name']}_step{point['step']}"
        existing = next((item for item in self.state["deployments"] if item["name"] == key), None)
        if existing:
            return existing
        output = self.root / "deployments" / key
        record = {"name": key, "candidate": candidate["name"], "step": point["step"],
                  "params": candidate["params"], "source_mean_percent": point["mean_percent"],
                  "source_kind": point["kind"],
                  "source_checkpoint": point["checkpoint"], "model_path": str(output / "model")}
        self.state["deployments"].append(record)
        record["export"] = {}
        self.execute(
            [sys.executable, str(HERE / "export_selected.py"), "--checkpoint", point["checkpoint"],
             "--output-dir", record["model_path"]],
            output / "export.log", record["export"], env_extra={"CUDA_VISIBLE_DEVICES": "0"},
        )
        record["evaluation"] = {}
        self.execute(
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
             str(HERE / "export_selected.py"), "--evaluate-only", "--checkpoint", record["model_path"],
             "--output-dir", str(output / "evaluation")],
            output / "reload_evaluation.log", record["evaluation"],
        )
        metrics = read_metrics(output / "evaluation/lm_eval/lm_eval_results_reloaded.json", point["step"], "reloaded")
        record.update(status="evaluated", **metrics)
        record["meets_target"] = record["mean_percent"] >= 69.0
        self.save()
        self.write_best()
        return record

    def write_best(self):
        valid = [item for item in self.state["deployments"] if item.get("status") == "evaluated"]
        if not valid:
            return
        best = max(valid, key=lambda item: item["mean_percent"])
        self.state["best_result"] = best
        atomic_json(self.root / "best_result.json", best)
        self.save()

    def finish(self, reason: str):
        self.state["status"] = "finalizing_best_candidates"
        self.save()
        for candidate, point in self.available_points()[:3]:
            self.deploy(candidate, point)
        self.write_best()
        best = self.state["best_result"]
        # Ended search: only the selected deliverable has a remaining weight use.
        for candidate in self.state["candidates"]:
            if candidate.get("run_output_dir"):
                self.state["cleanup"].extend(delete_checkpoints(
                    Path(candidate["run_output_dir"]), set(), (pilot.SEARCH_ROOT, self.root), remove_final=True,
                ))
                for point in candidate.get("evaluations", []):
                    point["available"] = False
                if candidate.get("final_evaluation"):
                    candidate["final_evaluation"]["model_available"] = False
                candidate["latest_checkpoint"] = None
                if candidate.get("status") == "paused":
                    candidate["status"] = "budget_stopped"
        for record in self.state["deployments"]:
            path = Path(record["model_path"])
            if record["name"] != best["name"] and path.is_dir():
                if path.is_symlink() or not path.resolve().is_relative_to((self.root / "deployments").resolve()):
                    raise RuntimeError(f"Invalid completed export cleanup path: {path}")
                size = sum(item.lstat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())
                self.state["cleanup"].append({"path": str(path), "bytes": size, "reason": "Inferior final export; raw metrics retained"})
                self.save()
                shutil.rmtree(path)
                record["model_retained"] = False
            else:
                record["model_retained"] = True
        self.state.update(status="completed", finished_at_utc=utc_now(), stop_reason=reason,
                          target_reached=best["mean_percent"] >= 69.0)
        self.write_best()
        print(f"SEARCH_COMPLETE best={best['mean_percent']:.4f}% model={best['model_path']}", flush=True)

    def run(self):
        try:
            self.check_identity()
            print(f"WAITING_FOR_PILOT pid={self.state['predecessor']['pid']}; no GPU allocation", flush=True)
            predecessor = wait_for_predecessor(
                self.state["predecessor"]["pid"], self.state["predecessor"]["start_ticks"],
                Path(self.state["predecessor"]["manifest"]),
            )
            self.check_identity()
            pilot.free_gpu_snapshot()
            self.deadline = time.time() + self.days * 86400
            self.state.update(status="running", started_at_utc=utc_now(), deadline_epoch=self.deadline)
            for trial in predecessor["trials"]:
                name = f"pilot_{trial['name']}"
                item = {"name": name, "params": {**BASE_PARAMS, "learning_rate": trial["learning_rate"]},
                        "base_command": trial["command"], "step": 1000, "stages": [], "origin": "pilot"}
                item.update(collect_candidate(Path(trial["run_output_dir"]), 1000))
                self.state["candidates"].append(item)
            self.save()
            # This is also the first real execution of the independent export path.
            # If export/reload fails, stop here instead of spending days on unusable artifacts.
            baseline, point = self.available_points()[0]
            self.deploy(baseline, point)
            pilots = list(self.state["candidates"])
            for index, candidate in enumerate(pilots):
                if not self.can_start(5000 - candidate["step"]):
                    return self.finish("time_budget")
                self.train_to(candidate, 5000)
                self.prune({item["name"] for item in pilots[index + 1:]})
            incumbent = next(item for item in pilots if item["name"] == coordinate_winner(pilots, through_step=5000))
            self.state["incumbent"] = incumbent["name"]
            for index, coordinate in enumerate(coordinates(self.scope, self.days), 1):
                if not self.can_start(7500):
                    return self.finish("time_budget_before_next_coordinate")
                phase = {"coordinate": asdict(coordinate), "starting_incumbent": incumbent["name"], "challengers": []}
                self.state["coordinate_results"].append(phase)
                challengers = []
                for number, value in enumerate(coordinate.values, 1):
                    if not self.can_start(2500):
                        phase.update(completed=False, stop_reason="time_budget_before_challenger")
                        return self.finish("time_budget_before_challenger")
                    name = f"c{index:02d}_{coordinate.key}_{number}"
                    params = {**incumbent["params"], coordinate.key: value}
                    if params == incumbent["params"]:
                        continue
                    trial_root = self.root / "trials" / name
                    command = replace_flags(pilot.command_for(name, str(params["learning_rate"])), {"run_root_dir": trial_root})
                    item = {"name": name, "params": params, "base_command": command,
                            "trial_root": str(trial_root), "step": 0, "origin": coordinate.key}
                    self.state["candidates"].append(item)
                    phase["challengers"].append(name)
                    challengers.append(item)
                    self.train_to(item, 2500)
                    self.prune({candidate["name"] for candidate in challengers})
                slots = MAX_EXTRA_PROMOTIONS - self.state["extra_promotions_used"]
                promoted = select_promotions(challengers, through_step=2500, extra_slots=slots)
                completed = [incumbent]
                for number, name in enumerate(promoted):
                    if not self.can_start(2500):
                        if number:
                            break
                        phase.update(completed=False, stop_reason="time_budget_before_promotion")
                        return self.finish("time_budget_before_promotion")
                    item = next(candidate for candidate in challengers if candidate["name"] == name)
                    self.train_to(item, 5000)
                    completed.append(item)
                    if number:
                        self.state["extra_promotions_used"] += 1
                    self.prune(set(promoted[number + 1:]))
                winning_name = coordinate_winner(completed, through_step=5000)
                incumbent = next(item for item in completed if item["name"] == winning_name)
                phase.update(completed=True, promoted=[item["name"] for item in completed[1:]], winning_incumbent=winning_name)
                self.state["incumbent"] = winning_name
                self.prune(set())
                self.save()
            self.finish("planned_coordinates_complete")
        except Exception as exc:
            for candidate in self.state["candidates"]:
                if candidate.get("status") == "running":
                    candidate["status"] = "failed"
            self.state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=utc_now())
            self.save()
            raise


def run_search(root: Path, scope: str, days: int, pid: int, start: str):
    if Path(sys.prefix).resolve() != Path("/root/miniconda3/envs/bitvae"):
        raise RuntimeError(f"Use the confirmed bitvae interpreter, got {sys.executable}")
    Search(root, scope, days, pid, start).run()
