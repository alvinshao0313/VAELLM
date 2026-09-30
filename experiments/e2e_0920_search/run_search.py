#!/usr/bin/env python3
"""Run the three authorized rank-8 LR trials in order; stop on any failure.

Activate bitvae in the calling shell and launch this entry with nohup or tmux.
This finite runner does not wait for occupied GPUs, retry, promote, or delete runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
BASE_SCRIPT = REPO / "compressed_e2e_fintuning/scripts/e2e_decoder.sh"
BASE_SCRIPT_SHA256 = "c582712089960a18e43e2ece57b098c0f4c57246cc8f006a53a8b8383f731f6d"
CHECKPOINT = Path("/root/data/ckpts/result/catlora/remaining_lora_mass/Qwen_Qwen3-8B_20260920_095822/final_model")
CHECKPOINT_ID = "1a6fa98c-6685-4dab-a222-e03695919bfb"
SEARCH_ROOT = Path("/root/data/ckpts/result/compressed_e2e_fintuning/e2e_0920_search_20260924")
TASKS = ("boolq", "rte", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa", "mmlu")
TRIALS = (("lr1e4", "1e-4"), ("lr3e5", "3e-5"), ("lr3e4", "3e-4"))
RUN_ENV = {"STUDENT_CKPT": str(CHECKPOINT), "PARALLEL_MODE": "dp", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "PYTHONPATH": str(REPO)}


def command_for(name: str, learning_rate: str) -> list[str]:
    return [
        "bash", str(BASE_SCRIPT),
        "--run_root_dir", str(SEARCH_ROOT / name),
        "--train_mode", "lora", "--lora_rank", "8", "--lora_alpha", "16",
        "--learning_rate", learning_rate,
        "--steps", "5000", "--stop_after_step", "1000", "--warmup_steps", "150",
        "--save_steps", "500", "--save_total_limit", "1",
        "--eval_after_save", "true", "--eval_tasks", ",".join(TASKS),
        "--eval_num_fewshot", "0",
    ]


def source_hashes() -> dict[str, str]:
    paths = {BASE_SCRIPT, Path(__file__).resolve()}
    for directory in ("compressed_e2e_fintuning", "e2e_common", "train_utils", "litebsq", "rotation"):
        paths.update((REPO / directory).rglob("*.py"))
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def free_gpu_snapshot() -> list[dict]:
    def query(fields: str, kind: str) -> list[list[str]]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines() if line.strip()]

    selected = []
    for index, uuid, used, total in query("index,uuid,memory.used,memory.total", "gpu"):
        if int(index) in (0, 1, 2, 3):
            selected.append({"index": int(index), "uuid": uuid, "memory_used_mib": int(used), "memory_total_mib": int(total)})
    if {gpu["index"] for gpu in selected} != {0, 1, 2, 3}:
        raise RuntimeError("This recipe requires physical GPUs 0,1,2,3; nvidia-smi did not report all four.")
    uuids = {gpu["uuid"] for gpu in selected}
    active = [row for row in query("gpu_uuid,pid", "compute-apps") if row[0] in uuids]
    occupied = [gpu for gpu in selected if gpu["memory_used_mib"] > 1024]
    if active or occupied:
        raise RuntimeError(f"GPUs 0,1,2,3 are occupied; no training started. compute_processes={active}, memory={occupied}")
    return selected


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_stage(trial_root: Path, learning_rate: str) -> dict:
    runs = sorted(path for path in trial_root.iterdir() if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"Expected one newly created training run under {trial_root}, found {runs}.")
    run = runs[0]
    config_path = run / "normalized_e2e_runtime_args.json"
    snapshot = read_json(config_path)
    config, training = snapshot["canonical_config"], snapshot["training_args"]
    evaluation = config["runtime"]["evaluation"]
    if evaluation["eval_limit"] is not None or evaluation["eval_num_fewshot"] != 0:
        raise RuntimeError(f"Trial did not use full, zero-shot evaluation: {config_path}")
    if set(evaluation["eval_tasks"].split(",")) != set(TASKS):
        raise RuntimeError(f"Trial task set differs from the eight-task protocol: {config_path}")
    if (config["lora"]["rank"], config["lora"]["alpha"], config["opt"]["learning_rate"]) != (8, 16, float(learning_rate)):
        raise RuntimeError(f"Resolved rank/alpha/LR differs from the search recipe: {config_path}")
    if (training["max_steps"], training["warmup_steps"], training["save_steps"], training["save_total_limit"]) != (5000, 150, 500, 1):
        raise RuntimeError(f"Resolved schedule differs from the search recipe: {config_path}")

    checkpoint = run / "trainer_state/checkpoint-1000"
    state = read_json(checkpoint / "trainer_state.json")
    meta = read_json(checkpoint / "checkpoint_meta.json")
    if state["global_step"] != 1000 or state["max_steps"] != 5000:
        raise RuntimeError(f"Unexpected paused trainer state: {checkpoint}")
    if meta["checkpoint_kind"] != "training_step" or meta["round_base_checkpoint_id"] != CHECKPOINT_ID:
        raise RuntimeError(f"Paused checkpoint has the wrong kind or input model: {checkpoint}")
    required = ("training_model_state.pt", "optimizer.pt", "scheduler.pt", *(f"rng_state_{rank}.pth" for rank in range(4)))
    for filename in required:
        if not (checkpoint / filename).is_file():
            raise RuntimeError(f"Missing continuation state: {checkpoint / filename}")
    # The v6 paused branch returns a dict to main, which does not write run_meta.json.
    # Its explicit terminal log plus checkpoint state supplies the completion evidence.
    log = (run / "compressed_e2e_fintuning.log").read_text(encoding="utf-8")
    if "E2E status=paused global_step=1000 max_steps=5000; finalization skipped." not in log:
        raise RuntimeError(f"Training did not report the planned pause: {run}")
    if (run / "final_model").exists():
        raise RuntimeError(f"A stage pause unexpectedly produced final_model: {run}")

    evaluations = []
    for step in (500, 1000):
        path = run / "lm_eval" / f"lm_eval_results_step_{step}.json"
        raw = read_json(path)
        metrics = {task: float(raw["task_metrics"][task]) for task in TASKS}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
            raise RuntimeError(f"Missing, non-finite, or invalid task accuracy: {path}")
        keys = {task: raw["task_metric_keys"][task] for task in TASKS}
        evaluations.append({"step": step, "raw_results": str(path), "task_metrics": metrics, "task_metric_keys": keys, "mean_percent": 100 * sum(metrics.values()) / len(TASKS)})
    return {"status": "paused", "run_output_dir": str(run), "configuration": str(config_path), "resume_from_checkpoint": str(checkpoint), "evaluations": evaluations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without checking GPUs or writing results.")
    args = parser.parse_args()
    if hashlib.sha256(BASE_SCRIPT.read_bytes()).hexdigest() != BASE_SCRIPT_SHA256:
        raise RuntimeError("The reviewed e2e_decoder.sh recipe changed; review the search configuration before launching.")
    if args.dry_run:
        for name, learning_rate in TRIALS:
            print(shlex.join(["env", *(f"{key}={value}" for key, value in RUN_ENV.items()), *command_for(name, learning_rate)]))
        return 0

    if Path(sys.prefix).resolve() != Path("/root/miniconda3/envs/bitvae"):
        raise RuntimeError(f"Activate the confirmed bitvae environment before launching; current interpreter: {sys.executable}")
    if read_json(CHECKPOINT / "checkpoint_meta.json")["checkpoint_id"] != CHECKPOINT_ID:
        raise RuntimeError("The initial checkpoint identity has changed.")
    manifest_path = SEARCH_ROOT / "search_manifest.json"
    if manifest_path.exists() or any((SEARCH_ROOT / name).exists() for name, _ in TRIALS):
        raise FileExistsError(f"Refusing to overwrite or rerun an existing search: {SEARCH_ROOT}")
    free_gpu_snapshot()
    SEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "input_checkpoint": str(CHECKPOINT), "input_checkpoint_id": CHECKPOINT_ID, "environment": RUN_ENV, "source_sha256": hashes, "trials": []}

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    for name, learning_rate in TRIALS:
        record = {"name": name, "rank": 8, "alpha": 16, "learning_rate": float(learning_rate), "command": command_for(name, learning_rate), "status": "pending", "exit_code": None}
        manifest["trials"].append(record)
        try:
            if source_hashes() != hashes:
                raise RuntimeError("Training source changed during this search; inspect before starting another trial.")
            record["gpus_before_launch"] = free_gpu_snapshot()
            trial_root = SEARCH_ROOT / name
            trial_root.mkdir(exist_ok=False)
            record.update(status="running", launcher_log=str(trial_root / "launcher.log"), started_at_utc=datetime.now(timezone.utc).isoformat())
            save_manifest()
            print(f"Starting {name}; log: {record['launcher_log']}", flush=True)
            with (trial_root / "launcher.log").open("w", encoding="utf-8") as handle:
                handle.write(shlex.join(record["command"]) + "\n")
                handle.flush()
                result = subprocess.run(record["command"], cwd=REPO, env={**os.environ, **RUN_ENV}, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)
            record["exit_code"] = result.returncode
            if result.returncode != 0:
                raise RuntimeError(f"{name} exited with code {result.returncode}; see {record['launcher_log']}")
            record.update(collect_stage(trial_root, learning_rate))
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_manifest()
            print(f"Paused {name} at step 1000; mean={record['evaluations'][-1]['mean_percent']:.4f}%", flush=True)
        except Exception as exc:
            record.update(status="failed", error=str(exc), finished_at_utc=datetime.now(timezone.utc).isoformat())
            manifest["status"] = "failed"
            save_manifest()
            raise
    manifest.update(status="screen_complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
    save_manifest()
    print(f"Three trials paused for review. Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
