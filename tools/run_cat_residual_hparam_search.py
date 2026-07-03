#!/usr/bin/env python3
"""Run two-phase channel_residual_vae hyperparameter search with dual-GPU scheduling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.summarize_cat_residual_hparam_search import find_phase1_anchor, load_eval_mean

BASE_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "catlora_residual_from_base.sh")
EVAL_TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"

PHASE1_STAGES = 2
PHASE1_STEPS = 2000
PHASE1_LR = 1e-2

PHASE2_MIN_RATIOS = (0.25, 0.5)
PHASE2_LRS = (5e-3, 1e-2, 1.5e-2)
PHASE2_STEPS = (2000, 3000)
PHASE2_STAGES = (1, 2)

SCOPES = ("layer", "category")
PROTECT_COUNTS = (32, 48, 64)
CODEBOOK_BITS = (32, 64, 96, 128)


@dataclass(frozen=True)
class TrialSpec:
    category: str
    phase: str
    scope: str
    protect_count: int
    min_per_layer: int
    stages: int
    steps: int
    lr: float
    codebook_bits: int

    @property
    def total_train_steps(self) -> int:
        return int(self.stages) * int(self.steps)

    def run_name(self) -> str:
        lr_text = f"{self.lr:.0e}" if self.lr < 0.01 else f"{self.lr:g}"
        return (
            f"scope={self.scope}_pc={self.protect_count}_mpl={self.min_per_layer}_"
            f"st={self.stages}_sp={self.steps}_lr={lr_text}_cb={self.codebook_bits}"
        )

    def run_dir(self, search_root: str) -> str:
        return os.path.join(search_root, self.category, self.phase, self.run_name())


class ManifestWriter:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _gpu_compute_pids(gpu_id: str) -> List[int]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []

    pids: List[int] = []
    in_processes = False
    for line in proc.stdout.splitlines():
        if line.strip() == "Processes:":
            in_processes = True
            continue
        if not in_processes:
            continue
        if not line.strip():
            break
        parts = line.split()
        if len(parts) >= 5 and parts[-2] in {"C", "G"}:
            try:
                pids.append(int(parts[0]))
            except ValueError:
                continue
    return pids


def wait_for_exclusive_gpus(gpus: Sequence[str], poll_sec: int = 60) -> None:
    while True:
        busy = [gpu_id for gpu_id in gpus if _gpu_compute_pids(gpu_id)]
        if not busy:
            return
        print(f"Waiting for idle GPUs: {','.join(busy)}", flush=True)
        time.sleep(int(poll_sec))


def _format_lr(lr: float) -> str:
    if lr == int(lr):
        return str(int(lr))
    return f"{lr:.0e}" if lr < 0.01 else f"{lr:g}"


def generate_phase1_trials(category: str) -> List[TrialSpec]:
    trials: List[TrialSpec] = []
    for scope in SCOPES:
        for protect_count in PROTECT_COUNTS:
            min_per_layer = protect_count // 2
            for codebook_bits in CODEBOOK_BITS:
                trials.append(
                    TrialSpec(
                        category=category,
                        phase="phase1",
                        scope=scope,
                        protect_count=protect_count,
                        min_per_layer=min_per_layer,
                        stages=PHASE1_STAGES,
                        steps=PHASE1_STEPS,
                        lr=PHASE1_LR,
                        codebook_bits=codebook_bits,
                    )
                )
    return trials


def generate_phase2_trials(category: str, anchor: TrialSpec) -> List[TrialSpec]:
    trials: List[TrialSpec] = []
    for ratio in PHASE2_MIN_RATIOS:
        min_per_layer = max(1, int(anchor.protect_count * ratio))
        if min_per_layer >= anchor.protect_count:
            continue
        for lr in PHASE2_LRS:
            for steps in PHASE2_STEPS:
                for stages in PHASE2_STAGES:
                    spec = TrialSpec(
                        category=category,
                        phase="phase2",
                        scope=anchor.scope,
                        protect_count=anchor.protect_count,
                        min_per_layer=min_per_layer,
                        stages=stages,
                        steps=steps,
                        lr=lr,
                        codebook_bits=anchor.codebook_bits,
                    )
                    if spec.total_train_steps >= 10000:
                        continue
                    trials.append(spec)
    return trials


def train_completed(run_dir: str) -> bool:
    completed_path = os.path.join(run_dir, "completed.json")
    if not os.path.isfile(completed_path):
        return False
    with open(completed_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return bool(payload.get("completed"))


def eval_completed(run_dir: str) -> bool:
    return os.path.isfile(os.path.join(run_dir, "eval_result.json"))


def checkpoint_dir_for_run(run_dir: str) -> str:
    completed = os.path.join(run_dir, "completed.json")
    if os.path.isfile(completed):
        with open(completed, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        ckpt = payload.get("checkpoint_dir")
        if isinstance(ckpt, str) and ckpt.strip():
            return ckpt
    return os.path.join(run_dir, "checkpoint")


def build_train_command(gpu_id: str, trial: TrialSpec, run_dir: str) -> List[str]:
    return [
        "bash",
        BASE_SCRIPT,
        gpu_id,
        "--target_categories",
        trial.category,
        "--output_dir",
        run_dir,
        "--overwrite",
        "--eval_after_residual",
        "false",
        "--outlier_channel_scope",
        trial.scope,
        "--outlier_protect_count",
        str(trial.protect_count),
        "--outlier_protect_min_per_layer",
        str(trial.min_per_layer),
        "--outlier_residual_vae_stages",
        str(trial.stages),
        "--outlier_residual_vae_steps",
        str(trial.steps),
        "--outlier_residual_vae_lr",
        _format_lr(trial.lr),
        "--outlier_residual_vae_codebook_bits",
        str(trial.codebook_bits),
    ]


def build_eval_command(run_dir: str, checkpoint_dir: str) -> List[str]:
    eval_log_dir = os.path.join(run_dir, "eval_log")
    return [
        sys.executable,
        os.path.join(_REPO_ROOT, "tools", "cat_eval.py"),
        "--checkpoint_dir",
        checkpoint_dir,
        "--eval_lm_eval",
        "--tasks",
        EVAL_TASKS,
        "--eval_device",
        "cuda",
        "--eval_hif4_act",
        "false",
        "--eval_log_dir",
        eval_log_dir,
    ]


def _compute_task_mean(task_metrics: Dict[str, Any], task_names: Sequence[str]) -> Optional[float]:
    values: List[float] = []
    for task_name in task_names:
        metric = task_metrics.get(task_name)
        if metric is None:
            continue
        values.append(float(metric))
    if not values:
        return None
    return float(sum(values) / len(values))


def _parse_cat_eval_summary(output: str) -> Optional[Dict[str, Any]]:
    marker = "Evaluation summary:"
    idx = output.rfind(marker)
    if idx < 0:
        return None
    payload = output[idx + len(marker) :].strip()
    start = payload.find("{")
    if start < 0:
        return None
    payload = payload[start:]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _read_latest_eval_summary(eval_log_dir: str) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(eval_log_dir):
        return None
    candidates = [
        os.path.join(eval_log_dir, name)
        for name in os.listdir(eval_log_dir)
        if name.startswith("cat_eval_") and name.endswith(".log")
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    with open(candidates[-1], "r", encoding="utf-8") as handle:
        return _parse_cat_eval_summary(handle.read())


# 清理 checkpoint 时保留：residual_from_base.log、config.json、metrics.json、
# payload_summary.json、completed.json、eval_result.json、eval.log、eval_log/
REQUIRED_AFTER_EVAL_CLEANUP = (
    "residual_from_base.log",
    "config.json",
    "metrics.json",
    "payload_summary.json",
    "completed.json",
    "eval_result.json",
)


def _assert_run_artifacts_preserved(run_dir: str) -> None:
    missing = [
        name for name in REQUIRED_AFTER_EVAL_CLEANUP if not os.path.isfile(os.path.join(run_dir, name))
    ]
    if missing:
        raise RuntimeError(
            f"Checkpoint cleanup removed or damaged preserved artifacts in {run_dir}: missing {missing}"
        )


def cleanup_checkpoint(run_dir: str) -> bool:
    run_dir = os.path.abspath(run_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir_for_run(run_dir))
    expected_checkpoint_dir = os.path.abspath(os.path.join(run_dir, "checkpoint"))
    if checkpoint_dir != expected_checkpoint_dir and not checkpoint_dir.startswith(run_dir + os.sep):
        raise ValueError(
            f"Refusing to delete checkpoint outside run_dir: run_dir={run_dir}, checkpoint_dir={checkpoint_dir}"
        )
    if not os.path.isdir(checkpoint_dir):
        return False
    shutil.rmtree(checkpoint_dir)
    _assert_run_artifacts_preserved(run_dir)
    print(f"[cleanup] removed checkpoint only, preserved logs/config under {run_dir}", flush=True)
    return True


def run_trial_train(
    *,
    gpu_id: str,
    trial: TrialSpec,
    search_root: str,
    manifest: ManifestWriter,
    dry_run: bool,
) -> Tuple[TrialSpec, int, str]:
    run_dir = trial.run_dir(search_root)
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)

    started_at = time.time()
    record: Dict[str, Any] = {
        "stage": "train",
        "category": trial.category,
        "phase": trial.phase,
        "gpu": gpu_id,
        "run_dir": run_dir,
        "run_name": trial.run_name(),
        "params": asdict(trial),
        "started_at_unix": started_at,
        "status": "started",
    }

    if train_completed(run_dir):
        record.update(
            {
                "status": "skipped_completed",
                "exit_code": 0,
                "finished_at_unix": started_at,
                "duration_sec": 0.0,
            }
        )
        manifest.append(record)
        print(f"[skip-train] gpu={gpu_id} {trial.category}/{trial.phase}/{trial.run_name()}", flush=True)
        return trial, 0, run_dir

    cmd = build_train_command(gpu_id, trial, run_dir)
    print(f"[train] gpu={gpu_id} {trial.category}/{trial.phase}/{trial.run_name()}", flush=True)
    if dry_run:
        record.update(
            {
                "status": "dry_run",
                "exit_code": 0,
                "command": cmd,
                "finished_at_unix": time.time(),
                "duration_sec": 0.0,
            }
        )
        manifest.append(record)
        return trial, 0, run_dir

    proc = subprocess.run(cmd, cwd=_REPO_ROOT)
    finished_at = time.time()
    record.update(
        {
            "status": "completed" if proc.returncode == 0 else "failed",
            "exit_code": int(proc.returncode),
            "finished_at_unix": finished_at,
            "duration_sec": finished_at - started_at,
            "command": cmd,
        }
    )
    manifest.append(record)
    return trial, int(proc.returncode), run_dir


def run_trial_eval(
    *,
    gpu_id: str,
    trial: TrialSpec,
    search_root: str,
    manifest: ManifestWriter,
    dry_run: bool,
    wait_gpu_poll_sec: int,
) -> Tuple[TrialSpec, int, str]:
    run_dir = trial.run_dir(search_root)
    started_at = time.time()
    record: Dict[str, Any] = {
        "stage": "eval",
        "category": trial.category,
        "phase": trial.phase,
        "gpu": gpu_id,
        "run_dir": run_dir,
        "run_name": trial.run_name(),
        "params": asdict(trial),
        "started_at_unix": started_at,
        "status": "started",
    }

    if eval_completed(run_dir):
        eval_mean = load_eval_mean(run_dir, trial.category)
        record.update(
            {
                "status": "skipped_completed",
                "exit_code": 0,
                "eval_mean": eval_mean,
                "finished_at_unix": started_at,
                "duration_sec": 0.0,
            }
        )
        manifest.append(record)
        print(f"[skip-eval] gpu={gpu_id} {trial.category}/{trial.phase}/{trial.run_name()}", flush=True)
        return trial, 0, run_dir

    if not train_completed(run_dir):
        record.update(
            {
                "status": "skipped_no_train",
                "exit_code": 1,
                "finished_at_unix": time.time(),
                "duration_sec": 0.0,
            }
        )
        manifest.append(record)
        print(f"[skip-eval-no-train] {trial.category}/{trial.phase}/{trial.run_name()}", flush=True)
        return trial, 1, run_dir

    checkpoint_dir = checkpoint_dir_for_run(run_dir)
    if not os.path.isdir(checkpoint_dir):
        record.update(
            {
                "status": "failed_missing_checkpoint",
                "exit_code": 1,
                "finished_at_unix": time.time(),
                "duration_sec": 0.0,
            }
        )
        manifest.append(record)
        print(f"[eval-missing-ckpt] {run_dir}", flush=True)
        return trial, 1, run_dir

    cmd = build_eval_command(run_dir, checkpoint_dir)
    print(f"[eval] gpu={gpu_id} {trial.category}/{trial.phase}/{trial.run_name()}", flush=True)
    if dry_run:
        record.update(
            {
                "status": "dry_run",
                "exit_code": 0,
                "command": cmd,
                "finished_at_unix": time.time(),
                "duration_sec": 0.0,
            }
        )
        manifest.append(record)
        return trial, 0, run_dir

    wait_for_exclusive_gpus([gpu_id], poll_sec=wait_gpu_poll_sec)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = _REPO_ROOT
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    finished_at = time.time()
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    eval_mean: Optional[float] = None
    task_metrics: Dict[str, Any] = {}
    if proc.returncode == 0:
        summary = _read_latest_eval_summary(os.path.join(run_dir, "eval_log")) or _parse_cat_eval_summary(output)
        lm_eval = ((summary or {}).get("evals") or {}).get("lm_eval") or {}
        task_metrics = lm_eval.get("task_metrics") or {}
        task_names = [item.strip() for item in EVAL_TASKS.split(",") if item.strip()]
        eval_mean = _compute_task_mean(task_metrics, task_names)

    eval_result = {
        "category": trial.category,
        "eval_tasks": EVAL_TASKS,
        "eval_mean": eval_mean,
        "task_metrics": task_metrics,
        "exit_code": int(proc.returncode),
        "checkpoint_dir_before_cleanup": checkpoint_dir,
    }
    with open(os.path.join(run_dir, "eval_result.json"), "w", encoding="utf-8") as handle:
        json.dump(eval_result, handle, ensure_ascii=False, indent=2, sort_keys=True)

    if eval_mean is not None:
        with open(os.path.join(run_dir, "eval.log"), "a", encoding="utf-8") as handle:
            handle.write(
                f"类别 {trial.category}/after_residual 下游任务均值: {eval_mean:.4f} ({eval_mean * 100.0:.2f}%)\n"
            )

    checkpoint_removed = False
    if proc.returncode == 0:
        checkpoint_removed = cleanup_checkpoint(run_dir)
    eval_result["checkpoint_removed"] = checkpoint_removed
    with open(os.path.join(run_dir, "eval_result.json"), "w", encoding="utf-8") as handle:
        json.dump(eval_result, handle, ensure_ascii=False, indent=2, sort_keys=True)

    if proc.returncode != 0:
        print(output[-4000:], flush=True)

    record.update(
        {
            "status": "completed" if proc.returncode == 0 and eval_mean is not None else "failed",
            "exit_code": int(proc.returncode),
            "eval_mean": eval_mean,
            "checkpoint_removed": checkpoint_removed,
            "finished_at_unix": finished_at,
            "duration_sec": finished_at - started_at,
            "command": cmd,
        }
    )
    manifest.append(record)
    return trial, int(proc.returncode), run_dir


def run_train_batch(
    *,
    trials: Sequence[TrialSpec],
    gpus: Sequence[str],
    search_root: str,
    manifest_path: str,
    dry_run: bool,
) -> None:
    manifest = ManifestWriter(manifest_path)
    pending = list(trials)
    if not pending:
        print("No train trials to run.")
        return

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        in_flight: Dict[Future, str] = {}
        gpu_queue = list(gpus)

        def submit_next() -> None:
            if not pending or not gpu_queue:
                return
            gpu_id = gpu_queue.pop(0)
            trial = pending.pop(0)
            future = executor.submit(
                run_trial_train,
                gpu_id=gpu_id,
                trial=trial,
                search_root=search_root,
                manifest=manifest,
                dry_run=dry_run,
            )
            in_flight[future] = gpu_id

        for _ in range(min(len(gpus), len(pending))):
            submit_next()

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                gpu_id = in_flight.pop(future)
                gpu_queue.append(gpu_id)
                exc = future.exception()
                if exc is not None:
                    raise exc
                submit_next()


def run_eval_batch(
    *,
    trials: Sequence[TrialSpec],
    eval_gpu: str,
    search_root: str,
    manifest_path: str,
    dry_run: bool,
    wait_gpu_poll_sec: int,
) -> None:
    manifest = ManifestWriter(manifest_path)
    pending = [trial for trial in trials if train_completed(trial.run_dir(search_root))]
    if not pending:
        print("No eval trials to run.")
        return

    for trial in pending:
        run_trial_eval(
            gpu_id=eval_gpu,
            trial=trial,
            search_root=search_root,
            manifest=manifest,
            dry_run=dry_run,
            wait_gpu_poll_sec=wait_gpu_poll_sec,
        )


def anchor_from_disk(category: str, search_root: str) -> Optional[TrialSpec]:
    from tools.summarize_cat_residual_hparam_search import discover_trials

    records = discover_trials(search_root, [category])
    anchor_record = find_phase1_anchor(records, category)
    if anchor_record is None:
        return None
    return TrialSpec(
        category=category,
        phase="phase1",
        scope=anchor_record.scope,
        protect_count=anchor_record.protect_count,
        min_per_layer=anchor_record.min_per_layer,
        stages=anchor_record.stages,
        steps=anchor_record.steps,
        lr=anchor_record.lr,
        codebook_bits=anchor_record.codebook_bits,
    )


def build_trials(categories: Sequence[str], search_root: str, phase: str) -> List[TrialSpec]:
    trials: List[TrialSpec] = []
    if phase in ("all", "phase1"):
        for category in categories:
            trials.extend(generate_phase1_trials(category))
    if phase in ("all", "phase2"):
        for category in categories:
            anchor = anchor_from_disk(category, search_root)
            if anchor is None:
                raise RuntimeError(
                    f"No phase1 anchor found for category={category}. Run phase1 train+eval first."
                )
            trials.extend(generate_phase2_trials(category, anchor))
    return trials


def reset_dry_run_artifacts(search_root: str) -> None:
    manifest_path = os.path.join(search_root, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        return
    has_real_completed = False
    for root, _dirs, files in os.walk(search_root):
        if "completed.json" in files or "eval_result.json" in files:
            has_real_completed = True
            break
    if has_real_completed:
        return
    os.remove(manifest_path)
    summary_path = os.path.join(search_root, "HPARAM_SEARCH_SUMMARY.md")
    if os.path.isfile(summary_path):
        os.remove(summary_path)


def summarize_results(search_root: str, categories: Sequence[str]) -> None:
    from tools.summarize_cat_residual_hparam_search import main as summarize_main

    summarize_main(
        [
            "--search_root",
            search_root,
            "--categories",
            ",".join(categories),
        ]
    )


def run_phase_pipeline(
    *,
    phase: str,
    categories: Sequence[str],
    search_root: str,
    gpus: Sequence[str],
    eval_gpu: str,
    manifest_path: str,
    dry_run: bool,
    wait_gpu_poll_sec: int,
    mode: str,
) -> None:
    trials = build_trials(categories, search_root, phase)
    if mode in ("all", "train"):
        print(f"{phase} train trials: {len(trials)}", flush=True)
        run_train_batch(
            trials=trials,
            gpus=gpus,
            search_root=search_root,
            manifest_path=manifest_path,
            dry_run=dry_run,
        )
    if mode in ("all", "eval"):
        print(f"{phase} eval trials: {len(trials)}", flush=True)
        run_eval_batch(
            trials=trials,
            eval_gpu=eval_gpu,
            search_root=search_root,
            manifest_path=manifest_path,
            dry_run=dry_run,
            wait_gpu_poll_sec=wait_gpu_poll_sec,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run cat residual hparam search.")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--eval_gpu", type=str, default="")
    parser.add_argument("--categories", type=str, default="up_proj,gate_proj")
    parser.add_argument(
        "--search_root",
        type=str,
        default=".result/catlora_residual_from_base/hparam_search",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=("all", "phase1", "phase2", "summarize"),
        default="all",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("all", "train", "eval"),
        default="all",
        help="Train and eval are split: train runs in parallel without lm_eval; eval runs sequentially.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--wait_gpu_poll_sec", type=int, default=60)
    args = parser.parse_args(argv)

    search_root = os.path.abspath(args.search_root)
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    eval_gpu = args.eval_gpu.strip() or gpus[0]
    manifest_path = os.path.join(search_root, "manifest.jsonl")
    os.makedirs(search_root, exist_ok=True)

    if args.phase == "summarize":
        summarize_results(search_root, categories)
        return

    if not args.dry_run:
        reset_dry_run_artifacts(search_root)

    if args.phase in ("all", "phase1"):
        run_phase_pipeline(
            phase="phase1",
            categories=categories,
            search_root=search_root,
            gpus=gpus,
            eval_gpu=eval_gpu,
            manifest_path=manifest_path,
            dry_run=bool(args.dry_run),
            wait_gpu_poll_sec=int(args.wait_gpu_poll_sec),
            mode=args.mode,
        )

    if args.phase in ("all", "phase2"):
        run_phase_pipeline(
            phase="phase2",
            categories=categories,
            search_root=search_root,
            gpus=gpus,
            eval_gpu=eval_gpu,
            manifest_path=manifest_path,
            dry_run=bool(args.dry_run),
            wait_gpu_poll_sec=int(args.wait_gpu_poll_sec),
            mode=args.mode,
        )

    if not args.dry_run and args.mode in ("all", "eval"):
        summarize_results(search_root, categories)


if __name__ == "__main__":
    main()
