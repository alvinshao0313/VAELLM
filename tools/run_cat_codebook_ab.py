#!/usr/bin/env python3
"""Schedule per-category codebook A/B trials on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

BASE_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "catlora_codebook_ab_single.sh")
EVAL_TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
DEFAULT_CATEGORIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

PPL_RE = re.compile(
    r"类别\s+(?P<category>\S+)\s+训练后 PPL:\s+(?P<ppl>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
TASK_RE = re.compile(
    r"类别\s+(?P<category>\S+)\s+下游任务\s+(?P<task>\S+):\s+\S+\s+=\s+"
    r"(?P<metric>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)
TASK_MEAN_RE = re.compile(
    r"类别\s+(?P<category>\S+)\s+下游任务均值:\s+(?P<mean>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class CodebookConfig:
    name: str
    codebook_bits: int
    codebook_dim: int
    residual_stages: int


CONFIGS: Tuple[CodebookConfig, ...] = (
    CodebookConfig(name="b32d32s2", codebook_bits=32, codebook_dim=32, residual_stages=2),
    CodebookConfig(name="b64d32s1", codebook_bits=64, codebook_dim=32, residual_stages=1),
)


@dataclass(frozen=True)
class TrialSpec:
    category: str
    config: CodebookConfig

    @property
    def config_name(self) -> str:
        return self.config.name

    def run_dir(self, search_root: str) -> str:
        return os.path.join(search_root, self.category, self.config_name)


class ManifestWriter:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def generate_trials(categories: Sequence[str]) -> List[TrialSpec]:
    trials: List[TrialSpec] = []
    for category in categories:
        for config in CONFIGS:
            trials.append(TrialSpec(category=category, config=config))
    return trials


def trial_completed(run_dir: str) -> bool:
    completed_path = os.path.join(run_dir, "completed.json")
    if not os.path.isfile(completed_path):
        return False
    with open(completed_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return bool(payload.get("completed"))


def parse_eval_from_log(log_text: str, category: str) -> Dict[str, Any]:
    ppl: Optional[float] = None
    task_mean: Optional[float] = None
    task_metrics: Dict[str, float] = {}

    for match in PPL_RE.finditer(log_text):
        if match.group("category") != category:
            continue
        ppl = float(match.group("ppl"))

    for match in TASK_RE.finditer(log_text):
        if match.group("category") != category:
            continue
        task_metrics[match.group("task")] = float(match.group("metric"))

    for match in TASK_MEAN_RE.finditer(log_text):
        if match.group("category") != category:
            continue
        task_mean = float(match.group("mean"))

    if task_mean is None and task_metrics:
        task_mean = float(sum(task_metrics.values()) / len(task_metrics))

    return {
        "wiki_ppl": ppl,
        "task_mean": task_mean,
        "task_metrics": task_metrics,
    }


def build_train_command(gpu_id: str, trial: TrialSpec, run_dir: str) -> List[str]:
    return [
        "bash",
        BASE_SCRIPT,
        str(gpu_id),
        "--compression_categories",
        trial.category,
        "--output_dir",
        run_dir,
        "--codebook_bits",
        f"default={trial.config.codebook_bits}",
        "--codebook_dim",
        f"default={trial.config.codebook_dim}",
        "--residual_stages",
        f"default={trial.config.residual_stages}",
    ]


def run_trial(
    *,
    gpu_id: str,
    trial: TrialSpec,
    search_root: str,
    manifest: ManifestWriter,
    dry_run: bool,
) -> Tuple[TrialSpec, int, str]:
    run_dir = trial.run_dir(search_root)
    trial_log = os.path.join(run_dir, "trial.log")
    started_at = time.time()
    record: Dict[str, Any] = {
        "category": trial.category,
        "config": trial.config_name,
        "gpu": gpu_id,
        "run_dir": run_dir,
        "params": {
            "category": trial.category,
            **asdict(trial.config),
        },
        "started_at_unix": started_at,
        "status": "started",
    }

    if trial_completed(run_dir):
        record.update(
            {
                "status": "skipped_completed",
                "exit_code": 0,
                "finished_at_unix": started_at,
                "duration_sec": 0.0,
            }
        )
        manifest.append(record)
        print(f"[skip] gpu={gpu_id} {trial.category}/{trial.config_name}", flush=True)
        return trial, 0, run_dir

    cmd = build_train_command(gpu_id, trial, run_dir)
    print(f"[run] gpu={gpu_id} {trial.category}/{trial.config_name}", flush=True)
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

    os.makedirs(run_dir, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT
    with open(trial_log, "w", encoding="utf-8") as log_handle:
        log_handle.write(" ".join(cmd) + "\n\n")
        log_handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finished_at = time.time()

    with open(trial_log, "r", encoding="utf-8") as handle:
        log_text = handle.read()
    parsed = parse_eval_from_log(log_text, trial.category)
    eval_result = {
        "category": trial.category,
        "config": trial.config_name,
        "codebook_bits": trial.config.codebook_bits,
        "codebook_dim": trial.config.codebook_dim,
        "residual_stages": trial.config.residual_stages,
        "eval_tasks": EVAL_TASKS,
        "wiki_ppl": parsed["wiki_ppl"],
        "task_mean": parsed["task_mean"],
        "task_metrics": parsed["task_metrics"],
        "exit_code": int(proc.returncode),
    }
    with open(os.path.join(run_dir, "eval_result.json"), "w", encoding="utf-8") as handle:
        json.dump(eval_result, handle, ensure_ascii=False, indent=2, sort_keys=True)

    success = (
        proc.returncode == 0
        and parsed["wiki_ppl"] is not None
        and parsed["task_mean"] is not None
    )
    completed_payload = {
        "completed": bool(success),
        "category": trial.category,
        "config": trial.config_name,
        "exit_code": int(proc.returncode),
        "wiki_ppl": parsed["wiki_ppl"],
        "task_mean": parsed["task_mean"],
        "finished_at_unix": finished_at,
    }
    with open(os.path.join(run_dir, "completed.json"), "w", encoding="utf-8") as handle:
        json.dump(completed_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    if proc.returncode != 0:
        print(log_text[-4000:], flush=True)

    record.update(
        {
            "status": "completed" if success else "failed",
            "exit_code": int(proc.returncode),
            "wiki_ppl": parsed["wiki_ppl"],
            "task_mean": parsed["task_mean"],
            "finished_at_unix": finished_at,
            "duration_sec": finished_at - started_at,
            "command": cmd,
        }
    )
    manifest.append(record)
    return trial, 0 if success else 1, run_dir


def run_batch(
    *,
    trials: Sequence[TrialSpec],
    gpus: Sequence[str],
    search_root: str,
    manifest_path: str,
    dry_run: bool,
) -> int:
    manifest = ManifestWriter(manifest_path)
    pending = list(trials)
    if not pending:
        print("No trials to run.")
        return 0

    failures = 0
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        in_flight: Dict[Future, str] = {}
        gpu_queue = list(gpus)

        def submit_next() -> None:
            if not pending or not gpu_queue:
                return
            gpu_id = gpu_queue.pop(0)
            trial = pending.pop(0)
            future = executor.submit(
                run_trial,
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
                _trial, exit_code, _run_dir = future.result()
                if int(exit_code) != 0:
                    failures += 1
                submit_next()
    return failures


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run per-category codebook A/B trials.")
    parser.add_argument("--gpus", type=str, default="4,5,6,7")
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
    )
    parser.add_argument(
        "--search_root",
        type=str,
        default=".result/catlora_codebook_ab",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    if os.path.isabs(args.search_root):
        search_root = os.path.abspath(args.search_root)
    else:
        search_root = os.path.abspath(os.path.join(_REPO_ROOT, args.search_root))
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not categories:
        raise ValueError("--categories must not be empty")
    if not gpus:
        raise ValueError("--gpus must not be empty")

    trials = generate_trials(categories)
    os.makedirs(search_root, exist_ok=True)
    manifest_path = os.path.join(search_root, "manifest.jsonl")

    print(f"search_root={search_root}", flush=True)
    print(f"gpus={','.join(gpus)}", flush=True)
    print(f"categories={','.join(categories)}", flush=True)
    print(f"trials={len(trials)}", flush=True)

    failures = run_batch(
        trials=trials,
        gpus=gpus,
        search_root=search_root,
        manifest_path=manifest_path,
        dry_run=bool(args.dry_run),
    )
    if failures:
        raise SystemExit(f"{failures} trial(s) failed. See {manifest_path}")

    summarize = os.path.join(_REPO_ROOT, "tools", "summarize_cat_codebook_ab.py")
    if os.path.isfile(summarize) and not args.dry_run:
        subprocess.run(
            [sys.executable, summarize, "--search_root", search_root],
            cwd=_REPO_ROOT,
            check=False,
        )


if __name__ == "__main__":
    main()
