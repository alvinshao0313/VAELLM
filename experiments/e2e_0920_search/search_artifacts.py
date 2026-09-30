"""Read search evidence and clean explicitly retired model artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile


TASK_KEYS = {
    "boolq": "acc,none", "rte": "acc,none", "winogrande": "acc,none",
    "arc_easy": "acc_norm,none", "arc_challenge": "acc_norm,none",
    "openbookqa": "acc_norm,none", "piqa": "acc_norm,none", "mmlu": "acc,none",
}
CHECKPOINT_ID = "1a6fa98c-6685-4dab-a222-e03695919bfb"


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_metrics(path: Path, step: int, kind: str = "training") -> dict:
    raw = read_json(path)
    tasks = raw["tasks"]
    if len(tasks) != len(TASK_KEYS) or set(tasks) != set(TASK_KEYS):
        raise ValueError(f"Expected exactly the eight search tasks: {path}")
    if set(raw["task_metrics"]) != set(TASK_KEYS) or raw["task_metric_keys"] != TASK_KEYS:
        raise ValueError(f"Task metrics or metric keys differ from the search protocol: {path}")
    metrics = {task: float(raw["task_metrics"][task]) for task in TASK_KEYS}
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in metrics.values()):
        raise ValueError(f"Task accuracy must be finite and in [0, 1]: {path}")
    return {"step": step, "mean_percent": 100 * sum(metrics.values()) / len(TASK_KEYS),
            "task_metrics": metrics, "task_metric_keys": dict(TASK_KEYS),
            "raw_results": str(Path(path).absolute()), "kind": kind}


def collect_candidate(run_dir: Path, expected_step: int) -> dict:
    run = Path(run_dir).absolute()
    if expected_step <= 0 or expected_step > 5000:
        raise ValueError("expected_step must be in [1, 5000]")
    configuration = run / "normalized_e2e_runtime_args.json"
    snapshot = read_json(configuration)
    cfg, training = snapshot["canonical_config"], snapshot["training_args"]
    evaluation = cfg["runtime"]["evaluation"]
    tasks = evaluation["eval_tasks"].split(",")
    rank = cfg["lora"]["rank"]
    if (cfg["train_mode"] != "lora" or not 1 <= rank <= 8
            or cfg["aux"]["residual_lora_mode"] != "none"
            or cfg["aux"]["lm_head_train_mode"] != "linear"
            or cfg["runtime"]["parallel_mode"] != "dp"
            or cfg["opt"]["steps"] != 5000 or training["max_steps"] != 5000):
        raise ValueError(f"Resolved model or training scope differs from the search: {configuration}")
    if (evaluation["eval_limit"] is not None or evaluation["eval_num_fewshot"] != 0
            or evaluation["eval_hif4_act"] or not evaluation["eval_after_save"]
            or len(tasks) != len(TASK_KEYS) or set(tasks) != set(TASK_KEYS)):
        raise ValueError(f"Expected full, zero-shot eight-task evaluation: {configuration}")
    checkpoint = run / "trainer_state" / f"checkpoint-{expected_step}"
    state = read_json(checkpoint / "trainer_state.json")
    meta = read_json(checkpoint / "checkpoint_meta.json")
    if (state["global_step"] != expected_step or state["max_steps"] != 5000
            or meta["checkpoint_kind"] != "training_step"
            or meta["round_base_checkpoint_id"] != CHECKPOINT_ID):
        raise ValueError(f"Checkpoint state or source identity differs: {checkpoint}")
    if not (checkpoint / "training_model_state.pt").is_file():
        raise FileNotFoundError(checkpoint / "training_model_state.pt")
    if expected_step < 5000:
        required = ("optimizer.pt", "scheduler.pt",
                    *(f"rng_state_{rank}.pth" for rank in range(4)))
        for name in required:
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        marker = f"E2E status=paused global_step={expected_step} max_steps=5000; finalization skipped."
        if marker not in (run / "compressed_e2e_fintuning.log").read_text(encoding="utf-8"):
            raise ValueError(f"No explicit successful pause evidence: {run}")
        status = "paused"
    else:
        result = read_json(run / "run_meta.json")
        meta = read_json(run / "final_model" / "checkpoint_meta.json")
        if (result["global_step"] != 5000 or result["round_base_checkpoint_id"] != CHECKPOINT_ID
                or meta["checkpoint_kind"] != "final_model"
                or result["final_checkpoint_id"] != meta["checkpoint_id"]
                or meta["train_mode"] != "lora" or meta["lm_head_train_mode"] != "linear"):
            raise ValueError(f"Final model metadata does not match the completed run: {run}")
        if not (run / "final_model" / meta["state_dict_file"]).is_file():
            raise FileNotFoundError(f"Final model weights are missing: {run}")
        status = "completed"
    rows = []
    for path in (run / "lm_eval").glob("lm_eval_results_step_*.json"):
        match = re.fullmatch(r"lm_eval_results_step_(\d+)\.json", path.name)
        if match is None:
            raise ValueError(f"Malformed training evaluation filename: {path}")
        step = int(match[1])
        if step > expected_step:
            raise ValueError(f"Run has advanced past expected step {expected_step}: {path}")
        row = read_metrics(path, step)
        step_checkpoint = run / "trainer_state" / f"checkpoint-{step}"
        row.update(checkpoint=str(step_checkpoint), available=(step_checkpoint / "training_model_state.pt").is_file())
        rows.append(row)
    final = None
    if expected_step == 5000:
        final = read_metrics(run / "lm_eval" / "lm_eval_results_final.json", 5000, "final_export")
        final.update(checkpoint=str(checkpoint), available=(checkpoint / "training_model_state.pt").is_file(),
                     model_path=str(run / "final_model"), model_available=True)
        # The existing pipeline may only evaluate the exported model at step 5000.
        if not any(row["step"] == 5000 for row in rows):
            rows.append(final)
    if not any(row["step"] == expected_step for row in rows):
        raise ValueError(f"Missing evaluation at completed step {expected_step}: {run}")
    return {"evaluations": sorted(rows, key=lambda row: row["step"]),
            "final_evaluation": final, "latest_checkpoint": str(checkpoint) if checkpoint.is_dir() else None,
            "configuration": str(configuration), "run_output_dir": str(run), "status": status}


def _safe_path(path: Path, allowed_roots: tuple[Path, ...]) -> Path:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Refusing a symlink path: {path}")
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root) and resolved != root for root in allowed_roots):
        raise ValueError(f"Artifact is outside the allowed search roots: {resolved}")
    return resolved


def delete_checkpoints(run_dir: Path, keep_steps: set[int], allowed_roots: tuple[Path, ...],
                       remove_final: bool = False) -> list[dict]:
    """The caller has already established that these outputs have no live users."""
    if any(not isinstance(step, int) or isinstance(step, bool) or step < 0 for step in keep_steps):
        raise ValueError("keep_steps must contain non-negative integer steps")
    for root in allowed_roots:
        path = Path(root).absolute()
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"Refusing a symlink allowed root: {path}")
    roots = tuple(Path(root).resolve() for root in allowed_roots)
    run = _safe_path(Path(run_dir), roots)
    trainer = _safe_path(run / "trainer_state", roots)
    targets = []
    if trainer.exists():
        for path in trainer.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if match and int(match[1]) not in keep_steps:
                targets.append((path, "retired training checkpoint; no scheduled continuation or export"))
    if remove_final and ((run / "final_model").exists() or (run / "final_model").is_symlink()):
        targets.append((run / "final_model", "retired export; not selected for delivery"))
    records = []
    # Validate and inventory every target before deleting any of them.
    for target, reason in targets:
        target = _safe_path(target, roots)
        if not target.is_dir():
            raise ValueError(f"Expected an artifact directory: {target}")
        size = 0
        for current, directories, files in os.walk(target, followlinks=False):
            for name in directories + files:
                item = Path(current) / name
                if item.is_symlink():
                    raise ValueError(f"Refusing a symlink inside artifact: {item}")
                if item.is_file():
                    size += item.stat().st_size
                elif not item.is_dir():
                    raise ValueError(f"Refusing a non-regular artifact entry: {item}")
        records.append({"path": str(target), "bytes": size, "reason": reason})
    for record in records:
        print(f"Deleting retired artifact: {json.dumps(record, ensure_ascii=False)}", flush=True)
        shutil.rmtree(record["path"])
        if Path(record["path"]).exists():
            raise RuntimeError(f"Artifact was not removed: {record['path']}")
    return records


def write_leaderboard(path: Path, candidates: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("name", "params", "step", "best_mean_percent", "best_step", "best_kind",
              "best_available_mean_percent", "best_available_step",
              "last_mean_percent", "final_export_mean_percent", "status")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            rows = candidate.get("evaluations", [])
            best = min(rows, key=lambda row: (-row["mean_percent"], row["step"])) if rows else {}
            available = [row for row in rows if row.get("available") is not False]
            best_available = min(available, key=lambda row: (-row["mean_percent"], row["step"])) if available else {}
            last = max(rows, key=lambda row: row["step"]) if rows else {}
            final = candidate.get("final_evaluation") or {}
            writer.writerow({"name": candidate["name"], "params": json.dumps(candidate["params"], ensure_ascii=False, sort_keys=True),
                             "step": last.get("step", ""), "best_mean_percent": best.get("mean_percent", ""),
                             "best_step": best.get("step", ""), "best_kind": best.get("kind", ""),
                             "best_available_mean_percent": best_available.get("mean_percent", ""),
                             "best_available_step": best_available.get("step", ""),
                             "last_mean_percent": last.get("mean_percent", ""),
                             "final_export_mean_percent": final.get("mean_percent", ""), "status": candidate["status"]})
