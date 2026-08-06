from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mix_bit.checkpoint_pool import CandidatePoolIndex
from mix_bit.cost_search import (
    audit_baseline_self_swap,
    create_cost_worker,
    evaluate_and_write_baseline_per_sample,
    module_safe_name,
    run_category_mode_job,
    summarize_paired_deltas,
    write_baseline_mode_zero_rows,
    write_json_atomic,
)
from mix_bit.kl_metric import (
    KL_MODE_EXACT_FULL_VOCAB,
    KL_MODE_TEACHER_TOPK,
    validate_kl_mode_arguments,
)
from mix_bit.model_inventory import ModelInventory
from mix_bit.schema import ResolvedRunConfig, sha256_file
from mix_bit.teacher_cache import load_teacher_cache_index

STATS_MATCH_TOL = 1e-12

BASELINE_STARTUP_TIMEOUT_SECONDS = 900.0
WORKER_STARTUP_TIMEOUT_SECONDS = 900.0
RESULT_QUEUE_POLL_SECONDS = 1.0
WORKER_JOIN_TIMEOUT_SECONDS = 30.0


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def compute_search_counts(
    *,
    category_count: int,
    target_linear_count: int,
    mode_count: int,
) -> dict[str, int | str]:
    c = int(category_count)
    l = int(target_linear_count)
    r = int(mode_count)
    if c < 1 or l < 1 or r < 1:
        raise ValueError(f"C/L/R must be >= 1, got C={c} L={l} R={r}")
    return {
        "C": c,
        "L": l,
        "R": r,
        "source_job_count": c * (r - 1),
        "non_baseline_module_evaluation_count": l * (r - 1),
        "complete_row_count": l * r,
        "formulas": {
            "source_jobs": "C * (R - 1)",
            "non_baseline_module_evaluations": "L * (R - 1)",
            "complete_rows": "L * R",
        },
    }


def validate_cost_run_arguments(
    *,
    kl_mode: str,
    teacher_topk: int | None,
    teacher_cache: str | Path | None,
    vocab_size: int | None = None,
):
    return validate_kl_mode_arguments(
        kl_mode=kl_mode,
        teacher_topk=teacher_topk,
        teacher_cache=teacher_cache,
        vocab_size=vocab_size,
    )


def cost_run_dirname(*, kl_mode: str, teacher_topk: int | None) -> str:
    mode = str(kl_mode)
    if mode == KL_MODE_TEACHER_TOPK:
        if teacher_topk is None:
            raise ValueError("teacher_topk mode requires an explicit positive teacher_topk K")
        k = int(teacher_topk)
        if k < 1:
            raise ValueError(f"teacher_topk must be >= 1, got {k}")
        return f"topk_k{k}"
    if mode == KL_MODE_EXACT_FULL_VOCAB:
        return "exact_full_vocab"
    raise ValueError(f"Unsupported kl_mode={mode!r}")


def derive_cost_run_root(
    resolved: ResolvedRunConfig,
    *,
    kl_mode: str,
    teacher_topk: int | None,
) -> Path:
    name = cost_run_dirname(kl_mode=kl_mode, teacher_topk=teacher_topk)
    return Path(resolved.canonical_run_root) / "costs" / name


def _non_baseline_modes(resolved: ResolvedRunConfig) -> list[str]:
    baseline = resolved.config.candidate_space.baseline_mode
    return [m.name for m in resolved.config.candidate_space.modes if m.name != baseline]


def _row_paths(cost_run_root: Path, module_name: str, mode: str) -> tuple[Path, Path]:
    safe = module_safe_name(module_name)
    stem = f"{safe}__{mode}"
    return (
        cost_run_root / "per_sample" / f"{stem}.npz",
        cost_run_root / "rows" / f"{stem}.json",
    )


def plan_cost_jobs(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    cost_run_root: str | Path,
    baseline_overlay_path: str | Path,
    dataset_manifest_path: str | Path,
    kl_mode: str,
    teacher_topk: int | None = None,
    teacher_cache: str | Path | None = None,
) -> dict[str, Any]:
    contract = validate_cost_run_arguments(
        kl_mode=kl_mode,
        teacher_topk=teacher_topk,
        teacher_cache=teacher_cache,
    )
    if contract.kl_mode == KL_MODE_TEACHER_TOPK:
        cache_index = load_teacher_cache_index(Path(teacher_cache) / "index.json")
        if int(cache_index["teacher_topk"]) != int(contract.teacher_topk):
            raise ValueError(
                "teacher_topk mismatch between CLI and cache index: "
                f"cli={contract.teacher_topk} cache={cache_index['teacher_topk']}"
            )
        cache_sha = sha256_file(Path(teacher_cache) / "index.json")
    else:
        cache_sha = ""

    counts = compute_search_counts(
        category_count=len(inventory.category_order),
        target_linear_count=len(inventory.targets),
        mode_count=len(resolved.config.candidate_space.modes),
    )
    cost_root = Path(cost_run_root)
    modes = _non_baseline_modes(resolved)
    jobs: list[dict[str, Any]] = []
    for category in inventory.category_order:
        module_names = [
            t.module_name for t in inventory.targets if t.category == category
        ]
        for mode in modes:
            key0 = (module_names[0], mode)
            if key0 not in pool_index.candidates:
                raise ValueError(f"Missing pool candidate for {key0}")
            source = pool_index.candidates[key0].source
            # All modules in the category/mode share one compact artifact.
            for name in module_names:
                cand = pool_index.candidates[(name, mode)]
                if cand.source.compact_state_path != source.compact_state_path:
                    raise ValueError(
                        f"Inconsistent compact artifact for category={category} mode={mode}"
                    )
            expected_rows = []
            for name in module_names:
                npz_path, row_path = _row_paths(cost_root, name, mode)
                expected_rows.append(
                    {
                        "module_name": name,
                        "mode": mode,
                        "row_path": str(row_path),
                        "per_sample_path": str(npz_path),
                    }
                )
            jobs.append(
                {
                    "job_id": f"{category}__{mode}",
                    "category": category,
                    "mode": mode,
                    "module_names": list(module_names),
                    "compact_state_path": source.compact_state_path,
                    "compact_state_sha256": source.compact_state_sha256,
                    "expected_rows": expected_rows,
                }
            )

    if len(jobs) != int(counts["source_job_count"]):
        raise ValueError(
            f"Planned job count {len(jobs)} != C*(R-1)={counts['source_job_count']}"
        )

    manifest = {
        "kind": "mix_bit_cost_jobs_manifest",
        "cost_run_root": str(Path(cost_run_root).resolve()),
        "kl_mode": contract.kl_mode,
        "metric_name": contract.metric_name,
        "teacher_topk": contract.teacher_topk,
        "teacher_cache_index_sha256": cache_sha,
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "calibration_manifest_sha256": sha256_file(dataset_manifest_path),
        "baseline_overlay_sha256": sha256_file(baseline_overlay_path),
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "baseline_mode": resolved.config.candidate_space.baseline_mode,
        "C": counts["C"],
        "L": counts["L"],
        "R": counts["R"],
        "source_job_count": counts["source_job_count"],
        "non_baseline_module_evaluation_count": counts["non_baseline_module_evaluation_count"],
        "complete_row_count": counts["complete_row_count"],
        "formulas": counts["formulas"],
        "jobs": jobs,
    }
    return manifest


def persist_jobs_manifest(cost_run_root: str | Path, manifest: Mapping[str, Any]) -> Path:
    root = Path(cost_run_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "jobs.json"
    # Canonical on-disk form: indent=2, sort_keys=True, trailing newline.
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        existing = path.read_bytes()
        if existing != rendered.encode("utf-8"):
            raise ValueError(
                "Existing jobs.json differs byte-for-byte from the newly planned manifest; "
                "refusing to resume with a metric/provenance mismatch"
            )
        return path
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    os.replace(tmp, path)
    return path


def _load_baseline_sample_ids(cost_run_root: Path) -> np.ndarray | None:
    path = cost_run_root / "baseline_per_sample.npz"
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["sample_ids"], dtype=np.int64)


def is_atomic_row_complete(
    cost_run_root: str | Path,
    *,
    module_name: str,
    mode: str,
    expected_provenance: Mapping[str, Any],
    baseline_sample_ids: np.ndarray | None = None,
) -> bool:
    root = Path(cost_run_root)
    npz_path, row_path = _row_paths(root, module_name, mode)
    if not npz_path.is_file() or not row_path.is_file():
        return False
    try:
        row = _read_json(row_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if row.get("status") != "complete":
        return False
    if str(row.get("module_name")) != str(module_name) or str(row.get("mode")) != str(mode):
        return False

    provenance_keys = (
        "kl_mode",
        "metric_name",
        "run_config_sha256",
        "model_inventory_sha256",
        "candidate_manifest_sha256",
        "calibration_manifest_sha256",
        "baseline_overlay_sha256",
        "teacher_cache_index_sha256",
    )
    for key in provenance_keys:
        if key not in expected_provenance:
            continue
        if row.get(key) != expected_provenance[key]:
            return False
    if "teacher_topk" in expected_provenance:
        if row.get("teacher_topk") != expected_provenance["teacher_topk"]:
            return False

    try:
        file_sha = sha256_file(npz_path)
    except OSError:
        return False
    if row.get("per_sample_sha256") != file_sha:
        return False

    try:
        npz = np.load(npz_path, allow_pickle=False)
        sample_ids = np.asarray(npz["sample_ids"], dtype=np.int64)
        delta = np.asarray(npz["delta_kl"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return False

    if int(row.get("sample_count", -1)) != int(sample_ids.size):
        return False
    if baseline_sample_ids is not None:
        if not np.array_equal(sample_ids, np.asarray(baseline_sample_ids, dtype=np.int64)):
            return False
    try:
        stats = summarize_paired_deltas(delta)
    except ValueError:
        return False
    for key in ("mean_delta_kl", "std_delta_kl", "standard_error_delta_kl"):
        if abs(float(row[key]) - float(stats[key])) > STATS_MATCH_TOL:
            return False
    for key in ("mean_delta_kl", "std_delta_kl", "standard_error_delta_kl"):
        val = float(row[key])
        if not math.isfinite(val):
            return False
    return True


def pending_jobs(
    manifest: Mapping[str, Any],
    cost_run_root: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    baseline_sample_ids: np.ndarray | None = None,
    recompute: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    root = Path(cost_run_root)
    if baseline_sample_ids is None:
        baseline_sample_ids = _load_baseline_sample_ids(root)
    recompute = recompute or set()
    pending: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        needs = False
        for module_name in job["module_names"]:
            key = (str(module_name), str(job["mode"]))
            if key in recompute:
                needs = True
                break
            if not is_atomic_row_complete(
                root,
                module_name=module_name,
                mode=job["mode"],
                expected_provenance=expected_provenance,
                baseline_sample_ids=baseline_sample_ids,
            ):
                needs = True
                break
        if needs:
            pending.append(dict(job))
    return pending


def provenance_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kl_mode": manifest["kl_mode"],
        "metric_name": manifest["metric_name"],
        "teacher_topk": manifest["teacher_topk"],
        "run_config_sha256": manifest["run_config_sha256"],
        "model_inventory_sha256": manifest["model_inventory_sha256"],
        "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
        "calibration_manifest_sha256": manifest["calibration_manifest_sha256"],
        "baseline_overlay_sha256": manifest["baseline_overlay_sha256"],
        "teacher_cache_index_sha256": manifest.get("teacher_cache_index_sha256", ""),
    }


def execute_category_mode_job(job: Mapping[str, Any], worker_state: Any) -> list[dict[str, Any]]:
    """Run one (category, mode) job using a resident CostWorkerContext."""
    return run_category_mode_job(
        worker_state,
        category=str(job["category"]),
        mode=str(job["mode"]),
    )


def _append_worker_log(log_path: Path, event: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


def _dead_process_descriptions(processes: Sequence[mp.Process]) -> list[str]:
    """Return stable pid/exitcode strings for processes that are not alive."""
    out: list[str] = []
    for proc in processes:
        if proc.is_alive():
            continue
        pid = getattr(proc, "pid", None)
        exitcode = getattr(proc, "exitcode", None)
        out.append(f"pid={pid} exitcode={exitcode}")
    return out


def _terminate_and_join(process: mp.Process, *, join_timeout: float = 5.0) -> None:
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.join(timeout=join_timeout)
    except Exception:
        pass


def _wait_for_single_process_message(
    *,
    process: mp.Process,
    result_queue: mp.Queue,
    expected_type: str,
    timeout_seconds: float,
    label: str,
) -> dict[str, Any]:
    """Wait with polling, child liveness checks and a total deadline."""
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_and_join(process, join_timeout=WORKER_JOIN_TIMEOUT_SECONDS)
            raise TimeoutError(
                f"{label} startup timed out after {timeout_seconds}s "
                f"(pid={getattr(process, 'pid', None)} "
                f"exitcode={getattr(process, 'exitcode', None)})"
            )
        poll = min(RESULT_QUEUE_POLL_SECONDS, remaining)
        try:
            msg = result_queue.get(timeout=poll)
        except queue.Empty:
            if not process.is_alive():
                pid = getattr(process, "pid", None)
                exitcode = getattr(process, "exitcode", None)
                raise RuntimeError(
                    f"{label} child exited before sending any message "
                    f"(pid={pid} exitcode={exitcode})"
                )
            time.sleep(min(0.001, max(0.0, remaining)))
            continue
        msg_type = msg.get("type")
        if msg_type == "failure":
            error = msg.get("error", "<no error>")
            tb = msg.get("traceback", "")
            _terminate_and_join(process)
            raise RuntimeError(
                f"{label} child reported failure: {error}\n{tb}"
            )
        if msg_type != expected_type:
            _terminate_and_join(process)
            raise RuntimeError(
                f"{label} received unexpected message type={msg_type!r} "
                f"(expected {expected_type!r}): {msg}"
            )
        try:
            process.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        except Exception:
            pass
        if process.is_alive():
            _terminate_and_join(process)
            raise RuntimeError(
                f"{label} child still alive after join "
                f"(pid={getattr(process, 'pid', None)})"
            )
        exitcode = getattr(process, "exitcode", None)
        if exitcode != 0:
            raise RuntimeError(
                f"{label} child exited with non-zero exitcode "
                f"(pid={getattr(process, 'pid', None)} exitcode={exitcode})"
            )
        return msg


def _wait_for_workers_ready(
    *,
    processes: Sequence[mp.Process],
    result_queue: mp.Queue,
    timeout_seconds: float,
) -> None:
    """Require one unique ready message from every logical worker."""
    expected = len(processes)
    ready_ids: set[int] = set()
    deadline = time.monotonic() + float(timeout_seconds)
    while len(ready_ids) < expected:
        dead = _dead_process_descriptions(processes)
        if dead:
            raise RuntimeError(
                f"worker died before ready: {'; '.join(dead)}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for proc in processes:
                if proc.is_alive():
                    _terminate_and_join(proc)
            raise TimeoutError(
                f"workers not ready after {timeout_seconds}s "
                f"(ready={sorted(ready_ids)} expected={expected})"
            )
        poll = min(RESULT_QUEUE_POLL_SECONDS, remaining)
        try:
            msg = result_queue.get(timeout=poll)
        except queue.Empty:
            dead = _dead_process_descriptions(processes)
            if dead:
                raise RuntimeError(
                    f"worker died before ready: {'; '.join(dead)}"
                )
            time.sleep(min(0.001, max(0.0, remaining)))
            continue
        msg_type = msg.get("type")
        if msg_type == "failure":
            error = msg.get("error", "<no error>")
            tb = msg.get("traceback", "")
            raise RuntimeError(
                f"worker startup failure "
                f"(logical_id={msg.get('logical_id')} "
                f"physical_gpu={msg.get('physical_gpu')}): {error}\n{tb}"
            )
        if msg_type == "ready":
            logical_id = int(msg.get("logical_id"))
            if logical_id in ready_ids:
                raise RuntimeError(
                    f"duplicate ready message for logical_id={logical_id}"
                )
            ready_ids.add(logical_id)
            continue
        raise RuntimeError(
            f"unexpected worker startup message type={msg_type!r}: {msg}"
        )


def _baseline_init_process_main(init_args: dict[str, Any]) -> None:
    """Single-GPU spawn helper: materialize baseline_per_sample.npz before workers."""
    physical_gpu = str(init_args["physical_gpu"])
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    result_queue: mp.Queue = init_args["result_queue"]
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            torch.cuda.set_device(0)

        from mix_bit.checkpoint_pool import build_candidate_pool_index_from_manifest
        from mix_bit.model_inventory import load_model_inventory
        from mix_bit.schema import resolve_run_config

        resolved = resolve_run_config(init_args["run_config"], write=False)
        inventory = load_model_inventory(init_args["inventory"])
        pool_index = build_candidate_pool_index_from_manifest(
            resolved, inventory, init_args["pool_manifest_path"]
        )
        info = evaluate_and_write_baseline_per_sample(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            baseline_overlay_path=init_args["baseline_overlay_path"],
            dataset_path=init_args["dataset_path"],
            dataset_manifest_path=init_args["dataset_manifest_path"],
            cost_run_root=init_args["cost_run_root"],
            kl_mode=init_args["kl_mode"],
            teacher_topk=init_args.get("teacher_topk"),
            teacher_cache=init_args.get("teacher_cache"),
            device=device,
            batch_size=int(init_args["batch_size"]),
            access_token=init_args.get("access_token"),
        )
        result_queue.put(
            {
                "type": "baseline_ready",
                "baseline_per_sample_path": info["baseline_per_sample_path"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "type": "failure",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _ensure_baseline_per_sample_spawn(
    *,
    resolved: ResolvedRunConfig,
    inventory_path: str | Path,
    pool_index: CandidatePoolIndex,
    baseline_overlay_path: str | Path,
    dataset_path: str | Path,
    dataset_manifest_path: str | Path,
    cost_run_root: Path,
    kl_mode: str,
    teacher_topk: int | None,
    teacher_cache: str | Path | None,
    first_gpu: str,
    batch_size: int,
    access_token: str | None,
) -> Path:
    out = cost_run_root / "baseline_per_sample.npz"
    if out.is_file():
        return out

    ctx_mp = mp.get_context("spawn")
    result_queue: mp.Queue = ctx_mp.Queue()
    init_args = {
        "physical_gpu": first_gpu,
        "result_queue": result_queue,
        "cost_run_root": str(cost_run_root.resolve()),
        "run_config": resolved.run_config_path,
        "inventory": str(inventory_path),
        "pool_manifest_path": str(Path(pool_index.manifest_path).resolve()),
        "baseline_overlay_path": str(baseline_overlay_path),
        "dataset_path": str(dataset_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "kl_mode": kl_mode,
        "teacher_topk": teacher_topk,
        "teacher_cache": None if teacher_cache is None else str(teacher_cache),
        "batch_size": batch_size,
        "access_token": access_token,
    }
    proc = ctx_mp.Process(target=_baseline_init_process_main, args=(init_args,), daemon=True)
    proc.start()
    msg = _wait_for_single_process_message(
        process=proc,
        result_queue=result_queue,
        expected_type="baseline_ready",
        timeout_seconds=BASELINE_STARTUP_TIMEOUT_SECONDS,
        label="baseline",
    )
    path = Path(msg["baseline_per_sample_path"])
    if not path.is_file():
        raise RuntimeError(f"baseline_per_sample missing after init: {path}")
    return path


def _worker_process_main(worker_args: dict[str, Any]) -> None:
    """Spawn entry: set CUDA device, load resident models, process jobs from queue."""
    physical_gpu = str(worker_args["physical_gpu"])
    logical_id = int(worker_args["logical_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu

    job_queue: mp.Queue = worker_args["job_queue"]
    result_queue: mp.Queue = worker_args["result_queue"]
    cost_run_root = Path(worker_args["cost_run_root"])
    log_path = cost_run_root / "worker_logs" / f"gpu_{physical_gpu}.jsonl"

    device = "cuda:0"
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logical_device = str(torch.cuda.current_device())
        else:
            device = "cpu"
            logical_device = "cpu"
        _append_worker_log(
            log_path,
            {
                "event": "worker_start",
                "physical_gpu": physical_gpu,
                "logical_id": logical_id,
                "logical_device": logical_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )

        from mix_bit.checkpoint_pool import build_candidate_pool_index_from_manifest
        from mix_bit.model_inventory import load_model_inventory
        from mix_bit.schema import resolve_run_config

        resolved = resolve_run_config(worker_args["run_config"], write=False)
        inventory = load_model_inventory(worker_args["inventory"])
        pool_index = build_candidate_pool_index_from_manifest(
            resolved, inventory, worker_args["pool_manifest_path"]
        )
        baseline_per_sample = worker_args.get("baseline_per_sample_path")
        ctx = create_cost_worker(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            baseline_overlay_path=worker_args["baseline_overlay_path"],
            dataset_path=worker_args["dataset_path"],
            dataset_manifest_path=worker_args["dataset_manifest_path"],
            cost_run_root=cost_run_root,
            kl_mode=worker_args["kl_mode"],
            teacher_topk=worker_args.get("teacher_topk"),
            teacher_cache=worker_args.get("teacher_cache"),
            device=device,
            batch_size=int(worker_args["batch_size"]),
            access_token=worker_args.get("access_token"),
            baseline_per_sample_path=baseline_per_sample,
        )
        result_queue.put(
            {
                "type": "ready",
                "physical_gpu": physical_gpu,
                "logical_id": logical_id,
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface to parent
        result_queue.put(
            {
                "type": "failure",
                "physical_gpu": physical_gpu,
                "logical_id": logical_id,
                "job_id": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        try:
            item = job_queue.get()
        except (EOFError, OSError):
            break
        if item is None:
            _append_worker_log(
                log_path,
                {"event": "worker_shutdown", "physical_gpu": physical_gpu},
            )
            break
        job = item
        job_id = job["job_id"]
        _append_worker_log(
            log_path,
            {"event": "job_start", "job_id": job_id, "physical_gpu": physical_gpu},
        )
        try:
            rows = execute_category_mode_job(job, ctx)
            _append_worker_log(
                log_path,
                {
                    "event": "job_end",
                    "job_id": job_id,
                    "physical_gpu": physical_gpu,
                    "row_count": len(rows),
                },
            )
            result_queue.put(
                {
                    "type": "success",
                    "physical_gpu": physical_gpu,
                    "logical_id": logical_id,
                    "job_id": job_id,
                    "row_count": len(rows),
                }
            )
        except Exception as exc:  # noqa: BLE001
            _append_worker_log(
                log_path,
                {
                    "event": "job_error",
                    "job_id": job_id,
                    "physical_gpu": physical_gpu,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            result_queue.put(
                {
                    "type": "failure",
                    "physical_gpu": physical_gpu,
                    "logical_id": logical_id,
                    "job_id": job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            # Do not continue after failure; parent will stop assigning.
            break


def run_cost_search_scheduler(
    *,
    manifest: Mapping[str, Any],
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    baseline_overlay_path: str | Path,
    dataset_path: str | Path,
    dataset_manifest_path: str | Path,
    cost_run_root: str | Path,
    kl_mode: str,
    teacher_topk: int | None,
    teacher_cache: str | Path | None,
    gpus: Sequence[str],
    batch_size: int = 1,
    access_token: str | None = None,
    device_override: str | None = None,
    in_process: bool = False,
    recompute: set[tuple[str, str]] | None = None,
    inventory_path: str | Path | None = None,
) -> int:
    """Dispatch pending jobs to GPU workers. Returns 0 on success, non-zero on failure.

    Parent process must not load Torch CUDA models when using spawn workers.
    Completion is determined only by atomic row files, never by worker JSONL logs.
    """
    if not gpus:
        raise ValueError("--gpus must be a non-empty comma-separated GPU id list")
    root = Path(cost_run_root)
    root.mkdir(parents=True, exist_ok=True)
    provenance = provenance_from_manifest(manifest)
    pending = pending_jobs(
        manifest,
        root,
        expected_provenance=provenance,
        recompute=recompute,
    )
    if not pending:
        return 0

    if in_process:
        device = device_override or "cpu"
        ctx = create_cost_worker(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            baseline_overlay_path=baseline_overlay_path,
            dataset_path=dataset_path,
            dataset_manifest_path=dataset_manifest_path,
            cost_run_root=root,
            kl_mode=kl_mode,
            teacher_topk=teacher_topk,
            teacher_cache=teacher_cache,
            device=device,
            batch_size=batch_size,
            access_token=access_token,
        )
        for job in pending:
            try:
                execute_category_mode_job(job, ctx)
            except Exception as exc:
                raise RuntimeError(
                    f"worker crash while running job {job['job_id']}: {exc}"
                ) from exc
        return 0

    if inventory_path is None:
        raise ValueError(
            "Spawn workers require inventory_path; pass the model_inventory.json path"
        )

    gpu_list = [str(g).strip() for g in gpus if str(g).strip()]
    pool_manifest_path = str(Path(pool_index.manifest_path).resolve())
    baseline_path = _ensure_baseline_per_sample_spawn(
        resolved=resolved,
        inventory_path=inventory_path,
        pool_index=pool_index,
        baseline_overlay_path=baseline_overlay_path,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        cost_run_root=root,
        kl_mode=kl_mode,
        teacher_topk=teacher_topk,
        teacher_cache=teacher_cache,
        first_gpu=gpu_list[0],
        batch_size=batch_size,
        access_token=access_token,
    )

    ctx_mp = mp.get_context("spawn")
    job_queue: mp.Queue = ctx_mp.Queue(maxsize=max(1, len(gpus)))
    result_queue: mp.Queue = ctx_mp.Queue()

    processes: list[mp.Process] = []
    for logical_id, gpu in enumerate(gpu_list):
        worker_args = {
            "physical_gpu": gpu,
            "logical_id": logical_id,
            "job_queue": job_queue,
            "result_queue": result_queue,
            "cost_run_root": str(root.resolve()),
            "run_config": resolved.run_config_path,
            "inventory": str(inventory_path),
            "pool_manifest_path": pool_manifest_path,
            "baseline_overlay_path": str(baseline_overlay_path),
            "dataset_path": str(dataset_path),
            "dataset_manifest_path": str(dataset_manifest_path),
            "kl_mode": kl_mode,
            "teacher_topk": teacher_topk,
            "teacher_cache": None if teacher_cache is None else str(teacher_cache),
            "batch_size": batch_size,
            "access_token": access_token,
            "baseline_per_sample_path": str(baseline_path),
        }
        proc = ctx_mp.Process(target=_worker_process_main, args=(worker_args,), daemon=True)
        proc.start()
        processes.append(proc)

    pending_iter = iter(pending)
    in_flight: dict[str, str] = {}  # job_id -> gpu
    stopping = False
    failures = 0
    failure_detail = ""

    try:
        # Wait for workers to become ready (or fail during load).
        _wait_for_workers_ready(
            processes=processes,
            result_queue=result_queue,
            timeout_seconds=WORKER_STARTUP_TIMEOUT_SECONDS,
        )

        def _submit_next() -> bool:
            nonlocal stopping
            if stopping:
                return False
            try:
                job = next(pending_iter)
            except StopIteration:
                return False
            job_queue.put(job)
            in_flight[job["job_id"]] = "pending"
            return True

        for _ in processes:
            if not _submit_next():
                break

        while in_flight and failures == 0:
            try:
                msg = result_queue.get(timeout=RESULT_QUEUE_POLL_SECONDS)
            except queue.Empty:
                dead = [p for p in processes if not p.is_alive()]
                if dead:
                    failures += 1
                    stopping = True
                    failure_detail = "; ".join(_dead_process_descriptions(dead))
                    break
                continue
            if msg["type"] == "success":
                in_flight.pop(msg["job_id"], None)
                _submit_next()
            elif msg["type"] == "failure":
                failures += 1
                stopping = True
                in_flight.pop(msg.get("job_id"), None)
                failure_detail = (
                    f"worker failure (job_id={msg.get('job_id')} "
                    f"logical_id={msg.get('logical_id')} "
                    f"physical_gpu={msg.get('physical_gpu')}): "
                    f"{msg.get('error', '<no error>')}"
                )
                if msg.get("traceback"):
                    failure_detail = f"{failure_detail}\n{msg['traceback']}"
                break
            elif msg["type"] == "ready":
                continue
    finally:
        # Drain: stop assigning, clear queued jobs, send sentinels, join/terminate
        # workers on every exit path (startup failure, runtime failure, or success).
        stopping = True
        drain_job_queue_and_stop_workers(
            job_queue, processes, join_timeout=WORKER_JOIN_TIMEOUT_SECONDS
        )

    if failures:
        base = (
            "One or more cost-search workers failed; "
            "atomic rows were not marked complete for crashed jobs"
        )
        if failure_detail:
            raise RuntimeError(f"{base}: {failure_detail}")
        raise RuntimeError(base)
    return 0


def drain_job_queue_and_stop_workers(
    job_queue: Any,
    processes: Sequence[Any],
    *,
    join_timeout: float = 30.0,
) -> None:
    """Clear pending jobs, send stop sentinels without hanging, join/terminate workers.

    A bounded ``job_queue`` that still holds unstarted jobs will block a naive
    ``put(None)`` forever once workers have stopped consuming. Drain first, then
    use non-blocking sentinel puts with a deadline.
    """
    # Drop any queued (not yet started) jobs so sentinel slots are available.
    while True:
        try:
            job_queue.get_nowait()
        except queue.Empty:
            break

    deadline = time.monotonic() + float(join_timeout)
    for _ in processes:
        while True:
            try:
                job_queue.put_nowait(None)
                break
            except queue.Full:
                # Another producer/consumer race may have refilled; keep draining.
                try:
                    job_queue.get_nowait()
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)

    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.join(timeout=remaining if remaining > 0 else 0.01)
        except Exception:
            pass
        if getattr(proc, "is_alive", lambda: False)():
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=5.0)
            except Exception:
                pass


def _mode_order(resolved: ResolvedRunConfig) -> dict[str, int]:
    return {m.name: idx for idx, m in enumerate(resolved.config.candidate_space.modes)}


def _category_order(inventory: ModelInventory) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(inventory.category_order)}


def _assert_row_matches_inventory(
    row: Mapping[str, Any],
    inventory: ModelInventory,
) -> None:
    target_map = {t.module_name: t for t in inventory.targets}
    name = str(row["module_name"])
    key = (name, str(row.get("mode")))
    target = target_map.get(name)
    if target is None:
        raise ValueError(f"Row module_name not in inventory: {key}")
    checks = {
        "category": target.category,
        "module_suffix": target.module_suffix,
        "block_index": int(target.block_index),
        "param_count": int(target.param_count),
    }
    for field, expected in checks.items():
        found = row.get(field)
        if field in ("block_index", "param_count"):
            found = int(found)
        if found != expected:
            raise ValueError(
                f"Row inventory metadata mismatch for {key}: "
                f"{field} row={found!r} inventory={expected!r}"
            )
    # Optional shape fields when present on the row.
    for field, expected in (
        ("in_features", int(target.in_features)),
        ("out_features", int(target.out_features)),
        ("has_bias", bool(target.has_bias)),
    ):
        if field not in row:
            continue
        found = row[field]
        if field == "has_bias":
            found = bool(found)
        else:
            found = int(found)
        if found != expected:
            raise ValueError(
                f"Row inventory metadata mismatch for {key}: "
                f"{field} row={found!r} inventory={expected!r}"
            )


def load_valid_atomic_rows(
    cost_run_root: str | Path,
    *,
    inventory: ModelInventory,
    resolved: ResolvedRunConfig,
    expected_provenance: Mapping[str, Any],
    baseline_sample_ids: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    root = Path(cost_run_root)
    if baseline_sample_ids is None:
        baseline_sample_ids = _load_baseline_sample_ids(root)
    rows: list[dict[str, Any]] = []
    modes = [m.name for m in resolved.config.candidate_space.modes]
    for target in inventory.targets:
        for mode in modes:
            if not is_atomic_row_complete(
                root,
                module_name=target.module_name,
                mode=mode,
                expected_provenance=expected_provenance,
                baseline_sample_ids=baseline_sample_ids,
            ):
                continue
            _, row_path = _row_paths(root, target.module_name, mode)
            rows.append(_read_json(row_path))
    return rows


def finalize_cost_table(
    *,
    rows: Sequence[Mapping[str, Any]],
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    cost_run_root: str | Path,
    expected_provenance: Mapping[str, Any],
    self_swap_audit: Mapping[str, Any],
    source_job_count: int,
    baseline_kl_mean: float,
) -> dict[str, Any]:
    if not self_swap_audit.get("passed"):
        raise ValueError("Refusing to finalize cost table: self-swap audit did not pass")

    seen: set[tuple[str, str]] = set()
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["module_name"]), str(row["mode"]))
        if key in seen:
            raise ValueError(f"duplicate module-mode row for {key}")
        seen.add(key)
        _assert_row_matches_inventory(row, inventory)
        for field in ("mean_delta_kl", "std_delta_kl", "standard_error_delta_kl"):
            val = float(row[field])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite {field} for {key}: {val}")
        for key_name, expected in expected_provenance.items():
            if key_name not in row:
                continue
            if row[key_name] != expected:
                raise ValueError(
                    f"Mixed provenance for {key}: {key_name} row={row[key_name]!r} "
                    f"expected={expected!r}"
                )
        cleaned.append(dict(row))

    counts = compute_search_counts(
        category_count=len(inventory.category_order),
        target_linear_count=len(inventory.targets),
        mode_count=len(resolved.config.candidate_space.modes),
    )
    expected_rows = int(counts["complete_row_count"])
    if len(cleaned) != expected_rows:
        raise ValueError(
            f"finalize requires L * R complete rows: expected={expected_rows} "
            f"got={len(cleaned)} (complete_row_count)"
        )

    # Require every inventory (module, mode) pair.
    required = {
        (t.module_name, m.name)
        for t in inventory.targets
        for m in resolved.config.candidate_space.modes
    }
    if seen != required:
        missing = sorted(required - seen)
        extra = sorted(seen - required)
        raise ValueError(
            f"Row set does not match inventory×modes; missing={missing[:5]} extra={extra[:5]}"
        )

    cat_order = _category_order(inventory)
    mode_order = _mode_order(resolved)

    def _sort_key(row: Mapping[str, Any]) -> tuple:
        return (
            int(row["block_index"]),
            cat_order[str(row["category"])],
            str(row["module_name"]),
            mode_order[str(row["mode"])],
        )

    cleaned.sort(key=_sort_key)

    baseline_mode = resolved.config.candidate_space.baseline_mode
    for row in cleaned:
        if row["mode"] == baseline_mode:
            if float(row["mean_delta_kl"]) != 0.0:
                raise ValueError(
                    f"Baseline row not exactly zero: {row['module_name']} "
                    f"mean={row['mean_delta_kl']}"
                )

    root = Path(cost_run_root)
    root.mkdir(parents=True, exist_ok=True)

    jsonl_path = root / "cost_table.jsonl"
    csv_path = root / "cost_table.csv"
    md_path = root / "cost_table_summary.md"
    meta_path = root / "cost_table_meta.json"

    jsonl_tmp = jsonl_path.with_name(jsonl_path.name + ".tmp")
    with open(jsonl_tmp, "w", encoding="utf-8") as handle:
        for row in cleaned:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(jsonl_tmp, jsonl_path)
    table_sha = sha256_file(jsonl_path)

    fieldnames = [
        "module_name",
        "category",
        "module_suffix",
        "block_index",
        "mode",
        "nominal_bit",
        "param_count",
        "mean_delta_kl",
        "std_delta_kl",
        "standard_error_delta_kl",
        "kl_mode",
        "metric_name",
        "teacher_topk",
    ]
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(csv_tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in cleaned:
            writer.writerow({k: row.get(k) for k in fieldnames})
    os.replace(csv_tmp, csv_path)

    summary = _build_summary(
        cleaned,
        inventory=inventory,
        resolved=resolved,
        source_job_count=source_job_count,
        baseline_mode=baseline_mode,
    )
    md_tmp = md_path.with_name(md_path.name + ".tmp")
    with open(md_tmp, "w", encoding="utf-8") as handle:
        handle.write(summary["markdown"])
    os.replace(md_tmp, md_path)

    meta = {
        "kind": "mix_bit_cost_table_meta",
        "row_count": len(cleaned),
        "C": counts["C"],
        "L": counts["L"],
        "R": counts["R"],
        "source_job_count": int(source_job_count),
        "non_baseline_module_evaluation_count": counts["non_baseline_module_evaluation_count"],
        "complete_row_count": counts["complete_row_count"],
        "formulas": counts["formulas"],
        "kl_mode": expected_provenance["kl_mode"],
        "metric_name": expected_provenance["metric_name"],
        "teacher_topk": expected_provenance.get("teacher_topk"),
        "run_config_sha256": expected_provenance["run_config_sha256"],
        "model_inventory_sha256": expected_provenance["model_inventory_sha256"],
        "candidate_manifest_sha256": expected_provenance["candidate_manifest_sha256"],
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "calibration_manifest_sha256": expected_provenance["calibration_manifest_sha256"],
        "baseline_overlay_sha256": expected_provenance["baseline_overlay_sha256"],
        "teacher_cache_index_sha256": expected_provenance.get("teacher_cache_index_sha256", ""),
        "baseline_kl_mean": float(baseline_kl_mean),
        "cost_table_sha256": table_sha,
        "self_swap_audit_sha256": self_swap_audit.get("audit_sha256"),
        "summary": summary["stats"],
    }
    write_json_atomic(meta_path, meta)
    return {
        "cost_table_jsonl": str(jsonl_path.resolve()),
        "cost_table_csv": str(csv_path.resolve()),
        "cost_table_summary_md": str(md_path.resolve()),
        "cost_table_meta": str(meta_path.resolve()),
        "cost_table_sha256": table_sha,
        "row_count": len(cleaned),
        "summary": summary["stats"],
    }


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    inventory: ModelInventory,
    resolved: ResolvedRunConfig,
    source_job_count: int,
    baseline_mode: str,
) -> dict[str, Any]:
    total_params = float(inventory.total_target_parameters) or 1.0
    cat_params: dict[str, int] = {c: 0 for c in inventory.category_order}
    mode_params: dict[str, int] = {m.name: 0 for m in resolved.config.candidate_space.modes}
    # Parameter shares: use each module once (any mode) for category; mode share uses rows.
    seen_modules: set[str] = set()
    for row in rows:
        name = str(row["module_name"])
        if name not in seen_modules:
            cat_params[str(row["category"])] += int(row["param_count"])
            seen_modules.add(name)
        mode_params[str(row["mode"])] += int(row["param_count"])

    deltas = np.asarray([float(r["mean_delta_kl"]) for r in rows], dtype=np.float64)
    non_baseline = [r for r in rows if r["mode"] != baseline_mode]
    nb_deltas = np.asarray([float(r["mean_delta_kl"]) for r in non_baseline], dtype=np.float64)
    negative_count = int(np.sum(nb_deltas < 0)) if nb_deltas.size else 0

    best_by_bit: dict[str, dict[str, Any]] = {}
    best_by_cat: dict[str, dict[str, Any]] = {}
    for row in non_baseline:
        bit_key = str(row["nominal_bit"])
        if bit_key not in best_by_bit or float(row["mean_delta_kl"]) < float(
            best_by_bit[bit_key]["mean_delta_kl"]
        ):
            best_by_bit[bit_key] = {
                "module_name": row["module_name"],
                "mode": row["mode"],
                "mean_delta_kl": row["mean_delta_kl"],
            }
        cat = str(row["category"])
        if cat not in best_by_cat or float(row["mean_delta_kl"]) < float(
            best_by_cat[cat]["mean_delta_kl"]
        ):
            best_by_cat[cat] = {
                "module_name": row["module_name"],
                "mode": row["mode"],
                "mean_delta_kl": row["mean_delta_kl"],
            }

    extremes = {}
    if nb_deltas.size:
        amin = int(np.argmin(nb_deltas))
        amax = int(np.argmax(nb_deltas))
        extremes = {
            "min": {
                "module_name": non_baseline[amin]["module_name"],
                "mode": non_baseline[amin]["mode"],
                "mean_delta_kl": float(nb_deltas[amin]),
            },
            "max": {
                "module_name": non_baseline[amax]["module_name"],
                "mode": non_baseline[amax]["mode"],
                "mean_delta_kl": float(nb_deltas[amax]),
            },
        }

    baseline_rows = [r for r in rows if r["mode"] == baseline_mode]
    all_baseline_zero = all(float(r["mean_delta_kl"]) == 0.0 for r in baseline_rows)

    stats = {
        "total_modules": len(inventory.targets),
        "total_rows": len(rows),
        "total_jobs": int(source_job_count),
        "category_parameter_shares": {
            k: float(v) / total_params for k, v in cat_params.items()
        },
        "mode_count": len(resolved.config.candidate_space.modes),
        "mode_parameter_shares": {
            k: float(v) / (total_params * len(resolved.config.candidate_space.modes))
            for k, v in mode_params.items()
        },
        "mean_delta_kl": float(deltas.mean()) if deltas.size else 0.0,
        "median_delta_kl": float(np.median(deltas)) if deltas.size else 0.0,
        "best_candidate_per_nominal_bit": best_by_bit,
        "best_candidate_per_category": best_by_cat,
        "negative_cost_count": negative_count,
        "extremes": extremes,
        "all_baseline_rows_exactly_zero": all_baseline_zero,
    }

    lines = [
        "# Cost table summary",
        "",
        f"- total_modules: {stats['total_modules']}",
        f"- total_rows: {stats['total_rows']}",
        f"- total_jobs: {stats['total_jobs']}",
        f"- mean_delta_kl: {stats['mean_delta_kl']}",
        f"- median_delta_kl: {stats['median_delta_kl']}",
        f"- negative_cost_count: {stats['negative_cost_count']}",
        f"- all_baseline_rows_exactly_zero: {stats['all_baseline_rows_exactly_zero']}",
        "",
        "## Category parameter shares",
    ]
    for cat, share in stats["category_parameter_shares"].items():
        lines.append(f"- {cat}: {share:.6f}")
    lines.append("")
    lines.append("## Mode parameter shares")
    for mode, share in stats["mode_parameter_shares"].items():
        lines.append(f"- {mode}: {share:.6f}")
    return {"stats": stats, "markdown": "\n".join(lines) + "\n"}


def parse_recompute_specs(specs: Sequence[str] | None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if not specs:
        return out
    for item in specs:
        if ":" not in item:
            raise ValueError(f"--recompute expects module_name:mode, got {item!r}")
        module_name, mode = item.rsplit(":", 1)
        if not module_name or not mode:
            raise ValueError(f"--recompute expects module_name:mode, got {item!r}")
        out.add((module_name, mode))
    return out


def parse_gpu_list(gpus: str) -> list[str]:
    parts = [p.strip() for p in str(gpus).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("--gpus must be a non-empty comma-separated GPU id list")
    return parts


@dataclass
class CostTableRunResult:
    cost_run_root: str
    source_job_count: int
    non_baseline_module_evaluation_count: int
    complete_row_count: int
    pending_job_count: int
    dry_run: bool
    finalized: bool
    meta_path: str | None = None


def compute_cost_table(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    baseline_overlay_path: str | Path,
    dataset_path: str | Path,
    dataset_manifest_path: str | Path,
    kl_mode: str,
    gpus: Sequence[str],
    teacher_topk: int | None = None,
    teacher_cache: str | Path | None = None,
    batch_size: int = 1,
    dry_run: bool = False,
    recompute: Sequence[str] | None = None,
    access_token: str | None = None,
    inventory_path: str | None = None,
    skip_finalize: bool = False,
) -> CostTableRunResult:
    contract = validate_cost_run_arguments(
        kl_mode=kl_mode,
        teacher_topk=teacher_topk,
        teacher_cache=teacher_cache,
    )
    cost_root = derive_cost_run_root(
        resolved, kl_mode=contract.kl_mode, teacher_topk=contract.teacher_topk
    )
    cost_root.mkdir(parents=True, exist_ok=True)

    manifest = plan_cost_jobs(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        cost_run_root=cost_root,
        baseline_overlay_path=baseline_overlay_path,
        dataset_manifest_path=dataset_manifest_path,
        kl_mode=contract.kl_mode,
        teacher_topk=contract.teacher_topk,
        teacher_cache=teacher_cache,
    )
    persist_jobs_manifest(cost_root, manifest)
    provenance = provenance_from_manifest(manifest)
    recompute_set = parse_recompute_specs(recompute)
    pending = pending_jobs(
        manifest,
        cost_root,
        expected_provenance=provenance,
        recompute=recompute_set,
    )

    print(f"cost_run_root={cost_root}")
    print(f"C={manifest['C']} L={manifest['L']} R={manifest['R']}")
    print(f"source_job_count={manifest['source_job_count']}")
    print(f"non_baseline_module_evaluation_count={manifest['non_baseline_module_evaluation_count']}")
    print(f"complete_row_count={manifest['complete_row_count']}")
    print(f"pending_jobs={len(pending)}")

    if dry_run:
        for job in manifest["jobs"]:
            print(
                f"job_id={job['job_id']} category={job['category']} mode={job['mode']} "
                f"modules={len(job['module_names'])}"
            )
        return CostTableRunResult(
            cost_run_root=str(cost_root),
            source_job_count=int(manifest["source_job_count"]),
            non_baseline_module_evaluation_count=int(
                manifest["non_baseline_module_evaluation_count"]
            ),
            complete_row_count=int(manifest["complete_row_count"]),
            pending_job_count=len(pending),
            dry_run=True,
            finalized=False,
        )

    # Ensure baseline per-sample exists before workers (first worker would also create it,
    # but parent stays CUDA-free: create via a temporary CPU path only when missing and
    # device allows — production workers create it on GPU).
    run_cost_search_scheduler(
        manifest=manifest,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        baseline_overlay_path=baseline_overlay_path,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        cost_run_root=cost_root,
        kl_mode=contract.kl_mode,
        teacher_topk=contract.teacher_topk,
        teacher_cache=teacher_cache,
        gpus=list(gpus),
        batch_size=batch_size,
        access_token=access_token,
        recompute=recompute_set,
        inventory_path=inventory_path,
    )

    if skip_finalize:
        return CostTableRunResult(
            cost_run_root=str(cost_root),
            source_job_count=int(manifest["source_job_count"]),
            non_baseline_module_evaluation_count=int(
                manifest["non_baseline_module_evaluation_count"]
            ),
            complete_row_count=int(manifest["complete_row_count"]),
            pending_job_count=0,
            dry_run=False,
            finalized=False,
        )

    # Finalization: self-swap audit + baseline zero rows on one worker context.
    # Parent would load CUDA here — use in-process CPU/GPU via create_cost_worker on
    # the first listed GPU by setting CUDA_VISIBLE_DEVICES temporarily only if needed.
    # For production, finalizer runs in the parent after workers exit; load on cuda:0
    # mapped from the first GPU id.
    first_gpu = list(gpus)[0]
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(first_gpu)
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        ctx = create_cost_worker(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            baseline_overlay_path=baseline_overlay_path,
            dataset_path=dataset_path,
            dataset_manifest_path=dataset_manifest_path,
            cost_run_root=cost_root,
            kl_mode=contract.kl_mode,
            teacher_topk=contract.teacher_topk,
            teacher_cache=teacher_cache,
            device=device,
            batch_size=batch_size,
            access_token=access_token,
        )
        audit = audit_baseline_self_swap(ctx)
        write_baseline_mode_zero_rows(ctx, audit=audit)
        rows = load_valid_atomic_rows(
            cost_root,
            inventory=inventory,
            resolved=resolved,
            expected_provenance=provenance,
            baseline_sample_ids=ctx.sample_ids,
        )
        result = finalize_cost_table(
            rows=rows,
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            cost_run_root=cost_root,
            expected_provenance=provenance,
            self_swap_audit=audit,
            source_job_count=int(manifest["source_job_count"]),
            baseline_kl_mean=float(ctx.baseline_kl.mean()),
        )
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev

    return CostTableRunResult(
        cost_run_root=str(cost_root),
        source_job_count=int(manifest["source_job_count"]),
        non_baseline_module_evaluation_count=int(
            manifest["non_baseline_module_evaluation_count"]
        ),
        complete_row_count=int(manifest["complete_row_count"]),
        pending_job_count=0,
        dry_run=False,
        finalized=True,
        meta_path=result["cost_table_meta"],
    )
