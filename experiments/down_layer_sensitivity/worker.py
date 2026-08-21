from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import transformers

from experiments.down_layer_sensitivity.core import (
    assert_down_original_devices,
    assert_down_restore_set,
    compute_down_weight_metrics,
    hoist_down_original_weights,
    load_worker_model,
    pin_down_original_weights_to_cpu,
    reset_all_vae_to_compressed,
    set_down_restore_set,
)
from experiments.down_layer_sensitivity.mmlu_eval import (
    build_tokenizer,
    evaluate_mmlu,
    extract_subject_metrics,
)

SEED = 31
PREWARM_GROUP_SIZE = 8
LOGICAL_DEVICE = "cuda:0"
ALLOWED_MODES = {"smoke", "formal"}
LAYER_INDEX_MIN = 0
LAYER_INDEX_MAX = 35
SMOKE_LM_LIMIT = 2
WEIGHT_METRICS_FILENAME = "weight_metrics_worker.json"


def validate_manifest(manifest: dict, *, worker_id: int, physical_gpu_id: str) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a dict.")

    if manifest.get("worker_id") != worker_id:
        raise ValueError(
            f"CLI worker_id={worker_id!r} does not match manifest worker_id={manifest.get('worker_id')!r}."
        )
    if manifest.get("physical_gpu_id") != physical_gpu_id:
        raise ValueError(
            f"CLI physical_gpu_id={physical_gpu_id!r} does not match "
            f"manifest physical_gpu_id={manifest.get('physical_gpu_id')!r}."
        )

    mode = manifest.get("mode")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"mode not in {{smoke, formal}}: {mode!r}")

    write_weight_metrics = manifest.get("write_weight_metrics")
    if type(write_weight_metrics) is not bool:
        raise ValueError("write_weight_metrics must be a bool.")

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Manifest jobs must be a list.")

    seen_job_ids: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each job must be a dict.")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"job_id must be a non-empty string, got {job_id!r}.")
        if job_id in seen_job_ids:
            raise ValueError(f"duplicate job_id: {job_id}")
        seen_job_ids.add(job_id)

        job_mode = job.get("mode")
        if job_mode not in ALLOWED_MODES:
            raise ValueError(f"mode not in {{smoke, formal}}: {job_mode!r}")
        if job_mode != mode:
            raise ValueError(
                f"job {job_id!r} mode={job_mode!r} does not match manifest mode={mode!r}."
            )

        if job_mode == "formal":
            if job.get("lm_limit") is not None:
                raise ValueError(f"formal job {job_id!r} requires lm_limit=None.")
        elif job.get("lm_limit") != SMOKE_LM_LIMIT:
            raise ValueError(f"smoke job {job_id!r} requires lm_limit={SMOKE_LM_LIMIT}.")

        restore_layers = job.get("restore_layers")
        if not isinstance(restore_layers, list):
            raise ValueError(f"job {job_id!r} restore_layers must be a list.")
        seen_layers: set[int] = set()
        for layer in restore_layers:
            if type(layer) is not int:
                raise ValueError(f"job {job_id!r} restore layer must be int, got {layer!r}.")
            if layer < LAYER_INDEX_MIN or layer > LAYER_INDEX_MAX:
                raise ValueError(
                    f"restore layer outside 0..35: {layer} (job {job_id!r})"
                )
            if layer in seen_layers:
                raise ValueError(f"duplicate layer in restore list: {layer} (job {job_id!r})")
            seen_layers.add(layer)


def _set_inference_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _lm_eval_version() -> str | None:
    try:
        import lm_eval
    except ImportError:
        return None
    version = getattr(lm_eval, "__version__", None)
    return None if version is None else str(version)


def _dump_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _collect_worker_metadata(
    *,
    worker_id: int,
    physical_gpu_id: str,
    checkpoint_dir: str,
    checkpoint_meta: dict[str, Any],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Worker requires CUDA; logical device is cuda:0.")
    return {
        "worker_id": worker_id,
        "physical_gpu_id": physical_gpu_id,
        "logical_device": LOGICAL_DEVICE,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "seed": SEED,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "lm_eval_version": _lm_eval_version(),
        "checkpoint_dir": checkpoint_dir,
        "base_model_path_from_checkpoint_meta": checkpoint_meta.get("base_model_path"),
    }


def _execute_job(
    *,
    model,
    tokenizer,
    checkpoint_dir: str,
    down_layers,
    job: dict[str, Any],
    worker_id: int,
    physical_gpu_id: str,
    device_name: str,
    prewarm_stats: dict[str, Any],
    device,
) -> dict[str, Any]:
    restore_layers = [int(layer) for layer in job["restore_layers"]]
    restore_set = set(restore_layers)

    reset_all_vae_to_compressed(model)
    pin_down_original_weights_to_cpu(down_layers)
    assert_down_restore_set(down_layers, set())
    assert_down_original_devices(down_layers, set(), device)
    set_down_restore_set(down_layers, restore_set)
    hoist_down_original_weights(down_layers, restore_set, device)
    assert_down_restore_set(down_layers, restore_set)
    assert_down_original_devices(down_layers, restore_set, device)
    try:
        started = time.perf_counter()
        eval_result = evaluate_mmlu(
            model,
            tokenizer,
            checkpoint_dir,
            lm_limit=job["lm_limit"],
        )
        runtime_sec = time.perf_counter() - started
        accuracy = float(eval_result["accuracy"])
        return {
            "job_id": job["job_id"],
            "mode": job["mode"],
            "restore_layers": restore_layers,
            "accuracy": accuracy,
            "accuracy_percent": 100.0 * accuracy,
            "metric_key": eval_result["metric_key"],
            "subject_metrics": extract_subject_metrics(eval_result),
            "n_samples_total": int(eval_result["n_samples_total"]),
            "runtime_sec": runtime_sec,
            "worker_id": worker_id,
            "physical_gpu_id": physical_gpu_id,
            "device_name": device_name,
            "prewarm_stats": prewarm_stats,
        }
    finally:
        reset_all_vae_to_compressed(model)
        pin_down_original_weights_to_cpu(down_layers)
        assert_down_restore_set(down_layers, set())
        assert_down_original_devices(down_layers, set(), device)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--jobs_dir", required=True)
    parser.add_argument("--worker_meta_path", required=True)
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--physical_gpu_id", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    with open(args.manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    validate_manifest(
        manifest,
        worker_id=args.worker_id,
        physical_gpu_id=args.physical_gpu_id,
    )

    _set_inference_seeds()

    if not torch.cuda.is_available():
        raise RuntimeError("Worker requires CUDA; logical device is cuda:0.")
    device = torch.device(LOGICAL_DEVICE)

    loaded = load_worker_model(
        args.checkpoint_dir,
        device,
        PREWARM_GROUP_SIZE,
    )
    model = loaded["model"]
    checkpoint_meta = loaded["meta"]
    down_layers = loaded["down_layers"]
    prewarm_stats = loaded["prewarm_stats"]
    tokenizer = build_tokenizer(args.checkpoint_dir)

    if manifest["write_weight_metrics"]:
        weight_metrics_path = os.path.join(
            os.path.dirname(os.path.abspath(args.jobs_dir)),
            WEIGHT_METRICS_FILENAME,
        )
        if not os.path.isfile(weight_metrics_path):
            weight_metrics = compute_down_weight_metrics(down_layers)
            _dump_json(weight_metrics_path, weight_metrics)

    worker_meta = _collect_worker_metadata(
        worker_id=args.worker_id,
        physical_gpu_id=args.physical_gpu_id,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_meta=checkpoint_meta,
    )
    _dump_json(args.worker_meta_path, worker_meta)

    os.makedirs(args.jobs_dir, exist_ok=True)
    device_name = worker_meta["device_name"]
    for job in manifest["jobs"]:
        job_path = os.path.join(args.jobs_dir, f"{job['job_id']}.json")
        if os.path.isfile(job_path):
            continue
        try:
            result = _execute_job(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=args.checkpoint_dir,
                down_layers=down_layers,
                job=job,
                worker_id=args.worker_id,
                physical_gpu_id=args.physical_gpu_id,
                device_name=device_name,
                prewarm_stats=prewarm_stats,
                device=device,
            )
            _dump_json(job_path, result)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == "__main__":
    main()
