from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Any

SEED = 31
RANDOM_CONTROL_SEEDS = (31, 32, 33, 34, 35)
PREWARM_GROUP_SIZE = 8
FORMAL_LM_LIMIT = None
SMOKE_LM_LIMIT = 2
EXPECTED_DOWN_LAYERS = 36
HISTORICAL_COMPRESSED_MMLU = 0.4171
HISTORICAL_PRE_DOWN_MMLU = 0.5199

ALLOWED_MODES = {"smoke", "formal"}
WORKER_SCRIPT = "experiments/down_layer_sensitivity/worker.py"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_CONFIG_NAME = "run_config.json"


def _dump_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _update_run_config(run_dir: str, **updates: Any) -> None:
    path = os.path.join(run_dir, RUN_CONFIG_NAME)
    config = _load_json(path)
    config.update(updates)
    _dump_json(path, config)


def parse_selected_gpus(raw: str) -> list[str]:
    gpus = [part.strip() for part in raw.split(",")]
    if not gpus or any(not gpu for gpu in gpus):
        raise ValueError("--gpus must be a comma-separated list of physical GPU IDs.")
    return gpus


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpus", default=None, help="comma-separated physical GPU IDs")
    parser.add_argument("--mode", default=None, choices=sorted(ALLOWED_MODES))
    parser.add_argument(
        "--resume_run_dir",
        default=None,
        help="Resume an existing run directory (skips completed job JSON files).",
    )
    return parser.parse_args(argv)


def _make_job(*, job_id: str, restore_layers: list[int], mode: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "restore_layers": list(restore_layers),
        "mode": mode,
        "lm_limit": SMOKE_LM_LIMIT if mode == "smoke" else FORMAL_LM_LIMIT,
    }


def _make_manifest(
    *,
    worker_id: int,
    physical_gpu_id: str,
    mode: str,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
        "physical_gpu_id": physical_gpu_id,
        "mode": mode,
        "write_weight_metrics": worker_id == 0,
        "jobs": list(jobs),
    }


def _least_loaded_worker_id(job_lists: list[list[dict[str, Any]]]) -> int:
    return min(
        range(len(job_lists)),
        key=lambda worker_id: (len(job_lists[worker_id]), worker_id),
    )


def build_phase1_manifests(*, selected_gpus: list[str], mode: str) -> list[dict]:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"mode not in {{smoke, formal}}: {mode!r}")
    if not selected_gpus:
        raise ValueError("selected_gpus must be non-empty.")

    worker_count = len(selected_gpus)
    if mode == "smoke" and worker_count != 1:
        raise ValueError("smoke mode allows exactly one GPU (W=1).")

    if mode == "smoke":
        jobs = [
            _make_job(job_id="compressed_baseline_worker00", restore_layers=[], mode=mode),
            _make_job(
                job_id="compressed_baseline_worker00_repeat",
                restore_layers=[],
                mode=mode,
            ),
            _make_job(job_id="restore_L00", restore_layers=[0], mode=mode),
            _make_job(
                job_id="all_down_original",
                restore_layers=list(range(EXPECTED_DOWN_LAYERS)),
                mode=mode,
            ),
        ]
        return [
            _make_manifest(
                worker_id=0,
                physical_gpu_id=selected_gpus[0],
                mode=mode,
                jobs=jobs,
            )
        ]

    worker_jobs: list[list[dict[str, Any]]] = []
    for worker_id in range(worker_count):
        jobs = [
            _make_job(
                job_id=f"compressed_baseline_worker{worker_id:02d}",
                restore_layers=[],
                mode=mode,
            )
        ]
        if worker_id == 0:
            jobs.append(
                _make_job(
                    job_id="compressed_baseline_worker00_repeat",
                    restore_layers=[],
                    mode=mode,
                )
            )
        worker_jobs.append(jobs)

    scientific_jobs = [
        _make_job(
            job_id="all_down_original",
            restore_layers=list(range(EXPECTED_DOWN_LAYERS)),
            mode=mode,
        )
    ]
    for layer_idx in range(EXPECTED_DOWN_LAYERS):
        scientific_jobs.append(
            _make_job(
                job_id=f"restore_L{layer_idx:02d}",
                restore_layers=[layer_idx],
                mode=mode,
            )
        )

    for job in scientific_jobs:
        worker_jobs[_least_loaded_worker_id(worker_jobs)].append(job)

    return [
        _make_manifest(
            worker_id=worker_id,
            physical_gpu_id=selected_gpus[worker_id],
            mode=mode,
            jobs=worker_jobs[worker_id],
        )
        for worker_id in range(worker_count)
    ]


def launch_phase_workers(
    *,
    checkpoint_dir: str,
    phase_dir: str,
    selected_gpus: list[str],
    manifests: list[dict],
) -> None:
    if len(manifests) != len(selected_gpus):
        raise ValueError(
            f"manifest count {len(manifests)} does not match selected GPU count {len(selected_gpus)}."
        )

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    phase_dir = os.path.abspath(phase_dir)
    manifests_dir = os.path.join(phase_dir, "manifests")
    jobs_dir = os.path.join(phase_dir, "jobs")
    logs_dir = os.path.join(phase_dir, "worker_logs")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen]] = []
    for worker_id, (gpu, manifest) in enumerate(zip(selected_gpus, manifests)):
        if manifest.get("worker_id") != worker_id:
            raise ValueError(
                f"manifest worker_id={manifest.get('worker_id')!r} does not match launch index {worker_id}."
            )
        if manifest.get("physical_gpu_id") != gpu:
            raise ValueError(
                f"manifest physical_gpu_id={manifest.get('physical_gpu_id')!r} "
                f"does not match selected GPU {gpu!r}."
            )
        manifest_path = os.path.join(manifests_dir, f"worker_{worker_id:02d}.json")
        _dump_json(manifest_path, manifest)
        command = [
            sys.executable,
            WORKER_SCRIPT,
            "--checkpoint_dir",
            checkpoint_dir,
            "--manifest_path",
            manifest_path,
            "--jobs_dir",
            jobs_dir,
            "--worker_meta_path",
            os.path.join(logs_dir, f"worker_{worker_id:02d}_meta.json"),
            "--worker_id",
            str(worker_id),
            "--physical_gpu_id",
            gpu,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        processes.append(
            (
                worker_id,
                subprocess.Popen(command, env=env, cwd=REPO_ROOT),
            )
        )

    failed_workers: list[int] = []
    for worker_id, process in processes:
        if process.wait() != 0:
            failed_workers.append(worker_id)

    if failed_workers:
        run_dir = os.path.dirname(phase_dir)
        run_config_path = os.path.join(run_dir, RUN_CONFIG_NAME)
        if os.path.isfile(run_config_path):
            config = _load_json(run_config_path)
            config["status"] = "failed"
            config["failed_workers"] = failed_workers
            _dump_json(run_config_path, config)
        raise SystemExit(1)


def build_phase2_manifests(
    *,
    selected_gpus: list[str],
    ranked_layers: list[int],
) -> list[dict]:
    if not selected_gpus:
        raise ValueError("selected_gpus must be non-empty.")
    ranked = list(ranked_layers)
    if len(ranked) != EXPECTED_DOWN_LAYERS or sorted(ranked) != list(range(EXPECTED_DOWN_LAYERS)):
        raise ValueError(
            "ranked_layers must be a permutation of "
            f"0..{EXPECTED_DOWN_LAYERS - 1}, got length {len(ranked)} values {ranked!r}."
        )

    w2 = min(len(selected_gpus), 9)
    phase2_gpus = selected_gpus[:w2]
    worker_jobs: list[list[dict[str, Any]]] = []
    for worker_id in range(w2):
        jobs = [
            _make_job(
                job_id=f"compressed_baseline_worker{worker_id:02d}",
                restore_layers=[],
                mode="formal",
            )
        ]
        if worker_id == 0:
            jobs.append(
                _make_job(
                    job_id="compressed_baseline_worker00_repeat",
                    restore_layers=[],
                    mode="formal",
                )
            )
        worker_jobs.append(jobs)

    scientific_jobs = [
        _make_job(job_id="top2", restore_layers=ranked[:2], mode="formal"),
        _make_job(job_id="top4", restore_layers=ranked[:4], mode="formal"),
        _make_job(job_id="top8", restore_layers=ranked[:8], mode="formal"),
        _make_job(job_id="top12", restore_layers=ranked[:12], mode="formal"),
    ]
    for seed in RANDOM_CONTROL_SEEDS:
        rng = random.Random(seed)
        restore_layers = sorted(rng.sample(list(range(EXPECTED_DOWN_LAYERS)), 8))
        scientific_jobs.append(
            _make_job(
                job_id=f"random8_seed{seed}",
                restore_layers=restore_layers,
                mode="formal",
            )
        )

    for job in scientific_jobs:
        worker_jobs[_least_loaded_worker_id(worker_jobs)].append(job)

    manifests = []
    for worker_id in range(w2):
        manifest = _make_manifest(
            worker_id=worker_id,
            physical_gpu_id=phase2_gpus[worker_id],
            mode="formal",
            jobs=worker_jobs[worker_id],
        )
        manifest["write_weight_metrics"] = False
        manifests.append(manifest)
    return manifests


def _write_initial_run_config(
    *,
    run_dir: str,
    run_id: str,
    checkpoint_dir: str,
    output_dir: str,
    selected_gpus: list[str],
    mode: str,
) -> None:
    _dump_json(
        os.path.join(run_dir, RUN_CONFIG_NAME),
        {
            "run_id": run_id,
            "checkpoint_dir": checkpoint_dir,
            "output_dir": output_dir,
            "selected_gpus": selected_gpus,
            "mode": mode,
            "seed": SEED,
            "random_control_seeds": list(RANDOM_CONTROL_SEEDS),
            "prewarm_group_size": PREWARM_GROUP_SIZE,
            "formal_lm_limit": FORMAL_LM_LIMIT,
            "smoke_lm_limit": SMOKE_LM_LIMIT,
            "expected_down_layers": EXPECTED_DOWN_LAYERS,
            "historical_compressed_mmlu": HISTORICAL_COMPRESSED_MMLU,
            "historical_pre_down_mmlu": HISTORICAL_PRE_DOWN_MMLU,
            "phase1_worker_count": len(selected_gpus),
            "status": "running",
        },
    )


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.resume_run_dir:
        run_dir = os.path.abspath(args.resume_run_dir)
        config_path = os.path.join(run_dir, RUN_CONFIG_NAME)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing run config: {config_path}")
        config = _load_json(config_path)
        selected_gpus = [str(gpu) for gpu in config["selected_gpus"]]
        mode = str(config["mode"])
        checkpoint_dir = str(config["checkpoint_dir"])
        output_dir = str(config["output_dir"])
        run_id = str(config["run_id"])
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported mode in run config: {mode!r}")
        _update_run_config(run_dir, status="running")
    else:
        if not args.checkpoint_dir or not args.output_dir or not args.gpus or not args.mode:
            raise ValueError(
                "Without --resume_run_dir, --checkpoint_dir, --output_dir, --gpus, and --mode are required."
            )
        selected_gpus = parse_selected_gpus(args.gpus)
        mode = args.mode
        checkpoint_dir = args.checkpoint_dir
        output_dir = args.output_dir
        run_id = datetime.now().strftime(f"%Y%m%d_%H%M%S_{mode}")
        run_dir = os.path.join(output_dir, run_id)
        os.makedirs(run_dir, exist_ok=False)
        _write_initial_run_config(
            run_dir=run_dir,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            selected_gpus=selected_gpus,
            mode=mode,
        )

    phase1_dir = os.path.join(run_dir, "phase1")
    phase1_jobs_dir = os.path.join(phase1_dir, "jobs")
    expected_phase1_jobs = 38 + len(selected_gpus) if mode == "formal" else 4
    existing_phase1 = (
        {
            name[:-5]
            for name in os.listdir(phase1_jobs_dir)
            if name.endswith(".json")
        }
        if os.path.isdir(phase1_jobs_dir)
        else set()
    )
    if len(existing_phase1) < expected_phase1_jobs:
        phase1_manifests = build_phase1_manifests(selected_gpus=selected_gpus, mode=mode)
        launch_phase_workers(
            checkpoint_dir=checkpoint_dir,
            phase_dir=phase1_dir,
            selected_gpus=selected_gpus,
            manifests=phase1_manifests,
        )

    if mode == "smoke":
        from experiments.down_layer_sensitivity.summarize import validate_smoke

        validate_smoke(run_dir=run_dir, selected_gpus=selected_gpus)
        _update_run_config(run_dir, status="smoke_completed")
        return

    from experiments.down_layer_sensitivity.summarize import summarize_final, summarize_phase1

    phase2_dir = os.path.join(run_dir, "phase2")

    try:
        ranked_layers = summarize_phase1(run_dir=run_dir, selected_gpus=selected_gpus)
        phase2_manifests = build_phase2_manifests(
            selected_gpus=selected_gpus,
            ranked_layers=ranked_layers,
        )
    except Exception:
        _update_run_config(run_dir, status="failed")
        raise

    w2 = min(len(selected_gpus), 9)
    phase2_gpus = selected_gpus[:w2]
    _update_run_config(run_dir, phase2_worker_count=len(phase2_manifests))

    # Always (re)launch phase2 workers; they skip completed job JSON files.
    launch_phase_workers(
        checkpoint_dir=checkpoint_dir,
        phase_dir=phase2_dir,
        selected_gpus=phase2_gpus,
        manifests=phase2_manifests,
    )
    try:
        summarize_final(run_dir=run_dir, selected_gpus=selected_gpus)
    except Exception:
        _update_run_config(run_dir, status="failed")
        raise
    _update_run_config(run_dir, status="completed")


if __name__ == "__main__":
    main()
