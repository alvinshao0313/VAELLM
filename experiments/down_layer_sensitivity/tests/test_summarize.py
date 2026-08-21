from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.down_layer_sensitivity.summarize import (
    spearman_rank_correlation,
    summarize_final,
    summarize_phase1,
    validate_smoke,
)

EXPECTED_DOWN_LAYERS = 36
DEVICE_A = "NVIDIA H100 80GB HBM3"
DEVICE_B = "NVIDIA A100-SXM4-80GB"
HISTORICAL_COMPRESSED_MMLU = 0.4171
HISTORICAL_PRE_DOWN_MMLU = 0.5199

SENSITIVITY_CSV_COLUMNS = [
    "rank",
    "layer_idx",
    "module_name",
    "mmlu_accuracy",
    "mmlu_accuracy_percent",
    "delta_mmlu_pp",
    "single_recovery_fraction",
    "weight_mse",
    "weight_nmse",
    "relative_fro_error",
    "original_rms",
    "error_rms",
    "subjects_improved",
    "subjects_worsened",
    "subjects_unchanged",
    "median_subject_delta_pp",
    "max_subject_gain_pp",
    "max_subject_drop_pp",
]


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _subject_metrics(acc_a: float, acc_b: float, acc_c: float, *, samples=(10, 20, 30)):
    return [
        {
            "subject_name": "mmlu_a",
            "metric_key": "acc,none",
            "accuracy": acc_a,
            "samples": samples[0],
        },
        {
            "subject_name": "mmlu_b",
            "metric_key": "acc,none",
            "accuracy": acc_b,
            "samples": samples[1],
        },
        {
            "subject_name": "mmlu_c",
            "metric_key": "acc,none",
            "accuracy": acc_c,
            "samples": samples[2],
        },
    ]


def _job_payload(
    *,
    job_id: str,
    accuracy: float,
    restore_layers: list[int],
    worker_id: int = 0,
    physical_gpu_id: str = "0",
    device_name: str = DEVICE_A,
    mode: str = "formal",
    subject_metrics=None,
    n_samples_total: int | None = None,
) -> dict:
    if subject_metrics is None:
        subject_metrics = _subject_metrics(0.4, 0.5, 0.6)
    if n_samples_total is None:
        n_samples_total = sum(int(row["samples"]) for row in subject_metrics)
    return {
        "job_id": job_id,
        "mode": mode,
        "restore_layers": list(restore_layers),
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "metric_key": "acc,none",
        "subject_metrics": subject_metrics,
        "n_samples_total": n_samples_total,
        "runtime_sec": 1.0,
        "worker_id": worker_id,
        "physical_gpu_id": physical_gpu_id,
        "device_name": device_name,
        "prewarm_stats": {"failed": 0},
    }


def _layer_accuracy(layer_idx: int) -> float:
    if layer_idx == 20:
        return 0.48
    if layer_idx in {7, 10}:
        return 0.45
    if layer_idx == 3:
        return 0.42
    return 0.401


def _layer_subjects(layer_idx: int):
    if layer_idx == 20:
        return _subject_metrics(0.5, 0.5, 0.5)
    return _subject_metrics(0.4, 0.5, 0.6)


def _weight_metrics_rows() -> list[dict]:
    rows = []
    for layer_idx in range(EXPECTED_DOWN_LAYERS):
        nmse = 0.01 * (layer_idx + 1)
        rows.append(
            {
                "layer_idx": layer_idx,
                "name": f"model.layers.{layer_idx}.mlp.down_proj",
                "numel": 16,
                "mse": 0.001 * (layer_idx + 1),
                "nmse": nmse,
                "relative_fro_error": math.sqrt(nmse),
                "original_rms": 1.0,
                "error_rms": 0.1,
            }
        )
    return rows


def _write_run_config(run_dir: Path, *, selected_gpus: list[str], mode: str) -> None:
    _dump_json(
        run_dir / "run_config.json",
        {
            "run_id": run_dir.name,
            "selected_gpus": selected_gpus,
            "mode": mode,
            "status": "running",
            "historical_compressed_mmlu": HISTORICAL_COMPRESSED_MMLU,
            "historical_pre_down_mmlu": HISTORICAL_PRE_DOWN_MMLU,
        },
    )


def _write_job(run_dir: Path, payload: dict) -> None:
    _dump_json(run_dir / "phase1" / "jobs" / f"{payload['job_id']}.json", payload)


def write_formal_phase1(
    run_dir: Path,
    *,
    selected_gpus: list[str] | None = None,
    baseline_accuracy: float = 0.40,
    all_down_accuracy: float = 0.50,
    layer_accuracy_fn=_layer_accuracy,
    layer_subjects_fn=_layer_subjects,
    device_name: str = DEVICE_A,
    job_device_overrides: dict[str, str] | None = None,
    skip_job_ids: set[str] | None = None,
    extra_jobs: list[dict] | None = None,
    mutate_jobs: dict[str, dict] | None = None,
    weight_metrics: list[dict] | None = None,
    mode: str = "formal",
) -> list[str]:
    if selected_gpus is None:
        selected_gpus = ["0", "1"]
    job_device_overrides = job_device_overrides or {}
    skip_job_ids = skip_job_ids or set()
    mutate_jobs = mutate_jobs or {}
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(run_dir, selected_gpus=selected_gpus, mode=mode)
    _dump_json(
        run_dir / "phase1" / "weight_metrics_worker.json",
        weight_metrics if weight_metrics is not None else _weight_metrics_rows(),
    )

    jobs: list[dict] = []
    for worker_id, gpu in enumerate(selected_gpus):
        jobs.append(
            _job_payload(
                job_id=f"compressed_baseline_worker{worker_id:02d}",
                accuracy=baseline_accuracy,
                restore_layers=[],
                worker_id=worker_id,
                physical_gpu_id=gpu,
                device_name=device_name,
                mode=mode,
            )
        )
    jobs.append(
        _job_payload(
            job_id="compressed_baseline_worker00_repeat",
            accuracy=baseline_accuracy,
            restore_layers=[],
            worker_id=0,
            physical_gpu_id=selected_gpus[0],
            device_name=device_name,
            mode=mode,
        )
    )
    jobs.append(
        _job_payload(
            job_id="all_down_original",
            accuracy=all_down_accuracy,
            restore_layers=list(range(EXPECTED_DOWN_LAYERS)),
            worker_id=0,
            physical_gpu_id=selected_gpus[0],
            device_name=device_name,
            mode=mode,
        )
    )
    for layer_idx in range(EXPECTED_DOWN_LAYERS):
        jobs.append(
            _job_payload(
                job_id=f"restore_L{layer_idx:02d}",
                accuracy=layer_accuracy_fn(layer_idx),
                restore_layers=[layer_idx],
                worker_id=0,
                physical_gpu_id=selected_gpus[0],
                device_name=device_name,
                mode=mode,
                subject_metrics=layer_subjects_fn(layer_idx),
            )
        )
    if extra_jobs:
        jobs.extend(extra_jobs)

    written_ids = []
    for job in jobs:
        job_id = job["job_id"]
        if job_id in skip_job_ids:
            continue
        if job_id in job_device_overrides:
            job["device_name"] = job_device_overrides[job_id]
        if job_id in mutate_jobs:
            job.update(mutate_jobs[job_id])
        _write_job(run_dir, job)
        written_ids.append(job_id)
    return written_ids


def write_smoke_phase1(
    run_dir: Path,
    *,
    baseline_accuracy: float = 0.40,
    all_down_accuracy: float = 0.41,
    restore_l00_accuracy: float = 0.405,
) -> None:
    selected_gpus = ["0"]
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(run_dir, selected_gpus=selected_gpus, mode="smoke")
    _dump_json(run_dir / "phase1" / "weight_metrics_worker.json", _weight_metrics_rows())
    jobs = [
        _job_payload(
            job_id="compressed_baseline_worker00",
            accuracy=baseline_accuracy,
            restore_layers=[],
            mode="smoke",
        ),
        _job_payload(
            job_id="compressed_baseline_worker00_repeat",
            accuracy=baseline_accuracy,
            restore_layers=[],
            mode="smoke",
        ),
        _job_payload(
            job_id="restore_L00",
            accuracy=restore_l00_accuracy,
            restore_layers=[0],
            mode="smoke",
        ),
        _job_payload(
            job_id="all_down_original",
            accuracy=all_down_accuracy,
            restore_layers=list(range(EXPECTED_DOWN_LAYERS)),
            mode="smoke",
        ),
    ]
    for job in jobs:
        _write_job(run_dir, job)


def test_spearman_rank_correlation_monotonic_is_one():
    rho = spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert rho == pytest.approx(1.0)


def test_spearman_rank_correlation_reverse_is_minus_one():
    rho = spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
    assert rho == pytest.approx(-1.0)


def test_spearman_rank_correlation_average_ranks_for_ties():
    rho = spearman_rank_correlation([1.0, 1.0, 2.0], [10.0, 10.0, 20.0])
    assert rho == pytest.approx(1.0)


def test_summarize_phase1_computes_delta_recovery_rank_and_subject_diagnostics(tmp_path):
    run_dir = tmp_path / "formal_ok"
    selected_gpus = ["0", "1"]
    write_formal_phase1(run_dir, selected_gpus=selected_gpus)

    ranked = summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)

    a_compressed = 0.40
    a_all = 0.50
    down_gap = 100.0 * (a_all - a_compressed)
    expected_special = {
        20: 0.48,
        7: 0.45,
        10: 0.45,
        3: 0.42,
    }
    expected_rows = []
    for layer_idx in range(EXPECTED_DOWN_LAYERS):
        accuracy = expected_special.get(layer_idx, 0.401)
        expected_rows.append(
            {
                "layer_idx": layer_idx,
                "delta_mmlu_pp": 100.0 * (accuracy - a_compressed),
            }
        )
    expected_rows.sort(key=lambda row: (-row["delta_mmlu_pp"], row["layer_idx"]))
    expected_ranked = [row["layer_idx"] for row in expected_rows]
    assert ranked == expected_ranked
    assert ranked[0] == 20
    assert ranked[1] == 7
    assert ranked[2] == 10
    assert ranked[3] == 3

    summary_path = run_dir / "phase1_summary.json"
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)

    scientific_fields = [
        "compressed_baseline",
        "cross_gpu_baseline_probes",
        "worker00_baseline_repeat",
        "all_down_original",
        "down_gap_pp",
        "historical_reference",
        "spearman_weight_nmse_vs_delta_mmlu",
        "ranked_layers",
    ]
    for field in scientific_fields:
        assert field in summary
    assert summary["compressed_baseline"]["job_id"] == "compressed_baseline_worker00"
    assert summary["compressed_baseline"]["accuracy"] == pytest.approx(0.40)
    assert summary["compressed_baseline"]["n_samples_total"] == 60
    assert summary["compressed_baseline"]["device_name"] == DEVICE_A
    probes = summary["cross_gpu_baseline_probes"]
    assert len(probes) == 2
    assert [row["job_id"] for row in probes] == [
        "compressed_baseline_worker00",
        "compressed_baseline_worker01",
    ]
    for row in probes:
        assert row["accuracy"] == pytest.approx(0.40)
        assert row["device_name"] == DEVICE_A
        assert row["n_samples_total"] == 60
        assert "physical_gpu_id" in row
        assert "passed" not in row or row.get("accuracy") is not None
    assert summary["worker00_baseline_repeat"]["job_id"] == "compressed_baseline_worker00_repeat"
    assert summary["worker00_baseline_repeat"]["accuracy"] == pytest.approx(0.40)
    assert summary["all_down_original"]["accuracy"] == pytest.approx(0.50)
    assert summary["down_gap_pp"] == pytest.approx(down_gap)
    historical = summary["historical_reference"]
    assert historical["historical_compressed_mmlu"] == HISTORICAL_COMPRESSED_MMLU
    assert historical["historical_pre_down_mmlu"] == HISTORICAL_PRE_DOWN_MMLU
    assert historical["current_baseline_minus_historical_compressed_pp"] == pytest.approx(
        100.0 * (0.40 - HISTORICAL_COMPRESSED_MMLU)
    )
    assert historical["current_all_down_original_minus_historical_pre_down_pp"] == pytest.approx(
        100.0 * (0.50 - HISTORICAL_PRE_DOWN_MMLU)
    )
    assert [row["layer_idx"] for row in summary["ranked_layers"]] == expected_ranked

    csv_path = run_dir / "single_layer_sensitivity.csv"
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == SENSITIVITY_CSV_COLUMNS
        csv_rows = list(reader)
    assert len(csv_rows) == EXPECTED_DOWN_LAYERS
    assert [int(row["rank"]) for row in csv_rows] == list(range(1, EXPECTED_DOWN_LAYERS + 1))
    assert [int(row["layer_idx"]) for row in csv_rows] == expected_ranked

    top = csv_rows[0]
    assert int(top["layer_idx"]) == 20
    assert float(top["mmlu_accuracy"]) == pytest.approx(0.48)
    assert float(top["mmlu_accuracy_percent"]) == pytest.approx(48.0)
    assert float(top["delta_mmlu_pp"]) == pytest.approx(8.0)
    assert float(top["single_recovery_fraction"]) == pytest.approx(0.8)
    assert top["module_name"] == "model.layers.20.mlp.down_proj"
    assert float(top["weight_mse"]) == pytest.approx(0.021)
    assert float(top["weight_nmse"]) == pytest.approx(0.21)
    assert int(top["subjects_improved"]) == 1
    assert int(top["subjects_worsened"]) == 1
    assert int(top["subjects_unchanged"]) == 1
    assert float(top["median_subject_delta_pp"]) == pytest.approx(0.0)
    assert float(top["max_subject_gain_pp"]) == pytest.approx(10.0)
    assert float(top["max_subject_drop_pp"]) == pytest.approx(-10.0)

    tie_first = csv_rows[1]
    tie_second = csv_rows[2]
    assert int(tie_first["layer_idx"]) == 7
    assert int(tie_second["layer_idx"]) == 10
    assert float(tie_first["delta_mmlu_pp"]) == pytest.approx(float(tie_second["delta_mmlu_pp"]))
    assert float(tie_first["delta_mmlu_pp"]) == pytest.approx(5.0)
    assert float(tie_first["single_recovery_fraction"]) == pytest.approx(0.5)

    nmse = [0.01 * (layer_idx + 1) for layer_idx in range(EXPECTED_DOWN_LAYERS)]
    deltas = [
        100.0 * (_layer_accuracy(layer_idx) - a_compressed)
        for layer_idx in range(EXPECTED_DOWN_LAYERS)
    ]
    assert summary["spearman_weight_nmse_vs_delta_mmlu"] == pytest.approx(
        spearman_rank_correlation(nmse, deltas)
    )

    weight_csv_path = run_dir / "weight_metrics.csv"
    with open(weight_csv_path, newline="", encoding="utf-8") as handle:
        weight_rows = list(csv.DictReader(handle))
    assert len(weight_rows) == EXPECTED_DOWN_LAYERS
    assert [int(row["layer_idx"]) for row in weight_rows] == list(range(EXPECTED_DOWN_LAYERS))
    assert weight_rows[20]["name"] == "model.layers.20.mlp.down_proj"


def test_baseline_mismatch_fails_without_ranking(tmp_path):
    run_dir = tmp_path / "baseline_mismatch"
    selected_gpus = ["0", "1"]
    write_formal_phase1(
        run_dir,
        selected_gpus=selected_gpus,
        mutate_jobs={"compressed_baseline_worker01": {"accuracy": 0.4000001}},
    )
    with pytest.raises(ValueError, match="baseline"):
        summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "single_layer_sensitivity.csv").exists()
    assert "ranked_layers" not in json.loads((run_dir / "phase1_summary.json").read_text()) if (
        run_dir / "phase1_summary.json"
    ).exists() else True


def test_worker00_repeat_mismatch_fails_without_ranking(tmp_path):
    run_dir = tmp_path / "repeat_mismatch"
    selected_gpus = ["0", "1"]
    write_formal_phase1(
        run_dir,
        selected_gpus=selected_gpus,
        mutate_jobs={"compressed_baseline_worker00_repeat": {"accuracy": 0.41}},
    )
    with pytest.raises(ValueError, match="baseline"):
        summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "single_layer_sensitivity.csv").exists()


def test_sample_count_mismatch_fails_without_ranking(tmp_path):
    run_dir = tmp_path / "sample_mismatch"
    selected_gpus = ["0", "1"]
    write_formal_phase1(
        run_dir,
        selected_gpus=selected_gpus,
        mutate_jobs={
            "restore_L05": {
                "subject_metrics": _subject_metrics(0.4, 0.5, 0.6, samples=(11, 20, 30)),
                "n_samples_total": 61,
            }
        },
    )
    with pytest.raises(ValueError, match="population|sample"):
        summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "single_layer_sensitivity.csv").exists()


def test_all_original_not_greater_than_baseline_writes_diagnostic_and_stops(tmp_path):
    run_dir = tmp_path / "no_gap"
    selected_gpus = ["0", "1"]
    write_formal_phase1(
        run_dir,
        selected_gpus=selected_gpus,
        baseline_accuracy=0.50,
        all_down_accuracy=0.50,
    )
    with pytest.raises(ValueError, match="A_all_down_original <= A_compressed"):
        summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)

    summary_path = run_dir / "phase1_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["compressed_baseline"]["accuracy"] == pytest.approx(0.50)
    assert summary["all_down_original"]["accuracy"] == pytest.approx(0.50)
    assert summary["down_gap_pp"] == pytest.approx(0.0)
    assert "ranked_layers" not in summary
    assert "spearman_weight_nmse_vs_delta_mmlu" not in summary
    assert not (run_dir / "single_layer_sensitivity.csv").exists()


def test_heterogeneous_gpu_fails_without_ranking(tmp_path):
    run_dir = tmp_path / "hetero_gpu"
    selected_gpus = ["0", "1"]
    write_formal_phase1(
        run_dir,
        selected_gpus=selected_gpus,
        job_device_overrides={"compressed_baseline_worker01": DEVICE_B},
    )
    with pytest.raises(ValueError, match="homogeneous GPU"):
        summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "single_layer_sensitivity.csv").exists()


def test_missing_job_inventory_fails(tmp_path):
    run_dir = tmp_path / "missing_job"
    selected_gpus = ["0", "1"]
    write_formal_phase1(run_dir, selected_gpus=selected_gpus, skip_job_ids={"restore_L35"})
    with pytest.raises(ValueError, match="job"):
        summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "single_layer_sensitivity.csv").exists()


def test_smoke_validate_does_not_write_formal_ranking(tmp_path):
    run_dir = tmp_path / "smoke_ok"
    write_smoke_phase1(run_dir)
    validate_smoke(run_dir=str(run_dir), selected_gpus=["0"])
    assert not (run_dir / "single_layer_sensitivity.csv").exists()
    assert not (run_dir / "phase1_summary.json").exists()
    assert not (run_dir / "weight_metrics.csv").exists()


def test_summarize_phase1_rejects_smoke_results(tmp_path):
    run_dir = tmp_path / "smoke_no_rank"
    write_smoke_phase1(run_dir)
    with pytest.raises(ValueError, match="smoke"):
        summarize_phase1(run_dir=str(run_dir), selected_gpus=["0"])
    assert not (run_dir / "single_layer_sensitivity.csv").exists()


CUMULATIVE_CSV_COLUMNS = [
    "configuration",
    "num_restored_layers",
    "restore_layers",
    "mmlu_accuracy",
    "mmlu_accuracy_percent",
    "delta_from_compressed_pp",
    "recovery_fraction",
]
CUMULATIVE_ROW_ORDER = [
    "top1",
    "top2",
    "top4",
    "top8",
    "top12",
    "random8_seed31",
    "random8_seed32",
    "random8_seed33",
    "random8_seed34",
    "random8_seed35",
    "all_down_original",
]
PHASE2_RANDOM8_LAYERS = {
    31: [0, 1, 4, 7, 9, 21, 25, 30],
    32: [0, 4, 9, 13, 15, 19, 23, 31],
    33: [10, 11, 14, 17, 19, 20, 28, 30],
    34: [1, 11, 12, 14, 22, 28, 33, 35],
    35: [8, 9, 13, 21, 31, 33, 34, 35],
}
DEFAULT_PHASE2_ACCURACIES = {
    "top2": 0.46,
    "top4": 0.47,
    "top8": 0.49,
    "top12": 0.485,
    "random8_seed31": 0.42,
    "random8_seed32": 0.43,
    "random8_seed33": 0.41,
    "random8_seed34": 0.30,
    "random8_seed35": 0.55,
}


def _parse_restore_layers_cell(raw: str) -> list[int]:
    return json.loads(raw)


def write_formal_phase2(
    run_dir: Path,
    *,
    selected_gpus: list[str],
    ranked: list[int],
    accuracies: dict[str, float] | None = None,
    baseline_accuracy: float = 0.40,
    device_name: str = DEVICE_A,
    mutate_jobs: dict[str, dict] | None = None,
    skip_job_ids: set[str] | None = None,
    extra_jobs: list[dict] | None = None,
) -> None:
    accuracies = dict(DEFAULT_PHASE2_ACCURACIES if accuracies is None else accuracies)
    mutate_jobs = mutate_jobs or {}
    skip_job_ids = skip_job_ids or set()
    w2 = min(len(selected_gpus), 9)
    jobs: list[dict] = []
    for worker_id, gpu in enumerate(selected_gpus[:w2]):
        jobs.append(
            _job_payload(
                job_id=f"compressed_baseline_worker{worker_id:02d}",
                accuracy=baseline_accuracy,
                restore_layers=[],
                worker_id=worker_id,
                physical_gpu_id=gpu,
                device_name=device_name,
            )
        )
    jobs.append(
        _job_payload(
            job_id="compressed_baseline_worker00_repeat",
            accuracy=baseline_accuracy,
            restore_layers=[],
            worker_id=0,
            physical_gpu_id=selected_gpus[0],
            device_name=device_name,
        )
    )
    scientific = [
        ("top2", ranked[:2]),
        ("top4", ranked[:4]),
        ("top8", ranked[:8]),
        ("top12", ranked[:12]),
        ("random8_seed31", PHASE2_RANDOM8_LAYERS[31]),
        ("random8_seed32", PHASE2_RANDOM8_LAYERS[32]),
        ("random8_seed33", PHASE2_RANDOM8_LAYERS[33]),
        ("random8_seed34", PHASE2_RANDOM8_LAYERS[34]),
        ("random8_seed35", PHASE2_RANDOM8_LAYERS[35]),
    ]
    for job_id, restore_layers in scientific:
        jobs.append(
            _job_payload(
                job_id=job_id,
                accuracy=accuracies[job_id],
                restore_layers=restore_layers,
                worker_id=0,
                physical_gpu_id=selected_gpus[0],
                device_name=device_name,
            )
        )
    if extra_jobs:
        jobs.extend(extra_jobs)
    for job in jobs:
        job_id = job["job_id"]
        if job_id in skip_job_ids:
            continue
        if job_id in mutate_jobs:
            job.update(mutate_jobs[job_id])
        _dump_json(run_dir / "phase2" / "jobs" / f"{job_id}.json", job)


def test_summarize_final_writes_cumulative_csv_reusing_phase1_top1(tmp_path):
    run_dir = tmp_path / "phase2_ok"
    selected_gpus = ["0", "1"]
    write_formal_phase1(run_dir, selected_gpus=selected_gpus)
    ranked = summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    write_formal_phase2(run_dir, selected_gpus=selected_gpus, ranked=ranked)

    summarize_final(run_dir=str(run_dir), selected_gpus=selected_gpus)

    csv_path = run_dir / "cumulative_results.csv"
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CUMULATIVE_CSV_COLUMNS
        rows = list(reader)
    assert [row["configuration"] for row in rows] == CUMULATIVE_ROW_ORDER
    assert "random8_mean" not in [row["configuration"] for row in rows]

    by_name = {row["configuration"]: row for row in rows}
    a_compressed = 0.40
    a_all = 0.50
    expected = {
        "top1": (0.48, ranked[:1]),
        "top2": (0.46, ranked[:2]),
        "top4": (0.47, ranked[:4]),
        "top8": (0.49, ranked[:8]),
        "top12": (0.485, ranked[:12]),
        "random8_seed31": (0.42, PHASE2_RANDOM8_LAYERS[31]),
        "random8_seed32": (0.43, PHASE2_RANDOM8_LAYERS[32]),
        "random8_seed33": (0.41, PHASE2_RANDOM8_LAYERS[33]),
        "random8_seed34": (0.30, PHASE2_RANDOM8_LAYERS[34]),
        "random8_seed35": (0.55, PHASE2_RANDOM8_LAYERS[35]),
        "all_down_original": (0.50, list(range(EXPECTED_DOWN_LAYERS))),
    }
    for name, (accuracy, restore_layers) in expected.items():
        row = by_name[name]
        assert int(row["num_restored_layers"]) == len(restore_layers)
        assert _parse_restore_layers_cell(row["restore_layers"]) == restore_layers
        assert float(row["mmlu_accuracy"]) == pytest.approx(accuracy)
        assert float(row["mmlu_accuracy_percent"]) == pytest.approx(100.0 * accuracy)
        assert float(row["delta_from_compressed_pp"]) == pytest.approx(
            100.0 * (accuracy - a_compressed)
        )
        assert float(row["recovery_fraction"]) == pytest.approx(
            (accuracy - a_compressed) / (a_all - a_compressed)
        )

    assert ranked[0] == 20
    assert float(by_name["top1"]["recovery_fraction"]) == pytest.approx(0.8)
    assert float(by_name["top12"]["mmlu_accuracy"]) < float(by_name["top8"]["mmlu_accuracy"])
    assert float(by_name["random8_seed34"]["recovery_fraction"]) == pytest.approx(-1.0)
    assert float(by_name["random8_seed35"]["recovery_fraction"]) == pytest.approx(1.5)

    summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    random_accs = np.asarray([0.42, 0.43, 0.41, 0.30, 0.55], dtype=np.float64)
    random_recs = (random_accs - a_compressed) / (a_all - a_compressed)
    aggregate = summary["random8_aggregate"]
    assert aggregate["accuracy_mean"] == pytest.approx(float(np.mean(random_accs)))
    assert aggregate["accuracy_std"] == pytest.approx(float(np.std(random_accs, ddof=0)))
    assert aggregate["recovery_mean"] == pytest.approx(float(np.mean(random_recs)))
    assert aggregate["recovery_std"] == pytest.approx(float(np.std(random_recs, ddof=0)))
    assert aggregate["top8_minus_random8_mean_recovery"] == pytest.approx(
        0.9 - float(np.mean(random_recs))
    )
    assert [row["layer_idx"] for row in summary["ranked_layers"]] == ranked
    assert summary["topk"]["top1"]["restore_layers"] == ranked[:1]
    assert summary["topk"]["top8"]["recovery_fraction"] == pytest.approx(0.9)
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "plots" / "layer_sensitivity.png").is_file()
    assert (run_dir / "plots" / "nmse_vs_mmlu_sensitivity.png").is_file()
    assert (run_dir / "plots" / "cumulative_recovery.png").is_file()

    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    for section in (
        "## 1. 实验目标",
        "## 2. 固定实验设置",
        "## 3. 有效性检查",
        "## 4. Down 压缩总损失",
        "## 5. 36 层敏感度排名",
        "## 6. Weight NMSE 与 MMLU 敏感度关系",
        "## 7. Top-K 累积恢复",
        "## 8. Random-8 五组对照",
        "## 9. 结论与后续压缩建议",
    ):
        assert section in report_text
    assert "Spearman rho" in report_text
    assert "Top-8 layer IDs" in report_text
    assert "Random-8 recovery mean" in report_text
    assert str(ranked[:8]) in report_text


def test_summarize_final_phase2_baseline_mismatch_writes_no_csv(tmp_path):
    run_dir = tmp_path / "phase2_baseline_mismatch"
    selected_gpus = ["0", "1"]
    write_formal_phase1(run_dir, selected_gpus=selected_gpus)
    ranked = summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    write_formal_phase2(
        run_dir,
        selected_gpus=selected_gpus,
        ranked=ranked,
        mutate_jobs={"compressed_baseline_worker01": {"accuracy": 0.41}},
    )
    with pytest.raises(ValueError, match="baseline"):
        summarize_final(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "cumulative_results.csv").exists()
    assert not (run_dir / "final_summary.json").exists()
    assert not (run_dir / "report.md").exists()
    assert not (run_dir / "plots").exists()
    assert not (run_dir / "report.md").exists()
    assert not (run_dir / "plots").exists()


def test_summarize_final_phase2_population_mismatch_writes_no_csv(tmp_path):
    run_dir = tmp_path / "phase2_population_mismatch"
    selected_gpus = ["0", "1"]
    write_formal_phase1(run_dir, selected_gpus=selected_gpus)
    ranked = summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    write_formal_phase2(
        run_dir,
        selected_gpus=selected_gpus,
        ranked=ranked,
        mutate_jobs={
            "top8": {
                "subject_metrics": _subject_metrics(0.4, 0.5, 0.6, samples=(11, 20, 30)),
                "n_samples_total": 61,
            }
        },
    )
    with pytest.raises(ValueError, match="population|sample"):
        summarize_final(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "cumulative_results.csv").exists()
    assert not (run_dir / "final_summary.json").exists()
    assert not (run_dir / "report.md").exists()
    assert not (run_dir / "plots").exists()


def test_summarize_final_phase2_device_mismatch_writes_no_csv(tmp_path):
    run_dir = tmp_path / "phase2_device_mismatch"
    selected_gpus = ["0", "1"]
    write_formal_phase1(run_dir, selected_gpus=selected_gpus)
    ranked = summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    write_formal_phase2(
        run_dir,
        selected_gpus=selected_gpus,
        ranked=ranked,
        mutate_jobs={"top4": {"device_name": DEVICE_B}},
    )
    with pytest.raises(ValueError, match="device"):
        summarize_final(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "cumulative_results.csv").exists()
    assert not (run_dir / "final_summary.json").exists()
    assert not (run_dir / "report.md").exists()
    assert not (run_dir / "plots").exists()


def test_summarize_final_missing_phase2_job_writes_no_csv(tmp_path):
    run_dir = tmp_path / "phase2_missing_job"
    selected_gpus = ["0", "1"]
    write_formal_phase1(run_dir, selected_gpus=selected_gpus)
    ranked = summarize_phase1(run_dir=str(run_dir), selected_gpus=selected_gpus)
    write_formal_phase2(
        run_dir,
        selected_gpus=selected_gpus,
        ranked=ranked,
        skip_job_ids={"top12"},
    )
    with pytest.raises(ValueError, match="job"):
        summarize_final(run_dir=str(run_dir), selected_gpus=selected_gpus)
    assert not (run_dir / "cumulative_results.csv").exists()
    assert not (run_dir / "final_summary.json").exists()
    assert not (run_dir / "report.md").exists()
    assert not (run_dir / "plots").exists()
