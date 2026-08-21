from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.down_layer_sensitivity.run import (
    EXPECTED_DOWN_LAYERS,
    HISTORICAL_COMPRESSED_MMLU,
    HISTORICAL_PRE_DOWN_MMLU,
    RANDOM_CONTROL_SEEDS,
)

ACCURACY_TOL = 1e-12
CANONICAL_BASELINE_JOB_ID = "compressed_baseline_worker00"
WORKER00_REPEAT_JOB_ID = "compressed_baseline_worker00_repeat"
ALL_DOWN_ORIGINAL_JOB_ID = "all_down_original"
WEIGHT_METRICS_FILENAME = "weight_metrics_worker.json"
RUN_CONFIG_NAME = "run_config.json"

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

WEIGHT_METRICS_CSV_COLUMNS = [
    "layer_idx",
    "name",
    "numel",
    "mse",
    "nmse",
    "relative_fro_error",
    "original_rms",
    "error_rms",
]

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
PHASE2_SCIENTIFIC_JOB_IDS = [
    "top2",
    "top4",
    "top8",
    "top12",
    "random8_seed31",
    "random8_seed32",
    "random8_seed33",
    "random8_seed34",
    "random8_seed35",
]


def _dump_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    n = int(values.shape[0])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(x, y) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape or x_arr.size < 2:
        raise ValueError(
            "spearman_rank_correlation requires two 1-D arrays of equal length >= 2."
        )
    rx = _average_ranks(x_arr)
    ry = _average_ranks(y_arr)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        raise ValueError("spearman_rank_correlation is undefined when a rank vector is constant.")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not math.isfinite(rho):
        raise ValueError("spearman_rank_correlation produced a non-finite result.")
    return rho


def _load_run_config(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, RUN_CONFIG_NAME)
    if not os.path.isfile(path):
        raise ValueError(f"missing {RUN_CONFIG_NAME} under {run_dir}")
    config = _load_json(path)
    if not isinstance(config, dict):
        raise ValueError(f"{RUN_CONFIG_NAME} must be a JSON object.")
    return config


def _expected_formal_job_ids(worker_count: int) -> list[str]:
    job_ids = [f"compressed_baseline_worker{worker_id:02d}" for worker_id in range(worker_count)]
    job_ids.append(WORKER00_REPEAT_JOB_ID)
    job_ids.append(ALL_DOWN_ORIGINAL_JOB_ID)
    job_ids.extend(f"restore_L{layer_idx:02d}" for layer_idx in range(EXPECTED_DOWN_LAYERS))
    return job_ids


def _expected_smoke_job_ids() -> list[str]:
    return [
        CANONICAL_BASELINE_JOB_ID,
        WORKER00_REPEAT_JOB_ID,
        "restore_L00",
        ALL_DOWN_ORIGINAL_JOB_ID,
    ]


def _load_jobs_from_dir(jobs_dir: str, *, phase_label: str) -> dict[str, dict[str, Any]]:
    if not os.path.isdir(jobs_dir):
        raise ValueError(f"missing {phase_label} jobs directory: {jobs_dir}")

    jobs: dict[str, dict[str, Any]] = {}
    for filename in sorted(os.listdir(jobs_dir)):
        if not filename.endswith(".json"):
            raise ValueError(f"unexpected non-JSON file in {phase_label} jobs: {filename}")
        path = os.path.join(jobs_dir, filename)
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"job result {filename} must be a JSON object.")
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"job result {filename} is missing job_id.")
        stem = os.path.splitext(filename)[0]
        if stem != job_id:
            raise ValueError(f"job file {filename} does not match job_id {job_id!r}.")
        if job_id in jobs:
            raise ValueError(f"duplicate {phase_label} job ID: {job_id}")
        jobs[job_id] = payload
    return jobs


def _load_phase1_jobs(run_dir: str) -> dict[str, dict[str, Any]]:
    return _load_jobs_from_dir(
        os.path.join(run_dir, "phase1", "jobs"),
        phase_label="phase-1",
    )


def _load_phase2_jobs(run_dir: str) -> dict[str, dict[str, Any]]:
    return _load_jobs_from_dir(
        os.path.join(run_dir, "phase2", "jobs"),
        phase_label="phase-2",
    )


def _load_phase1_job(run_dir: str, job_id: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "phase1", "jobs", f"{job_id}.json")
    if not os.path.isfile(path):
        raise ValueError(f"missing phase-1 job: {job_id}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"phase-1 job {job_id} must be a JSON object.")
    return payload


def _assert_job_inventory(
    jobs: dict[str, dict[str, Any]],
    expected_ids: list[str],
    *,
    phase_label: str = "phase-1",
) -> None:
    expected = set(expected_ids)
    found = set(jobs)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            f"{phase_label} job inventory mismatch: expected {len(expected_ids)} jobs "
            f"{sorted(expected_ids)}, missing={missing}, extra={extra}"
        )


def _subject_map(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = job.get("subject_metrics")
    if not isinstance(metrics, list):
        raise ValueError(f"job {job.get('job_id')!r} subject_metrics must be a list.")
    mapping: dict[str, dict[str, Any]] = {}
    for row in metrics:
        if not isinstance(row, dict):
            raise ValueError(f"job {job.get('job_id')!r} subject row must be a dict.")
        name = row.get("subject_name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"job {job.get('job_id')!r} subject_name must be a non-empty string.")
        if name in mapping:
            raise ValueError(f"duplicate subject {name} in job {job.get('job_id')!r}.")
        mapping[name] = row
    return mapping


def _assert_same_population(
    reference: dict[str, Any],
    job: dict[str, Any],
    *,
    context: str,
) -> None:
    ref_subjects = _subject_map(reference)
    job_subjects = _subject_map(job)
    ref_names = set(ref_subjects)
    job_names = set(job_subjects)
    if ref_names != job_names:
        raise ValueError(
            f"evaluation population mismatch for {context}: "
            f"subject-name set differs from canonical baseline "
            f"(missing={sorted(ref_names - job_names)}, extra={sorted(job_names - ref_names)})."
        )
    for name in sorted(ref_names):
        ref_samples = int(ref_subjects[name]["samples"])
        job_samples = int(job_subjects[name]["samples"])
        if job_samples != ref_samples:
            raise ValueError(
                f"evaluation population mismatch for {context}: "
                f"subject {name} sample count {job_samples} != canonical {ref_samples}."
            )
    ref_total = int(reference["n_samples_total"])
    job_total = int(job["n_samples_total"])
    if job_total != ref_total:
        raise ValueError(
            f"evaluation population mismatch for {context}: "
            f"n_samples_total {job_total} != canonical {ref_total}."
        )


def _assert_matching_accuracies(
    reference: dict[str, Any],
    job: dict[str, Any],
    *,
    context: str,
) -> None:
    ref_acc = float(reference["accuracy"])
    job_acc = float(job["accuracy"])
    if abs(job_acc - ref_acc) > ACCURACY_TOL:
        raise ValueError(
            f"baseline determinism failed for {context}: "
            f"accuracy {job_acc} vs canonical {ref_acc} exceeds {ACCURACY_TOL}."
        )
    ref_subjects = _subject_map(reference)
    job_subjects = _subject_map(job)
    for name in sorted(ref_subjects):
        ref_subject_acc = float(ref_subjects[name]["accuracy"])
        job_subject_acc = float(job_subjects[name]["accuracy"])
        if abs(job_subject_acc - ref_subject_acc) > ACCURACY_TOL:
            raise ValueError(
                f"baseline determinism failed for {context}: "
                f"subject {name} accuracy {job_subject_acc} vs canonical {ref_subject_acc} "
                f"exceeds {ACCURACY_TOL}."
            )


def _assert_homogeneous_devices(jobs: dict[str, dict[str, Any]]) -> str:
    names: set[str] = set()
    for job_id, job in jobs.items():
        device_name = job.get("device_name")
        if not isinstance(device_name, str) or not device_name:
            raise ValueError(f"job {job_id} is missing device_name.")
        names.add(device_name)
    if len(names) != 1:
        raise ValueError(
            "formal aggregation requires a homogeneous GPU set; "
            f"found device names {sorted(names)}. Rerun on a homogeneous GPU set."
        )
    return next(iter(names))


def _probe_record(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "worker_id": job["worker_id"],
        "physical_gpu_id": job["physical_gpu_id"],
        "device_name": job["device_name"],
        "accuracy": float(job["accuracy"]),
        "accuracy_percent": float(job["accuracy_percent"]),
        "n_samples_total": int(job["n_samples_total"]),
    }


def _historical_reference(*, a_compressed: float, a_all: float) -> dict[str, Any]:
    return {
        "historical_compressed_mmlu": HISTORICAL_COMPRESSED_MMLU,
        "historical_pre_down_mmlu": HISTORICAL_PRE_DOWN_MMLU,
        "current_baseline_accuracy": a_compressed,
        "current_baseline_minus_historical_compressed_pp": 100.0
        * (a_compressed - HISTORICAL_COMPRESSED_MMLU),
        "current_all_down_original_accuracy": a_all,
        "current_all_down_original_minus_historical_pre_down_pp": 100.0
        * (a_all - HISTORICAL_PRE_DOWN_MMLU),
    }


def _load_weight_metrics(run_dir: str) -> dict[int, dict[str, Any]]:
    path = os.path.join(run_dir, "phase1", WEIGHT_METRICS_FILENAME)
    if not os.path.isfile(path):
        raise ValueError(f"missing canonical weight metrics: {path}")
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("weight_metrics_worker.json must be a list.")
    by_layer: dict[int, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("each weight metric row must be a dict.")
        layer_idx = row.get("layer_idx")
        if type(layer_idx) is not int:
            raise ValueError(f"weight metric layer_idx must be int, got {layer_idx!r}.")
        if layer_idx in by_layer:
            raise ValueError(f"duplicate weight metric layer_idx: {layer_idx}")
        by_layer[layer_idx] = row
    expected = list(range(EXPECTED_DOWN_LAYERS))
    if sorted(by_layer) != expected:
        raise ValueError(
            f"weight metrics must cover layers {expected}, got {sorted(by_layer)}."
        )
    return by_layer


def _subject_diagnostics(
    layer_job: dict[str, Any],
    baseline_job: dict[str, Any],
) -> dict[str, Any]:
    baseline_subjects = _subject_map(baseline_job)
    deltas: list[float] = []
    improved = 0
    worsened = 0
    unchanged = 0
    for row in layer_job["subject_metrics"]:
        name = row["subject_name"]
        delta_pp = 100.0 * (float(row["accuracy"]) - float(baseline_subjects[name]["accuracy"]))
        deltas.append(delta_pp)
        if delta_pp > ACCURACY_TOL:
            improved += 1
        elif delta_pp < -ACCURACY_TOL:
            worsened += 1
        else:
            unchanged += 1
    deltas_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "subjects_improved": improved,
        "subjects_worsened": worsened,
        "subjects_unchanged": unchanged,
        "median_subject_delta_pp": float(np.median(deltas_arr)),
        "max_subject_gain_pp": float(np.max(deltas_arr)),
        "max_subject_drop_pp": float(np.min(deltas_arr)),
    }


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _format_accuracy_percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def _format_pp(value: float) -> str:
    return f"{value:.4f} pp"


def _format_fraction(value: float) -> str:
    return f"{value:.6f}"


def _plot_layer_sensitivity(
    *,
    plots_dir: str,
    sensitivity_rows: list[dict[str, str]],
    ranked_layers: list[dict[str, Any]],
) -> None:
    delta_by_layer = {
        int(row["layer_idx"]): float(row["delta_mmlu_pp"]) for row in sensitivity_rows
    }
    layer_indices = list(range(EXPECTED_DOWN_LAYERS))
    deltas = [delta_by_layer[layer_idx] for layer_idx in layer_indices]
    top8 = ranked_layers[:8]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(layer_indices, deltas, color="#4c72b0")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    for row in top8:
        layer_idx = int(row["layer_idx"])
        rank = int(row["rank"])
        y = delta_by_layer[layer_idx]
        ax.annotate(
            f"L{layer_idx}/#{rank}",
            xy=(layer_idx, y),
            xytext=(0, 8 if y >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("layer_idx")
    ax.set_ylabel("delta_mmlu_pp")
    ax.set_title("Down VAE layer sensitivity by model depth")
    ax.set_xticks(layer_indices)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "layer_sensitivity.png"), dpi=150)
    plt.close(fig)


def _plot_nmse_vs_mmlu_sensitivity(
    *,
    plots_dir: str,
    sensitivity_rows: list[dict[str, str]],
    ranked_layers: list[dict[str, Any]],
    spearman_rho: float,
) -> None:
    nmse = np.asarray([float(row["weight_nmse"]) for row in sensitivity_rows], dtype=np.float64)
    delta_pp = np.asarray(
        [float(row["delta_mmlu_pp"]) for row in sensitivity_rows],
        dtype=np.float64,
    )
    top8_layer_ids = {int(row["layer_idx"]) for row in ranked_layers[:8]}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(nmse, delta_pp, color="#4c72b0", alpha=0.8)
    for row in sensitivity_rows:
        layer_idx = int(row["layer_idx"])
        if layer_idx not in top8_layer_ids:
            continue
        x = float(row["weight_nmse"])
        y = float(row["delta_mmlu_pp"])
        ax.annotate(
            f"L{layer_idx}",
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("weight_nmse")
    ax.set_ylabel("delta_mmlu_pp")
    ax.set_title(f"Weight NMSE vs MMLU sensitivity (Spearman rho={spearman_rho:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "nmse_vs_mmlu_sensitivity.png"), dpi=150)
    plt.close(fig)


def _plot_cumulative_recovery(
    *,
    plots_dir: str,
    cumulative_rows: list[dict[str, str]],
    random8_aggregate: dict[str, Any],
) -> None:
    by_name = {row["configuration"]: row for row in cumulative_rows}
    topk_names = ["top1", "top2", "top4", "top8", "top12", "all_down_original"]
    x_vals = [int(by_name[name]["num_restored_layers"]) for name in topk_names]
    y_vals = [float(by_name[name]["recovery_fraction"]) for name in topk_names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_vals, y_vals, marker="o", color="#4c72b0", label="Top-K cumulative")
    ax.errorbar(
        [8],
        [float(random8_aggregate["recovery_mean"])],
        yerr=[float(random8_aggregate["recovery_std"])],
        fmt="s",
        color="#c44e52",
        capsize=4,
        label="Random-8 mean ± std",
    )
    ax.set_xlabel("restored layer count")
    ax.set_ylabel("cumulative recovery fraction")
    ax.set_title("Top-K cumulative down-gap recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "cumulative_recovery.png"), dpi=150)
    plt.close(fig)


def _render_report(
    *,
    run_dir: str,
    final_summary: dict[str, Any],
    sensitivity_rows: list[dict[str, str]],
    cumulative_rows: list[dict[str, str]],
) -> None:
    compressed = final_summary["compressed_baseline"]
    all_down = final_summary["all_down_original"]
    historical = final_summary["historical_reference"]
    ranked_layers = final_summary["ranked_layers"]
    topk = final_summary["topk"]
    random8_controls = final_summary["random8_controls"]
    random8_aggregate = final_summary["random8_aggregate"]
    spearman_rho = float(final_summary["spearman_weight_nmse_vs_delta_mmlu"])
    down_gap_pp = float(final_summary["down_gap_pp"])

    sensitivity_by_layer = {int(row["layer_idx"]): row for row in sensitivity_rows}
    top8_layers = ranked_layers[:8]
    top8_ids = [int(row["layer_idx"]) for row in top8_layers]

    baseline_lines = []
    for probe in final_summary["cross_gpu_baseline_probes"]:
        baseline_lines.append(
            f"- `{probe['job_id']}`: accuracy={_format_accuracy_percent(float(probe['accuracy']))}, "
            f"device={probe['device_name']}, n_samples_total={probe['n_samples_total']}"
        )

    ranking_lines = []
    for row in ranked_layers:
        layer_idx = int(row["layer_idx"])
        csv_row = sensitivity_by_layer[layer_idx]
        ranking_lines.append(
            f"| {int(row['rank'])} | {layer_idx} | {float(row['delta_mmlu_pp']):.4f} | "
            f"{float(row['single_recovery_fraction']):.6f} | {float(csv_row['weight_nmse']):.6f} |"
        )

    top8_detail_lines = []
    for row in top8_layers:
        layer_idx = int(row["layer_idx"])
        top8_detail_lines.append(
            f"- L{layer_idx}: ΔMMLU={float(row['delta_mmlu_pp']):.4f} pp, "
            f"single recovery={float(row['single_recovery_fraction']):.6f}"
        )

    topk_lines = []
    for name in ("top1", "top2", "top4", "top8", "top12"):
        payload = topk[name]
        topk_lines.append(
            f"- {name}: restore_layers={payload['restore_layers']}, "
            f"recovery_fraction={_format_fraction(float(payload['recovery_fraction']))}, "
            f"ΔMMLU={_format_pp(float(payload['delta_from_compressed_pp']))}"
        )

    random8_lines = []
    for seed_key in ("seed31", "seed32", "seed33", "seed34", "seed35"):
        payload = random8_controls[seed_key]
        random8_lines.append(
            f"- {seed_key}: restore_layers={payload['restore_layers']}, "
            f"recovery_fraction={_format_fraction(float(payload['recovery_fraction']))}, "
            f"ΔMMLU={_format_pp(float(payload['delta_from_compressed_pp']))}"
        )

    all_down_row = next(row for row in cumulative_rows if row["configuration"] == "all_down_original")
    all_down_recovery = float(all_down_row["recovery_fraction"])

    conclusion_layers = ", ".join(f"L{layer_idx}" for layer_idx in top8_ids)
    lines = [
        "# Down VAE 层敏感度消融实验报告",
        "",
        "## 1. 实验目标",
        "",
        "在固定 final_model 与 0-shot full-MMLU 设置下，识别 36 个 down_proj 层中哪些层对 VAE 压缩最敏感，"
        "并评估 Top-K 联合恢复与 Random-8 对照的 down gap 回收情况。",
        "",
        "## 2. 固定实验设置",
        "",
        f"- run_id: `{final_summary.get('run_id')}`",
        f"- down 层数: {EXPECTED_DOWN_LAYERS}",
        "- 评估: 0-shot full-MMLU, batch size auto, 无 HiF4 activation",
        "- 排名依据: 单层 restore 的 ΔMMLU (pp)，tie-break 为较小 layer_idx",
        "",
        "## 3. 有效性检查",
        "",
        "phase-1 与 phase-2 全部 GPU baseline 与 worker00 canonical baseline 在 1e-12 容差内一致，"
        "subject/sample population 一致，已通过一致性门禁。",
        "",
        "cross_gpu_baseline_probes:",
        *baseline_lines,
        "",
        f"- worker00_baseline_repeat accuracy: {_format_accuracy_percent(float(compressed['accuracy']))}",
        "",
        "## 4. Down 压缩总损失",
        "",
        f"- 当前 compressed baseline: {_format_accuracy_percent(float(compressed['accuracy']))}",
        f"- 历史 compressed 参考 (41.71%): {100.0 * HISTORICAL_COMPRESSED_MMLU:.2f}%",
        f"- 与历史参考差值: {_format_pp(float(historical['current_baseline_minus_historical_compressed_pp']))}",
        f"- all-down-original accuracy: {_format_accuracy_percent(float(all_down['accuracy']))}",
        f"- 历史 pre-down 参考 (51.99%): {100.0 * HISTORICAL_PRE_DOWN_MMLU:.2f}%",
        f"- 与历史参考差值: {_format_pp(float(historical['current_all_down_original_minus_historical_pre_down_pp']))}",
        f"- 总可恢复 down gap: {_format_pp(down_gap_pp)}",
        "",
        "## 5. 36 层敏感度排名",
        "",
        f"Top-8 layer IDs: {top8_ids}",
        "",
        "Top-8 单层恢复明细:",
        *top8_detail_lines,
        "",
        "| rank | layer_idx | delta_mmlu_pp | single_recovery_fraction | weight_nmse |",
        "| --- | --- | --- | --- | --- |",
        *ranking_lines,
        "",
        "![layer sensitivity](plots/layer_sensitivity.png)",
        "",
        "## 6. Weight NMSE 与 MMLU 敏感度关系",
        "",
        f"- Spearman rho (weight_nmse vs delta_mmlu_pp): {spearman_rho:.6f}",
        "- 该图为诊断用途；相关不等于因果。",
        "",
        "![nmse vs mmlu sensitivity](plots/nmse_vs_mmlu_sensitivity.png)",
        "",
        "## 7. Top-K 累积恢复",
        "",
        *topk_lines,
        f"- all_down_original (K=36): recovery_fraction={_format_fraction(all_down_recovery)}",
        "",
        "![cumulative recovery](plots/cumulative_recovery.png)",
        "",
        "## 8. Random-8 五组对照",
        "",
        *random8_lines,
        "",
        f"- Random-8 recovery mean: {_format_fraction(float(random8_aggregate['recovery_mean']))}",
        f"- Random-8 recovery std (ddof=0): {_format_fraction(float(random8_aggregate['recovery_std']))}",
        f"- Top-8 minus Random-8 mean recovery: {_format_fraction(float(random8_aggregate['top8_minus_random8_mean_recovery']))}",
        "",
        "## 9. 结论与后续压缩建议",
        "",
        f"在当前 final_model 与 0-shot full-MMLU 设置下，{conclusion_layers} 对 down VAE 压缩最敏感。",
        "Top-K 累积曲线与 Random-8 对照可用于判断后续是否应对少数敏感层采用 selective high precision / "
        "selective higher bit / 定向保护策略；本实验不自动修改压缩策略。",
        "",
    ]
    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _generate_report_and_plots(*, run_dir: str, final_summary: dict[str, Any]) -> None:
    sensitivity_path = os.path.join(run_dir, "single_layer_sensitivity.csv")
    cumulative_path = os.path.join(run_dir, "cumulative_results.csv")
    if not os.path.isfile(sensitivity_path):
        raise ValueError(f"missing {sensitivity_path}")
    if not os.path.isfile(cumulative_path):
        raise ValueError(f"missing {cumulative_path}")

    sensitivity_rows = _read_csv_rows(sensitivity_path)
    cumulative_rows = _read_csv_rows(cumulative_path)
    plots_dir = os.path.join(run_dir, "plots")
    _ensure_dir(plots_dir)

    ranked_layers = final_summary["ranked_layers"]
    _plot_layer_sensitivity(
        plots_dir=plots_dir,
        sensitivity_rows=sensitivity_rows,
        ranked_layers=ranked_layers,
    )
    _plot_nmse_vs_mmlu_sensitivity(
        plots_dir=plots_dir,
        sensitivity_rows=sensitivity_rows,
        ranked_layers=ranked_layers,
        spearman_rho=float(final_summary["spearman_weight_nmse_vs_delta_mmlu"]),
    )
    _plot_cumulative_recovery(
        plots_dir=plots_dir,
        cumulative_rows=cumulative_rows,
        random8_aggregate=final_summary["random8_aggregate"],
    )
    _render_report(
        run_dir=run_dir,
        final_summary=final_summary,
        sensitivity_rows=sensitivity_rows,
        cumulative_rows=cumulative_rows,
    )


def _require_job_mode(jobs: dict[str, dict[str, Any]], mode: str) -> None:
    for job_id, job in jobs.items():
        job_mode = job.get("mode")
        if job_mode != mode:
            raise ValueError(
                f"job {job_id} mode={job_mode!r} does not match required mode={mode!r}."
            )


def _validate_restore_layers(jobs: dict[str, dict[str, Any]]) -> None:
    for job_id, job in jobs.items():
        restore_layers = job.get("restore_layers")
        if not isinstance(restore_layers, list):
            raise ValueError(f"job {job_id} restore_layers must be a list.")
        if job_id.startswith("compressed_baseline_"):
            expected = []
        elif job_id == ALL_DOWN_ORIGINAL_JOB_ID:
            expected = list(range(EXPECTED_DOWN_LAYERS))
        elif job_id.startswith("restore_L"):
            layer_idx = int(job_id.split("L", 1)[1])
            expected = [layer_idx]
        else:
            raise ValueError(f"unexpected phase-1 job_id: {job_id}")
        if restore_layers != expected:
            raise ValueError(
                f"job {job_id} restore_layers={restore_layers} does not match expected {expected}."
            )


def _validate_baseline_determinism(
    jobs: dict[str, dict[str, Any]],
    *,
    worker_count: int,
) -> dict[str, Any]:
    canonical = jobs[CANONICAL_BASELINE_JOB_ID]
    repeat = jobs[WORKER00_REPEAT_JOB_ID]
    _assert_same_population(
        canonical,
        repeat,
        context=WORKER00_REPEAT_JOB_ID,
    )
    _assert_matching_accuracies(
        canonical,
        repeat,
        context=WORKER00_REPEAT_JOB_ID,
    )
    for worker_id in range(1, worker_count):
        job_id = f"compressed_baseline_worker{worker_id:02d}"
        probe = jobs[job_id]
        _assert_same_population(canonical, probe, context=job_id)
        _assert_matching_accuracies(canonical, probe, context=job_id)
    return canonical


def validate_smoke(*, run_dir: str, selected_gpus: list[str]) -> None:
    config = _load_run_config(run_dir)
    if config.get("mode") != "smoke":
        raise ValueError("validate_smoke requires run_config mode=smoke.")
    if len(selected_gpus) != 1:
        raise ValueError("smoke mode allows exactly one GPU (W=1).")
    jobs = _load_phase1_jobs(run_dir)
    _assert_job_inventory(jobs, _expected_smoke_job_ids())
    _require_job_mode(jobs, "smoke")
    _validate_restore_layers(jobs)
    canonical = jobs[CANONICAL_BASELINE_JOB_ID]
    _assert_same_population(canonical, jobs[WORKER00_REPEAT_JOB_ID], context=WORKER00_REPEAT_JOB_ID)
    _assert_matching_accuracies(
        canonical,
        jobs[WORKER00_REPEAT_JOB_ID],
        context=WORKER00_REPEAT_JOB_ID,
    )
    for job_id, job in jobs.items():
        _assert_same_population(canonical, job, context=job_id)


def summarize_phase1(*, run_dir: str, selected_gpus: list[str]) -> list[int]:
    if not selected_gpus:
        raise ValueError("selected_gpus must be non-empty.")
    config = _load_run_config(run_dir)
    mode = config.get("mode")
    if mode == "smoke":
        raise ValueError("smoke results must not produce formal sensitivity ranking")
    if mode != "formal":
        raise ValueError(f"summarize_phase1 requires run_config mode=formal, got {mode!r}.")

    worker_count = len(selected_gpus)
    expected_ids = _expected_formal_job_ids(worker_count)
    if len(expected_ids) != 38 + worker_count:
        raise ValueError("internal inventory length error.")
    jobs = _load_phase1_jobs(run_dir)
    _assert_job_inventory(jobs, expected_ids)
    _require_job_mode(jobs, "formal")
    _validate_restore_layers(jobs)

    canonical = _validate_baseline_determinism(jobs, worker_count=worker_count)
    for job_id, job in jobs.items():
        _assert_same_population(canonical, job, context=job_id)
    _assert_homogeneous_devices(jobs)

    all_down = jobs[ALL_DOWN_ORIGINAL_JOB_ID]
    a_compressed = float(canonical["accuracy"])
    a_all = float(all_down["accuracy"])
    down_gap_pp = 100.0 * (a_all - a_compressed)
    diagnostic = {
        "status": "phase1_gates_passed_pending_ranking",
        "run_id": config.get("run_id"),
        "compressed_baseline": _probe_record(canonical),
        "cross_gpu_baseline_probes": [
            _probe_record(jobs[f"compressed_baseline_worker{worker_id:02d}"])
            for worker_id in range(worker_count)
        ],
        "worker00_baseline_repeat": _probe_record(jobs[WORKER00_REPEAT_JOB_ID]),
        "all_down_original": _probe_record(all_down),
        "down_gap_pp": down_gap_pp,
        "historical_reference": _historical_reference(
            a_compressed=a_compressed,
            a_all=a_all,
        ),
    }
    if not (a_all > a_compressed):
        diagnostic["status"] = "all_down_original_not_greater_than_compressed"
        _dump_json(os.path.join(run_dir, "phase1_summary.json"), diagnostic)
        raise ValueError("A_all_down_original <= A_compressed")

    weight_by_layer = _load_weight_metrics(run_dir)
    rows: list[dict[str, Any]] = []
    for layer_idx in range(EXPECTED_DOWN_LAYERS):
        job = jobs[f"restore_L{layer_idx:02d}"]
        accuracy = float(job["accuracy"])
        delta_pp = 100.0 * (accuracy - a_compressed)
        recovery_fraction = (accuracy - a_compressed) / (a_all - a_compressed)
        weight = weight_by_layer[layer_idx]
        diagnostics = _subject_diagnostics(job, canonical)
        rows.append(
            {
                "layer_idx": layer_idx,
                "module_name": weight["name"],
                "mmlu_accuracy": accuracy,
                "mmlu_accuracy_percent": 100.0 * accuracy,
                "delta_mmlu_pp": delta_pp,
                "single_recovery_fraction": recovery_fraction,
                "weight_mse": weight["mse"],
                "weight_nmse": weight["nmse"],
                "relative_fro_error": weight["relative_fro_error"],
                "original_rms": weight["original_rms"],
                "error_rms": weight["error_rms"],
                **diagnostics,
            }
        )

    ranked_rows = sorted(rows, key=lambda r: (-r["delta_mmlu_pp"], r["layer_idx"]))
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    spearman = spearman_rank_correlation(
        [row["weight_nmse"] for row in rows],
        [row["delta_mmlu_pp"] for row in rows],
    )
    ranked_layers = [
        {
            "rank": row["rank"],
            "layer_idx": row["layer_idx"],
            "delta_mmlu_pp": row["delta_mmlu_pp"],
            "single_recovery_fraction": row["single_recovery_fraction"],
            "mmlu_accuracy": row["mmlu_accuracy"],
        }
        for row in ranked_rows
    ]
    summary = {
        "status": "phase1_ranked",
        "run_id": config.get("run_id"),
        "compressed_baseline": diagnostic["compressed_baseline"],
        "cross_gpu_baseline_probes": diagnostic["cross_gpu_baseline_probes"],
        "worker00_baseline_repeat": diagnostic["worker00_baseline_repeat"],
        "all_down_original": diagnostic["all_down_original"],
        "down_gap_pp": down_gap_pp,
        "historical_reference": diagnostic["historical_reference"],
        "spearman_weight_nmse_vs_delta_mmlu": spearman,
        "ranked_layers": ranked_layers,
    }
    _dump_json(os.path.join(run_dir, "phase1_summary.json"), summary)
    _write_csv(
        os.path.join(run_dir, "single_layer_sensitivity.csv"),
        SENSITIVITY_CSV_COLUMNS,
        ranked_rows,
    )
    weight_rows = [weight_by_layer[layer_idx] for layer_idx in range(EXPECTED_DOWN_LAYERS)]
    _write_csv(
        os.path.join(run_dir, "weight_metrics.csv"),
        WEIGHT_METRICS_CSV_COLUMNS,
        weight_rows,
    )
    return [row["layer_idx"] for row in ranked_rows]


def _expected_phase2_job_ids(worker_count: int) -> list[str]:
    job_ids = [f"compressed_baseline_worker{worker_id:02d}" for worker_id in range(worker_count)]
    job_ids.append(WORKER00_REPEAT_JOB_ID)
    job_ids.extend(PHASE2_SCIENTIFIC_JOB_IDS)
    return job_ids


def _ranked_layer_ids(phase1_summary: dict[str, Any]) -> list[int]:
    ranked_rows = phase1_summary.get("ranked_layers")
    if not isinstance(ranked_rows, list) or not ranked_rows:
        raise ValueError("phase1_summary.json is missing ranked_layers.")
    ranked: list[int] = []
    for row in ranked_rows:
        if not isinstance(row, dict) or "layer_idx" not in row:
            raise ValueError("phase1_summary ranked_layers entries must include layer_idx.")
        ranked.append(int(row["layer_idx"]))
    if len(ranked) != EXPECTED_DOWN_LAYERS or sorted(ranked) != list(range(EXPECTED_DOWN_LAYERS)):
        raise ValueError(
            "phase1_summary ranked_layers must be a permutation of "
            f"0..{EXPECTED_DOWN_LAYERS - 1}, got {ranked!r}."
        )
    return ranked


def _expected_random8_layers(seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(list(range(EXPECTED_DOWN_LAYERS)), 8))


def _validate_phase2_restore_layers(
    jobs: dict[str, dict[str, Any]],
    ranked: list[int],
) -> None:
    expected_by_id = {
        "top2": ranked[:2],
        "top4": ranked[:4],
        "top8": ranked[:8],
        "top12": ranked[:12],
    }
    for seed in RANDOM_CONTROL_SEEDS:
        expected_by_id[f"random8_seed{seed}"] = _expected_random8_layers(seed)
    for job_id, job in jobs.items():
        restore_layers = job.get("restore_layers")
        if not isinstance(restore_layers, list):
            raise ValueError(f"job {job_id} restore_layers must be a list.")
        if job_id.startswith("compressed_baseline_"):
            expected = []
        elif job_id in expected_by_id:
            expected = expected_by_id[job_id]
        else:
            raise ValueError(f"unexpected phase-2 job_id: {job_id}")
        if restore_layers != expected:
            raise ValueError(
                f"job {job_id} restore_layers={restore_layers} does not match expected {expected}."
            )


def _assert_device_name(job: dict[str, Any], expected_device: str, *, context: str) -> None:
    device_name = job.get("device_name")
    if device_name != expected_device:
        raise ValueError(
            f"device_name mismatch for {context}: {device_name!r} != phase-1 {expected_device!r}."
        )


def _assert_formal_lm_limit(job: dict[str, Any], *, context: str) -> None:
    if job.get("mode") != "formal":
        raise ValueError(f"{context} requires formal mode and lm_limit=None, got mode={job.get('mode')!r}.")
    if "lm_limit" in job and job.get("lm_limit") is not None:
        raise ValueError(
            f"{context} requires formal lm_limit=None, got {job.get('lm_limit')!r}."
        )


def _configuration_metrics(
    *,
    accuracy: float,
    restore_layers: list[int],
    a_compressed: float,
    a_all: float,
) -> dict[str, Any]:
    return {
        "num_restored_layers": len(restore_layers),
        "restore_layers": list(restore_layers),
        "mmlu_accuracy": float(accuracy),
        "mmlu_accuracy_percent": 100.0 * float(accuracy),
        "delta_from_compressed_pp": 100.0 * (float(accuracy) - a_compressed),
        "recovery_fraction": (float(accuracy) - a_compressed) / (a_all - a_compressed),
    }


def summarize_final(*, run_dir: str, selected_gpus: list[str]) -> None:
    if not selected_gpus:
        raise ValueError("selected_gpus must be non-empty.")
    config = _load_run_config(run_dir)
    if config.get("mode") != "formal":
        raise ValueError(
            f"summarize_final requires run_config mode=formal, got {config.get('mode')!r}."
        )

    phase1_summary_path = os.path.join(run_dir, "phase1_summary.json")
    if not os.path.isfile(phase1_summary_path):
        raise ValueError(f"missing phase1_summary.json under {run_dir}")
    phase1_summary = _load_json(phase1_summary_path)
    if not isinstance(phase1_summary, dict):
        raise ValueError("phase1_summary.json must be a JSON object.")
    ranked = _ranked_layer_ids(phase1_summary)

    w2 = min(len(selected_gpus), 9)
    expected_ids = _expected_phase2_job_ids(w2)
    if len(expected_ids) != 9 + w2 + 1:
        raise ValueError("internal phase-2 inventory length error.")
    phase2_jobs = _load_phase2_jobs(run_dir)
    _assert_job_inventory(phase2_jobs, expected_ids, phase_label="phase-2")
    _require_job_mode(phase2_jobs, "formal")
    _validate_phase2_restore_layers(phase2_jobs, ranked)

    canonical = _load_phase1_job(run_dir, CANONICAL_BASELINE_JOB_ID)
    phase1_device = canonical.get("device_name")
    if not isinstance(phase1_device, str) or not phase1_device:
        raise ValueError("phase-1 canonical baseline is missing device_name.")

    baseline_ids = [f"compressed_baseline_worker{worker_id:02d}" for worker_id in range(w2)]
    baseline_ids.append(WORKER00_REPEAT_JOB_ID)
    for job_id in baseline_ids:
        probe = phase2_jobs[job_id]
        _assert_same_population(canonical, probe, context=job_id)
        _assert_matching_accuracies(canonical, probe, context=job_id)
        _assert_device_name(probe, phase1_device, context=job_id)
        _assert_formal_lm_limit(probe, context=job_id)

    for job_id in PHASE2_SCIENTIFIC_JOB_IDS:
        job = phase2_jobs[job_id]
        _assert_same_population(canonical, job, context=job_id)
        _assert_device_name(job, phase1_device, context=job_id)
        _assert_formal_lm_limit(job, context=job_id)

    top1_layer = ranked[0]
    top1_job_id = f"restore_L{top1_layer:02d}"
    top1_job = _load_phase1_job(run_dir, top1_job_id)
    if top1_job.get("restore_layers") != [top1_layer]:
        raise ValueError(
            f"phase-1 {top1_job_id} restore_layers={top1_job.get('restore_layers')!r} "
            f"does not match rank-1 layer {[top1_layer]}."
        )
    _assert_same_population(canonical, top1_job, context=top1_job_id)
    _assert_device_name(top1_job, phase1_device, context=top1_job_id)
    _assert_formal_lm_limit(top1_job, context=top1_job_id)

    all_down = _load_phase1_job(run_dir, ALL_DOWN_ORIGINAL_JOB_ID)
    _assert_same_population(canonical, all_down, context=ALL_DOWN_ORIGINAL_JOB_ID)
    _assert_device_name(all_down, phase1_device, context=ALL_DOWN_ORIGINAL_JOB_ID)
    _assert_formal_lm_limit(all_down, context=ALL_DOWN_ORIGINAL_JOB_ID)
    if all_down.get("restore_layers") != list(range(EXPECTED_DOWN_LAYERS)):
        raise ValueError(
            f"phase-1 {ALL_DOWN_ORIGINAL_JOB_ID} restore_layers="
            f"{all_down.get('restore_layers')!r} does not match 0..35."
        )

    a_compressed = float(canonical["accuracy"])
    a_all = float(all_down["accuracy"])
    if not (a_all > a_compressed):
        raise ValueError("A_all_down_original <= A_compressed")

    restore_by_config = {
        "top1": ranked[:1],
        "top2": ranked[:2],
        "top4": ranked[:4],
        "top8": ranked[:8],
        "top12": ranked[:12],
        "random8_seed31": _expected_random8_layers(31),
        "random8_seed32": _expected_random8_layers(32),
        "random8_seed33": _expected_random8_layers(33),
        "random8_seed34": _expected_random8_layers(34),
        "random8_seed35": _expected_random8_layers(35),
        "all_down_original": list(range(EXPECTED_DOWN_LAYERS)),
    }
    accuracy_by_config = {
        "top1": float(top1_job["accuracy"]),
        "top2": float(phase2_jobs["top2"]["accuracy"]),
        "top4": float(phase2_jobs["top4"]["accuracy"]),
        "top8": float(phase2_jobs["top8"]["accuracy"]),
        "top12": float(phase2_jobs["top12"]["accuracy"]),
        "random8_seed31": float(phase2_jobs["random8_seed31"]["accuracy"]),
        "random8_seed32": float(phase2_jobs["random8_seed32"]["accuracy"]),
        "random8_seed33": float(phase2_jobs["random8_seed33"]["accuracy"]),
        "random8_seed34": float(phase2_jobs["random8_seed34"]["accuracy"]),
        "random8_seed35": float(phase2_jobs["random8_seed35"]["accuracy"]),
        "all_down_original": a_all,
    }

    csv_rows: list[dict[str, Any]] = []
    metrics_by_config: dict[str, dict[str, Any]] = {}
    for name in CUMULATIVE_ROW_ORDER:
        metrics = _configuration_metrics(
            accuracy=accuracy_by_config[name],
            restore_layers=restore_by_config[name],
            a_compressed=a_compressed,
            a_all=a_all,
        )
        metrics_by_config[name] = metrics
        csv_rows.append(
            {
                "configuration": name,
                "num_restored_layers": metrics["num_restored_layers"],
                "restore_layers": json.dumps(metrics["restore_layers"]),
                "mmlu_accuracy": metrics["mmlu_accuracy"],
                "mmlu_accuracy_percent": metrics["mmlu_accuracy_percent"],
                "delta_from_compressed_pp": metrics["delta_from_compressed_pp"],
                "recovery_fraction": metrics["recovery_fraction"],
            }
        )

    random8_accuracies = np.asarray(
        [metrics_by_config[f"random8_seed{seed}"]["mmlu_accuracy"] for seed in RANDOM_CONTROL_SEEDS],
        dtype=np.float64,
    )
    random8_recoveries = np.asarray(
        [
            metrics_by_config[f"random8_seed{seed}"]["recovery_fraction"]
            for seed in RANDOM_CONTROL_SEEDS
        ],
        dtype=np.float64,
    )
    random8_accuracy_mean = float(np.mean(random8_accuracies))
    random8_accuracy_std = float(np.std(random8_accuracies, ddof=0))
    random8_recovery_mean = float(np.mean(random8_recoveries))
    random8_recovery_std = float(np.std(random8_recoveries, ddof=0))
    top8_recovery = metrics_by_config["top8"]["recovery_fraction"]
    top8_minus_random8_mean_recovery = top8_recovery - random8_recovery_mean

    required_phase1_fields = [
        "compressed_baseline",
        "cross_gpu_baseline_probes",
        "all_down_original",
        "down_gap_pp",
        "ranked_layers",
        "spearman_weight_nmse_vs_delta_mmlu",
        "historical_reference",
    ]
    for field in required_phase1_fields:
        if field not in phase1_summary:
            raise ValueError(f"phase1_summary.json is missing {field}.")

    topk = {name: dict(metrics_by_config[name]) for name in ("top1", "top2", "top4", "top8", "top12")}
    random8_controls = {}
    for seed in RANDOM_CONTROL_SEEDS:
        payload = dict(metrics_by_config[f"random8_seed{seed}"])
        payload["seed"] = seed
        random8_controls[f"seed{seed}"] = payload

    summary = {
        "status": "phase2_aggregated",
        "run_id": config.get("run_id"),
        "compressed_baseline": phase1_summary["compressed_baseline"],
        "cross_gpu_baseline_probes": phase1_summary["cross_gpu_baseline_probes"],
        "all_down_original": phase1_summary["all_down_original"],
        "down_gap_pp": phase1_summary["down_gap_pp"],
        "ranked_layers": phase1_summary["ranked_layers"],
        "spearman_weight_nmse_vs_delta_mmlu": phase1_summary["spearman_weight_nmse_vs_delta_mmlu"],
        "topk": topk,
        "random8_controls": random8_controls,
        "random8_aggregate": {
            "accuracy_mean": random8_accuracy_mean,
            "accuracy_std": random8_accuracy_std,
            "recovery_mean": random8_recovery_mean,
            "recovery_std": random8_recovery_std,
            "top8_minus_random8_mean_recovery": top8_minus_random8_mean_recovery,
        },
        "historical_reference": phase1_summary["historical_reference"],
    }
    _write_csv(
        os.path.join(run_dir, "cumulative_results.csv"),
        CUMULATIVE_CSV_COLUMNS,
        csv_rows,
    )
    _dump_json(os.path.join(run_dir, "final_summary.json"), summary)
    _generate_report_and_plots(run_dir=run_dir, final_summary=summary)
