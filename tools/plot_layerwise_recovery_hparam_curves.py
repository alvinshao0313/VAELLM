#!/usr/bin/env python3
"""Plot diagnostics for layer-wise recovery hyper-parameter sweeps.

This script scans block-wise recovery runs, parses local distillation loss curves,
parses available prefix-evaluation curves, and produces figures for the paper
claim that local layer-wise objectives can be optimized while full-model prefix
performance still degrades under sequential replacement.

Default inputs are the paths used in the VAELLM workspace:
  - .result/block_vae_lora/Qwen_Qwen3-8B_*/block_vae_lora.log
  - .result/block_vae_lora/Qwen_Qwen3-8B_*/normalized_block_vae_lora_args.json
  - eval_log/block_prefix_eval/*

Outputs are written to:
  .result/block_vae_lora/figures/layerwise_hparam_diagnostic/

The script produces both averaged run-level curves and per-run all-block curves.
The averaged curves are intended for compact paper figures, while the all-block
curves and heatmaps are intended for appendix or diagnostic inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DISTILL_RE = re.compile(
    r"\[block (?P<block>\d+)\] distill step=(?P<step>\d+)/(?:\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) "
    r"attn_kl=(?P<attn>[0-9.eE+-]+) "
    r"linear=(?P<linear>[0-9.eE+-]+) "
    r"hidden=(?P<hidden>[0-9.eE+-]+) "
    r"lr=(?P<lr>[0-9.eE+-]+)"
)
PREFIX_LOG_RE = re.compile(
    r"\[prefix n=(?P<prefix>\d+)/(?:\d+)\].*?"
    r"mean=(?P<mean>[0-9.]+)(?: \((?P<pct>[0-9.]+)%\))?"
)
RUN_ID_RE = re.compile(r"Qwen_Qwen3-8B_\d{8}_\d{6}")


@dataclass
class RunData:
    run_id: str
    run_dir: Path
    args: dict[str, Any]
    setting_label: str
    curves_by_block: dict[int, list[dict[str, float]]]
    prefix_curve: list[tuple[int, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(".result/block_vae_lora"),
        help="Directory containing Qwen_Qwen3-8B_* block recovery runs.",
    )
    parser.add_argument(
        "--prefix-root",
        type=Path,
        default=Path("eval_log/block_prefix_eval"),
        help="Directory containing prefix evaluation logs and summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".result/block_vae_lora/figures/layerwise_hparam_diagnostic"),
        help="Output directory for figures and CSV summaries.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=101,
        help="Number of normalized-progress bins for averaged loss curves.",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=78,
        help="Maximum legend label length.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_arg(args_blob: dict[str, Any], key: str, default: Any = None) -> Any:
    return args_blob.get("args", {}).get(key, default)


def short_float(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value:.0e}"
        return f"{value:g}"
    return str(value)


def make_setting_label(run_id: str, args_blob: dict[str, Any], max_len: int) -> str:
    steps = get_arg(args_blob, "block_distill_steps", "NA")
    mode = get_arg(args_blob, "block_distill_train_mode", "NA")
    block_layers = get_arg(args_blob, "block_layers", "NA")
    alpha = get_arg(args_blob, "block_loss_alpha", "NA")
    beta = get_arg(args_blob, "block_loss_beta", "NA")
    lora_lr = get_arg(args_blob, "block_lora_lr", get_arg(args_blob, "lr", "NA"))
    lora_sched = get_arg(args_blob, "block_lora_lr_scheduler", "NA")
    rank = get_arg(args_blob, "block_lora_rank", "NA")

    step_label = f"{int(steps / 1000)}k" if isinstance(steps, int) and steps % 1000 == 0 else str(steps)
    label = (
        f"{run_id}: mode={mode}, layers={block_layers}, steps={step_label}, "
        f"alpha={short_float(alpha)}, beta={short_float(beta)}, "
        f"rank={rank}, lora_lr={short_float(lora_lr)}, sched={lora_sched}"
    )
    if len(label) > max_len:
        # Keep the run id and the most diagnostic parameters visible.
        label = (
            f"{run_id[-13:]} | {mode}, L={block_layers}, {step_label}, "
            f"a={short_float(alpha)}, b={short_float(beta)}, lr={short_float(lora_lr)}"
        )
    return label


def parse_distill_log(path: Path) -> dict[int, list[dict[str, float]]]:
    curves: dict[int, list[dict[str, float]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = DISTILL_RE.search(line)
            if not m:
                continue
            block = int(m.group("block"))
            curves[block].append(
                {
                    "step": float(m.group("step")),
                    "loss": float(m.group("loss")),
                    "attn_kl": float(m.group("attn")),
                    "linear": float(m.group("linear")),
                    "hidden": float(m.group("hidden")),
                    "lr": float(m.group("lr")),
                }
            )
    for block in list(curves):
        curves[block].sort(key=lambda row: row["step"])
    return dict(curves)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def run_id_from_text(text: str) -> str | None:
    match = RUN_ID_RE.search(text)
    return match.group(0) if match else None


def parse_prefix_log(path: Path) -> tuple[str | None, list[tuple[int, float]]]:
    text = read_text(path)
    run_id = run_id_from_text(text)
    rows: list[tuple[int, float]] = []
    for m in PREFIX_LOG_RE.finditer(text):
        prefix = int(m.group("prefix"))
        if m.group("pct") is not None:
            mean_pct = float(m.group("pct"))
        else:
            raw = float(m.group("mean"))
            mean_pct = raw * 100.0 if raw <= 1.0 else raw
        rows.append((prefix, mean_pct))
    rows = sorted(set(rows), key=lambda x: x[0])
    return run_id, rows


def parse_prefix_summary(path: Path) -> tuple[str | None, list[tuple[int, float]]]:
    if not path.exists():
        return None, []
    try:
        data = load_json(path)
    except Exception:
        return None, []
    run_id = run_id_from_text(json.dumps(data, ensure_ascii=False))
    rows_blob = data.get("rows", [])
    rows: list[tuple[int, float]] = []
    for row in rows_blob:
        if not isinstance(row, dict) or "prefix_layers" not in row:
            continue
        mean = row.get("mean_metric", row.get("average_accuracy", row.get("mean")))
        if mean is None:
            # Fall back to task-average if present.
            task_metrics = row.get("task_metrics") or row.get("tasks") or {}
            values: list[float] = []
            if isinstance(task_metrics, dict):
                for value in task_metrics.values():
                    if isinstance(value, dict):
                        metric = value.get("acc", value.get("accuracy", value.get("metric")))
                    else:
                        metric = value
                    if isinstance(metric, (int, float)):
                        values.append(float(metric))
            if values:
                mean = sum(values) / len(values)
        if mean is None:
            continue
        mean_pct = float(mean) * 100.0 if float(mean) <= 1.0 else float(mean)
        rows.append((int(row["prefix_layers"]), mean_pct))
    rows = sorted(set(rows), key=lambda x: x[0])
    return run_id, rows


def load_prefix_curves(prefix_root: Path) -> dict[str, list[tuple[int, float]]]:
    curves: dict[str, list[tuple[int, float]]] = {}
    if not prefix_root.exists():
        return curves

    for log_path in sorted(prefix_root.glob("block_prefix_eval_*.log")):
        run_id, rows = parse_prefix_log(log_path)
        if run_id and rows:
            # Keep the most complete curve if duplicated.
            if run_id not in curves or len(rows) > len(curves[run_id]):
                curves[run_id] = rows

    summary_path = prefix_root / "prefix_eval_summary.json"
    run_id, rows = parse_prefix_summary(summary_path)
    if run_id and rows:
        if run_id not in curves or len(rows) > len(curves[run_id]):
            curves[run_id] = rows
    return curves


def load_runs(run_root: Path, prefix_curves: dict[str, list[tuple[int, float]]], max_label_len: int) -> list[RunData]:
    runs: list[RunData] = []
    if not run_root.exists():
        return runs
    for run_dir in sorted(run_root.glob("Qwen_Qwen3-8B_*")):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        log_path = run_dir / "block_vae_lora.log"
        curves = parse_distill_log(log_path)
        if not curves:
            continue
        args_path = run_dir / "normalized_block_vae_lora_args.json"
        args_blob: dict[str, Any] = {}
        if args_path.exists():
            try:
                args_blob = load_json(args_path)
            except Exception:
                args_blob = {}
        label = make_setting_label(run_id, args_blob, max_label_len)
        runs.append(
            RunData(
                run_id=run_id,
                run_dir=run_dir,
                args=args_blob,
                setting_label=label,
                curves_by_block=curves,
                prefix_curve=prefix_curves.get(run_id, []),
            )
        )
    return runs


def normalized_loss_curve(
    curves_by_block: dict[int, list[dict[str, float]]],
    bins: int,
    metric: str = "loss",
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, bins)
    block_curves: list[np.ndarray] = []
    for rows in curves_by_block.values():
        if len(rows) < 2:
            continue
        steps = np.array([row["step"] for row in rows], dtype=float)
        values = np.array([row[metric] for row in rows], dtype=float)
        if not np.all(np.isfinite(values)) or values[0] == 0:
            continue
        progress = (steps - steps[0]) / max(steps[-1] - steps[0], 1.0)
        norm_values = values / values[0]
        order = np.argsort(progress)
        interp = np.interp(grid, progress[order], norm_values[order])
        block_curves.append(interp)
    if not block_curves:
        return grid, np.full_like(grid, np.nan)
    return grid, np.nanmean(np.vstack(block_curves), axis=0)


def run_loss_summary(run: RunData) -> dict[str, float]:
    first_values: list[float] = []
    last_values: list[float] = []
    decreased = 0
    total = 0
    for rows in run.curves_by_block.values():
        if len(rows) < 2:
            continue
        first = rows[0]["loss"]
        last = rows[-1]["loss"]
        if first == 0 or not (math.isfinite(first) and math.isfinite(last)):
            continue
        first_values.append(first)
        last_values.append(last)
        decreased += int(last < first)
        total += 1
    if not first_values:
        return {
            "num_blocks": 0,
            "loss_decreased_blocks": 0,
            "mean_loss_reduction_pct": float("nan"),
            "mean_loss_first": float("nan"),
            "mean_loss_last": float("nan"),
        }
    reductions = [(f - l) / f * 100.0 for f, l in zip(first_values, last_values)]
    return {
        "num_blocks": float(total),
        "loss_decreased_blocks": float(decreased),
        "mean_loss_reduction_pct": float(np.mean(reductions)),
        "mean_loss_first": float(np.mean(first_values)),
        "mean_loss_last": float(np.mean(last_values)),
    }


def prefix_summary(prefix_curve: list[tuple[int, float]]) -> dict[str, float]:
    if not prefix_curve:
        return {
            "prefix0_mean": float("nan"),
            "best_prefix_layers": float("nan"),
            "best_prefix_mean": float("nan"),
            "final_prefix_layers": float("nan"),
            "final_prefix_mean": float("nan"),
            "final_drop_points": float("nan"),
        }
    rows = sorted(prefix_curve, key=lambda x: x[0])
    prefix0 = rows[0][1]
    best = max(rows, key=lambda x: x[1])
    final = rows[-1]
    return {
        "prefix0_mean": prefix0,
        "best_prefix_layers": float(best[0]),
        "best_prefix_mean": best[1],
        "final_prefix_layers": float(final[0]),
        "final_prefix_mean": final[1],
        "final_drop_points": final[1] - prefix0,
    }


def write_summary_csv(runs: list[RunData], output_path: Path) -> None:
    fieldnames = [
        "run_id",
        "setting_label",
        "block_layers",
        "block_distill_train_mode",
        "block_distill_steps",
        "block_loss_alpha",
        "block_loss_beta",
        "block_lora_rank",
        "block_lora_lr",
        "block_lora_lr_scheduler",
        "lr",
        "lr_scheduler",
        "num_blocks_with_curves",
        "loss_decreased_blocks",
        "mean_loss_reduction_pct",
        "mean_loss_first",
        "mean_loss_last",
        "has_prefix_eval",
        "prefix0_mean",
        "best_prefix_layers",
        "best_prefix_mean",
        "final_prefix_layers",
        "final_prefix_mean",
        "final_drop_points",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            loss_stats = run_loss_summary(run)
            pref_stats = prefix_summary(run.prefix_curve)
            args = run.args.get("args", {})
            writer.writerow(
                {
                    "run_id": run.run_id,
                    "setting_label": run.setting_label,
                    "block_layers": args.get("block_layers"),
                    "block_distill_train_mode": args.get("block_distill_train_mode"),
                    "block_distill_steps": args.get("block_distill_steps"),
                    "block_loss_alpha": args.get("block_loss_alpha"),
                    "block_loss_beta": args.get("block_loss_beta"),
                    "block_lora_rank": args.get("block_lora_rank"),
                    "block_lora_lr": args.get("block_lora_lr"),
                    "block_lora_lr_scheduler": args.get("block_lora_lr_scheduler"),
                    "lr": args.get("lr"),
                    "lr_scheduler": args.get("lr_scheduler"),
                    "num_blocks_with_curves": int(loss_stats["num_blocks"]),
                    "loss_decreased_blocks": int(loss_stats["loss_decreased_blocks"]),
                    "mean_loss_reduction_pct": loss_stats["mean_loss_reduction_pct"],
                    "mean_loss_first": loss_stats["mean_loss_first"],
                    "mean_loss_last": loss_stats["mean_loss_last"],
                    "has_prefix_eval": bool(run.prefix_curve),
                    **pref_stats,
                }
            )


def save_line_plot(
    runs: list[RunData],
    output_dir: Path,
    bins: int,
    filename_stem: str = "layerwise_hparam_local_loss_curves",
) -> None:
    plt.figure(figsize=(9.5, 5.2))
    plotted = 0
    for run in runs:
        grid, values = normalized_loss_curve(run.curves_by_block, bins=bins, metric="loss")
        if np.all(~np.isfinite(values)):
            continue
        plt.plot(grid, values, linewidth=1.8, label=run.setting_label)
        plotted += 1
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Normalized training progress")
    plt.ylabel("Mean normalized local loss")
    plt.title("Layer-wise recovery: local objectives under different settings")
    if plotted <= 12:
        plt.legend(fontsize=7, loc="best")
    else:
        plt.legend(fontsize=6, loc="upper right", ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"{filename_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def block_normalized_curve(
    rows: list[dict[str, float]],
    bins: int,
    metric: str = "loss",
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, bins)
    if len(rows) < 2:
        return grid, np.full_like(grid, np.nan)
    steps = np.array([row["step"] for row in rows], dtype=float)
    values = np.array([row[metric] for row in rows], dtype=float)
    if not np.all(np.isfinite(values)) or values[0] == 0:
        return grid, np.full_like(grid, np.nan)
    progress = (steps - steps[0]) / max(steps[-1] - steps[0], 1.0)
    norm_values = values / values[0]
    order = np.argsort(progress)
    return grid, np.interp(grid, progress[order], norm_values[order])


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Lightly smooth a curve while preserving its length."""
    if window <= 1 or len(values) < window:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def save_all_block_curves(runs: list[RunData], output_dir: Path, bins: int) -> None:
    """Draw all layer-wise loss curves of each run in one trend figure.

    Each block curve is normalized by its first loss value and interpolated to
    the same number of progress bins. Individual block curves are shown as thin
    low-opacity lines, while the all-block mean trend is overlaid as a thick
    line. This figure is intended to show the broad downward local-loss trend
    without over-emphasizing step-level noise.
    """
    all_block_dir = output_dir / "all_blocks_curves"
    all_block_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        if not run.curves_by_block:
            continue
        plt.figure(figsize=(8.8, 5.0))
        plotted = 0
        all_curves: list[np.ndarray] = []
        for block in sorted(run.curves_by_block):
            grid, values = block_normalized_curve(run.curves_by_block[block], bins=bins, metric="loss")
            if np.all(~np.isfinite(values)):
                continue
            values = smooth_curve(values, window=5)
            all_curves.append(values)
            plt.plot(grid, values, linewidth=0.8, alpha=0.28)
            plotted += 1
        if plotted == 0 or not all_curves:
            plt.close()
            continue
        mean_curve = smooth_curve(np.nanmean(np.vstack(all_curves), axis=0), window=5)
        plt.plot(grid, mean_curve, linewidth=3.0, label="Mean trend")
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Normalized training progress")
        plt.ylabel("Normalized local loss")
        plt.title(f"All-layer local loss trends: {run.run_id}")
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        stem = safe_filename(f"{run.run_id}_all_blocks_loss_trend")
        for ext in ("png", "pdf"):
            plt.savefig(all_block_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close()


def save_all_block_heatmaps(runs: list[RunData], output_dir: Path, bins: int) -> None:
    heatmap_dir = output_dir / "all_blocks_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        blocks = sorted(run.curves_by_block)
        if not blocks:
            continue
        rows_for_heatmap: list[np.ndarray] = []
        valid_blocks: list[int] = []
        for block in blocks:
            _, values = block_normalized_curve(run.curves_by_block[block], bins=bins, metric="loss")
            if np.all(~np.isfinite(values)):
                continue
            rows_for_heatmap.append(values)
            valid_blocks.append(block)
        if not rows_for_heatmap:
            continue
        matrix = np.vstack(rows_for_heatmap)
        plt.figure(figsize=(9.5, max(4.2, 0.22 * len(valid_blocks))))
        im = plt.imshow(matrix, aspect="auto", interpolation="nearest", origin="lower")
        plt.colorbar(im, label="Normalized local loss")
        plt.xlabel("Normalized training progress bin")
        plt.ylabel("Block index")
        plt.title(f"All block local-loss heatmap: {run.run_id}")
        step = max(1, len(valid_blocks) // 18)
        yticks = list(range(0, len(valid_blocks), step))
        plt.yticks(yticks, [valid_blocks[i] for i in yticks])
        plt.tight_layout()
        stem = safe_filename(f"{run.run_id}_all_blocks_loss_heatmap")
        for ext in ("png", "pdf"):
            plt.savefig(heatmap_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close()


def write_block_level_csv(runs: list[RunData], output_path: Path) -> None:
    fieldnames = [
        "run_id",
        "setting_label",
        "block",
        "num_points",
        "first_step",
        "last_step",
        "loss_first",
        "loss_last",
        "loss_reduction_pct",
        "loss_decreased",
        "hidden_first",
        "hidden_last",
        "linear_first",
        "linear_last",
        "attn_kl_first",
        "attn_kl_last",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for block, rows in sorted(run.curves_by_block.items()):
                if len(rows) < 2:
                    continue
                first = rows[0]
                last = rows[-1]
                loss_first = first["loss"]
                loss_last = last["loss"]
                reduction = ((loss_first - loss_last) / loss_first * 100.0) if loss_first else float("nan")
                writer.writerow(
                    {
                        "run_id": run.run_id,
                        "setting_label": run.setting_label,
                        "block": block,
                        "num_points": len(rows),
                        "first_step": int(first["step"]),
                        "last_step": int(last["step"]),
                        "loss_first": loss_first,
                        "loss_last": loss_last,
                        "loss_reduction_pct": reduction,
                        "loss_decreased": loss_last < loss_first,
                        "hidden_first": first["hidden"],
                        "hidden_last": last["hidden"],
                        "linear_first": first["linear"],
                        "linear_last": last["linear"],
                        "attn_kl_first": first["attn_kl"],
                        "attn_kl_last": last["attn_kl"],
                    }
                )


def save_prefix_plot(runs: list[RunData], output_dir: Path) -> None:
    plt.figure(figsize=(9.5, 5.2))
    plotted = 0
    for run in runs:
        if not run.prefix_curve:
            continue
        rows = sorted(run.prefix_curve, key=lambda x: x[0])
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        plt.plot(xs, ys, marker="o", markersize=3.0, linewidth=1.8, label=run.setting_label)
        plotted += 1
    plt.xlabel("Number of prefix layers replaced")
    plt.ylabel("Average downstream accuracy (%)")
    plt.title("Prefix evaluation after sequentially activating recovered layers")
    if plotted > 0:
        plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"layerwise_hparam_prefix_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def save_scatter_plot(runs: list[RunData], output_dir: Path) -> None:
    points: list[tuple[float, float, str]] = []
    for run in runs:
        if not run.prefix_curve:
            continue
        loss_stats = run_loss_summary(run)
        pref_stats = prefix_summary(run.prefix_curve)
        x = loss_stats["mean_loss_reduction_pct"]
        y = pref_stats["final_drop_points"]
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y, run.run_id[-13:]))
    plt.figure(figsize=(6.8, 5.0))
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, s=40)
        for x, y, label in points:
            plt.annotate(label, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Mean local loss reduction (%)")
    plt.ylabel("Final prefix drop (points)")
    plt.title("Local loss reduction vs. full-model degradation")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"layerwise_hparam_loss_vs_drop.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def paper_hparam_label(run: RunData) -> str:
    """Compact setting label for paper figures.

    The long diagnostic label is useful for debugging, but it hides the actual
    message in a paper figure.  This label keeps only the parameters that show
    that the comparison spans different reasonable recovery settings.
    """
    args = run.args.get("args", {})
    steps = args.get("block_distill_steps", "NA")
    if isinstance(steps, int) and steps % 1000 == 0:
        steps_label = f"{steps // 1000}k"
    else:
        steps_label = str(steps)
    alpha = short_float(args.get("block_loss_alpha", "NA"))
    beta = short_float(args.get("block_loss_beta", "NA"))
    lr = short_float(args.get("block_lora_lr", args.get("lr", "NA")))
    mode = args.get("block_distill_train_mode", "NA")
    return f"{steps_label}, a={alpha}, b={beta}, lr={lr}, {mode}"


def matched_prefix_runs(runs: list[RunData]) -> list[RunData]:
    matched = [run for run in runs if run.prefix_curve]
    return matched if matched else runs


def save_failure_panel(runs: list[RunData], output_dir: Path, bins: int) -> None:
    """Save a compact paper figure that links three pieces of evidence.

    The intended claim is not merely that local losses decrease.  The intended
    claim is that several reasonable recovery hyper-parameters optimize the
    local objective, yet the corresponding sequential prefix evaluation remains
    degraded.  A single multi-panel figure makes this failure mode explicit.
    """
    selected = matched_prefix_runs(runs)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))

    # (a) Local objective curves under different hyper-parameter settings.
    ax = axes[0]
    for run in selected:
        grid, values = normalized_loss_curve(run.curves_by_block, bins=bins, metric="loss")
        if np.all(~np.isfinite(values)):
            continue
        values = smooth_curve(values, window=5)
        ax.plot(grid, values, linewidth=2.0, label=paper_hparam_label(run))
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Normalized training progress")
    ax.set_ylabel("Mean normalized local objective")
    ax.set_title("(a) Local objectives are optimized")
    ax.legend(fontsize=6.2, loc="best", frameon=False)

    # (b) Prefix evaluation for the same hyper-parameter settings.
    ax = axes[1]
    for run in selected:
        if not run.prefix_curve:
            continue
        rows = sorted(run.prefix_curve, key=lambda x: x[0])
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        ax.plot(xs, ys, marker="o", markersize=3.0, linewidth=2.0, label=paper_hparam_label(run))
    ax.set_xlabel("Number of replaced prefix layers")
    ax.set_ylabel("Average downstream accuracy (%)")
    ax.set_title("(b) Prefix performance still degrades")

    # (c) Summary: local improvement does not imply global recovery.
    ax = axes[2]
    points: list[tuple[float, float, RunData]] = []
    for run in selected:
        if not run.prefix_curve:
            continue
        loss_stats = run_loss_summary(run)
        pref_stats = prefix_summary(run.prefix_curve)
        x = loss_stats["mean_loss_reduction_pct"]
        y = pref_stats["final_drop_points"]
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y, run))
    if points:
        xs = [x for x, _, _ in points]
        ys = [y for _, y, _ in points]
        ax.scatter(xs, ys, s=55)
        for x, y, run in points:
            loss_stats = run_loss_summary(run)
            label = f"{int(loss_stats['loss_decreased_blocks'])}/{int(loss_stats['num_blocks'])} blocks"
            ax.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.text(
            0.98,
            0.05,
            "local loss down\nbut global drop remains",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Mean local loss reduction (%)")
    ax.set_ylabel("Final prefix drop (points)")
    ax.set_title("(c) Local/global objective mismatch")

    fig.suptitle(
        "Layer-wise recovery hyper-parameter sweep: local convergence does not ensure full-model recovery",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"layerwise_hparam_failure_panel.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_markdown_note(runs: list[RunData], output_path: Path) -> None:
    prefix_runs = [run for run in runs if run.prefix_curve]
    all_blocks = sum(int(run_loss_summary(run)["num_blocks"]) for run in runs)
    loss_runs = len(runs)
    lines: list[str] = []
    lines.append("# Layer-wise Recovery Hyper-parameter Diagnostic")
    lines.append("")
    lines.append("## Parsed runs")
    lines.append("")
    lines.append(f"- Runs with local distillation curves: **{loss_runs}**")
    lines.append(f"- Total parsed block-level curves: **{all_blocks}**")
    lines.append(f"- Runs with matched prefix-eval curves: **{len(prefix_runs)}**")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `layerwise_hparam_summary.csv`")
    lines.append("- `layerwise_hparam_failure_panel.png/pdf`")
    lines.append("- `layerwise_hparam_local_loss_curves.png/pdf`")
    lines.append("- `layerwise_hparam_prefix_curves.png/pdf`")
    lines.append("- `layerwise_hparam_loss_vs_drop.png/pdf`")
    lines.append("- `layerwise_hparam_block_level_summary.csv`")
    lines.append("- `all_blocks_curves/*_all_blocks_loss_curves.png/pdf`")
    lines.append("- `all_blocks_heatmaps/*_all_blocks_loss_heatmap.png/pdf`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The main paper figure should use `layerwise_hparam_failure_panel.png/pdf`. It links three "
        "facts in one place: multiple reasonable hyper-parameter settings reduce the local recovery "
        "objective, the matched prefix-evaluation curves still degrade as more layers are activated, "
        "and the run-level scatter remains in the loss-down/performance-drop region."
    )
    lines.append("")
    lines.append(
        "The paper should avoid claiming exhaustive hyper-parameter search. A safer formulation is that "
        "multiple reasonable layer-wise recovery settings were examined, and the observed local-objective "
        "optimization did not reliably prevent global degradation."
    )
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| run | setting | blocks | loss ↓ blocks | mean loss reduction | prefix final drop |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for run in runs:
        loss_stats = run_loss_summary(run)
        pref_stats = prefix_summary(run.prefix_curve)
        drop = pref_stats["final_drop_points"]
        drop_str = f"{drop:.2f}" if math.isfinite(drop) else "--"
        lines.append(
            f"| `{run.run_id}` | {run.setting_label.replace('|', '/')} | "
            f"{int(loss_stats['num_blocks'])} | {int(loss_stats['loss_decreased_blocks'])} | "
            f"{loss_stats['mean_loss_reduction_pct']:.2f}% | {drop_str} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix_curves = load_prefix_curves(args.prefix_root)
    runs = load_runs(args.run_root, prefix_curves, args.max_label_len)
    if not runs:
        raise SystemExit(f"No block distillation curves found under {args.run_root}")

    write_summary_csv(runs, args.output_dir / "layerwise_hparam_summary.csv")
    write_block_level_csv(runs, args.output_dir / "layerwise_hparam_block_level_summary.csv")
    save_line_plot(runs, args.output_dir, bins=args.bins)
    save_all_block_curves(runs, args.output_dir, bins=args.bins)
    save_all_block_heatmaps(runs, args.output_dir, bins=args.bins)
    save_prefix_plot(runs, args.output_dir)
    save_scatter_plot(runs, args.output_dir)
    save_failure_panel(runs, args.output_dir, bins=args.bins)
    save_markdown_note(runs, args.output_dir / "layerwise_hparam_diagnostic.md")

    print(f"Parsed {len(runs)} runs with local distillation curves.")
    print(f"Matched prefix-eval curves for {sum(bool(run.prefix_curve) for run in runs)} runs.")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
