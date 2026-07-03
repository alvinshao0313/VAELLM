#!/usr/bin/env python3
"""Summarize channel_residual_vae hyperparameter search results."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

EVAL_MEAN_RE = re.compile(
    r"类别\s+(?P<category>[^/]+)/after_residual\s+下游任务均值:\s+(?P<mean>[0-9.]+)"
)


@dataclass(frozen=True)
class TrialRecord:
    category: str
    phase: str
    run_dir: str
    scope: str
    protect_count: int
    min_per_layer: int
    stages: int
    steps: int
    lr: float
    codebook_bits: int
    eval_mean: Optional[float]
    train_time_sec: Optional[float]
    protected_residual_rms_after: Optional[float]
    residual_vae_final_recon: Optional[float]
    status: str


def _parse_run_dir_name(name: str) -> Dict[str, Any]:
    parts: Dict[str, Any] = {}
    for token in name.split("_"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in {"pc", "mpl", "st", "sp", "cb"}:
            parts[key] = int(value)
        elif key == "lr":
            parts[key] = float(value)
        else:
            parts[key] = value
    return parts


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_eval_mean(log_path: str, category: str) -> Optional[float]:
    if not os.path.isfile(log_path):
        return None
    last: Optional[float] = None
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = EVAL_MEAN_RE.search(line)
            if match is None:
                continue
            if match.group("category") != category:
                continue
            last = float(match.group("mean"))
    return last


def load_eval_mean(run_dir: str, category: str) -> Optional[float]:
    eval_result = _read_json(os.path.join(run_dir, "eval_result.json"))
    if isinstance(eval_result, dict) and eval_result.get("eval_mean") is not None:
        return float(eval_result["eval_mean"])
    for log_name in ("eval.log", "residual_from_base.log"):
        mean = parse_eval_mean(os.path.join(run_dir, log_name), category)
        if mean is not None:
            return mean
    return None


def collect_trial(run_dir: str, category: str, phase: str) -> TrialRecord:
    name = os.path.basename(run_dir.rstrip(os.sep))
    parsed = _parse_run_dir_name(name)
    completed = _read_json(os.path.join(run_dir, "completed.json"))
    metrics = _read_json(os.path.join(run_dir, "metrics.json"))

    eval_mean = load_eval_mean(run_dir, category)
    train_time_sec = None
    protected_residual_rms_after = None
    residual_vae_final_recon = None
    if metrics is not None:
        train_time_sec = metrics.get("train_time_sec")
        cat_metrics = (metrics.get("categories") or {}).get(category) or {}
        protected_residual_rms_after = cat_metrics.get("protected_residual_rms_after")
        residual_vae_final_recon = cat_metrics.get("residual_vae_final_recon")

    if completed and completed.get("completed"):
        status = "completed" if eval_mean is not None else "completed_no_eval"
    elif os.path.isdir(run_dir):
        status = "incomplete"
    else:
        status = "missing"

    return TrialRecord(
        category=category,
        phase=phase,
        run_dir=run_dir,
        scope=str(parsed.get("scope", "?")),
        protect_count=int(parsed.get("pc", -1)),
        min_per_layer=int(parsed.get("mpl", -1)),
        stages=int(parsed.get("st", -1)),
        steps=int(parsed.get("sp", -1)),
        lr=float(parsed.get("lr", float("nan"))),
        codebook_bits=int(parsed.get("cb", -1)),
        eval_mean=eval_mean,
        train_time_sec=float(train_time_sec) if train_time_sec is not None else None,
        protected_residual_rms_after=(
            float(protected_residual_rms_after) if protected_residual_rms_after is not None else None
        ),
        residual_vae_final_recon=(
            float(residual_vae_final_recon) if residual_vae_final_recon is not None else None
        ),
        status=status,
    )


def discover_trials(search_root: str, categories: Sequence[str]) -> List[TrialRecord]:
    records: List[TrialRecord] = []
    for category in categories:
        for phase in ("phase1", "phase2"):
            phase_dir = os.path.join(search_root, category, phase)
            if not os.path.isdir(phase_dir):
                continue
            for entry in sorted(os.listdir(phase_dir)):
                run_dir = os.path.join(phase_dir, entry)
                if not os.path.isdir(run_dir):
                    continue
                records.append(collect_trial(run_dir, category, phase))
    return records


def _format_lr(lr: float) -> str:
    if lr == int(lr):
        return str(int(lr))
    text = f"{lr:.0e}" if lr < 0.01 else f"{lr:g}"
    return text


def _format_hours(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    return f"{seconds / 3600.0:.2f}h"


def _trial_sort_key(record: TrialRecord) -> tuple:
    mean = record.eval_mean if record.eval_mean is not None else -1.0
    return (-mean, record.run_dir)


def _render_table(records: Sequence[TrialRecord]) -> List[str]:
    header = (
        "| rank | scope | pc | mpl | st | sp | lr | cb | eval_mean | rms_after | recon | time | status |"
    )
    sep = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for idx, record in enumerate(sorted(records, key=_trial_sort_key), start=1):
        lines.append(
            "| {rank} | {scope} | {pc} | {mpl} | {st} | {sp} | {lr} | {cb} | {mean} | {rms} | {recon} | {time} | {status} |".format(
                rank=idx,
                scope=record.scope,
                pc=record.protect_count,
                mpl=record.min_per_layer,
                st=record.stages,
                sp=record.steps,
                lr=_format_lr(record.lr),
                cb=record.codebook_bits,
                mean=f"{record.eval_mean:.4f}" if record.eval_mean is not None else "N/A",
                rms=(
                    f"{record.protected_residual_rms_after:.6f}"
                    if record.protected_residual_rms_after is not None
                    else "N/A"
                ),
                recon=(
                    f"{record.residual_vae_final_recon:.6f}"
                    if record.residual_vae_final_recon is not None
                    else "N/A"
                ),
                time=_format_hours(record.train_time_sec),
                status=record.status,
            )
        )
    return lines


def _best_completed(records: Sequence[TrialRecord]) -> List[TrialRecord]:
    completed = [r for r in records if r.eval_mean is not None and r.status.startswith("completed")]
    return sorted(completed, key=_trial_sort_key)


def _render_cli_snippet(record: TrialRecord) -> str:
    return "\n".join(
        [
            f'  --target_categories "{record.category}" \\',
            f'  --outlier_channel_scope "{record.scope}" \\',
            f'  --outlier_protect_count "{record.protect_count}" \\',
            f'  --outlier_protect_min_per_layer "{record.min_per_layer}" \\',
            f'  --outlier_residual_vae_stages "{record.stages}" \\',
            f'  --outlier_residual_vae_steps "{record.steps}" \\',
            f'  --outlier_residual_vae_lr "{_format_lr(record.lr)}" \\',
            f'  --outlier_residual_vae_codebook_bits "{record.codebook_bits}"',
        ]
    )


def find_phase1_anchor(records: Sequence[TrialRecord], category: str) -> Optional[TrialRecord]:
    phase1 = [r for r in records if r.category == category and r.phase == "phase1"]
    best = _best_completed(phase1)
    return best[0] if best else None


def build_summary_markdown(search_root: str, categories: Sequence[str]) -> str:
    records = discover_trials(search_root, categories)
    lines: List[str] = [
        "# channel_residual_vae Hyperparameter Search Summary",
        "",
        f"Search root: `{search_root}`",
        "",
    ]

    for category in categories:
        cat_records = [r for r in records if r.category == category]
        phase1 = [r for r in cat_records if r.phase == "phase1"]
        phase2 = [r for r in cat_records if r.phase == "phase2"]
        best_all = _best_completed(cat_records)

        lines.extend([f"## {category}", ""])
        lines.append(f"- trials: phase1={len(phase1)}, phase2={len(phase2)}")
        anchor = find_phase1_anchor(records, category)
        if anchor is not None:
            lines.append(
                f"- phase1 anchor: eval_mean={anchor.eval_mean:.4f}, "
                f"scope={anchor.scope}, pc={anchor.protect_count}, cb={anchor.codebook_bits}"
            )
        if best_all:
            best = best_all[0]
            phase1_best = _best_completed(phase1)
            phase2_best = _best_completed(phase2)
            if phase1_best and phase2_best:
                delta = phase2_best[0].eval_mean - phase1_best[0].eval_mean
                lines.append(
                    f"- phase1 best eval_mean={phase1_best[0].eval_mean:.4f}, "
                    f"phase2 best eval_mean={phase2_best[0].eval_mean:.4f}, delta={delta:+.4f}"
                )
            lines.append(f"- overall best eval_mean={best.eval_mean:.4f} ({best.phase})")
        lines.append("")

        lines.append("### Phase 1")
        lines.extend(_render_table(phase1))
        lines.append("")
        lines.append("### Phase 2")
        lines.extend(_render_table(phase2))
        lines.append("")

        top3 = _best_completed(cat_records)[:3]
        if top3:
            lines.append("### Top-3")
            for idx, record in enumerate(top3, start=1):
                lines.append(
                    f"{idx}. `{os.path.basename(record.run_dir)}` "
                    f"eval_mean={record.eval_mean:.4f} ({record.phase})"
                )
            lines.append("")

        if best_all:
            lines.append("### Recommended CLI overrides")
            lines.append("```bash")
            lines.append(_render_cli_snippet(best_all[0]))
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize cat residual hparam search.")
    parser.add_argument(
        "--search_root",
        type=str,
        default=".result/catlora_residual_from_base/hparam_search",
    )
    parser.add_argument("--categories", type=str, default="up_proj,gate_proj")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args(argv)

    search_root = os.path.abspath(args.search_root)
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    output_path = args.output.strip() or os.path.join(search_root, "HPARAM_SEARCH_SUMMARY.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    markdown = build_summary_markdown(search_root, categories)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
