#!/usr/bin/env python3
"""Summarize per-category codebook A/B results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_A = "b32d32s2"
CONFIG_B = "b64d32s1"
DEFAULT_CATEGORIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
EVAL_TASKS = (
    "boolq",
    "rte",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "piqa",
    "mmlu",
)


@dataclass(frozen=True)
class EvalRecord:
    category: str
    config: str
    wiki_ppl: Optional[float]
    task_mean: Optional[float]
    task_metrics: Dict[str, float]
    status: str
    run_dir: str


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_eval_record(run_dir: str, category: str, config: str) -> EvalRecord:
    eval_result = _read_json(os.path.join(run_dir, "eval_result.json"))
    completed = _read_json(os.path.join(run_dir, "completed.json"))

    wiki_ppl = None
    task_mean = None
    task_metrics: Dict[str, float] = {}
    if isinstance(eval_result, dict):
        if eval_result.get("wiki_ppl") is not None:
            wiki_ppl = float(eval_result["wiki_ppl"])
        if eval_result.get("task_mean") is not None:
            task_mean = float(eval_result["task_mean"])
        raw_metrics = eval_result.get("task_metrics") or {}
        if isinstance(raw_metrics, dict):
            task_metrics = {str(k): float(v) for k, v in raw_metrics.items() if v is not None}

    if completed and completed.get("completed") and wiki_ppl is not None and task_mean is not None:
        status = "completed"
    elif os.path.isdir(run_dir):
        status = "incomplete"
    else:
        status = "missing"

    return EvalRecord(
        category=category,
        config=config,
        wiki_ppl=wiki_ppl,
        task_mean=task_mean,
        task_metrics=task_metrics,
        status=status,
        run_dir=run_dir,
    )


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _ppl_winner(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "?"
    if a < b:
        return CONFIG_A
    if b < a:
        return CONFIG_B
    return "tie"


def _task_winner(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "?"
    if a > b:
        return CONFIG_A
    if b > a:
        return CONFIG_B
    return "tie"


def summarize(search_root: str, categories: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append(f"# Codebook A/B Summary")
    lines.append("")
    lines.append(f"- search_root: `{search_root}`")
    lines.append(f"- A: `{CONFIG_A}` = bits32 / dim32 / stages2")
    lines.append(f"- B: `{CONFIG_B}` = bits64 / dim32 / stages1")
    lines.append("")
    lines.append("| category | A PPL | B PPL | PPL winner | A task_mean | B task_mean | task winner | note |")
    lines.append("|---|---:|---:|---|---:|---:|---|---|")

    for category in categories:
        rec_a = load_eval_record(os.path.join(search_root, category, CONFIG_A), category, CONFIG_A)
        rec_b = load_eval_record(os.path.join(search_root, category, CONFIG_B), category, CONFIG_B)
        ppl_win = _ppl_winner(rec_a.wiki_ppl, rec_b.wiki_ppl)
        task_win = _task_winner(rec_a.task_mean, rec_b.task_mean)
        note = ""
        if rec_a.status != "completed" or rec_b.status != "completed":
            note = f"A={rec_a.status},B={rec_b.status}"
        elif ppl_win != task_win and ppl_win != "?" and task_win != "?":
            note = "conflict"
        lines.append(
            "| {cat} | {ap} | {bp} | {pw} | {am} | {bm} | {tw} | {note} |".format(
                cat=category,
                ap=_fmt(rec_a.wiki_ppl, 2),
                bp=_fmt(rec_b.wiki_ppl, 2),
                pw=ppl_win,
                am=_fmt(rec_a.task_mean, 4),
                bm=_fmt(rec_b.task_mean, 4),
                tw=task_win,
                note=note,
            )
        )

    lines.append("")
    lines.append("## Per-task deltas (B - A)")
    lines.append("")
    header = "| category | " + " | ".join(EVAL_TASKS) + " |"
    sep = "|---|" + "|".join(["---:" for _ in EVAL_TASKS]) + "|"
    lines.append(header)
    lines.append(sep)
    for category in categories:
        rec_a = load_eval_record(os.path.join(search_root, category, CONFIG_A), category, CONFIG_A)
        rec_b = load_eval_record(os.path.join(search_root, category, CONFIG_B), category, CONFIG_B)
        cells: List[str] = [category]
        for task in EVAL_TASKS:
            a_val = rec_a.task_metrics.get(task)
            b_val = rec_b.task_metrics.get(task)
            if a_val is None or b_val is None:
                cells.append("N/A")
            else:
                cells.append(_fmt(b_val - a_val, 4))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize codebook A/B results.")
    parser.add_argument(
        "--search_root",
        type=str,
        default=".result/catlora_codebook_ab",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
    )
    parser.add_argument(
        "--write_md",
        type=str,
        default="",
        help="Optional markdown output path. Default: <search_root>/CODEBOOK_AB_SUMMARY.md",
    )
    args = parser.parse_args(argv)

    search_root = args.search_root
    if not os.path.isabs(search_root):
        search_root = os.path.abspath(os.path.join(_REPO_ROOT, search_root))
    else:
        search_root = os.path.abspath(search_root)

    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    text = summarize(search_root, categories)
    print(text)

    out_path = args.write_md.strip() or os.path.join(search_root, "CODEBOOK_AB_SUMMARY.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
