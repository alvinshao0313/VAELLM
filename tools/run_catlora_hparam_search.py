#!/usr/bin/env python3
"""Resumable, dependency-free multi-fidelity search for the catlora pipeline.

The search deliberately keeps the nominal weight budget at strict 2-bit:
all BSQ categories use ``codebook_bits=32, codebook_dim=32, stages=2``.
It searches recovery and reconstruction choices first, then promotes the
best short trials to the full training recipe.  The runner uses the existing
``scripts/catlora_simple2.sh`` entry point and never reuses its output path.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts" / "catlora_simple2.sh"
TASKS = (
    "boolq",
    "rte",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "piqa",
    "mmlu",
)

TASK_RE = re.compile(
    r"类别\s+none\s+下游任务\s+(?P<task>\S+):\s+\S+\s+=\s+"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class Trial:
    name: str
    recon_loss: str
    mlp_metric: str
    recovery_mode: str
    lora_rank: int
    lora_dropout: float
    top_k: int
    temperature: float
    hidden_loss_weight: float
    pre_mlp_hidden_loss_weight: float
    vae_steps: int
    distill_steps: int
    eval_limit: int | None

    def run_dir(self, root: Path) -> Path:
        return root / self.name


def screen_trials() -> list[Trial]:
    """Six structural trials, followed by two recovery-capacity trials."""
    trials: list[Trial] = []
    for recon in ("mse", "wa_mse", "amse"):
        for mlp in ("none", "mlp_intermediate_aligned_actrms_abs"):
            trials.append(
                Trial(
                    name=f"struct_{recon}_{'aligned' if mlp != 'none' else 'independent'}",
                    recon_loss=recon,
                    mlp_metric=mlp,
                    recovery_mode="remaining_lora_current_decoder",
                    lora_rank=12,
                    lora_dropout=0.1,
                    top_k=100,
                    temperature=1.0,
                    hidden_loss_weight=0.1,
                    pre_mlp_hidden_loss_weight=0.01,
                    vae_steps=1000,
                    distill_steps=500,
                    eval_limit=64,
                )
            )
    trials.extend(
        [
            Trial(
                name="recovery_prefix_wa_mse_aligned",
                recon_loss="wa_mse",
                mlp_metric="mlp_intermediate_aligned_actrms_abs",
                recovery_mode="remaining_lora_prefix_decoder",
                lora_rank=12,
                lora_dropout=0.1,
                top_k=100,
                temperature=1.0,
                hidden_loss_weight=0.1,
                pre_mlp_hidden_loss_weight=0.01,
                vae_steps=1000,
                distill_steps=500,
                eval_limit=64,
            ),
            Trial(
                name="capacity_wa_mse_aligned",
                recon_loss="wa_mse",
                mlp_metric="mlp_intermediate_aligned_actrms_abs",
                recovery_mode="remaining_lora_current_decoder",
                lora_rank=24,
                lora_dropout=0.03,
                top_k=1000,
                temperature=2.0,
                hidden_loss_weight=0.03,
                pre_mlp_hidden_loss_weight=0.0,
                vae_steps=1000,
                distill_steps=500,
                eval_limit=64,
            ),
        ]
    )
    return trials


def promoted_trials(records: Iterable[dict[str, Any]], count: int) -> list[Trial]:
    completed = [r for r in records if r.get("status") == "completed" and r.get("score") is not None]
    completed.sort(key=lambda r: float(r["score"]), reverse=True)
    result: list[Trial] = []
    for record in completed[: max(0, int(count))]:
        params = dict(record.get("params") or {})
        result.append(
            Trial(
                name=f"full_{params['name']}",
                recon_loss=str(params["recon_loss"]),
                mlp_metric=str(params["mlp_metric"]),
                recovery_mode=str(params["recovery_mode"]),
                lora_rank=int(params["lora_rank"]),
                lora_dropout=float(params["lora_dropout"]),
                top_k=int(params["top_k"]),
                temperature=float(params["temperature"]),
                hidden_loss_weight=float(params["hidden_loss_weight"]),
                pre_mlp_hidden_loss_weight=float(params["pre_mlp_hidden_loss_weight"]),
                vae_steps=10000,
                distill_steps=5000,
                eval_limit=None,
            )
        )
    return result


def command_for(trial: Trial, run_dir: Path) -> list[str]:
    # The final override explicitly resets the old simple2 down_proj=64 entry.
    args = [
        "bash",
        str(BASE_SCRIPT),
        # The short stage focuses on the empirically sensitive SwiGLU
        # categories.  Promotion returns to the complete q/k/v/o/gate/up/down
        # pipeline so the final decision still covers the real run.
        *("--compression_categories", "gate_proj,up_proj,down_proj")
        if trial.eval_limit is not None
        else (),
        "--output_dir",
        str(run_dir),
        "--codebook_bits",
        "default=32,cat:down_proj=32",
        "--codebook_dim",
        "default=32",
        "--residual_stages",
        "default=2",
        "--recon_loss_type",
        f"default={trial.recon_loss}",
        "--channel_mlp_rank_metric",
        trial.mlp_metric,
        "--after_category_mode",
        trial.recovery_mode,
        "--distill_fp32_components",
        "lora,decoder",
        "--lora_rank",
        f"default={trial.lora_rank}",
        "--lora_alpha",
        f"default={2 * trial.lora_rank}",
        "--lora_dropout",
        f"default={trial.lora_dropout:g}",
        "--steps",
        f"default={trial.distill_steps}",
        "--vae_steps",
        f"default={trial.vae_steps}",
        "--top_k",
        f"default={trial.top_k}",
        "--temperature",
        f"default={trial.temperature:g}",
        "--hidden_loss_weight",
        f"default={trial.hidden_loss_weight:g}",
        "--pre_mlp_hidden_loss_weight",
        f"default={trial.pre_mlp_hidden_loss_weight:g}",
        "--eval_limit",
        str(trial.eval_limit) if trial.eval_limit is not None else "",
    ]
    if trial.eval_limit is None:
        args[-2:] = []
    return args


def parse_scores(log_path: Path) -> dict[str, float]:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, float] = {}
    for match in TASK_RE.finditer(text):
        values[match.group("task")] = float(match.group("value"))
    return {task: values[task] for task in TASKS if task in values}


def score(values: dict[str, float]) -> float | None:
    if len(values) != len(TASKS):
        return None
    ordered = [values[name] for name in TASKS]
    mean = sum(ordered) / len(ordered)
    spread = max(ordered) - min(ordered)
    # Mean quality matters, while the spread penalty prevents one task from
    # being sacrificed to improve the aggregate.
    return float(mean - 0.5 * spread)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def run_trial(trial: Trial, root: Path, manifest: Path, *, dry_run: bool) -> None:
    run_dir = trial.run_dir(root)
    result_path = run_dir / "tuner_result.json"
    if result_path.is_file():
        print(f"[skip] {trial.name}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(trial, run_dir)
    record: dict[str, Any] = {
        "name": trial.name,
        "params": asdict(trial),
        "run_dir": str(run_dir),
        "command": command,
        "started_at": time.time(),
        "status": "dry_run" if dry_run else "started",
    }
    if dry_run:
        append_record(manifest, record)
        print(" ".join(command))
        return

    log_path = run_dir / "tuner.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    values = parse_scores(run_dir / "linear_by_category.log")
    result = {
        **record,
        "finished_at": time.time(),
        "status": "completed" if proc.returncode == 0 else "failed",
        "exit_code": int(proc.returncode),
        "task_scores": values,
        "score": score(values),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_record(manifest, result)
    if proc.returncode != 0:
        raise RuntimeError(f"trial failed: {trial.name}, see {log_path}")
    print(f"[done] {trial.name} score={result['score']} tasks={values}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "full"), default="screen")
    parser.add_argument("--search_root", default="/root/data/ckpts/result/catlora/tuning_2bit")
    parser.add_argument("--promote", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.search_root).resolve()
    manifest = root / "manifest.jsonl"
    records = read_records(manifest)
    if args.stage == "screen":
        trials = screen_trials()
    else:
        trials = promoted_trials(records, args.promote)
        if not trials:
            raise SystemExit("No completed screen trials found; run --stage screen first.")
    for trial in trials:
        run_trial(trial, root, manifest, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
