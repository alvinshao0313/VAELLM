"""Actual LM-eval accuracy and paired item-level comparisons; offline only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

TASKS = ("piqa", "arc_easy", "arc_challenge")
PRIMARY_METRIC = "acc_norm"


def dump_json(path: Path, value) -> None:
    def encode(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (torch.dtype, torch.device, Path)):
            return str(obj)
        if callable(obj):
            return f"{obj.__module__}.{obj.__qualname__}"
        raise TypeError(f"Cannot serialize {type(obj)}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode) + "\n")


def load_lm(model_id: str, device: str, batch_size: int):
    from lm_eval.models.huggingface import HFLM
    root = Path(model_id)
    if not root.is_dir():
        root = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    lm = HFLM(
        pretrained=str(root), backend="causal", device=device, batch_size=batch_size,
        dtype="bfloat16", use_fast_tokenizer=False, attn_implementation="sdpa",
    )
    lm.model.eval().requires_grad_(False)
    return lm, str(root)


def evaluate(lm, *, limit: int, batch_size: int, seed: int, output: Path):
    from lm_eval import evaluator
    with torch.no_grad():
        result = evaluator.simple_evaluate(
            model=lm, tasks=list(TASKS), num_fewshot=0, batch_size=batch_size,
            limit=limit, log_samples=True, bootstrap_iters=0,
            random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
            fewshot_random_seed=seed, apply_chat_template=False,
        )
    dump_json(output / "lm_eval_raw.json", result)
    metrics, items = {}, {}
    for task in TASKS:
        task_result = result["results"][task]
        primary = float(task_result[PRIMARY_METRIC + ",none"])
        rows = result["samples"][task]
        if len(rows) != limit:
            raise RuntimeError(f"{task}: expected {limit} paired items, got {len(rows)}.")
        parsed = []
        for row in rows:
            doc_json = json.dumps(row["doc"], sort_keys=True, ensure_ascii=False)
            parsed.append({
                "doc_id": row["doc_id"],
                "doc_sha256": hashlib.sha256(doc_json.encode()).hexdigest(),
                "correct": int(row[PRIMARY_METRIC]), "acc": int(row["acc"]),
            })
        actual = sum(r["correct"] for r in parsed) / len(parsed)
        if abs(primary - actual) > 1e-10:
            raise RuntimeError("Sample-level accuracy disagrees with aggregate metric.")
        items[task] = parsed
        metrics[task] = {
            "metric": PRIMARY_METRIC, "accuracy": primary,
            "correct": sum(r["correct"] for r in parsed), "n": len(parsed),
            "acc": float(task_result["acc,none"]),
        }
    summary = {
        "tasks": metrics, "items": items,
        "macro_accuracy": sum(t["accuracy"] for t in metrics.values()) / len(metrics),
        "num_fewshot": 0, "chat_template": False,
        "sampling": "Fixed first N official evaluation documents per task; same IDs for all variants.",
    }
    dump_json(output / "accuracy.json", summary)
    print("ACCURACY " + json.dumps({k: v["accuracy"] for k, v in metrics.items()}), flush=True)
    return summary


def paired_comparison(reference: dict, candidate: dict, seed: int = 31) -> dict:
    rng = np.random.default_rng(seed)
    grouped, task_comparisons = [], {}
    for task in TASKS:
        a = reference["items"][task]
        b = candidate["items"][task]
        if [(r["doc_id"], r["doc_sha256"]) for r in a] != [(r["doc_id"], r["doc_sha256"]) for r in b]:
            raise ValueError("A/B tasks or document order differ.")
        diff = np.asarray([rb["correct"] - ra["correct"] for ra, rb in zip(a, b)], dtype=np.int8)
        grouped.append(diff)
        task_comparisons[task] = {
            "delta_pp": float(diff.mean() * 100),
            "candidate_only_correct": int((diff == 1).sum()),
            "reference_only_correct": int((diff == -1).sum()),
        }
    combined = np.concatenate(grouped)
    wins, losses = int((combined == 1).sum()), int((combined == -1).sum())
    discordant = wins + losses
    pvalue = 1.0 if discordant == 0 else min(
        1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant,
    )
    bootstrap = np.zeros(10000)
    for diff in grouped:
        ids = rng.integers(0, len(diff), size=(len(bootstrap), len(diff)))
        bootstrap += diff[ids].mean(axis=1) / len(grouped)
    ci = np.quantile(bootstrap, [0.025, 0.975]) * 100
    return {
        "tasks": task_comparisons,
        "macro_delta_pp": float(np.mean([d.mean() for d in grouped]) * 100),
        "candidate_only_correct": wins, "reference_only_correct": losses,
        "paired_bootstrap_95ci_pp": ci.tolist(), "mcnemar_exact_two_sided_p": pvalue,
        "scope": "Question-sampling uncertainty conditional on this training seed; not training-seed uncertainty.",
    }
