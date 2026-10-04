"""Compare two complete LongBench v1 score artifacts; never run inference/scoring.

This entry deliberately lives outside longbench_v1, whose Python files are pinned
by active prediction jobs. Scores and differences are benchmark score points,
not a uniform accuracy percentage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from experiments.longbench_v1.common import check_hashes, sha256
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.protocol import CATEGORIES, DATASETS, EXPECTED_COUNTS, protocol_record
from experiments.longbench_v1.scoring import METRICS


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100,
             "Expected a finite benchmark score on the 0–100 scale.")
    return float(value)


def _means(tasks):
    categories = {category: math.fsum(tasks[name] for name in names) / len(names)
                  for category, names in CATEGORIES.items()}
    return categories, math.fsum(tasks.values()) / len(tasks), math.fsum(categories.values()) / len(categories)


def _validated_scores(report, kind, archive_sha256):
    _require(report.get("schema") == "longbench_v1_scores_v1" and report.get("status") == "complete"
             and report.get("sample_count") == sum(EXPECTED_COUNTS.values()) == 4750,
             "Both score artifacts must cover all 4750 examples; partial results are forbidden.")
    _require(report.get("archive_sha256") == archive_sha256 and report.get("protocol") == protocol_record(),
             "Scores must use the same verified archive and fixed LongBench v1 protocol.")
    _require(report["model"]["kind"] == kind, f"Expected the {kind} model in this input position.")
    _require(set(report["tasks"]) == set(DATASETS), "Exactly the 21 official tasks are required.")
    tasks = {}
    for name in DATASETS:
        item = report["tasks"][name]
        _require(item["samples"] == EXPECTED_COUNTS[name] and item["metric"] == METRICS[name].__name__,
                 f"Wrong sample count or official metric: {name}.")
        tasks[name] = _number(item["score"])
        _require(item["official_rounded_score"] == round(tasks[name], 2), f"Rounded score mismatch: {name}.")
    categories, task_mean, category_mean = _means(tasks)
    _require(set(report["categories"]) == set(CATEGORIES), "Exactly six categories are required.")
    values = [(report["categories"][name], value) for name, value in categories.items()]
    values += [(report["task_macro_mean"], task_mean), (report["category_macro_mean"], category_mean)]
    _require(all(math.isclose(_number(actual), expected, abs_tol=1e-10, rel_tol=0)
                 for actual, expected in values), "Stored aggregates differ from unrounded official task scores.")
    return tasks, categories, task_mean, category_mean


def compare_reports(dense, compressed, *, archive_sha256):
    """Pure aggregation over complete, independently scored artifacts."""
    left = _validated_scores(dense, "dense", archive_sha256)
    right = _validated_scores(compressed, "compressed", archive_sha256)
    _require(all(dense["model"][key] == compressed["model"][key]
                 for key in ("hf_path", "hf_metadata_sha256", "hf_weight_stats")),
             "Dense and compressed results must share the original model/tokenizer identity.")

    def pair(a, b):
        return {"dense": a, "compressed": b, "delta_score_points": b - a}

    return {
        "schema": "longbench_v1_comparison_v1", "status": "complete", "sample_count_per_model": 4750,
        "archive_sha256": archive_sha256, "protocol": dense["protocol"],
        "models": {"dense": dense["model"], "compressed": compressed["model"]},
        "tasks": {name: {"samples": EXPECTED_COUNTS[name], "metric": METRICS[name].__name__,
                          **pair(left[0][name], right[0][name])} for name in DATASETS},
        "categories": {name: {"tasks": list(CATEGORIES[name]), **pair(left[1][name], right[1][name])}
                       for name in CATEGORIES},
        "task_macro_mean": pair(left[2], right[2]), "category_macro_mean": pair(left[3], right[3]),
        "aggregation_note": "Each category is an unweighted mean of its tasks. Task macro equally weights all21 tasks; "
                            "category macro equally weights all6 category means. Use unrounded official scores. "
                            "Differences are compressed minus dense in benchmark score points, not uniform accuracy.",
    }


def _load_scores(path, manifest_path, manifest):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    predictions = report["prediction_sha256"]
    paths = [Path(name).resolve() for name in predictions]
    _require(len(paths) == len(DATASETS) and {p.name for p in paths} == {name + ".jsonl" for name in DATASETS}
             and len({p.parent for p in paths}) == 1 and paths[0].parent.name == "predictions",
             "Official score provenance must contain exactly one prediction file per task.")
    config_path = paths[0].parent.parent / "config.json"
    config = json.loads(config_path.read_text())
    _require(sha256(config_path) == report["config_sha256"] and config["schema"] == "longbench_v1_run_v1"
             and config["mode"] == "full" and config["model"] == report["model"]
             and config["protocol"] == report["protocol"] == manifest["protocol"]
             and config["archive"]["sha256"] == sha256(manifest_path)
             and Path(config["archive"]["path"]).resolve() == Path(manifest_path).resolve(),
             "Score/model/config/archive provenance does not match.")
    check_hashes(predictions)
    return report, {"path": str(path), "sha256": sha256(path), "config_path": str(config_path),
                    "config_sha256": sha256(config_path), "prediction_sha256": predictions}


def comparison(dense_scores, compressed_scores, manifest_path):
    manifest, _ = load_archive(manifest_path)
    dense, dense_evidence = _load_scores(dense_scores, manifest_path, manifest)
    compressed, compressed_evidence = _load_scores(compressed_scores, manifest_path, manifest)
    result = compare_reports(dense, compressed, archive_sha256=sha256(manifest_path))
    result["inputs"] = {"manifest": str(Path(manifest_path).resolve()),
                        "dense_scores": dense_evidence, "compressed_scores": compressed_evidence,
                        "official_scoring_recomputed": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-scores", required=True, type=Path)
    parser.add_argument("--compressed-scores", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    outputs = [args.output] + ([args.csv] if args.csv else [])
    _require(len({p.resolve() for p in outputs}) == len(outputs), "Output paths must differ.")
    for path in outputs:
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to replace an existing comparison artifact: {path}")
    result = comparison(args.dense_scores, args.compressed_scores, args.manifest)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    if args.csv:
        fields = ("level", "name", "metric", "dense", "compressed", "delta_score_points")
        with args.csv.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for level in ("tasks", "categories"):
                for name, value in result[level].items():
                    writer.writerow({"level": level, "name": name,
                        "metric": value.get("metric", "unweighted_task_mean"),
                        **{key: value[key] for key in fields[3:]}})
            for name in ("task_macro_mean", "category_macro_mean"):
                writer.writerow({"level": "macro", "name": name, "metric": name, **result[name]})
    print(json.dumps({name: result[name] for name in ("task_macro_mean", "category_macro_mean")}))


if __name__ == "__main__":
    main()
