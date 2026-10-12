"""Score the fixed sensitive LongBench tasks without creating a full benchmark result."""
from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.longbench_v1 import common, data, protocol, scoring


def load_selection(path):
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if (selection["schema"] != "longbench_v1_sensitive_selection_v1"
            or selection["is_full_benchmark"] is not False
            or selection["purpose"] != "checkpoint_budget_screen_only"
            or selection["classification_rule"]["high_if_drop_greater_or_equal_pp"] != 5.0
            or selection["classification_rule"]["fixed_before_future_candidate_results"] is not True):
        raise ValueError("Use the fixed 5pp partial-screen selection manifest.")
    baselines = {}
    for name in ("bf16_scores", "source_scores"):
        record = selection["sources"][name]
        if common.sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Selection baseline changed: {name}")
        value = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        if (value["schema"] != "longbench_v1_scores_v1" or value["status"] != "complete"
                or value["sample_count"] != 4750 or set(value["tasks"]) != set(protocol.DATASETS)
                or value["archive_sha256"] != selection["archive_sha256"]):
            raise ValueError("Selection must bind complete scores on the same original archive.")
        baselines[name] = value
    if baselines["source_scores"]["model"]["checkpoint_id"] != selection["source_checkpoint_id"]:
        raise ValueError("Selection source UUID differs from its complete baseline.")
    high, low = [], []
    if set(selection["tasks"]) != set(protocol.DATASETS):
        raise ValueError("Selection must classify all 21 official tasks.")
    for task in protocol.DATASETS:
        record = selection["tasks"][task]
        bf16 = baselines["bf16_scores"]["tasks"][task]
        source = baselines["source_scores"]["tasks"][task]
        drop = bf16["score"] - source["score"]
        tier = "high" if drop >= 5.0 else "low"
        if (record["samples"] != protocol.EXPECTED_COUNTS[task]
                or bf16["samples"] != record["samples"] or source["samples"] != record["samples"]
                or record["bf16_score"] != bf16["score"] or record["source_score"] != source["score"]
                or record["absolute_drop_pp"] != drop or record["tier"] != tier):
            raise ValueError(f"Selection provenance/count/tier mismatch: {task}")
        (high if tier == "high" else low).append(task)
    if (selection["high_tasks"] != high or selection["low_tasks"] != low
            or len(high) != 12 or len(low) != 9
            or (selection["high_samples"], sum(protocol.EXPECTED_COUNTS[x] for x in high)) != (2950, 2950)
            or (selection["low_samples"], sum(protocol.EXPECTED_COUNTS[x] for x in low)) != (1800, 1800)):
        raise ValueError("The fixed official-order 12/2950 high and 9/1800 low partition changed.")
    return selection


def score_task(task, predictions, references, config_sha256):
    expected = protocol.EXPECTED_COUNTS[task]
    if len(predictions) != expected or len(references) != expected:
        raise ValueError(f"Missing or extra predictions for {task}: expected {expected}.")
    values = []
    for predicted, source in zip(predictions, references):
        scoring.check_prediction(predicted, source, config_sha256)
        values.append(scoring.score_example(task, predicted["pred"], source["answers"], source["all_classes"]))
    score = 100 * sum(values) / len(values)
    return {"samples": expected, "score": score, "official_rounded_score": round(score, 2),
            "metric": scoring.METRICS[task].__name__}


def score_directory(selection_path, archive_path, prediction_dir, output):
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Keep original metrics; use a new partial report: {output}")
    selection_path, archive_path = Path(selection_path).resolve(), Path(archive_path).resolve()
    selection = load_selection(selection_path)
    archive_sha = common.sha256(archive_path)
    if archive_sha != selection["archive_sha256"]:
        raise ValueError("Screen archive differs from the fixed selection archive.")
    manifest, rows = data.load_archive(archive_path)
    directory = Path(prediction_dir).resolve()
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["schema"] != "longbench_v1_run_v1" or config["mode"] != "full"
            or config["archive"]["sha256"] != archive_sha
            or Path(config["archive"]["path"]).resolve() != archive_path
            or config["protocol"] != manifest["protocol"]
            or config["model"]["kind"] not in ("dense", "compressed")
            or Path(config["model"]["hf_path"]).resolve() != Path(manifest["tokenizer"]["path"]).resolve()):
        raise ValueError("Only full-mode original-protocol configs on this archive/model family can be screened.")
    if config["model"]["kind"] == "compressed" and not config["model"].get("checkpoint_id"):
        raise ValueError("Compressed prediction config requires its checkpoint UUID.")
    config_sha = common.sha256(config_path)
    tasks, prediction_hashes = {}, {}
    for task in selection["high_tasks"]:
        references = [row for row in rows if row["dataset"] == task]
        path = directory / "predictions" / (task + ".jsonl")
        with path.open(encoding="utf-8") as stream:
            predictions = [json.loads(line) for line in stream]
        tasks[task] = score_task(task, predictions, references, config_sha)
        prediction_hashes[str(path)] = common.sha256(path)
    source_files = [Path(__file__), Path(__file__).with_name("run.py"),
                    Path(common.__file__), Path(data.__file__), Path(protocol.__file__),
                    Path(scoring.__file__), Path(scoring.metrics.__file__)]
    report = {
        "schema": "longbench_v1_sensitive_screen_v1", "status": "partial_screen_complete",
        "is_full_benchmark": False, "kind": "partial", "purpose": "checkpoint_budget_screen_only",
        "selection": {"path": str(selection_path), "sha256": common.sha256(selection_path)},
        "fixed_source_checkpoint_id": selection["source_checkpoint_id"],
        "selected_tasks": selection["high_tasks"], "task_count": len(tasks),
        "sample_count": sum(row["samples"] for row in tasks.values()), "tasks": tasks,
        "high_macro_mean": sum(row["score"] for row in tasks.values()) / len(tasks),
        "aggregation_note": "Equal task mean of unrounded official metrics; low tasks are unscored, not zero.",
        "acceptance_note": "Partial budget screen only; final acceptance requires all 21 tasks and the complete nine metrics.",
        "archive": {"path": str(archive_path), "sha256": archive_sha}, "protocol": manifest["protocol"],
        "prediction_dir": str(directory), "config_sha256": config_sha, "model": config["model"],
        "prediction_sha256": prediction_hashes,
        "sourcefiles_sha256": {str(path.resolve()): common.sha256(path) for path in source_files},
        "duration_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(output, report)
    return report
