from __future__ import annotations

import json
from pathlib import Path

from .common import sha256, write_json
from .data import load_archive
from .protocol import CATEGORIES, DATASETS, EXPECTED_COUNTS, official_files
from .upstream import metrics

METRICS = {
    "narrativeqa": metrics.qa_f1_score, "qasper": metrics.qa_f1_score,
    "multifieldqa_en": metrics.qa_f1_score, "multifieldqa_zh": metrics.qa_f1_zh_score,
    "hotpotqa": metrics.qa_f1_score, "2wikimqa": metrics.qa_f1_score,
    "musique": metrics.qa_f1_score, "dureader": metrics.rouge_zh_score,
    "gov_report": metrics.rouge_score, "qmsum": metrics.rouge_score,
    "multi_news": metrics.rouge_score, "vcsum": metrics.rouge_zh_score,
    "trec": metrics.classification_score, "triviaqa": metrics.qa_f1_score,
    "samsum": metrics.rouge_score, "lsht": metrics.classification_score,
    "passage_retrieval_en": metrics.retrieval_score, "passage_count": metrics.count_score,
    "passage_retrieval_zh": metrics.retrieval_zh_score,
    "lcc": metrics.code_sim_score, "repobench-p": metrics.code_sim_score,
}
FIRST_LINE_TASKS = frozenset(("trec", "triviaqa", "samsum", "lsht"))


def score_example(dataset, prediction, answers, all_classes):
    # This preprocessing and max-over-references match upstream eval.py.
    if dataset in FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    score = 0.0
    for answer in answers:
        score = max(score, METRICS[dataset](prediction, answer, all_classes=all_classes))
    return float(score)


def check_prediction(prediction, source, config_sha256):
    for name in ("dataset", "row_index", "_id", "input_ids_sha256", "answers", "all_classes", "length"):
        if prediction.get(name) != source[name]:
            raise ValueError(f"Prediction/source mismatch for {source['_id']}: {name}")
    if prediction.get("config_sha256") != config_sha256 or not isinstance(prediction.get("pred"), str):
        raise ValueError("Prediction has the wrong model/config or no response.")
    if prediction.get("diagnostic") is not False:
        raise ValueError("Diagnostic predictions cannot enter the full benchmark score.")


def score_directory(manifest_path, prediction_dir, output):
    official_files()
    manifest, index = load_archive(manifest_path)
    directory = Path(prediction_dir).resolve()
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    if config["archive"]["sha256"] != sha256(manifest_path) or config["mode"] != "full":
        raise ValueError("Only full runs on this shared archive can be scored.")
    config_sha = sha256(config_path)
    by_task, paths = {}, {}
    for dataset in DATASETS:
        reference = [row for row in index if row["dataset"] == dataset]
        path = directory / "predictions" / (dataset + ".jsonl")
        with path.open() as stream:
            predictions = [json.loads(line) for line in stream]
        if len(predictions) != EXPECTED_COUNTS[dataset]:
            raise ValueError(f"Missing or extra predictions for {dataset}.")
        scores = []
        for predicted, source in zip(predictions, reference):
            check_prediction(predicted, source, config_sha)
            scores.append(score_example(dataset, predicted["pred"], source["answers"], source["all_classes"]))
        score = 100 * sum(scores) / len(scores)
        by_task[dataset] = {"samples": len(scores), "score": score,
                            "official_rounded_score": round(score, 2), "metric": METRICS[dataset].__name__}
        paths[str(path)] = sha256(path)
    categories = {name: sum(by_task[task]["score"] for task in tasks) / len(tasks)
                  for name, tasks in CATEGORIES.items()}
    report = {"schema": "longbench_v1_scores_v1", "status": "complete", "sample_count": 4750,
              "protocol": manifest["protocol"], "archive_sha256": sha256(manifest_path),
              "config_sha256": config_sha, "model": config["model"],
              "tasks": by_task, "categories": categories,
              "task_macro_mean": sum(item["score"] for item in by_task.values()) / len(by_task),
              "category_macro_mean": sum(categories.values()) / len(categories),
              "aggregation_note": "Both macros use unrounded official task metrics; these are scores, not uniform accuracy.",
              "prediction_sha256": paths}
    write_json(output, report)
    return report
