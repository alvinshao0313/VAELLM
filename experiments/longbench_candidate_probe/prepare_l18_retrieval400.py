"""Prepare the two complete official retrieval tasks for the audited L18 endpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_candidate_probe.prepare import check_candidate_preparation
from experiments.longbench_candidate_probe.prepare_l18_gate import CHECKPOINT_ID, ROOT, R2, require
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.scoring import check_prediction, score_example

TASKS = ("passage_retrieval_en", "passage_retrieval_zh")
PROOF = R2 / "l18_gate_retrieval16"
LB = ROOT / "result/qwen3_8b_longbench_20261004"
CONTROLS = {"bf16": "dense", "f927": "compressed", "norm400": "compressed_norm400"}


def read(path):
    return json.loads(Path(path).read_text())


def prepare(output_dir):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "Use a CUDA-hidden CPU preparation.")
    require(Path.cwd() == ROOT, "Use the public source root and its native LongBench runner.")
    output_dir = Path(output_dir).resolve()
    require(not any((output_dir / name).exists() for name in ("config.json", "preparation.json", "run")),
            "Use a fresh output, without predictions or a prior prepared configuration.")
    old, _, manifest = check_candidate_preparation(PROOF / "preparation.json")
    report, review = read(PROOF / "run/report.json"), read(PROOF / "review.json")
    require(report["status"] == "diagnostic_complete" and report["exit_code"] == 0
            and report["sample_count"] == 16 and report["checkpoint_id"] == CHECKPOINT_ID
            and report["frozen_state_and_identity_unchanged"] is True
            and report["input_pins_unchanged_after"] is True
            and (PROOF / "outer.exit").read_text().strip() == "0"
            and review["status"] == "verified" and review["checkpoint_id"] == CHECKPOINT_ID
            and review["report_sha256"] == sha256(PROOF / "run/report.json")
            and report["preparation_sha256"] == sha256(PROOF / "preparation.json")
            and report["predictions_sha256"] == sha256(PROOF / "run/predictions.jsonl"),
            "Require the actual successful fixed16 native generation and frozen-state proof.")
    check_hashes(review["evidence_sha256"])
    old_config = old["config"]
    config = make_config(old_config["archive"]["path"], kind="compressed",
                         hf_path=old_config["model"]["hf_path"],
                         checkpoint=old_config["model"]["checkpoint"], mode="full")
    require({k: v for k, v in config.items() if k != "mode"}
            == {k: v for k, v in old_config.items() if k != "mode"},
            "Only native evaluation mode may differ from the verified sixteen-case configuration.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    rows = {task: [row for row in index if row["dataset"] == task] for task in TASKS}
    require(all(len(items) == 200 and [r["row_index"] for r in items] == list(range(200))
                for items in rows.values()), "Require all two hundred canonical rows per task.")
    sources = dict(old["source_sha256"])
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    for path, digest in config["source_sha256"].items():
        require(path not in sources or sources[path] == digest, "Source identity differs from the native proof.")
        sources[path] = digest
    evidence = dict(old["evidence_sha256"])
    evidence.update(review["evidence_sha256"])
    evidence.update({str(path): sha256(path) for path in (PROOF / "preparation.json", PROOF / "review.json",
        PROOF / "run/report.json", PROOF / "run/predictions.jsonl", PROOF / "outer.exit")})
    controls, baseline_rows = {}, {}
    paired_fields = ("dataset", "row_index", "_id", "answers", "all_classes", "length",
                     "input_ids_sha256", "input_tokens", "removed_tokens", "max_new_tokens",
                     "generation_options", "effective_generation_options", "use_model_defaults")
    for kind, relative in CONTROLS.items():
        directory = LB / relative
        control_config, scores = read(directory / "config.json"), read(directory / "scores.json")
        config_hash = sha256(directory / "config.json")
        require(control_config["mode"] == "full" and scores["status"] == "complete"
                and scores["sample_count"] == 4750 and scores["config_sha256"] == config_hash
                and scores["archive_sha256"] == config["archive"]["sha256"]
                and all(control_config[key] == config[key] for key in ("protocol", "archive", "package_versions")),
                "Historical full-task control protocol differs.")
        require(all(control_config["model"][key] == config["model"][key]
                    for key in ("hf_path", "hf_metadata_sha256", "hf_weight_stats")),
                "Historical control has a different original model/tokenizer.")
        task_scores = {}
        for task in TASKS:
            path = directory / "predictions" / (task + ".jsonl")
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(predictions) == 200 and scores["prediction_sha256"][str(path)] == sha256(path),
                    "Historical control predictions changed or are incomplete.")
            values = []
            for predicted, source in zip(predictions, rows[task]):
                check_prediction(predicted, source, config_hash)
                require(predicted["input_tokens"] == source["input_tokens"]
                        and predicted["max_new_tokens"] == source["max_new_tokens"] == 32,
                        "Historical input or output length differs from the shared archive.")
                values.append(score_example(task, predicted["pred"], source["answers"], source["all_classes"]))
            require(100 * sum(values) / 200 == scores["tasks"][task]["score"],
                    "Historical official retrieval scores do not reproduce.")
            comparable = [{key: item[key] for key in paired_fields} for item in predictions]
            if task in baseline_rows:
                require(comparable == baseline_rows[task], "Historical controls are not paired on actual inputs/options.")
            else:
                baseline_rows[task] = comparable
            task_scores[task] = {"samples": 200, "official_score_sum": sum(values),
                                "score_percent": 100 * sum(values) / 200, "prediction_path": str(path)}
            evidence[str(path)] = sha256(path)
        for name in ("config.json", "scores.json"):
            evidence[str(directory / name)] = sha256(directory / name)
        controls[kind] = {"directory": str(directory), "checkpoint_id": control_config["model"].get("checkpoint_id"),
                          "tasks": task_scores, "model_reload_required": False}
    check_hashes(sources)
    check_hashes(evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    write_json(config_path, config)
    record = {"schema": "l18_gate_full_retrieval_preparation_v1", "status": "prepared_unbound", "gpu": None,
        "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": CHECKPOINT_ID,
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "output_dir": str(output_dir / "run"), "datasets": list(TASKS), "sample_count": 400,
        "task_samples": {task: 200 for task in TASKS},
        "selection": "All original archived rows 0..199 of each selected task, in native archive order; no subsampling.",
        "input_lengths": {task: {"min": min(r["input_tokens"] for r in items),
                                     "max": max(r["input_tokens"] for r in items)} for task, items in rows.items()},
        "source_sha256": sources, "evidence_sha256": evidence,
        "source_bit_budget": old["source_bit_budget"], "completed_candidate": old["completed_candidate"],
        "native_generation_proof": {"directory": str(PROOF), "sample_count": 16,
            "scope": "Runtime and frozen-state evidence only; sixteen-case quality is not a complete-task result."},
        "controls": controls, "timeout_seconds": 900, "termination_grace_seconds": 20,
        "environment": dict(old["environment"]),
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict",
                          "--config", str(config_path), "--output", str(output_dir / "run"), "--datasets", *TASKS],
        "queued": False, "automatic_retry": False, "resume": False,
        "scope": "Two complete official LongBench v1 retrieval tasks (400 predictions), covering 2 of 21 tasks.",
        "scoring": "Reuse longbench_v1.scoring.check_prediction and score_example; report each unrounded task score and paired controls.",
        "full_longbench_status": "not_evaluated", "nine_item_goal_status": "not_evaluated",
        "model_mutations": False, "model_exported": False, "cuda_initialized": torch.cuda.is_initialized()}
    require(record["cuda_initialized"] is False, "CPU preparation unexpectedly initialized CUDA.")
    write_json(output_dir / "preparation.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output_dir)
    print(json.dumps({"status": result["status"], "checkpoint_id": result["checkpoint_id"],
                      "sample_count": result["sample_count"], "source_count": len(result["source_sha256"]),
                      "evidence_count": len(result["evidence_sha256"]), "cuda_initialized": result["cuda_initialized"],
                      "preparation_sha256": sha256(args.output_dir / "preparation.json")}, ensure_ascii=False))
