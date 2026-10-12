"""Bind the decoder-only FineWeb1600 Llama checkpoint to official retrieval400."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_candidate_probe.prepare_llama_best1600_d16 import DENSE, ROOT, R2, TASKS, read, require
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.scoring import check_prediction, score_example

STEP = R2 / "general_lm_recovery_s2000/llama/step_1600"
UUID = "dbde405a-328e-4bd2-84d3-f18ad60e3049"
BASE_UUID = "35776715-2707-45a9-9fb4-2da1efef3f0e"
CONTROLS = (("bf16", DENSE), ("original3577", R2 / "llama_original3577_retrieval400"),
            ("e767", R2 / "llama_best1600_d16_retrieval400"))


def prepare(output):
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not torch.cuda.is_initialized(), "Require public cwd and CUDA-hidden bitvae.")
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh output; no resume or overwrite.")
    checkpoint = STEP / "final_model"
    require((checkpoint / "pytorch_model.bin").is_file(), "The actual D1600 checkpoint must still exist.")
    launch = read(STEP.parent / "launch.json")
    review, audit, summary, callback, export, meta = (read(STEP / name) for name in
        ("independent_review.json", "audit.json", "full8_summary.json", "step_export.json",
         "final_model/export_record.json", "final_model/checkpoint_meta.json"))
    require(review["status"] == audit["status"] == "verified" and callback["status"] == "evaluated"
            and summary["status"] == "full8_evaluation_verified"
            and {x["checkpoint_id"] for x in (review, audit, summary, callback, export, meta)} == {UUID}
            and review["source_checkpoint_id"] == audit["source_checkpoint_id"] == BASE_UUID
            and review["global_step"] == audit["selected_global_step"] == callback["global_step"] == 1600,
            "Require the independently reviewed decoder-only D1600 endpoint.")
    require(review["callback"]["stage_exits"] == {k: 0 for k in ("export", "audit", "mmlu", "seven")}
            and all((STEP / (k + ".exit")).read_text().strip() == "0"
                    for k in ("export", "audit", "mmlu", "seven"))
            and summary["leaf_count"] == 64 and summary["sample_count"] == 24742
            and summary["mean_percent"] == review["mean_percent"] == callback["mean_percent"]
            and summary["audit_sha256"] == callback["audit_sha256"] == sha256(STEP / "audit.json")
            and callback["full8_summary_sha256"] == sha256(STEP / "full8_summary.json"),
            "Require completed native export/audit and all64 leaves; no quality threshold.")
    require(audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == 2979
            and audit["fresh_strict_reload_state_exact"] and audit["frozen_bytes_unchanged"]
            and audit["packed_codes_unchanged"] and audit["changed_code_bits"] == 0 and audit["rank_zero"]
            and audit["frozen_tensors_compared"] == 1635 and audit["decoder_tensor_count"] == 1344
            and audit["projection_count"] == len(meta["converted_modules"]) == 224
            and all(spec["codebook_dim"] == 32 for spec in meta["converted_modules"])
            and export["optimizer_steps_executed"] == 0,
            "Reuse2979 strict/native and frozen-code/norm/head proof; no D16 replacement.")
    budget = audit["bit_budget"]
    require(budget["user_linear_bits"] == budget["packed_linear_bits"] == 17100439552
            and budget["original_linear_weights"] == 6979321856 and budget["code_padding_bits"] == 0
            and budget["budget_passed"] and not budget["contract_violations"]
            and budget["components"]["decoder"]["dtypes"] == ["bfloat16"], "Storage contract differs.")
    check_hashes(launch["source_sha256"]); check_hashes(launch["input_evidence_sha256"])
    check_hashes(review["evidence_sha256"]); check_hashes(export["code_sha256"])
    dense_config = read(DENSE / "config.json")
    config = make_config(dense_config["archive"]["path"], kind="compressed",
                         hf_path=dense_config["model"]["hf_path"], checkpoint=checkpoint, mode="full")
    require(config["model"]["checkpoint_id"] == UUID and config["model_policy"]["attn_implementation"] == "sdpa"
            and config["model_policy"]["rope_scaling"]["rope_type"] == "llama3", "Wrong native Llama policy.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    controls, evidence, durations = {}, dict(review["evidence_sha256"]), {}
    for label, directory in CONTROLS:
        c_config, scores = read(directory / "config.json"), read(directory / "scores.json")
        for key in ("schema", "mode", "protocol", "archive", "package_versions", "model_policy", "prewarm_group_size", "device"):
            require(config[key] == c_config[key], "Control Llama protocol differs: " + key)
        predictions_root = directory if label == "bf16" else directory / "run"
        control_review_path = directory / ("score_verification.json" if label == "bf16" else "review.json")
        control_review = read(control_review_path)
        require(control_review["status"] == "verified"
                and control_review["scores_sha256"] == sha256(directory / "scores.json"), "Control scores not reviewed.")
        if label == "bf16":
            require(control_review["score_exit_code"] == 0
                    and control_review["all_prediction_raw_source_token_decode_generation_checks_passed"],
                    "Require complete native BF16 control evidence.")
        else:
            require(control_review["execution_exit_code"] == 0 and control_review["samples"] == 400
                    and control_review["actual_input_tokens_and_original_ids_exact"]
                    and control_review["generation_options_effective_and_decoded_tokens_verified"],
                    "Require complete native compressed control evidence.")
            durations[label] = control_review["duration_seconds"]
            path = Path(control_review["execution_report"]); evidence[str(path)] = sha256(path)
        tasks = {}
        for task in TASKS:
            rows = [r for r in index if r["dataset"] == task]
            path = predictions_root / "predictions" / (task + ".jsonl")
            preds = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows) == len(preds) == 200 and [r["row_index"] for r in rows] == list(range(200))
                    and sha256(path) == scores["prediction_sha256"][str(path)], "Use every original200 row.")
            values = []
            for pred, row in zip(preds, rows):
                check_prediction(pred, row, sha256(directory / "config.json"))
                values.append(score_example(task, pred["pred"], row["answers"], row["all_classes"]))
            actual = 100 * sum(values) / 200
            reported = (scores["tasks"][task]["score"] if label == "bf16" else
                        scores["tasks"][task]["models"][label]["score_percent"] if label == "original3577" else
                        scores["tasks"][task]["score_percent"])
            require(abs(actual - reported) < 1e-10, "Official fractional score differs.")
            tasks[task] = {"score": actual, "sample_count": 200}; evidence[str(path)] = sha256(path)
        controls[label] = {"directory": str(directory), "prediction_root": str(predictions_root), "tasks": tasks}
        for path in (control_review_path, directory / "scores.json", directory / "config.json"):
            evidence[str(path)] = sha256(path)
    sources = dict(launch["source_sha256"])
    for pins in (export["code_sha256"], dense_config["source_sha256"], config["source_sha256"],
                 {str(Path(__file__).resolve()): sha256(__file__),
                  str(ROOT / "experiments/longbench_candidate_probe/prepare_llama_best1600_d16.py"):
                  sha256(ROOT / "experiments/longbench_candidate_probe/prepare_llama_best1600_d16.py")}):
        for path, digest in pins.items():
            require(path not in sources or sources[path] == digest, "Source differs: " + path); sources[path] = digest
    for path in (STEP.parent / "launch.json", STEP / "independent_review.json", STEP / "audit.json",
                 STEP / "full8_summary.json", STEP / "step_export.json", checkpoint / "checkpoint_meta.json",
                 checkpoint / "config.json", checkpoint / "export_record.json"):
        evidence[str(path)] = sha256(path)
    check_hashes(sources); check_hashes(evidence)
    config_path = output / "config.json"
    record = {"schema": "llama_fineweb1600_retrieval400_preparation_v1", "status": "prepared_unbound", "gpu": None,
        "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": UUID,
        "config_path": str(config_path), "output_dir": str(output / "run"), "datasets": list(TASKS),
        "sample_count": 400, "task_samples": {task: 200 for task in TASKS}, "source_sha256": sources,
        "evidence_sha256": evidence, "source_bit_budget": budget, "controls": controls,
        "completed_candidate": {"checkpoint_dir": str(checkpoint), "checkpoint_id": UUID,
            "full8_mean_percent": review["mean_percent"], "review_path": str(STEP / "independent_review.json"),
            "native_tensor_count": 2979, "frozen_tensors": 1635, "code_bits_changed": 0, "global_step": 1600},
        "environment": {k: v for k, v in launch["environment"].items() if k != "CUDA_VISIBLE_DEVICES"},
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict", "--config", str(config_path),
                          "--output", str(output / "run"), "--datasets", *TASKS],
        "timeout_seconds": 900, "termination_grace_seconds": 20, "queued": False, "automatic_retry": False, "resume": False,
        "resource_evidence": {"actual_control_seconds": durations,
            "boundary": "900s planning cap, with completed same-native-protocol controls. Require parent callback window >=1220s."},
        "selection": "All archived rows0..199 for both official tasks; no filtering or sampling.",
        "scoring": "Native check_prediction/score_example with fractional credit; two-task macro only.",
        "validation_reuse": "Completed D1600 native2979 save/fresh/full8 proof and both compressed SDPA/FP32RoPE/group8 retrieval400 controls.",
        "comparison_boundary": "3577 to D1600 changes only decoder FineWeb recovery; D1600 to e767 replaces32 down projections with fresh D16 and removes their recovery history.",
        "scope": "Two of21 LongBench tasks, not full LongBench or nine-item acceptance.",
        "full_longbench_status": "not_evaluated", "nine_item_goal_status": "not_evaluated",
        "model_mutations": False, "model_exported": False, "cuda_initialized": torch.cuda.is_initialized()}
    require(not record["cuda_initialized"], "CPU preparation initialized CUDA.")
    output.mkdir(parents=True); write_json(config_path, config); record["config_sha256"] = sha256(config_path)
    write_json(output / "preparation.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.output_dir)
    print({"status": record["status"], "preparation_sha256": sha256(args.output_dir / "preparation.json"),
           "config_sha256": record["config_sha256"], "sources": len(record["source_sha256"]),
           "evidence": len(record["evidence_sha256"]), "cuda_initialized": record["cuda_initialized"]}, flush=True)
