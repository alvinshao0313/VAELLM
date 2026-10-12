"""Bind audited full-KL800 to the completed mass-KL800 retrieval400 protocol."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_candidate_probe.prepare_general_lm_qwen800 import read, require, TASKS
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.scoring import check_prediction, score_example

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "result/qwen3_8b_mmlu65_20261004"
TRAIN = R2 / "general_lm_full_kl_s2000"
STEP = TRAIN / "step_0800"
CONTROL = R2 / "general_lm_qwen800_retrieval400"
CHECKPOINT_ID = "d3638240-3c59-46ab-b6cc-e903d4d4a4cd"
CONTROL_ID = "8b209479-c6b7-4aa9-9c18-9156a2e9ffa4"
SOURCE_ID = "f927ffda-dd5b-4d00-b63a-5e839a182d70"
PROTOCOL_KEYS = ("schema", "mode", "protocol", "archive", "package_versions",
                 "model_policy", "prewarm_group_size", "device")


def check_endpoint():
    training = read(TRAIN / "launch.json")
    step, audit, review, exported, full8, meta = (read(STEP / name) for name in (
        "step_export.json", "audit.json", "independent_review.json", "final_model/export_record.json",
        "full8_summary.json", "final_model/checkpoint_meta.json"))
    require(step["status"] == "evaluated" and step["global_step"] == 800 and step["max_steps"] == 2000
            and step["diagnostic_only"] is False
            and set(step["stages"]) == {"export", "audit", "mmlu", "seven"}
            and all(item["exit_code"] == 0 for item in step["stages"].values())
            and all((STEP / (stage + ".exit")).read_text().strip() == "0" for stage in step["stages"]),
            "Require the actual completed full-KL800 callback.")
    require(audit["status"] == review["status"] == "verified" and exported["status"] == "exported"
            and {item["checkpoint_id"] for item in (step, audit, review, exported, full8, meta)} == {CHECKPOINT_ID}
            and audit["source_checkpoint_id"] == review["source_checkpoint_id"] == training["source_checkpoint_id"]
            == step["round_base_checkpoint_id"] == exported["round_base_checkpoint_id"] == SOURCE_ID
            and audit["selected_checkpoint_id"] == step["selected_checkpoint_id"] == exported["source_checkpoint_id"]
            and audit["selected_global_step"] == exported["source_global_step"] == review["global_step"] == 800,
            "Export, callback and independent review name different endpoints.")
    actual = review["results"]["full800"]
    require(actual["checkpoint_id"] == CHECKPOINT_ID and actual["mmlu_correct"] == 8599
            and actual["task_metrics"] == full8["task_metrics"] and actual["mean_percent"] == full8["mean_percent"]
            and full8["status"] == "full8_evaluation_verified" and full8["leaf_count"] == 64
            and full8["sample_count"] == 24742
            and step["audit_sha256"] == full8["audit_sha256"] == sha256(STEP / "audit.json")
            and step["full8_summary_sha256"] == sha256(STEP / "full8_summary.json")
            and review["configuration"]["actual_equals_prepared"]
            and review["configuration"]["mass_canonical_differences_only"] == ["run_root_dir", "loss.loss_type"]
            and training["cpu_configuration"]["canonical_config"]["loss"]["loss_type"] == "kl",
            "Require the reviewed full-vocabulary KL endpoint and its complete raw evaluation.")
    require(audit["train_mode"] == "decoder" and audit["selected_decoder_tensor_count"] == 1512
            and audit["selected_module_count"] == audit["changed_decoder_module_count"] == 252
            and audit["selected_decoder_fp32_changed_elements"] > 0
            and audit["selected_decoder_matches_fp32_cast_exactly"]
            and audit["frozen_tensors_compared"] == 1911 and audit["frozen_bytes_unchanged"]
            and audit["code_bank_count"] == 504 and audit["changed_code_bits"] == 0
            and audit["selected_codes_match_frozen_source_exactly"] and audit["packed_padding_unchanged"]
            and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == 3423
            and audit["fresh_strict_reload_state_exact"] and audit["rank_zero"]
            and audit["inference_specs_unchanged_except_activation_checkpointing"]
            and audit["native_decoder_dtype"] == "torch.bfloat16"
            and meta["norm_train_mode"] == meta["lm_head_train_mode"] == "none"
            and audit["selected_training_state_sha256"] == exported["source_files_sha256"][
                str(Path(exported["source_checkpoint"]) / "training_model_state.pt")],
            "Require the completed native decoder/frozen-code/save/fresh deployment proof.")
    budget = audit["bit_budget"]
    require(budget["user_linear_bits"] == budget["packed_linear_bits"] == 17065672704
            and budget["original_linear_weights"] == 6945767424 and budget["code_padding_bits"] == 0
            and budget["budget_passed"] and not budget["contract_violations"]
            and not budget["unclassified_target_keys"]
            and all(budget[key] == training["source_bit_budget"][key]
                    for key in ("user_linear_bits", "original_linear_weights", "components")),
            "Actual integer storage differs from the frozen source.")
    check_hashes(training["source_sha256"]); check_hashes(training["input_evidence_sha256"])
    check_hashes(review["evidence_sha256"]); check_hashes(training["hf_source"]["metadata_sha256"])
    for stat in (training["source_weight_stat"], *training["input_data_stat"], *training["hf_source"]["weight_files"]):
        check_stat(stat)
    return training, audit, review, actual


def prepare(output_dir):
    require(Path.cwd() == ROOT and os.environ.get("PYTHONPATH") == str(ROOT)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "Use public cwd/PYTHONPATH and CUDA-hidden bitvae.")
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "Use a fresh independent output; no resume or overwrite.")
    training, audit, endpoint_review, actual = check_endpoint()
    previous, control_review, old = (read(CONTROL / name) for name in
                                     ("preparation.json", "review.json", "config.json"))
    old_sha = sha256(CONTROL / "config.json")
    execution_path = Path(control_review["execution_report"]); execution = read(execution_path)
    require(previous["checkpoint_id"] == control_review["checkpoint_id"] == old["model"]["checkpoint_id"] == CONTROL_ID
            and previous["config_sha256"] == execution["config_sha256"] == old_sha
            and (CONTROL / "run/config.json").read_bytes() == (CONTROL / "config.json").read_bytes()
            and control_review["status"] == "verified" and control_review["samples"] == 400
            and execution["status"] == control_review["execution_status"] == "partition_complete"
            and execution["exit_code"] == control_review["execution_exit_code"] == 0
            and execution["new_predictions"] == 400 and execution["already_present"] == 0
            and execution["datasets"] == list(TASKS) and execution["quality_evaluation"]
            and control_review["report_sha256"] == sha256(execution_path)
            and (CONTROL / "outer.exit").read_text().strip() == "0"
            and control_review["actual_input_tokens_and_original_ids_exact"]
            and control_review["generation_options_effective_and_decoded_tokens_verified"]
            and control_review["historical_three_controls_official_scores_recomputed"]
            and control_review["all_pins_unchanged_after"], "Require completed audited mass-KL800 raw400.")
    check_hashes(previous["source_sha256"]); check_hashes(previous["evidence_sha256"])
    check_hashes(control_review["evidence_sha256"])
    # Old control weights were released after scoring; only the new endpoint is loaded by predict.
    config = make_config(old["archive"]["path"], kind="compressed", hf_path=old["model"]["hf_path"],
                         checkpoint=STEP / "final_model", mode="full")
    require(config["model"]["checkpoint_id"] == CHECKPOINT_ID
            and all(config[key] == old[key] for key in PROTOCOL_KEYS)
            and all(config["model"][key] == old["model"][key] for key in
                    ("kind", "hf_path", "hf_metadata_sha256", "hf_weight_stats"))
            and config["model_policy"]["model_type"] == "qwen3"
            and config["model_policy"]["attn_implementation"] == "flash_attention_2"
            and config["protocol"]["total_window_tokens"] == 32768
            and all(config["protocol"]["max_new_tokens"][task] == 32 for task in TASKS),
            "Only the compressed endpoint identity may differ from mass-KL800.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    evidence = dict(training["input_evidence_sha256"])
    for pins in (endpoint_review["evidence_sha256"], previous["evidence_sha256"], control_review["evidence_sha256"]):
        for path, digest in pins.items():
            require(path not in evidence or evidence[path] == digest, "Conflicting evidence pin: " + path)
            evidence[path] = digest
    tasks, cases = {}, []
    for task in TASKS:
        rows = [row for row in index if row["dataset"] == task]
        path = CONTROL / "run/predictions" / (task + ".jsonl")
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows) == len(predictions) == 200 and [row["row_index"] for row in rows] == list(range(200)),
                "Use all original EN200 and ZH200 rows in archived order.")
        values = []
        for prediction, row in zip(predictions, rows, strict=True):
            check_prediction(prediction, row, old_sha)
            require(prediction["input_tokens"] == row["input_tokens"]
                    and prediction["removed_tokens"] == row["removed_tokens"]
                    and prediction["max_new_tokens"] == row["max_new_tokens"] == 32,
                    "Mass-KL800 input geometry differs from the shared original archive.")
            values.append(score_example(task, prediction["pred"], row["answers"], row["all_classes"]))
            cases.append({key: row[key] for key in ("dataset", "row_index", "_id", "input_ids_sha256", "input_tokens")})
        score = 100 * math.fsum(values) / 200
        require(abs(score - control_review["task_results"][task]["candidate_score_percent"]) < 1e-10,
                "Official fractional mass-KL800 score changed.")
        tasks[task] = {"samples": 200, "official_score_sum": math.fsum(values),
                       "score_percent": score, "prediction_path": str(path)}
        evidence[str(path)] = sha256(path)
    require(math.fsum(item["score_percent"] for item in tasks.values()) / 2
            == control_review["selected_two_task_macro_percent"] == 52.0, "Control macro differs.")
    sources = dict(training["source_sha256"])
    for pins in (previous["source_sha256"], config["source_sha256"]):
        for path, digest in pins.items():
            require(path not in sources or sources[path] == digest, "Conflicting source pin: " + path)
            sources[path] = digest
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    for path in (TRAIN / "launch.json", STEP / "independent_review.json", STEP / "final_model/config.json",
                 CONTROL / "preparation.json", CONTROL / "config.json", CONTROL / "review.json"):
        evidence[str(path)] = sha256(path)
    check_hashes(sources); check_hashes(evidence)
    controls = {"mass800": {"directory": str(CONTROL), "checkpoint_id": CONTROL_ID, "tasks": tasks,
                            "review_path": str(CONTROL / "review.json"), "model_reload_required": False},
                **previous["controls"]}
    config_path = output_dir / "config.json"
    record = {"schema": "general_lm_full_kl_qwen800_retrieval400_preparation_v1", "status": "prepared_unbound",
        "gpu": None, "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": CHECKPOINT_ID,
        "config_path": str(config_path), "output_dir": str(output_dir / "run"), "datasets": list(TASKS),
        "sample_count": 400, "task_samples": {task: 200 for task in TASKS}, "cases": cases,
        "selection": "All original archived rows0..199 per task in native order; identical IDs/token hashes to mass800.",
        "source_sha256": sources, "evidence_sha256": evidence, "source_bit_budget": audit["bit_budget"],
        "completed_candidate": {"directory": str(STEP), "checkpoint_dir": str(STEP / "final_model"),
            "checkpoint_id": CHECKPOINT_ID, "source_checkpoint_id": SOURCE_ID, "global_step": 800,
            "mean_percent": actual["mean_percent"], "mmlu_correct": actual["mmlu_correct"],
            "audit_path": str(STEP / "audit.json"), "review_path": str(STEP / "independent_review.json"),
            "native_training_state_sha256": audit["selected_training_state_sha256"], "native_checkpoint_reload_required": False},
        "controls": controls, "primary_comparison": "full_kl800_minus_mass_kl800",
        "comparison_boundary": "Matched fresh f927/FineWeb800 decoder recovery; reviewed training difference is loss_type. Historical norm400 is only a secondary reference.",
        "validation_reuse": "Audited3423 native save/fresh,1512 BF16 decoders/1911 frozen/504 codebanks XOR0; same native QwenFA2/group8 official400 protocol. No new GPU short test or export.",
        "purpose": "Measure whether full-vocabulary KL restores retrieval and inform endpoint retention; no quality threshold for accepting a valid result.",
        "timeout_seconds": 900, "termination_grace_seconds": 20, "environment": previous["environment"],
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict", "--config", str(config_path),
                          "--output", str(output_dir / "run"), "--datasets", *TASKS],
        "resource_evidence": {"mass800_duration_seconds": control_review["duration_seconds"],
            "mass800_load_peak_allocated_bytes": execution["load_audit"]["load_peak_allocated_bytes"],
            "boundary": "Same-geometry reference, not a runtime guarantee; bind one available A80080GB within900+20s."},
        "queued": False, "automatic_retry": False, "resume": False,
        "scoring": previous["scoring"], "scope": "Two complete official retrieval tasks only,400 predictions/2of21tasks.",
        "full_longbench_status": "not_evaluated", "nine_item_goal_status": "not_evaluated",
        "model_mutations": False, "model_exported": False, "cuda_initialized": torch.cuda.is_initialized()}
    require(not record["cuda_initialized"], "CPU preparation initialized CUDA.")
    output_dir.mkdir(parents=True); write_json(config_path, config); record["config_sha256"] = sha256(config_path)
    write_json(output_dir / "preparation.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output_dir)
    print(json.dumps({"status": result["status"], "checkpoint_id": result["checkpoint_id"],
        "source_count": len(result["source_sha256"]), "evidence_count": len(result["evidence_sha256"]),
        "sample_count": result["sample_count"], "config_sha256": result["config_sha256"],
        "preparation_sha256": sha256(args.output_dir / "preparation.json"),
        "cuda_initialized": result["cuda_initialized"]}, ensure_ascii=False))
