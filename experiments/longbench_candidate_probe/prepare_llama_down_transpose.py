"""Bind the matched Llama D16 down orientations to official retrieval400."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.scoring import check_prediction, score_example

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "result/qwen3_8b_mmlu65_20261004"
ARMS = {
    "transpose": (R2 / "llama_down_dimension/block32", "8a4633a7-93bd-429a-8a0e-a519a2ae30ae"),
    "nontranspose": (R2 / "llama_down_transpose/nontranspose", "6d2f1b08-7ed6-4556-a707-71eb47e788d9"),
}
PAIR_REVIEW = ARMS["nontranspose"][0] / "independent_review.json"
PAIR_REVIEW_SHA = "f60ee9d24d85c4e3685762774267eed976b62f505fe6c1842ab170d56091573a"
BASE_ID = "35776715-2707-45a9-9fb4-2da1efef3f0e"
DENSE = ROOT / "result/llama31_8b_longbench_20261004/dense"
ORIGINAL = R2 / "llama_original3577_retrieval400"
TASKS = ("passage_retrieval_en", "passage_retrieval_zh")
PROTOCOL_KEYS = ("schema", "mode", "protocol", "archive", "package_versions", "model_policy", "prewarm_group_size", "device")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge_pins(target, incoming):
    for path, digest in incoming.items():
        require(path not in target or target[path] == digest, "Conflicting source/evidence pin: " + path)
        target[path] = digest


def checked_pair():
    review = read(PAIR_REVIEW)
    require(sha256(PAIR_REVIEW) == PAIR_REVIEW_SHA and review["status"] == "independent_cpu_verified"
            and review["same_native_recipe_except_output_and_down_transpose"]
            and review["native_audit_reused"] == {
                "source_union_save_fresh_tensors": 2979, "frozen_other_projection_count": 192,
                "down_count": 32, "same_protection_and_rotation_sign_records": True,
                "user_linear_bits": 17095180288, "user_linear_bpw": 17095180288 / 6979321856},
            "Require the independently reviewed pure transpose pair, not the recovered e767 combination.")
    sources, evidence, arms = {}, dict(review["evidence_sha256"]), {}
    check_hashes(evidence)
    for arm, (directory, uuid) in ARMS.items():
        launch, audit, summary, meta, assembly = (read(directory / name) for name in
            ("launch.json", "audit.json", "full8_summary.json", "final_model/checkpoint_meta.json", "final_model/assembly_report.json"))
        verified = review["control" if arm == "transpose" else "candidate"]
        require(launch["status"] == "completed" and launch["exit_code"] == 0
                and audit["status"] == assembly["status"] == "verified"
                and summary["status"] == "complete_full8"
                and all(item["checkpoint_id"] == uuid for item in (launch, audit, summary, meta, assembly, verified))
                and audit["base_checkpoint_id"] == launch["base_checkpoint_id"] == BASE_ID,
                "The completed native source identity differs: " + arm)
        require(all((directory / (stage + ".exit")).read_text().strip() == "0"
                    for stage in ("train", "assembly", "mmlu", "seven", "dispatcher", "outer"))
                and all(item["exit_code"] == 0 for item in launch["stages"].values())
                and summary["leaf_count"] == 64 and summary["sample_count"] == 24742
                and all(summary[key] == verified[key] for key in ("task_metrics", "integer_counts", "mean_percent"))
                and sha256(directory / "full8_summary.json") == launch["full8_summary_sha256"],
                "Require complete full8 without a quality threshold: " + arm)
        require(audit["source_union_save_fresh_tensors"] == assembly["serialized_tensors_compared"]
                == assembly["reloaded_tensors_compared"] == 2979
                and audit["replaced_projection_count"] == 32 and audit["preserved_projection_count"] == 192
                and assembly["sources"]["base_checkpoint_id"] == BASE_ID
                and assembly["sources"]["categories"] == ["down_proj"]
                and audit["assembly_report_sha256"] == sha256(directory / "final_model/assembly_report.json"),
                "Native source-union/save/fresh evidence changed: " + arm)
        down = [s for s in meta["converted_modules"] if s["name"].endswith(".down_proj")]
        require(len(meta["converted_modules"]) == 224 and len(down) == 32 and meta["lora_config"] is None
                and all(s["codebook_dim"] == 16 and s["transpose"] is (arm == "transpose")
                        and s["residual_stages"] == 2 and s["weight_rotation"]["block_size"] == 32
                        and s["protected_input_indices"]["shape"] == [256] for s in down),
                "Actual D16/block32/down256 orientation differs: " + arm)
        budget = audit["bit_budget"]
        require(budget == assembly["bit_budget"] and budget["user_linear_bits"] == budget["packed_linear_bits"] == 17095180288
                and budget["original_linear_weights"] == 6979321856 and budget["code_padding_bits"] == 0
                and budget["budget_passed"] and not budget["contract_violations"] and not budget["unclassified_target_keys"]
                and budget["components"]["decoder"]["dtypes"] == ["bfloat16"]
                and all(budget["components"].get(k, {}).get("storage_bits", 0) == 0 for k in ("lora", "residual_lora", "dense_weights")),
                "Measured rank-zero storage differs: " + arm)
        checkpoint = directory / "final_model"
        require((checkpoint / meta["state_dict_file"]).is_file(), "Actual candidate weight is missing: " + arm)
        for stat in launch["protected_stats"]:
            check_stat(stat)
        merge_pins(sources, launch["source_sha256"]); merge_pins(evidence, launch["input_sha256"])
        for name in ("launch.json", "audit.json", "full8_summary.json", "final_model/checkpoint_meta.json",
                     "final_model/config.json", "final_model/assembly_report.json"):
            path = directory / name; evidence[str(path)] = sha256(path)
        arms[arm] = {"directory": directory, "checkpoint": checkpoint, "launch": launch, "audit": audit, "summary": summary}
    for key in ("protection_tensor_records", "rotation_sign_records"):
        require(arms["transpose"]["audit"][key] == arms["nontranspose"]["audit"][key], "Pair payload differs: " + key)
    require(arms["transpose"]["audit"]["bit_budget"]["components"] == arms["nontranspose"]["audit"]["bit_budget"]["components"],
            "Pair component storage differs.")
    evidence[str(PAIR_REVIEW)] = PAIR_REVIEW_SHA
    return arms, sources, evidence


def auxiliary_controls(config, index, evidence):
    controls, durations = {}, {}
    for label, directory in (("bf16", DENSE), ("original3577", ORIGINAL)):
        old, scores = read(directory / "config.json"), read(directory / "scores.json")
        require(all(config[key] == old[key] for key in PROTOCOL_KEYS), "Auxiliary protocol differs: " + label)
        config_sha = sha256(directory / "config.json")
        review_path = directory / ("score_verification.json" if label == "bf16" else "review.json")
        review = read(review_path)
        require(review["status"] == "verified" and review["scores_sha256"] == sha256(directory / "scores.json")
                and scores["config_sha256"] == config_sha, "Auxiliary results changed: " + label)
        if label == "bf16":
            require(old["model"]["kind"] == "dense" and review["score_exit_code"] == 0
                    and review["all_prediction_raw_source_token_decode_generation_checks_passed"], "Require reviewed BF16 predictions.")
            prediction_root = directory
        else:
            require(old["model"]["checkpoint_id"] == review["checkpoint_id"] == BASE_ID
                    and review["execution_exit_code"] == 0 and review["samples"] == 400
                    and review["actual_input_tokens_and_original_ids_exact"]
                    and review["generation_options_effective_and_decoded_tokens_verified"], "Require reviewed original3577 predictions.")
            prediction_root = directory / "run"; durations[label] = review["duration_seconds"]
            path = Path(review["execution_report"]); evidence[str(path)] = sha256(path)
        tasks = {}
        for task in TASKS:
            rows = [row for row in index if row["dataset"] == task]
            path = prediction_root / "predictions" / (task + ".jsonl")
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows) == len(predictions) == 200 and [r["row_index"] for r in rows] == list(range(200))
                    and sha256(path) == scores["prediction_sha256"][str(path)], "Use all original200 rows: " + task)
            values = []
            for prediction, row in zip(predictions, rows, strict=True):
                check_prediction(prediction, row, config_sha)
                values.append(score_example(task, prediction["pred"], row["answers"], row["all_classes"]))
            actual = 100 * math.fsum(values) / 200
            reported = scores["tasks"][task]["score"] if label == "bf16" else scores["tasks"][task]["models"][label]["score_percent"]
            require(abs(actual - reported) < 1e-10, "Official fractional score differs.")
            tasks[task] = {"score": actual, "sample_count": 200}; evidence[str(path)] = sha256(path)
        controls[label] = {"directory": str(directory), "prediction_root": str(prediction_root), "tasks": tasks}
        for path in (review_path, directory / "scores.json", directory / "config.json"):
            evidence[str(path)] = sha256(path)
    return controls, durations


def prepare(arm, output):
    require(Path.cwd() == ROOT and os.environ.get("PYTHONPATH") == str(ROOT)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "Require public cwd/PYTHONPATH and CUDA-hidden bitvae.")
    output = Path(output).resolve()
    require(arm in ARMS and not output.exists(), "Use a known arm and fresh output; no resume or overwrite.")
    arms, sources, evidence = checked_pair(); item = arms[arm]
    dense = read(DENSE / "config.json")
    config = make_config(dense["archive"]["path"], kind="compressed", hf_path=dense["model"]["hf_path"],
                         checkpoint=item["checkpoint"], mode="full")
    require(config["model"]["checkpoint_id"] == ARMS[arm][1] and all(config[k] == dense[k] for k in PROTOCOL_KEYS)
            and config["model_policy"]["model_type"] == "llama" and config["model_policy"]["attn_implementation"] == "sdpa"
            and config["model_policy"]["rope_scaling"]["rope_type"] == "llama3"
            and config["protocol"]["total_window_tokens"] == 32768
            and all(config["protocol"]["max_new_tokens"][task] == 32 for task in TASKS), "Keep the official native Llama policy.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    controls, durations = auxiliary_controls(config, index, evidence)
    merge_pins(sources, config["source_sha256"]); sources[str(Path(__file__).resolve())] = sha256(__file__)
    check_hashes(sources); check_hashes(evidence)
    config_path = output / "config.json"
    environment = {**item["launch"]["environment"], "PYTHONPATH": str(ROOT)}
    require("CUDA_VISIBLE_DEVICES" not in environment, "GPU binding belongs to the later explicit launch.")
    record = {"schema": "llama_down_transpose_retrieval400_preparation_v1", "status": "prepared_unbound", "gpu": None,
        "arm": arm, "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": ARMS[arm][1],
        "config_path": str(config_path), "output_dir": str(output / "run"), "datasets": list(TASKS),
        "sample_count": 400, "task_samples": {task: 200 for task in TASKS}, "source_sha256": sources,
        "evidence_sha256": evidence, "source_bit_budget": item["audit"]["bit_budget"], "controls": controls,
        "completed_candidate": {"checkpoint_dir": str(item["checkpoint"]), "checkpoint_id": ARMS[arm][1],
            "base_checkpoint_id": BASE_ID, "full8_mean_percent": item["summary"]["mean_percent"],
            "native_tensor_count": 2979, "preserved_projections": 192, "replaced_projections": 32},
        "matched_pair": {"checkpoint_ids": {key: value[1] for key, value in ARMS.items()},
            "full8_review_path": str(PAIR_REVIEW), "full8_review_sha256": PAIR_REVIEW_SHA},
        "primary_comparison": "nontranspose_minus_transpose", "environment": environment,
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict", "--config", str(config_path),
                          "--output", str(output / "run"), "--datasets", *TASKS],
        "timeout_seconds": 900, "termination_grace_seconds": 20, "queued": False, "automatic_retry": False, "resume": False,
        "resource_evidence": {"actual_auxiliary_control_seconds": durations,
            "boundary": "900s per-arm planning cap plus20s termination grace; bind only an available sufficient GPU window."},
        "selection": "All archived rows0..199 for each official task, in order; no filtering or resampling.",
        "scoring": "Official check_prediction/score_example with fractional credit retained; two-task macro only.",
        "comparison_boundary": "Score both8a4633 and6d2f under identical official400 inputs before the transpose comparison. BF16/3577 are secondary references; e767 is not a pure transpose control.",
        "validation_reuse": "Completed paired2979 native save/fresh/full8, fixed192 projections and equal protection/signs; existing native Llama SDPA/FP32RoPE/group8 retrieval400.",
        "scope": "Two of21 official LongBench tasks only; no full LongBench or nine-item acceptance.",
        "full_longbench_status": "not_evaluated", "nine_item_goal_status": "not_evaluated",
        "model_mutations": False, "model_exported": False, "cuda_initialized": torch.cuda.is_initialized()}
    require(not record["cuda_initialized"], "CPU preparation initialized CUDA.")
    output.mkdir(parents=True); write_json(config_path, config); record["config_sha256"] = sha256(config_path)
    write_json(output / "preparation.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.arm, args.output_dir)
    print({"status": record["status"], "arm": args.arm, "config_sha256": record["config_sha256"],
           "preparation_sha256": sha256(args.output_dir / "preparation.json"), "cuda_initialized": False}, flush=True)
