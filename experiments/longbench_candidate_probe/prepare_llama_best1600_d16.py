"""Bind the audited Llama FineWeb1600+D16-down endpoint to official retrieval400."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.scoring import check_prediction, score_example

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "result/qwen3_8b_mmlu65_20261004"
CANDIDATE = R2 / "llama_fineweb1600_down_d16_combo"
DENSE = ROOT / "result/llama31_8b_longbench_20261004/dense"
CHECKPOINT_ID = "e767c232-56fd-4b03-a4ae-dd0781b6e047"
BASE_ID = "dbde405a-328e-4bd2-84d3-f18ad60e3049"
DONOR_ID = "8a4633a7-93bd-429a-8a0e-a519a2ae30ae"
REVIEW_SHA = "408d032b80ffd78d3b328897501157f1b7f46553e9673a519b136f1e585b3ac9"
TASKS = ("passage_retrieval_en", "passage_retrieval_zh")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(output_dir):
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not torch.cuda.is_initialized(), "Use public source root and CUDA-hidden bitvae.")
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "Use a fresh output directory; do not resume or overwrite predictions.")
    checkpoint = CANDIDATE / "final_model"
    launch, review, summary, meta, audit = (read(CANDIDATE / name) for name in
        ("launch.json", "review.json", "full8_summary.json", "final_model/checkpoint_meta.json",
         "final_model/assembly_report.json"))
    require(sha256(CANDIDATE / "review.json") == REVIEW_SHA and review["status"] == audit["status"] == "verified"
            and launch["status"] == "completed" and launch["exit_code"] == 0
            and {item["checkpoint_id"] for item in (launch, review, summary, meta, audit)} == {CHECKPOINT_ID},
            "Require the actual completed e767 endpoint and independent review.")
    require(set(launch["stages"]) == {"assembly", "mmlu", "seven"}
            and all(item["exit_code"] == 0 for item in launch["stages"].values())
            and all((CANDIDATE / (stage + ".exit")).read_text().strip() == "0"
                    for stage in ("assembly", "mmlu", "seven", "dispatcher", "outer")),
            "All assembly and full8 stages must have exited successfully.")
    require(summary["status"] == "full8_evaluation_verified" and summary["leaf_count"] == 64
            and summary["sample_count"] == 24742 and summary["task_metrics"] == review["task_metrics"]
            and all(summary["integer_counts"][key] == value for key, value in review["integer_counts"].items())
            and review["integer_counts"]["mmlu"]["correct"] == 7184
            and summary["mean_percent"] == review["mean_percent"] == launch["mean_percent"]
            and summary["assembly_report_sha256"] == sha256(checkpoint / "assembly_report.json")
            and launch["full8_summary_sha256"] == sha256(CANDIDATE / "full8_summary.json"),
            "Bind complete full8 evidence without imposing a retrieval score threshold.")
    union = audit["sources"]
    proof = review["native_proof"]
    require(union["base_checkpoint_id"] == launch["base_checkpoint_id"] == BASE_ID
            and union["donor_checkpoint_id"] == launch["donor_checkpoint_id"] == DONOR_ID
            and union["categories"] == ["down_proj"] and len(union["replaced_targets"]) == 32
            and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == 2979
            and proof["metadata_specs_exact_source_union"] and proof["preserved_base_projection_count"] == 192,
            "Reuse actual native union/save/fresh proof: 32 down projections, 192 base projections and backbone.")
    down = [spec for spec in meta["converted_modules"] if spec["name"].endswith(".down_proj")]
    require(len(down) == 32 and all(spec["codebook_dim"] == 16 and spec["transpose"]
            and spec["residual_stages"] == 2 and spec["weight_rotation"]["block_size"] == 32
            and spec["protected_input_indices"]["shape"] == [256] for spec in down),
            "Require the audited native D16/block32/down256 geometry.")
    budget = audit["bit_budget"]
    require(budget["user_linear_bits"] == budget["packed_linear_bits"] == 17095180288
            and budget["original_linear_weights"] == 6979321856 and budget["code_padding_bits"] == 0
            and budget["budget_passed"] and not budget["contract_violations"]
            and budget["user_linear_bpw"] < 2.5 and budget["components"]["decoder"]["dtypes"] == ["bfloat16"]
            and all(budget[key] == summary["bit_budget"][key] == review["bit_budget"][key]
                    for key in ("user_linear_bits", "original_linear_weights", "components")),
            "Keep measured integer bits including decoder/protection/indices/rotation; no new model export.")
    check_hashes(launch["source_sha256"]); check_hashes(launch["input_sha256"])
    check_hashes(review["evidence_sha256"])

    dense_config, dense_scores, dense_review = (read(DENSE / name) for name in
        ("config.json", "scores.json", "score_verification.json"))
    require(dense_review["status"] == "verified" and dense_review["score_exit_code"] == 0
            and dense_review["all_prediction_raw_source_token_decode_generation_checks_passed"]
            and dense_review["scores_sha256"] == sha256(DENSE / "scores.json")
            and dense_review["config_sha256"] == dense_scores["config_sha256"] == sha256(DENSE / "config.json"),
            "Require the independently verified full Llama BF16 archive and predictions.")
    config = make_config(dense_config["archive"]["path"], kind="compressed",
                         hf_path=dense_config["model"]["hf_path"], checkpoint=checkpoint, mode="full")
    require(config["model"]["checkpoint_id"] == CHECKPOINT_ID
            and all(config[key] == dense_config[key] for key in ("schema", "mode", "protocol", "archive",
                    "package_versions", "model_policy", "prewarm_group_size", "device"))
            and all(config["model"][key] == dense_config["model"][key]
                    for key in ("hf_path", "hf_metadata_sha256", "hf_weight_stats"))
            and config["model_policy"]["model_type"] == "llama"
            and config["model_policy"]["attn_implementation"] == "sdpa"
            and config["model_policy"]["rope_scaling"]["rope_type"] == "llama3",
            "Use the actual Llama tokenizer/chat/EOS/SDPA/RoPE protocol, never the Qwen archive.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    rows = {task: [row for row in index if row["dataset"] == task] for task in TASKS}
    timing, control_tasks = {}, {}
    for task, items in rows.items():
        path = DENSE / "predictions" / (task + ".jsonl")
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(items) == len(predictions) == 200 and [r["row_index"] for r in items] == list(range(200))
                and sha256(path) == dense_scores["prediction_sha256"][str(path)], "Use each complete original200 task.")
        scores = []
        for pred, row in zip(predictions, items):
            check_prediction(pred, row, dense_scores["config_sha256"])
            scores.append(score_example(task, pred["pred"], row["answers"], row["all_classes"]))
        require(abs(100 * sum(scores) / 200 - dense_scores["tasks"][task]["score"]) < 1e-10,
                "Historical official task scores must reproduce, including fractional credit.")
        control_tasks[task] = dense_scores["tasks"][task]
        timing[task] = {"generation_seconds": sum(p["elapsed_seconds"] for p in predictions),
            "generated_tokens": sum(p["generated_tokens"] for p in predictions),
            "hit_generation_limit": sum(p["hit_generation_limit"] for p in predictions),
            "input_tokens_min": min(r["input_tokens"] for r in items),
            "input_tokens_max": max(r["input_tokens"] for r in items)}
    run_proofs = []
    for item in dense_review["partitions"]:
        run = read(item["native_report"])
        if not set(TASKS).intersection(run["datasets"]):
            continue
        require(sha256(item["native_report"]) == item["native_report_sha256"]
                and run["status"] == "partition_complete" and run["exit_code"] == 0
                and run["config_sha256"] == dense_scores["config_sha256"]
                and run["load_audit"]["model_policy"] == config["model_policy"]
                and run["load_audit"]["parameter_dtypes"] == ["torch.bfloat16"]
                and run["load_audit"]["rope_buffers"]
                and all(v["dtype"] == "torch.float32" for v in run["load_audit"]["rope_buffers"].values()),
                "Require actual native BF16/SDPA/FP32-RoPE runtime evidence.")
        run_proofs.append(item["native_report"])
    require(run_proofs, "Missing actual retrieval runtime evidence.")
    sources = dict(launch["source_sha256"])
    for pins in (dense_config["source_sha256"], config["source_sha256"], {str(Path(__file__).resolve()): sha256(__file__)}):
        for path, digest in pins.items():
            require(path not in sources or sources[path] == digest, "Source identity differs: " + path)
            sources[path] = digest
    evidence = {**launch["input_sha256"], **review["evidence_sha256"]}
    for path in (CANDIDATE / "review.json", DENSE / "config.json", DENSE / "scores.json",
                 DENSE / "score_verification.json", *(Path(p) for p in run_proofs),
                 *(DENSE / "predictions" / (task + ".jsonl") for task in TASKS)):
        evidence[str(path)] = sha256(path)
    check_hashes(sources); check_hashes(evidence)
    config_path = output_dir / "config.json"
    record = {"schema": "llama_best1600_d16_retrieval400_preparation_v1", "status": "prepared_unbound", "gpu": None,
        "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": CHECKPOINT_ID,
        "config_path": str(config_path), "output_dir": str(output_dir / "run"), "datasets": list(TASKS),
        "sample_count": 400, "task_samples": {task: 200 for task in TASKS}, "source_sha256": sources,
        "evidence_sha256": evidence, "source_bit_budget": budget, "completed_candidate": {
            "directory": str(CANDIDATE), "checkpoint_dir": str(checkpoint), "checkpoint_id": CHECKPOINT_ID,
            "base_checkpoint_id": BASE_ID, "donor_checkpoint_id": DONOR_ID, "mmlu_correct": 7184,
            "full8_mean_percent": summary["mean_percent"], "native_proof": proof, "review_path": str(CANDIDATE / "review.json")},
        "controls": {"bf16": {"directory": str(DENSE), "tasks": control_tasks}},
        "original3577_control": "No existing complete200+200 predictions found; no baseline regeneration or fixed16 substitution.",
        "duplicate_check": "No prior e767 complete retrieval400 config/predictions found during source preparation.",
        "selection": "All archived rows0..199 of each official task; no filtering or resampling.",
        "environment": {**launch["environment"], "PYTHONPATH": str(ROOT)},
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict", "--config", str(config_path),
                          "--output", str(output_dir / "run"), "--datasets", *TASKS],
        "timeout_seconds": 1800, "termination_grace_seconds": 20, "queued": False, "automatic_retry": False, "resume": False,
        "resource_evidence": {"bf16_actual_tasks": timing, "native_reports": run_proofs,
            "bf16_generation_seconds": sum(v["generation_seconds"] for v in timing.values()),
            "boundary": "1800s is a planning cap, not measured compressed runtime. BF16 generation totals about422s; allow loading/group8 and longer responses. Llama SDPA compressed long-context resource/time remains unmeasured."},
        "scoring": "Official check_prediction/score_example on each complete200; unrounded task scores and two-task macro. Preserve fractional scores; do not run the21-task scorer on partial results.",
        "scope": "Two of21 official LongBench tasks only; not full LongBench or nine-item acceptance.",
        "full_longbench_status": "not_evaluated", "nine_item_goal_status": "not_evaluated",
        "validation_reuse": "Existing complete native2979 save/fresh/full8 D16 group8 proof; no additional GPU smoke test.",
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
    print({"status": result["status"], "checkpoint_id": result["checkpoint_id"],
           "source_count": len(result["source_sha256"]), "evidence_count": len(result["evidence_sha256"]),
           "preparation_sha256": sha256(args.output_dir / "preparation.json"), "cuda_initialized": False}, flush=True)
