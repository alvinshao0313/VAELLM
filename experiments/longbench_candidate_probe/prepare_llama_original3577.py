"""Bind original Llama3577 to the same official retrieval400 as e767 and BF16."""
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

FULL8 = ROOT / "result/llama31_8b_w25_20261004/remaining_prefix_block32_s500_full8"
E767 = R2 / "llama_best1600_d16_retrieval400"
UUID = "35776715-2707-45a9-9fb4-2da1efef3f0e"


def prepare(output):
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not torch.cuda.is_initialized(), "Require public cwd and CUDA-hidden bitvae.")
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh output; no resume or overwrite.")
    launch, audit, review = (read(FULL8 / name) for name in
                            ("launch.json", "acceptance.json", "independent_review.json"))
    checkpoint = Path(audit["checkpoint_dir"])
    meta = read(checkpoint / "checkpoint_meta.json")
    require({x["checkpoint_id"] for x in (launch, audit, review, meta)} == {UUID}
            and launch["status"] == "evaluated" and launch["exit_code"] == 0
            and review["status"] == "independently_reviewed", "Original3577 identity/evaluation mismatch.")
    require(all(launch["stages"][k]["exit_code"] == 0 for k in ("mmlu", "seven"))
            and launch["stages"]["summary"]["exit_code"] == 1 and audit["status"] == "not_met"
            and all((FULL8 / (k + ".exit")).read_text().strip() == "0" for k in ("mmlu", "seven", "run"))
            and (FULL8 / "summary.exit").read_text().strip() == "1",
            "Summary1 is the recorded quality failure, not a failed model evaluation.")
    require(audit["compressed"]["leaf_count"] == 64 and audit["compressed"]["sample_count"] == 24742
            and audit["compressed"]["mean_percent"] == launch["mean_percent"] == review["mean_percent"]
            and audit["checkpoint_meta_sha256"] == sha256(checkpoint / "checkpoint_meta.json")
            and review["tensor_count"] == 2979 and review["lora_state_keys"] == 0
            and review["original_norm_tensors_exact"] == 65 and len(meta["converted_modules"]) == 224,
            "Require complete full8 and native2979/BF16 original-source proof.")
    budget = audit["bit_budget"]
    require(budget["user_linear_bits"] == budget["packed_linear_bits"] == 17100439552
            and budget["original_linear_weights"] == 6979321856 and budget["code_padding_bits"] == 0
            and budget["budget_passed"] and not budget["contract_violations"]
            and budget["components"]["decoder"]["dtypes"] == ["bfloat16"], "Original storage contract differs.")
    check_hashes(review["checked_source_sha256"]); check_hashes(review["evidence_sha256"])
    proof, e_review = read(E767 / "preparation.json"), read(E767 / "review.json")
    require(e_review["status"] == "verified" and e_review["execution_exit_code"] == 0
            and e_review["samples"] == 400 and e_review["actual_input_tokens_and_original_ids_exact"]
            and e_review["generation_options_effective_and_decoded_tokens_verified"]
            and e_review["scores_sha256"] == sha256(E767 / "scores.json"), "Require complete audited e767400.")
    dense_config, e_config = read(DENSE / "config.json"), read(E767 / "config.json")
    config = make_config(dense_config["archive"]["path"], kind="compressed",
                         hf_path=dense_config["model"]["hf_path"], checkpoint=checkpoint, mode="full")
    for key in ("schema", "mode", "protocol", "archive", "package_versions", "model_policy", "prewarm_group_size", "device"):
        require(config[key] == dense_config[key] == e_config[key], "Llama protocol differs: " + key)
    require(config["model"]["checkpoint_id"] == UUID and config["model_policy"]["attn_implementation"] == "sdpa"
            and config["model_policy"]["rope_scaling"]["rope_type"] == "llama3", "Wrong model/native Llama policy.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    controls, evidence = {}, dict(review["evidence_sha256"])
    for label, directory in (("bf16", DENSE), ("e767", E767)):
        scores = read(directory / "scores.json")
        predictions_root = directory if label == "bf16" else directory / "run"
        tasks = {}
        for task in TASKS:
            rows = [r for r in index if r["dataset"] == task]
            path = predictions_root / "predictions" / (task + ".jsonl")
            preds = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows) == len(preds) == 200 and [r["row_index"] for r in rows] == list(range(200))
                    and sha256(path) == scores["prediction_sha256"][str(path)], "Require complete original200 rows.")
            values = []
            for pred, row in zip(preds, rows):
                check_prediction(pred, row, sha256(directory / "config.json"))
                values.append(score_example(task, pred["pred"], row["answers"], row["all_classes"]))
            actual = 100 * sum(values) / 200
            score_key = "score" if label == "bf16" else "score_percent"
            require(abs(actual - scores["tasks"][task][score_key]) < 1e-10, "Official fractional score differs.")
            tasks[task] = {"score": actual, "sample_count": 200}
            evidence[str(path)] = sha256(path)
        controls[label] = {"directory": str(directory), "prediction_root": str(predictions_root), "tasks": tasks}
        for name in ("scores.json", "config.json"):
            evidence[str(directory / name)] = sha256(directory / name)
    sources = dict(review["checked_source_sha256"])
    for pins in (proof["source_sha256"], config["source_sha256"], {str(Path(__file__).resolve()): sha256(__file__)}):
        for path, digest in pins.items():
            require(path not in sources or sources[path] == digest, "Source differs: " + path)
            sources[path] = digest
    for path in (FULL8 / "launch.json", FULL8 / "acceptance.json", FULL8 / "independent_review.json",
                 checkpoint / "checkpoint_meta.json", checkpoint / "config.json", E767 / "preparation.json",
                 E767 / "review.json", DENSE / "score_verification.json", Path(e_review["execution_report"])):
        evidence[str(path)] = sha256(path)
    check_hashes(sources); check_hashes(evidence)
    config_path = output / "config.json"
    record = {"schema": "llama_original3577_retrieval400_preparation_v1", "status": "prepared_unbound", "gpu": None,
        "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": UUID,
        "config_path": str(config_path), "output_dir": str(output / "run"), "datasets": list(TASKS),
        "sample_count": 400, "task_samples": {task: 200 for task in TASKS}, "source_sha256": sources,
        "evidence_sha256": evidence, "source_bit_budget": budget, "controls": controls,
        "completed_candidate": {"checkpoint_dir": str(checkpoint), "checkpoint_id": UUID,
            "full8_mean_percent": review["mean_percent"], "review_path": str(FULL8 / "independent_review.json"),
            "native_tensor_count": 2979, "original_norm_tensors_exact": 65, "summary_exit1": "historical quality_not_met"},
        "environment": {k: v for k, v in proof["environment"].items() if k != "CUDA_VISIBLE_DEVICES"},
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict", "--config", str(config_path),
                          "--output", str(output / "run"), "--datasets", *TASKS],
        "timeout_seconds": 900, "termination_grace_seconds": 20, "queued": False, "automatic_retry": False, "resume": False,
        "resource_evidence": {"e767_actual_seconds": e_review["duration_seconds"], "review_path": str(E767 / "review.json"),
            "boundary": "900s is a planning cap; e767 native LL retrieval400 completed in536s. Require parent callback window >=1220s."},
        "duplicate_check": "Known original3577 has fixed16 only; no existing complete200+200 predictions found.",
        "selection": "All archived rows0..199 for both official tasks; no filtering or sampling.",
        "scoring": "Native check_prediction/score_example, preserving fractional credit; two-task macro only.",
        "validation_reuse": "Original2979 native strict/full8/BF16 storage proof plus actual e767 SDPA/FP32RoPE/group8 retrieval400.",
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
