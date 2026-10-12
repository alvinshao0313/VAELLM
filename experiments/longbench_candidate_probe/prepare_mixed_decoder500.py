"""Bind the audited mixed-calibration gate decoder500 endpoint to official EN/ZH retrieval400."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "result/qwen3_8b_mmlu65_20261004"
CANDIDATE = R2 / "gate_kl_calibration_decoder_full8/alpaca64_fineweb64"
PROTOCOL_PROOF = R2 / "l18_gate_retrieval400"
CHECKPOINT_ID = "55a3aeb5-9b37-43ae-9da9-6964c9d6abac"
SOURCE_ID = "f927ffda-dd5b-4d00-b63a-5e839a182d70"
REVIEW_SHA256 = "6d938c07ecafd8f9dac641537e4b2ecfce90f2ffbeab2ab9938215ed764f1b86"
TASKS = ("passage_retrieval_en", "passage_retrieval_zh")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(output_dir):
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not torch.cuda.is_initialized(), "Use the public source root and CUDA-hidden bitvae.")
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "Use a fresh independent output directory.")
    checkpoint = CANDIDATE / "final_model"
    launch, audit, review, summary, meta = (read(CANDIDATE / name) for name in
        ("launch.json", "audit.json", "review.json", "full8_summary.json", "final_model/checkpoint_meta.json"))
    require(sha256(CANDIDATE / "review.json") == REVIEW_SHA256
            and launch["status"] == "completed" and launch["exit_code"] == launch["dispatcher_exit_code"] == 0
            and launch["arm"] == audit["arm"] == summary["arm"] == "alpaca64_fineweb64"
            and audit["status"] == review["status"] == "verified"
            and {x["checkpoint_id"] for x in (launch, audit, review, summary, meta)} == {CHECKPOINT_ID}
            and all(x["source_checkpoint_id"] == SOURCE_ID for x in (launch, audit, review, summary)),
            "Require the exact completed mixed-calibration fitted endpoint and independent review.")
    require(set(launch["stages"]) == {"deploy", "mmlu", "seven"}
            and all(stage["exit_code"] == 0 for stage in launch["stages"].values())
            and all((CANDIDATE / (stage + ".exit")).read_text().strip() == "0"
                    for stage in ("deploy", "mmlu", "seven", "dispatcher", "outer")),
            "All original deployment and complete-eight stages must have exited successfully.")
    require(summary["status"] == "full8_evaluation_verified" and summary["leaf_count"] == 64
            and summary["sample_count"] == 24742 and summary["actual_configs_versions_shots_counts_equal_three_controls"]
            and summary["audit_sha256"] == launch["audit_sha256"] == sha256(CANDIDATE / "audit.json")
            and launch["full8_summary_sha256"] == sha256(CANDIDATE / "full8_summary.json")
            and launch["mean_percent"] == summary["mean_percent"] == review["mean_percent"]
            and summary["task_metrics"] == review["task_metrics"]
            and review["integer_counts"]["mmlu"]["correct"] == 8576
            and review["integer_counts"]["mmlu"]["total"] == 14042,
            "Bind the real full8 result; its score is evidence, not a retrieval acceptance threshold.")
    require(audit["patched_local_tensor_count"] == 13 and audit["frozen_outside_target_tensors"] == 3410
            and audit["selected_decoder_tensors"] == 6 and audit["selected_projections"] == ["gate_proj"]
            and audit["stage_changed_bits"] == launch["patch_contract"]["stage_changed_bits"] == [15468, 660]
            and audit["changed_code_bits"] == 16128 and audit["source_packed_bank_count"] == 504
            and audit["frozen_outside_target_packed_bank_count"] == 502
            and audit["frozen_outside_target_packed_banks_exact"]
            and audit["frozen_local_nonmutable_tensors"] == 5 and audit["frozen_local_nonmutable_exact"]
            and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == 3423
            and all(audit[key] for key in ("fresh_strict_reload_state_exact", "endpoint_state_exact",
                    "frozen_outside_target_exact", "native_full_weight_reload_exact", "integer_budget_unchanged", "rank_zero"))
            and audit["endpoint_sha256"] == launch["endpoint_sha256"] == summary["endpoint_sha256"]
            and audit["checkpoint_meta_sha256"] == sha256(checkpoint / "checkpoint_meta.json"),
            "Reuse this endpoint's two-bank/six-decoder patch and original protection/rotation/frozen-complement proof.")
    require(len(audit["changed_decoder_elements"]) == 6 and all(n > 0 for n in audit["changed_decoder_elements"].values()),
            "Require the actual fitted decoder endpoint, not its unfitted code proposal.")
    budget = audit["bit_budget"]
    require(budget["user_linear_bits"] == budget["packed_linear_bits"] == 17065672704
            and budget["original_linear_weights"] == 6945767424 and budget["code_padding_bits"] == 0
            and budget["budget_passed"] and not budget["contract_violations"]
            and budget["components"]["decoder"]["dtypes"] == ["bfloat16"]
            and all(budget[key] == launch["source_bit_budget"][key]
                    for key in ("user_linear_bits", "original_linear_weights", "components")),
            "Keep the actual source integer storage budget, including decoder, protection, indices and rotation.")
    check_hashes(launch["source_sha256"]); check_hashes(launch["input_sha256"])
    check_hashes(review["evidence_sha256"])
    for item in (launch["source_weight_stat"], audit["checkpoint_state_stat"], *launch["hf_weight_stats"]):
        check_stat(item)

    previous, protocol_review = read(PROTOCOL_PROOF / "preparation.json"), read(PROTOCOL_PROOF / "review.json")
    old_config = read(previous["config_path"])
    require(protocol_review["status"] == "verified" and protocol_review["execution_status"] == "partition_complete"
            and protocol_review["execution_exit_code"] == 0 and sha256(previous["config_path"]) == previous["config_sha256"],
            "Require the existing complete400 native protocol proof; no repeat model short test.")
    config = make_config(old_config["archive"]["path"], kind="compressed",
                         hf_path=launch["architecture"]["base_model_path"], checkpoint=checkpoint, mode="full")
    require(config["model"]["checkpoint_id"] == CHECKPOINT_ID
            and all(config[key] == old_config[key] for key in ("schema", "mode", "protocol", "archive",
                    "package_versions", "model_policy", "prewarm_group_size", "device"))
            and all(config["model"][key] == old_config["model"][key]
                    for key in ("kind", "hf_path", "hf_metadata_sha256", "hf_weight_stats")),
            "Only candidate identity may differ from the original full400 official generation protocol.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    rows = {task: [row for row in index if row["dataset"] == task] for task in TASKS}
    require(all(len(items) == 200 and [r["row_index"] for r in items] == list(range(200)) for items in rows.values()),
            "Use both complete official tasks in original order, without filtering or sampling.")
    sources = dict(launch["source_sha256"])
    for path, digest in config["source_sha256"].items():
        require(path not in sources or sources[path] == digest, "Native source identity differs: " + path)
        sources[path] = digest
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    evidence = {**launch["input_sha256"], **review["evidence_sha256"]}
    for path in (CANDIDATE / "launch.json", CANDIDATE / "audit.json", CANDIDATE / "review.json",
                 CANDIDATE / "full8_summary.json", checkpoint / "config.json", checkpoint / "checkpoint_meta.json",
                 PROTOCOL_PROOF / "preparation.json", PROTOCOL_PROOF / "review.json", Path(previous["config_path"])):
        evidence[str(path)] = sha256(path)
    controls = {key: previous["controls"][key] for key in ("bf16", "f927")}
    for control in controls.values():
        directory = Path(control["directory"]); control_config = read(directory / "config.json")
        require(all(control_config[key] == config[key] for key in ("protocol", "archive", "package_versions")),
                "Historical controls must share the official generation protocol.")
        for name in ("config.json", "scores.json", *("predictions/" + task + ".jsonl" for task in TASKS)):
            path = directory / name
            require(previous["evidence_sha256"].get(str(path)) == sha256(path), "Previously verified control changed: " + str(path))
            evidence[str(path)] = sha256(path)
    check_hashes(sources); check_hashes(evidence)
    config_path = output_dir / "config.json"
    record = {"schema": "mixed_gate_decoder500_full_retrieval_preparation_v1", "status": "prepared_unbound", "gpu": None,
        "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": CHECKPOINT_ID,
        "config_path": str(config_path), "output_dir": str(output_dir / "run"), "datasets": list(TASKS),
        "sample_count": 400, "task_samples": {task: 200 for task in TASKS},
        "selection": "All original archived rows0..199 per task; no filtering/resampling.",
        "input_lengths": {task: {"min": min(r["input_tokens"] for r in items), "max": max(r["input_tokens"] for r in items)} for task, items in rows.items()},
        "source_sha256": sources, "evidence_sha256": evidence, "source_bit_budget": budget,
        "completed_candidate": {"directory": str(CANDIDATE), "checkpoint_dir": str(checkpoint), "checkpoint_id": CHECKPOINT_ID,
            "source_checkpoint_id": SOURCE_ID, "arm": "alpaca64_fineweb64", "full8_mean_percent": summary["mean_percent"],
            "mmlu_correct": 8576, "audit_path": str(CANDIDATE / "audit.json"), "review_path": str(CANDIDATE / "review.json"),
            "patched_local_tensor_count": 13, "selected_decoder_tensors": 6, "changed_code_bits": 16128,
            "stage_changed_bits": [15468, 660], "frozen_outside_target_tensors": 3410, "frozen_local_nonmutable_tensors": 5,
            "native_union_tensors": 3423},
        "controls": controls, "environment": {**previous["environment"], "PYTHONPATH": str(ROOT)},
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict", "--config", str(config_path),
                          "--output", str(output_dir / "run"), "--datasets", *TASKS],
        "timeout_seconds": 900, "termination_grace_seconds": 20, "queued": False, "automatic_retry": False, "resume": False,
        "resource_evidence": {"completed_native400_directory": str(PROTOCOL_PROOF),
            "boundary": "Existing original-tokenizer/native-group8/FA2/FP32RoPE protocol proof; this candidate has not generated retrieval predictions."},
        "scoring": previous["scoring"],
        "comparison_boundary": "BF16 and f927 are existing references; this endpoint includes four-group mixed-calibration code proposals and conditional MLP decoder500. Retrieval does not isolate the calibration or fitting effect.",
        "scope": "Two complete official LongBench v1 retrieval tasks; only2/21, not full LongBench or nine-item acceptance.",
        "full_longbench_status": "not_evaluated", "nine_item_goal_status": "not_evaluated", "model_mutations": False,
        "model_exported": False, "cuda_initialized": torch.cuda.is_initialized()}
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
