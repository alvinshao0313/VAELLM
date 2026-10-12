"""Bind the audited Qwen FineWeb decoder/all-norm step400 endpoint to two complete retrieval tasks."""
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
TRAIN = R2 / "general_lm_norm_recovery_s2000/qwen3"
STEP = TRAIN / "step_0400"
PROTOCOL_PROOF = R2 / "l18_gate_retrieval400"
CHECKPOINT_ID = "27eea2ff-3e5e-4be0-831e-0290a18e3993"
SOURCE_ID = "f927ffda-dd5b-4d00-b63a-5e839a182d70"
TASKS = ("passage_retrieval_en", "passage_retrieval_zh")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(output_dir):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "Use CUDA-hidden bitvae CPU preparation.")
    require(Path.cwd() == ROOT, "Use the public source root and native LongBench runner.")
    output_dir = Path(output_dir).resolve()
    require(not any((output_dir / name).exists() for name in ("config.json", "preparation.json", "run")),
            "Use a fresh independent output.")
    training = read(TRAIN / "launch.json")
    step, audit, review, exported, full8, meta = (read(STEP / name) for name in (
        "step_export.json", "audit.json", "independent_review.json", "final_model/export_record.json",
        "full8_summary.json", "final_model/checkpoint_meta.json"))
    require(step["status"] == "evaluated" and step["global_step"] == 400 and step["max_steps"] == 2000
            and step["checkpoint_id"] == CHECKPOINT_ID and step["round_base_checkpoint_id"] == SOURCE_ID
            and set(step["stages"]) == {"export", "audit", "mmlu", "seven"}
            and all(item["exit_code"] == 0 for item in step["stages"].values())
            and all((STEP / (stage + ".exit")).read_text().strip() == "0" for stage in step["stages"]),
            "Require the completed actual400 callback; parent2000 training may continue.")
    require(audit["status"] == review["status"] == "verified" and exported["status"] == "exported"
            and {x["checkpoint_id"] for x in (step, audit, review, exported, full8, meta)} == {CHECKPOINT_ID}
            and audit["source_checkpoint_id"] == review["source_checkpoint_id"]
            == exported["round_base_checkpoint_id"] == training["source_checkpoint_id"] == SOURCE_ID
            and audit["selected_checkpoint_id"] == step["selected_checkpoint_id"]
            == exported["source_checkpoint_id"] == review["selected_native_checkpoint_id"]
            and audit["selected_global_step"] == exported["source_global_step"] == review["global_step"] == 400,
            "The export, native400 identity and independent review must refer to this endpoint.")
    require(step["audit_sha256"] == full8["audit_sha256"] == sha256(STEP / "audit.json")
            and step["full8_summary_sha256"] == sha256(STEP / "full8_summary.json")
            and full8["status"] == "full8_evaluation_verified" and full8["leaf_count"] == 64
            and full8["sample_count"] == 24742 and review["tasks"]["mmlu"]["correct"] == 8560,
            "Bind the real complete-eight result and its previously verified raw evidence.")
    require(audit["train_mode"] == "decoder" and audit["selected_decoder_tensor_count"] == 1512
            and audit["selected_module_count"] == audit["changed_decoder_module_count"] == 252
            and audit["selected_decoder_fp32_changed_elements"] > 0
            and audit["selected_decoder_matches_fp32_cast_exactly"]
            and audit["frozen_tensors_compared"] == 1766 and audit["frozen_bytes_unchanged"]
            and audit["code_bank_count"] == 504 and audit["changed_code_bits"] == 0
            and audit["selected_codes_match_frozen_source_exactly"]
            and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == 3423
            and audit["fresh_strict_reload_state_exact"] and audit["rank_zero"]
            and audit["inference_specs_unchanged_except_activation_checkpointing"]
            and audit["native_decoder_dtype"] == "torch.bfloat16",
            "Reuse the completed FP32-to-BF16, frozen complement/code, geometry and strict deployment audit.")
    require(audit["norm_train_mode"] == meta["norm_train_mode"] == "all"
            and audit["lm_head_train_mode"] == meta["lm_head_train_mode"] == "none"
            and audit["mutable_tensor_count"] == 1657 and audit["norm_tensor_count"] == 145
            and audit["component_counts"] == exported["mutable_component_counts"] == {"decoder": 1512, "norm": 145}
            and audit["changed_norm_fp32_elements"] > 0 and audit["changed_norm_bf16_elements"] > 0
            and audit["norm_matches_selected_fp32_cast_exactly"]
            and audit["non_decoder_non_norm_bytes_unchanged"] and audit["original_head_bytes_unchanged"]
            and audit["all_packed_banks_match_source_exactly"]
            and audit["selected_training_state_sha256"] == exported["source_files_sha256"][
                str(Path(exported["source_checkpoint"]) / "training_model_state.pt")],
            "Require the completed actual decoder+145norm union; frozen complement is1766, not decoder-only1911.")
    budget = audit["bit_budget"]
    require(budget["user_linear_bits"] == 17065672704 and budget["original_linear_weights"] == 6945767424
            and budget["user_linear_bits"] * 2 <= budget["original_linear_weights"] * 5
            and all(budget[k] == training["source_bit_budget"][k]
                    for k in ("user_linear_bits", "original_linear_weights", "components")),
            "Require the actual unchanged integer source budget, including all storage components.")
    check_hashes(training["source_sha256"])
    check_hashes(training["input_evidence_sha256"])
    check_hashes(review["evidence_sha256"])
    for stat in (training["source_weight_stat"], *training["input_data_stat"], *training["hf_source"]["weight_files"]):
        check_stat(stat)
    check_hashes(training["hf_source"]["metadata_sha256"])

    previous, protocol_review = read(PROTOCOL_PROOF / "preparation.json"), read(PROTOCOL_PROOF / "review.json")
    previous_config = read(previous["config_path"])
    require(protocol_review["status"] == "verified" and sha256(previous["config_path"]) == previous["config_sha256"],
            "Require the completed official400 protocol/control proof.")
    check_hashes(previous["source_sha256"])
    check_hashes(previous["evidence_sha256"])
    check_hashes(protocol_review["evidence_sha256"])
    config = make_config(previous_config["archive"]["path"], kind="compressed",
                         hf_path=training["architecture"]["base_model_path"], checkpoint=STEP / "final_model", mode="full")
    require(config["model"]["checkpoint_id"] == CHECKPOINT_ID
            and all(config[k] == previous_config[k] for k in
                    ("schema", "mode", "protocol", "archive", "package_versions", "model_policy", "prewarm_group_size", "device"))
            and all(config["model"][k] == previous_config["model"][k] for k in
                    ("kind", "hf_path", "hf_metadata_sha256", "hf_weight_stats")),
            "Only the compressed endpoint identity may differ from the existing full400 native protocol.")
    check_config(config)
    _, index = load_archive(config["archive"]["path"], check_token_file=False)
    rows = {task: [row for row in index if row["dataset"] == task] for task in TASKS}
    require(all(len(items) == 200 and [r["row_index"] for r in items] == list(range(200))
                for items in rows.values()), "Require all original EN200 and ZH200 rows; no subsampling.")

    sources = dict(training["source_sha256"])
    for path, digest in config["source_sha256"].items():
        require(path not in sources or sources[path] == digest, "Training/native inference source identity differs.")
        sources[path] = digest
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    evidence = dict(training["input_evidence_sha256"])
    native_dir = Path(exported["source_checkpoint"])
    reusable_review_evidence = {path: digest for path, digest in review["evidence_sha256"].items()
                               if not Path(path).is_relative_to(native_dir)}
    evidence.update(reusable_review_evidence)
    for name in ("independent_review.json", "final_model/config.json"):
        evidence[str(STEP / name)] = sha256(STEP / name)
    for name in ("preparation.json", "config.json", "review.json"):
        evidence[str(PROTOCOL_PROOF / name)] = sha256(PROTOCOL_PROOF / name)
    for control in previous["controls"].values():
        directory = Path(control["directory"])
        for name in ("config.json", "scores.json", *("predictions/" + task + ".jsonl" for task in TASKS)):
            path = directory / name
            require(str(path) in previous["evidence_sha256"] and sha256(path) == previous["evidence_sha256"][str(path)],
                    "The already paired, officially scored full-task controls must remain byte-identical.")
            evidence[str(path)] = sha256(path)
    check_hashes(sources)
    check_hashes(evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    write_json(config_path, config)
    record = {"schema": "general_lm_norm_qwen400_full_retrieval_preparation_v1", "status": "prepared_unbound", "gpu": None,
        "source_root": str(ROOT), "python_executable": sys.executable, "checkpoint_id": CHECKPOINT_ID,
        "config_path": str(config_path), "config_sha256": sha256(config_path), "output_dir": str(output_dir / "run"),
        "datasets": list(TASKS), "sample_count": 400, "task_samples": {task: 200 for task in TASKS},
        "selection": "All original archived rows0..199 per task in native archive order, without filtering or resampling.",
        "input_lengths": {task: {"min": min(r["input_tokens"] for r in items), "max": max(r["input_tokens"] for r in items)}
                          for task, items in rows.items()},
        "source_sha256": sources, "evidence_sha256": evidence, "source_bit_budget": budget,
        "completed_candidate": {"directory": str(STEP), "checkpoint_dir": str(STEP / "final_model"),
            "checkpoint_id": CHECKPOINT_ID, "source_checkpoint_id": SOURCE_ID, "global_step": 400,
            "mean_percent": full8["mean_percent"], "mmlu_correct": 8560, "audit_path": str(STEP / "audit.json"),
            "review_path": str(STEP / "independent_review.json"),
            "native_training_state_sha256": audit["selected_training_state_sha256"],
            "native_checkpoint_reload_required": False,
            "mutable_component_counts": {"decoder": 1512, "norm": 145}, "frozen_tensors": 1766,
            "norm_training": "All native norms, LR1e-5, FP32 training master to audited BF16 deployment.",
            "rotating_native_evidence_not_pinned": sorted(set(review["evidence_sha256"]) - set(reusable_review_evidence)),
            "boundary": "Native400 may rotate later; retained audited exported state, not native optimizer files, is the inference input."},
        "native_generation_proof": {"directory": str(PROTOCOL_PROOF), "sample_count": 400,
            "scope": "Reuse original model/tokenizer, unchanged inference geometry and native runtime; this candidate has not generated LongBench yet."},
        "controls": previous["controls"], "timeout_seconds": 900, "termination_grace_seconds": 20,
        "environment": previous["environment"],
        "launcher_argv": [sys.executable, "-u", "-m", "experiments.longbench_v1.predict", "--config", str(config_path),
                          "--output", str(output_dir / "run"), "--datasets", *TASKS],
        "queued": False, "automatic_retry": False, "resume": False,
        "scope": "Two complete official LongBench v1 retrieval tasks;400 predictions cover only2/21 tasks.",
        "comparison_boundary": "Historical norm400 uses task50 data and norm-only updates; it is a secondary historical control, not an isolated FineWeb decoder+norm ablation.",
        "scoring": previous["scoring"], "full_longbench_status": "not_evaluated", "nine_item_goal_status": "not_evaluated",
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
        "source_count": len(result["source_sha256"]), "evidence_count": len(result["evidence_sha256"]),
        "sample_count": result["sample_count"], "cuda_initialized": result["cuda_initialized"],
        "preparation_sha256": sha256(args.output_dir / "preparation.json")}, ensure_ascii=False))
