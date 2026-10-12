"""Reuse the production subset audit and compare the separately exported final state."""
import argparse
import json
from pathlib import Path

from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import SOURCE_ID, require
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state


def verify(output):
    output = Path(output)
    training = json.loads((output / "training.json").read_text())
    exported = json.loads((output / "step_0004/final_model/export_record.json").read_text())
    audit = json.loads((output / "step_0004/audit.json").read_text())
    require(training["status"] == "trained" and training["training_result"]["global_step"] == 4,
            "Four-step native training is incomplete.")
    require(audit["status"] == "verified" and audit["source_checkpoint_id"] == SOURCE_ID
            and audit["train_mode"] == "decoder" and audit["selected_global_step"] == 4
            and audit["selected_modules"] == training["selected_targets"]
            and audit["frozen_tensors_compared"] == 2991
            and audit["frozen_code_bank_count"] == audit["code_bank_count"] == 504
            and audit["selected_decoder_tensor_count"] == 432 and audit["changed_code_bits"] == 0
            and audit["selected_decoder_fp32_changed_elements"] > 0
            and audit["selected_decoder_matches_fp32_cast_exactly"]
            and audit["selected_codes_match_frozen_source_exactly"]
            and audit["selected_exact_sidecar_sha256"] is None
            and audit["fresh_strict_reload_state_exact"] and audit["rank_zero"],
            "The unchanged native subset export/freeze/strict audit did not satisfy OV72.")
    require(exported["source_global_step"] == 4 and exported["source_max_steps"] == 2000
            and exported["restored_decoder_tensors"] == 432 and exported["optimizer_steps_executed"] == 0
            and exported["sparse_bit_restore"] is None, "Independent export changed step/schedule or bit state.")
    native = _load_state(Path(training["training_result"]["saved_model_dir"]))
    independent = _load_state(output / "step_0004/final_model")
    compared = _assert_state_equal(native, independent, "Native final versus independent long-context step4")
    require(compared == 3423, "Incomplete native/independent state inventory.")
    record = {"status": "verified", "diagnostic_not_quality": True, "source_checkpoint_id": SOURCE_ID,
              "checkpoint_id": audit["checkpoint_id"], "native_vs_independent_tensors_compared": compared,
              "training_sha256": sha256(output / "training.json"),
              "audit_sha256": sha256(output / "step_0004/audit.json"),
              "export_sha256": sha256(output / "step_0004/final_model/export_record.json"),
              "bit_budget": audit["bit_budget"], "steps": training["observations"]["steps"],
              "batches": training["observations"]["batches"], "quality_claim": "No downstream or LongBench score evaluated."}
    require(not (output / "verification.json").exists(), "Do not overwrite an existing verification.")
    write_json(output / "verification.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.output)
