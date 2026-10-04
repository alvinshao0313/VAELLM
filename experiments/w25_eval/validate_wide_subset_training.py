"""Four native H256/bit updates, followed by same-device independent export.

The dispatcher starts export only after this training process exits. This entry
does not implement periodic evaluation, resume, or an alternative loss/data path.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.qwen3_w25.subset_step_export_callback import SubsetStepExportCallback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe


def configuration(record):
    """Match the completed H128 control, changing only source/output/save interval."""
    control = json.loads(Path(record["control_launch"]).read_text())
    expected = list(control["argv"][3:])
    for option, value in (("--student_checkpoint_dir", record["source_checkpoint"]),
                          ("--run_root_dir", record["output_dir"]), ("--save_steps", "2")):
        _require(expected.count(option) == 1, f"Ambiguous control option {option}.")
        expected[expected.index(option) + 1] = value
    _require(record["training_argv"] == expected, "Four-step scientific argv differs from its H128 control.")
    cfg, hf, args = parse_args(expected)
    meta = json.loads((Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json").read_text())
    architecture = model_inventory(meta["base_model_path"])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    _require(list(names) == record["selected_targets"] == ["model.layers.0.mlp.down_proj"], "Expected only L0 down.")
    spec = next(item for item in meta["converted_modules"] if item["name"] == names[0])
    _require(all(item["hidden_dim"] == 256 and item["decoder_type"] == "asymmetric"
                 for item in spec["stage_decoders"]), "Expected the actual widened H256 source.")
    _require(args.max_steps == 2000 and args.save_steps == 2 and args.save_total_limit == 1
             and args.get_warmup_steps(2000) == 20, "Keep the real schedule/warmup and native save2.")
    return cfg, hf, args, names


class TrainingObservations(SubsetStepExportCallback):
    """Reuse real batch/optimizer/inventory observations, with no export callback."""

    def __init__(self, selected_targets):
        # The parent constructor binds a second GPU. Training-only observations
        # deliberately have no export environment or independent child process.
        self.selected_targets = tuple(selected_targets)
        self.diagnostic_stop_step = 4
        self.diagnostic = {"batches": [], "optimizer_steps": [], "native_saves": []}
        self._input_hook = None
        self._observed_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        _require(state.global_step == 0, "Short validation starts from a fresh round base.")
        return super().on_train_begin(args, state, control, **kwargs)

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        trainer = json.loads((checkpoint / "trainer_state.json").read_text())
        meta = json.loads((checkpoint / "checkpoint_meta.json").read_text())
        _require(trainer["global_step"] == state.global_step and state.global_step in (2, 4),
                 "Native saved step differs from the actual optimizer boundary.")
        self.diagnostic["native_saves"].append({"step": state.global_step, "checkpoint_dir": str(checkpoint),
            "checkpoint_id": meta["checkpoint_id"], "meta_sha256": _sha256(checkpoint / "checkpoint_meta.json"),
            "trainer_state_sha256": _sha256(checkpoint / "trainer_state.json"),
            "files": {p.name: p.stat().st_size for p in checkpoint.iterdir() if p.is_file()}})
        return control


def train(record):
    cfg, hf, args, names = configuration(record)
    callback = TrainingObservations(names)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    try:
        result = run(cfg, hf, args, extra_callbacks=(callback,))
        _require(result["global_step"] == 4 and result["round_base_checkpoint_id"] == record["source_checkpoint_id"],
                 "Native training did not finish at four steps from the pinned initial model.")
        observed = callback.diagnostic
        inventory = observed["trainable_inventory"]
        expected = record["expected_inventory"]
        _require(inventory["decoder_tensors"] == expected["selected_decoder_tensors"]
                 and inventory["decoder_elements"] == expected["selected_decoder_elements"]
                 and inventory["score_elements"] == expected["active_score_elements"], "Wrong trainable inventory.")
        _require([item["step"] for item in observed["optimizer_steps"]] == [1, 2, 3, 4]
                 and [item["step"] for item in observed["native_saves"]] == [2, 4]
                 and {item["step"] for item in observed["batches"]} == {1, 2, 3, 4}
                 and all(item["shape"][0] == 4 and item["shape"][1] <= 1024 for item in observed["batches"]),
                 "Actual B4 forward/update/native-save evidence is incomplete.")
        _require(all(item["main_moment_dtypes"] == ["torch.float32"] for item in observed["optimizer_steps"]),
                 "Decoder Adam moments lost FP32 precision.")
        report = {"status": "trained", "training_result": result, "observations": observed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "diagnostic_not_quality": True,
            "resource_coverage": "Actual four native dynamic batches; not a full-data worst-case bound."}
        (Path(record["output_dir"]) / "training.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    finally:
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def verify(record):
    output = Path(record["output_dir"])
    trained = json.loads((output / "training.json").read_text())
    audit = json.loads((output / "step_0004/audit.json").read_text())
    exported = json.loads((output / "step_0004/final_model/export_record.json").read_text())
    _require(audit["status"] == "verified" and audit["source_checkpoint_id"] == record["source_checkpoint_id"]
             and audit["selected_modules"] == record["selected_targets"]
             and audit["frozen_tensors_compared"] == 3415 and audit["code_bank_count"] == 504
             and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == 3423
             and audit["selected_decoder_fp32_changed_elements"] > 0
             and audit["selected_decoder_matches_fp32_cast_exactly"]
             and audit["selected_codes_match_current_snapshot_exactly"]
             and audit["fresh_strict_reload_state_exact"] and audit["rank_zero"], "Independent export audit failed.")
    _require(exported["source_global_step"] == 4 and exported["source_max_steps"] == 2000
             and exported["restored_decoder_tensors"] == 6 and exported["optimizer_steps_executed"] == 0,
             "Export must restore step4 without another training step.")
    native = _load_state(Path(trained["training_result"]["saved_model_dir"]))
    independent = _load_state(output / "step_0004/final_model")
    compared = _assert_state_equal(native, independent, "Native final versus independently restored step4")
    del native, independent
    gc.collect()
    report = {"status": "verified", "source_checkpoint_id": record["source_checkpoint_id"],
        "checkpoint_id": audit["checkpoint_id"], "native_vs_step4_tensors_compared": compared,
        "training_report_sha256": _sha256(output / "training.json"),
        "audit_sha256": _sha256(output / "step_0004/audit.json"),
        "export_record_sha256": _sha256(output / "step_0004/final_model/export_record.json"),
        "changed_decoder_fp32_elements": audit["selected_decoder_fp32_changed_elements"],
        "changed_decoder_bf16_elements": audit["changed_decoder_elements"],
        "changed_code_bits": audit["changed_code_bits"], "bit_budget": audit["bit_budget"],
        "export_peak_allocated_bytes": exported["peak_cuda_allocated_bytes"],
        "export_peak_reserved_bytes": exported["peak_cuda_reserved_bytes"], "diagnostic_not_quality": True}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("train", "verify"))
    args = parser.parse_args()
    record = json.loads(args.launch_record.read_text())
    {"train": train, "verify": verify}[args.phase](record)


if __name__ == "__main__":
    main()
