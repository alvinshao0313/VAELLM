"""Four native decoder-only updates and an independent saved-step audit.

Reuse the existing observations/save/stop callbacks with decoder-only inventory
derived from the real architecture and source checkpoint. Training, loss,
optimizer, finalization and export remain production paths.
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
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import persistent_inventory, selected_names, validate_recipe
from experiments.w25_eval.validate_wide_subset_training import TrainingObservations


def expected_inventory(source_dir, architecture, names):
    """Read persistent keys/shapes through CPU mmap, without loading a model."""
    state = _load_state(Path(source_dir))
    decoders = {name: [key for key in state if key.startswith(name + "._parallel_stage_decoder.")]
                for name in architecture["projection_shapes"]}
    banks = {name: [name + ".vq_weight", name + ".vq_weight_s1"] for name in decoders}
    _require(all(len(keys) == 6 for keys in decoders.values())
             and all(key in state and state[key].dtype == torch.uint8
                     for keys in banks.values() for key in keys),
             "Source does not contain every native six-leaf/two-bank decoder.")
    _require(bool(names) and len(set(names)) == len(names) and set(names) <= set(decoders),
             "Selected decoder inventory is absent or duplicated.")
    selected = [key for name in names for key in decoders[name]]
    result = {
        "deployed_projections": len(decoders), "serialized_tensors": len(state),
        "all_decoder_tensors": sum(map(len, decoders.values())),
        "all_code_banks": sum(map(len, banks.values())), "selected_projections": len(names),
        "selected_decoder_tensors": len(selected), "selected_code_banks": 0,
        "active_score_elements": 0, "frozen_tensors": len(state) - len(selected),
        "selected_projection_code_banks": sum(len(banks[name]) for name in names),
        "selected_decoder_elements": sum(state[key].numel() for key in selected),
    }
    del state
    return result


def configuration(record):
    control = json.loads(Path(record["control_launch"]).read_text())
    expected = list(control["argv"][3:])
    for option, value in (("--train_mode", "decoder"), ("--run_root_dir", record["output_dir"]),
                          ("--save_steps", "2")):
        _require(expected.count(option) == 1, f"Ambiguous control option {option}.")
        expected[expected.index(option) + 1] = value
    _require(record["training_argv"] == expected, "Only mode/output/save2 may differ from the matched control.")
    cfg, hf, args = parse_args(expected)
    meta = json.loads((Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json").read_text())
    architecture = model_inventory(meta["base_model_path"])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    _require(list(names) == record["selected_targets"], "Selected targets differ from the real architecture and CLI.")
    _require(record["expected_inventory"] == expected_inventory(cfg.student_checkpoint_dir, architecture, names),
             "Prepared decoder inventory differs from the actual source storage and architecture.")
    _require(cfg.student_checkpoint_dir == record["source_checkpoint"] == control["source_checkpoint"]
             and meta["checkpoint_id"] == record["source_checkpoint_id"] == control["source_checkpoint_id"],
             "Use the matched pinned fresh source, not a short-run endpoint.")
    _require(cfg.train_mode == "decoder" and tuple(cfg.opt.distill_fp32_components) == ("decoder",)
             and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none"
             and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, "Decoder-only fresh runtime required.")
    _require(args.max_steps == 2000 and args.save_steps == 2 and args.save_total_limit == 1
             and args.get_warmup_steps(2000) == 20 and args.bf16 and not args.fp16,
             "Keep original 2000/warmup20/BF16 and native save2.")
    _require(cfg.opt.batch_size == args.per_device_train_batch_size == 4
             and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
             and cfg.data.model_max_length == 1024, "Keep real B4/max1024/accumulation1.")
    return cfg, hf, args, names, architecture


class DecoderObservations(TrainingObservations):
    def __init__(self, selected_targets, architecture):
        super().__init__(selected_targets)
        self.architecture = architecture

    def on_train_begin(self, args, state, control, **kwargs):
        AuxStepExportCallback.on_train_begin(self, args, state, control, **kwargs)
        _require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 2
                 and args.get_warmup_steps(2000) == 20, "Expected fresh native four-step diagnostic.")
        model = kwargs["model"]
        inventory, banks = persistent_inventory(model, self.architecture)
        expected = {key for name in self.selected_targets for key in inventory[name]}
        active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        _require(set(active) == expected and all(p.dtype == torch.float32 for p in active.values())
                 and not hasattr(model, "sparse_bit_tuning"), "Only selected FP32 decoder leaves may train; no score runtime.")
        optimizer = kwargs["optimizer"]
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        _require(not hasattr(optimizer, "bit_optimizer")
                 and {id(p) for group in optimizer.param_groups for p in group["params"]}
                 == {id(p) for p in active.values()}, "Optimizer must contain exactly the selected decoders.")
        self.diagnostic["trainable_inventory"] = {
            "decoder_keys": sorted(active), "decoder_tensors": len(active),
            "decoder_elements": sum(p.numel() for p in active.values()), "score_keys": [],
            "score_elements": 0, "expected_active_bits": 0, "selected_code_banks": 0,
            "selected_projection_code_banks": sum(len(banks[name]) for name in self.selected_targets),
            "sparse_runtime_present": False}

        def observe_inputs(_module, positional, values):
            inputs = values.get("input_ids")
            if inputs is None and positional:
                inputs = positional[0]
            if inputs is not None:
                mask = values.get("attention_mask")
                self.diagnostic["batches"].append({"step": self._observed_step, "shape": list(inputs.shape),
                    "padded_tokens": inputs.numel(), "attention_tokens": None if mask is None else int(mask.sum().item())})
        self._input_hook = model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
        return control

    def on_save(self, args, state, control, **kwargs):
        result = super().on_save(args, state, control, **kwargs)
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        meta = json.loads((checkpoint / "checkpoint_meta.json").read_text())
        _require(meta["train_mode"] == "decoder" and not (checkpoint / "sparse_bit_tuning").exists()
                 and "proxy_coordinates" not in meta["immutable_resume_contract"]["sparse_bit"],
                 "Decoder native save must not include a SparseBit sidecar or active coordinates.")
        return result


def train(record):
    cfg, hf, args, names, architecture = configuration(record)
    callback = DecoderObservations(names, architecture)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    try:
        result = run(cfg, hf, args, extra_callbacks=(callback,))
        observed = callback.diagnostic
        inventory = observed["trainable_inventory"]
        expected = record["expected_inventory"]
        _require(result["global_step"] == 4 and result["round_base_checkpoint_id"] == record["source_checkpoint_id"],
                 "Native training did not complete exactly four updates from its pinned source.")
        _require(inventory["decoder_tensors"] == expected["selected_decoder_tensors"]
                 and inventory["decoder_elements"] == expected["selected_decoder_elements"]
                 and inventory["selected_projection_code_banks"] == expected["selected_projection_code_banks"]
                 and inventory["score_elements"] == expected["active_score_elements"] == 0,
                 "Wrong decoder-only trainable inventory.")
        _require([x["step"] for x in observed["optimizer_steps"]] == [1, 2, 3, 4]
                 and [x["step"] for x in observed["native_saves"]] == [2, 4]
                 and {x["step"] for x in observed["batches"]} == {1, 2, 3, 4}
                 and all(x["shape"][0] == 4 and x["shape"][1] <= 1024 for x in observed["batches"])
                 and all(x["main_moment_dtypes"] == ["torch.float32"] for x in observed["optimizer_steps"]),
                 "Actual B4 forward/Adam/save2 evidence is incomplete.")
        report = {"status": "trained", "training_result": result, "observations": observed,
            "expected_inventory": expected, "architecture": architecture,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "diagnostic_not_quality": True,
            "resource_coverage": "Four real native dynamic batches; not a full-data worst-case bound."}
        (Path(record["output_dir"]) / "training.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    finally:
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def verify(record):
    configuration(record)
    expected = record["expected_inventory"]
    output = Path(record["output_dir"])
    trained = json.loads((output / "training.json").read_text())
    audit = json.loads((output / "step_0004/audit.json").read_text())
    exported = json.loads((output / "step_0004/final_model/export_record.json").read_text())
    _require(audit["status"] == "verified" and audit["train_mode"] == "decoder"
             and audit["source_checkpoint_id"] == record["source_checkpoint_id"]
             and audit["selected_modules"] == record["selected_targets"]
             and audit["frozen_tensors_compared"] == expected["frozen_tensors"]
             and audit["frozen_code_bank_count"] == audit["code_bank_count"] == expected["all_code_banks"]
             and audit["selected_decoder_tensor_count"] == expected["selected_decoder_tensors"]
             and audit["selected_code_bank_count"] == audit["changed_code_bits"] == 0
             and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == expected["serialized_tensors"]
             and audit["selected_decoder_fp32_changed_elements"] > 0 and audit["selected_decoder_matches_fp32_cast_exactly"]
             and audit["selected_codes_match_frozen_source_exactly"] and audit["selected_exact_sidecar_sha256"] is None
             and audit["fresh_strict_reload_state_exact"] and audit["rank_zero"], "Independent decoder-only audit failed.")
    _require(exported["source_global_step"] == 4 and exported["source_max_steps"] == 2000
             and exported["restored_decoder_tensors"] == expected["selected_decoder_tensors"]
             and exported["optimizer_steps_executed"] == 0
             and exported["sparse_bit_restore"] is None, "Export must restore decoder step4 without bit state or training.")
    native = _load_state(Path(trained["training_result"]["saved_model_dir"]))
    independent = _load_state(output / "step_0004/final_model")
    compared = _assert_state_equal(native, independent, "Native decoder final versus independently restored step4")
    _require(compared == expected["serialized_tensors"], "Native/fresh deployment tensor count differs from source.")
    del native, independent
    gc.collect()
    report = {"status": "verified", "source_checkpoint_id": record["source_checkpoint_id"],
        "expected_inventory": expected,
        "checkpoint_id": audit["checkpoint_id"], "native_vs_step4_tensors_compared": compared,
        "training_report_sha256": _sha256(output / "training.json"), "audit_sha256": _sha256(output / "step_0004/audit.json"),
        "export_record_sha256": _sha256(output / "step_0004/final_model/export_record.json"),
        "changed_decoder_fp32_elements": audit["selected_decoder_fp32_changed_elements"],
        "changed_decoder_bf16_elements": audit["changed_decoder_elements"], "changed_code_bits": 0,
        "bit_budget": audit["bit_budget"], "export_peak_allocated_bytes": exported["peak_cuda_allocated_bytes"],
        "export_peak_reserved_bytes": exported["peak_cuda_reserved_bytes"], "diagnostic_not_quality": True}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("train", "verify"))
    args = parser.parse_args()
    {"train": train, "verify": verify}[args.phase](json.loads(args.launch_record.read_text()))


if __name__ == "__main__":
    main()
