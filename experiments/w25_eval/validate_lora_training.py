"""Private four-step selected-LoRA validation; no formal run CLI."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from compressed_e2e_fintuning.v6_runtime_state import _jsonable
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from e2e_common.full_lora import collect_logical_adapter_target_names
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.qwen3_w25.export_e2e_selected import _read_json, _sha256
from experiments.qwen3_w25.subset_step_export_callback import SubsetStepExportCallback
from experiments.w25_eval.lora_audit import load_state
from experiments.w25_eval.lora_state import lora_selection, lora_storage, project_lora_state, validate_source
from experiments.w25_eval.lora_step_callback import LoraStepExportCallback
from experiments.w25_eval.validate_wide_subset_training import TrainingObservations


def configuration(record):
    cfg, hf, args = parse_args(record["training_argv"])
    source_meta = _read_json(Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json")
    architecture = validate_source(source_meta)
    shapes, lora = lora_selection(architecture, cfg.target_layers, cfg.target_modules,
                                 rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout)
    _require(source_meta["checkpoint_id"] == record["source_checkpoint_id"], "The candidate source UUID changed.")
    _require(Path(cfg.student_checkpoint_dir).resolve() == Path(record["source_checkpoint"]).resolve()
             and cfg.train_mode == "lora", "Require fresh ordinary LoRA on the pinned source.")
    expected, storage = record["expected_inventory"], lora_storage(shapes, lora["rank"])
    _require(expected["lora_tensors"] == storage["tensors"] and expected["lora_parameters"] == storage["parameters"]
             and expected["added_BF16_bits"] == storage["storage_bits"]
             and expected["final_tensors"] == expected["original_frozen_tensors"] + storage["tensors"],
             "Prepared inventory does not match actual selected LoRA geometry.")
    _require(cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none"
             and tuple(cfg.opt.distill_fp32_components) == ("lora",)
             and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
             "Only FP32 LoRA may train; this diagnostic does not resume.")
    _require(cfg.loss.loss_type == "kl_top_mass" and cfg.loss.alpha == 0 and cfg.loss.top_k == 100
             and cfg.loss.temperature == 1 and cfg.loss.prompt_loss_weight == 0.3
             and cfg.opt.learning_rate == 3e-5, "Wrong conservative LoRA loss/LR recipe.")
    _require(cfg.runtime.parallel_mode == "dp" and cfg.runtime.offload_mode == "none"
             and not cfg.runtime.distill_hif4_act and not cfg.runtime.evaluation.eval_after_save
             and not str(cfg.runtime.evaluation.eval_tasks or "").strip(), "Use isolated single-process evaluation only.")
    _require(args.max_steps == 2000 and args.save_steps == 2 and args.save_total_limit == 1
             and args.get_warmup_steps(2000) == 20 and args.per_device_train_batch_size == 4
             and args.gradient_accumulation_steps == 1 and args.seed == args.data_seed == 0
             and cfg.data.model_max_length == 1024 and args.bf16 and not args.fp16,
             "Preserve the real schedule, B4/max1024 and BF16 compute.")
    _require(_jsonable(cfg.data) == record["canonical_data"], "The pinned five-way task50 data configuration changed.")
    return cfg, hf, args


class LoraValidationCallback(LoraStepExportCallback):
    def __init__(self, *, selected_shapes, lora_config, **kwargs):
        super().__init__(diagnostic_eval_limit=64, **kwargs)
        self.selected_shapes = selected_shapes
        self.lora_config = lora_config
        self.diagnostic_stop_step = 4
        self.diagnostic = {"batches": [], "optimizer_steps": [], "native_saves": []}
        self._input_hook = None
        self._observed_step = None

    # These existing observations inspect actual Trainer callbacks and Adam
    # moments; they do not reproduce the optimizer or alter its update rule.
    on_step_begin = SubsetStepExportCallback.on_step_begin
    on_step_end = SubsetStepExportCallback.on_step_end
    close = SubsetStepExportCallback.close
    on_train_end = SubsetStepExportCallback.on_train_end

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        _require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 2
                 and args.get_warmup_steps(2000) == 20, "Validation must start fresh with save2/warm20.")
        model = kwargs["model"]
        active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        manifest = [{"name": name, "component_class": "lora"} for name in active]
        shapes, lora = self.selected_shapes, self.lora_config
        storage = lora_storage(shapes, lora["rank"])
        projected = project_lora_state(active, manifest, shapes, rank=lora["rank"], alpha=lora["alpha"])
        del projected
        _require(collect_logical_adapter_target_names(model) == set(shapes)
                 and len(active) == storage["tensors"] and sum(v.numel() for v in active.values()) == storage["parameters"],
                 "Trainables escaped the selected adapter inventory.")
        optimizer = kwargs["optimizer"]
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        _require(not hasattr(optimizer, "bit_optimizer")
                 and {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(v) for v in active.values()},
                 "Optimizer must update exactly the LoRA factors.")
        self.diagnostic["trainable_inventory"] = {"lora_tensors": len(active), "lora_elements": storage["parameters"],
            "parameter_dtype": "torch.float32", "target_count": len(shapes), "lora_config": lora,
            "decoder_norm_head_code_trainables": 0}
        def observe(_module, positional, values):
            ids = values.get("input_ids", positional[0] if positional else None)
            if ids is not None:
                mask = values.get("attention_mask")
                self.diagnostic["batches"].append({"step": self._observed_step, "shape": list(ids.shape),
                    "attention_tokens": None if mask is None else int(mask.sum().item())})
        self._input_hook = model.register_forward_pre_hook(observe, with_kwargs=True)
        return control

    def on_save(self, args, state, control, **kwargs):
        TrainingObservations.on_save(self, args, state, control, **kwargs)
        return super().on_save(args, state, control, **kwargs)


def train(record):
    cfg, hf, args = configuration(record)
    _require(record.get("gpu") is not None and record.get("export_gpu") is not None
             and str(record["gpu"]) != str(record["export_gpu"]), "Bind two distinct GPUs.")
    architecture = validate_source(_read_json(Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json"))
    shapes, lora = lora_selection(architecture, cfg.target_layers, cfg.target_modules,
                                 rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout)
    callback = LoraValidationCallback(source_dir=cfg.student_checkpoint_dir, run_root_dir=cfg.run_root_dir,
                                      export_cuda_visible_devices=str(record["export_gpu"]), selected_shapes=shapes, lora_config=lora)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    try:
        result = run(cfg, hf, args, extra_callbacks=(callback,))
        _require(result["global_step"] == 4 and result["round_base_checkpoint_id"] == record["source_checkpoint_id"],
                 "Wrong native training endpoint.")
        obs = callback.diagnostic
        _require([x["step"] for x in obs["optimizer_steps"]] == [1, 2, 3, 4]
                 and [x["step"] for x in obs["native_saves"]] == [2, 4]
                 and {x["step"] for x in obs["batches"]} == {1, 2, 3, 4}
                 and all(x["shape"][0] == 4 and x["shape"][1] <= 1024 for x in obs["batches"])
                 and all(x["main_moment_dtypes"] == ["torch.float32"] for x in obs["optimizer_steps"]),
                 "Missing actual batches, updates, FP32 moments or save2/4 evidence.")
        report = {"status": "trained", "training_result": result, "observations": obs,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "diagnostic_not_quality": True, "resource_coverage": "Four real dynamic B4 batches; not a full-data peak bound."}
        (Path(record["output_dir"]) / "training.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    finally:
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def verify(record):
    cfg, _hf, _args = configuration(record)
    output = Path(record["output_dir"])
    trained = _read_json(output / "training.json")
    expected = record["expected_inventory"]
    architecture = validate_source(_read_json(Path(record["source_checkpoint"]) / "checkpoint_meta.json"))
    shapes, lora = lora_selection(architecture, cfg.target_layers, cfg.target_modules,
                                 rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout)
    points = {}
    for step in (2, 4):
        d = output / f"step_{step:04d}"
        callback, audit, export = (_read_json(d / p) for p in ("diagnostic.json", "audit.json", "final_model/export_record.json"))
        _require(callback["status"] == "diagnostic_complete" and callback["diagnostic_eval_limit"] == 64
                 and set(callback["stages"]) == {"export", "audit", "mmlu", "seven"}
                 and all(x["exit_code"] == 0 for x in callback["stages"].values()), "Incomplete limited callback.")
        _require(audit["status"] == "verified" and audit["source_checkpoint_id"] == record["source_checkpoint_id"]
                 and audit["frozen_tensors_compared"] == expected["original_frozen_tensors"]
                 and audit["frozen_code_banks"] == expected["frozen_code_banks"] and audit["changed_code_bits"] == 0
                 and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == expected["final_tensors"]
                 and audit["selected_lora_matches_native_bf16_cast_exactly"] and audit["lora_config"] == lora
                 and audit["selected_target_count"] == len(shapes) and audit["bit_budget"]["budget_passed"],
                 "Incomplete independent LoRA state audit.")
        _require(export["source_global_step"] == step and export["source_max_steps"] == 2000
                 and export["restored_lora_tensors"] == expected["lora_tensors"] and export["lora_config"] == lora
                 and export["optimizer_steps_executed"] == 0,
                 "Export must only restore the selected step.")
        for shard, batch, leaves, count in (("mmlu", 64, 57, 3648), ("seven", 32, 7, 448)):
            raw = _read_json(d / f"{shard}.json")
            e = raw["evals"]["lm_eval"]
            _require(raw["checkpoint_id"] == audit["checkpoint_id"] and raw["strict_load"]
                     and e["batch_size"] == batch and e["limit"] == 64 and e["num_fewshot"] == 0
                     and len(e["n_samples"]) == leaves and sum(x["effective"] for x in e["n_samples"].values()) == count,
                     "Diagnostic inference geometry or identity differs.")
        points[str(step)] = {"audit_sha256": _sha256(d / "audit.json"), "export_sha256": _sha256(d / "final_model/export_record.json"),
            "bit_budget": audit["bit_budget"], "nonzero_fp32_B_elements": audit["nonzero_fp32_B_elements"],
            "nonzero_bf16_low_rank_a_elements": audit["nonzero_bf16_low_rank_a_elements"],
            "export_peak_allocated_bytes": export["peak_cuda_allocated_bytes"], "export_peak_reserved_bytes": export["peak_cuda_reserved_bytes"]}
    native = load_state(trained["training_result"]["saved_model_dir"])
    independent = load_state(output / "step_0004/final_model")
    compared = _assert_state_equal(native, independent, "Native final versus independently restored step4")
    del native, independent
    gc.collect()
    _require(compared == expected["final_tensors"], "Native final has the wrong complete tensor inventory.")
    report = {"status": "verified", "source_checkpoint_id": record["source_checkpoint_id"], "points": points,
              "architecture": architecture, "lora_config": lora,
              "native_vs_step4_tensors_compared": compared, "training_report_sha256": _sha256(output / "training.json"),
              "diagnostic_not_quality": True}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "verify"), required=True)
    args = parser.parse_args()
    {"train": train, "verify": verify}[args.phase](_read_json(args.launch_record))


if __name__ == "__main__":
    main()
