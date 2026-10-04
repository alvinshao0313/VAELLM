"""Validate the migrated R1 decoder with four real steps and two isolated exports.

Invoke with ``--launch_record PATH``. The prepared JSON supplies training_argv
(without Python/module, including --export_cuda_visible_devices), source_checkpoint,
source_checkpoint_id, source_checkpoint_meta_sha256, source_weight_stat
({size, mtime_ns, inode}), source_sha256 and input_evidence_sha256 (path -> SHA256).
The run_root_dir must be the launch record's directory. The caller activates bitvae,
sets the recorded environment and assigns idle, disjoint training/export GPUs.
Diagnostic auto batch sizes are recorded, not substituted for the separately run
full baseline's batch-64/32 evidence. This entry never launches formal training.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.assemble_categories import CATEGORIES, _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.audit_e2e_decoder import audit_e2e_decoder, persistent_inventory
from experiments.qwen3_w25.step_export_callback import SEVEN_TASKS, StepExportCallback
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.checkpoint_v6 import load_v6_training_model_state
from train_utils.v6_model_loader import load_v6_model_checkpoint


REPO = Path(__file__).resolve().parents[2]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_inputs(record):
    for field in ("source_sha256", "input_evidence_sha256"):
        for name, digest in record[field].items():
            path = Path(name)
            path = path if path.is_absolute() else REPO / path
            _require(_sha(path) == digest, f"Pinned {field} changed: {path}.")
    source = Path(record["source_checkpoint"]).resolve()
    _require(_sha(source / "checkpoint_meta.json") == record["source_checkpoint_meta_sha256"],
             "Migrated source metadata changed.")
    current, expected = (source / "pytorch_model.bin").stat(), record["source_weight_stat"]
    _require((current.st_size, current.st_mtime_ns, current.st_ino)
             == (expected["size"], expected["mtime_ns"], expected["inode"]), "Migrated source weight identity changed.")
    return source


def _configuration(record, output):
    for name, value in record.get("environment", {}).items():
        _require(os.environ.get(name) == str(value), f"Launch environment differs: {name}.")
    training_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    _require(len(training_devices) == 1 and training_devices[0].strip() not in ("", "-1"),
             "Assign exactly one explicit training GPU.")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--export_cuda_visible_devices", required=True)
    export, remaining = parser.parse_known_args(record["training_argv"])
    cfg, hf_args, args = parse_args(remaining)
    _require(Path(cfg.student_checkpoint_dir).resolve() == Path(record["source_checkpoint"]).resolve()
             and Path(cfg.run_root_dir).resolve() == output, "Training source/output differs from the launch record.")
    _require(cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, "Validation must start fresh.")
    _require(cfg.train_mode == "decoder" and tuple(cfg.opt.distill_fp32_components) == ("decoder",)
             and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none",
             "Only FP32 decoders may train; norm, head and codes must remain frozen, without LoRA.")
    _require(tuple(cfg.target_layers) == tuple(range(36))
             and (cfg.target_modules == "all" or set(cfg.target_modules) == set(CATEGORIES)),
             "All 252 compressed projections must train.")
    _require(cfg.opt.steps == args.max_steps == 4 and args.save_steps == 2
             and getattr(args.save_strategy, "value", args.save_strategy) == "steps" and args.save_total_limit == 1,
             "Use four optimizer steps, save every two steps, and keep the latest trainer checkpoint.")
    _require(cfg.opt.batch_size == args.per_device_train_batch_size == 4
             and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
             and cfg.data.model_max_length == 1024, "Preserve B4, accumulation 1 and maximum length 1024.")
    _require(cfg.opt.decoder_lr == cfg.opt.learning_rate == 1e-5 and cfg.opt.warmup_ratio == 0.01
             and cfg.opt.lr_scheduler_type == "cosine" and cfg.opt.weight_decay == 0
             and cfg.opt.gradient_checkpointing and cfg.opt.gradient_checkpointing_kwargs == {"use_reentrant": False},
             "Use the matched decoder optimization and activation-checkpointing configuration.")
    _require(args.bf16 and not args.fp16 and cfg.data.seed == cfg.data.data_seed == args.seed == args.data_seed == 0,
             "Use BF16 compute/export and both training/data seeds zero.")
    _require(cfg.data.dataset_task == "sft" and cfg.data.dynamic_padding and cfg.data.group_by_length
             and tuple(cfg.data.dataset_mix_sources) == ("edgerazor_ii_7m", "edgerazor_ii_gen", "edgerazor_tulu",
                                                       "edgerazor_am", "vaellm_eval_task")
             and tuple(cfg.data.dataset_mix_weights) == (0.341, 0.067, 0.028, 0.064, 0.5),
             "Use the matched five-source task-50 data mixture.")
    _require(cfg.loss.loss_type == "kl_top_mass" and cfg.loss.top_k == 100 and cfg.loss.temperature == 1
             and cfg.loss.prompt_loss_weight == 0.3 and cfg.loss.hidden_loss_weight == 0
             and cfg.loss.pre_mlp_hidden_loss_weight == 0 and not cfg.loss.selective_student_topk,
             "Use the matched pure top-mass KL objective.")
    _require(cfg.runtime.parallel_mode == "dp" and cfg.runtime.offload_mode == "none"
             and not cfg.runtime.distill_hif4_act and cfg.runtime.vae_decoder_checkpoint,
             "Use the existing single-process DP decoder path without student offload or HiFloat4.")
    _require(not cfg.runtime.evaluation.eval_after_save and not cfg.runtime.evaluation.eval_tasks
             and cfg.runtime.evaluation.skip_ppl_eval
             and getattr(args.eval_strategy, "value", args.eval_strategy) == "no",
             "Only the isolated diagnostic callback may run evaluations.")
    callback = StepExportCallback(source_dir=cfg.student_checkpoint_dir, run_root_dir=cfg.run_root_dir,
        train_mode="decoder", export_cuda_visible_devices=export.export_cuda_visible_devices,
        diagnostic_eval_limit=64)
    return cfg, hf_args, args, callback


def _source_inventory(source, checkpoint_id):
    meta, budget = _validate_checkpoint(source)
    _require(meta["checkpoint_id"] == checkpoint_id and budget["budget_passed"], "Source identity/budget failed.")
    model, loaded, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True,
                                               expected_kind="final_model")
    _require(loaded["checkpoint_id"] == checkpoint_id, "Source identity changed during strict load.")
    inventory, banks = persistent_inventory(model)
    decoder_keys = {key for rows in inventory.values() for key in rows}
    zero_linear_keys = []
    for name in inventory:
        decoder = model.get_submodule(name)._parallel_stage_decoder
        _require(decoder.num_res_blocks == 1, f"Expected the newly migrated R1 decoder: {name}.")
        for key, parameter in decoder.blocks[0].linear1.named_parameters():
            _require(not bool(torch.count_nonzero(parameter)), f"New residual output is not zero: {name}.{key}.")
            zero_linear_keys.append(name + "._parallel_stage_decoder.blocks.0.linear1." + key)
    counts = {"projection_count": len(inventory), "serialized_tensors_compared": len(model.state_dict()),
              "decoder_tensor_count": len(decoder_keys), "code_bank_count": sum(len(rows) for rows in banks.values())}
    counts["frozen_tensors_compared"] = counts["serialized_tensors_compared"] - counts["decoder_tensor_count"]
    del model
    gc.collect()
    return counts, budget, decoder_keys, zero_linear_keys, meta


def _check_audit(audit, counts, budget, checkpoint_id):
    _require(audit["status"] == "verified" and audit["source_checkpoint_id"] == checkpoint_id
             and all(audit[key] == value for key, value in counts.items()), "Audit source or dynamic inventory mismatch.")
    _require(audit["source_reloaded_tensors_compared"] == audit["reloaded_tensors_compared"]
             == counts["serialized_tensors_compared"] and audit["fresh_strict_reload_state_exact"],
             "Complete source/saved/fresh state comparison failed.")
    _require(audit["changed_code_bits"] == 0 and audit["packed_codes_unchanged"]
             and audit["non_decoder_bytes_unchanged"] and audit["actual_tensor_storage_unchanged"]
             and audit["changed_decoder_elements"] > 0, "Decoder-only update or frozen storage contract failed.")
    for key in ("user_linear_bits", "packed_linear_bits", "user_linear_bpw", "code_padding_bits",
                "original_linear_weights", "full_model_tensor_bytes", "outside_linear_bits", "components"):
        _require(audit["bit_budget"][key] == budget[key], f"Deployment budget changed: {key}.")


def _check_boundary(folder, step, counts, budget, source_id):
    boundary = json.loads((folder / "diagnostic.json").read_text())
    audit = json.loads((folder / "audit.json").read_text())
    _require(boundary["status"] == "diagnostic_complete" and boundary["global_step"] == step
             and boundary["max_steps"] == 4 and boundary["diagnostic_eval_limit"] == 64,
             f"Incomplete diagnostic boundary {step}.")
    _require(set(boundary["stages"]) == {"export", "audit", "mmlu", "seven"}, "Unexpected diagnostic stages.")
    for name, stage in boundary["stages"].items():
        _require(stage["exit_code"] == 0 and (folder / f"{name}.exit").read_text().strip() == "0",
                 f"Diagnostic stage {step}/{name} failed.")
    _check_audit(audit, counts, budget, source_id)
    exported = json.loads((folder / "final_model/export_record.json").read_text())
    _require(exported["checkpoint_id"] == audit["checkpoint_id"] and exported["source_global_step"] == step
             and exported["source_max_steps"] == 4 and exported["restored_decoder_tensors"] == counts["decoder_tensor_count"]
             and exported["optimizer_steps_executed"] == 0 and not exported["trainer_or_main_optimizer_constructed"]
             and not exported["teacher_or_training_data_loaded"], "Export did not independently restore the selected decoder state.")
    shards = {}
    for name, tasks, leaves, formal_batch in (("mmlu", {"mmlu"}, 57, 64),
                                              ("seven", set(SEVEN_TASKS.split(",")), 7, 32)):
        path = folder / f"{name}.json"
        raw = json.loads(path.read_text())
        lm = raw["evals"]["lm_eval"]
        _require(raw["checkpoint_id"] == audit["checkpoint_id"] and raw["checkpoint_loader"] == "v6"
                 and raw["strict_load"] and raw["adapter_dir"] is None and not raw["eval_hif4_act"]
                 and raw["tokenizer_is_fast"], "Diagnostic inference loaded a different model/protocol.")
        _require(set(lm["tasks"]) == tasks and lm["num_fewshot"] == 0 and lm["limit"] == 64
                 and len(lm["n_samples"]) == leaves
                 and all(row["effective"] == 64 for row in lm["n_samples"].values()),
                 f"Incomplete limited inference for {name}.")
        batches = [int(value) for value in re.findall(r"Determined largest batch size:\s*(\d+)",
                    (folder / f"{name}.log").read_text(errors="replace"))]
        _require(bool(batches) and all(value > 0 for value in batches), "Missing actual auto-batch evidence.")
        shards[name] = {"raw_sha256": _sha(path), "effective_samples": leaves * 64,
                        "actual_auto_batches": batches, "formal_reference_batch": formal_batch,
                        "matches_formal_batch": all(value == formal_batch for value in batches)}
    return {"checkpoint_id": audit["checkpoint_id"], "diagnostic_sha256": _sha(folder / "diagnostic.json"),
            "audit_sha256": _sha(folder / "audit.json"), "inference": shards,
            "limited_inference_is_not_full_baseline_batch_or_quality_evidence": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", type=Path, required=True)
    launch = parser.parse_args().launch_record.resolve()
    record = json.loads(launch.read_text())
    _require(record["status"] == "prepared", "Validation must start from its prepared record.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.monotonic()
    record.update(status="running", worker_pid=os.getpid(),
        worker_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        started_at=datetime.now().astimezone().isoformat(), actual_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    launch.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    try:
        source = _check_inputs(record)
        cfg, hf_args, args, callback = _configuration(record, launch.parent)
        counts, budget, decoder_keys, zero_keys, meta = _source_inventory(source, record["source_checkpoint_id"])
        _require(Path(cfg.teacher_model_path).resolve() == Path(meta["base_model_path"]).resolve(), "Teacher/source identities differ.")
        record["source_inventory"] = counts
        record["source_bit_budget"] = {key: budget[key] for key in
            ("user_linear_bits", "user_linear_bpw", "original_linear_weights", "full_model_tensor_bytes", "outside_linear_bits")}
        configure_e2e_determinism(bool(args.full_determinism))
        set_e2e_seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()
        result = run(cfg, hf_args, args, extra_callbacks=(callback,))
        _require(result["global_step"] == 4 and callback._last_step == 4, "Training did not complete four real steps.")
        record["steps"] = {str(step): _check_boundary(launch.parent / f"step_{step:04d}", step,
            counts, budget, record["source_checkpoint_id"]) for step in (2, 4)}
        native = Path(result["saved_model_dir"])
        audit_path = launch.parent / "native_final_audit.json"
        native_audit = audit_e2e_decoder(source, native, audit_path, train_mode="decoder")
        _check_audit(native_audit, counts, budget, record["source_checkpoint_id"])
        compared = _assert_state_equal(_load_state(native), _load_state(launch.parent / "step_0004/final_model"),
                                       "Native final versus independent step-four export")
        mutable, manifest = load_v6_training_model_state(str(native.parent / "trainer_state/checkpoint-4"), map_location="cpu")
        _require(set(mutable) == decoder_keys and {row["name"] for row in manifest} == decoder_keys
                 and all(row["component_class"] == "decoder" for row in manifest)
                 and all(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()) for tensor in mutable.values()),
                 "The actual selected training payload must contain all and only finite FP32 decoders.")
        changed = sum(int(torch.count_nonzero(mutable[key])) for key in zero_keys)
        _require(changed > 0, "New zero-output residual linears received no actual FP32 update.")
        _check_inputs(record)
        record.update(status="verified", training_result=result, native_final_checkpoint_id=native_audit["checkpoint_id"],
            native_final_audit_sha256=_sha(audit_path), native_equals_step4_tensors_compared=compared,
            selected_fp32_decoder_tensors=len(mutable), new_residual_linear_updated_fp32_elements=changed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            training_resumed_after_first_export=True, protocol_is_diagnostic_not_quality=True, exit_code=0)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise
    finally:
        if torch.cuda.is_initialized():
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        record.update(completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started)
        launch.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
