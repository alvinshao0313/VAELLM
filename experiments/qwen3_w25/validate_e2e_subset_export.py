"""Four native optimizer steps, two subset exports and limited resource diagnostics.

The real schedule remains max_steps=2000/warmup20. A private callback stops after
step4, so these generated states are diagnostics and are never formal initials.
Launch records pin source/HF/data identities and bind two actually free GPUs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import time

import torch

from compressed_e2e_fintuning.args import parse_args
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.e2e_subset_audit import audit_e2e_subset
from experiments.qwen3_w25.e2e_subset_export import selected_names
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.qwen3_w25.train_subset_with_step_exports import run_training


def check_inputs(record):
    for field in ("source_sha256", "input_evidence_sha256"):
        for path, digest in record[field].items():
            _require(_sha256(path) == digest, f"Pinned {field} changed: {path}.")
    source = Path(record["source_checkpoint"])
    meta = json.loads((source / "checkpoint_meta.json").read_text())
    _require(_sha256(source / "checkpoint_meta.json") == record["source_checkpoint_meta_sha256"]
             and meta["checkpoint_id"] == record["source_checkpoint_id"], "Actual source identity changed.")
    for path, digest in record["hf_source"]["metadata_sha256"].items():
        _require(_sha256(path) == digest, f"HF metadata changed: {path}.")
    for item in [record["source_weight_stat"], *record["input_data_stat"], *record["hf_source"]["weight_files"]]:
        stat = Path(item["path"]).stat()
        _require((stat.st_size, stat.st_mtime_ns, stat.st_ino) == (item["size"], item["mtime_ns"], item["inode"]),
                 f"Protected source/data/HF file changed: {item['path']}.")


def configuration(record):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--export_cuda_visible_devices", required=True)
    export, rest = parser.parse_known_args(record["training_argv"])
    cfg, hf, args = parse_args(rest)
    names = selected_names(cfg.target_layers, cfg.target_modules)
    _require(list(names) == record["selected_targets"], "CLI subset differs from the prepared selection.")
    _require(cfg.student_checkpoint_dir == record["source_checkpoint"]
             and cfg.run_root_dir == record["output_dir"] and cfg.resume_from_checkpoint is None
             and cfg.stop_after_step is None, "Use the pinned fresh source and independent diagnostic output.")
    _require(cfg.train_mode == "decoder_sparse_bit" and tuple(cfg.opt.distill_fp32_components) == ("decoder",)
             and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none",
             "Only selected FP32 decoders and full bit banks may train.")
    _require(cfg.opt.steps == args.max_steps == 2000 and args.save_steps == 2 and args.save_total_limit == 1
             and args.get_warmup_steps(2000) == 20 and cfg.opt.warmup_ratio == .01,
             "Keep the full2000 schedule/warmup20 and save every2 for four-step diagnostics.")
    _require(cfg.opt.batch_size == args.per_device_train_batch_size == 4
             and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
             and cfg.data.model_max_length == 1024, "Keep real B4/max1024/accumulation1.")
    _require(cfg.opt.decoder_lr == cfg.opt.learning_rate == 1e-5 and cfg.opt.lr_scheduler_type == "cosine"
             and cfg.opt.weight_decay == 0 and cfg.opt.max_grad_norm == 1.5
             and cfg.opt.gradient_checkpointing and cfg.opt.gradient_checkpointing_kwargs == {"use_reentrant": False},
             "Subset decoder optimization differs from the approved recipe.")
    _require(cfg.bit_active_ratio == 1 and cfg.bit_optimizer == "adam" and float(cfg.bit_lr) == 4e-5
             and int(cfg.bit_round_steps) == 512 and cfg.bit_weight_decay == 0
             and cfg.bit_proxy_coordinates == "decoder_sensitivity", "Use all-code Adam4e-5/R512/sensitivity.")
    _require(args.bf16 and not args.fp16 and cfg.data.seed == cfg.data.data_seed == args.seed == args.data_seed == 0,
             "Use BF16 compute and both seeds0.")
    _require(cfg.data.dataset_task == "sft" and cfg.data.dynamic_padding and cfg.data.group_by_length
             and tuple(cfg.data.dataset_mix_sources) == ("vaellm_eval_task",)
             and tuple(cfg.data.dataset_mix_weights) == (1.0,), "Use native task100 ChatML data flow.")
    _require(cfg.loss.loss_type == "kd_top_mass" and cfg.loss.alpha == .5 and cfg.loss.prompt_loss_weight == 0
             and cfg.loss.top_k == 100 and cfg.loss.temperature == 1 and cfg.loss.hidden_loss_weight == 0
             and cfg.loss.pre_mlp_hidden_loss_weight == 0 and not cfg.loss.selective_student_topk,
             "Use existing CE50/response-only top100 mass objective.")
    _require(cfg.runtime.parallel_mode == "dp" and cfg.runtime.offload_mode == "none"
             and not cfg.runtime.distill_hif4_act and cfg.runtime.vae_decoder_checkpoint
             and not cfg.runtime.evaluation.eval_after_save and not cfg.runtime.evaluation.eval_tasks
             and cfg.runtime.evaluation.skip_ppl_eval, "Use native DP plus only isolated evaluations.")
    return cfg, hf, args, export


def _boundary(folder, step, record):
    boundary = json.loads((folder / "diagnostic.json").read_text())
    audit = json.loads((folder / "audit.json").read_text())
    n = len(record["selected_targets"])
    _require(boundary["status"] == "diagnostic_complete" and boundary["global_step"] == step
             and boundary["max_steps"] == 2000 and boundary["diagnostic_eval_limit"] == 64,
             "Incomplete independent diagnostic boundary.")
    _require(set(boundary["stages"]) == {"export", "audit", "mmlu", "seven"}, "Unexpected diagnostic stages.")
    for name, stage in boundary["stages"].items():
        _require(stage["exit_code"] == 0 and (folder / f"{name}.exit").read_text().strip() == "0",
                 f"Failed boundary {step}/{name}.")
    _require(audit["status"] == "verified" and audit["source_checkpoint_id"] == record["source_checkpoint_id"]
             and audit["selected_modules"] == record["selected_targets"]
             and audit["frozen_tensors_compared"] == 3423 - 8 * n
             and audit["selected_decoder_tensor_count"] == 6 * n and audit["selected_code_bank_count"] == 2 * n
             and audit["serialized_tensors_compared"] == audit["reloaded_tensors_compared"] == 3423
             and audit["code_bank_count"] == 504 and audit["selected_decoder_fp32_changed_elements"] > 0
             and audit["selected_decoder_matches_fp32_cast_exactly"] and audit["frozen_bytes_unchanged"]
             and audit["selected_codes_match_current_snapshot_exactly"] and audit["fresh_strict_reload_state_exact"],
             "Subset restore/freeze/full saved-state contract failed.")
    exported = json.loads((folder / "final_model/export_record.json").read_text())
    _require(exported["selected_targets"] == record["selected_targets"]
             and exported["source_global_step"] == step and exported["source_max_steps"] == 2000
             and exported["restored_decoder_tensors"] == 6 * n and exported["optimizer_steps_executed"] == 0,
             "Saved-step export did not restore the actual subset without training.")
    shards = {}
    for name, batch, leaves in (("mmlu", 64, 57), ("seven", 32, 7)):
        raw = json.loads((folder / f"{name}.json").read_text()); e = raw["evals"]["lm_eval"]
        _require(raw["checkpoint_id"] == audit["checkpoint_id"] and raw["strict_load"] and raw["tokenizer_is_fast"]
                 and not raw["eval_hif4_act"] and e["limit"] == 64 and e["num_fewshot"] == 0
                 and e["batch_size"] == batch and len(e["n_samples"]) == leaves,
                 "Limited diagnostic inference identity or fixed64/32 protocol changed.")
        shards[name] = {"sha256": _sha256(folder / f"{name}.json"), "batch_size": batch,
                        "limit": 64, "effective_samples": sum(x["effective"] for x in e["n_samples"].values())}
    return {"checkpoint_id": audit["checkpoint_id"], "diagnostic_sha256": _sha256(folder / "diagnostic.json"),
            "audit_sha256": _sha256(folder / "audit.json"), "export_sha256": _sha256(folder / "final_model/export_record.json"),
            "changed_decoder_fp32_elements": audit["selected_decoder_fp32_changed_elements"],
            "changed_decoder_bf16_elements": audit["changed_decoder_elements"], "changed_code_bits": audit["changed_code_bits"],
            "sparse_bit_restore": audit["sparse_bit_restore"], "bit_budget": audit["bit_budget"], "shards": shards,
            "export_peak_cuda_allocated_bytes": exported["peak_cuda_allocated_bytes"],
            "export_peak_cuda_reserved_bytes": exported["peak_cuda_reserved_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True, type=Path)
    cli = parser.parse_args(); record = json.loads(cli.launch_record.read_text())
    _require(record["status"] == "prepared" and not record["queued"], "Require unused prepared diagnostic record.")
    _cfg, _hf, _args, export = configuration(record)
    _require(type(record["gpu"]) is int and type(record["export_gpu"]) is int and record["gpu"] != record["export_gpu"]
             and os.environ.get("CUDA_VISIBLE_DEVICES") == str(record["gpu"])
             and export.export_cuda_visible_devices == str(record["export_gpu"]), "Bind distinct actual training/export GPUs.")
    for key, value in record["environment"].items():
        _require(os.environ.get(key) == str(value), f"Launch environment differs: {key}.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    record.update(status="running", started_at=datetime.now().astimezone().isoformat(), worker_pid=os.getpid(),
        worker_start_ticks=int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]))
    cli.launch_record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    started = time.monotonic()
    try:
        check_inputs(record); torch.cuda.reset_peak_memory_stats()
        result = run_training(record["training_argv"], diagnostic_eval_limit=64, diagnostic_stop_step=4)
        _require(result["global_step"] == 4 and result["round_base_checkpoint_id"] == record["source_checkpoint_id"],
                 "Native training did not complete exactly four steps from the pinned source.")
        output = Path(record["output_dir"])
        steps = {str(step): _boundary(output / f"step_{step:04d}", step, record) for step in (2, 4)}
        observed = result["subset_diagnostic"]
        _require(observed["trainable_inventory"]["decoder_tensors"] == record["expected_inventory"]["selected_decoder_tensors"]
                 and observed["trainable_inventory"]["score_elements"] == record["expected_inventory"]["active_score_elements"],
                 "Actual native trainable subset/full-bit inventory differs from source geometry.")
        _require([x["step"] for x in observed["optimizer_steps"]] == [1, 2, 3, 4]
                 and {x["step"] for x in observed["batches"]} == {1, 2, 3, 4}
                 and all(x["shape"][0] == 4 and x["shape"][1] <= 1024 for x in observed["batches"]),
                 "Actual B4 forward/update/resumption evidence is incomplete.")
        _require(all(x["main_moment_dtypes"] == ["torch.float32"] for x in observed["optimizer_steps"]),
                 "Decoder Adam moments lost FP32 precision.")
        native = audit_e2e_subset(record["source_checkpoint"], result["saved_model_dir"], output / "native_final_audit.json",
            training_checkpoint_dir=Path(result["run_output_dir"]) / "trainer_state/checkpoint-4")
        actual, independent = _load_state(Path(result["saved_model_dir"])), _load_state(output / "step_0004/final_model")
        compared = _assert_state_equal(actual, independent, "Native final versus independent step4 subset export")
        del actual, independent; gc.collect(); check_inputs(record)
        record.update(status="verified", training_result=result, steps=steps, native_final_checkpoint_id=native["checkpoint_id"],
            native_final_audit_sha256=_sha256(output / "native_final_audit.json"), native_vs_step4_tensors_compared=compared,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            training_resumed_after_first_export=True, diagnostic_not_quality=True,
            resource_coverage="Actual native four batches and two limited fixed64/32 inference shards; dynamic lengths and measured peak reported, not a full-data worst-case claim.")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started)
        cli.launch_record.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
