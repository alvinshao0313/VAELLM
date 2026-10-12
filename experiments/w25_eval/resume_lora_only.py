"""Continue a copied native LoRA checkpoint without restarting its schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import _build_e2e_runtime_parameter_payload, run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from e2e_common.lazy_datasets import default_dataloader_num_workers
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.lora_state import inspect_step, validate_configuration
from experiments.w25_eval.lora_step_callback import LoraStepExportCallback
from experiments.w25_eval.resume_head_only import ResumeBoundaryCallback


def validate_resume(cfg, hf, args, end_step, *, diagnostic=False):
    _require(cfg.resume_from_checkpoint is not None and cfg.stop_after_step is None,
             "Use a native resume checkpoint and the private absolute end step.")
    checkpoint = Path(cfg.resume_from_checkpoint).resolve()
    meta, state, snapshot, source, base, architecture = inspect_step(checkpoint)
    run_dir = checkpoint.parent.parent
    _require(Path(cfg.student_checkpoint_dir).resolve() == source,
             "Keep the original round base, not a BF16 exported endpoint.")
    _require(run_dir.parent == Path(cfg.run_root_dir).resolve()
             and not (run_dir / "final_model").exists(),
             "Use an unused continuation segment directly below its declared root.")
    _require(run_dir != Path(snapshot["resolved_runtime"]["run_output_dir"]).resolve(),
             "Copy the native checkpoint and original snapshot into a new segment; preserve the failed run.")
    _require(cfg.train_mode == "lora" and tuple(cfg.opt.distill_fp32_components) == ("lora",)
             and cfg.aux.norm_train_mode == cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == "none",
             "Only FP32 ordinary LoRA may train.")
    _require(cfg.runtime.parallel_mode == "dp" and cfg.runtime.offload_mode == "none"
             and not cfg.runtime.distill_hif4_act,
             "Preserve native single-process DP without student offload or HiFloat4.")
    _require(type(end_step) is int and state["global_step"] < end_step <= 2000
             and (diagnostic or end_step % 400 == 0), "Invalid future absolute boundary.")
    _require(args.max_steps == 2000 and args.save_steps == 400 and args.save_total_limit == 1
             and args.get_warmup_steps(2000) == 20,
             "Preserve the original 2000/warmup20/save400 schedule.")
    _require(not args.ignore_data_skip and not args.save_only_model
             and not args.restore_callback_states_from_checkpoint,
             "Restore data, optimizer and RNG without restoring old callback stop flags.")
    _require(not cfg.runtime.evaluation.eval_after_save
             and not str(cfg.runtime.evaluation.eval_tasks or "").strip(),
             "Use only independent saved-step evaluation.")
    current = _build_e2e_runtime_parameter_payload(
        cfg=cfg, hf_args=hf, training_args=args, run_output_dir=str(run_dir))
    validate_configuration(meta, state, current, architecture)
    _require(current["hf_args"] == snapshot["hf_args"]
             and args.world_size == meta["immutable_resume_contract"]["distributed"]["world_size"],
             "HF loading options or world size changed.")
    old = snapshot["training_args"]
    for key in ("seed", "data_seed", "full_determinism", "dataloader_drop_last",
                "dataloader_persistent_workers", "dataloader_prefetch_factor",
                "per_device_train_batch_size", "gradient_accumulation_steps", "max_grad_norm"):
        _require(getattr(args, key) == old[key], f"Resume changed training argument {key}.")
    workers = int(args.dataloader_num_workers or default_dataloader_num_workers())
    _require(workers == meta["immutable_resume_contract"]["dataloader"]["num_workers"],
             "Resolved worker count differs from the saved runtime.")
    teacher = str(cfg.teacher_model_path or base["base_model_path"])
    _require(teacher == meta["immutable_resume_contract"]["teacher_identity"]["model_path"],
             "Teacher path differs from the saved trajectory.")
    for name in ("checkpoint_meta.json", "trainer_state.json", "training_model_state.pt",
                 "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        _require((checkpoint / name).is_file(), f"Exact resume requires {name}.")
    return meta, state, snapshot, source, base, architecture


def run_training(argv=None, *, extra_callbacks=(), diagnostic=False):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--end_step", required=True, type=int)
    parser.add_argument("--export_cuda_visible_devices", required=True)
    private, remaining = parser.parse_known_args(argv)
    cfg, hf, args = parse_args(remaining)
    _meta, state, _snapshot, source, _base, _architecture = validate_resume(
        cfg, hf, args, private.end_step, diagnostic=diagnostic)
    callbacks = [ResumeBoundaryCallback(state["global_step"], private.end_step)]
    if not diagnostic:
        callbacks.append(LoraStepExportCallback(
            source_dir=source, run_root_dir=cfg.run_root_dir,
            export_cuda_visible_devices=private.export_cuda_visible_devices))
    callbacks.extend(extra_callbacks)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    try:
        result = run(cfg, hf, args, extra_callbacks=tuple(callbacks))
        _require(result["global_step"] == private.end_step, "Native resume stopped at the wrong step.")
        return result
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_training()
