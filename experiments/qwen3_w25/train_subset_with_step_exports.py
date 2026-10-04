"""Run native down-subset decoder/SparseBit training with isolated periodic export."""

from __future__ import annotations

import argparse

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.subset_step_export_callback import SubsetStepExportCallback
from experiments.qwen3_w25.e2e_subset_export import selected_names


def run_training(argv=None, *, diagnostic_eval_limit=None, diagnostic_stop_step=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--export_cuda_visible_devices", required=True)
    export_args, remaining = parser.parse_known_args(argv)
    cfg, hf_args, training_args = parse_args(remaining)
    names = selected_names(cfg.target_layers, cfg.target_modules)
    if (cfg.train_mode != "decoder_sparse_bit" or cfg.aux.norm_train_mode != "none"
            or cfg.aux.lm_head_train_mode != "none" or cfg.aux.residual_lora_mode != "none"
            or tuple(cfg.opt.distill_fp32_components) != ("decoder",)
            or cfg.bit_active_ratio != 1 or cfg.bit_proxy_coordinates != "decoder_sensitivity"):
        raise ValueError("Require selected FP32 decoders + all sensitivity-coordinate bits, frozen norm/head, no LoRA.")
    if cfg.runtime.parallel_mode != "dp" or cfg.runtime.offload_mode != "none" or cfg.runtime.distill_hif4_act:
        raise ValueError("This subset recipe uses native single-process DP without offload or HiFloat4.")
    if cfg.runtime.evaluation.eval_after_save or str(cfg.runtime.evaluation.eval_tasks or "").strip():
        raise ValueError("Use only isolated evaluation: eval_after_save=false and empty eval_tasks.")
    if cfg.resume_from_checkpoint is not None or cfg.stop_after_step is not None:
        raise ValueError("Use a fresh source; private diagnostic stopping does not create a resume workflow.")
    callback = SubsetStepExportCallback(source_dir=cfg.student_checkpoint_dir, run_root_dir=cfg.run_root_dir,
        selected_targets=names, export_cuda_visible_devices=export_args.export_cuda_visible_devices,
        diagnostic_eval_limit=diagnostic_eval_limit, diagnostic_stop_step=diagnostic_stop_step)
    configure_e2e_determinism(bool(training_args.full_determinism))
    set_e2e_seed(int(training_args.seed))
    try:
        result = run(cfg, hf_args, training_args, extra_callbacks=(callback,))
        if diagnostic_stop_step is not None:
            result["subset_diagnostic"] = callback.diagnostic
        return result
    finally:
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_training()
