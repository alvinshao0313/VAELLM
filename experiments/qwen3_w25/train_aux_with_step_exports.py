"""Canonical decoder plus fusible linear-head training with isolated step evaluation."""

from __future__ import annotations

import argparse

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback


def run_training(argv=None, *, diagnostic_eval_limit=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--export_cuda_visible_devices", required=True)
    export_args, remaining = parser.parse_known_args(argv)
    cfg, hf_args, training_args = parse_args(remaining)
    if (cfg.train_mode != "decoder" or cfg.aux.norm_train_mode != "none"
            or cfg.aux.lm_head_train_mode != "linear" or cfg.aux.residual_lora_mode != "none"
            or tuple(cfg.opt.distill_fp32_components) != ("decoder", "lm_head")):
        raise ValueError("This entry requires decoder + FP32 linear head, frozen norms/codes, and no LoRA.")
    if cfg.runtime.parallel_mode != "dp" or cfg.runtime.offload_mode != "none" or cfg.runtime.distill_hif4_act:
        raise ValueError("This entry uses DP without student offload or HiFloat4.")
    if cfg.runtime.evaluation.eval_after_save or str(cfg.runtime.evaluation.eval_tasks or "").strip():
        raise ValueError("Use only the isolated callback: eval_after_save=false and empty eval_tasks.")
    callback = AuxStepExportCallback(source_dir=cfg.student_checkpoint_dir, run_root_dir=cfg.run_root_dir,
                                    export_cuda_visible_devices=export_args.export_cuda_visible_devices,
                                    diagnostic_eval_limit=diagnostic_eval_limit)
    configure_e2e_determinism(bool(training_args.full_determinism))
    set_e2e_seed(int(training_args.seed))
    try:
        return run(cfg, hf_args, training_args, extra_callbacks=(callback,))
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_training()
