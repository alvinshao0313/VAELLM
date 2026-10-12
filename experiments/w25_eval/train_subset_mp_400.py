"""Run a layer-sharded joint subset; export separately after native training exits."""

from __future__ import annotations

from pathlib import Path

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.e2e_subset_export import _read_json
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe


def parse_run_arguments(argv=None):
    cfg, hf_args, training_args = parse_args(argv)
    source_meta = _read_json(Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json")
    architecture = model_inventory(source_meta["base_model_path"])
    validate_recipe(source_meta, architecture)
    selected_names(cfg.target_layers, cfg.target_modules, architecture)
    if (cfg.train_mode != "decoder_sparse_bit" or cfg.aux.norm_train_mode != "none"
            or cfg.aux.lm_head_train_mode != "none" or cfg.aux.residual_lora_mode != "none"
            or tuple(cfg.opt.distill_fp32_components) != ("decoder",)
            or cfg.bit_active_ratio != 1 or cfg.bit_proxy_coordinates != "decoder_sensitivity"):
        raise ValueError("Require selected FP32 decoders + all sensitivity-coordinate bits, frozen norm/head, no LoRA.")
    if (cfg.runtime.parallel_mode != "layer_mp" or cfg.runtime.offload_mode != "none"
            or cfg.runtime.teacher_model_offload != "none" or cfg.runtime.distill_hif4_act):
        raise ValueError("Require native single-process layer_mp without model/activation offload or HiFloat4.")
    if not training_args.bf16 or training_args.fp16:
        raise ValueError("Require BF16 compute/export with selected FP32 decoder storage.")
    if (cfg.runtime.evaluation.eval_after_save or str(cfg.runtime.evaluation.eval_tasks or "").strip()
            or getattr(training_args.eval_strategy, "value", training_args.eval_strategy) != "no"):
        raise ValueError("Use isolated evaluation after training exits; disable both native and Trainer evaluation.")
    if cfg.resume_from_checkpoint is not None or cfg.stop_after_step is not None:
        raise ValueError("Use a fresh source; this bounded comparison does not implement resume.")
    return cfg, hf_args, training_args


def run_training(argv=None):
    cfg, hf_args, training_args = parse_run_arguments(argv)
    callback = Budget400Callback()
    configure_e2e_determinism(bool(training_args.full_determinism))
    set_e2e_seed(int(training_args.seed))
    try:
        result = run(cfg, hf_args, training_args, extra_callbacks=(callback,))
        if result["global_step"] != 400:
            raise RuntimeError("Bounded layer-sharded subset did not finish at exactly 400 optimizer steps.")
        return result
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_training()
