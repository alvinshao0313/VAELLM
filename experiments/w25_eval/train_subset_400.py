"""Run a model-adaptive projection subset for 400 steps of the native 2000-step schedule."""

from __future__ import annotations

import torch
from pathlib import Path

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.e2e_subset_export import _read_json
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe


def run_training(argv=None):
    cfg, hf_args, training_args = parse_args(argv)
    source_meta = _read_json(Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json")
    architecture = model_inventory(source_meta["base_model_path"])
    validate_recipe(source_meta, architecture)
    selected_names(cfg.target_layers, cfg.target_modules, architecture)
    if (cfg.train_mode not in ("decoder", "decoder_sparse_bit") or cfg.aux.norm_train_mode != "none"
            or cfg.aux.lm_head_train_mode != "none" or cfg.aux.residual_lora_mode != "none"
            or tuple(cfg.opt.distill_fp32_components) != ("decoder",)):
        raise ValueError("Require selected FP32 decoders, frozen norm/head, no LoRA.")
    if cfg.train_mode == "decoder_sparse_bit" and (
            cfg.bit_active_ratio != 1 or cfg.bit_proxy_coordinates != "decoder_sensitivity"):
        raise ValueError("Joint training requires all sensitivity-coordinate bits.")
    if cfg.runtime.parallel_mode != "dp" or cfg.runtime.offload_mode != "none" or cfg.runtime.distill_hif4_act:
        raise ValueError("This comparison uses native single-process DP without offload or HiFloat4.")
    if cfg.runtime.evaluation.eval_after_save or str(cfg.runtime.evaluation.eval_tasks or "").strip():
        raise ValueError("Use only isolated evaluation: eval_after_save=false and empty eval_tasks.")
    if cfg.resume_from_checkpoint is not None or cfg.stop_after_step is not None:
        raise ValueError("Use a fresh source; this bounded comparison does not implement resume.")
    callback = Budget400Callback()
    configure_e2e_determinism(bool(training_args.full_determinism))
    set_e2e_seed(int(training_args.seed))
    try:
        result = run(cfg, hf_args, training_args, extra_callbacks=(callback,))
        if result["global_step"] != 400:
            raise RuntimeError("Bounded subset run did not finish at exactly 400 optimizer steps.")
        return result
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_training()
