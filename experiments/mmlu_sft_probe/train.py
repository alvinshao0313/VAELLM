"""Isolated MMLU SFT data registration over the existing native decoder path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.mmlu_sft_probe.data import registered_config, registered_preset, validate_saved_registration
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe


def parse_configuration(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mmlu_sft_manifest", required=True)
    local, remaining = parser.parse_known_args(argv)
    cfg, hf, training = parse_args(remaining)
    cfg.data = registered_config(cfg.data, local.mmlu_sft_manifest)
    if (cfg.train_mode != "decoder" or tuple(cfg.opt.distill_fp32_components) != ("decoder",)
            or cfg.aux.norm_train_mode != "none" or cfg.aux.lm_head_train_mode != "none"
            or cfg.aux.residual_lora_mode != "none"):
        raise ValueError("This probe only trains FP32 decoders with frozen codes/norm/head and no LoRA.")
    if (cfg.loss.loss_type != "kd_top_mass" or cfg.loss.alpha != .5
            or cfg.loss.prompt_loss_weight != 0 or cfg.loss.top_k != 100 or cfg.loss.temperature != 1
            or cfg.loss.hidden_loss_weight != 0 or cfg.loss.pre_mlp_hidden_loss_weight != 0):
        raise ValueError("Keep the reference D400 CE50/response-only loss.")
    if (cfg.runtime.parallel_mode != "dp" or cfg.runtime.offload_mode != "none"
            or cfg.runtime.distill_hif4_act or cfg.runtime.evaluation.eval_after_save
            or str(cfg.runtime.evaluation.eval_tasks or "").strip()
            or cfg.resume_from_checkpoint is not None or cfg.stop_after_step is not None):
        raise ValueError("Require fresh native DP and sequential export after training exits.")
    return cfg, hf, training


def run_training(argv=None):
    cfg, hf, training = parse_configuration(argv)
    meta = json.loads((Path(cfg.student_checkpoint_dir) / "checkpoint_meta.json").read_text())
    architecture = model_inventory(meta["base_model_path"])
    validate_recipe(meta, architecture)
    selected = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    if selected != ("model.layers.0.mlp.down_proj",):
        raise ValueError("This isolated comparison selects only L0 down.")
    configure_e2e_determinism(bool(training.full_determinism))
    set_e2e_seed(int(training.seed))
    try:
        with registered_preset(cfg.data):
            result = run(cfg, hf, training, extra_callbacks=(Budget400Callback(),))
        if result["global_step"] != 400:
            raise RuntimeError("Native probe did not finish at optimizer step400.")
        validate_saved_registration(result["run_output_dir"], cfg.data)
        return result
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_training()
