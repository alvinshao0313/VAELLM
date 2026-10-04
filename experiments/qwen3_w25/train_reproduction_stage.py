"""Run a reproduction stage without changing its original 2000-step schedule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import TrainerCallback

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.assemble_categories import CATEGORIES
from train_utils.checkpoint_v6 import _atomic_write_json


def validate_schedule(training_args, finish_step):
    if type(finish_step) is not int or not 0 < finish_step <= 2000 or finish_step % 400:
        raise ValueError("finish_step must be a positive multiple of 400, at most 2000.")
    save_strategy = getattr(training_args.save_strategy, "value", training_args.save_strategy)
    scheduler = getattr(training_args.lr_scheduler_type, "value", training_args.lr_scheduler_type)
    if (training_args.max_steps != 2000 or scheduler != "cosine"
            or training_args.get_warmup_steps(2000) != 20):
        raise ValueError("Keep the original 2000-step cosine schedule and 20 warmup steps.")
    if save_strategy != "steps" or training_args.save_steps != 400 or training_args.save_total_limit != 1:
        raise ValueError("Keep save_strategy=steps, save_steps=400 and save_total_limit=1.")


def validate_configuration(cfg, training_args, finish_step):
    validate_schedule(training_args, finish_step)
    layers, modules = cfg.target_layers, cfg.target_modules
    if cfg.train_mode == "decoder":
        all_layers = layers == "all" or tuple(layers) == tuple(range(36))
        all_modules = modules == "all" or (len(modules) == len(CATEGORIES) and set(modules) == set(CATEGORIES))
        if not all_layers or not all_modules:
            raise ValueError("Decoder-only reproduction requires all 36 layers and all seven projections.")
    elif cfg.train_mode == "decoder_sparse_bit":
        if (tuple(layers) != (0,) or tuple(modules) != ("down_proj",)
                or cfg.bit_active_ratio != 1 or cfg.bit_proxy_coordinates != "decoder_sensitivity"):
            raise ValueError("Joint reproduction requires only L0 down, all bits and sensitivity coordinates.")
    else:
        raise ValueError("Reproduction supports decoder or decoder_sparse_bit only.")
    if (cfg.aux.norm_train_mode != "none" or cfg.aux.lm_head_train_mode != "none"
            or cfg.aux.residual_lora_mode != "none"
            or tuple(cfg.opt.distill_fp32_components) != ("decoder",)):
        raise ValueError("Use FP32 decoder parameters, frozen norm/head and no residual LoRA.")
    if not training_args.bf16 or training_args.fp16:
        raise ValueError("Reproduction requires BF16 compute and export.")
    if (cfg.runtime.parallel_mode != "dp" or cfg.runtime.offload_mode != "none"
            or cfg.runtime.distill_hif4_act):
        raise ValueError("Use single-process DP without student offload or HiFloat4.")
    eval_strategy = getattr(training_args.eval_strategy, "value", training_args.eval_strategy)
    if (cfg.runtime.evaluation.eval_after_save or str(cfg.runtime.evaluation.eval_tasks or "").strip()
            or eval_strategy != "no"):
        raise ValueError("Evaluation belongs to the external pipeline; disable in-training evaluation.")
    if cfg.resume_from_checkpoint is not None or cfg.stop_after_step is not None:
        raise ValueError("Use a fresh source without resume_from_checkpoint or stop_after_step.")


class FinishStepCallback(TrainerCallback):
    """Stop after an actual optimizer update, then let Trainer save and finalize."""

    def __init__(self, finish_step):
        self.finish_step = finish_step
        self.saved_step = None
        self.global_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        validate_schedule(args, self.finish_step)
        if state.global_step != 0 or state.max_steps != 2000:
            raise ValueError("Reproduction starts at step zero with an unchanged 2000-step budget.")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.finish_step:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.saved_step = int(state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.global_step = int(state.global_step)
        if self.global_step != self.finish_step or self.saved_step != self.finish_step:
            raise RuntimeError("Training must finish at the requested, actually saved optimizer step.")
        return control


def write_stage_result(result, callback, run_root_dir):
    if (result.get("status") == "paused" or result.get("global_step") != callback.finish_step
            or callback.global_step != callback.finish_step or callback.saved_step != callback.finish_step
            or not result.get("saved_model_dir")):
        raise RuntimeError("Training did not finalize successfully at the requested step.")
    checkpoint = Path(result["run_output_dir"]).resolve() / "trainer_state" / f"checkpoint-{callback.finish_step}"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state["global_step"] != callback.finish_step or state["max_steps"] != 2000:
        raise RuntimeError("Saved Trainer state disagrees with the requested endpoint or original schedule.")
    payload = {"status": "completed", "finish_step": callback.finish_step,
        "global_step": state["global_step"], "max_steps": state["max_steps"],
        "run_output_dir": str(Path(result["run_output_dir"]).resolve()),
        "training_checkpoint_dir": str(checkpoint), "saved_model_dir": str(Path(result["saved_model_dir"]).resolve()),
        "runtime_result": result}
    _atomic_write_json(payload, str(Path(run_root_dir) / "stage_result.json"))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--finish_step", required=True, type=int)
    stage_args, remaining = parser.parse_known_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Reproduction stages require a single process and a single visible CUDA GPU.")
    cfg, hf_args, training_args = parse_args(remaining)
    validate_configuration(cfg, training_args, stage_args.finish_step)
    result_path = Path(cfg.run_root_dir) / "stage_result.json"
    if result_path.exists():
        raise FileExistsError(f"A completed stage already exists: {result_path}")
    if training_args.use_cpu or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU for this reproduction stage.")
    callback = FinishStepCallback(stage_args.finish_step)
    configure_e2e_determinism(bool(training_args.full_determinism))
    set_e2e_seed(int(training_args.seed))
    result = run(cfg, hf_args, training_args, extra_callbacks=(callback,))
    return write_stage_result(result, callback, cfg.run_root_dir)


if __name__ == "__main__":
    main()
