"""Run the existing full-head method for the first400 steps of the native2000 schedule."""
from pathlib import Path

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.export_e2e_selected import _read_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_recipe


def validate_runtime(cfg, args):
    meta = _read_json(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path'])
    validate_recipe(meta, architecture)
    _require(set(selected_names(cfg.target_layers, cfg.target_modules, architecture))
             == set(architecture['projection_shapes']), 'Head-only metadata uses the complete projection inventory.')
    _require(cfg.train_mode == 'none' and cfg.aux.lm_head_train_mode == 'full'
             and cfg.aux.norm_train_mode == cfg.aux.residual_lora_mode == 'none'
             and tuple(cfg.opt.distill_fp32_components) == ('lm_head',),
             'Only the original full head may train; decoders/codes/norm/LoRA are frozen.')
    _require(cfg.runtime.parallel_mode == 'dp' and cfg.runtime.offload_mode == 'none'
             and not cfg.runtime.distill_hif4_act and args.world_size == 1,
             'Use single-process native DP without student offload or HiFloat4.')
    _require(not cfg.runtime.evaluation.eval_after_save and not str(cfg.runtime.evaluation.eval_tasks or '').strip(),
             'Export/evaluate only after training exits.')
    _require(cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, 'Use a fresh source, without resume.')
    _require(args.bf16 and not args.fp16, 'Use BF16 compute/export and FP32 head parameters.')
    return meta, architecture


def run_training(argv=None):
    cfg, hf, args = parse_args(argv)
    validate_runtime(cfg, args)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    try:
        result = run(cfg, hf, args, extra_callbacks=(Budget400Callback(),))
        _require(result['global_step'] == 400, 'Native head-only training did not stop at400.')
        return result
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    run_training()
