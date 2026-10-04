"""Resume the native head-only trajectory to one absolute optimizer-step boundary."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import TrainerCallback

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import _build_e2e_runtime_parameter_payload, run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from e2e_common.lazy_datasets import default_dataloader_num_workers
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.head_state import inspect_step, validate_configuration


def validate_resume(cfg, hf, args, end_step, *, diagnostic=False):
    _require(cfg.resume_from_checkpoint is not None and cfg.stop_after_step is None,
             'Use a native resume checkpoint and the private absolute end step.')
    checkpoint = Path(cfg.resume_from_checkpoint).resolve()
    meta, state, snapshot, source, base, architecture = inspect_step(checkpoint)
    _require(Path(cfg.student_checkpoint_dir).resolve() == source,
             'The student must remain the original round base, not an exported endpoint.')
    _require(not (checkpoint.parent.parent / 'final_model').exists(),
             'Use a new segment directory; native finalization never overwrites an old model.')
    _require(checkpoint.parent.parent.parent == Path(cfg.run_root_dir).resolve(),
             'The segment must be directly below the declared continuation root.')
    _require(type(end_step) is int and state['global_step'] < end_step <= 2000
             and (diagnostic or end_step % 400 == 0), 'Invalid future absolute boundary.')
    _require(args.max_steps == 2000 and args.save_steps == 400 and args.save_total_limit == 1
             and args.get_warmup_steps(2000) == 20, 'Preserve the original 2000/warmup20/save400 schedule.')
    _require(not args.ignore_data_skip and not args.save_only_model
             and not args.restore_callback_states_from_checkpoint,
             'Restore data/optimizer/RNG, without reviving the previous stop flag.')
    _require(not cfg.runtime.evaluation.eval_after_save
             and not str(cfg.runtime.evaluation.eval_tasks or '').strip(),
             'Keep evaluation outside the native training process.')
    current = _build_e2e_runtime_parameter_payload(
        cfg=cfg, hf_args=hf, training_args=args, run_output_dir=str(checkpoint.parent.parent))
    validate_configuration(meta, state, current, architecture)
    _require(current['hf_args'] == snapshot['hf_args']
             and args.world_size == meta['immutable_resume_contract']['distributed']['world_size'],
             'HF loading options or world size changed.')
    old_training = snapshot['training_args']
    # These HF-only controls are not all represented by the immutable math contract.
    for key in ('seed', 'data_seed', 'full_determinism', 'dataloader_drop_last',
                'dataloader_persistent_workers', 'dataloader_prefetch_factor',
                'per_device_train_batch_size', 'gradient_accumulation_steps', 'max_grad_norm'):
        _require(getattr(args, key) == old_training[key], f'Resume changed training argument {key}.')
    workers = int(args.dataloader_num_workers or default_dataloader_num_workers())
    _require(workers == meta['immutable_resume_contract']['dataloader']['num_workers'],
             'Resolved worker count differs from the actual saved runtime.')
    teacher = str(cfg.teacher_model_path or base['base_model_path'])
    _require(teacher == meta['immutable_resume_contract']['teacher_identity']['model_path'],
             'Teacher path differs from the saved trajectory.')
    for name in ('training_model_state.pt', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        _require((checkpoint / name).is_file(), f'Exact resume requires {name}.')
    return meta, state, snapshot, source, base, architecture


class ResumeBoundaryCallback(TrainerCallback):
    def __init__(self, start_step, end_step):
        self.start_step = int(start_step)
        self.end_step = int(end_step)

    def on_train_begin(self, args, state, control, **kwargs):
        _require(state.global_step == self.start_step and args.max_steps == 2000
                 and args.get_warmup_steps(2000) == 20,
                 'The restored step or original scheduler horizon changed.')
        _require(not control.should_training_stop, 'The old callback stop flag was restored.')
        return control

    def on_step_end(self, args, state, control, **kwargs):
        _require(state.global_step <= self.end_step, 'Training exceeded the requested segment.')
        if state.global_step == self.end_step:
            control.should_save = True
            control.should_training_stop = True
        return control


def run_training(argv=None, *, extra_callbacks=(), diagnostic=False):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--end_step', required=True, type=int)
    private, remaining = parser.parse_known_args(argv)
    cfg, hf, args = parse_args(remaining)
    _meta, state, _snapshot, _source, _base, _architecture = validate_resume(
        cfg, hf, args, private.end_step, diagnostic=diagnostic)
    callback = ResumeBoundaryCallback(state['global_step'], private.end_step)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    try:
        result = run(cfg, hf, args, extra_callbacks=(callback, *extra_callbacks))
        _require(result['global_step'] == private.end_step, 'Native resume stopped at the wrong step.')
        return result
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    run_training()
