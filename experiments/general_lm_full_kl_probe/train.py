"""Four native full-vocabulary KL updates; no evaluation or concurrent export."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.general_lm_recovery.train import training_arguments as control_arguments
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.w25_eval.subset_state import persistent_inventory, selected_names, validate_recipe
from experiments.w25_eval.validate_wide_subset_training import TrainingObservations
from train_utils.runtime_snapshot import to_runtime_jsonable


def training_arguments(record):
    argv = control_arguments(record['source_checkpoint'], record['architecture']['base_model_path'],
                             record['output_dir'], record['data']['train_file'])
    argv[argv.index('--loss_type') + 1] = 'kl'
    return argv


def configuration(record):
    require(record['training_argv'] == training_arguments(record), 'Unexpected full-KL training arguments.')
    cfg, hf, args = parse_args(record['training_argv'])
    actual = to_runtime_jsonable(cfg)
    expected = deepcopy(read(record['control_normalized'])['canonical_config'])
    expected['run_root_dir'] = record['output_dir']
    expected['loss']['loss_type'] = 'kl'
    require(actual == expected, 'Only output and loss_type may differ from the actual FineWeb control.')
    architecture = record['architecture']
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(architecture['model_type'] == 'qwen3' and len(names) == 252
            and set(names) == set(architecture['projection_shapes'])
            and meta['checkpoint_id'] == record['source_checkpoint_id'], 'Require fresh complete f927 Qwen.')
    require(args.max_steps == 2000 and args.get_warmup_steps(2000) == 20 and args.save_steps == 400
            and cfg.loss.loss_type == 'kl' and cfg.loss.temperature == 1 and cfg.loss.alpha == 0
            and cfg.opt.batch_size == args.per_device_train_batch_size == 4
            and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
            and cfg.data.model_max_length == 1024 and args.bf16 and not args.fp16,
            'Keep the native2000/warm20/B4/max1024/BF16 schedule.')
    return cfg, hf, args, names, architecture


class FullKLObservations(TrainingObservations):
    """Reuse optimizer/save observations without the older save2-only boundary."""
    def __init__(self, names, architecture):
        super().__init__(names)
        self.architecture = architecture
        self.active = {}
        self.previous = None
        self.diagnostic.update(gradients=[], parameter_updates=[], losses=[])

    def on_train_begin(self, args, state, control, **kwargs):
        AuxStepExportCallback.on_train_begin(self, args, state, control, **kwargs)
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 400
                and args.get_warmup_steps(2000) == 20, 'Require the unchanged fresh native schedule.')
        model = kwargs['model']
        inventory, banks = persistent_inventory(model, self.architecture)
        expected = {key for name in self.selected_targets for key in inventory[name]}
        self.active = {key: p for key, p in model.named_parameters() if p.requires_grad}
        require(set(self.active) == expected and all(p.dtype == torch.float32 for p in self.active.values())
                and not hasattr(model, 'sparse_bit_tuning'), 'Only all FP32 decoder leaves may train.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'): optimizer = optimizer.optimizer
        require({id(p) for group in optimizer.param_groups for p in group['params']}
                == {id(p) for p in self.active.values()} and not hasattr(optimizer, 'bit_optimizer'),
                'Native optimizer contains unexpected parameters.')
        self.previous = torch.cat([p.detach().reshape(-1) for p in self.active.values()]).clone()
        self.diagnostic['trainable_inventory'] = {'decoder_tensors': len(self.active),
            'decoder_elements': self.previous.numel(), 'decoder_keys': sorted(self.active),
            'selected_projection_code_banks': sum(len(banks[name]) for name in self.selected_targets),
            'score_elements': 0, 'observation_previous_parameter_bytes': self.previous.numel() * 4}
        def inputs(_module, positional, values):
            ids = values.get('input_ids', positional[0] if positional else None)
            if ids is not None:
                mask = values.get('attention_mask')
                self.diagnostic['batches'].append({'step': self._observed_step, 'shape': list(ids.shape),
                    'padded_tokens': ids.numel(), 'attention_tokens': int(mask.sum()) if mask is not None else None,
                    'causal_tokens': int(mask[:, 1:].sum()) if mask is not None else None})
        self._input_hook = model.register_forward_pre_hook(inputs, with_kwargs=True)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        grads = [p.grad for p in self.active.values()]
        require(all(g is not None and g.dtype == torch.float32 for g in grads), 'Missing or non-FP32 decoder gradients.')
        norm = float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads])))
        require(math.isfinite(norm) and norm > 0, 'Decoder gradients must be finite and nonzero.')
        self.diagnostic['gradients'].append({'step': int(state.global_step) + 1,
            'post_clip_gradient_norm': norm, 'gradient_tensors': len(grads), 'dtype': 'torch.float32'})
        return control

    def on_step_end(self, args, state, control, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        now = torch.cat([p.detach().reshape(-1) for p in self.active.values()])
        delta = float((now - self.previous).norm())
        require(math.isfinite(delta), 'Nonfinite actual parameter update.')
        self.previous.copy_(now)
        self.diagnostic['parameter_updates'].append({'step': int(state.global_step), 'fp32_delta_l2': delta,
            'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()})
        # Logging observes the native loss; saving at the stop boundary leaves the
        # immutable 2000/save400 schedule intact and needs no second GPU callback.
        control.should_log = True
        if int(state.global_step) == 4: control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and 'distill_loss' in logs:
            value = float(logs['distill_loss'])
            require(math.isfinite(value), 'Native full KL became nonfinite.')
            self.diagnostic['losses'].append({'step': int(state.global_step), 'native_distill_loss': value})
        return control

    def close(self):
        super().close()
        self.previous = None
        self.active.clear()


def train(record):
    check_inputs(record)
    cfg, hf, args, names, architecture = configuration(record)
    callback = FullKLObservations(names, architecture)
    configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(int(args.seed))
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    report = {'status': 'failed', 'diagnostic_not_quality': True}
    try:
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        observed, expected = callback.diagnostic, record['expected_inventory']
        require(result['global_step'] == 4 and result['round_base_checkpoint_id'] == record['source_checkpoint_id'],
                'Native training did not finish exactly four updates from f927.')
        for key in ('optimizer_steps', 'gradients', 'parameter_updates', 'losses'):
            require([x['step'] for x in observed[key]] == [1, 2, 3, 4], 'Incomplete actual observations: ' + key)
        require([x['step'] for x in observed['native_saves']] == [4]
                and {x['step'] for x in observed['batches']} == {1, 2, 3, 4}
                and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps'])
                and sum(x['fp32_delta_l2'] for x in observed['parameter_updates']) > 0,
                'Native update/save/data observations differ.')
        require(observed['trainable_inventory']['decoder_tensors'] == expected['selected_decoder_tensors']
                and observed['trainable_inventory']['decoder_elements'] == expected['selected_decoder_elements'],
                'Actual trainable inventory differs from source.')
        check_inputs(record)
        report.update(status='trained', training_result=result, observations=observed,
            resource_coverage='Four actual native dynamic batches, with their shapes recorded; not a guarantee for every later batch.',
            full8_and_longbench='not_evaluated')
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}', observations=callback.diagnostic)
        raise
    finally:
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.time() - started)
        write_json(Path(record['output_dir']) / 'training.json', report)
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    train(read(parser.parse_args().launch_record))
