"""Native one-gate recovery with dynamic R0/R1 observation and stop128 callback."""
import argparse
import gc
import os
from pathlib import Path
import time

import torch
from transformers import TrainerCallback

from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.llama_gate_r1_probe.common import (
    PREFIX, STEPS, TARGET, check, configuration, read, require, write_json)
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.runtime_snapshot import to_runtime_jsonable


class Observations(TrainerCallback):
    def __init__(self, record, arm):
        self.record, self.arm = record, arm
        self.report = {'batches': [], 'gradients': [], 'updates': [], 'saved_steps': [], 'native_saves': []}
        self.parameters, self.before, self.hook = {}, {}, None
        self.step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        model, optimizer = kwargs['model'], kwargs['optimizer']
        gate = model.get_submodule(TARGET)
        expected = self.record['inventories'][self.arm]
        self.parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
        require(set(self.parameters) == set(expected['selected'])
                and all(p.dtype == torch.float32 for p in self.parameters.values())
                and sum(p.numel() for p in self.parameters.values()) == expected['selected_elements']
                and {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in self.parameters.values()},
                'Actual optimizer membership must equal all and only this native R0/R1 decoder.')
        require(state.global_step == 0 and state.max_steps == 2000 and args.save_steps == STEPS
                and args.get_warmup_steps(2000) == 20 and gate.trainable_decode
                and not gate.cache_decoded_weight and gate._parallel_stage_decoder.num_models == 2
                and gate._parallel_stage_decoder.num_res_blocks == (self.arm == 'r1')
                and list(gate._parallel_stage_layout) == [(0, 0), (1, 0)], 'Native packed R0/R1 runtime differs.')
        persistent = model.state_dict()
        require(len(persistent) == expected['state_tensors'], 'Training persistent inventory differs.')
        self.frozen = {k: (v.data_ptr(), v._version, tuple(v.shape), str(v.dtype))
                       for k, v in persistent.items() if k not in self.parameters}
        require(len(self.frozen) == expected['frozen_tensors'], 'Frozen inventory differs.')
        self.initial = {k: p.detach().cpu().clone() for k, p in self.parameters.items()}
        self.before = {k: p.detach().clone() for k, p in self.parameters.items()}
        def observe(_module, positional, inputs):
            if not _module.training or len(self.report['batches']) >= self.step:
                return
            expected_batch = self.record['data']['training_batches'][self.step - 1]
            observed = {k: tensor_record(inputs[k])['sha256'] for k in ('input_ids', 'attention_mask')}
            require(all(observed[k] == expected_batch['tensor_sha256'][k] for k in observed),
                    'Actual native sampler/encoding differs from the shared prepared prefix.')
            self.report['batches'].append({'step': self.step, 'shape': list(inputs['input_ids'].shape), **observed})
        self.hook = model.register_forward_pre_hook(observe, with_kwargs=True)
        self.report['actual_inventory'] = expected
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = int(state.global_step) + 1
        self.lr = [group['lr'] for group in kwargs['optimizer'].param_groups]
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        item = {'step': self.step, 'lr': self.lr, 'stages': {}}
        for stage in range(2):
            groups = {}
            for label in ('original', 'residual'):
                selected = [p for key, p in self.parameters.items() if ('blocks.0.' in key) == (label == 'residual')]
                if not selected:
                    continue
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected), 'Missing/nonfinite actual gradient.')
                values = [p.grad.detach().reshape(2, -1)[stage] for p in selected]
                groups[label] = {'l2': float(sum(v.double().square().sum() for v in values).sqrt()),
                                 'nonzero': sum(int(torch.count_nonzero(v)) for v in values)}
            require(groups['original']['nonzero'] > 0, 'Missing original-decoder gradient.')
            if self.arm == 'r1':
                require(groups['residual']['nonzero'] > 0, 'Residual branch has no real gradient.')
            item['stages'][str(stage)] = groups
        self.report['gradients'].append(item)
        if self.step in (1, 2, 20, STEPS):
            item['parameter_gradients'] = {k: {'l2': float(p.grad.detach().double().norm()),
                'nonzero': int(torch.count_nonzero(p.grad))} for k, p in self.parameters.items()}
        return control

    def on_step_end(self, args, state, control, **kwargs):
        item = {'step': int(state.global_step), 'stages': {}}
        for stage in range(2):
            counts = {label: sum(int(torch.count_nonzero(p.detach().reshape(2, -1)[stage]
                         != self.before[k].reshape(2, -1)[stage])) for k, p in self.parameters.items()
                         if ('blocks.0.' in k) == (label == 'residual')) for label in ('original', 'residual')}
            require(self.step == 1 or counts['original'] > 0, 'No original decoder update after warmup-zero first step.')
            if self.arm == 'r1':
                require(self.step == 1 or counts['residual'] > 0, 'No residual branch update after warmup-zero first step.')
            item['stages'][str(stage)] = counts
        for key, parameter in self.parameters.items():
            require(bool(torch.isfinite(parameter).all()), 'Nonfinite trainable parameter.')
            self.before[key].copy_(parameter.detach())
        self.report['updates'].append(item)
        moments = [value for entry in kwargs['optimizer'].state.values()
                   for key, value in entry.items() if key in ('exp_avg', 'exp_avg_sq')]
        require(len(moments) == 2 * len(self.parameters)
                and all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in moments),
                'Actual Adam moments must remain finite FP32 for every selected parameter.')
        if state.global_step == STEPS:
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        require(state.global_step == STEPS, 'Only the actual128 boundary may save.')
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        trainer, meta = read(checkpoint / 'trainer_state.json'), read(checkpoint / 'checkpoint_meta.json')
        require(trainer['global_step'] == STEPS and trainer['max_steps'] == 2000
                and (checkpoint / 'training_model_state.pt').is_file(), 'Actual native training save is incomplete.')
        self.report['native_saves'].append({'checkpoint': str(checkpoint), 'checkpoint_id': meta['checkpoint_id'],
            'trainer_global_step': trainer['global_step'], 'max_steps': trainer['max_steps']})
        self.report['saved_steps'].append(int(state.global_step))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        require(state.global_step == STEPS and len(self.report['batches']) == STEPS
                and self.report['saved_steps'] == [STEPS], 'Incomplete native128 training/save.')
        current = kwargs['model'].state_dict()
        frozen = {k: (v.data_ptr(), v._version, tuple(v.shape), str(v.dtype))
                  for k, v in current.items() if k not in self.parameters}
        require(frozen == self.frozen, 'A supposedly frozen tensor changed during training.')
        endpoint = {k: p.detach().cpu().clone() for k, p in self.parameters.items()}
        self.report['persistent_frozen_versions_unchanged'] = True
        self.report['changed_fp32_elements'] = {k: int(torch.count_nonzero(v != self.initial[k])) for k, v in endpoint.items()}
        self.report['bf16_changed_elements'] = {k: int(torch.count_nonzero(v.bfloat16() != self.initial[k].bfloat16())) for k, v in endpoint.items()}
        torch.save(endpoint, Path(self.record['output_dir']) / self.arm / 'selected_fp32.pt')
        self.hook.remove(); self.hook = None
        return control


def train(path, arm):
    record = read(path); check(record)
    output = Path(record['output_dir']) / arm
    require(not output.exists(), 'Do not overwrite or resume an arm.')
    output.mkdir()
    cfg, hf, args = configuration(record, arm)
    observer = Observations(record, arm)
    configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(); started = time.time()
    try:
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(observer,)))
        expected_id = record['source_checkpoint_id'] if arm == 'r0' else read(Path(record['output_dir']) / 'migration.json')['checkpoint_id']
        require(result['global_step'] == STEPS and result['round_base_checkpoint_id'] == expected_id, 'Wrong native result/source.')
        report = {'status': 'completed', 'arm': arm, 'training_result': result, 'observations': observer.report,
                  'elapsed_seconds': time.time() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        write_json(output / 'training.json', report)
    finally:
        if observer.hook is not None:
            observer.hook.remove()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        observer.parameters.clear(); observer.before.clear(); gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('r0', 'r1'))
    options = parser.parse_args(); train(options.launch_record, options.arm)
