"""Native all32-gate recovery; observe actual S4 gradients, saves and frozen state."""
import argparse
import gc
from pathlib import Path
import time

import torch
from transformers import TrainerCallback

from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.llama_gate_s4_r1_probe.common import (
    GATE_NAMES, check, configuration, read, require, write_json)
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.runtime_snapshot import to_runtime_jsonable


class Observations(TrainerCallback):
    def __init__(self, record, arm):
        self.record, self.arm = record, arm
        self.stop = record['schedule']['actual_stop_step']
        self.observe_steps = {1, 2, 3, 4, 20, self.stop}
        self.report = {'batches': [], 'gradients': [], 'updates': [], 'native_saves': []}
        self.parameters, self.before, self.hook, self.step = {}, {}, None, 0

    def on_train_begin(self, args, state, control, **kwargs):
        model, optimizer = kwargs['model'], kwargs['optimizer']
        expected = self.record['inventories'][self.arm]
        self.parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
        require(set(self.parameters) == set(expected['selected'])
                and all(p.dtype == torch.float32 for p in self.parameters.values())
                and sum(p.numel() for p in self.parameters.values()) == expected['selected_elements']
                and {id(p) for group in optimizer.param_groups for p in group['params']}
                == {id(p) for p in self.parameters.values()}, 'Actual FP32 optimizer must contain only all32 gate decoders.')
        require(state.global_step == 0 and state.max_steps == 2000 and args.get_warmup_steps(2000) == 20,
                'Wrong fresh2000 scheduler state.')
        for name in GATE_NAMES:
            gate = model.get_submodule(name)
            require(gate.trainable_decode and not gate.cache_decoded_weight
                    and gate._parallel_stage_decoder.num_models == 4
                    and gate._parallel_stage_decoder.num_res_blocks == int(self.arm == 'r1')
                    and list(gate._parallel_stage_layout) == [(i, 0) for i in range(4)],
                    'Wrong native packed four-stage runtime: ' + name)
        persistent = model.state_dict()
        require(len(persistent) == expected['state_tensors'], 'Training persistent inventory differs.')
        self.frozen = {k: (v.data_ptr(), v._version, tuple(v.shape), str(v.dtype))
                       for k, v in persistent.items() if k not in self.parameters}
        require(len(self.frozen) == expected['frozen_tensors'], 'Frozen inventory differs.')
        self.initial = {k: p.detach().cpu().clone() for k, p in self.parameters.items()}
        self.before = {k: p.detach().clone() for k, p in self.parameters.items()}
        self.groups = {name: {label: {k: p for k, p in self.parameters.items()
            if k.startswith(name + '.') and ('.blocks.0.' in k) == (label == 'residual')}
            for label in ('original', 'residual')} for name in GATE_NAMES}

        def observe(module, positional, inputs):
            if not module.training or len(self.report['batches']) >= self.step:
                return
            expected_batch = self.record['data']['training_batches'][self.step - 1]
            actual = {k: tensor_record(inputs[k])['sha256'] for k in ('input_ids', 'attention_mask')}
            require(all(actual[k] == expected_batch['tensor_sha256'][k] for k in actual),
                    'Actual native sampler/encoding differs from the shared prepared prefix.')
            self.report['batches'].append({'step': self.step, 'shape': list(inputs['input_ids'].shape), **actual})

        self.hook = model.register_forward_pre_hook(observe, with_kwargs=True)
        self.report['actual_inventory'] = expected
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = int(state.global_step) + 1
        self.lr = [group['lr'] for group in kwargs['optimizer'].param_groups]
        if self.step in self.observe_steps:
            for k, p in self.parameters.items():
                self.before[k].copy_(p.detach())
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.step not in self.observe_steps:
            return control
        row = {'step': self.step, 'lr': self.lr, 'gates': {}}
        for name, groups in self.groups.items():
            item = {}
            for label, selected in groups.items():
                if not selected:
                    continue
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected.values()),
                        'Missing/nonfinite actual gradient: ' + name)
                values = torch.cat([p.grad.detach().reshape(4, -1) for p in selected.values()], dim=1)
                counts = torch.count_nonzero(values, dim=1).tolist()
                require(all(n > 0 for n in counts), 'Missing actual gradient in a gate stage: ' + name + '/' + label)
                item[label] = {'l2_by_stage': values.double().square().sum(1).sqrt().tolist(),
                               'nonzero_by_stage': counts}
            # Zero-output initialization intentionally gives norm1 zero gradients before linear1 first updates.
            if self.arm == 'r1':
                item['residual_leaf_nonzero'] = {k.rsplit('._parallel_stage_decoder.', 1)[1]:
                    torch.count_nonzero(p.grad.detach().reshape(4, -1), dim=1).tolist()
                    for k, p in groups['residual'].items()}
            row['gates'][name] = item
        self.report['gradients'].append(row)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.step in self.observe_steps:
            row = {'step': int(state.global_step), 'gates': {}}
            for name, groups in self.groups.items():
                item = {}
                for label, selected in groups.items():
                    if not selected:
                        continue
                    changes = torch.cat([(p.detach() != self.before[k]).reshape(4, -1)
                                         for k, p in selected.items()], dim=1).sum(1).tolist()
                    require(self.step == 1 or all(n > 0 for n in changes),
                            'No update after warmup-zero first step: ' + name + '/' + label)
                    item[label] = changes
                row['gates'][name] = item
            require(all(torch.isfinite(p).all() for p in self.parameters.values()), 'Nonfinite selected parameter.')
            moments = [v for entry in kwargs['optimizer'].state.values()
                       for key, v in entry.items() if key in ('exp_avg', 'exp_avg_sq')]
            require(len(moments) == 2 * len(self.parameters)
                    and all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in moments),
                    'Actual Adam moments must be finite FP32 for every selected leaf.')
            self.report['updates'].append(row)
        if state.global_step == self.stop:
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        observed = [x['trainer_global_step'] for x in self.report['native_saves']]
        require(state.global_step == self.record['schedule']['export_steps'][len(observed)], 'Unexpected actual native save.')
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        trainer, meta = read(checkpoint / 'trainer_state.json'), read(checkpoint / 'checkpoint_meta.json')
        require(trainer['global_step'] == state.global_step and trainer['max_steps'] == 2000
                and (checkpoint / 'training_model_state.pt').is_file(), 'Incomplete native training save.')
        self.report['native_saves'].append({'checkpoint': str(checkpoint), 'checkpoint_id': meta['checkpoint_id'],
            'trainer_global_step': trainer['global_step'], 'max_steps': trainer['max_steps']})
        return control

    def on_train_end(self, args, state, control, **kwargs):
        require(state.global_step == self.stop and len(self.report['batches']) == self.stop
                and [x['trainer_global_step'] for x in self.report['native_saves']]
                == self.record['schedule']['export_steps'], 'Incomplete native training/saves.')
        current = kwargs['model'].state_dict()
        frozen = {k: (v.data_ptr(), v._version, tuple(v.shape), str(v.dtype))
                  for k, v in current.items() if k not in self.parameters}
        require(frozen == self.frozen, 'A supposedly frozen tensor changed during training.')
        self.report['persistent_frozen_versions_unchanged'] = True
        self.report['fp32_changed_elements'] = {k: int(torch.count_nonzero(p.detach().cpu() != self.initial[k]))
                                                 for k, p in self.parameters.items()}
        self.report['bf16_changed_elements'] = {k: int(torch.count_nonzero(p.detach().cpu().bfloat16()
            != self.initial[k].bfloat16())) for k, p in self.parameters.items()}
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
        require(result['global_step'] == record['schedule']['actual_stop_step']
                and result['round_base_checkpoint_id'] == expected_id, 'Wrong native result/source.')
        write_json(output / 'training.json', {'status': 'completed', 'arm': arm, 'training_result': result,
            'observations': observer.report, 'elapsed_seconds': time.time() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
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
