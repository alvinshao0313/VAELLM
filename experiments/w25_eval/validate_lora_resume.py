"""Observe two native LoRA updates after exact step1200 restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.validate_head_resume import assert_tensor_equal
from train_utils.distill_data import build_distill_dataset, build_distill_data_collator


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configuration(record):
    from experiments.w25_eval.resume_lora_only import validate_resume
    argv = record['training_argv']
    _require(argv[:4] == ['--end_step', '1202', '--export_cuda_visible_devices', '4'],
             'This diagnostic only observes the preserved1200 ->1202 trajectory.')
    cfg, hf, args = parse_args(argv[4:])
    details = validate_resume(cfg, hf, args, 1202, diagnostic=True)
    meta, state, _snapshot, source, base, _architecture = details
    _require(state['global_step'] == 1200 and meta['checkpoint_id'] == record['resume_checkpoint_id']
             and base['checkpoint_id'] == record['source_checkpoint_id']
             and str(source) == record['source_checkpoint'], 'Wrong native resume/source identity.')
    return cfg, hf, args, details


def prepare_reference(record):
    cfg, hf, args, details = configuration(record)
    meta, _state, _snapshot, _source, base, _architecture = details
    tokenizer = _build_v6_tokenizer(base['base_model_path'], hf)
    bundle = build_distill_dataset(cfg.data, tokenizer)
    _require(bundle.is_iterable and not bundle.group_by_length,
             'The native SFT iterable must preserve runtime-disabled length grouping.')
    collator = build_distill_data_collator(tokenizer, model_max_length=cfg.data.model_max_length,
                                          dynamic_padding=cfg.data.dynamic_padding)
    workers = meta['immutable_resume_contract']['dataloader']['num_workers']
    loader = DataLoader(bundle.train_dataset, batch_size=args.per_device_train_batch_size,
        num_workers=workers, drop_last=args.dataloader_drop_last, collate_fn=collator, pin_memory=False)
    observed = []
    for ordinal, batch in enumerate(loader, start=1):
        if ordinal > 1200:
            observed.append({'step': ordinal, 'shape': list(batch['input_ids'].shape),
                'hashes': {key: tensor_hash(value) for key, value in batch.items()},
                'attention_tokens': int(batch['attention_mask'].sum())})
        if ordinal == 1202:
            break
    _require([row['step'] for row in observed] == [1201, 1202], 'Reference stream ended early.')
    write_json(record['data_reference'], {'status': 'prepared', 'workers': workers,
        'batch_size': args.per_device_train_batch_size, 'is_iterable': True, 'group_by_length': False,
        'batches_replayed_without_training': 1202, 'batches': observed,
        'protocol': 'Same canonical dataset/collator and original worker sharding; no model or updates.'})


class ResumeObservations(TrainerCallback):
    def __init__(self, checkpoint, reference):
        self.checkpoint = Path(checkpoint)
        self.reference = {row['step']: row for row in reference['batches']}
        self.report = {'batches': [], 'optimizer_steps': []}
        self.hook = None

    def frozen_state(self, model):
        return {key: tensor_hash(value) for key, value in model.state_dict().items() if key not in self.active}

    def on_train_begin(self, args, state, control, **kwargs):
        self.model = kwargs['model']
        self.active = {name: value for name, value in self.model.named_parameters() if value.requires_grad}
        self.saved = torch.load(self.checkpoint / 'training_model_state.pt', mmap=True, map_location='cpu', weights_only=True)
        _require(set(self.active) == set(self.saved) and len(self.active) == 144
                 and all(value.dtype == torch.float32 for value in self.active.values()), 'Wrong native FP32 LoRA selection.')
        for key, value in self.active.items():
            assert_tensor_equal(value, self.saved[key], 'FP32 LoRA ' + key)
        self.optimizer = kwargs['optimizer']
        while hasattr(self.optimizer, 'optimizer'):
            self.optimizer = self.optimizer.optimizer
        _require({id(p) for group in self.optimizer.param_groups for p in group['params']}
                 == {id(p) for p in self.active.values()}, 'Optimizer includes a frozen parameter.')
        saved = torch.load(self.checkpoint / 'optimizer.pt', mmap=True, map_location='cpu', weights_only=True)
        actual = self.optimizer.state_dict()
        _require(saved['param_groups'] == actual['param_groups'] and saved['state'].keys() == actual['state'].keys(),
                 'Restored Adam groups or slots changed.')
        for key, slots in saved['state'].items():
            _require(slots.keys() == actual['state'][key].keys(), 'Adam slot inventory changed.')
            for slot, value in slots.items():
                if torch.is_tensor(value):
                    assert_tensor_equal(actual['state'][key][slot], value, 'Adam ' + slot)
                else:
                    _require(actual['state'][key][slot] == value, 'Adam scalar changed.')
        scheduler = torch.load(self.checkpoint / 'scheduler.pt', map_location='cpu', weights_only=True)
        _require(kwargs['lr_scheduler'].state_dict() == scheduler and scheduler['last_epoch'] == 1200
                 and all(int(slot['step']) == 1200 for slot in self.optimizer.state.values()),
                 'Optimizer/scheduler did not resume at the original1200 boundary.')
        self.frozen = self.frozen_state(self.model)
        self.frozen_parameters = [p for p in self.model.parameters() if not p.requires_grad]
        self.report.update(start_step=1200, restored_state_exact={'fp32_lora': True, 'adam': True, 'scheduler': True},
            mutable_tensors=len(self.active), frozen_tensors=len(self.frozen), restored_scheduler=scheduler,
            data_workers=args.dataloader_num_workers, restore_callback_states_from_checkpoint=args.restore_callback_states_from_checkpoint)

        def observe(_model, positional, values):
            inputs = values.get('input_ids', positional[0] if positional else None)
            observed = {'step': self.step, 'shape': list(inputs.shape),
                'hashes': {'input_ids': tensor_hash(inputs), 'attention_mask': tensor_hash(values['attention_mask'])},
                'attention_tokens': int(values['attention_mask'].sum())}
            expected = self.reference[self.step]
            _require(observed['shape'] == expected['shape'] and all(
                value == expected['hashes'][key] for key, value in observed['hashes'].items()),
                f'Native resumed input differs from canonical batch{self.step}.')
            self.report['batches'].append(observed)
        self.hook = self.model.register_forward_pre_hook(observe, with_kwargs=True)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = state.global_step + 1
        if state.global_step == 1200:
            saved = torch.load(self.checkpoint / 'rng_state.pth', map_location='cpu', weights_only=False)
            numpy_state = np.random.get_state()
            _require(random.getstate() == saved['python'] and numpy_state[0] == saved['numpy'][0]
                and np.array_equal(numpy_state[1], saved['numpy'][1]) and numpy_state[2:] == saved['numpy'][2:]
                and torch.equal(torch.get_rng_state(), saved['cpu'])
                and torch.equal(torch.cuda.get_rng_state(), saved['cuda']),
                'Native RNG was not restored after data skip and before the first resumed forward.')
            self.report['rng_restored_exact'] = True
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        _require(all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
                     for p in self.active.values()) and all(p.grad is None for p in self.frozen_parameters),
                 'Gradients escaped the original finite FP32 LoRA selection.')
        self.report['frozen_parameter_gradients_absent'] = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        _require(all(int(slot['step']) == state.global_step for slot in self.optimizer.state.values())
                 and kwargs['lr_scheduler'].state_dict()['last_epoch'] == state.global_step,
                 'Optimizer/scheduler did not advance from the original boundary.')
        self.report['optimizer_steps'].append({'step': state.global_step, 'lr': self.optimizer.param_groups[0]['lr'],
            'moment_dtypes': sorted({str(slot[key].dtype) for slot in self.optimizer.state.values() for key in ('exp_avg', 'exp_avg_sq')})})
        if state.global_step == 1202:
            changed = sum(int(torch.count_nonzero(value.detach().cpu() != self.saved[key])) for key, value in self.active.items())
            _require(changed > 0 and self.frozen_state(self.model) == self.frozen, 'No genuine LoRA update or a frozen tensor changed.')
            self.report.update(changed_fp32_lora_elements=changed, frozen_state_bytes_unchanged=True)
            self.close()
        return control

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def train(record):
    from experiments.w25_eval.resume_lora_only import run_training
    cfg, _hf, _args, _details = configuration(record)
    observer = ResumeObservations(cfg.resume_from_checkpoint, json.loads(Path(record['data_reference']).read_text()))
    started = time.perf_counter()
    try:
        result = run_training(record['training_argv'], extra_callbacks=(observer,), diagnostic=True)
        _require([row['step'] for row in observer.report['batches']] == [1201, 1202]
                 and [row['step'] for row in observer.report['optimizer_steps']] == [1201, 1202], 'Missing actual resumed updates.')
        write_json(Path(record['output_dir']) / 'training.json', {'status': 'trained', 'observations': observer.report,
            'training_result': result, 'data_reference_sha256': sha256(record['data_reference']),
            'data_skip_inputs_exact': True, 'labels_boundary': 'Labels retained only in canonical CPU reference; actual model input IDs/mask observed.',
            'elapsed_seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'diagnostic_not_quality': True})
    finally:
        observer.close()


def verify(record):
    from experiments.w25_eval.lora_state import inspect_step
    output = Path(record['output_dir'])
    trained = json.loads((output / 'training.json').read_text())
    native = Path(trained['training_result']['run_output_dir']) / 'trainer_state/checkpoint-1202'
    meta, state, _snapshot, _source, _base, _architecture = inspect_step(native)
    scheduler = torch.load(native / 'scheduler.pt', map_location='cpu', weights_only=True)
    optimizer = torch.load(native / 'optimizer.pt', map_location='cpu', weights_only=True, mmap=True)
    _require(state['global_step'] == scheduler['last_epoch'] == 1202 and state['max_steps'] == 2000
             and all(int(slot['step']) == 1202 for slot in optimizer['state'].values())
             and len(meta['mutable_state_manifest']) == 144, 'Saved native1202 state is incomplete.')
    write_json(output / 'verification.json', {'status': 'verified', 'start_step': 1200, 'end_step': 1202,
        'resume_checkpoint_id': record['resume_checkpoint_id'], 'source_checkpoint_id': record['source_checkpoint_id'],
        'native1202_checkpoint_id': meta['checkpoint_id'], 'training_report_sha256': sha256(output / 'training.json'),
        'native1202_meta_sha256': sha256(native / 'checkpoint_meta.json'), 'observations': trained['observations'],
        'data_skip_inputs_exact': True, 'export_validation': 'Reuse audited LoRA step1200 native-to-BF16/strict export; no additional full8 or deployment audit.',
        'diagnostic_not_quality': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', type=Path, required=True)
    parser.add_argument('--phase', choices=('reference', 'train', 'verify'), required=True)
    args = parser.parse_args()
    record = json.loads(args.launch_record.read_text())
    {'reference': prepare_reference, 'train': train, 'verify': verify}[args.phase](record)
