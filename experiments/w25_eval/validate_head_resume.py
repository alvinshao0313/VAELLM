"""Observe native400 ->404 restoration; evaluation and model export stay in existing tools."""
from __future__ import annotations

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
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.head_state import HEAD_KEY
from experiments.w25_eval.resume_head_only import run_training, validate_resume
from train_utils.distill_data import build_distill_dataset, build_distill_data_collator


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fixed_head_probe_indices(numel, device=None):
    # FP32 linspace can round the final coordinate up to numel for a large head.
    _require(numel >= 64, 'The head must contain at least64 diagnostic coordinates.')
    return torch.arange(64, dtype=torch.int64, device=device) * (numel - 1) // 63


def assert_tensor_equal(current, saved, label):
    _require(current.shape == saved.shape and current.dtype == saved.dtype, f'{label}: shape/dtype changed.')
    left, right = current.detach().reshape(-1), saved.reshape(-1)
    for start in range(0, left.numel(), 4 * 1024 * 1024):
        _require(torch.equal(left[start:start + 4 * 1024 * 1024].cpu(),
                             right[start:start + 4 * 1024 * 1024].cpu()), f'{label}: restored values differ.')


def configuration(record):
    argv = record['training_argv']
    _require(argv[:2] == ['--end_step', '404'], 'This validation only observes native400 ->404.')
    cfg, hf, args = parse_args(argv[2:])
    details = validate_resume(cfg, hf, args, 404, diagnostic=True)
    meta, state, _snapshot, source, base, _architecture = details
    _require(state['global_step'] == 400 and meta['checkpoint_id'] == record['resume_checkpoint_id']
             and base['checkpoint_id'] == record['source_checkpoint_id']
             and str(source) == record['source_checkpoint'], 'Wrong actual resume/source identity.')
    return cfg, hf, args, details


def prepare_reference(record):
    """CPU tokenization replay, without model construction or training updates."""
    cfg, hf, args, details = configuration(record)
    meta, _state, _snapshot, _source, base, _architecture = details
    tokenizer = _build_v6_tokenizer(base['base_model_path'], hf)
    bundle = build_distill_dataset(cfg.data, tokenizer)
    _require(bundle.is_iterable and not bundle.group_by_length, 'Expected the original ungrouped five-source iterable.')
    collator = build_distill_data_collator(tokenizer, model_max_length=cfg.data.model_max_length,
                                          dynamic_padding=cfg.data.dynamic_padding)
    workers = meta['immutable_resume_contract']['dataloader']['num_workers']
    loader = DataLoader(bundle.train_dataset, batch_size=args.per_device_train_batch_size,
                        num_workers=workers, drop_last=args.dataloader_drop_last,
                        collate_fn=collator, pin_memory=False)
    observed = []
    for ordinal, batch in enumerate(loader, start=1):
        if ordinal > 400:
            observed.append({'step': ordinal, 'shape': list(batch['input_ids'].shape),
                'hashes': {key: tensor_hash(value) for key, value in batch.items()},
                'attention_tokens': int(batch['attention_mask'].sum()),
                'labels_observed': True})
        if ordinal == 404:
            break
    _require([x['step'] for x in observed] == [401, 402, 403, 404], 'Reference stream ended early.')
    report = {'status': 'prepared', 'workers': workers, 'batch_size': args.per_device_train_batch_size,
        'seed': cfg.data.seed, 'data_seed': cfg.data.data_seed, 'group_by_length': bundle.group_by_length,
        'is_iterable': True, 'batches_replayed_without_training': 404,
        'batches': observed, 'protocol': 'Same canonical builder/collator and worker sharding; no model loaded.'}
    Path(record['data_reference']).write_text(json.dumps(report, indent=2) + '\n')


class ResumeObservations(TrainerCallback):
    def __init__(self, checkpoint, reference):
        self.checkpoint = Path(checkpoint)
        self.reference = {row['step']: row for row in reference['batches']}
        self.report = {'batches': [], 'optimizer_steps': []}
        self.hook = None
        self.step = None

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        _require(set(active) == {HEAD_KEY} and active[HEAD_KEY].dtype == torch.float32
                 and not hasattr(model, 'sparse_bit_tuning'), 'Resume escaped the head-only FP32 selection.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        self.optimizer, self.head = optimizer, active[HEAD_KEY]
        self.frozen_parameters = [p for name, p in model.named_parameters() if name != HEAD_KEY]
        self.probe_indices = fixed_head_probe_indices(self.head.numel(), self.head.device)
        self.initial_head_probe = self.head.detach().reshape(-1)[self.probe_indices].cpu().clone()
        _require({id(p) for group in optimizer.param_groups for p in group['params']} == {id(self.head)},
                 'Adam contains parameters other than the original head.')
        saved_head = torch.load(self.checkpoint / 'training_model_state.pt', mmap=True, map_location='cpu', weights_only=True)
        _require(set(saved_head) == {HEAD_KEY}, 'Saved mutable payload is not solely the head.')
        assert_tensor_equal(self.head, saved_head[HEAD_KEY], 'FP32 head')
        saved = torch.load(self.checkpoint / 'optimizer.pt', mmap=True, map_location='cpu', weights_only=True)
        actual = optimizer.state_dict()
        _require(saved['param_groups'] == actual['param_groups'] and saved['state'].keys() == actual['state'].keys(),
                 'Restored Adam groups/state inventory changed.')
        compared = []
        for param, saved_slots in saved['state'].items():
            _require(saved_slots.keys() == actual['state'][param].keys(), 'Adam slot keys changed.')
            for key, value in saved_slots.items():
                current = actual['state'][param][key]
                if torch.is_tensor(value):
                    assert_tensor_equal(current, value, 'Adam ' + key)
                else:
                    _require(current == value, 'Adam scalar changed.')
                compared.append(key)
        adam = optimizer.state[self.head]
        _require(int(adam['step']) == state.global_step == 400
                 and adam['exp_avg'].dtype == adam['exp_avg_sq'].dtype == torch.float32,
                 'Adam step/moments were not restored from400.')
        scheduler = torch.load(self.checkpoint / 'scheduler.pt', map_location='cpu', weights_only=True)
        _require(kwargs['lr_scheduler'].state_dict() == scheduler and scheduler['last_epoch'] == 400,
                 'The original scheduler was not restored.')
        self.report.update(start_step=400, restored_state_exact={'head': True, 'adam': True, 'scheduler': True},
            restored_adam_slots=compared, restored_scheduler=scheduler, trainable_keys=list(active),
            mutable_dtype=str(self.head.dtype), data_workers=args.dataloader_num_workers,
            restore_callback_states_from_checkpoint=args.restore_callback_states_from_checkpoint)

        def observe(_module, positional, values):
            inputs = values.get('input_ids', positional[0] if positional else None)
            _require(inputs is not None and self.step in self.reference, 'Unexpected student forward boundary.')
            observed = {'step': self.step, 'shape': list(inputs.shape),
                'hashes': {'input_ids': tensor_hash(inputs), 'attention_mask': tensor_hash(values['attention_mask'])},
                'attention_tokens': int(values['attention_mask'].sum())}
            expected = self.reference[self.step]
            _require(observed['shape'] == expected['shape'] and all(
                value == expected['hashes'][key] for key, value in observed['hashes'].items()),
                f'Actual resumed inputs differ from canonical batch{self.step}.')
            self.report['batches'].append(observed)
        self.hook = model.register_forward_pre_hook(observe, with_kwargs=True)
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        _require(self.head.grad is not None and self.head.grad.dtype == torch.float32
                 and all(p.grad is None for p in self.frozen_parameters),
                 'Gradient ownership escaped the original FP32 head.')
        self.report['frozen_parameter_gradients_absent'] = True
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = state.global_step + 1
        if state.global_step == 400:
            saved = torch.load(self.checkpoint / 'rng_state.pth', map_location='cpu', weights_only=False)
            actual_numpy = np.random.get_state()
            _require(random.getstate() == saved['python']
                     and actual_numpy[0] == saved['numpy'][0]
                     and np.array_equal(actual_numpy[1], saved['numpy'][1])
                     and actual_numpy[2:] == saved['numpy'][2:]
                     and torch.equal(torch.get_rng_state(), saved['cpu'])
                     and torch.equal(torch.cuda.get_rng_state(), saved['cuda']),
                     'RNG differs after native data skip and before the first resumed forward.')
            self.report['rng_restored_exact'] = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        adam = self.optimizer.state[self.head]
        _require(int(adam['step']) == state.global_step
                 and kwargs['lr_scheduler'].state_dict()['last_epoch'] == state.global_step,
                 'Adam/scheduler failed to advance with the restored native step.')
        self.report['optimizer_steps'].append({'step': state.global_step,
            'adam_step': int(adam['step']), 'lr': self.optimizer.param_groups[0]['lr'],
            'moment_dtypes': [str(adam[key].dtype) for key in ('exp_avg', 'exp_avg_sq')]})
        if state.global_step == 404:
            current = self.head.detach().reshape(-1)[self.probe_indices].cpu()
            changed = int(torch.count_nonzero(current != self.initial_head_probe))
            _require(changed > 0, 'No update observed at the64 fixed head coordinates.')
            self.report['head_probe_changed_coordinates'] = changed
            self.report['head_probe_coordinate_count'] = 64
            self.hook.remove()
            self.hook = None
        return control

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def train(record):
    cfg, _hf, _args, _details = configuration(record)
    reference = json.loads(Path(record['data_reference']).read_text())
    callback = ResumeObservations(cfg.resume_from_checkpoint, reference)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = run_training(record['training_argv'], extra_callbacks=(callback,), diagnostic=True)
        observed = callback.report
        _require([row['step'] for row in observed['batches']] == [401, 402, 403, 404]
                 and [row['step'] for row in observed['optimizer_steps']] == [401, 402, 403, 404],
                 'Four genuine resumed forwards/updates were not observed.')
        report = {'status': 'trained', 'source_checkpoint_id': record['source_checkpoint_id'],
            'resume_checkpoint_id': record['resume_checkpoint_id'], 'start_step': 400, 'end_step': 404,
            'observations': observed, 'training_result': result,
            'data_reference_sha256': _sha256(Path(record['data_reference'])),
            'data_skip_inputs_exact': True,
            'data_boundary': 'Actual student input_ids/attention_mask observed; labels only recorded in canonical CPU reference.',
            'elapsed_seconds': time.perf_counter() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'diagnostic_not_quality': True}
        Path(record['output_dir'], 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        callback.close()


def verify(record):
    output = Path(record['output_dir'])
    trained = json.loads((output / 'training.json').read_text())
    observed = trained['observations']
    native = Path(trained['training_result']['run_output_dir']) / 'trainer_state/checkpoint-404'
    meta = json.loads((native / 'checkpoint_meta.json').read_text())
    state = json.loads((native / 'trainer_state.json').read_text())
    scheduler = torch.load(native / 'scheduler.pt', map_location='cpu', weights_only=True)
    _require(meta['round_base_checkpoint_id'] == record['source_checkpoint_id']
             and meta['train_mode'] == 'none' and meta['lm_head_train_mode'] == 'full'
             and len(meta['mutable_state_manifest']) == 1
             and meta['mutable_state_manifest'][0]['name'] == HEAD_KEY
             and not (native / 'sparse_bit_tuning').exists()
             and state['global_step'] == scheduler['last_epoch'] == 404
             and state['max_steps'] == 2000, 'Native404 role/step/scheduler mismatch.')
    report = {'status': 'verified', 'source_checkpoint_id': record['source_checkpoint_id'],
        'resume_checkpoint_id': record['resume_checkpoint_id'], 'start_step': 400, 'end_step': 404,
        'optimizer_steps': [row['step'] for row in observed['optimizer_steps']], 'head_only': True,
        'restored_state_exact': observed['restored_state_exact'], 'rng_restored_exact': observed['rng_restored_exact'],
        'data_skip_inputs_exact': trained['data_skip_inputs_exact'],
        'frozen_parameter_gradients_absent': observed['frozen_parameter_gradients_absent'],
        'head_probe_changed_coordinates': observed['head_probe_changed_coordinates'],
        'head_probe_coordinate_count': 64, 'native404_checkpoint_id': meta['checkpoint_id'],
        'native404_meta_sha256': _sha256(native / 'checkpoint_meta.json'),
        'training_report_sha256': _sha256(output / 'training.json'),
        'export_validation': 'Reuse completed head-only4/head-only400 export evidence; no new full-state audit or full8 here.',
        'diagnostic_not_quality': True}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    parser.add_argument('--phase', choices=('reference', 'train', 'verify'), required=True)
    args = parser.parse_args()
    record = json.loads(args.launch_record.read_text())
    {'reference': prepare_reference, 'train': train, 'verify': verify}[args.phase](record)


if __name__ == '__main__':
    main()
