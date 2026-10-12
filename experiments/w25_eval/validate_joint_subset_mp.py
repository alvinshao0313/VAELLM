"""Observe two native layer-sharded joint updates, then audit an independent export.

The real 2000-step scheduler and save400 contract stay intact. Only a callback
requests an actual step2 save and stop; no training state or step is fabricated.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import persistent_inventory, selected_names
from experiments.w25_eval.train_subset_mp_400 import parse_run_arguments
from experiments.w25_eval.validate_wide_subset_training import TrainingObservations


def configuration(record):
    cfg, hf, args = parse_run_arguments(record['training_argv'])
    meta = json.loads((Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json').read_text())
    architecture = model_inventory(meta['base_model_path'])
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    _require(list(names) == record['selected_targets'] and len(names) == 108,
             'The diagnostic requires the prepared complete MLP subset.')
    _require(meta['checkpoint_id'] == record['source_checkpoint_id']
             and cfg.student_checkpoint_dir == record['source_checkpoint'], 'Fresh source changed.')
    _require(args.max_steps == 2000 and args.save_steps == 400 and args.get_warmup_steps(2000) == 20
             and cfg.opt.batch_size == 4 and args.gradient_accumulation_steps == 1
             and cfg.data.model_max_length == 1024, 'Keep the actual schedule, batch and sequence contract.')
    return cfg, hf, args, names, architecture


class JointObservations(TrainingObservations):
    def __init__(self, names, architecture, expected):
        super().__init__(names)
        self.diagnostic_stop_step = 2
        self.architecture, self.expected = architecture, expected
        self.diagnostic['gradient_steps'] = []

    def on_train_begin(self, args, state, control, **kwargs):
        Budget400Callback().on_train_begin(args, state, control, **kwargs)
        model = kwargs['model']
        inventory, banks = persistent_inventory(model, self.architecture)
        active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        decoder = {key for name in self.selected_targets for key in inventory[name]}
        scores = set(active) - decoder
        _require(set(active) == decoder | scores and len(decoder) == 648
                 and all(key.startswith('sparse_bit_tuning.') for key in scores)
                 and all(value.dtype == torch.float32 for value in active.values()),
                 'Only selected FP32 decoder leaves and sensitivity scores may train.')
        runtime = model.sparse_bit_tuning
        specs = runtime.bank_specs
        _require(len(specs) == 216 and {s.module_path for s in specs} == set(self.selected_targets)
                 and all(s.n_active == s.n_bits for s in specs)
                 and sum(s.n_active for s in specs) == self.expected['active_score_elements']
                 == sum(active[key].numel() for key in scores), 'All selected logical bits must be active.')
        devices = {str(s.device) for s in specs}
        _require(devices == {f'cuda:{i}' for i in range(4)}
                 and all(str(s.device) == f'cuda:{int(s.module_path.split(".")[2]) // 9}' for s in specs),
                 'Native auto placement must use nine layers per logical GPU.')
        self._model = model
        self._manager = model.get_submodule(self.selected_targets[0])._sparse_bit_binding.manager
        manager = self._manager
        _require(manager.bit_round_steps == 512 and manager.stable_steps == 10
                 and manager.config.active_ratio == 1 and manager.config.proxy_coordinates == 'decoder_sensitivity',
                 'Keep native sensitivity coordinates and real early-round termination.')
        # Small deterministic observations only; retain neither full score banks
        # nor GPU tensors. These samples establish real bit updates, not all-bit
        # update counts or a new training/data sampling rule.
        self._initial_score_samples = {
            spec.canonical_key: runtime.score_view(spec).detach().reshape(-1)[:256].cpu().clone()
            for spec in specs}
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        _require(optimizer.bit_manager is manager.bit_optimizer
                 and {id(p) for g in optimizer.main_optimizer.param_groups for p in g['params']}
                 == {id(active[k]) for k in decoder}, 'Main Adam must contain only selected decoders.')
        self.diagnostic['trainable_inventory'] = {
            'decoder_keys': sorted(decoder), 'decoder_tensors': len(decoder),
            'decoder_elements': sum(active[key].numel() for key in decoder),
            'score_keys': sorted(scores), 'score_elements': sum(active[key].numel() for key in scores),
            'selected_code_banks': len(specs), 'all_bits_active': True,
            'score_devices': sorted(devices), 'bit_round_steps': manager.bit_round_steps,
            'native_stable_steps': manager.stable_steps,
            'selected_projection_code_banks': sum(len(banks[name]) for name in self.selected_targets)}

        def observe_inputs(_module, positional, values):
            inputs = values.get('input_ids')
            if inputs is None and positional:
                inputs = positional[0]
            if inputs is not None:
                mask = values.get('attention_mask')
                self.diagnostic['batches'].append({'step': self._observed_step, 'shape': list(inputs.shape),
                    'padded_tokens': inputs.numel(), 'attention_tokens': None if mask is None else int(mask.sum().item())})
        self._input_hook = model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._manager.bit_optimizer.validate_gradients()
        _require(all(p.grad is None for p in self._model.parameters() if not p.requires_grad),
                 'A frozen parameter received a gradient.')
        self.diagnostic['gradient_steps'].append({'step': int(state.global_step) + 1,
            'all_score_chunk_gradients_connected': True, 'frozen_parameter_gradients_absent': True})
        return control

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        manager = self._manager
        bit_state = manager.bit_optimizer._state
        _require(all(s.exp_avg is not None and s.exp_avg_sq is not None
                     and s.exp_avg.dtype == s.exp_avg_sq.dtype == torch.float32 for s in bit_state.values()),
                 'Bit Adam moments must exist in FP32 for every score chunk.')
        self.diagnostic['optimizer_steps'][-1].update({
            'bit_moment_chunks': len(bit_state), 'bit_moment_dtype': 'torch.float32',
            'bit_lr': manager.bit_optimizer.lr, 'bit_round': manager.global_bit_round,
            'bit_round_step': manager.bit_round_step, 'stable_counter': manager.stable_counter,
            'cumulative_flip_count': manager.cumulative_flip_count})
        samples = []
        for spec in manager.score_module.bank_specs:
            before = self._initial_score_samples[spec.canonical_key]
            now = manager.score_module.score_view(spec).detach().reshape(-1)[:before.numel()].cpu()
            moment = bit_state[spec.chunk_id].exp_avg[spec.score_start:spec.score_start + before.numel()].detach().cpu()
            samples.append({'bank': spec.canonical_key, 'sampled_positions': before.numel(),
                            'changed_scores': int(torch.count_nonzero(now != before)),
                            'nonzero_adam_first_moments': int(torch.count_nonzero(moment))})
        _require(sum(x['changed_scores'] for x in samples) > 0
                 and sum(x['nonzero_adam_first_moments'] for x in samples) > 0,
                 'The selected observed score samples did not update.')
        self.diagnostic['optimizer_steps'][-1]['first256_score_observations_per_bank'] = samples
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        meta = json.loads((checkpoint / 'checkpoint_meta.json').read_text())
        trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
        _require(state.global_step == trainer['global_step'] == 2 and trainer['max_steps'] == 2000
                 and (checkpoint / 'sparse_bit_tuning/exact_state.pt').is_file(), 'Real native step2 save is missing.')
        self.diagnostic['native_saves'].append({'step': 2, 'checkpoint_dir': str(checkpoint),
            'checkpoint_id': meta['checkpoint_id'], 'meta_sha256': _sha256(checkpoint / 'checkpoint_meta.json'),
            'trainer_state_sha256': _sha256(checkpoint / 'trainer_state.json')})
        return control


def train(record):
    cfg, hf, args, names, architecture = configuration(record)
    _require(torch.cuda.device_count() == 4, 'The resource validation requires exactly four visible GPUs.')
    callback = JointObservations(names, architecture, record['expected_inventory'])
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    for i in range(4):
        torch.cuda.reset_peak_memory_stats(i)
    try:
        result = run(cfg, hf, args, extra_callbacks=(callback,))
        observed = callback.diagnostic
        _require(result['global_step'] == 2 and result['round_base_checkpoint_id'] == record['source_checkpoint_id'],
                 'Diagnostic must finish at actual step2 from fresh f927.')
        _require([x['step'] for x in observed['optimizer_steps']] == [1, 2]
                 and [x['step'] for x in observed['gradient_steps']] == [1, 2]
                 and len(observed['native_saves']) == 1
                 and {x['step'] for x in observed['batches']} == {1, 2}
                 and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                 and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps']),
                 'Actual batch/update/save or FP32 Adam observations are incomplete.')
        memory = {str(i): {'peak_allocated_bytes': torch.cuda.max_memory_allocated(i),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved(i),
                          'free_bytes_at_finish': torch.cuda.mem_get_info(i)[0],
                          'total_bytes': torch.cuda.mem_get_info(i)[1]} for i in range(4)}
        report = {'status': 'trained', 'training_result': result, 'observations': observed,
                  'gpu_memory': memory, 'diagnostic_not_quality': True,
                  'resource_coverage': 'Two actual native dynamic batches; these may not reach max_length1024. '
                                       'Reuse the previous full252/B4/1024 four-GPU peak evidence separately.'}
        (Path(record['output_dir']) / 'training.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    finally:
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def verify(record):
    output, expected = Path(record['output_dir']), record['expected_inventory']
    trained = json.loads((output / 'training.json').read_text())
    audit = json.loads((output / 'step_0002/audit.json').read_text())
    exported = json.loads((output / 'step_0002/final_model/export_record.json').read_text())
    _require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == record['source_checkpoint_id']
             and audit['selected_modules'] == record['selected_targets']
             and audit['frozen_tensors_compared'] == expected['frozen_tensors']
             and audit['selected_code_bank_count'] == 216 and audit['code_bank_count'] == 504
             and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
             and audit['selected_decoder_fp32_changed_elements'] > 0
             and audit['selected_decoder_matches_fp32_cast_exactly']
             and audit['selected_codes_match_current_snapshot_exactly']
             and audit['fresh_strict_reload_state_exact'] and audit['rank_zero'], 'Independent joint subset audit failed.')
    _require(exported['source_global_step'] == 2 and exported['source_max_steps'] == 2000
             and exported['restored_decoder_tensors'] == 648 and exported['optimizer_steps_executed'] == 0,
             'Export must restore real step2 with no additional training.')
    native = _load_state(Path(trained['training_result']['saved_model_dir']))
    independent = _load_state(output / 'step_0002/final_model')
    compared = _assert_state_equal(native, independent, 'Native joint final versus independent step2')
    del native, independent
    gc.collect()
    _require(compared == 3423, 'Native/independent tensor count differs.')
    report = {'status': 'verified', 'source_checkpoint_id': record['source_checkpoint_id'],
        'checkpoint_id': audit['checkpoint_id'], 'native_vs_step2_tensors_compared': compared,
        'training_report_sha256': _sha256(output / 'training.json'),
        'audit_sha256': _sha256(output / 'step_0002/audit.json'),
        'export_record_sha256': _sha256(output / 'step_0002/final_model/export_record.json'),
        'changed_decoder_fp32_elements': audit['selected_decoder_fp32_changed_elements'],
        'changed_decoder_bf16_elements': audit['changed_decoder_elements'], 'changed_code_bits': audit['changed_code_bits'],
        'bit_budget': audit['bit_budget'], 'gpu_memory': trained['gpu_memory'],
        'export_peak_allocated_bytes': exported['peak_cuda_allocated_bytes'],
        'export_peak_reserved_bytes': exported['peak_cuda_reserved_bytes'], 'diagnostic_not_quality': True}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('train', 'verify'))
    args = parser.parse_args()
    record = json.loads(args.launch_record.read_text())
    {'train': train, 'verify': verify}[args.phase](record)


if __name__ == '__main__':
    main()
