"""Llama Q/K scope around native periodic export; inventory comes from the source."""
from pathlib import Path
import subprocess
import time

import torch

from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.subset_state import inspect_step, persistent_inventory, selected_names

SAVE_STEPS = [400, 800, 1200, 1600, 2000]


class QKStepCallback(AuxStepExportCallback):
    def __init__(self, *, record, **kwargs):
        super().__init__(**kwargs)
        self.record, self.completed_steps = record, []

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 400
                and args.get_warmup_steps(2000) == 20, 'Keep the fresh2000/warm20/save400 schedule.')
        model, selected = kwargs['model'], set(self.record['selected_targets'])
        inv = self.record['expected_inventory']
        inventory, banks = persistent_inventory(model, self.record['architecture'])
        expected = {key for name in selected for key in inventory[name]}
        active = {key: p for key, p in model.named_parameters() if p.requires_grad}
        require(len(selected) == 2 * self.record['architecture']['num_hidden_layers'] == inv['selected_projections']
                and len(expected) == len(active) == inv['selected_decoder_tensors'] and set(active) == expected
                and all(p.dtype == torch.float32 for p in active.values())
                and sum(p.numel() for p in active.values()) == inv['selected_decoder_elements'] and not hasattr(model, 'sparse_bit_tuning'),
                'Only the actual Q/K decoder leaves may train.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        require(not hasattr(optimizer, 'bit_optimizer')
                and {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in active.values()},
                'Optimizer scope differs from the actual Q/K decoder parameters.')
        serialized = model.state_dict()
        require(len(serialized) == inv['serialized_tensors'] and len(set(serialized) - expected) == inv['frozen_tensors']
                and sum(len(v) for v in banks.values()) == inv['all_code_banks'], 'Wrong actual complete/frozen inventory.')
        modules = dict(model.named_modules())
        for name in inventory:
            module, chosen = modules[name], name in selected
            require(module.trainable_decode == chosen and module.cache_decoded_weight == (not chosen)
                    and module.weight_rotation.block_size == self.record['rotation_blocks'][name],
                    'Native decode/cache/rotation policy differs: ' + name)
        write_json(self.run_root_dir / 'actual_scope.json', {'status': 'verified', 'selected_targets': sorted(selected),
            'decoder_keys': sorted(active), 'decoder_tensors': len(active),
            'decoder_elements': sum(p.numel() for p in active.values()),
            'frozen_tensors': len(serialized) - len(active), 'all_code_banks': sum(map(len, banks.values())),
            'serialized_tensors': len(serialized), 'selected_trainable_decode_no_cache': len(selected),
            'frozen_cached_projections': len(inventory) - len(selected),
            'optimizer_exact': True, 'persistent_values_checked_later_by_native_step_audit': True})
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == self._last_step:
            return control
        require(step == SAVE_STEPS[len(self.completed_steps)], 'Unexpected native save order.')
        checkpoint = Path(args.output_dir).resolve() / f'checkpoint-{step}'
        meta, saved, _snapshot, source, base, architecture = inspect_step(checkpoint)
        require(saved['global_step'] == step and meta['train_mode'] == 'decoder' and source == self.source_dir
                and base['checkpoint_id'] == self.record['source_checkpoint_id']
                and architecture == self.record['architecture']
                and list(selected_names(meta['target_layers'], meta['target_modules'], architecture)) == self.record['selected_targets'],
                'Saved step/source/QK-only scope differs.')
        check_inputs(self.record)
        output = self.run_root_dir / f'step_{step:04d}'
        output.mkdir(exist_ok=False)
        path = output / 'step_export.json'
        item = {'status': 'running', 'global_step': step, 'max_steps': 2000, 'source_checkpoint': str(checkpoint),
            'selected_checkpoint_id': meta['checkpoint_id'], 'round_base_checkpoint_id': base['checkpoint_id'],
            'export_gpu': self.env['CUDA_VISIBLE_DEVICES'], 'stages': {}, 'started_unix': time.time()}
        started = time.monotonic()
        try:
            query = subprocess.run(['nvidia-smi', '--id=' + self.env['CUDA_VISIBLE_DEVICES'],
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], stdin=subprocess.DEVNULL,
                text=True, capture_output=True, check=True)
            require(not query.stdout.strip(), 'The dedicated export GPU is occupied: ' + query.stdout.strip())
            for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=source, output_dir=output):
                item['stages'][name] = {'argv': command, 'started_unix': time.time()}
                write_json(path, item)
                try:
                    status = run_logged_stage(command, name=name, output_dir=output, env=self.env, allowed_exit_codes=allowed)
                finally:
                    exit_path = output / (name + '.exit')
                    if exit_path.exists():
                        value = exit_path.read_text().strip()
                        item['stages'][name]['exit_code'] = int(value) if value.lstrip('-').isdigit() else value
                    item['stages'][name]['finished_unix'] = time.time()
                    write_json(path, item)
                require(status == 0, 'Every native export/audit/full8 stage must succeed.')
            audit = read(output / 'audit.json')
            inv = self.record['expected_inventory']
            require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == base['checkpoint_id']
                    and audit['selected_checkpoint_id'] == meta['checkpoint_id'] and audit['selected_global_step'] == step
                    and audit['selected_modules'] == self.record['selected_targets']
                    and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == inv['serialized_tensors']
                    and audit['selected_decoder_tensor_count'] == inv['selected_decoder_tensors']
                    and audit['frozen_tensors_compared'] == inv['frozen_tensors']
                    and audit['code_bank_count'] == audit['frozen_code_bank_count'] == inv['all_code_banks'] and audit['changed_code_bits'] == 0
                    and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['fresh_strict_reload_state_exact']
                    and audit['frozen_bytes_unchanged'] and audit['rank_zero'], 'Saved subset/frozen-state audit failed.')
            budget = audit['bit_budget']
            for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components'):
                require(budget[key] == self.record['source_bit_budget'][key], 'Persistent storage changed: ' + key)
            final = output / 'final_model'
            final_meta = read(final / 'checkpoint_meta.json')
            require(final_meta['checkpoint_id'] == audit['checkpoint_id'] and budget['budget_passed'], 'Export identity/budget differs.')
            paths = [output / (name + '.json') for name in ('mmlu', 'seven')]
            result = collect_results(paths, architecture, checkpoint=(final, final_meta))
            for actual, reference in zip(paths, self.record['control_result_paths'], strict=True):
                a, b = (read(p)['evals']['lm_eval'] for p in (actual, reference))
                require(all(a[k] == b[k] for k in ('configs', 'versions', 'n_shot', 'n_samples')), 'Full8 protocol differs.')
            summary = {'status': 'full8_evaluation_verified', 'checkpoint_id': final_meta['checkpoint_id'],
                'checkpoint_dir': str(final), 'global_step': step, **result, 'bit_budget': budget,
                'source_mean_percent': self.record['control_mean_percent'],
                'source_delta_percentage_points': result['mean_percent'] - self.record['control_mean_percent'],
                'source_checkpoint_id': self.record['source_checkpoint_id'],
                'historical_e767_context': self.record['historical_e767_context'],
                'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated', 'goal_acceptance': False,
                'audit_sha256': sha256(output / 'audit.json')}
            write_json(output / 'full8_summary.json', summary)
            check_inputs(self.record)
            item.update(status='evaluated', checkpoint_id=final_meta['checkpoint_id'], mean_percent=result['mean_percent'],
                full8_summary_sha256=sha256(output / 'full8_summary.json'), audit_sha256=sha256(output / 'audit.json'))
            self._last_step = step
            self.completed_steps.append(step)
        except BaseException as error:
            item.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            item.update(elapsed_seconds=time.monotonic() - started, finished_unix=time.time())
            write_json(path, item)
        return control
