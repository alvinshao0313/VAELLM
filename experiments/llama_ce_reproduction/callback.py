"""Model-adaptive synchronous decoder export, audit and full-eight evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import time

from experiments.longbench_v1.common import write_json, sha256
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback, stage_commands as fixed_commands
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.subset_state import inspect_step, selected_names

LOG = logging.getLogger(__name__)


def stage_commands(*, checkpoint, source_dir, output_dir, diagnostic=False):
    commands = [row for row in fixed_commands(checkpoint=checkpoint, source_dir=source_dir,
                                            output_dir=output_dir) if row[0] != 'summary']
    commands[0][1][2] = 'experiments.w25_eval.subset_export'
    commands[1][1][2] = 'experiments.w25_eval.subset_audit'
    commands[1][1].extend(['--training_checkpoint_dir', str(checkpoint)])
    return commands[:2] if diagnostic else commands


class DecoderStepCallback(AuxStepExportCallback):
    def __init__(self, *, record, **kwargs):
        super().__init__(**kwargs)
        self.record = record
        self.diagnostic_only = record['kind'] == 'validate_s4'
        self.completed_steps = []

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        require(state.global_step == 0 and args.max_steps == 2000 and args.get_warmup_steps(2000) == 20
                and args.save_steps == (2 if self.diagnostic_only else 400), 'Wrong fresh CE schedule or save interval.')
        return control

    def on_save(self, args, state, control, **kwargs):
        from experiments.llama_ce_reproduction.train import check_inputs, read
        step = int(state.global_step)
        if step == self._last_step:
            return control
        expected_steps = [2, 4] if self.diagnostic_only else [400, 800, 1200, 1600, 2000]
        require(step == expected_steps[len(self.completed_steps)], 'Unexpected native save order.')
        checkpoint = Path(args.output_dir).resolve() / f'checkpoint-{step}'
        meta, saved, _snapshot, source, base, architecture = inspect_step(checkpoint)
        require(saved['global_step'] == step and meta['train_mode'] == 'decoder'
                and source == self.source_dir and base['checkpoint_id'] == self.record['source_checkpoint_id']
                and architecture == self.record['architecture']
                and set(selected_names(meta['target_layers'], meta['target_modules'], architecture))
                == set(architecture['projection_shapes']), 'The saved step/source/all-projection scope differs.')
        check_inputs(self.record)
        output = self.run_root_dir / f'step_{step:04d}'
        output.mkdir(parents=True, exist_ok=False)
        path = output / 'step_export.json'
        item = {'status': 'running', 'global_step': step, 'max_steps': 2000,
            'source_checkpoint': str(checkpoint), 'selected_checkpoint_id': meta['checkpoint_id'],
            'round_base_checkpoint_id': base['checkpoint_id'], 'export_gpu': self.env['CUDA_VISIBLE_DEVICES'],
            'diagnostic_only': self.diagnostic_only, 'stages': {}, 'started_unix': time.time()}
        start = time.monotonic()
        try:
            query = subprocess.run(['nvidia-smi', '--id=' + self.env['CUDA_VISIBLE_DEVICES'],
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], stdin=subprocess.DEVNULL,
                text=True, capture_output=True, check=True)
            require(not query.stdout.strip(), 'The dedicated export GPU is occupied: ' + query.stdout.strip())
            item['export_gpu_idle_checked'] = True
            for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=source,
                    output_dir=output, diagnostic=self.diagnostic_only):
                item['stages'][name] = {'argv': command, 'started_unix': time.time()}
                write_json(path, item)
                try:
                    status = run_logged_stage(command, name=name, output_dir=output, env=self.env,
                                              allowed_exit_codes=allowed)
                finally:
                    exit_path = output / (name + '.exit')
                    if exit_path.exists():
                        value = exit_path.read_text().strip()
                        item['stages'][name]['exit_code'] = int(value) if value.lstrip('-').isdigit() else value
                    item['stages'][name]['finished_unix'] = time.time()
                    write_json(path, item)
                require(status == 0, 'Every independent stage must complete successfully.')
            audit = read(output / 'audit.json')
            inv = self.record['expected_inventory']
            require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == base['checkpoint_id']
                    and audit['selected_checkpoint_id'] == meta['checkpoint_id'] and audit['selected_global_step'] == step
                    and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == inv['serialized_tensors']
                    and audit['selected_decoder_tensor_count'] == inv['selected_decoder_tensors']
                    and audit['frozen_tensors_compared'] == inv['frozen_tensors']
                    and audit['code_bank_count'] == inv['all_code_banks'] and audit['changed_code_bits'] == 0
                    and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['fresh_strict_reload_state_exact']
                    and audit['frozen_bytes_unchanged'] and audit['rank_zero'], 'Saved decoder/frozen-state audit failed.')
            budget = audit['bit_budget']
            for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components'):
                require(budget[key] == self.record['source_bit_budget'][key], 'Persistent storage changed: ' + key)
            require(budget['budget_passed'] and budget['user_linear_bpw'] <= 2.5, 'Bit budget violated.')
            final = output / 'final_model'
            final_meta = read(final / 'checkpoint_meta.json')
            require(final_meta['checkpoint_id'] == audit['checkpoint_id'], 'Export and audit identities differ.')
            if not self.diagnostic_only:
                current_paths = [output / (name + '.json') for name in ('mmlu', 'seven')]
                current = collect_results(current_paths, architecture, checkpoint=(final, final_meta))
                for current_path, reference_path in zip(current_paths, self.record['control_result_paths'], strict=True):
                    actual_raw, reference_raw = (read(p)['evals']['lm_eval'] for p in (current_path, reference_path))
                    for field in ('configs', 'versions', 'n_shot', 'n_samples'):
                        require(actual_raw[field] == reference_raw[field], f'Full8 {field} differs from CAT control.')
                summary = {'status': 'full8_evaluation_verified', 'checkpoint_id': final_meta['checkpoint_id'],
                    'checkpoint_dir': str(final), 'global_step': step, **current, 'bit_budget': budget,
                    'control_mean_percent': self.record['control_mean_percent'],
                    'mean_delta_percentage_points': current['mean_percent'] - self.record['control_mean_percent'],
                    'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated',
                    'goal_acceptance': False, 'audit_sha256': sha256(output / 'audit.json')}
                write_json(output / 'full8_summary.json', summary)
                item.update(mean_percent=current['mean_percent'], mmlu_percent=100 * current['task_metrics']['mmlu'],
                            full8_summary_sha256=sha256(output / 'full8_summary.json'))
            check_inputs(self.record)
            item.update(status='diagnostic_complete' if self.diagnostic_only else 'evaluated',
                        checkpoint_id=final_meta['checkpoint_id'], audit_sha256=sha256(output / 'audit.json'))
            self._last_step = step
            self.completed_steps.append(step)
        except BaseException as error:
            item.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            item.update(elapsed_seconds=time.monotonic() - start, finished_unix=time.time())
            write_json(path, item)
        LOG.info('Decoder step %d independent export/evaluation finished; returning control to native training.', step)
        return control
