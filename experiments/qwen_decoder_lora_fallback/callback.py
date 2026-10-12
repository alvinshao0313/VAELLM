"""Joint saved-step export/audit and the unchanged complete eight-task protocol."""
from pathlib import Path
import subprocess
import time

from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback, stage_commands as native_commands
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.qwen_decoder_lora_fallback.config import check_inputs, read, require, verify_audit
from experiments.qwen_decoder_lora_fallback.state import inspect_step
from experiments.w25_eval.acceptance import collect_results


def stage_commands(checkpoint, source_dir, output_dir, diagnostic=False):
    commands = [row for row in native_commands(checkpoint=checkpoint, source_dir=source_dir, output_dir=output_dir)
                if row[0] != 'summary']
    commands[0][1][2] = 'experiments.qwen_decoder_lora_fallback.export'
    commands[1][1][2] = 'experiments.qwen_decoder_lora_fallback.audit'
    commands[1][1].extend(['--training_checkpoint_dir', str(checkpoint)])
    return commands[:2] if diagnostic else commands


class JointStepCallback(AuxStepExportCallback):
    def __init__(self, record, **kwargs):
        super().__init__(**kwargs)
        self.record, self.completed_steps = record, []

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == self._last_step: return control
        require(step == [400, 800, 1200, 1600, 2000][len(self.completed_steps)], 'Unexpected formal save order.')
        checkpoint = Path(args.output_dir).resolve() / f'checkpoint-{step}'
        meta, saved, _, source, base, architecture = inspect_step(checkpoint)
        require(saved['global_step'] == step and source == self.source_dir
                and base['checkpoint_id'] == self.record['source_checkpoint_id'], 'Saved source/step differs.')
        check_inputs(self.record)
        output = self.run_root_dir / f'step_{step:04d}'; output.mkdir(exist_ok=False)
        path = output / 'step_export.json'
        record = {'status': 'running', 'global_step': step, 'max_steps': 2000,
                  'source_checkpoint': str(checkpoint), 'selected_checkpoint_id': meta['checkpoint_id'], 'stages': {}}
        started = time.time()
        try:
            query = subprocess.run(['nvidia-smi', '--id=' + self.env['CUDA_VISIBLE_DEVICES'],
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, check=True)
            require(not query.stdout.strip(), 'Export GPU is occupied: ' + query.stdout.strip())
            for name, argv, allowed in stage_commands(checkpoint, source, output):
                record['stages'][name] = {'argv': argv}; write_json(path, record)
                record['stages'][name]['exit_code'] = run_logged_stage(argv, name=name, output_dir=output,
                    env=self.env, allowed_exit_codes=allowed)
            audited = read(output / 'audit.json'); verify_audit(audited, self.record, step)
            final = output / 'final_model'; final_meta = read(final / 'checkpoint_meta.json')
            require(final_meta['checkpoint_id'] == audited['checkpoint_id']
                    and audited['selected_checkpoint_id'] == meta['checkpoint_id'], 'Export/audit UUID differs.')
            paths = [output / (name + '.json') for name in ('mmlu', 'seven')]
            result = collect_results(paths, architecture, checkpoint=(final, final_meta))
            for actual_path, reference_path in zip(paths, self.record['control_result_paths'], strict=True):
                actual, reference = (read(p)['evals']['lm_eval'] for p in (actual_path, reference_path))
                require(all(actual[k] == reference[k] for k in ('configs', 'versions', 'n_shot', 'n_samples')),
                        'Complete evaluation protocol differs from the original control.')
            write_json(output / 'full8_summary.json', {'status': 'full8_evaluation_verified', **result,
                'checkpoint_id': final_meta['checkpoint_id'], 'global_step': step, 'bit_budget': audited['bit_budget'],
                'audit_sha256': sha256(output / 'audit.json'), 'longbench_status': 'not_evaluated',
                'current_nine_item_goal_status': 'not_evaluated', 'goal_acceptance': False})
            record.update(status='evaluated', checkpoint_id=final_meta['checkpoint_id'], mean_percent=result['mean_percent'])
            self.completed_steps.append(step); self._last_step = step
            check_inputs(self.record)
        except BaseException as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            record['elapsed_seconds'] = time.time() - started
            write_json(path, record)
        return control
