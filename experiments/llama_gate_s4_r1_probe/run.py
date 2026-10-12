"""One explicitly bound GPU: native training, then independent export and full8."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_gate_s4_r1_probe.common import check, read, require, sha256, write_json
from experiments.qwen3_w25.aux_step_export_callback import stage_commands
from experiments.w25_eval.acceptance import collect_results


def run(path):
    path = Path(path).resolve(); record = read(path); root = Path(record['output_dir'])
    code = 1
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path.cwd() == Path(record['cwd'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and not record['queued'] and not record['automatic_retry']
                and all(os.environ.get(k) == v for k, v in record['environment'].items()),
                'Root must bind one free GPU and the exact S environment.')
        require(not any((root / name).exists() for name in ('migration.json', 'r0', 'r1', 'report.json')),
                'Never retry or overwrite a prior run.')
        check(record)
        if record['validation']['required']:
            proof = Path(record['validation']['report_path'])
            require(record['validation']['report_sha256'] is not None
                    and sha256(proof) == record['validation']['report_sha256'],
                    'Bind the completed real S4/all-gate R1 validation evidence before formal execution.')
            validated = read(proof)
            require(validated['status'] == 'completed_native_validation'
                    and validated['source_checkpoint_id'] == record['source_checkpoint_id']
                    and validated['source_sha256'] == record['source_sha256']
                    and validated['exported_steps'] == [2, 4], 'GPU proof is not for this source and implementation.')
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)

        def stage(name, command, output=root):
            started = time.time()
            with (output / (name + '.log')).open('w') as log:
                result = subprocess.run(command, cwd=record['cwd'], stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            (output / (name + '.exit')).write_text(str(result.returncode) + '\n')
            record['stages'][str(output.relative_to(root) / name)] = {
                'argv': command, 'exit_code': result.returncode, 'elapsed_seconds': time.time() - started}
            write_json(path, record)
            require(result.returncode == 0, name + ' failed; no retry or dependent continuation.')

        prefix = [sys.executable, '-u', '-m']
        stage('migration', [*prefix, 'experiments.llama_gate_s4_r1_probe.migrate', '--launch-record', str(path)])
        summaries = {}
        for arm in record['schedule']['arms']:
            stage(arm + '_training', [*prefix, 'experiments.llama_gate_s4_r1_probe.train',
                                     '--launch-record', str(path), '--arm', arm])
            for step in record['schedule']['export_steps']:
                stage(f'{arm}_export_{step:04d}', [*prefix, 'experiments.llama_gate_s4_r1_probe.export',
                    '--launch-record', str(path), '--arm', arm, '--step', str(step)])
                output = root / arm / f'step_{step:04d}'
                audit = read(output / 'audit.json')
                require(audit['status'] == 'verified', 'Independent export/fresh audit did not pass.')
                if not record['schedule']['full8']:
                    continue
                for name, command, allowed in stage_commands(checkpoint='', source_dir=record['source_checkpoint'],
                                                             output_dir=output):
                    if name in ('mmlu', 'seven'):
                        stage(name, command, output)
                final = output / 'final_model'; meta = read(final / 'checkpoint_meta.json')
                paths = [output / (name + '.json') for name in ('mmlu', 'seven')]
                current = collect_results(paths, record['architecture'], checkpoint=(final, meta))
                for actual, reference in zip(paths, record['control_result_paths'], strict=True):
                    left, right = (read(p)['evals']['lm_eval'] for p in (actual, reference))
                    require(all(left[key] == right[key] for key in ('configs', 'versions', 'n_shot', 'n_samples')),
                            'Full8 protocol differs from the fixed source evaluation.')
                summary = {'status': 'full8_evaluation_verified', 'checkpoint_id': meta['checkpoint_id'],
                    'checkpoint_dir': str(final), 'global_step': step, **current, 'bit_budget': audit['bit_budget'],
                    'source_checkpoint_id': record['source_checkpoint_id'],
                    'control_mean_percent': record['control_mean_percent'],
                    'mean_delta_percentage_points': current['mean_percent'] - record['control_mean_percent'],
                    'longbench_status': 'not_evaluated_root_to_schedule_complete21',
                    'current_nine_item_goal_status': 'not_evaluated', 'goal_acceptance': False,
                    'audit_sha256': sha256(output / 'audit.json')}
                write_json(output / 'full8_summary.json', summary)
                summaries[arm] = {'path': str(output / 'full8_summary.json'),
                    'sha256': sha256(output / 'full8_summary.json'), 'mean_percent': current['mean_percent']}
        training = {arm: read(root / arm / 'training.json') for arm in record['schedule']['arms']}
        matched = None
        if len(training) == 2:
            require(training['r0']['observations']['batches'] == training['r1']['observations']['batches'],
                    'The two actual training prefixes differ.')
            matched = len(training['r0']['observations']['batches'])
        report = {'status': 'completed_native_validation' if record['kind'] == 'validate_s4' else 'completed_paired_full8',
            'source_checkpoint_id': record['source_checkpoint_id'], 'source_sha256': record['source_sha256'],
            'migration_sha256': sha256(root / 'migration.json'), 'exported_steps': record['schedule']['export_steps'],
            'training_sha256': {arm: sha256(root / arm / 'training.json') for arm in training},
            'audit_sha256': {arm: {str(step): sha256(root / arm / f'step_{step:04d}/audit.json')
                                  for step in record['schedule']['export_steps']} for arm in training},
            'matched_training_batches': matched, 'full8': summaries,
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
            'boundary': record['scientific_boundary'], 'elapsed_seconds': time.time() - record['started_unix']}
        check(record); write_json(root / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(root / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (root / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    sys.exit(run(parser.parse_args().launch_record))
