"""Execute one explicitly bound donor, or an explicitly bound final composition."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.full_rht_gate_pair.common import ROOT, SHARDS, check, eval_argv, read
from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_gate_down_balanced_bits.audit import check_assembly, donor_audit
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1

    def stage(name, argv):
        record['status'] = name; log = output / (name + '.log')
        _require(not log.exists(), 'Do not overwrite an executed stage.')
        with log.open('w') as handle:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': process.pid, 'started_unix': time.time()}
            write_json(path, record); value = process.wait()
        (output / (name + '.exit')).write_text(str(value) + '\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(path, record)
        _require(value == 0, f'{name} exited {value}; inspect {log}')

    try:
        _require(record['status'] == 'bound_ready' and not record['queued'] and not record['automatic_retry']
                 and Path.cwd().resolve() == ROOT and str(ROOT) == os.environ.get('PYTHONPATH')
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Use one explicitly bound snapshot invocation and its actual environment.')
        check(record)
        record.update(worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={}); write_json(path, record)
        if record['phase'] == 'donor':
            stage('train', record['training_argv'])
            finals = list((output / 'donor').glob('*/final_model'))
            _require(len(finals) == 1, 'Expected exactly one native independent donor.')
            donor = finals[0].resolve(); audit = donor_audit(donor, record)
            write_json(output / 'audit.json', audit)
            check(record)
            record.update(status='donor_completed', donor_checkpoint_dir=str(donor),
                donor_checkpoint_id=audit['checkpoint_id'], audit_sha256=sha256(output / 'audit.json'),
                full8_evaluated=False, next_phase_started=False)
        else:
            _require(record['phase'] == 'compose' and record['assembly_order'] == ['down_proj', 'gate_proj'],
                     'Only the explicitly prepared final composition is supported.')
            base = record['base_dir']
            for category, destination in (('down_proj', output / 'down_base'), ('gate_proj', output / 'final_model')):
                stage('assemble_' + category, [sys.executable, '-m', 'experiments.w25_eval.assemble_categories',
                    '--base', str(base), '--donor', record['donors'][category]['checkpoint_dir'],
                    '--categories', category, '--output', str(destination)])
                base = str(destination)
            audit = check_assembly(output, record); write_json(output / 'audit.json', audit)
            final = output / 'final_model'
            for name, tasks, batch in SHARDS:
                stage(name, eval_argv(final, output / (name + '.json'), tasks=tasks, batch=batch,
                                     python=sys.executable))
            current, raw = score([output / (name + '.json') for name, _, _ in SHARDS], record['architecture'], final)
            differences = {}
            for label, directory, paths in (
                    ('matched_gate32_down48', Path(record['historical_control_dir']) / 'final_model',
                     record['historical_control_result_paths']),
                    ('original3577', Path(record['base_dir']), record['control_result_paths'])):
                control, shards = score([Path(p) for p in paths], record['architecture'], directory)
                for left, right in zip(raw, shards, strict=True):
                    for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                        _require(left[key] == right[key], 'Control evaluation protocol differs: ' + key)
                differences[label] = delta(current, control)
            summary = {'status': 'full8_evaluation_verified', **current,
                'bit_budget': audit['bit_budget'], 'primary_control': 'matched_gate32_down48',
                'deltas': differences, 'audit_sha256': sha256(output / 'audit.json'),
                'comparison_boundary': record['comparison_boundary'],
                'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
            write_json(output / 'full8_summary.json', summary); check(record)
            record.update(status='completed', checkpoint_id=current['checkpoint_id'],
                mean_percent=current['mean_percent'], mmlu_correct=current['integer_counts']['mmlu']['correct'],
                full8_summary_sha256=sha256(output / 'full8_summary.json'))
        code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    run(parser.parse_args().launch_record)
