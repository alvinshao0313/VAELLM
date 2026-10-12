"""One fresh transposed-gate BSQ run with native assembly and matched complete full8."""
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
from experiments.llama_gate_transpose.audit import check_assembly, check_donor, results
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1

    def stage(name, argv):
        record['status'] = name; log = output / (name + '.log')
        _require(not log.exists(), 'Do not overwrite an executed stage.')
        with log.open('x') as handle:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': process.pid, 'started_unix': time.time()}
            write_json(path, record); value = process.wait()
        (output / (name + '.exit')).write_text(str(value) + '\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(path, record)
        _require(value == 0, f'{name} exited {value}; inspect {log}')

    try:
        _require(record['status'] == 'bound_ready' and not record['queued'] and not record['automatic_retry']
                 and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and Path(record['output_dir']).resolve() == output
                 and record['prepared_sha256'] == sha256(output / 'launch.prepared.json'),
                 'Require the explicitly bound prepared snapshot invocation.')
        _require(all(os.environ.get(key) == value for key, value in record['environment'].items())
                 and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Use the pinned native environment.')
        check(record)
        record.update(worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={}); write_json(path, record)
        stage('train', record['training_argv'])
        finals = list((output / 'donor').glob('*/final_model'))
        _require(len(finals) == 1, 'Require one actual fresh transposed gate donor.')
        donor = finals[0].resolve(); donor_meta, _ = check_donor(donor, record)
        record.update(donor_checkpoint_dir=str(donor), donor_checkpoint_id=donor_meta['checkpoint_id']); write_json(path, record)
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories',
            '--base', record['base_dir'], '--donor', str(donor), '--categories', 'gate_proj',
            '--output', str(output / 'final_model')])
        audit = check_assembly(output / 'final_model', donor, record)
        primary_audit = read(record['primary_control_audit_path'])
        _require(primary_audit['status'] == 'verified'
                 and primary_audit['checkpoint_id'] == record['primary_control_checkpoint_id'], 'Primary gateTrue identity differs.')
        for key in ('protection_tensor_records', 'rotation_sign_records'):
            _require(audit[key] == primary_audit[key], 'The coefficient bundle changed primary protection/signs: ' + key)
        for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components'):
            _require(audit['bit_budget'][key] == primary_audit['bit_budget'][key], 'Primary persistent storage differs: ' + key)
        audit.update(primary_control_checkpoint_id=primary_audit['checkpoint_id'],
            primary_control_audit_sha256=sha256(record['primary_control_audit_path']),
            protection_and_signs_exact_primary_gate_true=True, bsq_control=record['bsq_control'], bsq_candidate=record['bsq_candidate'])
        write_json(output / 'audit.json', audit)
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(output / 'final_model', output / (name + '.json'),
                                 tasks=tasks, batch=batch, python=sys.executable))
        summary = results(output, record)
        current, raw = score([output / (name + '.json') for name, _, _ in SHARDS],
                             record['architecture'], output / 'final_model')
        control, control_raw = score([Path(p) for p in record['primary_control_result_paths']],
            record['architecture'], Path(record['primary_control_dir']) / 'final_model')
        _require(control['checkpoint_id'] == record['primary_control_checkpoint_id'], 'Primary gateTrue full8 identity differs.')
        for left, right in zip(raw, control_raw, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(left[key] == right[key], 'Matched gateTrue task protocol differs: ' + key)
        summary.update(integer_counts=current['integer_counts'], user_linear_bpw=audit['bit_budget']['user_linear_bpw'],
            bsq_recipe_on_transpose_true_vs_gate_true=delta(current, control),
            primary_control_checkpoint_id=control['checkpoint_id'], bsq_recipe_comparison='complete',
            bsq_control=record['bsq_control'], bsq_candidate=record['bsq_candidate'],
            comparison_boundary=record['comparison_boundary'])
        summary.pop('rotation_pair_comparison', None)
        write_json(output / 'full8_summary.json', summary); check(record)
        record.update(status='completed', checkpoint_id=current['checkpoint_id'], mean_percent=summary['mean_percent'],
            mmlu_correct=summary['mmlu_correct'], full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    run(parser.parse_args().launch_record)
