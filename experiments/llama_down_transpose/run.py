"""Native nontransposed D16 down run and matched complete eight-task comparison."""
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
from experiments.llama_down_transpose.audit import check_assembly, check_donor, results
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
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve(),
                 'Require one explicitly bound snapshot invocation.')
        _require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Use the pinned environment and ordinary numerical policy.')
        check(record)
        record.update(worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={}); write_json(path, record)
        stage('train', record['training_argv'])
        finals = list((output / 'donor').glob('*/final_model'))
        _require(len(finals) == 1, 'Expected one actual native down donor.')
        donor = finals[0].resolve(); donor_meta, _ = check_donor(donor, record)
        record.update(donor_checkpoint_dir=str(donor), donor_checkpoint_id=donor_meta['checkpoint_id'])
        write_json(path, record)
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories',
            '--base', record['base_dir'], '--donor', str(donor), '--categories', 'down_proj',
            '--output', str(output / 'final_model')])
        audit = check_assembly(output / 'final_model', donor, record); write_json(output / 'audit.json', audit)
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(output / 'final_model', output / (name + '.json'),
                                 tasks=tasks, batch=batch, python=sys.executable))
        summary = results(output, record)
        current, raw = score([output / (name + '.json') for name, _, _ in SHARDS],
                             record['architecture'], output / 'final_model')
        control, old_raw = score([Path(p) for p in record['historical_control_result_paths']],
                                 record['architecture'], Path(record['historical_control_dir']) / 'final_model')
        _require(control['checkpoint_id'] == record['historical_control_checkpoint_id'],
                 'Matched transposed D16 control names a different checkpoint.')
        for left, right in zip(raw, old_raw, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(left[key] == right[key], 'Historical control evaluation protocol differs: ' + key)
        summary.update(integer_counts=current['integer_counts'],
            user_linear_bpw=audit['bit_budget']['user_linear_bpw'],
            nontranspose_vs_transpose=delta(current, control),
            historical_control_checkpoint_id=control['checkpoint_id'],
            transpose_pair_comparison='complete')
        summary.pop('rotation_pair_comparison', None)
        write_json(output / 'full8_summary.json', summary); check(record)
        record.update(status='completed', checkpoint_id=current['checkpoint_id'],
            mean_percent=summary['mean_percent'], mmlu_correct=summary['mmlu_correct'],
            full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    args = parser.parse_args()
    run(args.launch_record)
