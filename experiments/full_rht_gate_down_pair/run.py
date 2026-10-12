"""Assemble the bound gate/down categories and evaluate all eight tasks."""
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
from experiments.full_rht_gate_down_pair.prepare import BUDGET_FIELDS
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.checkpoint_v6 import load_v6_meta


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1

    def stage(name, argv):
        log = output / (name + '.log')
        _require(not log.exists(), 'Do not overwrite an executed stage.')
        record['status'] = name
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
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path.cwd() == ROOT and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Use one explicitly bound snapshot invocation and its actual environment.')
        check(record)
        record.update(worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={}); write_json(path, record)
        final = output / 'final_model'
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories',
            '--base', record['base_dir'], '--donor', record['donor_dir'],
            '--categories', 'gate_proj', '--output', str(final)])
        meta = load_v6_meta(str(final)); assembly = read(final / 'assembly_report.json')
        sources = assembly['sources']; budget = assembly['bit_budget']
        gates = {name for name in record['architecture']['projection_shapes'] if name.endswith('.gate_proj')}
        count = 3423 if record['model_type'] == 'qwen3' else 2979
        _require(assembly['status'] == 'verified' and assembly['checkpoint_id'] == meta['checkpoint_id']
                 and sources['base_checkpoint_id'] == record['base_checkpoint_id']
                 and sources['donor_checkpoint_id'] == record['donor_checkpoint_id']
                 and sources['categories'] == ['gate_proj'] and set(sources['replaced_targets']) == gates
                 and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared'] == count,
                 'Native source union, save or strict reload differs from the bound composition.')
        _require(all(budget[key] == record['source_bit_budget'][key] for key in BUDGET_FIELDS)
                 and budget['budget_passed']
                 and not budget['contract_violations'] and not budget['unclassified_target_keys']
                 and budget['user_linear_bits'] == record['expected_user_linear_bits'],
                 'Actual composition storage differs from its source budget.')
        changed = [s for s in meta['converted_modules'] if s['name'].endswith(('.gate_proj', '.down_proj'))]
        _require(len(changed) == 2 * record['architecture']['num_hidden_layers']
                 and all(s['weight_rotation']['block_size'] == record['block_size'] for s in changed),
                 'The complete gate/down categories do not use the intended common rotation.')
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(final, output / (name + '.json'), tasks=tasks, batch=batch, python=sys.executable))
        current, raw = score([output / (name + '.json') for name, _, _ in SHARDS], record['architecture'], final)
        references = {'original': (record['original_base_dir'], record['control_result_paths']),
                      'gate': (record['donor_dir'], record['category_result_paths']['gate']),
                      'down': (record['base_dir'], record['category_result_paths']['down'])}
        differences = {}
        for label, (checkpoint, paths) in references.items():
            reference, shards = score([Path(p) for p in paths], record['architecture'], Path(checkpoint))
            for left, right in zip(raw, shards, strict=True):
                for field in ('configs', 'versions', 'n_shot', 'n_samples'):
                    _require(left[field] == right[field], 'Reference evaluation protocol differs: ' + field)
            differences[label] = delta(current, reference)
        summary = {'status': 'full8_evaluation_verified', **current, 'bit_budget': budget,
            'secondary_deltas': differences, 'source_union_save_fresh_tensors': count,
            'assembly_report_sha256': sha256(final / 'assembly_report.json'),
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'full8_summary.json', summary); check(record)
        record.update(status='completed', checkpoint_id=meta['checkpoint_id'], mean_percent=current['mean_percent'],
            mmlu_correct=current['integer_counts']['mmlu']['correct'],
            full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
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
