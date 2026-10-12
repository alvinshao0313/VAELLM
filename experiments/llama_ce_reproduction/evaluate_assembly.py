"""Run the original CE50-attention/CE10-MLP assembly and full eight tasks once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.full_rht_gate_pair.compare import delta, score
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.w25_eval.acceptance import validate_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record['source_sha256'])
    check_hashes(record['input_sha256'])
    for row in record['protected_stats']:
        check_stat(row)


def assembly(record):
    value = read(Path(record['output_dir']) / 'final_model/assembly_report.json')
    _require(value['status'] == 'verified'
             and value['serialized_tensors_compared'] == value['reloaded_tensors_compared'] == 2979
             and value['sources']['base_checkpoint_id'] == record['base_checkpoint_id']
             and value['sources']['donor_checkpoint_id'] == record['donor_checkpoint_id']
             and value['sources']['categories'] == ['gate_proj', 'up_proj', 'down_proj']
             and set(value['sources']['replaced_targets']) == set(record['selected_targets'])
             and len(record['selected_targets']) == 96,
             'Native source union must preserve CE50 attention and replace all96 MLP with CE10.')
    return value


def run(path):
    path = Path(path).resolve(); record = read(path); output = Path(record['output_dir']); code = 1
    try:
        _require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path.cwd().resolve() == Path(record['cwd']).resolve()
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and not record['queued'] and not record['automatic_retry'],
                 'Use the explicitly bound public-source invocation without retry.')
        _require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Use the original native evaluation environment.')
        _require(not any((output / name).exists() for name in
                         ('final_model', 'mmlu.json', 'seven.json', 'full8_summary.json', 'dispatcher.exit')),
                 'Do not overwrite existing model or results.')
        check(record)
        for label in ('ce10', 'ce50'):
            review = read(record['terminal_reviews'][label])
            _require(review['status'] == 'verified' and review['global_step'] == review['max_steps'] == 2000
                     and review['training_status_observed']['natural_termination_at2000']
                     and review['checkpoint_id'] == record['controls'][label]['checkpoint_id'],
                     'Require both independently verified natural2000 terminals before assembly.')
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        _require([row['name'] for row in record['stage_commands']] == ['assembly', 'mmlu', 'seven'],
                 'Execute only the original assembly and two complete shards.')
        for row in record['stage_commands']:
            name = row['name']; record['status'] = name
            record['stages'][name] = {'argv': row['argv'], 'started_unix': time.time()}
            write_json(path, record)
            try:
                run_logged_stage(row['argv'], name=name, output_dir=output, env=os.environ.copy())
            finally:
                exit_path = output / (name + '.exit')
                if exit_path.exists():
                    record['stages'][name]['exit_code'] = int(exit_path.read_text())
                record['stages'][name]['finished_unix'] = time.time(); write_json(path, record)
            if name == 'assembly':
                assembly(record)
        proof = assembly(record)
        directory, meta, budget = validate_checkpoint(output / 'final_model', record['architecture'])
        _require(meta['checkpoint_id'] == proof['checkpoint_id'] and budget == proof['bit_budget']
                 and budget['user_linear_bits'] == 17100439552
                 and budget['original_linear_weights'] == 6979321856
                 and budget['user_linear_bpw'] == 17100439552 / 6979321856
                 and budget['budget_passed'], 'Actual saved storage differs from both original CE endpoints.')
        current, raw = score([output / (name + '.json') for name in ('mmlu', 'seven')],
                             record['architecture'], directory)
        controls, differences = {}, {}
        for label, item in record['controls'].items():
            control, rows = score([Path(p) for p in item['result_paths']], record['architecture'],
                                   Path(item['checkpoint_dir']))
            _require(control['checkpoint_id'] == item['checkpoint_id'], 'Control identity changed: ' + label)
            for actual, old in zip(raw, rows, strict=True):
                for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                    _require(actual[key] == old[key], label + ' protocol differs: ' + key)
            controls[label], differences[label] = control, delta(current, control)
        check(record)
        summary = {'status': 'full8_evaluation_verified', **current, 'bit_budget': budget,
            'base_checkpoint_id': record['base_checkpoint_id'], 'donor_checkpoint_id': record['donor_checkpoint_id'],
            'controls': controls, 'candidate_minus_controls': differences,
            'assembly_report_sha256': sha256(output / 'final_model/assembly_report.json'),
            'scope': 'CE50/task100/prompt0 attention and backbone; CE10/task50/prompt.3 gate/up/down. No new updates.',
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'full8_summary.json', summary)
        record.update(status='completed', checkpoint_id=meta['checkpoint_id'], mean_percent=current['mean_percent'],
            full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    sys.exit(run(parser.parse_args().launch_record))
