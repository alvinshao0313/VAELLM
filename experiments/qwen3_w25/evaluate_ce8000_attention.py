"""Evaluate the prepared CE8000-attention/f927-MLP assembly once."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.gate_student_context_evaluation.run import read
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require


def check(record):
    check_hashes(record['source_sha256'])
    check_hashes(record['input_sha256'])
    for row in record['protected_stats']:
        check_stat(row)


def assembled(record):
    output = Path(record['output_dir'])
    value = read(output / 'final_model/assembly_report.json')
    _require(value['status'] == 'verified'
             and value['serialized_tensors_compared'] == value['reloaded_tensors_compared'] == 3423
             and value['sources']['base_checkpoint_id'] == record['base_checkpoint_id']
             and value['sources']['donor_checkpoint_id'] == record['donor_checkpoint_id']
             and value['sources']['categories'] == ['q_proj', 'k_proj', 'v_proj', 'o_proj']
             and set(value['sources']['replaced_targets']) == set(record['selected_targets']),
             'The 144-attention source union does not match native assembly evidence.')
    return value


def run(record_path):
    record_path = Path(record_path).resolve()
    record = read(record_path)
    output = Path(record['output_dir'])
    code = 1
    try:
        _require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path.cwd() == Path(record['cwd'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and not record['queued'] and not record['automatic_retry'],
                 'Use the explicitly bound GPU and bitvae environment without retry.')
        _require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'The prepared runtime environment changed.')
        _require(not any((output / name).exists() for name in
                         ('final_model', 'mmlu.json', 'seven.json', 'full8_summary.json', 'dispatcher.exit')),
                 'Do not overwrite existing evaluation results.')
        check(record)
        record.update(status='running', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={})
        write_json(record_path, record)
        from experiments.qwen3_w25.step_export_callback import run_logged_stage
        _require([row['name'] for row in record['stage_commands']] == ['assembly', 'mmlu', 'seven'],
                 'Expected native assembly followed by the two complete evaluation shards.')
        for row in record['stage_commands']:
            name, command = row['name'], row['argv']
            record.update(status=name)
            record['stages'][name] = {'argv': command, 'started_unix': time.time()}
            write_json(record_path, record)
            try:
                run_logged_stage(command, name=name, output_dir=output,
                                 env=os.environ.copy(), allowed_exit_codes=(0,))
            finally:
                path = output / (name + '.exit')
                if path.exists():
                    record['stages'][name]['exit_code'] = int(path.read_text())
                record['stages'][name]['finished_unix'] = time.time()
                write_json(record_path, record)
            if name == 'assembly':
                assembled(record)
        from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
        assembly = assembled(record)
        model, meta, budget = validate_checkpoint(output / 'final_model', record['architecture'])
        _require(meta['checkpoint_id'] == assembly['checkpoint_id']
                 and budget['user_linear_bits'] == 17065672704
                 and budget['original_linear_weights'] == 6945767424
                 and budget['user_linear_bpw'] == 17065672704 / 6945767424,
                 'The deployed checkpoint identity or actual bit budget changed.')
        paths = [output / (name + '.json') for name in ('mmlu', 'seven')]
        current = collect_results(paths, record['architecture'], checkpoint=(model, meta))
        _require(current['leaf_count'] == 64 and current['sample_count'] == 24742,
                 'Require complete 57-subject MMLU and the seven other tasks.')
        base = Path(record['base_dir'])
        control = collect_results(record['control_result_paths'], record['architecture'],
                                  checkpoint=(base, read(base / 'checkpoint_meta.json')))
        for path, reference in zip(paths, record['control_result_paths'], strict=True):
            actual, old = read(path)['evals']['lm_eval'], read(reference)['evals']['lm_eval']
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(actual[key] == old[key], 'Control protocol differs: ' + key)
        check(record)
        summary = {'status': 'full8_evaluation_verified', 'checkpoint_id': meta['checkpoint_id'],
            'base_checkpoint_id': record['base_checkpoint_id'], 'donor_checkpoint_id': record['donor_checkpoint_id'],
            **current, 'bit_budget': budget, 'control_mean_percent': control['mean_percent'],
            'mean_delta_pp': current['mean_percent'] - control['mean_percent'],
            'task_delta_pp': {k: 100 * (v - control['task_metrics'][k]) for k, v in current['task_metrics'].items()},
            'assembly_report_sha256': sha256(output / 'final_model/assembly_report.json'),
            'scope': 'Only 144 attention projections from CE8000; 108 MLP and backbone from f927; no updates.',
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'full8_summary.json', summary)
        record.update(status='completed', checkpoint_id=meta['checkpoint_id'],
            mean_percent=current['mean_percent'], full8_summary_sha256=sha256(output / 'full8_summary.json'))
        code = 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time())
        write_json(record_path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    sys.exit(run(parser.parse_args().launch_record))
