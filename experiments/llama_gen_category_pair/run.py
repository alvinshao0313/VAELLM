"""Execute one bound native category union and its complete eight-task comparison."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.full_rht_gate_pair.common import read
from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_gen_category_pair.prepare import ARMS, BASE_ID, BITS, DENOMINATOR, GEN_ID, ROOT
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.step_export_callback import run_logged_stage


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for row in record['protected_stats']:
        check_stat(row)


def assembly(record):
    directory = Path(record['output_dir']) / 'final_model'
    value, meta = (read(directory / name) for name in ('assembly_report.json', 'checkpoint_meta.json'))
    selected = sorted(name for name in record['architecture']['projection_shapes']
                      if name.rsplit('.', 1)[1] in ARMS[record['arm']])
    sources, budget = value['sources'], value['bit_budget']
    _require(value['status'] == 'verified' and value['checkpoint_id'] == meta['checkpoint_id']
             and meta['checkpoint_id'] not in (BASE_ID, GEN_ID)
             and value['serialized_tensors_compared'] == value['reloaded_tensors_compared'] == 2979
             and sources['base_checkpoint_id'] == record['base_checkpoint_id'] == BASE_ID
             and sources['donor_checkpoint_id'] == record['donor_checkpoint_id'] == GEN_ID
             and sources['categories'] == record['categories'] == ARMS[record['arm']]
             and sources['replaced_targets'] == record['selected_targets'] == selected
             and record['selected_decoder_tensors'] == 6 * len(selected)
             and record['e767_complement_tensor_count'] == 2979 - 6 * len(selected)
             and not any(sources['selected_spec_differences'].values()),
             'Native source union, metadata, save/fresh or selected category contract differs.')
    _require(budget['budget_passed'] and not budget['contract_violations']
             and not budget['unclassified_target_keys'] and budget['user_linear_bits'] == BITS
             and budget['packed_linear_bits'] == BITS and budget['original_linear_weights'] == DENOMINATOR
             and budget['code_padding_bits'] == 0 and budget['compressed_target_count'] == 224
             and not any(budget['components'].get(k, {}).get('storage_bits', 0)
                         for k in ('lora', 'residual_lora', 'dense_weights')),
             'Native actual storage must remain the original rank0 mixed-D16/D32 budget.')
    for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components'):
        _require(budget[key] == record['source_bit_budget'][key], 'Union storage changed: ' + key)
    return value


def run(path):
    path = Path(path).resolve(); record = read(path); output = Path(record['output_dir']); code = 1
    try:
        _require(record['status'] == 'bound_ready' and record['arm'] in ARMS
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path.cwd() == ROOT and output == path.parent
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and not record['queued'] and not record['automatic_retry'],
                 'Use one explicitly bound public-source bitvae invocation.')
        _require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Use the native pinned evaluation environment.')
        _require(not any((output / name).exists() for name in
                         ('final_model', 'mmlu.json', 'seven.json', 'full8_summary.json', 'dispatcher.exit')),
                 'Do not overwrite any executed arm.')
        check(record)
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        _require([row['name'] for row in record['stage_commands']] == ['assembly', 'mmlu', 'seven'],
                 'Run only native assembly and two complete evaluation shards.')
        for row in record['stage_commands']:
            name = row['name']; record['status'] = name
            record['stages'][name] = {'argv': row['argv'], 'started_unix': time.time()}; write_json(path, record)
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
        current, raw = score([output / (name + '.json') for name in ('mmlu', 'seven')],
                             record['architecture'], output / 'final_model')
        controls, differences = {}, {}
        for label, item in record['controls'].items():
            control, rows = score([Path(p) for p in item['result_paths']], record['architecture'],
                                   Path(item['checkpoint_dir']))
            _require(control['checkpoint_id'] == item['checkpoint_id'], 'Control identity changed: ' + label)
            for actual, old in zip(raw, rows, strict=True):
                _require(all(actual[k] == old[k] for k in ('configs', 'versions', 'n_shot', 'n_samples')),
                         label + ' complete evaluation protocol differs.')
            controls[label], differences[label] = control, delta(current, control)
        check(record)
        summary = {'status': 'full8_evaluation_verified', **current, 'arm': record['arm'],
            'bit_budget': proof['bit_budget'], 'base_checkpoint_id': BASE_ID, 'donor_checkpoint_id': GEN_ID,
            'controls': controls, 'candidate_minus_controls': differences,
            'assembly_report_sha256': sha256(output / 'final_model/assembly_report.json'),
            'frozen_contract_reused': record['validation_reuse'],
            'selected_decoder_tensors': record['selected_decoder_tensors'],
            'e767_complement_tensor_count': record['e767_complement_tensor_count'],
            'comparison_boundary': record['comparison_boundary'],
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'full8_summary.json', summary)
        record.update(status='completed', checkpoint_id=current['checkpoint_id'], mean_percent=current['mean_percent'],
            mmlu_correct=current['integer_counts']['mmlu']['correct'],
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
