"""Keep K40 gate/down, transplant FineWeb800 attention/up, and evaluate full8 once."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_ce_reproduction.evaluate_assembly import check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.w25_eval.acceptance import validate_checkpoint


def assembly(record):
    proof = read(Path(record['output_dir']) / 'final_model/assembly_report.json')
    _require(proof['status'] == 'verified'
             and proof['serialized_tensors_compared'] == proof['reloaded_tensors_compared'] == 2979
             and proof['sources']['base_checkpoint_id'] == record['base_checkpoint_id']
             and proof['sources']['donor_checkpoint_id'] == record['donor_checkpoint_id']
             and proof['sources']['categories'] == record['categories']
             == ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj']
             and set(proof['sources']['replaced_targets']) == set(record['selected_targets'])
             and len(record['selected_targets']) == 160 and record['preserved_projection_count'] == 64
             and proof['bit_budget']['user_linear_bits'] == record['expected_user_linear_bits'] == 17102536704
             and proof['bit_budget']['components'] == record['expected_budget_components'],
             'Require the native exact union:160 FineWeb attention/up,64 K40 gate/down, unchanged backbone and K40 storage.')
    return proof


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
                 'Do not overwrite a model or an evaluation.')
        check(record)
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        _require([row['name'] for row in record['stage_commands']] == ['assembly', 'mmlu', 'seven'],
                 'Only native assembly and the two complete evaluation shards are defined.')
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
                 and budget['user_linear_bits'] == record['expected_user_linear_bits']
                 and budget['original_linear_weights'] == record['architecture']['original_linear_weights']
                 and budget['components'] == record['expected_budget_components'] and budget['code_padding_bits'] == 0
                 and budget['components']['decoder']['dtypes'] == ['bfloat16']
                 and not budget['contract_violations'] and budget['budget_passed'] and budget['user_linear_bpw'] <= 2.5,
                 'Actual assembled tensor storage differs from the selected source union.')
        current, raw = score([output / (name + '.json') for name in ('mmlu', 'seven')],
                             record['architecture'], directory)
        _require(set(record['controls']) == {'k40', 'fineweb800', 'fineweb800_down_d16'}
                 and record['primary_control'] == 'k40', 'Use the three prepared controls and K40 primary comparison.')
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
            'controls': controls, 'candidate_minus_controls': differences, 'primary_control': 'k40',
            'selected_categories': record['categories'], 'replaced_projection_count': 160, 'preserved_projection_count': 64,
            'all_budget_components_match_k40': True, 'rank_zero': True, 'persistent_decoder_dtype': 'bfloat16',
            'assembly_report_sha256': sha256(output / 'final_model/assembly_report.json'),
            'scope': record['scientific_difference'], 'longbench_status': 'not_evaluated',
            'nine_item_goal_status': 'not_evaluated', 'additional_training_steps': 0}
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
