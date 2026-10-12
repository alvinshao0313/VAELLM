"""Run one explicitly bound Llama Q/K arm, or CPU-compare two completed arms."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.full_rht_gate_pair.common import ROOT, SHARDS, check, eval_argv, read
from experiments.full_rht_gate_pair.compare import delta, normalized_recipe, score
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.llama_qk_rotation_pair.audit import check_assembly, check_donor
from experiments.full_rht_gate_pair.audit import results
from train_utils.checkpoint_v6 import load_v6_meta


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
                 'Use the pinned offline environment and ordinary numerical policy.')
        check(record)
        record.update(worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={}); write_json(path, record)
        stage('train', record['training_argv'])
        finals = list((output / 'donor').glob('*/final_model'))
        _require(len(finals) == 1, 'Expected one final donor with both fresh Q/K categories.')
        donor = finals[0].resolve(); donor_meta, _ = check_donor(donor, record)
        record.update(donor_checkpoint_dir=str(donor), donor_checkpoint_id=donor_meta['checkpoint_id'])
        write_json(path, record)
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories',
            '--base', record['base_dir'], '--donor', str(donor), '--categories', 'q_proj,k_proj',
            '--output', str(output / 'final_model')])
        audit = check_assembly(output / 'final_model', donor, record); write_json(output / 'audit.json', audit)
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(output / 'final_model', output / (name + '.json'),
                                 tasks=tasks, batch=batch, python=sys.executable))
        summary = results(output, record)
        summary['user_linear_bpw'] = audit['bit_budget']['user_linear_bpw']
        write_json(output / 'full8_summary.json', summary); check(record)
        record.update(status='completed', checkpoint_id=summary['checkpoint_id'],
            mean_percent=summary['mean_percent'], mmlu_correct=summary['mmlu_correct'],
            full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


def completed_arm(directory, expected_arm):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json')
    _require(record['status'] == 'completed' and record['exit_code'] == 0
             and record['arm'] == expected_arm and Path(record['output_dir']).resolve() == directory,
             'Require the correct completed paired arm.')
    check(record)
    for stage in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'):
        _require((directory / (stage + '.exit')).read_text().strip() == '0', 'Incomplete stage: ' + stage)
        if stage in record['stages']:
            _require(record['stages'][stage]['exit_code'] == 0, 'Failed recorded stage: ' + stage)
    donor = Path(record['donor_checkpoint_dir']); donor_meta, names = check_donor(donor, record)
    checkpoint = directory / 'final_model'; meta = load_v6_meta(str(checkpoint))
    audit = read(directory / 'audit.json'); assembly = read(checkpoint / 'assembly_report.json')
    _require(audit['status'] == assembly['status'] == 'verified'
             and audit['checkpoint_id'] == assembly['checkpoint_id'] == meta['checkpoint_id'] == record['checkpoint_id']
             and audit['donor_checkpoint_id'] == record['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and audit['base_checkpoint_id'] == record['base_checkpoint_id']
             and audit['source_union_save_fresh_tensors'] == assembly['serialized_tensors_compared']
             == assembly['reloaded_tensors_compared'] == 2979
             and audit['replaced_projection_count'] == 64 and audit['preserved_projection_count'] == 160
             and set(audit['protection_tensor_records']) == set(audit['rotation_sign_records']) == names
             and audit['bit_budget'] == assembly['bit_budget']
             and audit['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
             and audit['actual_native_normalized_sha256'] == sha256(donor.parent / 'normalized_cat_runtime_args.json'),
             'Saved native source-union audit or identity changed.')
    paths = [directory / (name + '.json') for name, _, _ in SHARDS]
    result, raw = score(paths, record['architecture'], checkpoint)
    summary = read(directory / 'full8_summary.json')
    _require(sha256(directory / 'full8_summary.json') == record['full8_summary_sha256']
             and summary['task_metrics'] == result['task_metrics']
             and summary['mean_percent'] == result['mean_percent'] == record['mean_percent']
             and summary['mmlu_correct'] == record['mmlu_correct'] == result['integer_counts']['mmlu']['correct'],
             'Raw scores disagree with the completed summary.')
    evidence = [directory / name for name in ('launch.json', 'audit.json', 'full8_summary.json',
        'train.log', 'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
    evidence += [*paths, checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json',
                 donor / 'checkpoint_meta.json', donor.parent / 'normalized_cat_runtime_args.json']
    return record, audit, result, raw, {str(p): sha256(p) for p in evidence}


def compare(block32, full, output):
    output = Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(),
             'Pair comparison is CPU-only and must not overwrite a result.')
    left, right = completed_arm(block32, 'block32'), completed_arm(full, 'full')
    a, aa, ar, araw, ae = left; b, ba, br, braw, be = right
    _require(normalized_recipe(a) == normalized_recipe(b)
             and a['base_dir'] == b['base_dir'] and a['base_checkpoint_id'] == b['base_checkpoint_id']
             and a['architecture'] == b['architecture'] and a['category_rng'] == b['category_rng']
             and a['source_sha256'] == b['source_sha256'] and a['input_sha256'] == b['input_sha256']
             and a['protected_stats'] == b['protected_stats'], 'Paired recipe, source or calibration differs.')
    for key in ('protection_tensor_records', 'rotation_sign_records', 'calibration_cache'):
        _require(aa[key] == ba[key], 'Actual Q/K pair differs: ' + key)
    for audit in (aa, ba):
        budget = audit['bit_budget']
        _require(budget['budget_passed'] and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17095180288
                 and budget['original_linear_weights'] == 6979321856
                 and budget['components'] == a['expected_budget_components'], 'Actual paired storage differs.')
    base_paths = [Path(p) for p in a['control_result_paths']]
    base, base_raw = score(base_paths, a['architecture'], Path(a['base_dir']))
    for rows in (araw, braw):
        for (_, _, batch), actual, original in zip(SHARDS, rows, base_raw, strict=True):
            _require(actual['batch_size'] == batch and actual['num_fewshot'] == 0 and actual['limit'] is None,
                     'Require actual64/32, zero-shot, complete tasks.')
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(actual[key] == original[key], 'Full8 protocol differs: ' + key)
    evidence = {**ae, **be, **{str(p): sha256(p) for p in [*base_paths, Path(a['base_dir']) / 'checkpoint_meta.json']}}
    report = {'status': 'verified_pair', 'model_type': 'llama', 'categories': ['q_proj', 'k_proj'],
        'primary_comparison': 'full_minus_new_fresh_block32', 'full_vs_block32': delta(br, ar),
        'secondary_vs_e767': {'block32': delta(ar, base), 'full': delta(br, base)},
        'arms': {'block32': ar, 'full': br, 'base': base}, 'bit_budget': aa['bit_budget'],
        'pair_checks': {'same_recipe_except_output_and_block': True, 'same_category_rng': True,
            'q_and_k_two_complete_10000_stages': True, 'protection_records_exact': True,
            'rotation_sign_records_exact': True, 'native_union_save_fresh_tensors': 2979,
            'same_actual_calibration_inputs': True, 'calibration_stats_exact_diagnostic': aa['calibration_stats'] == ba['calibration_stats'],
            'leaf_count': 64, 'sample_count': 24742, 'mmlu_subjects': 57, 'mmlu_samples': 14042},
        'evidence_sha256': evidence, 'comparison_source_sha256': {str(Path(__file__).resolve()): sha256(__file__)},
        'quality_boundary': 'Negative full8 is a valid result, not a protocol failure or a gate to skip either arm retrieval.',
        'future_retrieval': a['future_retrieval'],
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    write_json(output, report)
    print({'status': report['status'], 'full_minus_block32_pp': report['full_vs_block32']['mean_delta_pp'],
           'comparison_sha256': sha256(output)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch-record', type=Path)
    mode.add_argument('--compare', type=Path, nargs=2, metavar=('BLOCK32', 'FULL'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.compare:
        parser.error('--compare requires --output') if args.output is None else compare(*args.compare, args.output)
    else:
        parser.error('--output belongs to --compare') if args.output is not None else run(args.launch_record)
