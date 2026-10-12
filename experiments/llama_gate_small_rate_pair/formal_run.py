"""Native formal training/full8 and the eventual CPU-only matched comparison."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, BASE_ID, DENOMINATOR, check, read, _require
from experiments.full_rht_gate_pair.common import SHARDS, eval_argv
from experiments.full_rht_gate_pair.compare import score, delta
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state


def audit(record, native):
    output = Path(record['output_dir']); final = output / 'final_model'; dm = load_v6_meta(str(native))
    actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == native.parent, 'Wrong actual donor output.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Native formal recipe changed.')
    source_specs = {s['name']: s for s in load_v6_meta(record['base_dir'])['converted_modules']}
    names = {n for n in source_specs if n.endswith('.gate_proj')}; specs = {s['name']: s for s in dm['converted_modules']}
    _require(len(names) == 32 and set(specs) == set(dm['compressed_targets']) == names
             and dm['compression_categories'] == dm['completed_categories'] == ['gate_proj']
             and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
             and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong complete fresh gate donor.')
    k = record['codebook_bits']
    for name, spec in specs.items():
        expected = deepcopy(source_specs[name])
        for field in ('vq_weights', 'stage_vq_weights'):
            for bank in expected[field]:
                bank['shape'][-1] = (k + 7) // 8; bank['logical_shape'][-1] = k
        for field in ('decoders', 'stage_decoders'):
            for decoder in expected[field]: decoder['in_dim'] = k
        _require(spec == expected, 'Actual geometry differs beyond declared K: ' + name)
    log = (output / 'train.log').read_text()
    for stage in (1, 2):
        _require(f'[gate_proj.L0-31/stage{stage}] step=10000/10000 ' in log, 'Missing completed formal stage.')
    meta = load_v6_meta(str(final)); report = read(final / 'assembly_report.json'); provenance = report['sources']; budget = report['bit_budget']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and provenance['base_checkpoint_id'] == BASE_ID and provenance['donor_checkpoint_id'] == dm['checkpoint_id']
             and provenance['categories'] == ['gate_proj'] and set(provenance['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979,
             'Native source-union and saved/fresh2979 verification failed.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == record['expected_user_linear_bits']
             and budget['packed_linear_bits'] == record['expected_packed_linear_bits']
             and budget['code_padding_bits'] == record['expected_padding_bits']
             and budget['original_linear_weights'] == DENOMINATOR and budget['compressed_target_count'] == 224
             and budget['components'] == record['expected_budget_components'], 'Actual logical/physical budget differs.')
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 2979, 'Wrong saved inventory.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {f: tensor_record(state[n + '.weight_rotation.' + f]) for f in ('input_signs', 'output_signs')} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
             and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and act['seqlen'] == cache['seqlen'] == 256 and act['seed'] == cache['seed'] == 31
             and names <= set(act['stats_by_linear']), 'Actual original512x256 HF calibration differs.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'bit_budget': budget, 'source_union_save_fresh_tensors': 2979,
        'replaced_projection_count': 32, 'preserved_projection_count': 192, 'protection_tensor_records': protection,
        'rotation_sign_records': signs, 'calibration_cache': tensor_tree(cache), 'calibration_stats': tensor_tree(act['stats_by_linear']),
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json'),
        'cat_runtime_state_sha256': sha256(native / 'cat_runtime_state.pt')}


def same_protocol(left, right):
    for a, b in zip(left, right, strict=True):
        for key in ('configs', 'versions', 'n_shot', 'n_samples'):
            _require(a[key] == b[key], 'Full8 raw protocol differs: ' + key)


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1

    def stage(name, argv):
        log = output / (name + '.log'); _require(not log.exists(), 'Do not overwrite an executed stage.')
        record['status'] = name
        with log.open('w') as handle:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            record['stages'][name] = {'argv': argv, 'pid': process.pid, 'started_unix': time.time()}
            write_json(path, record); value = process.wait()
        (output / (name + '.exit')).write_text(str(value) + '\n')
        record['stages'][name].update(exit_code=value, finished_unix=time.time()); write_json(path, record)
        _require(value == 0, f'{name} exited {value}; inspect {log}')

    try:
        _require(record['schema'] == 'llama_gate_small_rate_formal_v1' and record['status'] == 'bound_ready'
                 and record['steps_per_stage'] == 10000 and record['full_evaluation']
                 and record['codebook_bits'] == {'control': 32, 'candidate': 34}[record['arm']]
                 and not record['queued'] and not record['automatic_retry'] and Path.cwd().resolve() == ROOT
                 and Path(record['output_dir']) == output and type(record['gpu']) is int
                 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Require one explicitly bound formal arm.')
        check(record); record.update(worker_pid=os.getpid(), worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat')
            .read_text().rsplit(')', 1)[1].split()[19]), started_unix=time.time(), stages={}); write_json(path, record)
        stage('train', record['training_argv'])
        finals = list((output / 'donor').glob('*/final_model')); _require(len(finals) == 1, 'Expected one native donor.')
        native = finals[0].resolve(); record['donor_checkpoint_dir'] = str(native); write_json(path, record)
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories', '--base', record['base_dir'],
            '--donor', str(native), '--categories', 'gate_proj', '--output', str(output / 'final_model')])
        proof = audit(record, native); write_json(output / 'audit.json', proof)
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(output / 'final_model', output / (name + '.json'), tasks=tasks, batch=batch, python=sys.executable))
        current, raw = score([output / (n + '.json') for n, _, _ in SHARDS], record['architecture'], output / 'final_model')
        base, br = score(record['control_result_paths'], record['architecture'], Path(record['base_dir'])); same_protocol(raw, br)
        _require(base['checkpoint_id'] == BASE_ID, 'Wrong secondary e767 source.')
        summary = {'status': 'complete_full8', **current, 'bit_budget': proof['bit_budget'], 'secondary_vs_e767': delta(current, base),
            'primary_pair_status': 'pending_both_completed', 'comparison_boundary': record['comparison_boundary'],
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated', 'audit_sha256': sha256(output / 'audit.json')}
        write_json(output / 'full8_summary.json', summary); check(record)
        record.update(status='completed', checkpoint_id=current['checkpoint_id'], mean_percent=current['mean_percent'],
                      full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()); raise
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')


def compare(directory):
    directory = Path(directory).resolve(); target = directory / 'comparison.json'
    _require(not target.exists(), 'Do not overwrite the matched comparison.')
    rows = {}
    for arm, k in (('control', 32), ('candidate', 34)):
        output = directory / arm; record = read(output / 'launch.json'); check(record)
        _require(record['status'] == 'completed' and record['exit_code'] == 0 and record['codebook_bits'] == k
                 and all((output / (s + '.exit')).read_text().strip() == '0'
                         for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')), 'Require two natural full completions.')
        result, raw = score([output / (n + '.json') for n, _, _ in SHARDS], record['architecture'], output / 'final_model')
        proof = read(output / 'audit.json'); summary = read(output / 'full8_summary.json')
        _require(result['checkpoint_id'] == proof['checkpoint_id'] == record['checkpoint_id']
                 and result['integer_counts'] == summary['integer_counts'] and result['mean_percent'] == record['mean_percent']
                 and summary['audit_sha256'] == sha256(output / 'audit.json')
                 and record['full8_summary_sha256'] == sha256(output / 'full8_summary.json'), 'Completed raw evidence changed.')
        value = deepcopy(record['normalized_expected'])
        for section in ('cat_args', 'canonical_config'): value[section]['output_dir'] = '<arm>'
        for section in ('vae_args', 'canonical_config'): value[section]['codebook_bits']['by_category']['gate_proj'] = '<K>'
        value['resolved_category_runtime']['gate_proj']['codebook_bits'] = '<K>'
        rows[arm] = (record, proof, result, raw, value)
    c, ca, cs, cr, cn = rows['control']; t, ta, ts, tr, tn = rows['candidate']
    _require(cn == tn, 'Recipes differ beyond K and output.')
    for key in ('source_sha256', 'input_sha256', 'protected_stats', 'base_dir', 'base_checkpoint_id', 'architecture', 'environment', 'protocol'):
        _require(c[key] == t[key], 'Source/protocol mismatch: ' + key)
    for key in ('protection_tensor_records', 'rotation_sign_records', 'calibration_cache'):
        _require(ca[key] == ta[key], 'Actual matched protection/sign/calibration differs: ' + key)
    same_protocol(cr, tr)
    base, _ = score(c['control_result_paths'], c['architecture'], Path(c['base_dir']))
    report = {'status': 'verified_pair', 'primary_comparison': 'fresh_gate_K34_minus_fresh_gate_K32',
        'candidate_minus_control': delta(ts, cs), 'arms': {'control': cs, 'candidate': ts, 'e767': base},
        'secondary_vs_e767': {'control': delta(cs, base), 'candidate': delta(ts, base)},
        'bit_budgets': {'control': ca['bit_budget'], 'candidate': ta['bit_budget']},
        'same_actual_protection_payload_signs_calibration_inputs': True,
        'calibration_stats_records_exact': ca['calibration_stats'] == ta['calibration_stats'],
        'comparison_boundary': c['comparison_boundary'], 'seed_boundary': c['seed_boundary'],
        'evidence_sha256': {str(directory / arm / name): sha256(directory / arm / name) for arm in rows
                           for name in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')},
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    write_json(target, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch-record', type=Path); group.add_argument('--compare-dir', type=Path)
    args = parser.parse_args()
    run(args.launch_record) if args.launch_record else compare(args.compare_dir)
