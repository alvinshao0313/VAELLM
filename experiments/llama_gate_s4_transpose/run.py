"""One native transposed S4 donor, source union, and complete matched full8."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.llama_gate_s4_transpose.prepare import ROOT, CONTROL_ID, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_gate_small_rate_pair.formal_run import same_protocol
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.full_rht_gate_pair.common import SHARDS, eval_argv
from experiments.full_rht_gate_pair.compare import score, delta
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state
from train_utils.checkpoint_bpw import _code_specs


def audit(record, native):
    output = Path(record['output_dir']); final = output / 'final_model'; cd = Path(record['historical_control_dir'])
    control = read(cd / 'audit.json'); dm = load_v6_meta(str(native)); actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == native.parent, 'Wrong actual donor output.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Actual native recipe differs.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.gate_proj')}
    refs = {s['name']: s for s in load_v6_meta(str(cd / 'final_model'))['converted_modules']}
    specs = {s['name']: s for s in dm['converted_modules']}
    _require(len(names) == 32 and set(specs) == set(dm['compressed_targets']) == names
             and dm['completed_categories'] == dm['compression_categories'] == ['gate_proj']
             and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
             and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong fresh native donor.')
    for name, spec in specs.items():
        expected = deepcopy(refs[name]); _require(expected['transpose'] is False, 'Control must be False.')
        expected['transpose'] = True
        _require(spec == expected, 'Only the native gate transpose may change: ' + name)
    log = (output / 'train.log').read_text()
    _require(all(f'[gate_proj.L0-31/stage{s}] step=10000/10000 ' in log for s in range(1, 5)), 'Missing full stages.')
    meta = load_v6_meta(str(final)); report = read(final / 'assembly_report.json'); source = report['sources']; budget = report['bit_budget']
    _require(report['status'] == control['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and control['checkpoint_id'] == CONTROL_ID and source['base_checkpoint_id'] == BASE_ID
             and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == ['gate_proj']
             and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 3043,
             'Native3043 source-union/save/fresh evidence differs.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == record['expected_user_linear_bits'] == 17099800576
             and budget['code_padding_bits'] == 0 and budget['original_linear_weights'] == DENOMINATOR
             and budget['compressed_target_count'] == 224 and budget['components'] == control['bit_budget']['components']
             == record['expected_budget_components'], 'Actual S4 storage differs from control.')
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 3043 and sum(1 for s in meta['converted_modules'] for _ in _code_specs(s)) == 512, 'Wrong bank inventory.')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')} for n in sorted(names)}
    signs = {n: {f: tensor_record(state[n + '.weight_rotation.' + f]) for f in ('input_signs', 'output_signs')} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
             and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and act['seqlen'] == cache['seqlen'] == 256 and act['seed'] == cache['seed'] == 31, 'Calibration changed.')
    cache_record, stats = tensor_tree(cache), tensor_tree(act['stats_by_linear'])
    _require(protection == control['protection_tensor_records'] and signs == control['rotation_sign_records']
             and cache_record == control['calibration_cache'], 'Protection/sign/calibration inputs differ.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'historical_control_checkpoint_id': CONTROL_ID, 'bit_budget': budget,
        'source_union_save_fresh_tensors': 3043, 'full_code_bank_count': 512, 'gate_code_bank_count': 128,
        'replaced_projection_count': 32, 'preserved_projection_count': 192, 'actual_gate_transpose': True,
        'protection_tensor_records': protection, 'rotation_sign_records': signs, 'calibration_cache': cache_record,
        'calibration_stats': stats, 'matched_control_protection_signs_calibration_exact': True,
        'calibration_stats_exact_control': stats == control['calibration_stats'],
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json')}


def run(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1
    def stage(name, argv):
        record['stages'][name] = {'argv': argv, 'started_unix': time.time()}; write_json(path, record)
        try:
            run_logged_stage(argv, name=name, output_dir=output, env=os.environ.copy())
        finally:
            p = output / (name + '.exit')
            if p.exists(): record['stages'][name]['exit_code'] = int(p.read_text())
            record['stages'][name]['finished_unix'] = time.time(); write_json(path, record)
    try:
        _require(record['schema'] == 'llama_gate_s4_transpose_v1' and record['status'] == 'bound_ready'
                 and record['transpose'] is True and record['residual_stages'] == 4 and record['codebook_bits'] == 16
                 and record['steps_per_stage'] == 10000 and record['total_stage_updates'] == 40000
                 and record['full_evaluation'] and not record['queued'] and not record['automatic_retry']
                 and Path.cwd() == ROOT and Path(record['output_dir']) == output
                 and type(record['gpu']) is int and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and all(os.environ.get(k) == v for k, v in record['environment'].items()), 'Bind one GPU and the unchanged S environment.')
        _require(not any((output / n).exists() for n in ('donor', 'final_model', 'train.log', 'mmlu.json', 'seven.json')), 'Fresh output only.')
        check(record); record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), stages={},
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])); write_json(path, record)
        stage('train', record['training_argv'])
        finals = list((output / 'donor').glob('*/final_model')); _require(len(finals) == 1, 'Expected one donor.')
        native = finals[0].resolve(); record['donor_checkpoint_dir'] = str(native); write_json(path, record)
        stage('assembly', [sys.executable, '-m', 'experiments.w25_eval.assemble_categories', '--base', record['base_dir'],
            '--donor', str(native), '--categories', 'gate_proj', '--output', str(output / 'final_model')])
        proof = audit(record, native); write_json(output / 'audit.json', proof)
        for name, tasks, batch in SHARDS:
            stage(name, eval_argv(output / 'final_model', output / (name + '.json'), tasks=tasks, batch=batch, python=sys.executable))
        current, raw = score([output / (n + '.json') for n, _, _ in SHARDS], record['architecture'], output / 'final_model')
        cd = Path(record['historical_control_dir'])
        control, cr = score([cd / (n + '.json') for n, _, _ in SHARDS], record['architecture'], cd / 'final_model')
        base, br = score(record['control_result_paths'], record['architecture'], Path(record['base_dir']))
        _require(control['checkpoint_id'] == CONTROL_ID and base['checkpoint_id'] == BASE_ID, 'Wrong controls.')
        same_protocol(raw, cr); same_protocol(raw, br); check(record)
        write_json(output / 'full8_summary.json', {'status': 'complete_full8', **current, 'bit_budget': proof['bit_budget'],
            'primary_vs_s4_false': delta(current, control), 'secondary_vs_e767': delta(current, base),
            'historical_control_checkpoint_id': CONTROL_ID, 'comparison_boundary': record['comparison_boundary'],
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated', 'audit_sha256': sha256(output / 'audit.json')})
        record.update(status='completed', checkpoint_id=current['checkpoint_id'], mean_percent=current['mean_percent'],
            full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch-record', required=True, type=Path)
    sys.exit(run(parser.parse_args().launch_record))
