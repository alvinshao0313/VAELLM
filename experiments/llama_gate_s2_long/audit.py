"""Actual two20000-stage geometry, protection and complete source-union audit."""
from pathlib import Path

import torch

from experiments.llama_gate_small_rate_pair.common import BASE_ID, DENOMINATOR, read, _require
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state

TOTAL_BITS = 17095180288
CONTROL_ID = 'd9a4d29b-8e6f-4464-9b79-a4ff96bc4d7d'
S4_ID = '13bd608f-c3a4-43d3-b3c0-be269743c544'


def audit(record, native):
    output = Path(record['output_dir']); final = output / 'final_model'; dm = load_v6_meta(str(native))
    cd = Path(record['historical_control_dir']); control = read(cd / 'audit.json')
    sd = Path(record['equal_update_s4_dir']); s4 = read(sd / 'audit.json')
    actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == native.parent, 'Wrong actual native output.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Native formal recipe changed.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.gate_proj')}
    reference = {s['name']: s for s in load_v6_meta(str(cd / 'final_model'))['converted_modules']}
    specs = {s['name']: s for s in dm['converted_modules']}
    _require(len(names) == 32 and set(specs) == set(dm['compressed_targets']) == names
             and dm['compression_categories'] == dm['completed_categories'] == ['gate_proj']
             and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
             and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong complete fresh gate donor.')
    for name, spec in specs.items():
        _require(spec == reference[name] and spec['residual_stages'] == 2 and not spec['transpose'], 'Original K32/S2 geometry differs: ' + name)
    log = (output / 'train.log').read_text()
    for stage in (1, 2):
        _require(log.count(f'[gate_proj.L0-31/stage{stage}] step=20000/20000 ') == 1, 'Missing actual20000-stage completion.')
    meta = load_v6_meta(str(final)); report = read(final / 'assembly_report.json'); source = report['sources']; budget = report['bit_budget']
    _require(report['status'] == control['status'] == s4['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and control['checkpoint_id'] == CONTROL_ID and s4['checkpoint_id'] == S4_ID
             and control['base_checkpoint_id'] == s4['base_checkpoint_id'] == source['base_checkpoint_id'] == BASE_ID
             and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == ['gate_proj']
             and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979, 'Complete saved/fresh union differs.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
             and budget['code_padding_bits'] == 0 and budget['original_linear_weights'] == DENOMINATOR
             and budget['compressed_target_count'] == 224 and budget['components'] == record['expected_budget_components'], 'Actual storage differs.')
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 2979 and sum(1 for s in meta['converted_modules'] for _ in _code_specs(s)) == 448,
             'Require actual2979 saved tensors and448 complete code banks.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {f: tensor_record(state[n + '.weight_rotation.' + f]) for f in ('input_signs', 'output_signs')} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
             and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and act['seqlen'] == cache['seqlen'] == 256 and act['seed'] == cache['seed'] == 31
             and names <= set(act['stats_by_linear']), 'Original HF512x256 calibration differs.')
    cache_record = tensor_tree(cache); stats = tensor_tree(act['stats_by_linear'])
    for proof in (control, s4):
        _require(protection == proof['protection_tensor_records'] and signs == proof['rotation_sign_records']
                 and cache_record == proof['calibration_cache'], 'Actual protection/sign/cache differs from completed controls.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'historical_control_checkpoint_id': CONTROL_ID, 'equal_update_s4_checkpoint_id': S4_ID,
        'bit_budget': budget, 'source_union_save_fresh_tensors': 2979, 'full_code_bank_count': 448, 'gate_code_bank_count': 64,
        'replaced_projection_count': 32, 'preserved_projection_count': 192, 'protection_tensor_records': protection,
        'rotation_sign_records': signs, 'calibration_cache': cache_record, 'calibration_stats': stats,
        'matched_controls_protection_signs_calibration_exact': True, 'calibration_stats_exact_control': stats == control['calibration_stats'],
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json'),
        'cat_runtime_state_sha256': sha256(native / 'cat_runtime_state.pt'),
        'control_audit_sha256': sha256(cd / 'audit.json'), 's4_audit_sha256': sha256(sd / 'audit.json')}
