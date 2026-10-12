"""Check fresh S4 asymmetric donor and the native e767 source-union export."""
from pathlib import Path

import torch

from experiments.llama_gate_small_rate_pair.common import BASE_ID, DENOMINATOR, read, _require
from experiments.llama_gate_s4_width.prepare import width_spec, REFERENCE_ID
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state


def audit(record, native):
    output = Path(record['output_dir']); final = output / 'final_model'; ref = Path(record['reference_dir'])
    dm = load_v6_meta(str(native)); reference = read(ref / 'audit.json')
    actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == native.parent, 'Wrong actual native output.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Native formal recipe changed.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.gate_proj')}
    historical = {s['name']: s for s in load_v6_meta(str(ref / 'final_model'))['converted_modules']}
    specs = {s['name']: s for s in dm['converted_modules']}
    _require(len(names) == 32 and set(specs) == set(dm['compressed_targets']) == names
             and dm['compression_categories'] == dm['completed_categories'] == ['gate_proj']
             and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
             and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong fresh HF gate donor.')
    for name, spec in specs.items():
        _require(spec == width_spec(historical[name], record['decoder_hidden_dim']), 'S4 decoder-width geometry differs: ' + name)
    log = (output / 'train.log').read_text()
    for stage in range(1, 5):
        _require(f'[gate_proj.L0-31/stage{stage}] step=10000/10000 ' in log, 'Missing completed native stage.')
    meta = load_v6_meta(str(final)); report = read(final / 'assembly_report.json')
    source = report['sources']; budget = report['bit_budget']
    _require(report['status'] == reference['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and reference['checkpoint_id'] == REFERENCE_ID and reference['base_checkpoint_id'] == BASE_ID
             and source['base_checkpoint_id'] == BASE_ID and source['donor_checkpoint_id'] == dm['checkpoint_id']
             and source['categories'] == ['gate_proj'] and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared']
             == record['expected_state_tensors'] == 3043, 'Saved/fresh complete source union differs.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == record['expected_user_linear_bits']
             and budget['code_padding_bits'] == 0 and budget['original_linear_weights'] == DENOMINATOR
             and budget['compressed_target_count'] == 224 and budget['components'] == record['expected_budget_components'],
             'Actual complete stored bit budget differs.')
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 3043 and sum(1 for s in meta['converted_modules'] for _ in _code_specs(s)) == 512,
             'Require actual3043 tensors and512 native banks.')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')} for n in sorted(names)}
    signs = {n: {f: tensor_record(state[n + '.weight_rotation.' + f])
                    for f in ('input_signs', 'output_signs')} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
             and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and act['seqlen'] == cache['seqlen'] == 256 and act['seed'] == cache['seed'] == 31
             and names <= set(act['stats_by_linear']), 'Original HF calibration differs.')
    cache_record = tensor_tree(cache); stats = tensor_tree(act['stats_by_linear'])
    _require(protection == reference['protection_tensor_records'] and signs == reference['rotation_sign_records']
             and cache_record == reference['calibration_cache'], 'Protection/sign/calibration differs from original S4.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'bit_budget': budget, 'decoder_hidden_dim': record['decoder_hidden_dim'],
        'source_union_save_fresh_tensors': len(state), 'full_code_bank_count': 512, 'gate_code_bank_count': 128,
        'replaced_projection_count': 32, 'preserved_projection_count': 192,
        'protection_tensor_records': protection, 'rotation_sign_records': signs, 'calibration_cache': cache_record,
        'calibration_stats': stats, 'calibration_stats_exact_historical': stats == reference['calibration_stats'],
        'protection_signs_calibration_exact_historical': True, 'historical_checkpoint_id': REFERENCE_ID,
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json'),
        'cat_runtime_state_sha256': sha256(native / 'cat_runtime_state.pt')}
