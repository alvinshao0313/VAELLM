"""Actual encoder-only widening: unchanged deployed shapes, source union and calibration."""
from copy import deepcopy
from pathlib import Path

import torch

from experiments.llama_gate_small_rate_pair.common import BASE_ID, DENOMINATOR, read, _require
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state


def audit(record, native):
    output = Path(record['output_dir']); final = output / 'final_model'; dm = load_v6_meta(str(native))
    control_dir = Path(record['historical_control_dir']); control = read(control_dir / 'audit.json')
    actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == native.parent, 'Wrong actual donor output.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Actual native recipe differs from CPU preparation.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.gate_proj')}
    reference = {s['name']: s for s in load_v6_meta(str(control_dir / 'final_model'))['converted_modules']}
    specs = {s['name']: s for s in dm['converted_modules']}
    _require(len(names) == 32 and set(specs) == set(dm['compressed_targets']) == names
             and dm['compression_categories'] == dm['completed_categories'] == ['gate_proj']
             and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
             and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong complete native donor.')
    for name, spec in specs.items():
        expected = deepcopy(reference[name]); _require(expected['transpose'] is True, 'Control must be transposed.')
        for decoder in expected['decoders'] + expected['stage_decoders']:
            decoder['decoder_type'] = 'asymmetric'
        _require(spec == expected, 'Native deployed spec differs beyond asymmetric metadata: ' + name)
    log = (output / 'train.log').read_text()
    for stage in (1, 2):
        _require(f'[gate_proj.L0-31/stage{stage}] step=10000/10000 ' in log, 'Missing complete formal stage.')
    meta = load_v6_meta(str(final)); report = read(final / 'assembly_report.json'); source = report['sources']; budget = report['bit_budget']
    _require(report['status'] == control['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and control['checkpoint_id'] == record['historical_control_checkpoint_id'] and control['base_checkpoint_id'] == BASE_ID
             and source['base_checkpoint_id'] == BASE_ID and source['donor_checkpoint_id'] == dm['checkpoint_id']
             and source['categories'] == ['gate_proj'] and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979,
             'Native source union and saved/fresh2979 verification differs.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == record['expected_user_linear_bits']
             and budget['packed_linear_bits'] == record['expected_packed_linear_bits']
             and budget['code_padding_bits'] == record['expected_padding_bits']
             and budget['original_linear_weights'] == DENOMINATOR and budget['compressed_target_count'] == 224
             and budget['components'] == record['expected_budget_components'] == control['bit_budget']['components'],
             'Actual logical/physical component ledger differs from K34 True H128 encoder.')
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 2979, 'Saved state inventory differs.')
    bank_keys = [key for spec in specs.values() for key, _ in _code_specs(spec)]
    _require(len(bank_keys) == 64 and all(not bool((state[key][..., -1] & 0xfc).any()) for key in bank_keys),
             'K34 high six padding bits must remain zero.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {f: tensor_record(state[n + '.weight_rotation.' + f]) for f in ('input_signs', 'output_signs')} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
             and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and act['seqlen'] == cache['seqlen'] == 256 and act['seed'] == cache['seed'] == 31
             and names <= set(act['stats_by_linear']), 'Actual HF calibration differs.')
    cache_record = tensor_tree(cache); stats = tensor_tree(act['stats_by_linear'])
    _require(protection == control['protection_tensor_records'] and signs == control['rotation_sign_records']
             and cache_record == control['calibration_cache'], 'Actual protection/sign/calibration inputs differ from K34 True H128 encoder.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'historical_control_checkpoint_id': control['checkpoint_id'], 'bit_budget': budget,
        'source_union_save_fresh_tensors': 2979, 'replaced_projection_count': 32, 'preserved_projection_count': 192,
        'actual_gate_transpose': True, 'actual_encoder_hidden_dim': 256, 'actual_decoder_hidden_dim': 128,
        'actual_decoder_type': 'asymmetric', 'encoder_is_not_deployed': True, 'zero_padding_banks': 64, 'protection_tensor_records': protection,
        'rotation_sign_records': signs, 'calibration_cache': cache_record, 'calibration_stats': stats,
        'matched_control_protection_signs_calibration_exact': True, 'calibration_stats_exact_control': stats == control['calibration_stats'],
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json'),
        'cat_runtime_state_sha256': sha256(native / 'cat_runtime_state.pt'), 'control_audit_sha256': sha256(control_dir / 'audit.json')}
