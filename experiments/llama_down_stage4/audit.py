"""Native K12/D16/S4 down geometry, padding and e767 source-union proof."""
from copy import deepcopy
from pathlib import Path

import torch

from experiments.llama_gate_small_rate_pair.common import BASE_ID, DENOMINATOR, read, _require
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state

TOTAL_BITS = 17097687040
PACKED_BITS = 18943180800
PADDING_BITS = 1845493760
DOWN_ID = '8a4633a7-93bd-429a-8a0e-a519a2ae30ae'


def stage4_spec(source):
    result = deepcopy(source)
    _require(source['residual_stages'] == 2 and source['stage_codebook_dims'] == [16, 16]
             and source['transpose'] and source['stage_decoders'][0]['in_dim'] == 24,
             'Require the retained original K24/D16/S2 transposeTrue down geometry.')
    bank = deepcopy(source['stage_vq_weights'][0]); bank['shape'][-1] = 2; bank['logical_shape'][-1] = 12
    decoder = deepcopy(source['stage_decoders'][0]); decoder['in_dim'] = 12
    result.update(residual_stages=4, stage_codebook_dims=[16] * 4,
        vq_weights=[deepcopy(bank)], decoders=[deepcopy(decoder)],
        stage_vq_weights=[deepcopy(bank) for _ in range(4)], stage_decoders=[deepcopy(decoder) for _ in range(4)])
    return result


def audit(record, native):
    output = Path(record['output_dir']); final = output / 'final_model'; dm = load_v6_meta(str(native))
    old = Path(record['fresh_down_control_dir']); control = read(old / 'audit.json')
    actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == native.parent, 'Wrong native output directory.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Actual formal recipe differs from preparation.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.down_proj')}
    reference = {s['name']: s for s in load_v6_meta(record['base_dir'])['converted_modules']}
    specs = {s['name']: s for s in dm['converted_modules']}
    _require(len(names) == 32 and set(specs) == set(dm['compressed_targets']) == names
             and dm['compression_categories'] == dm['completed_categories'] == ['down_proj']
             and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
             and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong complete fresh down donor.')
    for name, spec in specs.items():
        _require(spec == stage4_spec(reference[name]), 'Actual native four-stage spec differs: ' + name)
    log = (output / 'train.log').read_text()
    for stage in range(1, 5):
        _require(f'[down_proj.L0-31/stage{stage}] step=10000/10000 ' in log, 'Missing actual stage completion.')
    meta = load_v6_meta(str(final)); report = read(final / 'assembly_report.json'); source = report['sources']; budget = report['bit_budget']
    _require(report['status'] == control['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and control['checkpoint_id'] == DOWN_ID and source['base_checkpoint_id'] == BASE_ID
             and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == ['down_proj']
             and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 3043,
             'Actual complete source union/save/fresh proof differs.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == TOTAL_BITS and budget['packed_linear_bits'] == PACKED_BITS
             and budget['code_padding_bits'] == PADDING_BITS and budget['original_linear_weights'] == DENOMINATOR
             and budget['compressed_target_count'] == 224 and budget['components'] == record['expected_budget_components'],
             'Actual logical/physical six-component storage differs; padding is not part of logical acceptance.')
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 3043 and sum(1 for s in meta['converted_modules'] for _ in _code_specs(s)) == 512,
             'Wrong actual full state or complete code-bank inventory.')
    bank_count = 0
    for spec in meta['converted_modules']:
        if spec['name'] not in names:
            continue
        for key, code in _code_specs(spec):
            value = state[key]
            _require(code['logical_shape'] == [3604480, 1, 12] and list(value.shape) == [3604480, 1, 2]
                     and value.dtype == torch.uint8, 'Wrong actual K12 bank: ' + key)
            for chunk in value.split(262144):
                _require(not bool((chunk[..., -1] & 0xf0).any()), 'Nonzero unused K12 padding: ' + key)
            bank_count += 1
    _require(bank_count == 128, 'Expected128 physical down banks.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {k: tensor_record(v) for k, v in state.items()
                   if k.startswith(n + '.') and k.endswith(('.input_signs', '.output_signs'))} for n in sorted(names)}
    _require(protection == control['protection_tensor_records'] and signs == control['rotation_sign_records'],
             'Actual protection values/indices/scales or signs differ from the original fresh S2 control.')
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
             and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and act['seqlen'] == cache['seqlen'] == 256 and act['seed'] == cache['seed'] == 31
             and names <= set(act['stats_by_linear']), 'Actual original-HF calibration contract differs.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'fresh_down_control_checkpoint_id': DOWN_ID, 'bit_budget': budget,
        'source_union_save_fresh_tensors': 3043, 'full_code_bank_count': 512, 'down_code_bank_count': bank_count,
        'replaced_projection_count': 32, 'preserved_projection_count': 192, 'k12_padding_all_zero': True,
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'protection_and_signs_exact_original_s2': True, 'calibration_cache': tensor_tree(cache),
        'calibration_stats': tensor_tree(act['stats_by_linear']),
        'calibration_boundary': 'Same original HF/calibration recipe and pinned native source. Original S2 runtime cache was retired; no unavailable old token-cache equality is claimed.',
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json'),
        'cat_runtime_state_sha256': sha256(native / 'cat_runtime_state.pt'), 'control_audit_sha256': sha256(old / 'audit.json')}
