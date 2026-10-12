"""Audit K38 Q/K, high-two-bit padding, calibration and native source union."""
from copy import deepcopy
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, read
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _require
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state

CONTROL_ID = '43b0c5a2-4ef7-4a00-a6d1-ed887be18d01'
LOGICAL_BITS = 17346134016
PACKED_BITS = 17439031296
DENOMINATOR = 6945767424


def k38_spec(source):
    result = deepcopy(source)
    _require(source['residual_stages'] == 2 and source['stage_codebook_dims'] == [32, 32]
             and source['weight_rotation']['block_size'] == 0, 'Require same full-RHT K32/S2 source.')
    for key in ('vq_weights', 'stage_vq_weights'):
        for bank in result[key]:
            _require(bank['logical_shape'][-1] == 32 and bank['shape'][-1] == 4, 'Wrong original code bank.')
            bank['logical_shape'][-1] = 38; bank['shape'][-1] = 5
    for key in ('decoders', 'stage_decoders'):
        for decoder in result[key]:
            _require(decoder['in_dim'] == 32 and decoder['hidden_dim'] == 128 and decoder['out_dim'] == 32
                     and decoder['num_res_blocks'] == 0 and decoder['param_dtype'] == 'bfloat16', 'Wrong original decoder.')
            decoder['in_dim'] = 38
    return result


def check_donor(directory, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent, 'Wrong actual native output.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Actual recipe differs from preparation.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith(('.q_proj', '.k_proj'))}
    specs = {s['name']: s for s in meta['converted_modules']}
    reference = {s['name']: s for s in load_v6_meta(str(Path(record['control_dir']) / 'final_model'))['converted_modules']}
    _require(len(names) == 72 and set(specs) == set(meta['compressed_targets']) == names
             and meta['compression_categories'] == meta['completed_categories'] == ['q_proj', 'k_proj']
             and meta['after_category_mode'] == 'none' and meta['lora_config'] is None
             and meta['base_model_path'] == record['architecture']['base_model_path'], 'Wrong fresh HF Q/K donor.')
    for name in names: _require(specs[name] == k38_spec(reference[name]), 'Actual spec differs: ' + name)
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for category in ('q_proj', 'k_proj'):
        for stage in (1, 2):
            _require(f'[{category}.L0-35/stage{stage}] step=10000/10000 ' in log, 'Missing complete 10K native stage.')
    return meta, names


def check_assembly(directory, donor, record):
    directory = Path(directory); meta = load_v6_meta(str(directory)); dm, names = check_donor(donor, record)
    report = read(directory / 'assembly_report.json'); source = report['sources']; budget = report['bit_budget']
    control = read(Path(record['control_dir']) / 'audit.json')
    _require(report['status'] == control['status'] == 'verified' and control['checkpoint_id'] == CONTROL_ID
             and report['checkpoint_id'] == meta['checkpoint_id'] and source['base_checkpoint_id'] == record['base_checkpoint_id'] == EXPECTED_IDS['qwen3']
             and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == ['q_proj', 'k_proj']
             and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 3423, 'Wrong complete source union/save/fresh.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == LOGICAL_BITS and budget['packed_linear_bits'] == PACKED_BITS
             and budget['code_padding_bits'] == PACKED_BITS - LOGICAL_BITS
             and budget['original_linear_weights'] == DENOMINATOR and budget['compressed_target_count'] == 252
             and budget['components'] == record['expected_budget_components'], 'Actual logical/physical storage differs.')
    state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    banks = [k for s in meta['converted_modules'] for k, _ in _code_specs(s)]
    local = [k for k in state if any(k.startswith(n + '.') for n in names)]
    _require(len(state) == 3423 and len(local) == 936 and len(banks) == 504
             and all(state[k].dtype == torch.uint8 for k in banks), 'Persistent inventory differs.')
    selected_banks = [k for s in dm['converted_modules'] for k, _ in _code_specs(s)]
    _require(len(selected_banks) == 144 and all(not bool((state[k][..., -1] & 0xc0).any()) for k in selected_banks), 'Nonzero K38 high-two-bit padding.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {k: tensor_record(v) for k, v in state.items() if k.startswith(n + '.')
                and k.endswith(('.input_signs', '.output_signs'))} for n in sorted(names)}
    _require(protection == control['protection_tensor_records'] and signs == control['rotation_sign_records'], 'Protection/sign differs from K32 control.')
    runtime = load_v6_cat_runtime_state(str(donor), required=True); activation = runtime['activation_runtime']; cache = activation['cache']
    _require(runtime['global_adaptive_plan'] is None and activation['dataset'] == cache['dataset'] == 'alpaca=1'
             and activation['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and activation['seqlen'] == cache['seqlen'] == 256 and activation['seed'] == cache['seed'] == 31
             and names <= set(activation['stats_by_linear']), 'Actual HF calibration changed.')
    cache_record = tensor_tree(cache); stats = tensor_tree(activation['stats_by_linear'])
    _require(cache_record == record['control_calibration']['cache'], 'Actual512 calibration cache differs from K32.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': record['base_checkpoint_id'], 'control_checkpoint_id': CONTROL_ID, 'bit_budget': budget,
        'source_union_save_fresh_tensors': 3423, 'packed_bank_count': 504, 'high_two_padding_bits_zero_banks': 144,
        'replaced_projection_count': 72, 'preserved_projection_count': 180, 'replaced_local_tensor_count': 936, 'frozen_complement_tensor_count': 2487,
        'frozen_evidence': 'Native assembler checks complete3423 expected source union, serialized and fresh states exactly.',
        'protection_tensor_records': protection, 'rotation_sign_records': signs, 'calibration_cache': cache_record, 'calibration_stats': stats,
        'matched_protection_sign_cache_exact': True, 'calibration_stats_exact_control': stats == record['control_calibration']['stats'],
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json'),
        'cat_runtime_state_sha256': sha256(Path(donor) / 'cat_runtime_state.pt')}
