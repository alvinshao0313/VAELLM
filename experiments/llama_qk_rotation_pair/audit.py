"""Audit the real Llama64 donor, e767160 complement, and paired calibration."""
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import read
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _require
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state


def check_donor(directory, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent, 'Wrong native output identity.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Actual native recipe differs from CPU parsing.')
    specs = {s['name']: s for s in meta['converted_modules']}; names = set(record['expected_target_specs'])
    _require(record['categories'] == ['q_proj', 'k_proj'] and record['architecture']['num_hidden_layers'] == 32
             and record['architecture']['projection_count'] == 224 and len(names) == 64
             and specs == record['expected_target_specs'] and set(meta['compressed_targets']) == names
             and meta['completed_categories'] == meta['compression_categories'] == record['categories']
             and meta['after_category_mode'] == 'none' and meta['lora_config'] is None
             and meta['base_model_path'] == record['architecture']['base_model_path'], 'Wrong complete native Q/K donor/specs.')
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for category in record['categories']:
        for stage in (1, 2):
            _require(f'[{category}.L0-31/stage{stage}] step=10000/10000 ' in log, 'Missing complete native stage.')
    return meta, names


def check_assembly(directory, donor, record):
    directory = Path(directory); meta = load_v6_meta(str(directory)); dm, names = check_donor(donor, record)
    report = read(directory / 'assembly_report.json'); source = report['sources']; budget = report['bit_budget']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id']
             and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == record['categories']
             and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979, 'Wrong source union/save/fresh.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == record['expected_user_linear_bits']
             and budget['original_linear_weights'] == record['architecture']['original_linear_weights']
             and budget['code_padding_bits'] == 0 and budget['compressed_target_count'] == 224
             and budget['components'] == record['expected_budget_components'], 'Actual integer deployment budget differs.')
    state = torch.load(directory / meta['state_dict_file'], weights_only=True, map_location='cpu', mmap=True)
    local = [k for k in state if any(k.startswith(n + '.') for n in names)]
    banks = [k for s in meta['converted_modules'] for k, _ in _code_specs(s)]
    _require(all(state[k].dtype == torch.uint8 for k in banks), 'Native banks are not packed uint8.')
    _require(len(state) == 2979 and len(local) == 832 and len(banks) == 448, 'Wrong actual persistent inventory.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {k: tensor_record(v) for k, v in state.items() if k.startswith(n + '.')
                     and k.endswith(('.input_signs', '.output_signs'))} for n in sorted(names)}
    _require(all(len(v) == 2 for v in signs.values()), 'Incomplete actual rotation signs.')
    runtime = load_v6_cat_runtime_state(str(donor), required=True); activation = runtime['activation_runtime']; cache = activation['cache']
    _require(runtime['global_adaptive_plan'] is None and activation['dataset'] == cache['dataset'] == 'alpaca=1'
             and activation['nsamples'] == cache['nsamples'] == 512 and activation['seed'] == cache['seed'] == 31
             and activation['seqlen'] == cache['seqlen'] == 256 and len(cache['input_ids']) == 512
             and names <= set(activation['stats_by_linear']), 'Actual shared HF calibration differs.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': record['base_checkpoint_id'], 'bit_budget': budget, 'source_union_save_fresh_tensors': 2979,
        'replaced_projection_count': 64, 'preserved_projection_count': 160, 'replaced_local_tensor_count': 832,
        'frozen_complement_tensor_count': 2147, 'packed_bank_count': len(banks),
        'frozen_evidence': 'Native assembler compared all2979 source-union, serialized and fresh-reloaded states exactly.',
        'protection_tensor_records': protection, 'rotation_sign_records': signs, 'calibration_cache': tensor_tree(cache),
        'calibration_stats': tensor_tree(activation['stats_by_linear']), 'global_adaptive_plan': None,
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json'),
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'cat_runtime_state_sha256': sha256(Path(donor) / 'cat_runtime_state.pt')}
