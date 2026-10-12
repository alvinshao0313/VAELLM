"""Dynamic RMS state inventory, exact e767 complement, and matched calibration."""
from pathlib import Path

import torch

from experiments.llama_gate_rms.common import check, read, _require, rms_spec
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state


def audit(record, native):
    check(record); output = Path(record['output_dir']); final = output / 'final_model'
    control_dir = Path(record['historical_control_dir']); control = read(control_dir / 'audit.json')
    dm, meta = load_v6_meta(str(native)), load_v6_meta(str(final))
    actual = read(native.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == native.parent, 'Wrong native donor path.')
    actual['cat_args']['output_dir'] = str(output / 'donor')
    _require(actual == record['normalized_expected'], 'Actual CAT settings differ from prepared RMS recipe.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.gate_proj')}
    reference = {s['name']: s for s in load_v6_meta(str(control_dir / 'final_model'))['converted_modules']}
    specs = {s['name']: s for s in dm['converted_modules']}
    _require(set(specs) == set(dm['compressed_targets']) == names
        and dm['compression_categories'] == dm['completed_categories'] == ['gate_proj']
        and dm['after_category_mode'] == 'none' and dm['lora_config'] is None
        and dm['base_model_path'] == record['architecture']['base_model_path'], 'Wrong complete native donor.')
    for name, spec in specs.items(): _require(spec == rms_spec(reference[name]), 'Unexpected native RMS spec: ' + name)
    log = (output / 'train.log').read_text(); steps = record['steps_per_stage']
    for stage in range(1, record['expected_inventory']['stages_per_projection'] + 1):
        _require(f'[gate_proj.L0-31/stage{stage}] step={steps}/{steps} ' in log, 'Incomplete native stage.')
    report = read(final / 'assembly_report.json'); source, budget = report['sources'], report['bit_budget']
    inv = record['expected_inventory']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
        and source['base_checkpoint_id'] == record['base_checkpoint_id']
        and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == ['gate_proj']
        and set(source['replaced_targets']) == names
        and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == inv['serialized_tensors'],
        'Native source union/save/fresh is incomplete.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
        and budget['user_linear_bits'] == record['expected_user_linear_bits']
        and budget['packed_linear_bits'] == record['expected_packed_linear_bits']
        and budget['code_padding_bits'] == record['expected_padding_bits']
        and budget['original_linear_weights'] == record['original_linear_weights']
        and budget['compressed_target_count'] == record['architecture']['projection_count']
        and budget['components'] == record['expected_budget_components'], 'Actual RMS storage ledger differs.')
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == inv['serialized_tensors'], 'Actual serialized inventory differs.')
    all_banks = [key for spec in meta['converted_modules'] for key, _ in _code_specs(spec)]
    keys = [key for spec in specs.values() for key, _ in _code_specs(spec)]
    _require(len(all_banks) == inv['code_banks'] and len(keys) == inv['selected_stages']
        and all(not bool((state[key][..., -1] & 0xfc).any()) for key in keys), 'K34 padding/code inventory differs.')
    updated = 0
    for name in names:
        leaves = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + '._parallel_stage_decoder.')}
        _require({k: [list(v.shape), str(v.dtype)] for k, v in leaves.items()} == record['native_cpu_construction']['decoder_tensor_shapes'],
                 'Actual RMS packed decoder leaves differ from native construction.')
        gamma = leaves['_parallel_stage_decoder.norm_out.weight']
        _require(bool(torch.isfinite(gamma).all()), 'Nonfinite RMS gamma.')
        updated += int((gamma != 1).any(dim=1).sum())
    if not record['full_evaluation']:
        _require(updated == inv['selected_stages'], 'New RMS gamma did not update in every diagnostic stage.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {f: tensor_record(state[n + '.weight_rotation.' + f]) for f in ('input_signs', 'output_signs')} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); act = runtime['activation_runtime']; cache = act['cache']
    _require(runtime['global_adaptive_plan'] is None and act['dataset'] == cache['dataset'] == 'alpaca=1'
        and act['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
        and act['seqlen'] == cache['seqlen'] == 256 and act['seed'] == cache['seed'] == 31
        and names <= set(act['stats_by_linear']), 'Actual HF calibration differs.')
    cache_record, stats = tensor_tree(cache), tensor_tree(act['stats_by_linear'])
    _require(protection == control['protection_tensor_records'] and signs == control['rotation_sign_records']
        and cache_record == control['calibration_cache'], 'Protection/sign/calibration input differs from98a8.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': record['base_checkpoint_id'], 'historical_control_checkpoint_id': control['checkpoint_id'],
        'bit_budget': budget, 'source_union_save_fresh_tensors': len(state), 'expected_inventory': inv,
        'replaced_projection_count': len(names), 'preserved_projection_count': record['architecture']['projection_count'] - len(names),
        'rms_updated_stage_count': updated, 'zero_padding_banks': len(keys),
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'calibration_cache': cache_record, 'calibration_stats': stats,
        'matched_control_protection_signs_calibration_exact': True, 'calibration_stats_exact_control': stats == control['calibration_stats'],
        'assembly_report_sha256': sha256(final / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(native.parent / 'normalized_cat_runtime_args.json'),
        'cat_runtime_state_sha256': sha256(native / 'cat_runtime_state.pt'), 'control_audit_sha256': sha256(control_dir / 'audit.json')}
