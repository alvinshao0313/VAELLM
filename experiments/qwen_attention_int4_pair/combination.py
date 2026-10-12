"""Validate the actual donor, then build an in-memory source union and INT4 sidecar."""
import gc
from pathlib import Path

import torch

from experiments.qwen_attention_int4_pair.common import CATEGORIES, BITS, read, _require
from experiments.qwen3_w25.assemble_categories import _expected_state, _load_state, _replace_modules, _assert_state_equal
from experiments.w25_eval.assemble_categories import validate_sources
from experiments.longbench_attribution.restore import tensor_digest
from experiments.longbench_v1.common import sha256, file_stat
from experiments.protected_int4_probe.codec import encode_output_scale, decode_output_scale
from experiments.protected_int4_scale_search.search import RATIOS, search_scales
from experiments.protected_int4_quality.protection import payload_from, replay
from litebsq.protected_channel_quant import decode_protected_channel_weight
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.v6_model_loader import load_v6_model_checkpoint


def check_donor(donor, record):
    actual = read(donor.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']) == donor.parent, 'Donor output identity differs.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Actual native training parameters differ.')
    meta = read(donor / 'checkpoint_meta.json')
    _require(meta['compression_categories'] == meta['completed_categories'] == CATEGORIES
             and meta['after_category_mode'] == 'none' and len(meta['converted_modules']) == 144,
             'Require exactly four fresh complete attention categories.')
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for cat in CATEGORIES:
        for stage in (1, 2):
            _require(f'[{cat}.L0-35/stage{stage}] step=10000/10000 ' in log, 'Incomplete native schedule.')
    for spec in meta['converted_modules']:
        cat = spec['name'].rsplit('.', 1)[1]; bits = BITS[record['arm']][cat]
        _require(spec['residual_stages'] == 2 and spec['stage_codebook_dims'] == [32, 32]
                 and spec['transpose'] == (cat != 'k_proj') and spec['parallel_parts'] == 1
                 and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == 32, 'Actual donor geometry differs.')
        for packed, decoder in zip(spec['stage_vq_weights'], spec['stage_decoders'], strict=True):
            groups = spec['out_features'] * (spec['in_features'] - 64) // 32
            _require(packed == {'shape': [groups, 1, bits//8], 'dtype': 'uint8', 'storage_format': 'bitpack_u8',
                     'logical_shape': [groups, 1, bits], 'logical_dtype': 'bool', 'pack_bits': 8}, 'Donor packed layout differs.')
            _require(decoder['in_dim'] == bits and decoder['out_dim'] == 32 and decoder['hidden_dim'] == 128
                     and decoder['num_res_blocks'] == 0 and decoder['decoder_type'] == 'symmetric'
                     and decoder['param_dtype'] == 'bfloat16', 'Donor decoder layout differs.')
    return meta


def sidecar(state, meta, record, native_budget):
    output = Path(record['output_dir']); payload_path = output / 'protected_payloads.pt'
    _require(not payload_path.exists(), 'Never overwrite an executed simulation payload.')
    payloads, manifest, diagnostics = {}, {}, {}; old_bits = new_bits = expanded_bits = 0
    for spec in meta['converted_modules']:
        name = spec['name']; q, s, indices = (state[name + '.protected_input_' + key] for key in ('qvalues', 'scales', 'indices'))
        expected_count = 256 if name.endswith('.down_proj') else 64
        _require(spec['protected_channel_quant_format'] == 'int8' and spec['protected_output_indices'] is None
                 and spec['protected_input_weight'] is None and indices.numel() == expected_count,
                 'Actual union protection differs from the fixed recipe.')
        reference = decode_protected_channel_weight(q, s, quant_format='int8', dtype=torch.bfloat16,
                                                    device=torch.device('cpu')).t().contiguous()
        original = encode_output_scale(reference, bits=4)
        selected, choices, before, after = search_scales(reference, original, row_chunk=512)
        _require(original.storage() == selected.storage() and selected.storage()['padding_bits'] == 0
                 and bool((after <= before).all()), 'Scale-search storage or no-worse row contract failed.')
        decoded = decode_output_scale(selected).t().contiguous()
        payloads[name] = {'shape': selected.shape, 'bits': 4, **selected.tensors(), 'indices': indices.clone()}
        native_bits = sum(t.untyped_storage().nbytes()*8 for t in (q, s))
        old_bits += native_bits; new_bits += selected.storage()['total_bits']; expanded_bits += decoded.untyped_storage().nbytes()*8
        manifest[name] = {'native_qvalues_sha256': tensor_digest(q), 'native_scales_sha256': tensor_digest(s),
            'indices_sha256': tensor_digest(indices), 'decoded_input_by_output_sha256': tensor_digest(decoded),
            'source_payload_bits_without_indices': native_bits, 'candidate_payload_storage': selected.storage(),
            'payload_tensor_sha256': {k: tensor_digest(v) for k,v in payloads[name].items() if torch.is_tensor(v)}}
        diagnostics[name] = {'baseline_sse': float(before.sum()), 'selected_sse': float(after.sum()),
            'reference_energy': float(reference.double().square().sum()), 'rows': len(choices),
            'strictly_improved_rows': int((after < before).sum()), 'rows_with_worse_sse': int((after > before).sum()),
            'selection_counts': torch.bincount(choices, minlength=17).tolist()}
    _require(len(payloads) == 252, 'Require every actual union protection matrix.')
    artifact = {'schema': 'protected_int4_payload_v1', 'source_checkpoint_id': record['base_checkpoint_id'],
        'donor_checkpoint_id': meta['donor_checkpoint_id'], 'arm': record['arm'], 'projections': payloads}
    torch.save(artifact, payload_path)
    restored = torch.load(payload_path, map_location='cpu', weights_only=True)
    _require(restored.keys() == artifact.keys() and all(restored[k] == artifact[k] for k in
             ('schema','source_checkpoint_id','donor_checkpoint_id','arm')) and restored['projections'].keys() == payloads.keys(),
             'Sidecar fresh-reload identity differs.')
    for name, row in restored['projections'].items():
        _require(row.keys() == payloads[name].keys() and all(torch.equal(row[k],v) if torch.is_tensor(v) else row[k] == v
                 for k,v in payloads[name].items()), 'Sidecar fresh-reload bytes differ.')
        _require(tensor_digest(decode_output_scale(payload_from(row)).t().contiguous())
                 == manifest[name]['decoded_input_by_output_sha256'], 'Persisted BF16 decode differs.')
    total = native_budget['user_linear_bits']; denominator = native_budget['original_linear_weights']
    account = {'source_user_linear_bits': total, 'original_linear_weights': denominator,
        'replaced_native_qvalues_scales_bits': old_bits, 'candidate_packed_values_scales_signs_bits': new_bits,
        'unchanged_indices_already_in_source_budget': True, 'candidate_projected_linear_bits': total-old_bits+new_bits,
        'expanded_simulation_linear_bits': total-old_bits+expanded_bits, 'sidecar_container_bytes': payload_path.stat().st_size,
        'boundary': 'Actual sidecar tensor storage plus actual native union remainder; indices counted once. BF16 expanded carrier is not a native INT4 checkpoint.'}
    for key in ('candidate_projected_linear','expanded_simulation_linear'):
        account[key+'_bpw'] = account[key+'_bits']/denominator
    _require(account['candidate_projected_linear_bits'] == record['forecast']['packed_sidecar_combination_bits']
             and account['expanded_simulation_linear_bits'] == record['forecast']['expanded_simulation_bits']
             and account['candidate_projected_linear_bits']*2 <= denominator*5, 'Actual sidecar budget differs from forecast.')
    runtime = dict(record, source_checkpoint_id=record['base_checkpoint_id'], selected_targets=sorted(payloads),
                   expected_serialized_tensors=3423, payload_path=str(payload_path), payload_manifest=manifest)
    return runtime, {'accounting': account, 'payload_manifest': manifest, 'diagnostics': diagnostics,
        'ratios': list(RATIOS), 'sidecar_sha256': sha256(payload_path), 'sidecar_saved_fresh_reload_exact': True,
        'reference': 'Actual final source union native INT8 BF16; no mismatched f927/donor payload reused.'}


def build(record):
    base_dir = Path(record['base_dir']); donor = Path(record['donor_checkpoint_dir'])
    donor_meta = check_donor(donor, record)
    base_meta, dm, names, architecture, _, _, _ = validate_sources(base_dir, donor, CATEGORIES)
    _require(base_meta['checkpoint_id'] == record['base_checkpoint_id'] and dm['checkpoint_id'] == donor_meta['checkpoint_id']
             and len(names) == 144, 'Source-union identity differs.')
    stats = [file_stat(p / 'pytorch_model.bin') for p in (base_dir, donor)]
    base_state, donor_state = _load_state(base_dir), _load_state(donor)
    expected = _expected_state(base_state, donor_state, names, architecture['projection_shapes'])
    _require(len(expected) == 3423, 'Source union inventory changed.')
    donor_specs = {s['name']: s for s in dm['converted_modules']}
    meta = dict(base_meta, converted_modules=[donor_specs[s['name']] if s['name'] in names else s for s in base_meta['converted_modules']],
                donor_checkpoint_id=dm['checkpoint_id'])
    budget = audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0)
    allowed = [] if record['arm'] == 'control' else [f"User Linear bpw {budget['user_linear_bpw']:.9f} exceeds 2.5."]
    _require(budget['contract_violations'] == allowed and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == record['forecast']['native_int8_union_bits']
             and budget['code_padding_bits'] == 0 and budget['components']['decoder']['dtypes'] == ['bfloat16'],
             'Unexpected native union structure or storage. Native over-budget union is not an exported candidate.')
    frozen_source = {k: tensor_digest(v) for k,v in expected.items()}
    runtime, payload_report = sidecar(expected, meta, record, budget)
    base, bm, bl = load_v6_model_checkpoint(str(base_dir), map_location='cpu', strict=True)
    fresh, fm, fl = load_v6_model_checkpoint(str(donor), map_location='cpu', strict=True)
    _require(not bl.missing_keys and not bl.unexpected_keys and not fl.missing_keys and not fl.unexpected_keys
             and bm['checkpoint_id'] == base_meta['checkpoint_id'] and fm['checkpoint_id'] == dm['checkpoint_id'], 'Fresh strict source load differs.')
    _replace_modules(base, fresh, names); _assert_state_equal(base.state_dict(), expected, 'In-memory source union')
    del fresh, base_state, donor_state, expected; gc.collect()
    base.eval().requires_grad_(False); base.config.use_cache = False
    replay_report, frozen = replay(base, runtime)
    _require([file_stat(p / 'pytorch_model.bin') for p in (base_dir, donor)] == stats, 'Sources changed during composition.')
    return base, runtime, frozen, {'native_union_budget': budget, 'source_union_tensor_sha256': frozen_source,
        'source_union_exact_tensors': 3423, 'fresh_strict_source_ids': [bm['checkpoint_id'], fm['checkpoint_id']],
        'replaced_attention': 144, 'preserved_mlp': 108, 'preserved_backbone': 147,
        'source_weight_stats': stats, 'payload': payload_report, 'replay': replay_report, 'model_exported': False}
