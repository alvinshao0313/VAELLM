"""Audit unchanged D1200 tensors and byte-identical native donor sparse buffers."""
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import read
from experiments.llama_decoder_sparse_combo.prepare import BASE, PATCH, BASE_ID, PATCH_ID, E767_ID, check
from experiments.llama_gate_sparse_residual.audit import FIELDS, PROBE, local_forward, state
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.v6_model_loader import load_v6_model_checkpoint


def frozen_and_budget(meta, current, record):
    base_meta, original = state(BASE); patch_meta, donor = state(PATCH)
    _require(base_meta['checkpoint_id'] == BASE_ID and patch_meta['checkpoint_id'] == PATCH_ID
             and len(original) == 2979 and len(current) == 3171, 'Source identity or state inventory differs.')
    _assert_state_equal({k: current[k].detach().cpu() for k in original}, original, 'unchanged D1200 tensors')
    added = {name + '.sparse_residual_' + field for name in record['targets'] for field in FIELDS}
    _require(set(current) - set(original) == added and len(added) == 192,
             'Only192 native sparse buffers may be appended.')
    _assert_state_equal({k: current[k].detach().cpu() for k in added}, {k: donor[k] for k in added},
                        'unchanged e767 sparse payload')
    prior, patches, actual = [{s['name']: s for s in m['converted_modules']}
                              for m in (base_meta, patch_meta, meta)]
    _require(set(actual) == set(prior) and meta['compressed_targets'] == base_meta['compressed_targets']
             and not meta['pending_dense_targets'] and not meta['skip_targets'] and meta['lora_config'] is None,
             'The complete compressed topology changed.')
    for name, item in actual.items():
        nonsparse = lambda spec: {k: v for k, v in spec.items() if not k.startswith('sparse_residual_')}
        sparse = lambda spec: {k: v for k, v in spec.items() if k.startswith('sparse_residual_')}
        _require(nonsparse(item) == nonsparse(prior[name]) and sparse(item) == sparse(patches[name]),
                 'Native source/patch metadata union differs: ' + name)
    extra = meta['extra_meta']
    _require(extra['source_base_checkpoint_id'] == BASE_ID and extra['sparse_patch_checkpoint_id'] == PATCH_ID
             and extra['sparse_residual_source_checkpoint_id'] == E767_ID,
             'Saved metadata lost the separate decoder source and residual origin.')
    budget = audit_state_dict(meta, current, max_bpw=2.5, max_lora_rank=0)
    expected = record['expected_budget']
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and all(budget[k] == expected[k] for k in
                     ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components')),
             'Unchanged source/patch storage or budget differs.')
    total = sum(current[k].untyped_storage().nbytes() for k in added)
    _require(total == 42680128, 'Frozen native codec byte count changed.')
    return {'original_frozen_tensors': 2979, 'added_frozen_sparse_tensors': 192, 'saved_tensors': 3171,
            'copied_payload_bytes': total, 'source_tensors_exact': True, 'sparse_payload_exact': True,
            'native_metadata_union_exact': True, 'bit_budget': budget}


def verify(record):
    check(record); output = Path(record['output_dir']); checkpoint = output / 'final_model'
    meta, saved = state(checkpoint); report = frozen_and_budget(meta, saved, record)
    model, fresh_meta, _ = load_v6_model_checkpoint(str(checkpoint), strict=True, map_location='cpu')
    model.eval().requires_grad_(False)
    compared = _assert_state_equal(model.state_dict(), saved, 'fresh strict native patch-only combination')
    probe = torch.load(output / 'local_probe.pt', weights_only=True, map_location='cpu')
    module = model.get_submodule(PROBE).to('cuda:0')
    after = local_forward(module, probe['input'])
    _require(torch.equal(after, probe['after']), 'Fresh native BF16 probe differs from saved combination.')
    module.clear_decoded_weight_cache(); module.to('cpu')
    _assert_state_equal(model.state_dict(), saved, 'after native BF16 probe')
    _require(not any(p.requires_grad for p in model.parameters()), 'Unexpected trainable parameter.')
    report.update(status='verified', checkpoint_id=fresh_meta['checkpoint_id'],
        base_checkpoint_id=BASE_ID, patch_checkpoint_id=PATCH_ID, residual_source_checkpoint_id=E767_ID,
        strict_reloaded_tensors=compared, trainable_parameters=0, hf_residual_recomputed=False,
        residual_target=record['residual_target'], comparison_boundary=record['comparison_boundary'],
        native_before_after_local_forward=read(output / 'build_report.json')['local_forward'],
        fresh_native_bf16_local_forward={'matches_saved_probe': True, 'dtype': str(after.dtype),
            'shape': list(after.shape), 'finite': True, 'quality_evidence': False},
        metadata_sha256=sha256(checkpoint / 'checkpoint_meta.json'),
        local_probe_sha256=sha256(output / 'local_probe.pt'))
    write_json(output / 'audit.json', report)
    print({'status': 'verified', 'source': BASE_ID, 'patch': PATCH_ID, 'frozen': 3171,
           'user_linear_bpw': report['bit_budget']['user_linear_bpw']}, flush=True)
