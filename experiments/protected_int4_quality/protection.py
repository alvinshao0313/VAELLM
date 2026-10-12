"""Sidecar replay into existing native unquantized protection buffers on CPU."""
import torch

from experiments.longbench_attribution.restore import tensor_digest
from experiments.long_context_recovery.contract import require
from experiments.protected_int4_probe.codec import Payload, decode_output_scale
from litebsq.vae_linear import VAELinear


def payload_from(row):
    return Payload(row['values'], row['scales'], tuple(row['shape']), 4, row.get('signs'))


def immutable_inventory(model, names):
    excluded = {name + '.' + suffix for name in names for suffix in
                ('protected_input_qvalues', 'protected_input_scales', 'protected_input_weight')}
    runtime_caches = set()
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            require('_cached_weight' in module._non_persistent_buffers_set,
                    'Only native nonpersistent materialized weight caches may be excluded.')
            runtime_caches.update(name + '.' + suffix for suffix in module._non_persistent_buffers_set)
    return {'state': {key: tensor_digest(value) for key, value in model.state_dict().items() if key not in excluded},
        'buffers': {key: tensor_digest(value) for key, value in model.named_buffers()
                    if key not in excluded and key not in runtime_caches},
        'parameter_identity': {key: [id(value), value.data_ptr(), value._version]
            for key, value in model.named_parameters() if key not in excluded}}


def replay(model, record):
    artifact = torch.load(record['payload_path'], map_location='cpu', weights_only=True)
    require(artifact['source_checkpoint_id'] == record['source_checkpoint_id'] and artifact['arm'] == record['arm']
            and set(artifact['projections']) == set(record['selected_targets']), 'Sidecar identity or target set changed.')
    names = record['selected_targets']; before = immutable_inventory(model, names)
    manifest = record['payload_manifest']; restored = {}
    for name in names:
        module = model.get_submodule(name); row = artifact['projections'][name]; item = manifest[name]
        require(isinstance(module, VAELinear) and module.protected_channel_quant_format == 'int8'
                and module.protected_input_weight is None and module.protected_output_indices is None
                and module._cached_weight is None and module.protected_input_qvalues.device.type == 'cpu',
                'Apply only to untouched native INT8 input-protected CPU projections, before prewarm.')
        require(torch.equal(module.protected_input_indices, row['indices'])
                and tensor_digest(module.protected_input_qvalues) == item['native_qvalues_sha256']
                and tensor_digest(module.protected_input_scales) == item['native_scales_sha256'],
                'Native source payload or original protection indices differ.')
        decoded = decode_output_scale(payload_from(row), dtype=torch.bfloat16).t().contiguous()
        require(decoded.shape == (module.protected_input_indices.numel(), module.out_features)
                and tensor_digest(decoded) == item['decoded_input_by_output_sha256'], 'Sidecar BF16 replay changed.')
        module.protected_input_qvalues = None
        module.protected_input_scales = None
        module.protected_channel_quant_format = 'none'
        module.protected_input_weight = torch.nn.Parameter(decoded, requires_grad=False)
        native = module._decode_protected_input_weight_rows(dtype=torch.bfloat16, device=torch.device('cpu'))
        require(torch.equal(native, decoded), 'Native none protection does not return the exact replayed BF16 payload.')
        restored[name] = {'decoded_sha256': tensor_digest(native), 'shape': list(native.shape),
                          'expanded_storage_bits': native.untyped_storage().nbytes() * 8}
    after = immutable_inventory(model, names)
    require(after == before, 'Protection replay changed a main-VQ, index, backbone, buffer or parameter identity.')
    require(len(model.state_dict()) == record['expected_serialized_tensors'] - len(names),
            'Simulation must replace exactly two INT8 buffers with one BF16 parameter per projection.')
    return {'projections': restored, 'projection_count': len(names),
        'frozen_state_tensors': len(before['state']), 'all_non_protection_state_exact': True,
        'cpu_parameter_identity_exact': True, 'schema_delta': 'Per projection: remove qvalues/scales, add frozen BF16 protected_input_weight, format int8->none.',
        'sidecar_replayed_exact': True, 'model_exported': False}, before


def verify_native_cache(model, record):
    expanded_bits = 0
    for name in record['selected_targets']:
        module = model.get_submodule(name); expected = module.protected_input_weight
        require(module.protected_channel_quant_format == 'none' and module.protected_input_qvalues is None
                and module.protected_input_scales is None and expected.dtype == torch.bfloat16
                and not expected.requires_grad, 'Native protection simulation schema/precision changed.')
        require(tensor_digest(expected) == record['payload_manifest'][name]['decoded_input_by_output_sha256'],
                'GPU move or inference changed the replayed BF16 payload.')
        cached = module._cached_weight
        require(cached is not None and cached.dtype == torch.bfloat16
                and torch.equal(cached.index_select(1, module.protected_input_indices), expected.t()),
                'Native prewarm/cache did not scatter the exact decoded protected columns.')
        expanded_bits += expected.untyped_storage().nbytes() * 8
    return {'projection_count': len(record['selected_targets']), 'protected_cached_columns_exact': True,
            'expanded_protected_weight_storage_bits': expanded_bits, 'native_forward_unchanged': True}
