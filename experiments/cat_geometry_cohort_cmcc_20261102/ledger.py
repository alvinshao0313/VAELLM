"""Actual partial-node storage measurement, without stable-final budget assertions."""
from collections import defaultdict
import math

from . import bindings as b


def measure(meta, state):
    from train_utils import checkpoint_bpw as native
    b.require(meta['checkpoint_kind'] == 'round_base' and meta['converted_module_count'] == 180
        and len(meta['pending_dense_targets']) == 72, 'Measure the actual partial node only.')
    specs = {value['name']: value for value in meta['converted_modules']}
    names = set(meta['compressed_targets']) | set(meta['pending_dense_targets']) | set(meta['skip_targets'])
    b.require(len(names) == 252 and set(specs) == set(meta['compressed_targets']), 'Incomplete projection inventory.')
    counts = {name: int(specs[name]['in_features'])*int(specs[name]['out_features']) if name in specs
        else state[name+'.weight'].numel() for name in names}
    codes = dict(item for spec in specs.values() for item in native._code_specs(spec))
    groups = {}
    for key, value in state.items():
        owner = next((name for name in names if key.startswith(name+'.')), None)
        component = 'code' if key in codes else native._component(key[len(owner)+1:]) if owner else 'outside_projection'
        b.require(component != 'other_target', 'Unclassified target storage: '+key)
        storage = value.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes()) if storage.nbytes() else ('empty', key)
        row = groups.setdefault(identity, dict(owners=set(), scope=owner is not None, component=component,
            storage_bits=storage.nbytes()*8, views={}, dtypes=set(), keys=[]))
        b.require(row['scope'] == (owner is not None) and row['component'] == component,
            'Storage alias crossed accounting scope.')
        if owner:
            row['owners'].add(owner)
        row['keys'].append(key)
        row['dtypes'].add(str(value.dtype))
        if key in codes:
            spec = codes[key]
            b.require(value.dtype == b.torch.uint8 and list(value.shape) == spec['shape'] and value.is_contiguous(),
                'Invalid actual packed code: '+key)
            span = (value.storage_offset(), value.storage_offset()+value.numel())
            logical = math.prod(spec['logical_shape'])
            for previous, previous_logical in row['views'].items():
                b.require((span == previous and logical == previous_logical)
                    or (span != previous and max(span[0], previous[0]) >= min(span[1], previous[1])),
                    'Packed code views overlap or disagree.')
            row['views'][span] = logical
    components = defaultdict(lambda: dict(budget_bits=0, storage_bits=0, unique_storages=0, dtypes=set()))
    gate_components = defaultdict(lambda: dict(budget_bits=0, storage_bits=0, unique_storages=0, dtypes=set()))
    outside = 0
    for row in groups.values():
        if not row['scope']:
            outside += row['storage_bits']
            continue
        logical = sum(row['views'].values()) if row['component'] == 'code' else row['storage_bits']
        selected = [components]
        gate_owners = [name.endswith('.gate_proj') for name in row['owners']]
        b.require(all(gate_owners) or not any(gate_owners), 'Storage alias crossed gate/non-gate ledger scopes.')
        if all(gate_owners):
            selected.append(gate_components)
        for table in selected:
            item = table[row['component']]
            item['budget_bits'] += logical
            item['storage_bits'] += row['storage_bits']
            item['unique_storages'] += 1
            item['dtypes'].update(row['dtypes'])
    for table in (components, gate_components):
        for value in table.values():
            value['dtypes'] = sorted(value['dtypes'])
    dim = next(value['codebook_dim'] for value in specs.values() if value['name'].endswith('.gate_proj'))
    return dict(status='measured_actual_partial_storage', checkpoint_id=meta['checkpoint_id'],
        checkpoint_kind=meta['checkpoint_kind'], converted=180, future_dense=72,
        projection=native._summary(components, sum(counts.values())),
        gate=native._summary(gate_components, sum(count for name,count in counts.items() if name.endswith('.gate_proj'))),
        full_model_tensor_bytes=sum(row['storage_bits'] for row in groups.values())//8,
        outside_projection_bits=outside, serialized_file_bytes=None,
        layout=dict(gate_D=dim, gate_K=dim, stages=2, protected_input_count=64,
            compressed_shape=[12288,4032], vectors_per_gate=12288*4032//dim,
            flatten_remainder=12288*4032%dim, weight_padding_elements=0,
            rotation_block_size=32, gate_payload_bpw=2),
        stable_audit_boundary='Original audit excludes round_base and has a protected-channel multiple-of-D policy; '
            'D128/input64 does not meet that additional policy. Actual flatten/native deployment is validated separately. '
            'Resolve this stable audit gap before any complete-suffix acceptance; no final <=2.6 or budget-pass claim.',
        scope='Actual unique tensor storages; main codes, decoder, protection values/scales/indices, rotation and 72 future dense '
            'included in projection ledger; encoder is training-only and absent from deployment.')
