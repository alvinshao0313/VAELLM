"""Full-inventory native v6 measurement and explicitly separate prefit fee quotes."""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path

from .adapter import category_projections, file_sha, public_consumer_identity
from .contracts import require, validate_inventory, validate_ledger
from .geometry import component_plan


def measure_native(meta, state, inventory, *, directory=None, max_bpw='2.6', max_lora_rank=0):
    import torch
    from train_utils import checkpoint_bpw as native

    public_consumer_identity()
    inventory = validate_inventory(inventory)
    projections = {row['actual_name']: row for row in inventory['projections']}
    specs = {row['name']: row for row in meta.get('converted_modules', [])}
    require(set(specs) <= set(projections), 'Native checkpoint has targets absent from the actual model inventory.')
    require(set(specs) == set(meta.get('compressed_targets') or []), 'Actual compressed specs and source target inventory disagree.')
    future = set(projections) - set(specs)
    for name, row in projections.items():
        if name in specs:
            spec = specs[name]
            require([spec['out_features'], spec['in_features']] == row['shape'], 'Actual native target shape changed: ' + name)
        else:
            require(name + '.weight' in state and list(state[name + '.weight'].shape) == row['shape'], 'Missing actual future dense payload: ' + name)
    # This is an accounting view only. Source phase/order/completed metadata is
    # retained and never rewritten on disk or offered as seven-class resume.
    view = deepcopy(meta)
    view['compressed_targets'] = sorted(specs)
    view['pending_dense_targets'] = sorted(future)
    view['skip_targets'] = []
    if meta.get('checkpoint_kind') == 'round_base':
        require(meta.get('lora_config') is None and not any('lora_' in key for key in state),
            'An active-LoRA round_base is training state, not a complete deployment accounting view.')
        require(meta.get('checkpoint_id') and meta.get('format') == 'vaellm_model_checkpoint_v6'
            and meta.get('schema_version') == 6, 'Round-base accounting needs its actual complete native identity.')
        # The original phase/prefix/UUID stay in the result. This private view
        # admits only LoRA-null full payloads; it never becomes a resume artifact.
        view['checkpoint_kind'] = 'category_boundary'
    measured = native.audit_state_dict(view, state, max_bpw=100., max_lora_rank=max_lora_rank)
    require(not measured['contract_violations'] and not measured['unclassified_target_keys'], 'Actual native payload has structural/unclassified/rank violations.')
    require(measured['original_linear_weights'] == inventory['total_target_numel']
        and measured['target_count'] == len(projections), 'Every single-class ledger must keep the complete target denominator.')
    code_specs = dict(item for spec in specs.values() for item in native._code_specs(spec))
    groups = {}
    names = sorted(projections, key=len, reverse=True)
    for key, tensor in state.items():
        owner = next((name for name in names if key.startswith(name + '.')), None)
        if owner is None:
            continue
        component = 'code' if key in code_specs else native._component(key[len(owner) + 1:])
        storage = tensor.untyped_storage()
        storage_key = (storage.data_ptr(), storage.nbytes()) if storage.nbytes() else ('empty', key)
        group = groups.setdefault(storage_key, dict(component=component, owners=set(), packed=storage.nbytes() * 8, code_views={}))
        require(group['component'] == component, 'Actual storage alias crosses components.')
        group['owners'].add(owner)
        if key in code_specs:
            span = (tensor.storage_offset(), tensor.storage_offset() + tensor.numel())
            group['code_views'][span] = math.prod(code_specs[key]['logical_shape'])
    categories = {category: dict(N=sum(row['numel'] for row in category_projections(inventory, category)),
        logical_bits=0, packed_bits=0, component_fees={}) for category in inventory['category_ids']}
    future_dense_bits = 0
    for group in groups.values():
        owners = group['owners']
        semantic = {projections[name]['category_id'] for name in owners}
        require(len(semantic) == 1, 'impl_blocked: shared deployment storage across categories needs an explicit category-cap allocation contract.')
        category = next(iter(semantic))
        logical = sum(group['code_views'].values()) if group['component'] == 'code' else group['packed']
        entry = categories[category]
        entry['logical_bits'] += logical
        entry['packed_bits'] += group['packed']
        fee = entry['component_fees'].setdefault(group['component'], dict(logical_bits=0, packed_bits=0))
        fee['logical_bits'] += logical
        fee['packed_bits'] += group['packed']
        if owners <= future and group['component'] in ('dense_weights', 'linear_bias'):
            future_dense_bits += group['packed']
    require(sum(row['logical_bits'] for row in categories.values()) == measured['user_linear_bits']
        and sum(row['packed_bits'] for row in categories.values()) == measured['packed_linear_bits'], 'Per-category actual fees must close exactly to the native full ledger.')
    directory_bytes = None
    files = None
    if directory is not None:
        directory = Path(directory).resolve()
        files = {path.name: path.stat().st_size for path in directory.iterdir() if path.is_file()}
        directory_bytes = sum(files.values())
    n = inventory['total_target_numel']
    limit = math.floor(Fraction(str(max_bpw)) * n)
    result = dict(N=n, logical_bits=measured['user_linear_bits'], packed_bits=measured['packed_linear_bits'],
        tensor_bytes=measured['full_model_tensor_bytes'], directory_bytes=directory_bytes,
        per_category=categories, future_dense_bits=future_dense_bits, component_fees=measured['components'],
        assumptions=['No source checkpoint metadata is rewritten', 'Actual alias storages counted once; code padding separate',
            'Future dense weights and biases charged at their actual persisted dtype'],
        source=dict(kind='actual_native_v6_tensor_storage', checkpoint_id=meta.get('checkpoint_id'),
            source_checkpoint_kind=meta.get('checkpoint_kind'), source_compression_categories=meta.get('compression_categories'),
            source_completed_categories=meta.get('completed_categories'), full_inventory_accounting_view=True,
            formal_seven_class_resume_claim=False),
        actual_target_count=len(projections), active_compressed_count=len(specs), future_dense_count=len(future),
        final_budget_applicable=not future, final_budget_passed=not future and measured['user_linear_bits'] <= limit,
        partial_exceeds_final_limit=bool(future) and measured['user_linear_bits'] > limit,
        logical_bpw=measured['user_linear_bpw'], packed_bpw=measured['packed_linear_bpw'],
        code_padding_bits=measured['code_padding_bits'], actual_storage_aliases=measured['storage_aliases'],
        outside_linear_bits=measured['outside_linear_bits'], files_bytes=files)
    return validate_ledger(result)


def measure_checkpoint(directory, inventory, *, max_bpw='2.6', max_lora_rank=0):
    import json
    import torch

    directory = Path(directory).resolve()
    meta_path = directory / 'checkpoint_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    require(meta['format'] == 'vaellm_model_checkpoint_v6' and meta['schema_version'] == 6,
        'This adapter measures actual native v6; mixed compact needs its own explicit ledger adapter.')
    state_path = directory / meta['state_dict_file']
    state = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    result = measure_native(meta, state, inventory, directory=directory, max_bpw=max_bpw, max_lora_rank=max_lora_rank)
    result['source'].update(directory=str(directory), metadata_sha256=file_sha(meta_path), state_file_sha256=file_sha(state_path))
    return result


def quote_partial(inventory, recipes):
    inventory = validate_inventory(inventory)
    require(set(recipes) <= set(inventory['category_ids']), 'Unknown requested category in the fee quote.')
    rows = {}
    for category in inventory['category_ids']:
        if category in recipes:
            plan = component_plan(inventory, recipes[category])
            require(plan['capability']['constructible'], 'Cannot quote an illegal/unconstructible category geometry.')
            rows[category] = dict(N=sum(row['numel'] for row in category_projections(inventory, category)),
                logical_bits=plan['logical_bits'], packed_bits=plan['packed_bits'], main_compressed=True, plan=plan)
        else:
            dense = sum((row['numel'] + row.get('bias_numel', 0)) * row['dtype_bits'] for row in category_projections(inventory, category))
            rows[category] = dict(N=sum(row['numel'] for row in category_projections(inventory, category)),
                logical_bits=dense, packed_bits=dense, main_compressed=False)
    return dict(N=inventory['total_target_numel'], logical_bits=sum(row['logical_bits'] for row in rows.values()),
        packed_bits=sum(row['packed_bits'] for row in rows.values()), per_category=rows,
        future_dense_bits=sum(row['logical_bits'] for row in rows.values() if not row['main_compressed']),
        measurement='forecast_from_actual_geometry_and_factories_not_checkpoint', tensor_bytes=None, directory_bytes=None,
        source='Actual inventory dtypes; actual Decoder/bitpack/rotation specs; no empirical artifact claim',
        final_budget_applicable=len(recipes) == len(rows), actual_artifact_closure_required=True)
