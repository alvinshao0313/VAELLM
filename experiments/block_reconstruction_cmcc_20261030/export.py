"""Save only the committed BF16 mixed BR endpoint and its frozen complement."""
from . import bindings as b
from . import fit

torch = b.torch


def deployment_contract(model):
    b.require(model.config.num_hidden_layers == 36 and model.config.hidden_size == 4096
        and model.config.intermediate_size == 12288,
        'This BR prototype requires the complete real Qwen3-8B architecture.')
    b.require(all(value.dtype == torch.bfloat16 and not value.requires_grad and value.grad is None
                  for value in model.parameters()) and all(not module.training for module in model.modules()),
              'Save only actual frozen eval BF16 parameters after each local decoder commit.')
    rotary = model.model.rotary_emb
    b.require(rotary.inv_freq.dtype == rotary.original_inv_freq.dtype == torch.float32
        and bool(torch.isfinite(rotary.inv_freq).all()) and bool(torch.isfinite(rotary.original_inv_freq).all()),
        'Preserve finite original FP32 root RoPE; never cast the complete model to BF16.')
    rope = [(name, value) for name, value in model.named_buffers() if 'inv_freq' in name]
    b.require(rope and all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for name, value in rope),
              'All actual RoPE buffers must remain finite FP32.')
    groups = b.checkpoint.inventories(model)
    b.require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [216, 36, 0],
              'The complete BR endpoint must retain its actual 216 native plus 36 compact projections.')
    return groups


def frozen_contract(model, source_records, allowed):
    allowed = set(allowed)
    actual = fit.state_records(model)
    b.require(allowed and allowed.issubset(source_records) and set(actual) == set(source_records),
              'BR may update only existing decoder leaves, without a new persistent deployment field.')
    b.require({name: value for name, value in actual.items() if name not in allowed}
        == {name: value for name, value in source_records.items() if name not in allowed},
        'A code, route, protected payload, rotation, norm, other block or nondecoder source field changed.')
    b.require(all(actual[name]['shape'] == source_records[name]['shape']
        and actual[name]['dtype'] == source_records[name]['dtype'] == 'torch.bfloat16' for name in allowed),
        'Every trained existing decoder must be committed to its original BF16 shape and dtype.')
    return actual, fit.proof_from_records(actual, allowed)


def budget_contract(budget, source_budget):
    b.require(budget['budget_passed'] and budget['final_budget_applicable']
        and [budget[key] for key in ('native_target_count', 'compact_target_count', 'pending_dense_target_count')]
            == [216, 36, 0], 'The BR endpoint lost its complete compressed budget contract.')
    keys = ('original_linear_weights', 'target_count', 'user_linear_bits', 'packed_linear_bits',
            'full_model_tensor_bytes', 'outside_linear_bits', 'components')
    b.require(all(budget[key] == source_budget[key] for key in keys),
              'BR changed logical/packed storage or a charged component relative to the actual converted source.')
    return dict(logical_bits_unchanged=True, packed_bits_unchanged=True,
                complete_tensor_bytes_unchanged=True, compared_source_components=True)


@torch.no_grad()
def save(model, directory, identity, source_audit, source_records, allowed):
    groups = deployment_contract(model)
    expected, frozen = frozen_contract(model, source_records, allowed)
    source_budget = source_audit['post_conversion_budget']
    provenance = dict(schema='BR_adjacent_blocks_short2_export_v1', execution=identity,
        validation_only=True, prototype_only=identity['prototype_only'],
        source_binding=identity['source_binding'], source_audit=source_audit,
        completed_layers=[0, 1], updates_per_layer=2, group_indices=list(fit.GROUPS),
        schedule_steps=500, trained_decoder_names=sorted(allowed), source_frozen_complement=frozen,
        scope='Adjacent-block mechanism diagnostic only; not a new C0, quality result or exact-resume state.')
    meta, budget, state = b.checkpoint.save(model, directory, provenance=provenance)
    b.require(meta['checkpoint_kind'] == 'mixed_final_model'
        and {name: fit.tensor_record(value) for name, value in state.items()} == expected,
        'Canonical mixed serialization changed a committed full persistent tensor.')
    budget_audit = budget_contract(budget, source_budget)
    return meta, budget, expected, dict(groups=groups, frozen_complement=frozen,
        all_parameters_BF16=True, FP32_RoPE_preserved=True, only_existing_trained_D_changed=True,
        storage=budget_audit)
