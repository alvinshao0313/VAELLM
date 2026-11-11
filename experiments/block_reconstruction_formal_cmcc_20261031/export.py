"""Canonical full BR36 BF16 export with the exact C0 frozen complement."""
import hashlib
import json

from . import bindings as b


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def geometry(meta):
    return json_sha({key: meta[key] for key in ('native_modules', 'compact_modules', 'pending_dense_modules')})


def decoder_names(meta):
    names = set()
    for family, prefix in (('native_modules', '_parallel_stage_decoder.'), ('compact_modules', 'decoders.')):
        for module in meta[family]:
            selected = {name for name in meta['tensor_table'] if name.startswith(module['name']+'.'+prefix)}
            b.require(selected, 'Every actual projection needs its existing persistent decoder leaves: '+module['name'])
            names.update(selected)
    return names


@b.torch.no_grad()
def save(model, directory, identity, source_meta, source_records, allowed, block_reports, source_state):
    b.deployment.deployment_contract(model)
    b.require(set(allowed) == decoder_names(source_meta) and len(block_reports) == len(b.LAYERS),
              'Export the actual existing D union after all36 completed blocks only.')
    expected, frozen = b.deployment.frozen_contract(model, source_records, allowed)
    provenance = dict(schema='BR36_H96_32_provenance_v1', validation_only=False, prototype_only=False,
        execution=identity, contract=b.CONTRACT, source_C0_binding=identity['source_C0_binding'],
        completed_layers=list(b.LAYERS), updates_per_layer=32, total_updates=1152,
        group_indices=list(b.GROUPS), trained_decoder_names=sorted(allowed), source_frozen_complement=frozen,
        source_geometry_sha256=geometry(source_meta), source_state=source_state, block_reports=block_reports,
        scope='Complete fixed-code D-only BR36 endpoint on actual fresh C0; no quality or exact-resume claim.')
    meta, budget, state = b.checkpoint.save(model, directory, provenance=provenance)
    b.require(meta['checkpoint_kind'] == 'mixed_final_model' and geometry(meta) == geometry(source_meta)
        and {name: b.fit.tensor_record(value) for name, value in state.items()} == expected,
        'Canonical mixed serialization changed complete committed state or the source geometry.')
    storage = b.deployment.budget_contract(budget, identity['source_C0_binding']['bit_budget'])
    return meta, budget, expected, dict(frozen_complement=frozen, storage=storage,
        source_geometry_sha256=geometry(source_meta), all_parameters_BF16=True, FP32_RoPE_preserved=True,
        trained_D_leaves=len(allowed), frozen_fields=frozen['tensors'])
