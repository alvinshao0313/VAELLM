"""Only replace actual down modules; preserve the complete non-down complement."""
from . import bindings as b
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear


def complement(state):
    return {name: value for name, value in state.items() if '.mlp.down_proj.' not in name}


def frozen_down(module):
    return {name: value for name, value in module.state_dict().items()
        if name != 'length_frame' and name != 'codes_2' and not name.startswith('decoders.2.')}


def frozen_metadata(module):
    spec = module.metadata()
    return dict(geometry=spec['geometry'], vector_dim=spec['vector_dim'], n_vectors=spec['n_vectors'],
        first_two_stages=spec['stages'][:2])


def topology(model, *, mode):
    groups = b.checkpoint.inventories(model)
    b.c.require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [180, 72, 0]
        and set(groups['compact']) == set(b.NAMES) | {name.replace('down_proj', 'gate_proj') for name in b.NAMES},
        'Require real180 native,36 original compact gate,36 compact down and no dense projections.')
    for name in b.NAMES:
        module = model.get_submodule(name)
        b.c.require(type(module) is CompactRVQLinear and module.resolved['n_vectors'] == b.N
            and module.resolved['vector_dim'] == 16 and module.resolved['geometry']['transpose'], 'Actual down compact geometry changed.')
        stages = module.resolved['stages']
        expected = 2 if mode == 'control' else 3
        b.c.require(mode == 'fixture' or len(stages) == expected, 'Every formal down must have its actual selected stage count.')
        b.c.require(all(item['decoder_spec']['in_dim'] == 24 for item in stages)
            and all(item['decoder_spec']['out_dim'] == 16 for item in stages)
            and bool(((module.lengths == 2) | (module.lengths == 3)).all())
            and len(module.codes_0) == len(module.codes_1) == b.N, 'Keep full original banks and strict2->3 routes.')
    return groups


def build(raw, specs, source):
    model, meta, loaded, _budget = b.checkpoint.load(b.RAW)
    b.c.require(meta['checkpoint_id'] == b.RAW_ID and not loaded.missing_keys and not loaded.unexpected_keys,
        'Start the complete new bridge from the genuine original D16 raw checkpoint.')
    before = b.c.digest(complement(raw))
    b.c.require(b.c.digest(complement(model.state_dict())) == before, 'Strict raw load changed a non-down field.')
    bundles, down, metadata = {}, {}, {}
    for spec in specs:
        bundle, lengths, _resolved, _fields = b.short.prefix(raw, spec, source)
        module = CompactRVQLinear(bundle, lengths)
        b.checkpoint.native.set_module_by_name(model, spec['name'], module)
        bundles[spec['name']] = bundle
        down[spec['name']] = b.c.digest(frozen_down(module))
        metadata[spec['name']] = frozen_metadata(module)
    topology(model, mode='control')
    b.c.require(b.c.digest(complement(model.state_dict())) == before, 'Converting only down changed the native/gate/backbone complement.')
    return model, bundles, dict(non_down_digest=before, frozen_down_digests=down, frozen_down_metadata=metadata)


def verify_frozen(model, identity):
    b.c.require(b.c.digest(complement(model.state_dict())) == identity['non_down_digest'], 'A non-down persistent field changed.')
    for name in b.NAMES:
        b.c.require(b.c.digest(frozen_down(model.get_submodule(name))) == identity['frozen_down_digests'][name],
            'A frozen down stage/code/protection/sign field changed: ' + name)
        b.c.require(frozen_metadata(model.get_submodule(name)) == identity['frozen_down_metadata'][name],
            'Frozen down geometry/stage decoder specification/normalization changed: ' + name)


def provenance(arm, *, phase, source):
    return dict(schema=b.SCHEMA, phase=phase, arm=arm, source_raw_id=b.RAW_ID, live_target_id=b.LIVE_ID,
        source_binding=source, actual_category='down_proj', actual_layers=36,
        actual_stage3_updates_per_layer=10000 if arm == 'k24' and phase == 'formal' else 0,
        encoder_init=('Fresh per layer2031; never resume a short deployment' if phase == 'formal'
            else 'No new fitting; real short3 deployments used only as a CPU bridge fixture'),
        recovery_updates=0, no_automatic_suffix=True)
