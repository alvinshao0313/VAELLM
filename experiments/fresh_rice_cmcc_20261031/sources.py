"""Build all36 Rice gates from this completed common raw2 and its own after_o."""
from pathlib import Path

# This import closes the selected N namespace before any legacy sparse helper.
from experiments.fresh_fixed2_cmcc_20261030 import sources as control

import torch

from experiments.adaptive_class_20261030 import sparse

common, c = control.common, control.c
checkpoint, recovery = control.checkpoint, control.recovery
HERE = Path(__file__).resolve().parent
ROOT, HF = control.ROOT, control.HF
COMMON = common.ARTIFACTS / 'fresh_common_cmcc_20261030/formal'
RAW2_ID = 'f4669891-9028-4665-85a5-71c26ac45c33'
AFTER_O_ID = '7eeb82b4-3868-4cc5-9fa3-51870ee55894'
ARM = 'C1_Rice'


def source_pins():
    paths = set(HERE.glob('*.py'))
    paths.update(ROOT / 'experiments/adaptive_class_20261030' / name
        for name in ('__init__.py', 'sources.py', 'sparse.py', 'artifact.py'))
    paths.update((Path(sparse.rice.__file__),
        ROOT / 'experiments/cmcc_runtime_20261030/sequence.py',
        ROOT / 'experiments/cmcc_runtime_20261030/adaptive_evaluation.py'))
    return {**control.source_pins(), **{str(path.resolve()): c.sha(path) for path in paths}}


def bind(directory):
    c.require(Path(directory).resolve() == COMMON.resolve(),
        'C1 Rice consumes only the reviewed new common formal source; no old archive or short source.')
    binding = control.bind(directory)
    c.require(binding['source_raw2']['checkpoint_id'] == RAW2_ID
        and binding['source_after_o']['checkpoint_id'] == AFTER_O_ID,
        'This C1 requires its reviewed fresh raw2 and own common after_o UUIDs.')
    return binding


recipe = control.recipe


@torch.no_grad()
def build_model(binding):
    raw = Path(binding['source_raw2']['path'])
    after_o = Path(binding['source_after_o']['path'])
    meta, live_meta = c.read(raw / 'checkpoint_meta.json'), c.read(after_o / 'checkpoint_meta.json')
    model, state, strict = checkpoint.load_parent(raw, meta)
    c.require(not strict.missing_keys and not strict.unexpected_keys, 'Strictly load the complete actual new raw2.')
    live = torch.load(after_o / live_meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    before = {name: value for name, value in state.items() if '.mlp.gate_proj.' not in name}
    before_digest = c.digest(before)
    c.require(c.digest({name: value for name, value in live.items() if '.mlp.gate_proj.' not in name}) == before_digest,
        'Actual new raw2/own after_o non-gate complement must be exact before constructing Rice.')
    gates = [item for item in meta['converted_modules'] if item['name'].endswith('.gate_proj')]
    c.require({item['name'] for item in gates} == {f'model.layers.{i}.mlp.gate_proj' for i in range(36)},
        'Construct all36 gate payloads from this raw2; never borrow missing old patches.')
    rows = []
    for spec in gates:
        name = spec['name']
        relative = {key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + '.')}
        bundle, lengths, _resolved = control.from_native(spec, relative, dict(
            source_raw2_checkpoint_id=RAW2_ID, source_after_o_checkpoint_id=AFTER_O_ID,
            source_module=name, actual_raw2=str(raw), actual_live_after_o=str(after_o),
            purpose='fresh C1 Rice abs005; no new gate fit or third stage'))
        # Match the verified comparison's actual CUDA BF16 compact consumer.
        base = control.CompactRVQLinear(bundle, lengths, 'cuda:0')
        deployed = base.decode_weight().detach().cpu()
        fields, measured = sparse.build(bundle, live[name + '.weight'], deployed)
        module = sparse.Fixed2RiceLinear(bundle, fields)
        control.checkpoint.native.set_module_by_name(model, name, module)
        rows.append(dict(name=name, source_raw2=RAW2_ID, source_after_o=AFTER_O_ID,
            live_target_digest=c.digest({'weight': live[name + '.weight']}),
            compact_prefix_digest=c.digest({'weight': deployed}),
            sparse_fields_digest=c.digest(fields), sparse_persistent_fields=len(fields), **measured))
        print(dict(event='fresh_Rice_gate_constructed', **rows[-1]), flush=True)
        del base, deployed, fields, bundle, module
    c.require(c.digest({name: value for name, value in model.state_dict().items() if '.mlp.gate_proj.' not in name}) == before_digest,
        'Building Rice changed a non-gate field of the actual common parent.')
    groups = checkpoint.inventories(model)
    c.require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [144, 36, 72],
        'Fresh Rice raw inventory must be144 native/36 compact/72 live dense.')
    result = dict(inventories=groups, original_non_gate_fields_exact=True, non_gate_digest=before_digest,
        raw2_weight_sha256=c.sha(raw / meta['state_dict_file']),
        live_after_o_weight_sha256=c.sha(after_o / live_meta['state_dict_file']),
        imported_gate_count=len(gates), rice_payloads=rows,
        rice_sparse_persistent_fields=sum(row['sparse_persistent_fields'] for row in rows),
        rice_recipe=dict(rank_metric='sparse_residual_abs', top_p=.005, value_bits=8, index_bits=8,
            block_shape=[256, 256], rice_k=7, protected_columns_excluded=True),
        additional_gate_fit_updates=0, third_stage_fit_updates=0)
    del state, live, before
    torch.cuda.empty_cache()
    return model.eval().requires_grad_(False), result
