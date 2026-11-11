"""Bind C0 exclusively to one completed new BF16 common prerequisite."""
import importlib
from pathlib import Path

from experiments.fresh_common_cmcc_20261030 import bindings as common

common.s.enable_snapshot()
# Legacy helpers prepend the repository to sys.path. Close namespace child
# imports before those helpers load so the selected complete N runtime stays N.
for _package_name in ('train_utils', 'litebsq', 'rotation', 'e2e_common'):
    _package = importlib.import_module(_package_name)
    _package.__path__ = [str(common.s.SNAPSHOT / _package_name)]

import torch

from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.native_prefix import from_native
from experiments.adaptive_vector_rvq import common as c
from train_utils.cat_runtime_state_v6 import validate_cat_runtime_identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HF = common.HF


def source_pins():
    paths = set(HERE.glob('*.py'))
    for folder, names in (
        ('adaptive_class_20261030', ('continuation.py', 'initialization.py', 'run_pipeline.py')),
        ('adaptive_native_model_20261029', ('__init__.py', 'checkpoint.py', 'recovery.py', 'consumer.py',
            'codec.py', 'spec.py', 'native_prefix.py', 'training_checkpoint.py', 'handoff_validation.py')),
        ('adaptive_vector_rvq', ('common.py', 'codec.py', 'length_codec_probe.py')),
    ):
        paths.update(ROOT / 'experiments' / folder / name for name in names)
    return {**common.source_pins(), **{str(path.resolve()): c.sha(path) for path in paths}}


def bind(directory):
    """Resolve actual paths from terminal saves; no historical archive or UUID."""
    directory = Path(directory).resolve(strict=True)
    run = c.read(directory / 'run.json')
    request = c.read(directory / 'request.json')
    c.require(run['status'] == 'common_prerequisite_complete' and run['exit_code'] == 0
        and run['validation_only'] is False and request['validation_only'] is False
        and request['formal_parent_eligible'] is True and Path(request['output_dir']).resolve() == directory
        and Path(request['base_model_path']).resolve() == HF.resolve()
        and run['request_sha256'] == c.sha(directory / 'request.json')
        and run['source_sha256'] == request['source_sha256'],
        'C0 requires this completed formal BF16 common prerequisite, never a short or old parent.')
    verification = c.read(directory / 'boundary_verification.json')
    common.s.check_pins(request['source_sha256'])
    c.require(verification['status'] == 'verified' and verification['validation_only'] is False
        and verification['non_gate_after_o_complement_exact'] and verification['runtime_identity_equal']
        and verification['gate_recovery_updates'] == 0, 'The actual shared raw2 boundary was not verified.')
    after_o, after_event = common.s.own_after_o(request)
    gate = c.read(directory / 'gate_raw2.json')
    prefix = c.read(directory / 'prefix.json')
    c.require(gate['status'] == 'raw2_completed' and gate['exit_code'] == 0
        and gate['validation_only'] is False and gate['formal_parent_eligible'] is True
        and gate['parent_event'] == after_event and gate['recoveries'] == [] and gate['seed_calls'] == [31]
        and gate['request_sha256'] == c.sha(directory / 'request.json')
        and [item['category'] for item in gate['actual_groups']] == ['gate_proj']
        and gate['actual_groups'][0]['actual_steps_per_stage'] == 10000
        and gate['actual_groups'][0]['planned_steps_per_stage'] == 10000
        and gate['actual_groups'][0]['stages'] == 2, 'Require the real common B S2x10K gate before recovery.')
    c.require(prefix['validation_only'] is False and prefix['formal_parent_eligible'] is True
        and [item['category'] for item in prefix['actual_groups']] == list(recovery.CATEGORIES[:4])
        and all(item['actual_steps_per_stage'] == item['planned_steps_per_stage'] == 10000
            and item['stages'] == 2 for item in prefix['actual_groups'])
        and [item['category'] for item in prefix['recoveries']] == list(recovery.CATEGORIES[:4])
        and all(item['status'] == 'completed' for item in prefix['recoveries'])
        and [item['exit_code'] for item in run['phases']] == [0, 0], 'Common formal attention exposure is incomplete.')
    raw_event = gate['saved_boundaries'][-1]
    raw = Path(raw_event['path']).resolve(strict=True)
    c.require(raw.is_relative_to(directory / 'gate_raw2/native') and raw.name == 'round_base'
        and run['gate_raw2'] == raw_event and run['after_o'] == after_event,
        'C0 can consume only the newly saved canonical raw2 from this run.')
    meta = c.read(raw / 'checkpoint_meta.json')
    common.validate_raw2_meta(meta, HF)
    c.require(meta['checkpoint_id'] == raw_event['checkpoint_id'] == verification['raw2_checkpoint_id']
        and after_event['checkpoint_id'] == verification['after_o_checkpoint_id']
        and c.sha(raw / 'checkpoint_meta.json') == raw_event['metadata_sha256']
        and c.sha(raw / 'cat_runtime_state.pt') == raw_event['runtime_sha256']
        and common.s.stat(raw / meta['state_dict_file']) == raw_event['weight_stat'], 'The actual formal raw2 changed.')
    files = ('request.json', 'run.json', 'prefix.json', 'gate_raw2.json', 'boundary_verification.json')
    return dict(schema='fresh_C0_fixed2_parent_v1', common_directory=str(directory),
        common_files_sha256={str(directory / name): c.sha(directory / name) for name in files},
        source_after_o=after_event, source_raw2=raw_event, source_common_pid=run['pid'],
        source_common_start_ticks=run['start_ticks'], source_runtime_identity=gate['runtime_identity'],
        common_elapsed_seconds=run['elapsed_seconds'], formal_parent_eligible=True)


def recipe(output, binding):
    raw = Path(binding['source_raw2']['path'])
    runtime = torch.load(raw / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    values = common.recipe.argv('B', HF, output)
    cat, hf, training, vae, cfgs, identity, _normalized = common.recipe.parse('B', values)
    validate_cat_runtime_identity(runtime['runtime_identity'], identity)
    c.require(identity == binding['source_runtime_identity'] and cat.seed == training.seed == 31
        and vae.model_path == str(HF) and cfgs['gate_proj'].codebook_dim == cfgs['gate_proj'].codebook_bits == 64,
        'Use the actual new parent runtime and unchanged seven-category fresh B recipe.')
    stage = recovery.canonical.resolve_cat_after_category_stage(cat, training, category='gate_proj', round_idx=4)
    c.require(stage.mode == 'remaining_lora_prefix_decoder' and stage.config.opt.steps == 500
        and stage.config.opt.batch_size == 4 and stage.config.data.model_max_length == 1024
        and stage.config.opt.gradient_checkpointing and stage.config.data.dynamic_padding
        and stage.config.lora.rank == 12 and stage.config.opt.resolved_decoder_lr() == 1e-5
        and stage.config.opt.learning_rate == 1e-4, 'Keep the original500/B4/1024/D+future-dense-rank12 recipe.')
    return cat, hf, training, vae, cfgs, runtime


@torch.no_grad()
def build_model(binding):
    raw = Path(binding['source_raw2']['path'])
    meta = c.read(raw / 'checkpoint_meta.json')
    model, state, strict = checkpoint.load_parent(raw, meta)
    c.require(not strict.missing_keys and not strict.unexpected_keys, 'Load the complete actual native raw2 strictly.')
    before = {name: value for name, value in state.items() if '.mlp.gate_proj.' not in name}
    before_digest = c.digest(before)
    gates = [item for item in meta['converted_modules'] if item['name'].endswith('.gate_proj')]
    for spec in gates:
        name = spec['name']
        relative = {key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + '.')}
        bundle, lengths, _resolved = from_native(spec, relative, dict(
            source_raw2_checkpoint_id=meta['checkpoint_id'], source_after_o_checkpoint_id=binding['source_after_o']['checkpoint_id'],
            source_module=name, actual_raw2=str(raw), purpose='C0 ordinary fixed2; no fitting, third stage or Rice'))
        checkpoint.native.set_module_by_name(model, name, CompactRVQLinear(bundle, lengths))
    actual = {name: value for name, value in model.state_dict().items() if '.mlp.gate_proj.' not in name}
    c.require(c.digest(actual) == before_digest, 'Importing C0 must preserve all144 native and72 live dense fields exactly.')
    groups = checkpoint.inventories(model)
    c.require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [144, 36, 72],
        'C0 raw state must contain exactly144 native,36 fixed2 compact,72 live dense projections.')
    del state, before, actual
    return model.eval().requires_grad_(False), dict(inventories=groups,
        original_non_gate_fields_exact=True, non_gate_digest=before_digest,
        raw2_weight_sha256=c.sha(raw / meta['state_dict_file']), imported_gate_count=len(gates),
        additional_gate_fit_updates=0, third_stage_fit_updates=0, rice_sparse_fields=0)
