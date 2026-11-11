"""Bind the completed E256 raw2 to the genuine same-parent C0 protocol."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from experiments.fresh_fixed2_cmcc_20261030 import sources as control
from experiments.cat_geometry_cohort_cmcc_20261102 import bindings as geometry

common, c, torch, checkpoint, recovery = control.common, control.c, control.torch, control.checkpoint, control.recovery
ROOT, HF = control.ROOT, control.HF
HERE = Path(__file__).resolve().parent
RAW_ROOT = geometry.OUTPUT / 'encoder256'
CONTROL_ROOT = geometry.ARTIFACTS / 'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0'
CONTROL_EVAL = geometry.RESULT.parent / 'fresh_fixed2_cmcc_20261030/single_20261030_v3'
AFTER_O = '7eeb82b4-3868-4cc5-9fa3-51870ee55894'
RAW_ID = '66122de5-a5da-41a6-a64a-ca03b2c3868d'


def source_pins():
    paths = set(HERE.glob('*.py')) | set(control.HERE.glob('*.py'))
    paths.add(Path(geometry.__file__))
    return {**control.source_pins(), **{str(path.resolve()): c.sha(path) for path in paths}}


def bind(directory):
    original = control.bind(directory)
    run, fit, strict, mmlu, readout = (c.read(path) for path in (
        RAW_ROOT / 'run.json', RAW_ROOT / 'formal/report.json', RAW_ROOT / 'formal/strict_reload.json',
        RAW_ROOT / 'mmlu.json', RAW_ROOT / 'readout.json'))
    event = run['raw2']
    raw = Path(event['path']).resolve(strict=True)
    c.require(original['source_after_o']['checkpoint_id'] == AFTER_O
        and fit['source_binding']['after_o']['checkpoint_id'] == AFTER_O
        and run['status'] == 'completed_fit_fresh_full_mmlu_readout' and run['exit_code'] == 0
        and fit['status'] == 'raw2_completed' and fit['exit_code'] == 0 and not fit['validation_only']
        and fit['arm'] == 'encoder256' and fit['phase'] == 'formal'
        and fit['seed_calls'] == [31] and fit['gate_recovery_updates'] == 0
        and event == fit['raw2'] and event['checkpoint_id'] == RAW_ID,
        'Require the completed formal E256 raw2 from the genuine same after_o, never a short or refit.')
    c.require(geometry.event(raw) == event and raw.is_relative_to(RAW_ROOT / 'formal')
        and strict['status'] == 'verified_native_full_state_strict' and strict['exit_code'] == 0
        and strict['checkpoint_id'] == RAW_ID and strict['expected_state_equal']
        and strict['non_gate_after_o_complement_exact'] and strict['non_gate_tensor_count'] == 2091
        and mmlu['status'] == 'completed_full_mmlu' and mmlu['checkpoint_id'] == RAW_ID
        and readout['status'] == 'completed_deployment_readout' and readout['checkpoint_id'] == RAW_ID,
        'The real E256 identity/full strict/non-gate complement/stage evaluations are incomplete or changed.')
    groups = fit['actual_groups']
    c.require(len(groups) == 1 and groups[0]['category'] == 'gate_proj' and groups[0]['stages'] == 2
        and groups[0]['actual_updates_per_stage'] == 10000 and groups[0]['shuffle_seed'] == 400031
        and groups[0]['vector_batch_per_linear'] == 4096 and groups[0]['status'] == 'fitted_and_encoded',
        'E256 must have both complete original 10K stages and its original vector exposure.')
    meta = c.read(raw / 'checkpoint_meta.json')
    geometry.validate_raw2(meta, 'encoder256')
    common.validate_raw2_meta(meta, HF)
    baseline = c.read(CONTROL_ROOT / 'report.json')
    quality = c.read(CONTROL_EVAL / 'evaluation.json')
    c.require(baseline['status'] == 'completed_fresh_C0_fixed2_pipeline' and baseline['exit_code'] == 0
        and baseline['source_binding'] == original and baseline['source_after_o'] == AFTER_O
        and baseline['source_raw2'] == original['source_raw2']['checkpoint_id']
        and quality['checkpoint_id'] == baseline['final_checkpoint_id'],
        'Bind the actual completed same-parent C0, not the old-parent native B suffix.')
    proofs = (RAW_ROOT / 'run.json', RAW_ROOT / 'formal/report.json', RAW_ROOT / 'formal/strict_reload.json',
        RAW_ROOT / 'mmlu.json', RAW_ROOT / 'readout.json', CONTROL_ROOT / 'report.json', CONTROL_EVAL / 'evaluation.json')
    binding = deepcopy(original)
    binding.update(schema='same_parent_E256_fixed2_parent_v1', source_raw2=event,
        source_runtime_identity=fit['runtime_identity'], control_binding=original,
        evidence_sha256={str(path): c.sha(path) for path in proofs},
        control_checkpoint_id=baseline['final_checkpoint_id'], control_report=str(CONTROL_ROOT / 'report.json'),
        control_quality_report=str(CONTROL_EVAL / 'evaluation.json'),
        fixed_predecessor='Completed E256 native raw2 reused; only the original C0 full suffix is new.')
    return binding


def recipe(output, binding):
    raw = Path(binding['source_raw2']['path'])
    runtime = torch.load(raw / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    _argv, parsed, identity, _normalized = geometry.recipe('encoder256', output)
    cat, hf, training, vae, cfgs = parsed
    control_values = control.recipe(output, binding['control_binding'])
    original_cat, _hf, original_training, _vae, original_cfgs, _runtime = control_values
    control_identity = deepcopy(_runtime['runtime_identity'])
    control_identity['resolved_category_runtime']['gate_proj']['base_ch'] = 256
    control_identity['experiment_runtime'] = identity['experiment_runtime']
    c.require(runtime['runtime_identity'] == identity == binding['source_runtime_identity'] == control_identity
        and cat.seed == training.seed == original_cat.seed == original_training.seed == 31
        and all(asdict(cfgs[name]) == asdict(original_cfgs[name]) for name in recovery.CATEGORIES if name != 'gate_proj')
        and cfgs['gate_proj'].base_ch == 256 and cfgs['gate_proj'].decoder_base_ch == 128,
        'Only the completed gate encoder factor may differ; suffix configs and actual runtime must match C0.')
    for category, index in zip(('gate_proj', 'up_proj', 'down_proj'), (4, 5, 6), strict=True):
        actual = recovery.canonical.resolve_cat_after_category_stage(cat, training, category=category, round_idx=index)
        expected = recovery.canonical.resolve_cat_after_category_stage(original_cat, original_training, category=category, round_idx=index)
        c.require(asdict(actual) == asdict(expected), 'The actual original500 recovery protocol changed: ' + category)
    return cat, hf, training, vae, cfgs, runtime


def build_model(binding):
    # The unchanged C0 loader performs strict native readback and the actual
    # from_native -> CompactRVQLinear conversion, retaining all non-gate fields.
    return control.build_model(binding)


def verify_suffix_initializations(binding, stages):
    baseline = c.read(binding['control_report'])
    evidence = []
    for actual, reference in zip(stages, baseline['stages'], strict=False):
        if 'next_live_category_fit' not in actual:
            continue
        left, right = actual['next_live_category_fit'], reference['next_live_category_fit']
        fields = ('category', 'actual_stages', 'actual_updates_per_stage', 'schedule_steps',
            'vector_batch_per_layer', 'shuffle_seed', 'fresh_VAE_initialization_seed', 'actual_optimizer_initializations')
        c.require(all(left[key] == right[key] for key in fields),
            'Own live suffix initializations, optimizer shapes/LR, seeds or budgets differ from genuine C0.')
        evidence.append(dict(category=left['category'], actual_optimizer_initializations_equal_to_C0=True,
            initial_parameter_digests=[row['initial_parameter_digest'] for row in left['actual_optimizer_initializations']],
            shuffle_seed=left['shuffle_seed'], fresh_VAE_initialization_seed=left['fresh_VAE_initialization_seed']))
    return evidence
