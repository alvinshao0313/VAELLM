"""Fresh real-parent J32/B32 bindings; frozen03 modules keep their own identities."""
from copy import deepcopy
import importlib
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTECTION = ROOT / 'experiments/cat_aligned_protection_20261028'
previous = list(sys.path)
try:
    sys.path.insert(0, str(PROTECTION))
    mature = importlib.import_module('bindings')
    inputs = importlib.import_module('data')
    rollout = importlib.import_module('rollout')
    native_loader = importlib.import_module('native')
    deployment = importlib.import_module('deployment')
    retention = importlib.import_module('retention_cache')
    pair_loader = importlib.import_module('run')
    assembly = importlib.import_module('assemble')
    for value in (mature, inputs, rollout, native_loader, deployment, retention, pair_loader, assembly):
        if Path(value.__file__).resolve().parent != PROTECTION:
            raise ValueError('Frozen03 import was shadowed: ' + value.__name__)
finally:
    sys.path[:] = previous
sys.path.insert(0, str(mature.SNAPSHOT))
sys.dont_write_bytecode = True

R, SNAPSHOT, PROFILE, PYTHON = mature.R, mature.SNAPSHOT, mature.PROFILE, mature.PYTHON
PARENT = mature.PARENT
core, qa, torch = mature.core, mature.qa, mature.torch
fit_core, base, mmlu, engine = mature.fit_core, mature.base, mature.mmlu, mature.engine
read, require, binding, sha256 = mature.read, mature.require, mature.binding, mature.sha256
load_state, model_config, teacher_identity = mature.load_state, mature.model_config, mature.teacher_identity
TASKS, LIMITS, BUDGET_FIELDS = mature.TASKS, mature.LIMITS, mature.BUDGET_FIELDS
STAGES = {'J32': {'directory': '05_J32', 'selected': ['model.layers.18.mlp.down_proj'], 'decoder_tensors': 6,
                    'mutable_tensors': 8, 'frozen_tensors': 3415},
          'B32': {'directory': '06_B32', 'selected': [f'model.layers.9.mlp.{kind}_proj' for kind in ('gate', 'up', 'down')],
                    'decoder_tensors': 18, 'mutable_tensors': 18, 'frozen_tensors': 3405}}
CODE_KEYS = ('model.layers.18.mlp.down_proj.vq_weight', 'model.layers.18.mlp.down_proj.vq_weight_s1')
SUPPORT, TRUST, ETA = 482, .01, .001
R_CACHE = PARENT / '04_initial_R'
R_SCHEMA = 'CG_aligned04_native_R896_v1'
PV = R / 'qwen_c1_subspace_pv_accumulator128_20261018'
CONTRACT = dict(
    role='Fixed historical-chain benchmark; source/selected scope explicit for later ablation; not a learned model-independent layer rule',
    sources='J32 from this profile actual verified04_C1; B32 from its actual verified05_J32; fresh seed137/Adam per stage, never exact resume',
    data='Original N32 groups0..31 once; Maux1024/Q32/N32/H192; M32+QA8; .5raw-fullF+.5gold mixed once',
    QA='Current stage student nativegreedy; complete original prompt/EOS; Q/N128,H32; frozen full-F full151936 KL on same complete prefix',
    R='Both stages share this profile actual frozen04 initial R896, original train7x128/allbranches/15462tokens; zero HF restore; never02/oldC1/05 teacher',
    loss='M.5+QA.5+R.5 without renormalization; R task/doc/branch/real-continuation-token means',
    D='FP32 masters/BF16 compute, AdamW(.9,.95)/eps1e-8/WD0/commonclip1.5/LR1e-6*(1-(step-1)/500); honest32 updates',
    J32='L18down6D+2uint8banks; sixteen original FW/BW capture actualBF16gW into FP32, hook returnsNone/no extraVJP; D.step then U.step then projection',
    U='AbsoluteFP32[4096,12032], initialized once from first actual training-packedBF16W; continuous separateAdamW(.9,.95)/eps1e-8/WD0/eta.001constant/no clip/no division16/no reanchor',
    projection='NewD+oldcode actualBF16/group8 Wp; stable top482/48128 original32x32tiles; current plus48 true singlebit neighbors/stage0then1/position0..31/strictSSE/current wins ties/earliestbit',
    trust='Complete proposal actualgroup8; FP64 actual tile displacement; longest priority prefix <=.01||Wp_unprotected||; worse native targetSSE rolls back codes only; D/U/bothAdams preserved; zero flips/prefix0/rejection legal',
    B32='L9 gate/up/down18D only; noU/no code/no guard/no newloss;249 frozen caches; three independent same-signaturegroup8 chunks; actual gate geometry retained',
    runtime='FA2/FP32RoPE/fullRHT; preserve source frozen caches; nativegreedy swaps only selectedD BF16 views and restores sameFP32Parameter/storage/gradient/nonemptyAdam; clear selected/KV caches',
    endpoint='J32 only6BF16D+2uint8banks, B32 only18BF16D; inference-only honestterminal32 or genuine independent short2; no masters/U/gW/Adam/RNG; newUUID/strict3423/504banks/geometry/actual parent component ledger equal/logical<=2.6/rank0',
    observation='No old baseline100 or16/32 observations; no borrowed oldJ32/C1 panel; equal updates not equal compute')


def module(path, name):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


tiles = module(PV / 'pv_tiles.py', '_aligned_selected_original_tiles')
proxy = module(PV / 'pv_proxy.py', '_aligned_selected_original_proxy')


def stage_dir(stage):
    return PARENT / STAGES[stage]['directory']


def source04():
    audit_path = PARENT / '04_C1/assembly_report.json'
    require(audit_path.is_file(), 'Actual aligned04 is not independently audited yet: ' + str(audit_path))
    audit = read(audit_path)
    directory = PARENT / '04_C1/final_model'
    meta = read(directory / 'checkpoint_meta.json')
    prepared03 = mature.read_prepared()
    _, bm, _ = assembly.arm_source('scale32', prepared03)
    _, dm, _ = assembly.arm_source('q64', prepared03)
    sources = audit['sources']
    require(audit['schema'] == 'CG_aligned_C1_native_assembly_v1' and audit['status'] == 'verified'
        and audit['profile'] == sources['profile'] == PROFILE and audit['checkpoint_dir'] == str(directory)
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['fresh_strict_reload_state_exact']
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['code_bank_count'] == 504 and audit['source_geometry_unchanged']
        and audit['nonprojection_tensors_compared'] == 147 and audit['replaced_projection_count'] == 144
        and audit['bit_budget']['budget_passed'] and sources['aligned02_parent_checkpoint_id'] == prepared03['source_checkpoint_id']
        and sources['parent_checkpoint_ids'] == [bm['checkpoint_id'], dm['checkpoint_id']]
        and sources['base_audit'] == binding(mature.OUTPUT / 'scale32/native_audit.json')
        and sources['donor_audit'] == binding(mature.OUTPUT / 'q64/native_audit.json')
        and meta['checkpoint_id'] not in sources['parent_checkpoint_ids']
        and audit['checkpoint_meta_sha256'] == sha256(directory / 'checkpoint_meta.json'),
        'Require this profile actual verified04 whole-projection assembly from both real03 arms.')
    mature.check_hashes(sources['input_evidence_sha256'])
    mature.check_stat(audit['checkpoint_weight_stat'])
    return directory, meta, audit, audit_path


def source(stage):
    if stage == 'J32':
        return source04()
    require(stage == 'B32', 'Choose fixed J32 or B32 stage.')
    root = stage_dir('J32')
    audit_path = root / 'native_audit.json'
    require(audit_path.is_file(), 'Actual aligned05 is not independently audited yet: ' + str(audit_path))
    audit = read(audit_path)
    directory = root / 'native_final/final_model'
    meta = read(directory / 'checkpoint_meta.json')
    report_path = root / 'formal/report.json'
    report = read(report_path)
    _, parent_meta, _, _ = source04()
    require(audit['schema'] == 'CG_aligned_selected_native_audit_v1' and audit['status'] == 'verified'
        and audit['profile'] == report['profile'] == PROFILE and audit['stage'] == report['stage'] == 'J32'
        and audit['phase'] == report['phase'] == 'formal' and audit['actual_steps'] == audit['planned_steps'] == report['steps'] == 32
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['checkpoint_dir'] == str(directory)
        and audit['source_checkpoint_id'] == report['source_checkpoint_id'] == parent_meta['checkpoint_id']
        and audit['endpoint_report'] == binding(report_path) and report['contract'] == CONTRACT
        and report['status'] == 'aligned_selected_formal_complete' and (report_path.parent / 'run.exit').read_text().strip() == '0'
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['frozen_tensors_compared'] == 3415 and audit['code_bank_count'] == 504
        and audit['fresh_strict_reload_state_exact'] and audit['source_geometry_unchanged'] and audit['bit_budget']['budget_passed']
        and audit['checkpoint_meta_sha256'] == sha256(directory / 'checkpoint_meta.json'), 'Require actual same-profile complete verified05 J32.')
    mature.check_hashes(audit['sources']['input_evidence_sha256'])
    mature.check_stat(audit['checkpoint_weight_stat'])
    return directory, meta, audit, audit_path


def scope(state, meta, stage):
    spec = STAGES[stage]
    selected = spec['selected']
    decoder = {key: dict(shape=list(v.shape), elements=v.numel()) for key, v in state.items()
        if any(key.startswith(name + '._parallel_stage_decoder.') for name in selected)}
    codes = {key: dict(shape=list(state[key].shape), elements=state[key].numel()) for key in CODE_KEYS} if stage == 'J32' else {}
    mutable = set(decoder) | set(codes)
    require(len(state) == 3423 and len(decoder) == spec['decoder_tensors'] and len(mutable) == spec['mutable_tensors']
        and all(state[key].dtype == torch.bfloat16 for key in decoder) and all(state[key].dtype == torch.uint8 for key in codes),
        'Actual selected native scope/dtype changed.')
    inventory = {item['name']: item for item in meta['converted_modules']}
    geometry = {name: inventory[name] for name in selected}
    for name, item in geometry.items():
        down = name.endswith('down_proj')
        require(item['transpose'] == down and item['residual_stages'] == 2 and item['parallel_parts'] == 1
            and item['protected_input_indices']['shape'] == [256 if down else 64]
            and item['protected_output_indices'] is None and item['weight_rotation']['block_size'] == 32,
            'Keep real selected MLP native geometry: ' + name)
        if stage == 'J32':
            require((item['out_features'], item['in_features'], item['compressed_in_features']) == (4096, 12288, 12032)
                and item['stage_codebook_dims'] == [32, 32]
                and all(d['in_dim'] == 48 and d['out_dim'] == 32 and d['hidden_dim'] == 128 for d in item['stage_decoders'])
                and all(tuple(state[key].shape) == (1540096, 1, 6) for key in codes), 'Keep actual L18 K48/D32/S2 top482 support.')
    frozen = mature.frozen_state(state, mutable)
    require(frozen['tensors'] == spec['frozen_tensors'], 'Wrong source-derived frozen complement.')
    return dict(selected=selected, decoder_parameters=decoder, code_parameters=codes, mutable=sorted(mutable),
        trainable_elements=sum(v['elements'] for v in decoder.values()), frozen=frozen, geometry=geometry)


def read_prepared(stage, path=None):
    expected = stage_dir(stage) / 'prepared.json'
    path = expected if path is None else Path(path).resolve()
    record = read(path)
    directory, meta, _, audit_path = source(stage)
    r_directory, r_meta, _, r_audit_path = source04()
    require(path == expected and record['schema'] == 'CG_aligned_selected_prepared_v1'
        and record['status'] == 'CPU_prepared_GPU_unvalidated' and record['profile'] == PROFILE and record['stage'] == stage
        and record['contract'] == CONTRACT and record['stages'] == STAGES and record['source_checkpoint'] == str(directory)
        and record['source_checkpoint_id'] == meta['checkpoint_id'] and record['source_audit'] == binding(audit_path)
        and record['R_source_checkpoint'] == str(r_directory) and record['R_source_checkpoint_id'] == r_meta['checkpoint_id']
        and record['R_source_audit'] == binding(r_audit_path) and record['R_cache'] == binding(R_CACHE / 'cache.json'),
        'Changed actual selected-stage/04-retention preparation.')
    for item in record['bindings'].values():
        require(binding(item['path']) == item, 'Changed selected implementation/material: ' + item['path'])
    mature.check_stat(record['source_weight_stat'])
    qa.check_config(record['model_config'])
    qa.check_config(record['teacher_identity']['carrier_model_config'])
    return record


def runtime_record(prepared, phase, directory):
    require(phase in ('short2', 'formal'), 'Choose genuine short2 or formal32.')
    return {**deepcopy(prepared), 'phase': phase, 'run_dir': str(directory),
        'group_indices': [19, 0] if phase == 'short2' else list(range(32)),
        'R_group_indices': [17, 28] if phase == 'short2' else list(range(32)),
        'steps': 2 if phase == 'short2' else 32, 'planned_steps': 2 if phase == 'short2' else 32,
        'prepared_sha256': sha256(stage_dir(prepared['stage']) / 'prepared.json')}
