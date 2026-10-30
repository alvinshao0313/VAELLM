"""Actual aligned02 source binding; immutable old inputs are separate from the student."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SNAPSHOT = R / 'rotation_factor_probe/source_snapshot_20261005_01'
PROFILE = os.environ.get('VAELLM_ALIGNED_PROFILE', 'G')
if PROFILE not in ('C', 'G'):
    raise ValueError('Choose fixed C or G profile.')
PARENT = Path('/data/star/VAELLM/cat_aligned_recovery_20261028') / PROFILE
STAGE02 = PARENT / '02_online_opd32'
SOURCE_AUDIT = STAGE02 / 'native_audit.json'
OUTPUT = PARENT / '03_protection'
ASSEMBLY = PARENT / '04_C1'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT))


def module(path, name):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(Path(path).parent))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path[:] = previous


aligned02 = module(ROOT / 'experiments/cat_aligned_opd_20261028/bindings.py', '_protection_readonly_aligned02')
core, qa, torch = aligned02.core, aligned02.qa, aligned02.torch
fit_core, engine, base, mmlu, hot = (aligned02.fit_core, aligned02.engine, aligned02.base,
    aligned02.mmlu, aligned02.hot)
read, require, sha256 = core.read, core.require, core.sha256
binding, load_state = aligned02.binding, aligned02.load_state
model_config, teacher_recipe = aligned02.model_config, aligned02.teacher_recipe
selected_modules = aligned02.selected_modules
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat

TASKS = ('boolq', 'rte', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa', 'piqa')
LIMITS = {'qasper': 128, 'narrativeqa': 128, 'hotpotqa': 32}
ARMS = {'scale32': {'steps': 32, 'kind': 'scale', 'learning_rate': 1e-3},
        'q64': {'steps': 64, 'kind': 'q', 'learning_rate': .03}}
CONTRACT = dict(source='Each profile actual completed independently audited02 native_final/final_model; fresh seed137 per arm',
    M='scale32 exact B_med64 groups0..31 aux16+Med16; q64 original N32 M1024 twice; .5raw-fullF+.5gold mixed once',
    QA='Original N32 groups0..31 once/twice; current arm nativegreedy Q/N128,H32,originalEOS; full-prefix full-F full151936 KL',
    R='Each profile actual frozen02 initial complete native model; seven original train128/tasks; all candidate branches; no HF restoration',
    loss='M.5 + QA.5 + R.5 independently; R mean7task(mean4doc(meanbranch(mean real continuation token KL)))',
    scope='All252 D1512 plus external protection252 FP32 leaves; BF16 compute;1659 frozen fields; no LoRA',
    optimizer='Continuous AdamW(.9,.95),eps1e-8,WD0,all1764 commonclip1.5,seed137; no reset at33',
    LR='D1e-6;scale1e-3 or q.03; each times1-(step-1)/500; genuine32/64 budgets',
    scale='FP32 logdelta0; BF16(s0.float*exp(delta)) then BF16(qsigned.float*rounded_scale.float)',
    q='FP32 signed-q0 master; clamp[-128,127],ties-even-round STE; fixed originalBF16scale; no trust restriction',
    runtime='FA2/FP32RoPE/group8/fullRHT; exact reversible D/protection/grad/nonemptyAdam nativegreedy transition; clear caches',
    endpoint='Only actual terminal BF16D plus nativeBF16scale or uint8q; no FP32masters/Adam/RNG; independently strict3423,504banks,2.6/rank0')
BUDGET_FIELDS = aligned02.BUDGET_FIELDS


def source():
    require(SOURCE_AUDIT.is_file(), 'Actual aligned02 is not independently audited yet: ' + str(SOURCE_AUDIT))
    audit = read(SOURCE_AUDIT)
    directory = STAGE02 / 'native_final/final_model'
    meta = read(directory / 'checkpoint_meta.json')
    report_path = STAGE02 / 'formal32/report.json'
    report = read(report_path)
    require(audit['schema'] == 'aligned_online_OPD_native_audit_v1' and audit['status'] == 'verified'
        and audit['profile'] == PROFILE and audit['checkpoint_dir'] == str(directory)
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['source_checkpoint_id'] == report['source_checkpoint_id']
        and report['status'] == 'aligned_nativegreedy_OPD_formal32_complete' and report['phase'] == 'formal32'
        and report['steps'] == 32 and report['profile'] == PROFILE
        and (report_path.parent / 'run.exit').read_text().strip() == '0'
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['frozen_tensors_compared'] == 1911 and audit['code_bank_count'] == 504
        and audit['fresh_strict_reload_state_exact'] and audit['source_geometry_unchanged']
        and audit['bit_budget']['budget_passed'] and audit['endpoint_report'] == binding(report_path)
        and audit['checkpoint_meta_sha256'] == sha256(directory / 'checkpoint_meta.json'),
        'Use only this profile actual verified aligned02 native endpoint, never09 or olde3.')
    check_hashes(audit['sources']['input_evidence_sha256'])
    check_stat(audit['checkpoint_weight_stat'])
    return directory, meta, audit


def teacher_identity(state, meta, names, directory):
    """Bind full-F to the actual shared-HF loader, not saved training flags."""
    from transformers import AutoConfig
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal
    from experiments.qwen3_w25.assemble_runtime_categories import _validate_runtime_configs
    teacher = teacher_recipe()
    old_dir = Path(teacher['model']['checkpoint'])
    qa.check_config(teacher)
    original = Path(meta['base_model_path']).resolve()
    require(original == Path(teacher['model']['hf_path']).resolve(),
        'Student source references another original HF teacher.')
    tokenizer = core.build_tokenizer(str(original))
    configs = [read(path / 'config.json') for path in (old_dir, Path(directory))]
    runtime = _validate_runtime_configs(*configs, tokenizer.pad_token_id)
    # Both canonical v6 loaders construct from this shared HF configuration and
    # only then install checkpoint tensors; phase calls override cache settings.
    effective = AutoConfig.from_pretrained(original, local_files_only=True)
    runtime.update(shared_HF_config=binding(original / 'config.json'),
        effective_pad_token_id=effective.pad_token_id, effective_use_cache=effective.use_cache,
        M_R_batch_pad_token_id=effective.pad_token_id if effective.pad_token_id is not None else 0,
        forward_policy='M/R: explicit left-pad mask and positions, use_cache=False; QA: ones mask/use_cache=False; '
                       'nativegreedy: explicit GenerationConfig pad/cache, use_model_defaults=False.')
    old = load_state(old_dir)
    actual = aligned02.outside_projections(state, names)
    count = _assert_state_equal(actual, aligned02.outside_projections(old, names), 'Full-HF F teacher nonprojection backbone')
    digest = core.mapping_digest({key: core.tensor_record(value) for key, value in actual.items()})
    return dict(teacher_kind='full7', carrier_model_config=teacher,
        carrier_source_checkpoint_id=teacher['model']['checkpoint_id'], restored_projection_count=252,
        nonprojection_tensors_compared=count, nonprojection_sha256=digest,
        new_student_backbone_matches_teacher_carrier_exact=True, configuration_runtime_fields=runtime,
        hf_metadata_sha256=teacher['model']['hf_metadata_sha256'], hf_weight_stats=teacher['model']['hf_weight_stats'],
        dataset_and_tokenizer='Original fixed train inputs and original tokenizer; full-F targets only.',
        reused_student_baseline=False, reused_student_retention=False)


def frozen_state(state, mutable):
    values = {key: core.tensor_record(value) for key, value in state.items() if key not in mutable}
    return dict(tensors=len(values), sha256=core.mapping_digest(values))


def inventory(state, names, kind):
    decoder, _ = hot.state_inventory(state, names)
    suffix = '.protected_input_scales' if kind == 'scale' else '.protected_input_qvalues'
    protection = {name + suffix: dict(shape=list(state[name + suffix].shape),
        elements=state[name + suffix].numel(), source_tensor=core.tensor_record(state[name + suffix])) for name in names}
    require(len(state) == 3423 and len(names) == 252 and len(decoder) == 1512 and len(protection) == 252,
        'Require original complete native all252 inventory.')
    frozen = frozen_state(state, set(decoder) | set(protection))
    require(frozen['tensors'] == 1659, 'Wrong source-derived D/protection complement.')
    return decoder, protection, frozen


def read_prepared(path=None):
    path = OUTPUT / 'prepared.json' if path is None else Path(path).resolve()
    value = read(path)
    directory, meta, _ = source()
    require(path == OUTPUT / 'prepared.json' and value['schema'] == 'CG_aligned_protection_prepared_v1'
        and value['status'] == 'CPU_prepared_GPU_unvalidated' and value['profile'] == PROFILE
        and value['source_checkpoint'] == str(directory) and value['source_checkpoint_id'] == meta['checkpoint_id']
        and value['source_audit'] == binding(SOURCE_AUDIT) and value['contract'] == CONTRACT and value['arms'] == ARMS,
        'Changed actual aligned02 protection preparation.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Changed bound protection implementation/material: ' + item['path'])
    check_stat(value['source_weight_stat'])
    qa.check_config(value['model_config'])
    qa.check_config(value['teacher_identity']['carrier_model_config'])
    return value
