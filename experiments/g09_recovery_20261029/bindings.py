"""Actual G09 direct OPD32 identity; no Q/K2000 stage or old student targets."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SNAPSHOT = R / 'rotation_factor_probe/source_snapshot_20261005_01'
OPD = ROOT / 'experiments/cat_aligned_opd_20261028'
PROTECTION = ROOT / 'experiments/cat_aligned_protection_20261028'
REFERENCE = R / 'qwen_train_only_opd_hotpot_20261012'
PROFILE = 'G'
OUTPUT = Path('/data/star/VAELLM/g09_recovery_20261029/direct_opd32')
SOURCE_AUDIT = R / 'qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/G/09_joint1200/native_audit.json'
SOURCE_ID = 'e8e5efd4-655e-462c-b6e0-fedf4e2e0346'
ALIGNED = Path('/data/star/VAELLM/cat_aligned_recovery_20261028/G')
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
STEPS = {'formal32': list(range(32))}
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


aligned = module(OPD / 'bindings.py', '_g09_reference_OPD_bindings')
core, qa, torch = aligned.core, aligned.qa, aligned.torch
fit_core, engine, base, mmlu, hot = aligned.fit_core, aligned.engine, aligned.base, aligned.mmlu, aligned.hot
read, require, sha256 = core.read, core.require, core.sha256
binding, load_state = aligned.binding, aligned.load_state
teacher_recipe, model_config = aligned.teacher_recipe, aligned.model_config
selected_modules, load_inputs = aligned.selected_modules, aligned.load_inputs
LIMITS, BUDGET_FIELDS = deepcopy(aligned.LIMITS), aligned.BUDGET_FIELDS
CONTRACT = {**deepcopy(aligned.CONTRACT),
    'student_source': 'Verified G09 complete native endpoint, fresh Adam OPD32 directly; QK2000 omitted',
    'initialization': 'Source BF16 decoders -> FP32 masters; seed137; no optimizer/RNG continuation',
    'data_and_optimizer': 'Unchanged M1024 Q32 H96 train; M.5+QA.5 R0; 32 actual updates of original linear500 prefix'}

protection = module(PROTECTION / 'bindings.py', '_g09_runtime_config_semantics')
teacher_identity = protection.teacher_identity
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat


def source():
    audit = read(SOURCE_AUDIT)
    directory = Path(audit['checkpoint_dir']).resolve()
    meta = read(directory / 'checkpoint_meta.json')
    require(audit['status'] == 'verified' and audit['profile'] == 'G'
        and audit['checkpoint_id'] == meta['checkpoint_id'] == SOURCE_ID
        and audit['selected_global_step'] == 1200 and audit['code_bank_count'] == 504
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['fresh_strict_reload_state_exact'] and audit['saved_training_decoder_BF16_exact']
        and audit['actual_native_runtime_finalization'] and audit['selected_decoder_matches_fp32_cast_exactly']
        and audit['bit_budget']['budget_passed'], 'Use only the actual verified G09 native1200 endpoint.')
    # The completed aligned01 preparation already pinned this exact G09 input.
    pinned = read(ALIGNED / '01_qk_fineweb2000/prepared.json')
    require(pinned['source_checkpoint'] == str(directory) and pinned['source_checkpoint_id'] == SOURCE_ID
        and pinned['input_evidence_sha256'][str(SOURCE_AUDIT)] == sha256(SOURCE_AUDIT)
        and pinned['input_evidence_sha256'][str(directory / 'checkpoint_meta.json')]
            == sha256(directory / 'checkpoint_meta.json'), 'G09 differs from its completed aligned source binding.')
    check_stat(pinned['source_weight_stat'])
    return directory, meta, audit


def reused_gpu_evidence(meta):
    from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
    prepared_path = ALIGNED / '02_online_opd32/prepared.json'
    short_path = ALIGNED / '02_online_opd32/short2/report.json'
    prepared, short = read(prepared_path), read(short_path)
    other = read(Path(prepared['source_checkpoint']) / 'checkpoint_meta.json')
    expected = {item['name']: _inference_spec(item) for item in meta['converted_modules']}
    observed = {item['name']: _inference_spec(item) for item in other['converted_modules']}
    require(expected == observed and short['status'] == 'aligned_nativegreedy_OPD_short2_complete'
        and short['source_checkpoint_id'] == prepared['source_checkpoint_id'] == other['checkpoint_id']
        and short['prepared_sha256'] == sha256(prepared_path)
        and short['steps'] == 2 and short['source_restored_exact']
        and short['endpoint']['save_reload_install_exact']
        and (short_path.parent / 'run.exit').read_text().strip() == '0'
        and short['training']['frozen_state_before'] == short['training']['frozen_state_after']
        and short['training']['trainable_tensors'] == 1512, 'Existing OPD GPU evidence does not cover this geometry/runtime.')
    for key in ('native.py', 'fit.py', 'rollout.py', 'run.py'):
        item = prepared['bindings'][str(OPD / key)]
        require(binding(item['path']) == item, 'Changed reused OPD runtime: ' + key)
    return dict(report=binding(short_path), prepared=binding(prepared_path),
        geometry_matches_G09_exact=True, fresh_GPU_short2_repeated=False,
        scope='Reuse actual all252 nativegreedy/master/Adam/save/restore two-step evidence; source identity and frozen complement newly checked on CPU.')


def read_prepared(path=None):
    path = OUTPUT / 'prepared.json' if path is None else Path(path).resolve()
    prepared = read(path)
    directory, meta, _ = source()
    require(path == OUTPUT / 'prepared.json' and prepared['schema'] == 'G09_direct_online_OPD32_prepared_v1'
        and prepared['status'] == 'CPU_prepared_reused_GPU_evidence' and prepared['profile'] == PROFILE
        and prepared['source_checkpoint_id'] == SOURCE_ID and prepared['source_checkpoint'] == str(directory)
        and prepared['source_audit'] == binding(SOURCE_AUDIT)
        and prepared['actual_target_contract'] == CONTRACT and prepared['phases'] == STEPS
        and prepared['GPU_evidence_reuse'] == reused_gpu_evidence(meta), 'Changed G09 direct OPD source/contract/evidence.')
    for item in prepared['bindings'].values():
        require(binding(item['path']) == item, 'Changed bound direct OPD implementation: ' + item['path'])
    check_stat(prepared['source_weight_stat'])
    qa.check_config(prepared['record']['model_config'])
    qa.check_config(prepared['record']['teacher_identity']['carrier_model_config'])
    return prepared, prepared['record']
