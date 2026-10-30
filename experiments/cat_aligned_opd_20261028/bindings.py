"""Separate actual aligned-student identity from the unchanged full-HF teacher."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SNAPSHOT = R / 'rotation_factor_probe/source_snapshot_20261005_01'
REFERENCE = R / 'qwen_train_only_opd_hotpot_20261012'
PROFILE = os.environ.get('VAELLM_ALIGNED_PROFILE', 'G')
if PROFILE not in ('C', 'G'):
    raise ValueError('Use the fixed C or G aligned profile.')
PARENT = Path('/data/star/VAELLM/cat_aligned_recovery_20261028') / PROFILE
STAGE01 = PARENT / '01_qk_fineweb2000'
OUTPUT = PARENT / '02_online_opd32'
SOURCE_AUDIT = STAGE01 / 'native_audit.json'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
sys.dont_write_bytecode = True
# The extended full-RHT implementation must win before any native loader import.
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


reference = module(REFERENCE / 'opd_hotpot_bindings.py', '_aligned_OPD_reference')
core, qa, torch = reference.core, reference.qa, reference.torch
fit_core, engine, base, mmlu = reference.fit_core, reference.engine, reference.base, reference.mmlu
hot = reference.hot
CONTRACT = deepcopy(reference.CONTRACT)
LIMITS, STEPS = deepcopy(reference.LIMITS), deepcopy(reference.STEPS)
read, require, sha256 = core.read, core.require, core.sha256

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat

BUDGET_FIELDS = ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights',
                 'components', 'outside_linear_bits', 'code_padding_bits')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def load_state(directory, meta=None):
    directory = Path(directory)
    meta = read(directory / 'checkpoint_meta.json') if meta is None else meta
    return torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)


def teacher_recipe():
    """This record describes the old, independently restored F teacher carrier only."""
    prepared = read(reference.HOT / 'prepared_hotpot.json')
    old = prepared['arms']['F_A_HOT']
    require(prepared['source_checkpoint_id'] == reference.SOURCE_ID and old['teacher_kind'] == 'full7'
            and old['student_scope'] == 'all' and prepared['retention_lambda'] == 0,
            'Changed immutable reference F teacher/data recipe.')
    return deepcopy(prepared['model_config'])


def source():
    require(SOURCE_AUDIT.is_file(), 'Aligned 01 is not independently audited yet: ' + str(SOURCE_AUDIT))
    audit, training, execution = (read(path) for path in
        (SOURCE_AUDIT, STAGE01 / 'training.json', STAGE01 / 'execution.json'))
    directory = Path(training['training_result']['saved_model_dir']).resolve()
    meta = read(directory / 'checkpoint_meta.json')
    require(directory.is_relative_to(STAGE01) and directory.name == 'final_model'
            and audit['schema'] == 'aligned_QK2000_terminal_native_v1' and audit['status'] == 'verified'
            and execution['status'] == training['status'] == 'completed_native_pending_audit'
            and execution['exit_code'] == 0 and (STAGE01 / 'run.exit').read_text().strip() == '0'
            and training['training_result']['global_step'] == audit['selected_global_step'] == 2000
            and audit['checkpoint_id'] == meta['checkpoint_id'] == training['training_result']['checkpoint_id']
            and audit['checkpoint_dir'] == str(directory) and audit['profile'] == PROFILE
            and audit['frozen_tensors_compared'] == 2991 and audit['code_bank_count'] == 504
            and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
            and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['fresh_strict_reload_state_exact']
            and audit['bit_budget']['budget_passed'], 'Use only the actual completed and independently audited aligned 01.')
    check_hashes(audit['input_evidence_sha256'])
    check_stat(audit['checkpoint_weight_stat'])
    return directory, meta, audit


def model_config(directory, meta):
    config = teacher_recipe()
    require(Path(meta['base_model_path']).resolve() == Path(config['model']['hf_path']).resolve(),
            'Aligned student and F teacher must use the same original HF model.')
    config['model'].update(checkpoint=str(directory), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=sha256(directory / 'checkpoint_meta.json'),
        weight_stat=file_stat(directory / meta['state_dict_file']))
    qa.check_config(config)
    return config


def outside_projections(state, names):
    prefixes = tuple(name + '.' for name in names)
    return {key: value for key, value in state.items() if not key.startswith(prefixes)}


def teacher_identity(state, meta, names, directory):
    """Full252 HF restoration discards all source VAELinear payloads, not the backbone."""
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal
    teacher = teacher_recipe()
    old_dir = Path(teacher['model']['checkpoint'])
    qa.check_config(teacher)
    require(Path(meta['base_model_path']).resolve() == Path(teacher['model']['hf_path']).resolve(),
            'Student source references another original HF teacher.')
    old = load_state(old_dir)
    configs = [read(path / 'config.json') for path in (old_dir, Path(directory))]
    for config in configs:
        for key in ('_name_or_path', 'transformers_version'):
            config.pop(key, None)
    require(configs[0] == configs[1], 'Student and full-F carrier have different effective model configurations.')
    actual = outside_projections(state, names)
    count = _assert_state_equal(actual, outside_projections(old, names), 'Full-HF F teacher nonprojection backbone')
    # These are exactly the persistent tensors left from each carrier after restoring all252.
    digest = core.mapping_digest({key: core.tensor_record(value) for key, value in actual.items()})
    return dict(teacher_kind='full7', carrier_model_config=teacher,
        carrier_source_checkpoint_id=teacher['model']['checkpoint_id'], restored_projection_count=252,
        nonprojection_tensors_compared=count, nonprojection_sha256=digest,
        new_student_backbone_matches_teacher_carrier_exact=True,
        hf_metadata_sha256=teacher['model']['hf_metadata_sha256'],
        hf_weight_stats=teacher['model']['hf_weight_stats'],
        dataset_and_tokenizer='Original fixed train inputs and original tokenizer; full-F targets only.',
        reused_student_baseline=False, reused_student_retention=False)


def load_inputs():
    """Validate the actual old full-F cache; retain only F-M targets and train cases."""
    dataset, old, cache, evidence = reference.load_inputs()
    recovery = {}
    for key, item in old.items():
        case = item['case']
        # Never retain an old student's logits, QA answers, or a retention target.
        recovery[key] = {'case': case}
        if case['domain'] == 'mmlu':
            recovery[key]['target_log_probs'] = item['target_log_probs']
    require(len(recovery) == 1152 and sum('target_log_probs' in item for item in recovery.values()) == 1024,
            'Only the once-mixed frozen F-M1024 distribution may be reused for OPD.')
    return dataset, recovery, cache, evidence


def selected_modules(model, record):
    return hot.selected_modules(model, record)


def read_prepared(path=None):
    path = OUTPUT / 'prepared.json' if path is None else Path(path).resolve()
    prepared = read(path)
    directory, meta, audit = source()
    require(path == OUTPUT / 'prepared.json' and prepared['schema'] == 'CG_aligned_online_OPD32_v1'
            and prepared['status'] == 'CPU_prepared_GPU_unvalidated' and prepared['profile'] == PROFILE
            and prepared['source_checkpoint_id'] == meta['checkpoint_id']
            and prepared['source_checkpoint'] == str(directory) and prepared['actual_target_contract'] == CONTRACT
            and prepared['source_audit'] == binding(SOURCE_AUDIT), 'Changed aligned OPD source or contract.')
    for item in prepared['bindings'].values():
        require(sha256(item['path']) == item['sha256'], 'Changed bound OPD implementation/input: ' + item['path'])
    check_stat(prepared['source_weight_stat'])
    qa.check_config(prepared['record']['model_config'])
    qa.check_config(prepared['record']['teacher_identity']['carrier_model_config'])
    return prepared, prepared['record']
