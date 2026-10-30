"""Real aligned06 source adapter for the unchanged abs005/Rice7 bundle consumers."""
import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SELECTED = ROOT / 'experiments/cat_aligned_selected_20261028'
NIGHT = ROOT / 'experiments/night_sequence_20261026'
previous = list(sys.path)
try:
    sys.path.insert(0, str(SELECTED))
    selected = importlib.import_module('selected_bindings')
    sys.path.insert(0, str(NIGHT))
    c = importlib.import_module('sparse_common')
    builder = importlib.import_module('sparse_build')
    bundle = importlib.import_module('sparse_bundle')
    runtime = importlib.import_module('sparse_runtime')
    evaluator = importlib.import_module('sparse_eval')
    for value in (c, builder, bundle, runtime, evaluator):
        if Path(value.__file__).resolve().parent != NIGHT:
            raise ValueError('Mature sparse import was shadowed: ' + value.__name__)
finally:
    sys.path[:] = previous
sys.path.insert(0, str(c.SNAPSHOT))
sys.dont_write_bytecode = True

PROFILE, PARENT = selected.PROFILE, selected.PARENT
OUTPUT = PARENT / '07_sparse005'
ARTIFACT = OUTPUT / 'artifacts'
CHECKPOINT = ARTIFACT / 'final_model'
R, SNAPSHOT, PYTHON, HF = c.RESULT, c.SNAPSHOT, c.PYTHON, c.HF
read, write, require, binding, torch = c.read, c.write, c.require, c.binding, c.torch
RECIPE, FORMAT = c.RECIPE, c.FORMAT
REFERENCE = c.OUTPUT
PLAN_SOURCE = R / 'qwen_b32_s3_gain_calibration_20261024/evaluation_adapter/full9.py'
RICE_MULTI = R / 'qwen_sparse_mlp0075_rice7_probe_20261024/evaluation_adapter'
CONTRACT = dict(source='Only this profile actual verified06_B32/native_final/final_model; never09/rawC/rawG/oldB32',
    support='108 MLP eachceil(4096*12288*.005)=251659; abs(HF_BF16-current06_native_BF16), exclude protected input columns; rebuild per new source',
    values='Original256x256 blocked INT8 q/FP32scale/zero; BF16 decode then BF16 cache addition; no density/value/selection changes',
    indices='Lossless original Rice7 coordinates; actual stream/offset/header lengths counted; no persisted expanded local_indices',
    native='NewUUID self-contained base3423+Rice864=4287 freshstrict; all3423 original fields and504banks frozen;108 protection columns exact',
    budget='Actual logical<=2.6/rank0 including all source andRice fields; packed padding/container separate; measured over_budget is valid construction but not quality-evaluation eligible; no automatic adjustment',
    evaluation='Real Rice bundle consumer; full8 0shot M64/seven32/64leaves/24742, two canonical whole-task LB partitions/21/4750/32K/officialcaps/greedyseed42; originalnine target67.24714491790849; not_met valid',
    execution='Explicit prepare/build/export/consume/evaluate; no future manifest or UUID, scheduler, predecessor wait, retry or partial scoring')


def helpers():
    return c.module(R / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_aligned07_mature_eval_helpers')


def source():
    root = PARENT / '06_B32'
    audit_path = root / 'native_audit.json'
    require(audit_path.is_file(), 'Actual aligned06 is not independently audited yet: ' + str(audit_path))
    audit = read(audit_path)
    directory = root / 'native_final/final_model'
    meta = read(directory / 'checkpoint_meta.json')
    report_path = root / 'formal/report.json'
    report = read(report_path)
    prepared = selected.read_prepared('B32')
    require(audit['schema'] == 'CG_aligned_selected_native_audit_v1' and audit['status'] == 'verified'
        and audit['stage'] == report['stage'] == 'B32' and audit['profile'] == report['profile'] == PROFILE
        and audit['phase'] == report['phase'] == 'formal' and audit['actual_steps'] == audit['planned_steps'] == report['steps'] == report['planned_steps'] == 32
        and audit['checkpoint_id'] == meta['checkpoint_id'] and audit['checkpoint_dir'] == str(directory)
        and audit['source_checkpoint_id'] == report['source_checkpoint_id'] == prepared['source_checkpoint_id']
        and audit['R_source_checkpoint_id'] == report['R_source_checkpoint_id'] == prepared['R_source_checkpoint_id']
        and report['prepared_sha256'] == selected.sha256(root / 'prepared.json') and report['contract'] == selected.CONTRACT
        and audit['endpoint_report'] == binding(report_path) and report['status'] == 'aligned_selected_formal_complete'
        and (report_path.parent / 'run.exit').read_text().strip() == '0' and report['source_restored_exact']
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['frozen_tensors_compared'] == 3405 and audit['mutable_tensors'] == 18 and audit['code_bank_count'] == 504
        and audit['fresh_strict_reload_state_exact'] and audit['source_geometry_unchanged'] and audit['parent_bit_ledger_equal']
        and audit['bit_budget']['budget_passed'] and meta['checkpoint_id'] != prepared['source_checkpoint_id']
        and audit['checkpoint_meta_sha256'] == selected.sha256(directory / 'checkpoint_meta.json'),
        'Only this profile actual complete independently verified06 B32 may source07.')
    selected.mature.check_hashes(audit['sources']['input_evidence_sha256'])
    selected.mature.check_stat(audit['checkpoint_weight_stat'])
    return directory, meta, audit, audit_path


def read_prepared():
    path = OUTPUT / 'prepared.json'
    record = c.check_prepared(path)
    directory, meta, _, audit_path = source()
    require(record['schema'] == 'CG_aligned06_abs005_Rice7_prepared_v1' and record['profile'] == PROFILE
        and record['contract'] == CONTRACT and record['source_dir'] == str(directory)
        and record['source_checkpoint_id'] == meta['checkpoint_id'] and record['source_native_audit'] == binding(audit_path)
        and record['artifact_root'] == str(ARTIFACT), 'Changed actual06-bound sparse07 preparation.')
    return record


def check_export(*, quality=False):
    prepared = read_prepared()
    path = OUTPUT / 'audit.json'
    audit = read(path)
    require(audit['status'] in ('verified', 'over_budget') and audit['source_checkpoint_id'] == prepared['source_checkpoint_id']
        and audit['checkpoint_dir'] == str(CHECKPOINT) and audit['actual_native_save'] and audit['independent_fresh_strict']
        and audit['base_frozen_tensors_compared'] == 3423 and audit['rice_tensors_compared'] == 864
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 4287
        and audit['source_banks_compared'] == 504 and audit['no_LoRA_optimizer'] and not audit['unpacked_local_indices_persisted']
        and audit['sources']['prepared'] == binding(OUTPUT / 'prepared.json') and audit['sources']['build'] == binding(OUTPUT / 'build.json')
        and not audit['bit_budget']['contract_violations'], 'Require the actual new4287 strict native/Rice export.')
    if audit['status'] == 'over_budget':
        require(not audit['bit_budget']['budget_passed']
            and audit['bit_budget']['user_linear_bits']*5 > audit['bit_budget']['original_linear_weights']*13,
            'Only measured fixed-density storage overflow is a valid over_budget result.')
        require(not quality, 'Measured over_budget candidate is ineligible for quality evaluation; density is unchanged.')
        return CHECKPOINT, audit, None
    meta, actual = runtime.check_audit(CHECKPOINT)
    require(actual == audit and meta['source_checkpoint_id'] == prepared['source_checkpoint_id'], 'Bundle/base parent differs from actual06.')
    return CHECKPOINT, audit, meta


def pins():
    prepared = read_prepared()
    values = {item['path']: item['sha256'] for item in prepared['bindings']}
    for path in (OUTPUT / 'prepared.json', OUTPUT / 'build.json', OUTPUT / 'audit.json', OUTPUT / 'gpu_validation.json',
            CHECKPOINT / 'checkpoint_meta.json', CHECKPOINT / 'base/checkpoint_meta.json'):
        values[str(path)] = binding(path)['sha256']
    return values
