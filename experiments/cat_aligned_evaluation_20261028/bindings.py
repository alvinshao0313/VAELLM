"""CPU identity binding for the aligned chain's ordinary native3423 endpoints."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / 'result/qwen3_8b_mmlu65_20261004'
OUTPUT = R / 'cat_aligned_evaluation_20261028'
DATA = Path('/data/star/VAELLM/cat_aligned_recovery_20261028')
SNAPSHOT = R / 'rotation_factor_probe/source_snapshot_20261005_01'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
SOURCE_IDS = {'C': '726b7b36-bb95-4418-b626-cc029b4f0aec', 'G': 'e8e5efd4-655e-462c-b6e0-fedf4e2e0346'}
STAGES = ('01', '03a', '03b', '04', '06')
BUDGET_FIELDS = ('components', 'user_linear_bits', 'packed_linear_bits', 'outside_linear_bits',
                 'original_linear_weights', 'code_padding_bits')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SNAPSHOT))

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


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


def helpers():
    return module(R / 'qwen_mlp_night_eval/evaluate.py', '_aligned_evaluation_helpers')


def recovery():
    return module(ROOT / 'experiments/night_sequence_20261026/recovery_eval.py', '_aligned_full8')


def formula():
    return module(R / 'qwen_b32_mlp_s3_20261023/summarize_nine.py', '_aligned_original_nine')


def cpu_environment():
    import torch
    require(Path(sys.executable).resolve() == PYTHON.resolve()
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Use caller-activated bitvae and CUDA_VISIBLE_DEVICES empty; the driver remains on CPU.')


def source(profile, stage, checkpoint=None, native_audit=None):
    require(profile in SOURCE_IDS and stage in STAGES, 'Only fixed C/G ordinary3423 stages01/03a/03b/04/06; Rice07 needs sparse_runtime.')
    pins = {}
    if stage == '01':
        require(checkpoint is None and native_audit is None, '01 resolves only its own completed training ledger; no source override.')
        stage_dir = DATA / profile / '01_qk_fineweb2000'
        audit_path = stage_dir / 'native_audit.json'
        require(audit_path.is_file(), 'Aligned01 is not independently audited yet: ' + str(audit_path))
        training, execution, prepared = (read(stage_dir / name) for name in ('training.json', 'execution.json', 'prepared.json'))
        checkpoint = Path(training['training_result']['saved_model_dir']).resolve()
        audit = read(audit_path)
        require(training['status'] == execution['status'] == 'completed_native_pending_audit'
                and execution['exit_code'] == 0 and (stage_dir / 'run.exit').read_text().strip() == '0'
                and training['training_result']['global_step'] == audit['selected_global_step'] == 2000
                and training['completed_steps'] == [400, 800, 1200, 1600, 2000]
                and checkpoint.is_relative_to(stage_dir) and checkpoint.name == 'final_model'
                and audit['schema'] == 'aligned_QK2000_terminal_native_v1'
                and audit['source_checkpoint_id'] == training['source_checkpoint_id'] == prepared['source_checkpoint_id'] == SOURCE_IDS[profile]
                and audit['checkpoint_id'] == training['training_result']['checkpoint_id']
                and audit['checkpoint_id'] != audit['selected_checkpoint_id']
                and audit['frozen_tensors_compared'] == 2991 and audit['packed_codes_unchanged']
                and audit['selected_decoder_matches_fp32_cast_exactly'],
                '01 requires its actual2000/new UUID/2991 frozen/packed banks/FP32-cast readback audit.')
        for key in ('source_sha256', 'input_evidence_sha256'):
            check_hashes(prepared[key]); pins.update(prepared[key])
        check_stat(prepared['source_weight_stat'])
        pins.update({str(stage_dir / name): sha256(stage_dir / name)
                     for name in ('training.json', 'execution.json', 'prepared.json', 'run.exit')})
    else:
        require(checkpoint is not None and native_audit is not None,
                'Later stages require explicit actual --checkpoint and --native-audit; no predecessor inference.')
        checkpoint, audit_path = Path(checkpoint).resolve(), Path(native_audit).resolve()
        audit = read(audit_path)
        require(checkpoint.is_relative_to(DATA / profile) and audit_path.is_relative_to(DATA / profile),
                'Explicit later endpoints must belong to this new aligned profile, never an old09/other chain.')
        if stage in ('03a', '03b'):
            arm, steps = ('scale32', 32) if stage == '03a' else ('q64', 64)
            require(audit['schema'] == 'CG_aligned_protection_native_audit_v1' and audit['arm'] == arm
                    and audit['phase'] == 'formal' and audit['actual_steps'] == audit['planned_steps'] == steps,
                    'Use the real formal03a32/03b64 native audit.')
        elif stage == '04':
            require(audit['schema'] == 'CG_aligned_C1_native_assembly_v1'
                    and audit['nonprojection_tensors_compared'] == 147 and audit['replaced_projection_count'] == 144,
                    '04 requires the actual whole-projection C1 assembly audit.')
        elif stage == '06':
            require(audit['schema'] == 'CG_aligned_selected_native_audit_v1' and audit['stage'] == 'B32'
                    and audit['phase'] == 'formal' and audit['actual_steps'] == audit['planned_steps'] == 32
                    and audit['mutable_tensors'] == 18 and audit['frozen_tensors_compared'] == 3405
                    and audit['parent_bit_ledger_equal'], '06 requires the actual formal selected B32 native audit.')
    meta = read(checkpoint / 'checkpoint_meta.json')
    require(audit['status'] == 'verified' and audit['profile'] == profile
            and Path(audit['checkpoint_dir']).resolve() == checkpoint
            and audit['checkpoint_id'] == meta['checkpoint_id']
            and audit['checkpoint_id'] != audit.get('source_checkpoint_id')
            and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
            and audit['fresh_strict_reload_state_exact'] and audit['code_bank_count'] == 504,
            'Require the actual same-profile/new-UUID native3423/504-bank strict readback audit.')
    uuid.UUID(meta['checkpoint_id'])
    sources = audit.get('sources', {})
    require(meta['checkpoint_id'] not in sources.get('parent_checkpoint_ids', []), 'Native export must have a new UUID.')
    for record in (audit, sources):
        for key in ('input_evidence_sha256', 'source_sha256', 'audit_code_sha256'):
            if key in record:
                check_hashes(record[key]); pins.update(record[key])
    check_stat(audit['checkpoint_weight_stat'])
    require(audit['checkpoint_weight_stat'] == file_stat(checkpoint / meta['state_dict_file']), 'Audit weights do not belong to this endpoint.')
    if 'checkpoint_meta_sha256' in audit:
        require(sha256(checkpoint / 'checkpoint_meta.json') == audit['checkpoint_meta_sha256'], 'Native metadata changed after audit.')
    api = recovery()
    actual_meta, budget = api.checkpoint(checkpoint)
    require(actual_meta == meta and all(budget[key] == audit['bit_budget'][key] for key in BUDGET_FIELDS),
            'Actual rank0/logical2.6 native storage differs from its verified audit.')
    if stage == '01':
        require(all(budget[key] == audit['source_bit_budget'][key] for key in BUDGET_FIELDS), '01 changed native storage.')
    pins.update({str(path): sha256(path) for path in (audit_path, checkpoint / 'checkpoint_meta.json')})
    return dict(profile=profile, stage=stage, checkpoint_dir=str(checkpoint), checkpoint_id=meta['checkpoint_id'],
                native_audit=str(audit_path), native_audit_sha256=sha256(audit_path),
                checkpoint_weight_stat=file_stat(checkpoint / meta['state_dict_file']), bit_budget=budget,
                evidence_sha256=pins), meta


def make_config(binding, meta, pins):
    from experiments.longbench_v1.models import make_config as native_config, check_config
    reference = read(helpers().REFERENCE_CONFIG)
    config = native_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'],
                           checkpoint=binding['checkpoint_dir'], mode='full')
    require(all(config[key] == reference[key] for key in reference if key != 'model'),
            'Keep the original canonical archive/32K/BF16/seed42/official generation protocol.')
    config['model']['recompression'] = dict(max_linear_bpw=2.6, max_lora_rank=0,
        bit_budget=binding['bit_budget'], source_sha256=pins)
    check_config(config)
    return config


def source_pins(binding):
    full8_api = module(R / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_aligned_sample_helpers')
    paths = [*full8_api.source_paths(), *helpers().source_paths(), *HERE.glob('*.py'),
        ROOT / 'experiments/night_sequence_20261026/recovery_core.py',
        ROOT / 'experiments/night_sequence_20261026/recovery_eval.py',
        ROOT / 'experiments/cat_recompression/longbench.py',
        R / 'qwen_b32_mlp_s3_20261023/summarize_nine.py',
        R / 'qwen_b32_mlp_s3_20261023/s3_bindings.py',
        R / 'qwen_b32_s3_gain_calibration_20261024/evaluation_adapter/full9.py']
    return {**binding['evidence_sha256'], **{str(path): sha256(path) for path in paths}}
