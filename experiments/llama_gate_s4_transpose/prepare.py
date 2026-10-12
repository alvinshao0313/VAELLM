"""Prepare the existing native S4 gate recipe with only gate transpose changed."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_gate_transpose.prepare import recipe
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import sha256, write_json

CONTROL = RESULT / 'llama_gate_stage4/s4'
CONTROL_ID = '13bd608f-c3a4-43d3-b3c0-be269743c544'
OUTPUT = RESULT / 'llama_gate_s4_transpose/true'


def prepare():
    _require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not OUTPUT.exists(), 'Prepare once in CPU-only S.')
    record = deepcopy(read(CONTROL / 'launch.prepared.json')); check(record)
    completed, audit, review, summary = (read(CONTROL / n) for n in
        ('launch.json', 'audit.json', 'independent_review.json', 'full8_summary.json'))
    _require(completed['status'] == 'completed' and completed['exit_code'] == 0
             and review['status'] == 'independent_cpu_verified'
             and review['checkpoint_id'] == audit['checkpoint_id'] == completed['checkpoint_id'] == CONTROL_ID
             and audit['base_checkpoint_id'] == BASE_ID
             and all((CONTROL / (n + '.exit')).read_text().strip() == '0'
                     for n in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the completed matched S4 False arm.')
    control, _ = score([CONTROL / (n + '.json') for n in ('mmlu', 'seven')], record['architecture'], CONTROL / 'final_model')
    _require(control['mean_percent'] == summary['mean_percent'] == completed['mean_percent'] == 61.69747742212227
             and control['integer_counts'] == summary['integer_counts']
             and summary['audit_sha256'] == sha256(CONTROL / 'audit.json')
             and completed['full8_summary_sha256'] == sha256(CONTROL / 'full8_summary.json'), 'Control raw summary changed.')
    argv, normalized = recipe(record, OUTPUT)
    cfg = normalized['resolved_category_runtime']['gate_proj']
    _require(cfg['codebook_bits'] == 16 and cfg['codebook_dim'] == 32 and cfg['residual_stages'] == 4
             and cfg['base_ch'] == 128 and cfg['num_res_blocks'] == 0 and cfg['channel_protect_count'] == 64
             and (4096 - 64) % 32 == 14336 % 32 == 0, 'Keep S4/D32 and both vector orientations aligned.')
    budget = audit['bit_budget']
    _require(budget['user_linear_bits'] == budget['packed_linear_bits'] == 17099800576
             and budget['original_linear_weights'] == DENOMINATOR and budget['code_padding_bits'] == 0
             and budget['components'] == record['expected_budget_components']
             and audit['source_union_save_fresh_tensors'] == 3043 and audit['full_code_bank_count'] == 512,
             'Actual S4 control geometry/budget differs.')
    transposed = RESULT / 'llama_gate_k34_transpose/true'
    proof = read(transposed / 'independent_review.json')
    _require(proof['checkpoint_id'] == '98a80541-fd2b-4ce4-b88d-0809f82d4ef3'
             and read(transposed / 'audit.json')['actual_gate_transpose'] is True,
             'Require the already executed native gateTrue path.')
    record.pop('native_cpu_construction', None)
    record.update(schema='llama_gate_s4_transpose_v1', output_dir=str(OUTPUT), transpose=True,
        historical_control_dir=str(CONTROL), historical_control_checkpoint_id=CONTROL_ID,
        training_argv=[sys.executable, 'tools/cat_train.py', *argv], normalized_expected=normalized,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_s4_transpose.run',
            '--launch-record', str(OUTPUT / 'launch.json')], timeout_seconds=2700, termination_grace_seconds=20,
        budget_estimate={'user_linear_bits': 17099800576, 'packed_linear_bits': 17099800576,
            'bpw': 17099800576 / DENOMINATOR, 'code_padding_bits': 0, 'additional_bits_vs_s4_false': 0},
        validation_reuse={'new_gpu_short': False, 'matched_native_s4': str(CONTROL),
            'executed_gate_true': str(transposed), 'boundary': 'Native four-stage packed CAT/save/group8 and gateTrue already executed. This crossed recipe is not a measured quality result.'},
        comparison_boundary='Only gate transpose False toTrue versus13bd; fresh HF32gate, same K16/D32/S4/H128R0/input64INT8/block32/B8192/seed31/10Kx4. Other192 and backbone remain e767. Orientation changes code vectors; gains from separate S4/transpose results cannot be added. No independent MMLU/eight-task quality gate; nine-item acceptance needs full21 LongBench.',
        resource_estimate={'estimated_seconds': [1800, 2400], 'seconds_limit': 2700,
            'matched_s4_false_seconds': completed['finished_unix'] - completed['started_unix'],
            'single_arm_disk_provision_bytes': 32326744064, 'filesystem_free_at_preparation': shutil.disk_usage(RESULT).free,
            'boundary': 'Same persistent geometry and stage count; orientation throughput is unmeasured. Keep final/raw for later LongBench or recovery decisions; no automatic queue/retry.'})
    for p in [CONTROL / n for n in ('launch.prepared.json', 'launch.json', 'audit.json', 'independent_review.json',
            'full8_summary.json', 'mmlu.json', 'seven.json', 'final_model/checkpoint_meta.json', 'final_model/assembly_report.json')]:
        record['input_sha256'][str(p)] = sha256(p)
    for p in [transposed / 'independent_review.json', transposed / 'audit.json', ROOT / 'experiments/llama_gate_transpose/prepare.py']:
        record['input_sha256'][str(p)] = sha256(p)
    for p in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(p.read_text()); record['source_sha256'][str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/S new source differs.')
    check(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    OUTPUT.mkdir(parents=True); write_json(OUTPUT / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(OUTPUT / 'launch.prepared.json'), 'sources': len(record['source_sha256']),
        'inputs': len(record['input_sha256']), 'stats': len(record['protected_stats']), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
