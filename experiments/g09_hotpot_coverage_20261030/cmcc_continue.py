"""Reuse the completed H192 full8; run only canonical LongBench and strict summaries."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.g09_hotpot_coverage_20261030 import cmcc_evaluate as e
from experiments.g09_hotpot_coverage_20261030 import cmcc_strict_summary as strict

c = e.c
FULL8 = c.OUT / 'evaluation/Muon/full8'
CONTROL = c.ROOT / 'result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/evaluation/Muon/transfer_acceptance.json'
CONTROL_MODEL = Path('/data1/cmcc/VAELLM/artifacts/g09_recovery_20261029/muon_opd32/native_final/final_model')
UUID = '74329fcf-c762-4498-9f41-218df9645b3f'


def continuation_plan(plan):
    argv = plan['nine_task_argv']
    c.require(argv[:4] == [sys.executable, '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task'],
              'Unexpected current-profile summary entry.')
    argv[3] = 'experiments.g09_hotpot_coverage_20261030.cmcc_strict_summary'
    index = argv.index('--compressed_results')
    argv[index + 1:index + 3] = [str(FULL8 / 'mmlu.json'), str(FULL8 / 'seven.json')]
    return plan


def raw_pins():
    pins = {}
    leaves = 0
    for name, count in (('mmlu', 14042), ('seven', 10700)):
        report_path = FULL8 / (name + '.json')
        exit_path = FULL8 / (name + '.exit')
        c.require(exit_path.read_text().strip() == '0', 'Require the actual completed full8 worker exit.')
        report = c.read(report_path)
        lm = report['evals']['lm_eval']
        artifacts = lm['sample_artifacts']
        c.require(report['checkpoint_id'] == UUID and artifacts['total_count'] == count
                  and set(artifacts['tasks']) == set(lm['n_samples'])
                  and sum(row['count'] for row in artifacts['tasks'].values()) == count, 'Incomplete same-UUID raw full8.')
        leaves += len(artifacts['tasks'])
        for task, row in artifacts['tasks'].items():
            path = Path(row['path']).resolve()
            c.require(path.is_relative_to(FULL8) and row['count'] == lm['n_samples'][task]['effective']
                      and c.sha(path) == row['sha256'], 'Changed full8 raw sample: ' + task)
            pins[str(path)] = row['sha256']
        pins[str(report_path)] = c.sha(report_path)
        pins[str(exit_path)] = c.sha(exit_path)
    c.require(leaves == 64, 'Require all64 leaves/24742 actual full8 samples.')
    return pins


def validate_full8(original, output):
    recovery = original.b.recovery()
    old_write = recovery.write

    def write_summary(path, value):
        c.require(Path(path) == FULL8 / 'full8_summary.json', 'Only redirect the new full8 summary output.')
        c.write(output / 'full8_summary.json', value)

    # All original full8_summary computation and raw checks run against the
    # existing source files. Only its new summary is written to this continuation.
    recovery.write = write_summary
    try:
        result, _, _ = recovery.full8_summary(c.OUT / 'native_final/final_model', FULL8)
    finally:
        recovery.write = old_write
    c.require(result['leaf_count'] == 64 and result['sample_count'] == 24742, 'Incomplete original full8 summary.')
    return result


def prepare(output, gpus):
    c.require(not output.exists(), 'Use a fresh continuation preparation directory.')
    output.mkdir(parents=True)
    import evaluate as hf_evaluate
    expected = Path(importlib.metadata.distribution('evaluate').locate_file('evaluate/__init__.py')).resolve()
    c.require(Path(hf_evaluate.__file__).resolve() == expected, 'Import must resolve the installed HF evaluate package.')
    original = e.runtime()
    binding, meta = original.b.source('Muon', '')
    c.require(meta['checkpoint_id'] == UUID, 'Do not substitute another H192 endpoint.')
    pins = {**original.b.source_pins(binding), **raw_pins()}
    result = validate_full8(original, output)
    control = strict.rank0_report(c.read(CONTROL), CONTROL_MODEL)
    c.require(control['status'] == 'not_met' and not control['passed']
              and control['compressed_nine_task_score'] == 66.06887949148307
              and control['protocol']['max_lora_rank'] == 0, 'Valid old Muon not_met must remain a valid completed result.')
    c.write(output / 'control_rank0_validation.json', control)
    (output / 'control_rank0_validation.exit').write_text('1\n', encoding='utf-8')
    for path in (Path(__file__), Path(strict.__file__), CONTROL, output / 'full8_summary.json',
                 output / 'control_rank0_validation.json', c.OUT / 'evaluation/Muon/evaluation.json', c.OUT / 'full9.exit',
                 c.OUT / 'failed_full9_source_20261030/cmcc_evaluate.py'):
        pins[str(path.resolve())] = c.sha(path)
    config = original.b.make_config(binding, meta, pins)
    config_path = output / 'validated_config.json'
    c.write(config_path, config)
    plan = continuation_plan(original.make_plan(config, config_path, binding['checkpoint_dir'], output / 'evaluation/Muon', gpus, pins))
    c.require(sum(unit['expected_samples'] for unit in plan['execution_units']) == 4750
              and len(plan['execution_units']) == 3, 'Keep canonical21/4750 across the same three GPUs.')
    c.write(output / 'validated_plan.json', plan)
    c.write(output / 'prepared.json', dict(schema='cmcc_H192_LB_continuation_v1', status='CPU_verified',
        checkpoint_id=UUID, output=str(output), gpus=gpus, source_full8=str(FULL8),
        full8_leaf_tasks=64, full8_samples=24742, full8_mean_percent=result['mean_percent'],
        hf_evaluate_binding=c.bind(expected), input_sha256=pins,
        validated_config=c.bind(config_path), validated_plan=c.bind(output / 'validated_plan.json'),
        control_rank0_report=c.bind(output / 'control_rank0_validation.json'),
        completion_report=str(output / 'evaluation/Muon/evaluation.json'),
        strict_result=str(output / 'evaluation/Muon/current_goal_result.json'),
        pair_result=str(output / 'muon_pair.json'),
        scope='Reuse actual successful full8 without writes; new LB21/4750, strict rank0/current quality, mechanical Muon comparison.'))
    print(json.dumps(dict(status='CPU_verified', checkpoint_id=UUID, full8_samples=24742,
        full8_leaf_tasks=64, full8_mean_percent=result['mean_percent'], hf_evaluate=str(expected))), flush=True)


def run(output, gpus):
    prepared = c.read(output / 'prepared.json')
    c.require(prepared['status'] == 'CPU_verified' and prepared['checkpoint_id'] == UUID
              and prepared['output'] == str(output) and prepared['gpus'] == gpus, 'Changed continuation binding.')
    for path, digest in prepared['input_sha256'].items():
        c.require(c.sha(path) == digest, 'Changed continuation dependency: ' + path)
    original = e.runtime()
    source = original.b.source
    source_pins = original.b.source_pins
    original.b.OUTPUT = output / 'evaluation'
    recovery = original.b.recovery()

    def completed_full8(checkpoint, destination, gpu, *, full9, evidence):
        c.require(Path(checkpoint) == c.OUT / 'native_final/final_model' and not full9,
                  'Only reuse this complete H192 full8.')
        raw_pins()
        summary = c.read(output / 'full8_summary.json')
        return dict(status='complete_full8', full8_mean_percent=summary['mean_percent'],
                    checkpoint_id=UUID, source_summary=str(output / 'full8_summary.json'))

    recovery.evaluate = completed_full8

    def full8_pins(destination, pins):
        pins.update(raw_pins())
        pins[str(output / 'full8_summary.json')] = c.sha(output / 'full8_summary.json')

    original.full8_pins = full8_pins

    def continued_source(*args):
        binding, meta = source(*args)
        binding.update(reused_full8=str(FULL8), reused_full8_summary=str(output / 'full8_summary.json'))
        return binding, meta

    original.b.source = continued_source
    original.b.source_pins = lambda binding: {**source_pins(binding), **prepared['input_sha256'],
                                              str(output / 'prepared.json'): c.sha(output / 'prepared.json')}
    make_plan = original.make_plan

    def continued_plan(*args):
        return continuation_plan(make_plan(*args))

    original.make_plan = continued_plan
    record = original.evaluate(SimpleNamespace(profile='Muon', stage='', checkpoint=None, native_audit=None,
                                               check_only=False, gpus=gpus))
    result = e.finish(record, output / 'evaluation/Muon')
    control = c.read(output / 'control_rank0_validation.json')
    c.require(set(result['compressed_scores']) == set(control['compressed_scores'])
              and result['baseline_nine_task_score'] == control['baseline_nine_task_score'],
              'Require the same nine metrics and BF16 reference for mechanical pairing.')
    c.write(output / 'muon_pair.json', dict(status='complete_H192_Muon_fixed32_pair',
        source_checkpoint_id=c.SOURCE_ID, h192_checkpoint_id=UUID, control_checkpoint_id=control['checkpoint_id'],
        h192_report=c.bind(record['nine_task_report']), control_report=c.bind(CONTROL),
        h192_score=result['compressed_nine_task_score'], control_score=control['compressed_nine_task_score'],
        delta_points=result['compressed_nine_task_score'] - control['compressed_nine_task_score'],
        task_deltas={key: result['compressed_scores'][key] - control['compressed_scores'][key] for key in result['compressed_scores']},
        h192_passed=result['passed'], control_passed=control['passed'],
        interpretation='Mechanical same-parent fixed32 recipe comparison; no significance or generalization claim.'))
    print(json.dumps(dict(status='complete', checkpoint_id=UUID, score=result['compressed_nine_task_score'],
                          passed=result['passed'], quality_not_met_is_technical_failure=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'The continuation driver remains on CPU.')
    gpus = args.gpus.split(',')
    c.require(gpus == ['5', '6', '7'], 'Use the fixed H192 GPU5/6/7 allocation.')
    output = args.output.resolve()
    if args.prepare_only:
        prepare(output, gpus)
    else:
        run(output, gpus)
