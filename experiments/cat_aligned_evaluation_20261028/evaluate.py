"""One fixed aligned native3423 endpoint: complete full8, then canonical full21/nine."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import bindings as b
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS

# Exact partition policy from the mature gain-calibration full9 driver; no model-specific constraints.
LB_GROUPS = (('gov_report',), ('multi_news',), ('vcsum',),
    ('passage_retrieval_en', 'passage_retrieval_zh', 'repobench-p'),
    ('qasper', 'triviaqa', 'lcc'), ('multifieldqa_zh', 'musique', 'dureader', 'trec'),
    ('narrativeqa', '2wikimqa', 'samsum', 'passage_count'),
    ('multifieldqa_en', 'hotpotqa', 'qmsum', 'lsht'))


def gpu_list(value):
    devices = value.split(',')
    b.require(1 <= len(devices) <= 8 and len(set(devices)) == len(devices)
        and all(device in tuple(map(str, range(8))) for device in devices), 'Choose distinct physical GPUs0..7.')
    return devices


def task_groups(gpus):
    buckets = [set() for _ in gpus]
    for index, group in enumerate(LB_GROUPS):
        buckets[index % len(gpus)].update(group)
    groups = [[task for task in DATASETS if task in bucket] for bucket in buckets]
    assigned = [task for group in groups for task in group]
    b.require(len(assigned) == len(set(assigned)) == 21 and set(assigned) == set(DATASETS)
        and sum(EXPECTED_COUNTS[task] for task in assigned) == 4750, 'Canonical21/4750 must be assigned exactly once.')
    return groups


def check_resources(gpus):
    devices = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    observed = {line.split(',')[0].strip(): line.split(',')[1].strip() for line in devices.stdout.splitlines()}
    b.require(all(gpu in observed for gpu in gpus), 'An assigned GPU is absent.')
    selected = {gpu: observed[gpu] for gpu in gpus}
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    blockers = [line for line in active.stdout.splitlines() if line.split(',')[0].strip() in selected.values()]
    b.require(not blockers, 'Assigned GPUs are occupied; stop without waiting/retry: ' + repr(blockers))
    return dict(checked_unix=time.time(), gpu_uuids=selected, other_compute_processes=blockers)


def make_plan(config, config_path, checkpoint, output, gpus, pins):
    api = b.helpers()
    template = b.read(api.TEMPLATE)
    units = []
    for index, (gpu, tasks) in enumerate(zip(gpus, task_groups(gpus), strict=True)):
        group = output / 'longbench/groups' / f'part_{index + 1:02d}'
        runtime = 86400
        argv = [str(b.PYTHON), '-B', '-u', str(b.HERE / 'predict.py'), '--config', str(config_path),
            '--output', str(group), '--datasets', *tasks, '--max-runtime-seconds', str(runtime)]
        units.append(dict(output_dir=str(group), datasets=tasks,
            expected_samples=sum(EXPECTED_COUNTS[task] for task in tasks), native_reports=[],
            outer_exit=None, launch_records=[], terminal_attempts=[], gpu=gpu, argv=argv,
            native_runtime_seconds=runtime, outer_timeout_seconds=runtime + 180, termination_grace_seconds=20))
    original = template['nine_task_argv']
    b.require(original[1:3] == ['-m', 'experiments.w25_eval.summarize_nine_task'], 'Wrong original BF16 reference template.')
    nine = [str(b.PYTHON), str(b.HERE / 'summarize.py'), *original[3:]]
    merged = output / 'longbench/merged'
    nine[nine.index('--checkpoint_dir') + 1] = str(checkpoint)
    index = nine.index('--compressed_results')
    nine[index + 1:index + 3] = [str(output / 'full8/mmlu.json'), str(output / 'full8/seven.json')]
    nine[nine.index('--compressed_longbench_scores') + 1] = str(merged / 'scores.json')
    nine[nine.index('--output') + 1] = str(merged / 'nine_task_acceptance.json')
    b.require(nine[nine.index('--base_model_path') + 1] == config['model']['hf_path']
        and nine[nine.index('--longbench_manifest') + 1] == config['archive']['path'], 'Keep the original HF/tokenizer/BF16/archive reference.')
    return dict(schema='fixed_full21_merge_plan_v1', status='draft_awaiting_explicit_terminal_binding',
        checkpoint_id=config['model']['checkpoint_id'], expected_tasks=list(DATASETS), expected_samples=4750,
        config_path=str(config_path), config_sha256=b.sha256(config_path), protocol=config['protocol'],
        execution_units=units, task_file_sources={task: str(Path(unit['output_dir']) / 'predictions' / (task + '.jsonl'))
            for unit in units for task in unit['datasets']}, pins={**pins, str(config_path): b.sha256(config_path)},
        merged_output=str(merged), scoring_cwd=str(b.ROOT), python_executable=str(b.PYTHON), nine_task_argv=nine,
        scoring_environment=dict(PYTHONPATH=str(b.ROOT), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
        acceptance={**template['acceptance'], 'max_linear_bpw': 2.6, 'max_lora_rank': 0,
            'source_boundary': 'Actual independently verified aligned native3423 endpoint; not sparse/Rice07.'},
        nine_task_exit_semantics=template['nine_task_exit_semantics'], automatic_retry=False,
        automatic_resume=False, partial_scoring=False,
        boundary='One same-UUID complete full8+21/4750 evaluation; no quality gate, predecessor wait or retry.')


def full8_pins(output, pins):
    leaves = 0
    for name, count in (('mmlu', 14042), ('seven', 10700)):
        path = output / (name + '.json')
        lm = b.read(path)['evals']['lm_eval']
        artifacts = lm['sample_artifacts']
        b.require(artifacts['total_count'] == count and set(artifacts['tasks']) == set(lm['n_samples'])
            and sum(row['count'] for row in artifacts['tasks'].values()) == count, 'Incomplete full8 question artifacts.')
        leaves += len(artifacts['tasks'])
        for task, row in artifacts['tasks'].items():
            artifact = Path(row['path']).resolve()
            b.require(artifact.is_relative_to(output) and row['count'] == lm['n_samples'][task]['effective']
                and b.sha256(artifact) == row['sha256'], 'Full8 raw sample binding changed.')
            pins[str(artifact)] = row['sha256']
        pins[str(path)] = b.sha256(path)
    b.require(leaves == 64, 'Require all64 leaf tasks/24742 samples.')
    pins[str(output / 'full8_summary.json')] = b.sha256(output / 'full8_summary.json')


def evaluate(args):
    b.cpu_environment()
    binding, meta = b.source(args.profile, args.stage, args.checkpoint, args.native_audit)
    if args.check_only:
        print(b.read(binding['native_audit'])['status'], binding['checkpoint_id'], flush=True)
        return binding
    output = b.OUTPUT / args.profile / args.stage
    b.require(not os.path.lexists(output), 'Never overwrite an endpoint evaluation: ' + str(output))
    pins = b.source_pins(binding)
    config = b.make_config(binding, meta, pins)
    manifest, rows = load_archive(config['archive']['path'], check_token_file=False)
    b.require(len(rows) == 4750 and manifest['data']['counts'] == EXPECTED_COUNTS
        and all(row['input_tokens'] + row['max_new_tokens'] <= 32768 for row in rows), 'Canonical archive/window changed.')
    resources = check_resources(args.gpus)
    b.check_hashes(pins)
    output.mkdir(parents=True)
    report_path = output / 'evaluation.json'
    record = dict(status='running', **binding, gpus=args.gpus, started_unix=time.time(),
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        argv=sys.argv, initial_GPU_resource_check=resources, source_sha256=pins,
        automatic_retry=False, automatic_resume=False, stages={})
    b.write_json(report_path, record)
    recovery, api = b.recovery(), b.helpers()
    runner = b.module(b.ROOT / 'experiments/qwen3_w25/step_export_callback.py', '_aligned_logged_stage').run_logged_stage

    def stage(name, argv):
        record['stages'][name] = dict(command=argv, started_unix=time.time())
        b.write_json(report_path, record)
        try:
            command = ['timeout', '--signal=TERM', '--kill-after=20s', '1800s', *argv]
            return runner(command, name=name, output_dir=output, env=api.environment(b.ROOT, ''))
        finally:
            record['stages'][name].update(finished_unix=time.time(),
                exit_code=(output / (name + '.exit')).read_text().strip() if (output / (name + '.exit')).exists() else None)
            b.write_json(report_path, record)

    try:
        full8 = recovery.evaluate(binding['checkpoint_dir'], output / 'full8', args.gpus[0],
                                  full9=False, evidence=binding['native_audit'])
        full8_pins(output / 'full8', pins)
        record.update(full8_mean_percent=full8['full8_mean_percent'], full8_sample_count=24742, full8_leaf_tasks=64)
        lb = output / 'longbench'
        lb.mkdir()
        config_path, plan_path = lb / 'config.json', lb / 'merge_plan.json'
        b.write_json(config_path, config)
        plan = make_plan(config, config_path, binding['checkpoint_dir'], output, args.gpus, pins)
        plan['actual_GPU_resource_check'] = check_resources(args.gpus)
        b.write_json(plan_path, plan)
        b.check_hashes(plan['pins']); check_config(config)
        processes = []
        try:
            for unit in plan['execution_units']:
                Path(unit['output_dir']).mkdir(parents=True)
                processes.append(api.start_group(unit, config, plan['config_sha256']))
        finally:
            codes = [process.wait() for process in processes]
            record['longbench_outer_exit_codes'] = codes
            b.write_json(report_path, record)
        b.require(len(codes) == len(args.gpus) and all(code == 0 for code in codes), 'A partition failed; no retry or partial score.')
        for index, unit in enumerate(plan['execution_units']):
            group = Path(unit['output_dir'])
            reports = list((group / 'runs').glob('*.json'))
            b.require(len(reports) == 1, 'One actual native inference attempt per partition is required.')
            native = b.read(reports[0])
            b.require(native['status'] == 'partition_complete' and native['exit_code'] == 0
                and native['already_present'] == 0 and native['new_predictions'] == unit['expected_samples']
                and native['load_audit']['persistent_tensors'] == 3423,
                'Require complete actual native3423 partition output.')
            stage(f'bind_{index}', [str(b.PYTHON), str(api.BINDER), '--plan', str(plan_path), '--unit-index', str(index),
                '--native-report', str(reports[0]), '--outer-launch', str(group / 'outer.launch.json'),
                '--outer-exit', str(group / 'outer.exit'), '--outer-log', str(group / 'outer.log')])
        plan = b.read(plan_path)
        b.require(sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750
            and not plan['pending_output_directories_at_last_terminal'], 'Complete canonical21 coverage is required.')
        b.check_hashes(plan['pins']); check_config(config)
        plan['status'] = 'prepared_not_scored'
        b.write_json(plan_path, plan)
        # The mature merger accepts nine_exit1 as a valid complete not_met result.
        stage('merge_nine', [str(b.PYTHON), str(api.MERGER), '--plan', str(plan_path)])
        merged = Path(plan['merged_output'])
        scores, nine = b.read(merged / 'scores.json'), b.read(merged / 'nine_task_acceptance.json')
        nine_exit = int((merged / 'nine_task.exit').read_text().strip())
        b.require(nine_exit in (0, 1) and scores['status'] == 'complete' and scores['sample_count'] == 4750
            and set(scores['tasks']) == set(DATASETS) and scores['model']['checkpoint_id'] == meta['checkpoint_id']
            and nine['quality_target_evaluated'] and nine['checkpoint_id'] == meta['checkpoint_id']
            and nine['protocol']['max_linear_bpw'] == 2.6 and nine['protocol']['max_lora_rank'] == 0,
            'Require same-UUID complete native2.6/rank0 nine-term evidence.')
        b.check_hashes(plan['pins']); b.check_stat(binding['checkpoint_weight_stat']); check_config(config)
        record.update(status='complete_full8_full21_nine', longbench_plan=str(plan_path),
            longbench_sample_count=4750, longbench_scores=str(merged / 'scores.json'), nine_exit_code=nine_exit,
            nine_task_report=str(merged / 'nine_task_acceptance.json'), nine_task_score=nine['compressed_nine_task_score'],
            quality_passed=nine['mean_passed'], authorized_2_6_acceptance_passed=nine['passed'])
    except BaseException as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        b.write_json(report_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('C', 'G'), required=True)
    parser.add_argument('--stage', choices=b.STAGES, required=True)
    parser.add_argument('--gpus', type=gpu_list, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--native-audit', type=Path)
    parser.add_argument('--check-only', action='store_true')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
