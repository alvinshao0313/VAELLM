"""Original-D native full8 and LongBench21 with strict current-goal scoring."""
import argparse
import importlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from . import evaluation_bindings as e
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def full8(arguments):
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--shard', choices=('mmlu', 'seven'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(arguments)
    b.require(not args.output.exists(), 'Never overwrite a full8 shard.')
    config = b.read(args.config)
    b.require(config['mode'] == 'full', 'Complete full8 has no diagnostic limit.')
    model, audit = e.load_model(config, logging.getLogger('original_D_full8'))
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(str(b.BASE), use_fast=True,
        trust_remote_code=True, local_files_only=True)
    api = b.module(b.SAMPLE_DIR / 'eval_utils_with_samples.py', '_original_D_hflm_samples')
    options = SimpleNamespace(tasks='mmlu' if args.shard == 'mmlu' else b.SEVEN,
        num_fewshot=0, batch_size='64' if args.shard == 'mmlu' else '32', lm_limit=None,
        model_path=str(b.BASE), eval_log_dir=None, eval_run_ts=None, mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(args.output.parent / (args.shard + '_logs')),
        mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    result = api.run_lm_eval(model, tokenizer, options)
    payload = dict(checkpoint_dir=config['model']['checkpoint'], checkpoint_id=audit['checkpoint_id'],
        checkpoint_loader='v6', checkpoint_format=e.FORMAT, base_model_path=str(b.BASE),
        strict_load=True, adapter_dir=None, eval_hif4_act=False,
        tokenizer_class=type(tokenizer).__name__, tokenizer_is_fast=tokenizer.is_fast,
        checkpoint_files_sha256=config['model']['checkpoint_files_sha256'], load_audit=audit,
        evals=dict(lm_eval=result))
    e.check_config(config)
    b.write_json(args.output, payload)
    return 0


def predict(arguments):
    from experiments.longbench_v1 import runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--max-runtime-seconds', required=True, type=float)
    args = parser.parse_args(arguments)
    b.require(b.read(args.config)['mode'] == 'full', 'Only the complete official partition is allowed.')
    runner.check_config, runner.load_model = e.check_config, e.load_model
    report = runner.run(args.config, args.output, datasets=args.datasets,
        max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({key:report[key] for key in ('status', 'new_predictions', 'already_present')}), flush=True)
    return 0


def full8_summary(config, paths, binding, meta, output):
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    inventory = b.model_inventory(b.BASE)
    compressed = collect_results(paths, inventory,
        checkpoint=(Path(binding['checkpoint_dir']), meta))
    baseline_paths, _lb = b.template_inputs()
    baseline = collect_results(baseline_paths, inventory)
    b.require(compressed['versions'] == baseline['versions'], 'Task versions differ from BF16.')
    for path, reference in zip(paths, baseline_paths, strict=True):
        payload = b.read(path)
        audit = payload['load_audit']
        b.require(payload['checkpoint_files_sha256'] == binding['checkpoint_files_sha256']
            and audit['checkpoint_format'] == e.FORMAT and audit['persistent_tensors'] == 3423
            and audit['checkpoint_files_sha256'] == binding['checkpoint_files_sha256']
            and payload['evals']['lm_eval']['configs'] == b.read(reference)['configs'],
            'Require the actual native inventory, immutable files and matching task configurations.')
    compressed['sample_artifacts'] = _sample_artifacts(paths)
    b.write_json(output, dict(checkpoint_id=meta['checkpoint_id'], checkpoint_format=e.FORMAT, **compressed))
    return compressed


def summarize(arguments):
    from experiments.w25_eval.transfer_nine_task_acceptance import summarize as official
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pipeline-report', 'compressed-longbench-scores'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--compressed-results', nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(arguments)
    b.cpu_environment()
    b.require(not args.output.exists(), 'Never overwrite acceptance evidence.')
    binding, meta = e.source(args.checkpoint, args.pipeline_report)
    baseline, lb = b.template_inputs()
    report = official(base_model_path=str(b.BASE), checkpoint_dir=binding['checkpoint_dir'],
        longbench_manifest=b.read(b.REFERENCE)['archive']['path'], baseline_results=baseline,
        compressed_results=args.compressed_results, baseline_longbench_scores=lb,
        compressed_longbench_scores=args.compressed_longbench_scores)
    b.require(report['checkpoint_id'] == meta['checkpoint_id']
        and report['protocol']['max_linear_bpw'] == 2.6
        and report['protocol']['maximum_absolute_drop_points'] == .9
        and report['protocol']['strict_score_drop_bound'], 'The native current-goal profile changed.')
    report.update(pipeline_report=binding['pipeline_report'], originalA_control=binding['originalA_control'],
        source_binding=binding['source_binding'], training_contract=binding['training_contract'])
    b.write_json(args.output, report)
    return 0 if report['passed'] else 1


def evaluate(arguments):
    from experiments.adaptive_class_evaluation_20261030 import evaluate as execution
    from experiments.qwen3_w25.step_export_callback import run_logged_stage
    from experiments.longbench_v1.data import load_archive
    from experiments.longbench_v1.protocol import EXPECTED_COUNTS, DATASETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pipeline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=execution.canonical().gpu_list, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(arguments)
    b.cpu_environment()
    b.require(args.gpus == ['0'], 'This fixed batch assigns only physicalGPU0 to original-D quality evaluation.')
    binding, meta = e.source(args.checkpoint, args.pipeline_report)
    pins = b.source_pins(binding)
    config = e.make_config(binding, pins)
    manifest, rows = load_archive(config['archive']['path'], check_token_file=False)
    b.require(len(rows) == 4750 and manifest['data']['counts'] == EXPECTED_COUNTS
        and all(row['input_tokens']+row['max_new_tokens'] <= 32768 for row in rows),
        'Keep complete21/4750 and the original32K input policy.')
    if args.check_only:
        print(json.dumps(dict(status='verified_completed_original_D_native_source',
            checkpoint_id=meta['checkpoint_id'])), flush=True)
        return 0
    output = args.output.resolve()
    b.require(not output.exists() and output.is_relative_to(b.TOPIC), 'Use a fresh topic evaluation directory.')
    resources = execution.canonical().check_resources(args.gpus)
    b.check_hashes(pins)
    output.mkdir(parents=True)
    record = dict(status='running', **binding, source_sha256=pins, gpus=args.gpus,
        started_unix=time.time(), argv=list(sys.argv), initial_GPU_resource_check=resources,
        automatic_retry=False, automatic_resume=False, stages={})
    api = b.module(b.HELPERS, '_original_D_full9_execution')
    api.S = b.ROOT

    def stage(name, command, gpu='', seconds=1800):
        actual = ['timeout', '--signal=TERM', '--kill-after=20s', str(seconds)+'s', *command]
        record['stages'][name] = dict(command=actual, gpu=gpu, started_unix=time.time())
        b.write_json(output/'evaluation.json', record)
        try:
            return run_logged_stage(actual, name=name, output_dir=output, env=api.environment(b.ROOT, gpu))
        finally:
            path = output/(name+'.exit')
            record['stages'][name].update(finished_unix=time.time(),
                exit_code=path.read_text().strip() if path.exists() else None)
            b.write_json(output/'evaluation.json', record)

    try:
        full = output/'full8'
        full.mkdir()
        b.write_json(full/'config.json', config)
        for shard in ('mmlu', 'seven'):
            stage(shard, b.command('full8', '--config', str(full/'config.json'), '--shard', shard,
                '--output', str(full/(shard+'.json'))), args.gpus[0], 3600)
        paths = [full/(shard+'.json') for shard in ('mmlu', 'seven')]
        summary = full8_summary(config, paths, binding, meta, full/'full8_summary.json')
        pins.update(summary['sample_artifacts']['sha256'])
        pins.update({str(path):b.sha256(path) for path in (*paths, full/'full8_summary.json')})
        record.update(full8_mean_percent=summary['mean_percent'], full8_sample_count=24742, full8_leaf_tasks=64)
        lb = output/'longbench'
        lb.mkdir()
        config_path, plan_path = lb/'config.json', lb/'merge_plan.json'
        b.write_json(config_path, config)
        plan = execution.make_plan(config, config_path, binding, output, args.gpus, pins)
        plan['boundary'] = 'One actual original-D native3423 endpoint; complete full8+LongBench21 and strict current goal.'
        plan['actual_GPU_resource_check'] = execution.canonical().check_resources(args.gpus)
        b.write_json(plan_path, plan)
        b.check_hashes(plan['pins'])
        e.check_config(config)
        processes = []
        try:
            for unit in plan['execution_units']:
                Path(unit['output_dir']).mkdir(parents=True)
                processes.append(api.start_group(unit, config, plan['config_sha256']))
        finally:
            codes = [process.wait() for process in processes]
            record['longbench_outer_exit_codes'] = codes
            b.write_json(output/'evaluation.json', record)
        b.require(codes == [0], 'The native partition failed; no retry or partial scoring.')
        for index, unit in enumerate(plan['execution_units']):
            group = Path(unit['output_dir'])
            reports = list((group/'runs').glob('*.json'))
            b.require(len(reports) == 1, 'Require exactly one native partition attempt.')
            actual = b.read(reports[0])
            b.require(actual['status'] == 'partition_complete' and actual['exit_code'] == 0
                and actual['already_present'] == 0 and actual['new_predictions'] == unit['expected_samples']
                and actual['load_audit']['checkpoint_format'] == e.FORMAT
                and actual['load_audit']['persistent_tensors'] == 3423
                and actual['load_audit']['checkpoint_files_sha256'] == binding['checkpoint_files_sha256'],
                'Require the complete original-D native partition with its immutable identity.')
            stage('bind_'+str(index), [str(b.PYTHON), str(api.BINDER), '--plan', str(plan_path),
                '--unit-index', str(index), '--native-report', str(reports[0]),
                '--outer-launch', str(group/'outer.launch.json'), '--outer-exit', str(group/'outer.exit'),
                '--outer-log', str(group/'outer.log')])
        plan = b.read(plan_path)
        b.require(sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750
            and not plan['pending_output_directories_at_last_terminal'], 'Require full21 coverage.')
        plan['status'] = 'prepared_not_scored'
        b.write_json(plan_path, plan)
        stage('merge_nine', b.command('merge', '--plan', str(plan_path)))
        merged = Path(plan['merged_output'])
        report = b.read(merged/'nine_task_acceptance.json')
        code = int((merged/'nine_task.exit').read_text().strip())
        b.require(code in (0, 1) and report['quality_target_evaluated']
            and report['checkpoint_id'] == meta['checkpoint_id']
            and report['protocol']['max_linear_bpw'] == 2.6
            and report['protocol']['maximum_absolute_drop_points'] == .9
            and report['protocol']['strict_score_drop_bound'], 'Require the same-UUID strict current-goal outcome.')
        b.check_hashes(plan['pins'])
        e.check_config(config)
        record.update(status='complete_full8_full21_current_goal', longbench_plan=str(plan_path),
            longbench_sample_count=4750, longbench_scores=str(merged/'scores.json'),
            acceptance_report=str(merged/'nine_task_acceptance.json'), acceptance_exit_code=code,
            score=report['compressed_nine_task_score'], passed=report['passed'])
        b.write_json(output/'current_goal_result.json', record)
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        b.write_json(output/'evaluation.json', record)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', type=Path, required=True)
    parser.add_argument('--component', choices=('evaluate', 'full8', 'predict', 'summarize', 'merge'), default='evaluate')
    args, remaining = parser.parse_known_args()
    e.setup(args.support.resolve(strict=True))
    if args.component == 'merge':
        api = importlib.import_module('experiments.adaptive_class_evaluation_20261030.merge')
        sys.argv = [sys.argv[0], *remaining]
        return api.main()
    return globals()[args.component](remaining)


if __name__ == '__main__':
    raise SystemExit(main())
