"""One actual Rice07 bundle: full8 then two canonical full21 partitions and mature merge."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
import sparse07_bindings as b
import sparse07_plan as plan_api
from experiments.longbench_v1.common import check_hashes
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS


def validate_full8(output, checkpoint, meta, pins):
    from experiments.w25_eval.model_inventory import model_inventory
    architecture = model_inventory(meta['base_model_path'])
    paths = [output / 'mmlu.json', output / 'seven.json']
    full8 = b.evaluator.collect_results(paths, architecture, checkpoint=(checkpoint, meta))
    template = b.read(b.helpers().TEMPLATE)['nine_task_argv']
    index = template.index('--baseline_results')
    baseline_paths = [Path(path) for path in template[index+1:index+3]]
    baseline = b.evaluator.collect_results(baseline_paths, architecture)
    b.require(full8['versions'] == baseline['versions'], 'Full8 task versions differ from BF16.')
    leaves = 0
    for candidate, base, expected in zip(paths, baseline_paths, (14042, 10700), strict=True):
        lm = b.read(candidate)['evals']['lm_eval']
        artifacts = lm['sample_artifacts']
        b.require(lm['configs'] == b.read(base)['configs'] and artifacts['total_count'] == expected
            and set(artifacts['tasks']) == set(lm['n_samples'])
            and sum(row['count'] for row in artifacts['tasks'].values()) == expected, 'Original full8 samples/configurations differ.')
        leaves += len(artifacts['tasks'])
        for task, row in artifacts['tasks'].items():
            path = Path(row['path']).resolve()
            b.require(path.is_relative_to(output) and row['count'] == lm['n_samples'][task]['effective']
                and b.binding(path)['sha256'] == row['sha256'], 'Full8 raw sample binding changed.')
            pins[str(path)] = row['sha256']
    b.require(leaves == 64 and full8['sample_count'] == 24742, 'Require all64 leaves/24742 samples.')
    b.write(output / 'full8_summary.json', dict(checkpoint_id=meta['checkpoint_id'], **full8))
    for path in (*paths, output / 'full8_summary.json'):
        pins[str(path)] = b.binding(path)['sha256']
    return full8


def evaluate(gpus, check_only=False):
    b.c.cpu_only()
    checkpoint, audit, meta = b.check_export()
    if check_only or audit['status'] == 'over_budget':
        return dict(status=audit['status'], checkpoint_id=audit['checkpoint_id'], quality_evaluation_started=False,
            bit_budget=audit['bit_budget'])
    b.require(len(gpus) == 2, 'Use two explicitly assigned physical GPUs for final07 full9.')
    b.runtime.check_gpu(checkpoint)
    output = b.OUTPUT / 'full9'
    b.require(not os.path.lexists(output), 'Never overwrite actual07 evaluation.')
    pins = b.pins()
    config = b.evaluator.make_config(checkpoint, pins)
    manifest, rows = load_archive(config['archive']['path'], check_token_file=False)
    b.require(len(rows) == 4750 and manifest['data']['counts'] == EXPECTED_COUNTS
        and all(row['input_tokens']+row['max_new_tokens'] <= 32768 for row in rows), 'Original full21/archive/window differs.')
    resources = plan_api.check_resources(gpus)
    check_hashes(pins)
    output.mkdir(parents=True)
    record = dict(status='running', profile=b.PROFILE, checkpoint_id=meta['checkpoint_id'], checkpoint_dir=str(checkpoint),
        source_checkpoint_id=meta['source_checkpoint_id'], checkpoint_loader=b.FORMAT, gpus=gpus, source_sha256=pins,
        bit_budget=audit['bit_budget'], started_unix=time.time(), initial_GPU_resource_check=resources,
        automatic_retry=False, automatic_resume=False, stages={})
    report_path = output / 'evaluation.json'
    b.write(report_path, record)
    api = b.helpers()
    runners = {root: b.c.module(root / 'experiments/qwen3_w25/step_export_callback.py',
        '_aligned07_logged_stage_'+root.name).run_logged_stage for root in (b.ROOT, b.SNAPSHOT)}

    def stage(name, argv, *, gpu='', seconds=1800, root=b.ROOT):
        item = dict(argv=argv, cwd=str(root), physical_gpu=gpu, timeout_seconds=seconds, started_unix=time.time())
        record['stages'][name] = item
        b.write(report_path, record)
        try:
            return runners[root](['timeout', '--signal=TERM', '--kill-after=20s', str(seconds)+'s', *argv],
                name=name, output_dir=output, env=api.environment(root, gpu))
        finally:
            item.update(finished_unix=time.time(), exit_code=(output / (name+'.exit')).read_text().strip()
                if (output / (name+'.exit')).exists() else None)
            b.write(report_path, record)

    try:
        for name, tasks, batch in (('mmlu', 'mmlu', '64'), ('seven', api.SEVEN, '32')):
            stage(name, [str(b.PYTHON), '-B', '-u', str(b.NIGHT / 'sparse_runtime.py'), 'full8',
                '--checkpoint_dir', str(checkpoint), '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0',
                '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', batch, '--tokenizer_use_fast', 'true',
                '--prewarm_group_size', '8', '--eval_log_dir', str(output / (name+'_logs')),
                '--output_json', str(output / (name+'.json'))], gpu=gpus[0], seconds=3600, root=b.SNAPSHOT)
        full8 = validate_full8(output, checkpoint, meta, pins)
        record.update(full8_mean_percent=full8['mean_percent'], full8_sample_count=24742, full8_leaf_tasks=64)
        lb = output / 'longbench'
        lb.mkdir()
        config_path, plan_path = lb / 'config.json', lb / 'merge_plan.json'
        b.write(config_path, config)
        plan = plan_api.make_plan(config, config_path, checkpoint, output, gpus, pins)
        plan['actual_GPU_resource_check'] = plan_api.check_resources(gpus)
        b.write(plan_path, plan)
        check_hashes(plan['pins'])
        check_config(config)
        processes = []
        try:
            for unit in plan['execution_units']:
                Path(unit['output_dir']).mkdir(parents=True)
                processes.append(api.start_group(unit, config, plan['config_sha256']))
        finally:
            codes = [process.wait() for process in processes]
            record['longbench_outer_exit_codes'] = codes
            b.write(report_path, record)
        b.require(len(codes) == 2 and all(code == 0 for code in codes), 'A full21 partition failed; no retry/partial score.')
        for index, unit in enumerate(plan['execution_units']):
            group = Path(unit['output_dir'])
            reports = list((group / 'runs').glob('*.json'))
            b.require(len(reports) == 1, 'Require one actual inference attempt per partition.')
            native = b.read(reports[0])
            b.require(native['status'] == 'partition_complete' and native['exit_code'] == 0 and native['already_present'] == 0
                and native['new_predictions'] == unit['expected_samples'], 'Every assigned official row must be complete.')
            b.runtime.check_loaded(native['load_audit'], meta['checkpoint_id'], audit['bit_budget'])
            stage(f'bind_{index}', [str(b.PYTHON), str(api.BINDER), '--plan', str(plan_path), '--unit-index', str(index),
                '--native-report', str(reports[0]), '--outer-launch', str(group / 'outer.launch.json'),
                '--outer-exit', str(group / 'outer.exit'), '--outer-log', str(group / 'outer.log')])
        plan = b.read(plan_path)
        b.require(sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750
            and not plan['pending_output_directories_at_last_terminal'], 'Require complete canonical21/4750 coverage.')
        check_hashes(plan['pins'])
        check_config(config)
        plan['status'] = 'prepared_not_scored'
        b.write(plan_path, plan)
        stage('merge_nine', [str(b.PYTHON), str(api.MERGER), '--plan', str(plan_path)])
        merged = Path(plan['merged_output'])
        scores, nine = b.read(merged / 'scores.json'), b.read(merged / 'nine_task_acceptance.json')
        code = int((merged / 'nine_task.exit').read_text().strip())
        b.require(code in (0, 1) and scores['status'] == 'complete' and scores['sample_count'] == 4750
            and set(scores['tasks']) == set(DATASETS) and scores['model']['checkpoint_id'] == meta['checkpoint_id']
            and nine['checkpoint_id'] == meta['checkpoint_id'] and nine['quality_target_evaluated']
            and nine['protocol']['max_linear_bpw'] == 2.6 and nine['protocol']['max_lora_rank'] == 0,
            'Require valid same-UUID full21/nine; not_met is a complete result.')
        check_hashes(plan['pins'])
        b.check_export(quality=True)
        record.update(status='complete_full8_full21_nine', longbench_sample_count=4750, nine_exit_code=code,
            nine_task_report=str(merged / 'nine_task_acceptance.json'), nine_task_score=nine['compressed_nine_task_score'],
            quality_passed=nine['mean_passed'], authorized_2_6_acceptance_passed=nine['passed'])
    except BaseException as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        b.write(report_path, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=plan_api.gpu_list, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(evaluate(args.gpus, args.check_only), flush=True)
