"""One-shot pidfd resource handoffs, D terminal donors, and fixed05/full9."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

import core as d


def donor(recipe):
    d.check()
    gpu = d.GPUS[recipe]
    output = d.OUTPUT / recipe
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'pipeline.json'
    record = dict(status='running', profile='D', recipe=recipe, gpu=gpu,
        source_checkpoint_id=d.SOURCE_ID, started_unix=time.time(), stages={})
    d.write(path, record)
    try:
        command = [sys.executable, '-B', '-u', str(d.HERE / 'run.py'), 'train', '--recipe', recipe]
        record['stages']['train'] = d.stages.stage('train', command, output, gpu=gpu, seconds=50400)
        trained = d.read(output / 'training_result.json')
        final = Path(trained['native_result']['saved_model_dir'])
        step = Path(trained['saved_training_checkpoints'][-1]['path'])
        audit = d.audit_final(d.SOURCE, final, step, output / 'native_audit.json')
        # The reused audit is CPU-only; the original native runtime finalized on GPU.
        audit.pop('export_used_cuda', None)
        audit.update(audit_used_cuda=False, native_runtime_finalization_used_cuda=True)
        d.write(output / 'native_audit.json', audit)
        record['stages']['native_audit'] = dict(exit_code=0, evidence=str(output / 'native_audit.json'))
        record.update(final_model=str(final), checkpoint_id=audit['checkpoint_id'])
        d.write(path, record)
        evaluation = d.evaluate_full8(final, output / 'evaluation_full8', gpu, evidence=output / 'native_audit.json')
        record['stages']['full8'] = dict(exit_code=0, evidence=str(output / 'evaluation_full8/evaluation.json'))
        record.update(status='complete_donor', full8_mean_percent=evaluation['full8_mean_percent'])
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        d.write(path, record)
    return record


def predecessor_terminal(recipe):
    binding = d.PREDECESSORS[recipe]
    d.processes.wait_for_process(binding['pid'], binding['start_ticks'])
    report = d.read(binding['report'])
    d.require(report['pid'] == binding['pid'] and int(report['start_ticks']) == binding['start_ticks'], 'The exact exited resource predecessor identity changed.')
    terminal = report.get('status') in ('complete', 'completed', 'failed')
    result = dict(binding=binding, process_exit_event=True, report_status=report.get('status'),
        report_terminal=terminal, execution_exit_code=report.get('exit_code'), report_sha256=d.sha(binding['report']),
        resource_dependency_only=True, status='terminal_report_recorded' if terminal else 'exited_without_terminal_report')
    evaluation = d.TOPIC / 'qwen_recompression_20261026/final_full9/D/evaluation.json' if recipe == 'ce10' else d.PREVIOUS / 'sparse/G/full9/evaluation.json'
    if evaluation.exists():
        actual = d.read(evaluation)
        result['quality_evaluation'] = dict(path=str(evaluation), status=actual.get('status'), passed=actual.get('passed'),
            checkpoint_id=actual.get('checkpoint_id'), nine_task_score=actual.get('nine_task_score'), sha256=d.sha(evaluation))
    # Neither an unrelated execution failure nor quality_not_met changes D's source/recipe.
    d.check()
    d.native.check_gpu(d.GPUS[recipe])
    result['assigned_gpu_checked_free_once'] = d.GPUS[recipe]
    return result


def lane(recipe, peer_pid=None, peer_start=None):
    d.check()
    root = d.OUTPUT / 'queue' / recipe
    root.mkdir(parents=True, exist_ok=False)
    path = root / 'pipeline.json'
    record = dict(status='waiting_exact_resource_predecessor', recipe=recipe, gpu=d.GPUS[recipe],
        pid=d.os.getpid(), start_ticks=d.processes.process_start(d.os.getpid()),
        predecessor=d.PREDECESSORS[recipe], started_unix=time.time(), automatic_retry=False, resource_polling=False)
    d.write(path, record)
    try:
        record['predecessor_terminal'] = predecessor_terminal(recipe)
        record['status'] = 'running_donor'
        d.write(path, record)
        record['donor'] = donor(recipe)
        if recipe == 'ce10':
            d.require(peer_pid is not None and peer_start is not None, 'The fixed CE10 leader requires its actual CE50 producer identity.')
            record.update(status='waiting_actual_same_source_CE50_donor', peer_pid=peer_pid, peer_start_ticks=peer_start)
            d.write(path, record)
            d.processes.wait_for_process(peer_pid, peer_start)
            peer = d.read(d.OUTPUT / 'queue/ce50/pipeline.json')
            d.require(peer['pid'] == peer_pid and int(peer['start_ticks']) == peer_start and peer['status'] == 'complete'
                and d.read(d.OUTPUT / 'ce50/pipeline.json')['status'] == 'complete_donor', 'Actual D CE50 producer must finish successfully before05.')
            record['status'] = 'running_fixed05_full9'
            d.write(path, record)
            audit = d.assemble()
            from evaluate import evaluate
            d.native.check_gpu(d.GPUS[recipe])
            result = evaluate(Path(audit['checkpoint_dir']), d.OUTPUT / '05_ce50_attention_ce10_mlp/evaluation_full9', d.GPUS[recipe],
                              d.OUTPUT / '05_ce50_attention_ce10_mlp/native_audit.json')
            record.update(assembly_checkpoint_id=audit['checkpoint_id'], nine_task_score=result['nine_task_score'], passed=result['passed'])
        record.update(status='complete', exit_code=0)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        d.write(path, record)


def launch():
    d.check()
    path = d.OUTPUT / 'launch.json'
    d.require(not path.exists(), 'Launch the finite D resource handoff only once.')
    record = dict(status='launching', prepared_sha256=d.sha(d.PREPARED), started_unix=time.time(), workers={}, automatic_retry=False, resource_polling=False)
    d.write(path, record)
    for recipe in ('ce50', 'ce10'):
        argv = [sys.executable, '-B', '-u', str(d.HERE / 'run.py'), 'lane', '--recipe', recipe]
        if recipe == 'ce10':
            peer = record['workers']['ce50']
            argv.extend(['--peer-pid', str(peer['pid']), '--peer-start', str(peer['start_ticks'])])
        _child, row = d.stages.spawn(argv, d.OUTPUT, recipe + '_lane')
        record['workers'][recipe] = row
        d.write(path, record)
    record['status'] = 'two_exact_resource_waits_started'
    d.write(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'launch', 'lane', 'train'])
    parser.add_argument('--recipe', choices=['ce10', 'ce50'])
    parser.add_argument('--peer-pid', type=int)
    parser.add_argument('--peer-start', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = d.prepare()
        print(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds']), flush=True)
    elif args.command == 'launch':
        launch()
    elif args.command == 'train':
        d.train(args.recipe)
    else:
        lane(args.recipe, args.peer_pid, args.peer_start)
