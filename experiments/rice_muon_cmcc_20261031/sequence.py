"""Wait once for complete C1/full9, then Rice lifecycle2 -> fresh original-C1 Muon32/full9."""
import argparse
import fcntl
import os
from pathlib import Path
import select
import sys
import time
import traceback

from . import bindings as b
from experiments.fresh_muon_cmcc_20261030 import after_control as events

DIRECTORY = b.EVIDENCE / 'after_C1_complete'
TARGET = b.OUTPUT / 'C1_H96_Muon32'
CONTROLLER = (502968, 15251008)
INITIAL_WORKER = (503348, 15251666)
BUDGET = dict(engineering_Rice_updates=2, formal_recovery_updates=32, original_schedule_steps=500,
    complete_nine_evaluations=1, extra_C1_training_updates=0, seed=137, group_indices=list(range(32)),
    training_gpus=[0], evaluation_gpus=[0, 1, 2])


def requests():
    prepared = TARGET / 'prepared.json'

    def request(name, module, arguments, gpu='', devices=()):
        return dict(name=name, argv=[str(b.PYTHON), '-B', '-u', '-m', module, *arguments],
            environment=b.executor.environment(gpu), gpus=list(devices),
            log=str(DIRECTORY / (name + '.log')), exit_file=str(DIRECTORY / (name + '.exit')))

    module = 'experiments.rice_muon_cmcc_20261031.'
    return [
        request('prepare_actual_C1', module + 'prepare', ['--checkpoint', str(b.SOURCE / 'final_model'),
            '--pipeline-report', str(b.SOURCE / 'report.json'), '--output', str(TARGET)]),
        request('Rice_short2', module + 'run', ['--prepared', str(prepared), '--gpu', '0', '--phase', 'short2',
            '--timeout-seconds', '7200'], '0', ('0',)),
        request('Rice_short2_fresh', module + 'fresh', ['--prepared', str(prepared), '--gpu', '0', '--phase', 'short2'], '0', ('0',)),
        request('Rice_formal32', module + 'run', ['--prepared', str(prepared), '--gpu', '0', '--phase', 'formal32',
            '--timeout-seconds', '43200'], '0', ('0',)),
        request('Rice_formal32_fresh', module + 'fresh', ['--prepared', str(prepared), '--gpu', '0', '--phase', 'formal32'], '0', ('0',)),
        request('Rice_Muon_full9', module + 'evaluate', ['--support', str(b.SUPPORT), '--checkpoint', str(TARGET / 'formal32/final_model'),
            '--pipeline-report', str(TARGET / 'formal32/report.json'), '--output', str(b.EVIDENCE / 'evaluation'), '--gpus', '0,1,2'],
            devices=('0', '1', '2')),
    ]


def current_parent():
    """Bind actual active identities now, without waiting or treating a future final as real."""
    control = b.read(b.CONTROL_PLAN)
    run = b.read(b.CONTROL_PLAN.parent / 'run.json')
    stage = run['stages']['C1_Rice_formal']
    b.require(control['schema'] == 'fresh_C1_Rice_single_sequence_v1'
        and Path(control['output']).resolve(strict=True) == b.CONTROL_PLAN.parent
        and control['formal']['gpus'] == ['0'] and control['evaluation']['gpus'] == ['0', '1', '2']
        and control['order'] == ['C1_Rice_formal', 'C1_Rice_actual_audit', 'C1_Rice_complete_nine']
        and (run['pid'], run['start_ticks']) == CONTROLLER and (stage['pid'], stage['start_ticks']) == INITIAL_WORKER
        and run['plan_sha256'] == b.executor.sha(b.CONTROL_PLAN)
        and stage['status'] == 'running' and run['status'] == 'C1_Rice_formal_running',
        'Bind only the actual active fresh C1 controller/formal pair and its fixed full9 plan.')
    controller, worker = events.process_identity(CONTROLLER[0]), events.process_identity(INITIAL_WORKER[0])
    b.require(controller['start_ticks'] == CONTROLLER[1] and worker['start_ticks'] == INITIAL_WORKER[1]
        and worker['parent_pid'] == controller['pid'] and worker['argv'] == stage['argv'] == control['formal']['argv']
        and controller['argv'] == [str(b.PYTHON), '-B', '-u', '-m',
            'experiments.fresh_rice_cmcc_20261031.sequence', '--run', str(b.CONTROL_PLAN)],
        'Actual C1 commands, process generations or worker ownership changed.')
    b.executor.check_pins(control['source_sha256'])
    return dict(controller=controller, initial_formal_worker=worker,
        record_path=str(b.CONTROL_PLAN.parent / 'run.json'), control_plan_sha256=b.executor.sha(b.CONTROL_PLAN))


def terminal_parent(parent):
    """Only an actual stopped successful controller can release the full dependency."""
    control, run = b.read(b.CONTROL_PLAN), b.read(parent['record_path'])
    b.require(b.executor.sha(b.CONTROL_PLAN) == parent['control_plan_sha256']
        and (run['pid'], run['start_ticks']) == (parent['controller']['pid'], parent['controller']['start_ticks'])
        and run['plan_sha256'] == parent['control_plan_sha256']
        and run.get('status') == 'completed_C1_Rice_and_complete_nine' and run.get('exit_code') == 0,
        'The actual C1 controller did not publish successful complete-nine; technical failure stops this dependency.')
    for key in ('formal', 'audit', 'evaluation'):
        expected = control[key]
        stage = run['stages'][expected['name']]
        b.require(stage['status'] == 'exited' and type(stage['exit_code']) is int and stage['exit_code'] == 0
            and stage['argv'] == expected['argv'] and stage['exit_file'] == expected['exit_file']
            and Path(expected['exit_file']).read_text().strip() == '0', 'Actual C1 stage did not complete exit0: ' + key)
        if key == 'formal':
            worker = parent['initial_formal_worker']
            b.require((stage['pid'], stage['start_ticks']) == (worker['pid'], worker['start_ticks']),
                'The C1 final belongs to another formal worker.')
    b.executor.check_pins(control['source_sha256'])
    source, _meta = b.source(b.SOURCE / 'final_model', b.SOURCE / 'report.json')
    audit = b.read(b.CONTROL_PLAN.parent / 'final_audit.json')
    b.require(audit['status'] == 'verified_actual_fresh_C1_Rice_final' and audit['exit_code'] == 0
        and audit['binding'] == source and audit['checkpoint_id'] == run['final_checkpoint_id'] == source['checkpoint_id'],
        'Actual C1 final audit and successful controller disagree on source UUID or binding.')
    quality = b.executor.validate_evaluation(control['evaluation']['output'], source['checkpoint_id'])
    b.require(run['complete_nine'] == quality, 'Actual same-UUID full9 differs from the C1 controller terminal result.')
    return source, quality


def wait_controller(parent, observe):
    descriptor = events.open_identity(parent['controller'])
    try:
        if descriptor is not None:
            observe('waiting_actual_C1_complete_controller_pidfd', parent['controller'])
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            poller.poll()  # One kernel exit wait, no interval, log polling or retry.
        observe('actual_C1_controller_exit_observed', parent['controller'])
        # After process exit every atomic terminal write is closed; no earlier
        # formal-only exit or open publication can race this read.
        return terminal_parent(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def prepare():
    b.executor.cpu_environment()
    b.configure()
    b.require(not os.path.lexists(DIRECTORY) and not os.path.lexists(TARGET)
        and not os.path.lexists(b.EVIDENCE / 'evaluation'), 'Never overwrite a plan, recovery or quality output.')
    parent = current_parent()
    control = b.read(b.CONTROL_PLAN)
    pins = dict(control['source_sha256'])
    for path, item in b.code_bindings().items():
        b.require(path not in pins or pins[path] == item['sha256'], 'Parent and new recovery pins disagree: ' + path)
        pins[path] = item['sha256']
    pins[str(b.CONTROL_PLAN)] = parent['control_plan_sha256']
    b.executor.check_pins(pins)
    plan = dict(schema='fresh_C1_Rice_after_complete_Muon32_v1', created_unix=time.time(),
        control_plan=str(b.CONTROL_PLAN), parent=parent, target=str(TARGET), source_sha256=pins,
        stages=requests(), budget=BUDGET, automatic_retry=False, automatic_resume=False,
        boundary='Wait once for this entire C1 controller and same-UUID full9. Then actual-source CPU prepare, '
            'Rice engineering2/fresh, independent original-C1 fresh32/fresh/full9. No short continuation, old donor, '
            'resource polling, reservation, scheduler or modification of active files.')
    DIRECTORY.mkdir(parents=True)
    b.core.write(DIRECTORY / 'plan.json', plan)
    print(dict(status='prepared_single_Rice_Muon32_after_complete_C1', plan=str(DIRECTORY / 'plan.json'),
        target=str(TARGET), controller=parent['controller']), flush=True)


def run():
    b.executor.cpu_environment()
    b.configure()
    plan_path = DIRECTORY / 'plan.json'
    plan = b.read(plan_path)
    b.require(plan['schema'] == 'fresh_C1_Rice_after_complete_Muon32_v1'
        and plan['control_plan'] == str(b.CONTROL_PLAN) and plan['target'] == str(TARGET)
        and plan['stages'] == requests() and plan['budget'] == BUDGET
        and (plan['parent']['controller']['pid'], plan['parent']['controller']['start_ticks']) == CONTROLLER
        and (plan['parent']['initial_formal_worker']['pid'], plan['parent']['initial_formal_worker']['start_ticks']) == INITIAL_WORKER
        and plan['automatic_retry'] is plan['automatic_resume'] is False,
        'The fixed reviewed C1/Rice-Muon dependency identity or requests changed.')
    b.require(not (DIRECTORY / 'run.json').exists(), 'This one-shot continuation already started.')
    record = dict(schema=plan['schema'], status='waiting_complete_C1_controller', pid=os.getpid(),
        start_ticks=b.executor.proc_start(os.getpid()), started_unix=time.time(), argv=[sys.executable, *sys.argv],
        plan_sha256=b.executor.sha(plan_path), target=str(TARGET), events=[], stages={})
    with (DIRECTORY / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (DIRECTORY / 'run.json').open('x', encoding='utf-8') as stream:
            import json
            stream.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')

        def observe(event, evidence):
            record['events'].append(dict(event=event, unix=time.time(), evidence=evidence))
            b.core.write(DIRECTORY / 'run.json', record)
            print(dict(event=event, evidence=evidence), flush=True)

        def changed(stage):
            record['stages'][stage['name']] = stage
            record['status'] = stage['name'] + '_' + stage['status']
            b.core.write(DIRECTORY / 'run.json', record)

        code = 1
        try:
            b.require(not os.path.lexists(TARGET), 'Recovery target already exists; never resume or retry.')
            b.executor.check_pins(plan['source_sha256'])
            parent, control = wait_controller(plan['parent'], observe)
            record.update(source_binding=parent, control=control)
            for request in plan['stages']:
                b.executor.check_pins(plan['source_sha256'])
                if request['gpus']:
                    record[request['name'] + '_resources'] = b.executor.check_resources(request['gpus'])
                b.executor.launch_child(request, changed)
            audit = b.read(TARGET / 'formal32/fresh_audit.json')
            restored = b.executor.validate_evaluation(b.EVIDENCE / 'evaluation', audit['checkpoint_id'])
            b.require(audit['source_checkpoint_id'] == parent['checkpoint_id'], 'Recovery audit belongs to another actual C1.')
            record.update(status='completed_actual_fresh_C1_Rice_Muon32_pair', exit_code=0, restored=restored,
                restored_minus_C1={task:restored['task_scores'][task] - control['task_scores'][task] for task in b.executor.TASKS},
                nine_delta=restored['nine_task_score'] - control['nine_task_score'])
            b.executor.check_pins(plan['source_sha256'])
            code = 0
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            b.core.write(DIRECTORY / 'run.json', record)
            (DIRECTORY / 'run.exit').write_text(str(code) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
