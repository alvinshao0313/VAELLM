"""One actual C0 -> prepare -> Muon32 -> fresh -> complete-nine execution."""
import argparse
import fcntl
import os
from pathlib import Path
import select
import sys
import time
import traceback

from . import bindings as b
from experiments.cmcc_runtime_20261030 import sequence as executor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=5, choices=(5,))
    args = parser.parse_args()
    executor.cpu_environment()
    b.configure()
    control_plan = b.read(args.control_plan)
    b.require(control_plan['schema'] == 'fresh_C0_fixed2_single_sequence_v1'
        and control_plan['evaluation']['gpus'] == ['6','7'], 'Use the actual fixed C0 control and its reserved evaluation pair.')
    control_state = b.read(args.control_plan.parent/'run.json')
    parent, _ = b.source(Path(control_plan['formal']['output'])/'final_model', control_plan['formal']['report'])
    b.require(Path(control_plan['formal']['exit_file']).read_text().strip() == '0',
        'Start only after the real C0 formal process exited0, independently of its evaluation.')
    descriptor = None
    actual = executor.proc_start(control_state['pid'])
    terminal_control = control_state.get('status') in ('completed_C0_and_complete_nine', 'failed')
    if actual is not None and not terminal_control:
        b.require(actual == control_state['start_ticks'], 'C0 resource-owner PID was recycled.')
        try:
            descriptor = executor.pidfd_open(control_state['pid'])
        except ProcessLookupError:
            descriptor = None
    b.require(actual is not None or terminal_control,
        'The C0 controller disappeared without a terminal record; inspect its actual state before recovery.')
    root = args.output.resolve()
    b.require(not root.exists(), 'Never overwrite or automatically resume a recovery pipeline.')
    root.mkdir(parents=True)
    record = dict(schema='fresh_C0_Muon32_pipeline_v1', status='starting', pid=os.getpid(),
        start_ticks=executor.proc_start(os.getpid()), started_unix=time.time(), argv=sys.argv,
        parent=parent, control_plan=b.binding(args.control_plan), source=b.code_bindings(), stages={},
        scope='One already-budgeted C0 recovery32 and one full9; no change to active C0/control pipeline.')
    path = root/'run.json'
    lock_path = root.parent/'formal32.lock'
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            b.core.write(path, record)
            executor.check_pins(control_plan['source_sha256'])
            study = root/'recovery'

            def changed(stage):
                record['stages'][stage['name']] = stage
                record['status'] = stage['name']+'_'+stage['status']
                b.core.write(path, record)

            def stage(name, module, arguments, gpu='', devices=()):
                for item in record['source'].values():
                    b.require(b.binding(item['path']) == item, 'A recovery source changed: '+item['path'])
                if devices:
                    record[name+'_resources'] = executor.check_resources(list(devices))
                row = dict(name=name, argv=[str(b.PYTHON),'-B','-u','-m',module,*arguments],
                    environment=executor.environment(gpu), log=str(root/(name+'.log')), exit_file=str(root/(name+'.exit')))
                return executor.launch_child(row, changed)

            stage('prepare', 'experiments.fresh_muon_cmcc_20261030.prepare',
                ['--checkpoint',parent['checkpoint_dir'],'--pipeline-report',parent['pipeline_report'],'--output',str(study)])
            prepared = study/'prepared.json'
            stage('train32', 'experiments.fresh_muon_cmcc_20261030.run',
                ['--prepared',str(prepared),'--gpu',str(args.gpu)], str(args.gpu), (str(args.gpu),))
            stage('fresh', 'experiments.fresh_muon_cmcc_20261030.fresh',
                ['--prepared',str(prepared),'--gpu',str(args.gpu)], str(args.gpu), (str(args.gpu),))
            # Training does not wait for control scoring. Only the new full9
            # waits for the existing control process to release GPUs6/7.
            record['status'] = 'waiting_control_resource_release'
            b.core.write(path, record)
            if descriptor is not None:
                poller = select.poll()
                poller.register(descriptor, select.POLLIN)
                poller.poll()
            quality = root/'evaluation'
            stage('full9', 'experiments.fresh_muon_cmcc_20261030.evaluate',
                ['--support',str(b.SUPPORT),'--checkpoint',str(study/'formal32/final_model'),
                 '--pipeline-report',str(study/'formal32/report.json'),'--output',str(quality),'--gpus','5,6,7'],
                devices=('5','6','7'))
            audit = b.read(study/'formal32/fresh_audit.json')
            restored = executor.validate_evaluation(quality, audit['checkpoint_id'])
            control = executor.validate_evaluation(control_plan['evaluation']['output'], parent['checkpoint_id'])
            record.update(status='complete_actual_C0_Muon32_pair', exit_code=0, control=control, restored=restored,
                restored_minus_C0={task:restored['task_scores'][task]-control['task_scores'][task] for task in executor.TASKS},
                nine_delta=restored['nine_task_score']-control['nine_task_score'])
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        record['finished_unix'] = time.time()
        b.core.write(path, record)


if __name__ == '__main__':
    main()
