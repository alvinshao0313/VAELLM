"""One fixed two-arm validation, fresh128 learning, and fixed endpoint readout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from . import bindings as b

DIRECTORY = b.EVIDENCE/'single_20261101'
PAIRS = dict(decoder='0,1',dense='2,3')
READOUT_GPU = dict(decoder='4',dense='5')
PACKAGE = 'experiments.mlp_learnability_formal_cmcc_20261101.'


def model_root(arm,short):
    return b.ROOT/(('short_'+arm) if short else arm+'128')


def requests(arm,short):
    model = model_root(arm,short)
    name = ('short_' if short else 'formal_')+arm

    def row(suffix,module,arguments,gpu=''):
        return dict(name=name+'_'+suffix,argv=[str(b.PYTHON),'-B','-u','-m',PACKAGE+module,*map(str,arguments)],
            environment=b.executor.environment(gpu),log=str(DIRECTORY/(name+'_'+suffix+'.log')),
            exit_file=str(DIRECTORY/(name+'_'+suffix+'.exit')))

    rows = [row('train','run',['--arm',arm,'--mode','short' if short else 'formal',
        '--checkpoint',b.C0,'--output',model,'--gpus',PAIRS[arm],
        '--timeout-seconds','14400' if short else '43200'],PAIRS[arm])]
    for step in b.snapshot_steps(short):
        rows.append(row('fresh'+str(step),'fresh',['--directory',model/'snapshots'/f'step_{step:06d}',
            '--gpus',PAIRS[arm]],PAIRS[arm]))
    if short:
        snapshot = model/'snapshots/step_000002'
        rows.append(row('consumer','evaluate',['--support',b.SUPPORT,'--component','consumer',
            '--checkpoint',snapshot/'final_model','--pipeline-report',snapshot/'report.json',
            '--output',b.EVIDENCE/('consumer_'+arm)],READOUT_GPU[arm]))
    return rows


def evaluations(arm):
    model = model_root(arm,False)
    base = b.EVIDENCE/(arm+'128')
    rows = []
    for step,component,option in ((64,'midpoint','--gpu'),(128,'evaluate','--gpus')):
        snapshot = model/'snapshots'/f'step_{step:06d}'
        name = arm+'_'+component
        rows.append(dict(name=name,argv=[str(b.PYTHON),'-B','-u','-m',PACKAGE+'evaluate',
            '--support',str(b.SUPPORT),'--component',component,'--checkpoint',str(snapshot/'final_model'),
            '--pipeline-report',str(snapshot/'report.json'),'--output',str(base/component),option,READOUT_GPU[arm]],
            environment=b.executor.environment(''),log=str(DIRECTORY/(name+'.log')),
            exit_file=str(DIRECTORY/(name+'.exit'))))
    return rows


def prepare():
    b.executor.cpu_environment()
    b.require(not DIRECTORY.exists() and all(not model_root(arm,short).exists()
        for arm in PAIRS for short in (True,False)), 'Use fresh fixed study plan and arm outputs.')
    source = b.source_C0()
    pins = b.code_bindings()
    b.executor.check_pins(pins)
    plan = dict(schema='C0_all_MLP_fixed_learning_sequence_v1',created_unix=time.time(),output=str(DIRECTORY),
        source_binding=source,source_sha256=pins,training_contract=b.CONTRACT,
        short={arm:requests(arm,True) for arm in PAIRS},formal={arm:requests(arm,False) for arm in PAIRS},
        evaluation={arm:evaluations(arm) for arm in PAIRS},automatic_retry=False,automatic_resume=False,
        resource_wait=False,boundary='One actual-C0 compact/snapshot bridge per arm, then fresh continuous128. '
            'Fixed64 observation and fixed128 full9 after training; no score gate, no GPUs6/7.')
    DIRECTORY.mkdir(parents=True)
    b.write(DIRECTORY/'plan.json',plan)
    print(json.dumps(dict(status='prepared_not_started',plan=str(DIRECTORY/'plan.json'),source_pins=len(pins))),flush=True)


def run():
    b.executor.cpu_environment()
    path = DIRECTORY/'plan.json'
    plan = b.read(path)
    b.require(plan['schema']=='C0_all_MLP_fixed_learning_sequence_v1' and plan['source_binding']==b.source_C0()
        and plan['training_contract']==b.CONTRACT and plan['short']=={arm:requests(arm,True) for arm in PAIRS}
        and plan['formal']=={arm:requests(arm,False) for arm in PAIRS}
        and plan['evaluation']=={arm:evaluations(arm) for arm in PAIRS}, 'The reviewed fixed study plan changed.')
    b.executor.check_pins(plan['source_sha256'])
    record_path = DIRECTORY/'run.json'
    with (DIRECTORY/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.require(not record_path.exists(), 'This fixed study already started; no resume or retry.')
        record = dict(status='starting_short_arms',pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),
            argv=[sys.executable,*sys.argv],plan_sha256=b.sha(path),started_unix=time.time(),stages={})
        b.write(record_path,record)
        mutex = threading.Lock()

        def changed(stage):
            with mutex:
                record['stages'][stage['name']] = stage
                b.write(record_path,record)

        def execute(rows):
            for row in rows:
                b.executor.check_pins(plan['source_sha256'])
                if row['name'].endswith('_consumer'):
                    resources = b.executor.check_resources([row['environment']['CUDA_VISIBLE_DEVICES']])
                    with mutex:
                        record[row['name']+'_resource_check'] = resources
                        b.write(record_path,record)
                b.executor.launch_child(row,changed)

        def pair(mapping):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(execute,mapping[arm]) for arm in PAIRS]
                for future in futures: future.result()

        code = 1
        try:
            pair(plan['short'])
            for arm in PAIRS:
                consumer = b.read(b.EVIDENCE/('consumer_'+arm)/'consumer.json')
                b.require(consumer['status']=='verified_mlp_single_GPU_consumer'
                    and consumer['arm']==arm and consumer['validation_only'] is True,
                    'Both actual saved-short consumers must complete before the fresh formal arms.')
            record['status']='both_short_consumers_verified_starting_fresh128'
            b.write(record_path,record)
            pair(plan['formal'])
            record['status']='formal128_and_all_snapshot_fresh_complete_readouts'
            b.write(record_path,record)
            pair(plan['evaluation'])
            terminal = {}
            for arm in PAIRS:
                base = b.EVIDENCE/(arm+'128')
                mid = b.read(base/'midpoint/midpoint.json')
                full = b.read(base/'evaluate/evaluation.json')
                b.require(mid['status']=='complete_fixed64_mmlu_observation'
                    and full['status']=='complete_full8_full21_current_goal'
                    and mid['source_C0_checkpoint_id']==full['source_C0_checkpoint_id']==b.C0_ID
                    and full['arm']==arm and full['actual_updates']==128,
                    'Fixed64 and fixed128 actual same-C0 readout did not complete.')
                terminal[arm]=dict(midpoint=str(base/'midpoint/midpoint.json'),
                    full9=str(base/'evaluate/evaluation.json'),checkpoint_id=full['checkpoint_id'])
            record.update(status='completed_fixed_C0_learning_comparison',exit_code=0,results=terminal)
            code = 0
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record['finished_unix']=time.time()
            b.write(record_path,record)
            (DIRECTORY/'run.exit').write_text(str(code)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--run',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else run()


if __name__=='__main__':
    main()
