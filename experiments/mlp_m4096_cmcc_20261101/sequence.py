"""One fresh decoder128, strict64/128 fresh, fixed64 MMLU and128 complete-nine."""
import argparse
import fcntl
import os
import sys
import time
import traceback

from . import bindings as b

DIRECTORY=b.EVIDENCE/'single_20261101'
MODEL=b.ROOT/'decoder128'
PACKAGE='experiments.mlp_m4096_cmcc_20261101.'


def requests():
    def row(name,module,arguments,gpus):
        return dict(name=name,argv=[str(b.PYTHON),'-B','-u','-m',PACKAGE+module,*map(str,arguments)],
            environment=b.executor.environment(gpus),log=str(DIRECTORY/(name+'.log')),
            exit_file=str(DIRECTORY/(name+'.exit')))
    rows=[row('train','run',['--arm','decoder','--mode','formal','--checkpoint',b.C0,
        '--output',MODEL,'--gpus','2,3','--timeout-seconds','43200'],'2,3')]
    for step in (64,128):
        rows.append(row('fresh'+str(step),'fresh',['--directory',MODEL/'snapshots'/f'step_{step:06d}',
            '--gpus','2,3'],'2,3'))
    for step,component,flag in ((64,'midpoint','--gpu'),(128,'evaluate','--gpus')):
        snapshot=MODEL/'snapshots'/f'step_{step:06d}'
        rows.append(row(component,'evaluate',['--support',b.SUPPORT,'--component',component,
            '--checkpoint',snapshot/'final_model','--pipeline-report',snapshot/'report.json',
            '--output',b.EVIDENCE/component,flag,'3'],''))
    return rows


def prepare():
    b.executor.cpu_environment()
    b.require(not DIRECTORY.exists() and not MODEL.exists(),'Use fresh independent coverage outputs.')
    validation=b.read(b.EVIDENCE/'cpu/validation.json')
    b.require(validation['status']=='verified_same_C0_M4096_real_data_targets_and_dispatch'
        and validation['data_contract']==b.data_binding(),'The actual CPU data/targets contract must be complete.')
    plan=dict(schema='same_C0_M4096_fixed128_sequence_v1',source_binding=b.source_C0(),
        source_sha256=b.code_bindings(),training_contract=b.CONTRACT,data_contract=b.data_binding(),
        requests=requests(),created_unix=time.time(),automatic_retry=False,automatic_resume=False,
        resource_wait=False,validation_reused='Same completed C0 D128 training/snapshot/fresh/consumer path; only CPU coverage/source changes',
        boundary='Only M unique coverage changes; measured tokens differ. Fixed128 no quality gate or grid. GPU2/3 only.')
    DIRECTORY.mkdir(parents=True)
    b.write(DIRECTORY/'plan.json',plan)
    print(dict(status='prepared_not_started',plan=str(DIRECTORY/'plan.json'),pins=len(plan['source_sha256'])),flush=True)


def run():
    b.executor.cpu_environment()
    path=DIRECTORY/'plan.json'
    plan=b.read(path)
    b.require(plan['schema']=='same_C0_M4096_fixed128_sequence_v1' and plan['source_binding']==b.source_C0()
        and plan['training_contract']==b.CONTRACT and plan['data_contract']==b.data_binding()
        and plan['requests']==requests(),'The reviewed fixed128 coverage plan changed.')
    b.executor.check_pins(plan['source_sha256'])
    with (DIRECTORY/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.require(not (DIRECTORY/'run.json').exists(),'No retry/resume of this fixed run.')
        record=dict(status='starting_fresh128',pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),
            argv=[sys.executable,*sys.argv],plan_sha256=b.sha(path),started_unix=time.time(),stages={})
        def changed(stage):
            record['stages'][stage['name']]=stage
            b.write(DIRECTORY/'run.json',record)
        code=1
        try:
            for row in plan['requests']:
                b.executor.check_pins(plan['source_sha256'])
                b.executor.launch_child(row,changed)
            mid=b.read(b.EVIDENCE/'midpoint/midpoint.json')
            full=b.read(b.EVIDENCE/'evaluate/evaluation.json')
            b.require(mid['status']=='complete_fixed64_mmlu_observation'
                and full['status']=='complete_full8_full21_current_goal'
                and full['arm']=='decoder' and full['actual_updates']==128
                and mid['data_contract']==full['data_contract']==b.data_binding(),
                'Require this same-data64 observation and fixed128 complete nine.')
            record.update(status='completed_same_C0_M4096_fixed128',exit_code=0,
                checkpoint_id=full['checkpoint_id'],midpoint=str(b.EVIDENCE/'midpoint/midpoint.json'),
                complete_nine=str(b.EVIDENCE/'evaluate/evaluation.json'))
            code=0
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record['finished_unix']=time.time()
            b.write(DIRECTORY/'run.json',record)
            (DIRECTORY/'run.exit').write_text(str(code)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--run',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else run()


if __name__=='__main__':main()
