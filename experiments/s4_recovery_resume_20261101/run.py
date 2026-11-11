"""Evaluate completed terminal once, then run only the missing sequential56 tail."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as r

b=r.b


def prepare(root):
    b.require(Path(sys.executable).resolve()==b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'CPU prepare in caller bitvae only.')
    root=root.resolve()
    b.require(root.is_relative_to(b.OUTPUT) and not root.exists(), 'Use a new independent continuation directory.')
    inherited=r.old_binding();imports=r.import_proof();resources=r.available()
    for path in r.HERE.glob('*.py'):compile(path.read_text(),str(path),'exec')
    root.mkdir(parents=True);view=root/'terminal_readout';view.mkdir()
    for name in ('report.json','fresh.json','final_model'):
        (view/name).symlink_to(r.TERMINAL/name,target_is_directory=name=='final_model')
    proof=dict(status='verified_import_context_only_resume_CPU',original=inherited,imports=imports,resources=resources,
        old1062_pins_unchanged=True,no_model_CPU_or_repeated_short=True)
    b.write(root/'CPU_validation.json',proof)
    plan=dict(schema='S4_recovery_import_context_resume_v1',output=str(root),terminal_view=str(view),
        original=inherited,CPU_validation=b.binding(root/'CPU_validation.json'),resume_source_sha256=r.own_pins(),
        old_source_sha256=inherited['original_source_sha256'],original56_contract=b.CONTRACT,
        worker_PYTHONPATH=r.environment('')['PYTHONPATH'],
        steps=['terminal_MMLU_seven_first_scores','sequential56_from_original_b120','sequential_fresh','sequential_MMLU_seven'],
        automatic_retry=False,physical_GPUs=[6,7])
    b.write(root/'plan.json',plan)
    print(dict(status='prepared',plan=str(root/'plan.json'),plan_sha256=b.sha(root/'plan.json')),flush=True)


def results(view,expected_id):
    metrics={};ids=[]
    for shard in ('mmlu','seven'):
        payload=b.read(view/'evaluation'/(shard+'.json'))
        b.require(payload['strict_load'] and not payload['validation_only'] and payload['actual_updates']==56,
            'Use complete formal56 same-protocol endpoint scores.')
        ids.append(payload['checkpoint_id'])
        metrics.update({name:100*value for name,value in payload['verified_task_metrics'].items()})
    b.require(len(metrics)==8 and set(ids)=={expected_id}, 'The eight scores must bind one real endpoint.')
    baseline=b.read(b.BASE/'formal/result.json')['scores_percent']
    return dict(checkpoint_id=expected_id,scores_percent=metrics,eight_mean_percent=sum(metrics.values())/8,
        deltas_from_no_recovery_S4_pp={name:value-baseline[name] for name,value in metrics.items()},
        model_physical_path=str((view/'final_model').resolve()),readout_root=str(view),completed_updates=56,
        scope='Eight original HFLM tasks only; no LongBench, full-nine or final2.6 acceptance.')


def execute(path):
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and Path(sys.executable).resolve()==b.PYTHON.resolve(), 'CPU controller only.')
    plan=b.read(path);root=Path(plan['output']);view=Path(plan['terminal_view'])
    b.require(not (root/'run.json').exists() and r.own_pins()==plan['resume_source_sha256']
        and b.pins()==plan['old_source_sha256'] and r.old_binding()==plan['original'], 'Changed source/control or reused output.')
    resources=r.available()
    b.require(b.binding(root/'CPU_validation.json')==plan['CPU_validation'], 'CPU import proof changed.')
    record=dict(status='evaluating_completed_terminal_first_scores',**b.parent.process_event(os.getpid()),
        started_unix=time.time(),plan_sha256=b.sha(path),resources=resources,stages=[],sequential_formal_started=False)
    b.write(root/'run.json',record)
    def launch(name,module,args,gpu):
        command=[str(b.PYTHON),'-B','-u','-m',r.PACKAGE+'.'+module,*map(str,args)]
        env=r.environment(gpu);log=(root/(name+'.log')).open('x',encoding='utf-8')
        process=subprocess.Popen(command,cwd=b.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        row=dict(name=name,command=command,**b.parent.process_event(process.pid),started_unix=time.time(),
            CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=env['PYTHONPATH'])
        record['stages'].append(row);b.write(root/'run.json',record)
        return process,row,log
    def finish(job):
        process,row,log=job;code=process.wait();log.close()
        row.update(exit_code=code,elapsed_seconds=time.time()-row['started_unix']);b.write(root/'run.json',record)
        return code
    def evaluate(name,directory):
        jobs=[launch(name+'_'+shard,'evaluate',['--root',directory,'--component',shard],gpu)
            for shard,gpu in [('mmlu','6'),('seven','7')]]
        codes=[]
        for job in jobs:
            code=finish(job);codes.append(code)
            if code==0 and job[1]['name'].endswith('_mmlu'):
                payload=b.read(directory/'evaluation/mmlu.json')
                print(dict(event=name+'_MMLU_complete',checkpoint_id=payload['checkpoint_id'],
                    MMLU_percent=100*payload['verified_task_metrics']['mmlu'],output=str(directory/'evaluation/mmlu.json')),flush=True)
        b.require(codes==[0,0], 'A required endpoint evaluation failed; stop without retry or more training.')
    code=1
    try:
        evaluate('terminal',view)
        terminal=results(view,r.TERMINAL_ID);b.write(root/'terminal_result.json',terminal)
        b.require(r.old_binding()==plan['original'], 'Readout view changed original terminal or failure evidence.')
        record.update(status='executing_missing_sequential56_from_b120',sequential_formal_started=True)
        b.write(root/'run.json',record)
        sequential=root/'formal/sequential'
        b.require(finish(launch('formal_sequential','run',['--arm','sequential','--mode','formal','--output',sequential,
            '--cpu-proof',r.OLD/'CPU_validation.json'],'6,7'))==0, 'The missing sequential arm failed; stop without retry.')
        b.require(finish(launch('formal_sequential_fresh','evaluate',['--root',sequential,'--component','fresh'],'6,7'))==0,
            'The actual sequential endpoint failed its existing fresh contract.')
        evaluate('sequential',sequential)
        report=b.read(sequential/'report.json');other=results(sequential,report['export']['checkpoint_id'])
        b.write(root/'result.json',dict(status='complete_original_S4_placement_comparison_tail',
            arms=dict(terminal=terminal,sequential=other),original_S4_parent=b.SOURCE_ID,
            original_failure_preserved=plan['original']['preserved_evidence'],
            old1062_pins_unchanged=True,contract=b.CONTRACT))
        b.require(r.old_binding()==plan['original'], 'Continuation changed original failure evidence.')
        record.update(status='complete_original_S4_placement_comparison_tail',exit_code=0);code=0
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        record['elapsed_seconds']=time.time()-record['started_unix'];b.write(root/'run.json',record)
        (root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare(args.prepare)
    else:execute(args.plan.resolve(strict=True))
