"""One finite short-both -> fresh-parent formal-both chain, with no retry."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b

PACKAGE='experiments.progressive_residual_bsq_20261030.recovery_compare_20261101'


def prepare(root):
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and Path(sys.executable).resolve()==b.PYTHON.resolve()
        and not b.torch.cuda.is_initialized(), 'CPU prepare in caller bitvae only.')
    root=root.resolve()
    b.require(root.is_relative_to(b.OUTPUT) and not root.exists(), 'Use one new comparison directory.')
    source=b.source()
    # Actual immutable cache parser and source checks; no full-model CPU construction or fake updates.
    dataset,recovery,evidence=b.data.load()
    for path in b.HERE.glob('*.py'):compile(path.read_text(),str(path),'exec')
    indices=[index for _,items in b.segments(False) for index in items]
    b.require(indices==list(range(32))+list(range(24)) and len(indices)==56
        and [56-8*i for i in range(7)]==[56,48,40,32,24,16,8], 'Fixed matched update/activation schedule changed.')
    require_fields=('original','effective')
    for name in ('mmlu','seven'):
        baseline=b.read(b.parent.BASELINE/(name+'.json'))
        from experiments.w25_eval.acceptance import leaf_metrics
        metrics,_,batch=leaf_metrics(baseline['results'],baseline['n-samples'],baseline['n-shot'],baseline['configs'])
        b.require((name=='mmlu' and len(metrics)==1 and batch==64) or
            (name=='seven' and len(metrics)==7 and batch==32), 'Original evaluation schema/config changed.')
    root.mkdir(parents=True)
    proof=dict(status='verified_S4_recovery_compare_CPU',source_sha256=b.pins(),source_binding=source,
        data_evidence=evidence,formal_exposure=b.data.exposure(dataset,recovery,indices),
        segment_exposure=[b.data.exposure(dataset,recovery,items) for _,items in b.segments(False)],
        contract=b.CONTRACT,compile_all_new_sources=True,original1033_pins_unchanged=True,
        cuda_initialized=b.torch.cuda.is_initialized(),
        scope='Actual existing static target/source/header/control checks. Factory state and new S4 backward/export require genuine GPU short.')
    b.require(proof['cuda_initialized'] is False, 'CPU preparation must not initialize CUDA.')
    b.write(root/'CPU_validation.json',proof)
    plan=dict(schema='S4_placement_comparison_plan_v1',output=str(root),CPU_validation=b.binding(root/'CPU_validation.json'),
        source_binding=source,source_sha256=b.pins(),contract=b.CONTRACT,
        fixed_order=['short_terminal','fresh_terminal','short_sequential','fresh_sequential',
            'formal_terminal','fresh_terminal','terminal_MMLU_seven','formal_sequential','fresh_sequential','sequential_MMLU_seven'],
        physical_GPUs=[6,7],automatic_retry=False)
    b.write(root/'plan.json',plan)
    print(dict(status='prepared',plan=str(root/'plan.json'),pins=len(plan['source_sha256'])),flush=True)


def execute(path):
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and Path(sys.executable).resolve()==b.PYTHON.resolve(), 'CPU controller only.')
    plan=b.read(path);root=Path(plan['output'])
    b.require(not (root/'run.json').exists() and b.binding(root/'CPU_validation.json')==plan['CPU_validation'],
        'Never overwrite a comparison or changed CPU proof.')
    b.parent.s.check_pins(plan['source_sha256']);b.source()
    record=dict(status='validating_both_S4_recovery_short',**b.parent.process_event(os.getpid()),
        started_unix=time.time(),plan_sha256=b.sha(path),stages=[],formal_started=False)
    b.write(root/'run.json',record)
    def launch(name,module,arguments,gpu):
        command=[str(b.PYTHON),'-B','-u','-m',PACKAGE+'.'+module,*map(str,arguments)]
        log=(root/(name+'.log')).open('x',encoding='utf-8')
        process=subprocess.Popen(command,cwd=b.REPO,env=b.parent.environment(gpu),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
        row=dict(name=name,command=command,CUDA_VISIBLE_DEVICES=str(gpu),**b.parent.process_event(process.pid),started_unix=time.time())
        record['stages'].append(row);b.write(root/'run.json',record)
        return process,row,log
    def finish(job):
        process,row,log=job;code=process.wait();log.close()
        row.update(exit_code=code,elapsed_seconds=time.time()-row['started_unix'])
        b.write(root/'run.json',record)
        b.require(code==0, 'Fixed comparison component failed, stopping without retry: '+row['name'])
    def arm(mode,name):
        directory=root/mode/name
        finish(launch(mode+'_'+name,'run',['--arm',name,'--mode',mode,'--output',directory,
            '--cpu-proof',root/'CPU_validation.json'],'6,7'))
        finish(launch(mode+'_'+name+'_fresh','evaluate',['--root',directory,'--component','fresh'],'6,7'))
        return directory
    code=1
    try:
        for name in ('terminal','sequential'):arm('short',name)
        record.update(status='executing_fresh_parent_formal_comparison',formal_started=True)
        b.write(root/'run.json',record)
        summaries={}
        for name in ('terminal','sequential'):
            directory=arm('formal',name)
            jobs=[launch('formal_'+name+'_'+shard,'evaluate',['--root',directory,'--component',shard],gpu)
                for shard,gpu in [('mmlu','6'),('seven','7')]]
            codes=[]
            # Both launched consumers are always reaped, even if one fails.
            for process,row,log in jobs:
                exit_code=process.wait();log.close();codes.append(exit_code)
                row.update(exit_code=exit_code,elapsed_seconds=time.time()-row['started_unix'])
                b.write(root/'run.json',record)
            b.require(codes==[0,0], 'A required endpoint evaluation failed; do not start the next arm.')
            metrics={}
            ids=[]
            for shard in ('mmlu','seven'):
                result=b.read(directory/'evaluation'/(shard+'.json'))
                metrics.update({key:100*value for key,value in result['verified_task_metrics'].items()})
                ids.append(result['checkpoint_id'])
            b.require(len(metrics)==8 and len(set(ids))==1, 'Endpoint eight tasks must bind the same model.')
            parent=b.read(b.BASE/'formal/result.json')['scores_percent']
            summaries[name]=dict(checkpoint_id=ids[0],scores_percent=metrics,eight_mean_percent=sum(metrics.values())/8,
                deltas_from_no_recovery_S4_pp={key:value-parent[key] for key,value in metrics.items()},
                completed_updates=56,scope='Eight tasks only; no full-nine or final2.6 acceptance.')
        b.write(root/'result.json',dict(status='complete_S4_placement_comparison',arms=summaries,
            original_S4_parent=b.SOURCE_ID,contract=b.CONTRACT))
        record.update(status='complete_S4_placement_comparison',exit_code=0);code=0
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
