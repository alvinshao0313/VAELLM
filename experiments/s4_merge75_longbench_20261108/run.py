"""Finite four-part generation, explicit terminal binding, canonical merge and score."""
import os
from pathlib import Path
import subprocess
import time
import traceback
from . import bindings as b


def cpu(name,argv,report):
    env=b.environment('');env['PYTHONPATH']=str(b.ROOT)
    with (b.RUN/(name+'.log')).open('x') as log:
        child=subprocess.Popen(argv,cwd=b.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        report['cpu_stages'][name]=dict(command=argv,**b.parent.process_event(child.pid));b.write(b.RUN/'run.json',report)
        code=child.wait()
    report['cpu_stages'][name]['exit_code']=code;b.write(b.RUN/'run.json',report)
    (b.RUN/(name+'.exit')).write_text(str(code)+'\n');b.require(code==0,name+' failed; stop without retry.')


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (b.RUN/'run.json').exists(),
        'Fresh CPU controller required; no duplicate or retry.')
    request=b.read(b.RUN/'request.json');b.check(request);config=b.read(b.RUN/'config.json')
    b.require(b.sha(b.RUN/'config.json')==request['config_sha256'],'Config changed.')
    plan_path=b.RUN/'merge_plan.json';plan=b.read(plan_path)
    resource=b.partition_api().check_resources(b.GPUS)
    report=dict(status='running_selected_merge75_full21',**b.parent.process_event(os.getpid()),
        request=b.binding(b.RUN/'request.json'),checkpoint_id=b.SOURCE_ID,initial_resources=resource,
        execution_units={},cpu_stages={});b.write(b.RUN/'run.json',report)
    start=time.monotonic();code=1;processes=[]
    try:
        helper=b.helper()
        try:
            for index,unit in enumerate(plan['execution_units']):
                group=Path(unit['output_dir']);group.mkdir(parents=True,exist_ok=False)
                child=helper.start_group(unit,config,plan['config_sha256']);processes.append((index,child))
                report['execution_units'][str(index)]=dict(gpu=unit['gpu'],datasets=unit['datasets'],
                    **b.parent.process_event(child.pid));b.write(b.RUN/'run.json',report)
        finally:
            for index,child in processes:
                status=child.wait();report['execution_units'][str(index)]['exit_code']=status;b.write(b.RUN/'run.json',report)
        b.require(len(processes)==4 and all(x['exit_code']==0 for x in report['execution_units'].values()),
            'One canonical partition failed; no partial scoring or retry.')
        for index,unit in enumerate(plan['execution_units']):
            group=Path(unit['output_dir']);reports=list((group/'runs').glob('*.json'))
            b.require(len(reports)==1,'One actual fresh native attempt per partition required.')
            native=b.read(reports[0])
            b.require(native['status']=='partition_complete' and native['exit_code']==0
                and native['already_present']==0 and native['new_predictions']==unit['expected_samples']
                and native['load_audit']['checkpoint_id']==b.SOURCE_ID,'Partition terminal/coverage/UUID invalid.')
            cpu('bind_'+str(index),[str(b.PYTHON),'-B',str(b.TERMINAL/'bind_terminal.py'),
                '--plan',str(plan_path),'--unit-index',str(index),'--native-report',str(reports[0]),
                '--outer-launch',str(group/'outer.launch.json'),'--outer-exit',str(group/'outer.exit'),
                '--outer-log',str(group/'outer.log')],report)
        bound=b.read(plan_path)
        b.require(not bound['pending_output_directories_at_last_terminal']
            and sum(u['completed_samples_at_last_terminal'] for u in bound['execution_units'])==4750,
            'All21 tasks must be complete before scoring.')
        bound['status']='prepared_not_scored';b.write(plan_path,bound)
        cpu('merge_full21_nine',[str(b.PYTHON),'-B',str(b.TERMINAL/'merge_full21.py'),'--plan',str(plan_path)],report)
        final=b.read(b.RUN/'merged/nine_task_acceptance.json')
        b.require(final['checkpoint_id']==b.SOURCE_ID and final['status']=='complete_S4_merge75_same_checkpoint_nine',
            'Same-checkpoint nine summary invalid.')
        b.check(request);report.update(status='complete_selected_merge75_full21_and_nine',
            result=b.binding(b.RUN/'merged/nine_task_acceptance.json'),quality_acceptance=final['quality_acceptance']);code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-start);b.write(b.RUN/'run.json',report)
        (b.RUN/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
