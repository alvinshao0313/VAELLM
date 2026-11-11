"""One GPU, exactly three independent declared candidates, finite owned workers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cat_up_rate_profiles_cmcc_20261104.run import _kill_group


def wait_child(command, environment, output, deadline):
    with (output/'worker.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        from . import bindings as b
        row=dict(command=command,pid=child.pid,process_group_id=child.pid,started_epoch=time.time(),absolute_deadline_epoch=deadline)
        if '--request' in command:
            request=Path(command[command.index('--request')+1]);row.update(request=str(request),request_sha256=b.sha(request))
        b.write(output/'process.json',row)
        try:
            code=child.wait(timeout=max(0.,deadline-time.time()))
        except BaseException as error:
            row['error']=type(error).__name__+': '+str(error); _kill_group(child.pid); child.wait(); raise
        finally:
            _kill_group(child.pid); child.wait(); row.update(exit_code=child.returncode,finished_epoch=time.time()); b.write(output/'exit.json',row)
        return code


def process_start(pid):
    try:return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None


def wait_for_prior(plan_path, pid, start_ticks, gpu):
    from . import bindings as b
    plan=b.read(plan_path);reference=b.bound(plan_path);auth=plan['authorization']
    b.require(auth and auth['physical_GPU']==gpu and gpu in (1,3,4), 'Bind the actual existing full8 plan on this same GPU.')
    deadline=auth['absolute_deadline_epoch']+30
    while True:
        actual=process_start(pid)
        b.require(actual is None or actual==start_ticks, 'Prior controller PID generation changed; do not wait for or disturb a different process.')
        if actual is not None:
            command=Path('/proc',str(pid),'cmdline').read_bytes().decode().split('\0')
            b.require('experiments.cat_offline_execution_cmcc_20261104.evaluate' in command
                and '--component' in command and command[command.index('--component')+1]=='run'
                and '--plan' in command and Path(command[command.index('--plan')+1]).resolve()==Path(plan_path).resolve(),
                'Prior controller live command does not belong to this exact existing full8 plan.')
        if actual is None:
            evaluation=b.read(Path(plan['output_dir'])/'evaluation.json')
            b.require(evaluation['source_plan']==reference and evaluation['checkpoint_id']==plan['endpoint']['checkpoint_id']
                and evaluation.get('finished_unix') and ((evaluation['status']=='complete_original_full8' and evaluation['exit_code']==0)
                    or (evaluation['status']=='failed' and evaluation['exit_code']!=0)), 'Prior process exited without its actual terminal evaluation record; stop this queue.')
            return dict(plan=reference,pid=pid,start_ticks=start_ticks,terminal_status=evaluation['status'],finished_unix=evaluation['finished_unix'])
        b.require(time.time()<deadline, 'Existing upstream full8 exceeded its own bounded deadline/cleanup; no queue takeover.')
        time.sleep(min(5.,max(.1,deadline-time.time())))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=range(6),required=True)
    parser.add_argument('--queue-root',type=Path,required=True);parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--wait-plan',type=Path);parser.add_argument('--wait-controller-pid',type=int);parser.add_argument('--wait-controller-start-ticks',type=int)
    args=parser.parse_args(); declared=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['ACCELERATE_USE_CPU']='true'
    from . import bindings as b
    from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
    from experiments.cmcc_runtime_20261030.sequence import check_resources
    from experiments.cat_offline_execution_cmcc_20261104 import evaluate
    from experiments.cat_offline_execution_cmcc_20261104.results import worker_result,attach_full8
    require(os.environ.get('CONDA_DEFAULT_ENV')=='bitvae' and declared==str(args.gpu), 'Use caller bitvae and the actual assigned GPU.')
    manifest_path=args.queue_root.resolve()/'manifest.json';manifest=b.read(manifest_path);user=b.approval(manifest,manifest_path,args.authorization)
    dependency=None
    waits=(args.wait_plan,args.wait_controller_pid,args.wait_controller_start_ticks)
    require(all(value is None for value in waits) or all(value is not None for value in waits), 'An existing resource dependency requires its exact plan/PID/start ticks.')
    if args.wait_plan is not None:dependency=wait_for_prior(args.wait_plan,args.wait_controller_pid,args.wait_controller_start_ticks,args.gpu)
    check_resources([str(args.gpu)])
    root=manifest_path.parent/('gpu'+str(args.gpu));root.mkdir();started=time.time()
    state=dict(status='running',gpu=args.gpu,category=b.CARDS[args.gpu],controller_pid=os.getpid(),started_epoch=started,
        absolute_deadline_epoch=started+16200,manifest=b.bound(manifest_path),authorization=b.bound(args.authorization),prior_resource_dependency=dependency,candidates=[])
    b.write(root/'queue.json',state)
    try:
        for row in manifest['queues'][str(args.gpu)]:
            b.integrity(manifest)
            directory=root/row['label'];directory.mkdir();candidate_started=time.time()
            item=dict(label=row['label'],candidate_id=row['candidate_id'],status='running',started_epoch=candidate_started,
                absolute_deadline_epoch=min(candidate_started+5400,started+16200),stages=[])
            state['candidates'].append(item);b.write(root/'queue.json',state)
            try:
                for phase in ('technical_validation','formal'):
                    check_resources([str(args.gpu)])
                    request=b.request(manifest,manifest_path,args.authorization,args.gpu,row,directory,phase,started,candidate_started)
                    registry=directory/(phase+'_request.json');b.write_new(registry,request);output=directory/phase;output.mkdir()
                    command=[sys.executable,'-B','-u','-m','experiments.cat_night_profiles_cmcc_20261104.search','--request',str(registry),'--worker']
                    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),ACCELERATE_USE_CPU='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                    item['stages'].append(dict(phase=phase,request=b.bound(registry),deadline=request['absolute_deadline_epoch']));b.write(root/'queue.json',state)
                    code=wait_child(command,environment,output,request['absolute_deadline_epoch'])
                    require(code==0,'Actual '+phase+' worker failed; no dependent stages or retry for this candidate.')
                    completed=b.read(output/'completed.json');require(completed['status']=='completed' and completed['exit_code']==0 and completed['request_identity']==b.digest(request),'Missing or mismatched actual completed receipt.')
                    if phase=='technical_validation':b.technical_proof(manifest,row,directory)
                result=worker_result(directory/'formal/completed.json',row['candidate_id'],request_reference=b.bound(directory/'formal_request.json'))
                b.write_new(directory/'formal/profile_result.json',result)
                b.integrity(manifest);check_resources([str(args.gpu)])
                now=time.time();deadline=b.phase_window(started,candidate_started,now,'full8')
                authorization=dict(authorization_reference=user['reference'],physical_GPU=args.gpu,authorized_GPU_hours=.25,
                    authorized_start_epoch=now,absolute_deadline_epoch=deadline)
                full8=directory/'full8';plan=evaluate.make_plan(result=result,baseline=b.load_baseline(b.BASELINE),baseline_path=b.BASELINE,output=full8,authorization=authorization)
                full8.mkdir();b.write_new(full8/'plan.json',plan)
                item['stages'].append(dict(phase='full8',entrypoint='experiments.cat_offline_execution_cmcc_20261104.evaluate.run',plan=b.bound(full8/'plan.json'),deadline=deadline));b.write(root/'queue.json',state)
                evaluate.run(full8/'plan.json')
                scored=attach_full8(result,summary_path=full8/'full8_summary.json',evaluation_path=full8/'evaluation.json');b.write_new(directory/'formal/profile_full8_result.json',scored)
                item.update(status='completed',exit_code=0,full8_mean=scored['full8_mean'])
            except Exception as error:
                item.update(status='failed',exit_code=1,error=type(error).__name__+': '+str(error))
                b.integrity(manifest)  # Common source/protocol failures stop all remaining work on this queue.
            finally:
                item['finished_epoch']=time.time();b.write(root/'queue.json',state)
        failed=sum(row['status']=='failed' for row in state['candidates'])
        state.update(status='completed_with_failures' if failed else 'completed_finite_queue',exit_code=1 if failed else 0,failed_candidates=failed)
    except BaseException as error:
        state.update(status='stopped',exit_code=1,error=type(error).__name__+': '+str(error));raise
    finally:
        state['finished_epoch']=time.time();b.write(root/'queue.json',state)
    return state['exit_code']


if __name__=='__main__':raise SystemExit(main())
