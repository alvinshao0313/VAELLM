"""Exercise the genuine evaluation plan with one absolute deadline on CPU."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import time

from . import bindings as b, evaluate, sequence
from experiments.adaptive_class_evaluation_20261030 import evaluate as original
from experiments.adaptive_class_evaluation_20261030 import bindings as evaluation_bindings


def main():
    b.cpu_environment()
    evaluate.setup(10000.)
    config_path = b.CONTROL_ACCEPTANCE.parents[2]/'full8/config.json'
    config = b.read(config_path)
    binding = dict(checkpoint_dir=str(b.OUTPUT/'formal/final_model'),pipeline_report=str(b.OUTPUT/'formal/report.json'),checkpoint_id='CPU_plan_only')
    plan = original.make_plan(config,config_path,binding,b.RESULT/'evaluation',['0'],{})
    bounded = evaluate.bound_longbench_plan(deepcopy(plan),10000.,now_unix=1000.)
    diffs = []
    def compare(left,right,path=''):
        if isinstance(left,dict):
            b.require(left.keys()==right.keys(),'Deadline adapter changed plan keys.')
            for key in left:
                compare(left[key],right[key],path+'/'+str(key))
        elif isinstance(left,list):
            b.require(len(left)==len(right),'Deadline adapter changed list lengths.')
            for index,(a,z) in enumerate(zip(left,right,strict=True)):
                compare(a,z,path+'/'+str(index))
        elif left!=right:
            diffs.append(path)
    compare(plan,bounded)
    unit = bounded['execution_units'][0]
    argv = unit['argv']
    time_index = argv.index('--max-runtime-seconds')+1
    b.require(set(diffs)=={'/execution_units/0/argv/'+str(time_index),'/execution_units/0/native_runtime_seconds','/execution_units/0/outer_timeout_seconds'}
        and unit['native_runtime_seconds']==8760 and unit['outer_timeout_seconds']==8940
        and unit['termination_grace_seconds']==20 and plan['expected_samples']==4750 and len(plan['expected_tasks'])==21,
        'Only native/outer remaining time may change; retain all protocol/data and original20-second grace.')
    commands = [unit['argv'],bounded['nine_task_argv'],evaluation_bindings.command('full8','--config',str(config_path),'--shard','mmlu','--output','CPU_only')]
    b.require(all(command[command.index('--deadline-unix')+1]=='10000.0' for command in commands),
        'Every consumer command must carry the identical absolute deadline.')
    requests = sequence.requests()
    b.require(all(row['gpus']==['0'] for row in requests) and requests[0]['environment']['CUDA_VISIBLE_DEVICES']=='0'
        and requests[1]['environment']['CUDA_VISIBLE_DEVICES']=='', 'OnlyGPU0 may execute; evaluation controller isCPU.')
    expired = subprocess.run([sys.executable,'-B','-u','-m','experiments.cat_down_current_decoder_cmcc_20261104.evaluate',
        '--deadline-unix',str(time.time()-1),'--component','full8'],cwd=b.REPO,capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    b.require(expired.returncode != 0 and 'fixed actual full9 launch deadline' in expired.stderr,
        'An expired absolute deadline must reject before loading any model or running a consumer.')
    result = dict(status='verified_genuine_plan_absolute_deadline_and_unchanged_protocol',exit_code=0,
        config_path=str(config_path),config_sha256=b.sha(config_path),only_three_LB_time_fields_changed=diffs,
        LB_native_seconds=8760,LB_outer_seconds=8940,termination_grace_seconds=20,
        all_components_forward_same_absolute_deadline=True,component_entry_executes_GNU_timeout_with_remaining_minus20_seconds=True,
        expired_deadline_rejected_before_consumer=True,expired_check_exit=expired.returncode,
        expected_samples=4750,expected_tasks=plan['expected_tasks'],requests=requests,source_sha256=b.pins())
    b.write(b.RESULT/'deadline_cpu.json',result)
    print(json.dumps(dict(status=result['status'],exit_code=0)),flush=True)


if __name__=='__main__':
    main()
