"""Independent single-card short-to128 chains and their finite approved readouts."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from . import bindings as b,diagnostics

MODULE='experiments.s4_webmed_recovery_20261105'
TERMINAL=b.ROOT/'result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21'
HELPERS=b.ROOT/'result/qwen3_8b_mmlu65_20261004/qwen_aux_retention32_batch_20261011/evaluate.py'


def prepare(root):
    b.initialize();b.require(root==b.RUN and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fixed isolated CPU preparation required.')
    b.require(not (root/'plan.json').exists(),'Never overwrite a prepared plan.')
    dataset,targets,evidence=b.data.load()
    parent=b.source();control=b.event(b.CONTROL)
    b.require(control['checkpoint_id']==b.CONTROL_ID,'Fixed half-LR control changed.')
    inputs=b.read(b.DATA/'dataset_prepared.json');pins=b.pins();requests={}
    prior_validation=b.prior.OUTPUT/'trial_20261104_1940/joint_half_lr/validation/training.json'
    reused=b.read(prior_validation)
    b.require(reused['exit_code']==0 and reused['full_frozen_complement_exact']
        and reused['fresh']['full_state_exact'] and reused['midpoint_readout_followed_by_real_update']
        and reused['parent']==parent,'Existing same-parent single-card lifecycle evidence required.')
    reused_validation=b.binding(prior_validation)
    for short,name in [(True,'validation'),(False,'formal')]:
        output=root/'webmed'/name;output.mkdir(parents=True)
        request=dict(schema='S4_WebMed_same_parent_fixed128_v1',output=str(output),arm='webmed',gpu='6',
            validation_only=short,parent=parent,matched_control=control,source_sha256=pins,contract=b.CONTRACT,
            data_digest=evidence['actual_input_target_records_sha256'],data_contract=b.binding(b.DATA/'data_contract.json'),
            heldout=inputs['heldout'],short_indices=inputs['short_indices'],fixture=b.binding(root/'fixture.pt'),
            reused_validation=reused_validation)
        b.write(output/'request.json',request);requests[name]=b.binding(output/'request.json')
    cpu=dict(status='verified_fixed_Web2048_Med2048_original_QA_CPU',source_sha256=pins,parent=parent,matched_control=control,
        data_digest=evidence['actual_input_target_records_sha256'],source_summary=inputs['summary'],
        actual_shapes=inputs['shapes'],short_indices=inputs['short_indices'],
        original_QA_inputs_targets_group_order_weights_exact=True,heldout_disjoint128=True,
        actual_native_M32_packer_all128_groups_verified=True,reused_validation=reused_validation,
        existing_verified_M_max_length=1222,existing_verified_QA_max_length=10593,
        CUDA_initialized=b.torch.cuda.is_initialized())
    b.require(not cpu['CUDA_initialized'],'CPU finalize initialized CUDA.')
    b.write(root/'CPU_validation.json',cpu)
    plan=dict(schema='S4_WebMed_same_parent_plan_v1',root=str(root),source_sha256=pins,contract=b.CONTRACT,
        parent=parent,matched_control=control,requests={'webmed':requests},CPU_validation=b.binding(root/'CPU_validation.json'),
        automatic_retry=False,automatic_research=False,wall_clock_cutoff=None)
    b.write(root/'plan.json',plan)
    print(dict(event='CPU_plan_complete',source_count=len(pins),plan_sha256=b.sha(root/'plan.json')),flush=True)


def stage(root,name,argv,gpu,report):
    root=Path(root)
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        child=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        report['processes'][name]=dict(command=argv,gpu=gpu,**b.parent.process_event(child.pid));b.write(root/'run.json',report)
        code=child.wait()
    report['processes'][name]['exit_code']=code;b.write(root/'run.json',report)
    (root/(name+'.outer.exit')).write_text(str(code)+'\n')
    b.require(code==0,name+' failed; no automatic retry or dependent execution.')


def cpu_stage(root,name,argv):
    env=b.environment('');env['PYTHONPATH']=str(b.ROOT)
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        result=subprocess.run(argv,cwd=b.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (root/(name+'.exit')).write_text(str(result.returncode)+'\n');b.require(result.returncode==0,name+' CPU terminal failed.')


def full_nine(path,root,checkpoint,gpu,mmlu_reuse=None):
    from experiments.longbench_v1.models import make_config,check_config
    from experiments.longbench_v1.common import file_stat
    from experiments.longbench_v1.protocol import DATASETS
    root.mkdir(parents=True,exist_ok=True);request=b.read(path);event=b.event(checkpoint)
    endpoint_arm='parent0' if event['checkpoint_id']==b.SOURCE_ID else request['arm']
    report=dict(status='running_full_nine_local_endpoint',checkpoint=event,parent=request['parent'],arm=endpoint_arm,processes={})
    evaluation=root/'evaluation'
    if mmlu_reuse is None:
        stage(root,'mmlu',[str(b.PYTHON),'-B','-u','-m',MODULE+'.evaluate','--request',str(path),
            '--checkpoint',str(checkpoint),'--component','mmlu','--output',str(evaluation)],gpu,report)
        mmlu=evaluation/'mmlu.json'
    else:
        mmlu=Path(mmlu_reuse);b.require(b.read(mmlu)['checkpoint_id']==event['checkpoint_id'],'Reused M must belong to this exact endpoint.')
        b.write(root/'mmlu_reuse.json',dict(source=b.binding(mmlu),source_checkpoint=event,new_inference_process=False))
    stage(root,'seven',[str(b.PYTHON),'-B','-u','-m',MODULE+'.evaluate','--request',str(path),
        '--checkpoint',str(checkpoint),'--component','seven','--output',str(evaluation)],gpu,report)
    endpoint_request=dict(request,output=str(root),arm=endpoint_arm,checkpoint=str(checkpoint),mmlu_path=str(mmlu),seven_path=str(evaluation/'seven.json'))
    endpoint_path=root/'endpoint_request.json';b.write(endpoint_path,endpoint_request)
    base=root/'longbench';base.mkdir()
    config=make_config(b.parent.MANIFEST,kind='dense',hf_path=b.HF,mode='full')
    config['model'].update(kind='compressed',checkpoint=str(checkpoint),checkpoint_id=event['checkpoint_id'],
        metadata_sha256=event['metadata_sha256'],weight_stat=file_stat(Path(checkpoint)/'pytorch_model.bin'))
    config['source_sha256'].update(request['source_sha256']);config['research_parent']=dict(scope=b.CONTRACT['scope'],parent_id=b.SOURCE_ID)
    check_config(config);config_path=base/'config.json';b.write(config_path,config)
    output=base/'groups/part_01'
    argv=[str(b.PYTHON),'-B','-u','-m',MODULE+'.evaluate','--request',str(path),'--component','predict',
        '--config',str(config_path),'--output',str(output),'--datasets',*DATASETS]
    unit=dict(output_dir=str(output),datasets=list(DATASETS),expected_samples=4750,native_reports=[],outer_exit=None,
        launch_records=[],terminal_attempts=[],gpu=gpu,argv=argv,native_runtime_seconds=86400,outer_timeout_seconds=86580,
        termination_grace_seconds=20)
    merged=base/'merged'
    plan=dict(schema='fixed_full21_merge_plan_v1',status='draft_awaiting_explicit_terminal_binding',checkpoint_id=event['checkpoint_id'],
        expected_tasks=list(DATASETS),expected_samples=4750,config_path=str(config_path),config_sha256=b.sha(config_path),protocol=config['protocol'],
        execution_units=[unit],task_file_sources={task:str(output/'predictions'/(task+'.jsonl')) for task in DATASETS},
        pins={**request['source_sha256'],str(config_path):b.sha(config_path)},merged_output=str(merged),scoring_cwd=str(b.ROOT),python_executable=str(b.PYTHON),
        nine_task_argv=[str(b.PYTHON),'-B','-m',MODULE+'.run','--request',str(endpoint_path),'--summarize',str(merged/'nine_task_acceptance.json')],
        automatic_retry=False,automatic_resume=False,partial_scoring=False)
    plan_path=base/'merge_plan.json';b.write(plan_path,plan)
    helper=b.parent.module(HELPERS,'_S4_diagnosis_existing_LB_outer');helper.S=b.ROOT;helper.PYTHON=b.PYTHON
    output.mkdir(parents=True);child=helper.start_group(unit,config,plan['config_sha256'])
    report['processes']['LongBench']=dict(**b.parent.process_event(child.pid),gpu=gpu);b.write(root/'run.json',report)
    code=child.wait();b.require(code==0,'LongBench failed; stop without retry.')
    reports=list((output/'runs').glob('*.json'));b.require(len(reports)==1,'One fresh canonical LongBench attempt required.')
    native=b.read(reports[0]);b.require(native['status']=='partition_complete' and native['exit_code']==0
        and native['new_predictions']==4750 and native['already_present']==0
        and native['load_audit']['checkpoint_id']==event['checkpoint_id'],'LongBench native identity/count/exit failed.')
    cpu_stage(root,'bind_LB',[str(b.PYTHON),'-B',str(TERMINAL/'bind_terminal.py'),'--plan',str(plan_path),'--unit-index','0',
        '--native-report',str(reports[0]),'--outer-launch',str(output/'outer.launch.json'),'--outer-exit',str(output/'outer.exit'),'--outer-log',str(output/'outer.log')])
    bound=b.read(plan_path);b.require(not bound['pending_output_directories_at_last_terminal'],'Incomplete LongBench endpoint.')
    bound['status']='prepared_not_scored';b.write(plan_path,bound)
    cpu_stage(root,'merge_nine',[str(b.PYTHON),'-B',str(TERMINAL/'merge_full21.py'),'--plan',str(plan_path)])
    result=b.read(merged/'nine_task_acceptance.json');b.require(result['checkpoint_id']==event['checkpoint_id'],'Nine-task UUID mismatch.')
    (root/'result.json').symlink_to(merged/'nine_task_acceptance.json')
    report.update(status='complete_same_parent_endpoint_full_nine',exit_code=0);b.write(root/'run.json',report)


def summarize(path,output):
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    from experiments.compare_longbench_v1 import comparison
    from train_utils.checkpoint_bpw import audit_checkpoint
    request=b.read(path);root=Path(request['output']);checkpoint=Path(request['checkpoint']);meta=b.read(checkpoint/'checkpoint_meta.json')
    inventory=model_inventory(b.HF);paths=[Path(request[k]) for k in ['mmlu_path','seven_path']]
    low=collect_results(paths,inventory,checkpoint=(checkpoint,meta))
    baseline=collect_results([b.parent.BASELINE/'mmlu.json',b.parent.BASELINE/'seven.json'],inventory)
    b.require(low['versions']==baseline['versions'],'Full-eight versions differ from BF16.')
    lb=comparison(b.parent.BASELINE/'longbench/scores.json',root/'longbench/merged/scores.json',str(b.parent.MANIFEST))
    budget=audit_checkpoint(checkpoint,max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,'Actual native ledger changed.')
    scores={k:v*100 for k,v in low['task_metrics'].items()};dense={k:v*100 for k,v in baseline['task_metrics'].items()}
    scores['longbench']=lb['task_macro_mean']['compressed'];dense['longbench']=lb['task_macro_mean']['dense']
    b.write(output,dict(status='complete_same_parent_local_S4_recovery_nine',scope=b.CONTRACT['scope'],
        checkpoint_id=meta['checkpoint_id'],checkpoint=b.event(checkpoint),parent=request['parent'],arm=request['arm'],
        scores_percent=scores,BF16_percent=dense,nine_mean_percent=math.fsum(scores.values())/9,BF16_nine_mean_percent=math.fsum(dense.values())/9,
        deltas_pp={k:scores[k]-dense[k] for k in scores},bit_budget=budget,raw_samples=_sample_artifacts(paths),
        source_sha256=request['source_sha256'],contract=b.CONTRACT,upstream_recipe_frozen_for_local_comparison=True,full_BF16_chain_rerun=False))


def execute(plan_path,arm):
    plan=b.read(plan_path);b.parent.s.check_pins(plan['source_sha256']);root=Path(plan['root'])/arm
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (root/'run.json').exists(),'New CPU controller required.')
    began=time.monotonic();code=1;gpu=b.ARMS[arm]['gpu']
    report=dict(status='running_incremental_single_GPU_validation',arm=arm,**b.parent.process_event(os.getpid()),
        plan_sha256=b.sha(plan_path),source_sha256=plan['source_sha256'],processes={})
    try:
        formal_request=b.read(plan['requests'][arm]['formal']['path'])
        names=['validation','formal'] if formal_request['short_indices'] else ['formal']
        for name in names:
            request_path=Path(plan['requests'][arm][name]['path']);request=b.read(request_path)
            b.require(b.binding(request_path)==plan['requests'][arm][name],'Actual request binding changed.')
            if name=='formal':
                b.require(b.binding(request['reused_validation']['path'])==request['reused_validation'],
                    'Reused lifecycle proof changed.')
                if request['short_indices']:
                    proof=b.read(root/'validation/training.json')
                    b.require(proof['exit_code']==0 and proof['full_frozen_complement_exact'] and proof['BF16_decoder_change']
                        and proof['actual_updates']==len(request['short_indices']),'New peak-shape update proof incomplete.')
                report['status']='formal_same_parent_fixed128_started';b.write(root/'run.json',report)
            stage(root,name+'_training',[str(b.PYTHON),'-B','-u','-m',MODULE+'.train','--request',str(request_path)],gpu,report)
        formal=root/'formal';training=b.read(formal/'training.json');checkpoint=training['final']['checkpoint']['path']
        full_nine(formal/'request.json',formal/'full_nine',checkpoint,gpu)
        report.update(status='complete_fixed_local_comparison_arm',exit_code=0);code=0
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-began;b.write(root/'run.json',report);(root/'run.exit').write_text(str(code)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--arm',choices=list(b.ARMS));parser.add_argument('--request',type=Path);parser.add_argument('--summarize',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare(args.prepare.resolve())
    elif args.summarize:summarize(args.request,args.summarize)
    else:execute(args.plan.resolve(strict=True),args.arm)


if __name__=='__main__':main()
