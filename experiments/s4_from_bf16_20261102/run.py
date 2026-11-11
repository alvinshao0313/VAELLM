"""One explicit fresh algorithm trial: necessary short, then fixed formal chain."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b

TERMINAL=b.ROOT/'result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21'
HELPERS=b.ROOT/'result/qwen3_8b_mmlu65_20261004/qwen_aux_retention32_batch_20261011/evaluate.py'
PARTITIONS=b.ROOT/'experiments/cat_aligned_evaluation_20261028/evaluate.py'


def prepare(root):
    b.initialize()
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not root.exists()
        and root.resolve().is_relative_to(b.OUTPUT),'Prepare a new CPU-only isolated trial.')
    b.require(b.CONTRACT['recovery']['budget_rule']!='pending_Astra_adaptive_rule','Adaptive budget rule is not finalized.')
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text())
    from . import budget
    budget_cases=budget.validation_cases()
    from . import fit,recovery,evaluate
    cat,_hf,_training,vae,cfgs,_identity,_normalized,_argv=b.native_recipe(root/'formal/native')
    b.require(cat.after_category_mode=='none' and not cat.rot_llm and vae.model_path==str(b.HF)
        and len(cfgs)==7 and all(cfg.residual_stages==4 and cfg.steps==10000 for cfg in cfgs.values()),
        'Actual native factory/recipe parser differs from the new algorithm contract.')
    from train_utils import cat_train_pipeline as native
    b.require(Path(native.__file__).resolve().is_relative_to(b.parent.s.SNAPSHOT),'Actual native import escaped its frozen source.')
    api=b.parent.module(b.parent.SAMPLE/'eval_utils_with_samples.py','_fresh_S4_CPU_actual_eval_api')
    b.require(callable(api.run_lm_eval),'Actual HFLM worker import context/API is unavailable.')
    shared=b.read(b.data.b.DATA/'data_contract.json')
    pins=b.pins();b.parent.s.check_pins(pins)
    old_proof=b.ROOT/'result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/cpu/validation.json'
    proof=b.read(old_proof)
    b.require(proof['exit_code']==0 and proof['old_M1024_mixed_and_QA_FP32_targets_exact']
        and proof['original_QA_group_suffixes128_exact'],'Reuse the actual verified shared static targets.')
    dataset=b.torch.load(b.data.checked(shared['dataset']),map_location='cpu',weights_only=True)
    case=next(case for case in dataset['training_cases'] if case['case_id']==dataset['groups'][2]['case_ids'][0])
    root.mkdir(parents=True)
    b.torch.save(dict(input_ids=case['input_ids'].unsqueeze(0),
        attention_mask=b.torch.ones_like(case['input_ids']).unsqueeze(0),
        source_case_id=case['case_id'],input_signature=case['input_signature']),root/'fixture.pt')
    identity=b.parent.s.hf_identity(b.HF)
    for short,name in [(True,'short'),(False,'formal')]:
        output=root/name;output.mkdir()
        (output/'fixture.pt').symlink_to(root/'fixture.pt')
        b.write(output/'request.json',dict(schema='fresh_BF16_S4_prefix_trial_v1',output=str(output),
            validation_only=short,source_sha256=pins,contract=b.CONTRACT,HF_identity=identity,
            data_digest=shared['actual_input_target_records_sha256'],data_contract=b.binding(b.data.b.DATA/'data_contract.json'),
            reused_data_CPU=b.binding(old_proof),old_compressed_inputs=[],
            training_sources=shared['source_capture'],physical_GPUs=[6,7]))
    b.write(root/'CPU_validation.json',dict(status='verified_incremental_fresh_S4_driver',exit_code=0,
        source_sha256=pins,old_native_and_data_evidence_reused=True,new_scope='CLI/native imports/hooks/metadata/data identity',
        data_contract=b.binding(b.data.b.DATA/'data_contract.json'),shared_CPU=b.binding(old_proof),
        actual128_exposure=shared['full_exposure'],cuda_initialized=b.torch.cuda.is_initialized(),
        adaptive_budget_arithmetic_cases=budget_cases,
        actual_native_import=str(Path(native.__file__).resolve()),actual_HFLM_API=b.binding(b.parent.SAMPLE/'eval_utils_with_samples.py'),
        numerical_scope='No CPU model reconstruction, GPU computation, or claim of new GPU-path validation.'))
    plan=dict(schema='fresh_BF16_S4_prefix_complete_trial_v1',root=str(root),source_sha256=pins,
        contract=b.CONTRACT,CPU_validation=b.binding(root/'CPU_validation.json'),
        short_request=b.binding(root/'short/request.json'),formal_request=b.binding(root/'formal/request.json'),
        physical_GPUs=[6,7],automatic_retry=False,automatic_research=False)
    b.write(root/'plan.json',plan)
    return plan


def stage(root,name,argv,gpus,report):
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(gpus),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
        report['processes'][name]=dict(command=argv,gpus=gpus,**b.parent.process_event(process.pid))
        b.write(root/'run.json',report)
        code=process.wait()
    report['processes'][name]['exit_code']=code;b.write(root/'run.json',report)
    (root/(name+'.outer.exit')).write_text(str(code)+'\n')
    b.require(code==0,name+' failed; no automatic retry or affected dependent execution.')


def cpu_stage(root,name,argv):
    env=b.environment('');env['PYTHONPATH']=str(b.ROOT)
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        result=subprocess.run(argv,cwd=b.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=False)
    (root/(name+'.exit')).write_text(str(result.returncode)+'\n')
    b.require(result.returncode==0,name+' CPU terminal/merge failed.')


def longbench(request_path,root,checkpoint,report):
    from experiments.longbench_v1.models import make_config,check_config
    from experiments.longbench_v1.common import file_stat
    from experiments.longbench_v1.protocol import DATASETS,EXPECTED_COUNTS
    request=b.read(request_path);event=b.event(checkpoint)
    base=root/'longbench';base.mkdir()
    config=make_config(b.parent.MANIFEST,kind='dense',hf_path=b.HF,mode='full')
    config['model'].update(kind='compressed',checkpoint=str(checkpoint),checkpoint_id=event['checkpoint_id'],
        metadata_sha256=event['metadata_sha256'],weight_stat=file_stat(Path(checkpoint)/'pytorch_model.bin'))
    config['source_sha256'].update(request['source_sha256'])
    config['research_parent']=dict(stage_count=4,unprotected_main_bpw=4,final_lowbit_acceptance_evaluated=False)
    check_config(config);config_path=base/'config.json';b.write(config_path,config)
    # Reuse the established canonical task partition and exact terminal validators.
    if str(b.parent.HERE) not in sys.path:sys.path.insert(0,str(b.parent.HERE))
    partition=b.parent.module(PARTITIONS,'_fresh_S4_original_LB_partitions')
    groups=partition.task_groups(['7','6'])
    units=[]
    for index,(gpu,tasks) in enumerate(zip(['7','6'],groups,strict=True)):
        output=base/'groups'/f'part_{index+1:02d}'
        argv=[str(b.PYTHON),'-B','-u','-m','experiments.s4_from_bf16_20261102.evaluate',
            '--request',str(request_path),'--component','predict','--config',str(config_path),
            '--output',str(output),'--datasets',*tasks]
        units.append(dict(output_dir=str(output),datasets=tasks,expected_samples=sum(EXPECTED_COUNTS[x] for x in tasks),
            native_reports=[],outer_exit=None,launch_records=[],terminal_attempts=[],gpu=gpu,argv=argv,
            native_runtime_seconds=86400,outer_timeout_seconds=86580,termination_grace_seconds=20))
    merged=base/'merged'
    plan=dict(schema='fixed_full21_merge_plan_v1',status='draft_awaiting_explicit_terminal_binding',
        checkpoint_id=event['checkpoint_id'],expected_tasks=list(DATASETS),expected_samples=4750,
        config_path=str(config_path),config_sha256=b.sha(config_path),protocol=config['protocol'],execution_units=units,
        task_file_sources={task:str(Path(unit['output_dir'])/'predictions'/(task+'.jsonl')) for unit in units for task in unit['datasets']},
        pins={**request['source_sha256'],str(config_path):b.sha(config_path)},merged_output=str(merged),
        scoring_cwd=str(b.ROOT),python_executable=str(b.PYTHON),
        nine_task_argv=[str(b.PYTHON),'-B','-m','experiments.s4_from_bf16_20261102.run',
            '--request',str(request_path),'--summarize',str(merged/'nine_task_acceptance.json')],
        automatic_retry=False,automatic_resume=False,partial_scoring=False)
    plan_path=base/'merge_plan.json';b.write(plan_path,plan)
    helper=b.parent.module(HELPERS,'_fresh_S4_original_LB_outer');helper.S=b.ROOT;helper.PYTHON=b.PYTHON
    processes=[]
    try:
        Path(units[0]['output_dir']).mkdir(parents=True);processes.append(helper.start_group(units[0],config,plan['config_sha256']))
        final=root/'nodes/down_proj/post'
        stage(root,'final_seven',[str(b.PYTHON),'-B','-u','-m','experiments.s4_from_bf16_20261102.evaluate',
            '--request',str(request_path),'--checkpoint',str(checkpoint),'--component','seven',
            '--output',str(final/'evaluation')],'6',report)
        Path(units[1]['output_dir']).mkdir(parents=True);processes.append(helper.start_group(units[1],config,plan['config_sha256']))
    finally:codes=[process.wait() for process in processes]
    b.require(len(codes)==2 and all(code==0 for code in codes),'Assigned canonical LongBench partition failed.')
    for index,unit in enumerate(units):
        output=Path(unit['output_dir']);reports=list((output/'runs').glob('*.json'))
        b.require(len(reports)==1,'Require one actual fresh LongBench attempt per partition.')
        native=b.read(reports[0])
        b.require(native['status']=='partition_complete' and native['exit_code']==0
            and native['new_predictions']==unit['expected_samples'] and native['already_present']==0
            and native['load_audit']['checkpoint_id']==event['checkpoint_id'],'Native LongBench terminal/UUID/count differs.')
        cpu_stage(root,f'bind_LB_{index}',[str(b.PYTHON),'-B',str(TERMINAL/'bind_terminal.py'),
            '--plan',str(plan_path),'--unit-index',str(index),'--native-report',str(reports[0]),
            '--outer-launch',str(output/'outer.launch.json'),'--outer-exit',str(output/'outer.exit'),
            '--outer-log',str(output/'outer.log')])
    bound=b.read(plan_path)
    b.require(sum(unit['completed_samples_at_last_terminal'] for unit in bound['execution_units'])==4750
        and not bound['pending_output_directories_at_last_terminal'],'All21 tasks/4750 canonical predictions required.')
    bound['status']='prepared_not_scored';b.write(plan_path,bound)
    cpu_stage(root,'merge_LB_nine',[str(b.PYTHON),'-B',str(TERMINAL/'merge_full21.py'),'--plan',str(plan_path)])
    result=b.read(merged/'nine_task_acceptance.json')
    b.require(result['status']=='complete_fresh_BF16_S4_prefix_trial'
        and result['checkpoint_id']==event['checkpoint_id'],'Final same-UUID nine-task result missing.')
    (root/'result.json').symlink_to(merged/'nine_task_acceptance.json')


def summarize(request_path,output):
    from experiments.w25_eval.model_inventory import model_inventory
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.transfer_nine_task_acceptance import _sample_artifacts
    from experiments.compare_longbench_v1 import comparison
    from train_utils.checkpoint_bpw import audit_checkpoint
    request=b.read(request_path);root=Path(request['output']);fitted=b.read(root/'fit.json')
    checkpoint=Path(fitted['nodes'][-1]['post']['checkpoint']['path']);meta=b.read(checkpoint/'checkpoint_meta.json')
    paths=[root/'nodes/down_proj/post/evaluation'/name for name in ['mmlu.json','seven.json']]
    inventory=model_inventory(b.HF);low=collect_results(paths,inventory,checkpoint=(checkpoint,meta))
    baseline=collect_results([b.parent.BASELINE/'mmlu.json',b.parent.BASELINE/'seven.json'],inventory)
    b.require(low['versions']==baseline['versions'],'Full-eight versions differ from BF16.')
    lb=comparison(b.parent.BASELINE/'longbench/scores.json',root/'longbench/merged/scores.json',str(b.parent.MANIFEST))
    budget=audit_checkpoint(checkpoint,max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,'Final actual S4 ledger changed.')
    scores={key:value*100 for key,value in low['task_metrics'].items()};dense={key:value*100 for key,value in baseline['task_metrics'].items()}
    scores['longbench']=lb['task_macro_mean']['compressed'];dense['longbench']=lb['task_macro_mean']['dense']
    b.write(output,dict(status='complete_fresh_BF16_S4_prefix_trial',checkpoint_id=meta['checkpoint_id'],
        checkpoint=b.event(checkpoint),scores_percent=scores,BF16_percent=dense,
        deltas_pp={key:scores[key]-dense[key] for key in scores},nine_mean_percent=math.fsum(scores.values())/9,
        BF16_nine_mean_percent=math.fsum(dense.values())/9,bit_budget=budget,raw_samples=_sample_artifacts(paths),
        contract=b.CONTRACT,actual_category_updates=[row['actual_updates'] for row in fitted['recoveries']],
        source_HF_identity=request['HF_identity'],old_compressed_inputs=[],final_lowbit_acceptance_evaluated=False))


def execute(plan_path):
    plan=b.read(plan_path);root=Path(plan['root'])
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (root/'run.json').exists(),'New CPU controller only.')
    b.parent.s.check_pins(plan['source_sha256'])
    report=dict(status='validating_new_fresh_algorithm_path',**b.parent.process_event(os.getpid()),
        started_unix=time.time(),plan_sha256=b.sha(plan_path),source_sha256=plan['source_sha256'],processes={})
    b.write(root/'run.json',report);code=1
    try:
        for name in ['short','formal']:
            request_path=root/name/'request.json'
            if name=='formal':
                proof=b.read(root/'short/validation.json')
                b.require(proof['status']=='verified_fresh_BF16_S4_prefix_GPU_short'
                    and proof['source_sha256']==plan['source_sha256'],'Formal needs this actual new path short evidence.')
                report['status']='formal_fresh_BF16_algorithm_started';b.write(root/'run.json',report)
            stage(root,name+'_fit',[str(b.PYTHON),'-B','-u','-m','experiments.s4_from_bf16_20261102.fit',
                '--request',str(request_path)],'6,7',report)
            fitted=b.read(root/name/'fit.json')
            b.require(fitted['status']=='complete_fresh_BF16_S4_prefix_recovery' and len(fitted['recoveries'])==7,
                'Real seven-category fit and recovery chain incomplete.')
            if name=='short':
                midpoint=root/name/'nodes/down_proj/recovery/step_000002'
                final=root/name/'nodes/down_proj/post/evaluation/fresh.json'
                b.require(b.read(final)['full_state_exact'] and b.read(midpoint/'evaluation/fresh.json')['full_state_exact']
                    and b.read(root/name/'nodes/down_proj/recovery/resume_validation.json')['status']=='actual_FP32_Adam_RNG_state_restored'
                    and all(group['complete_noncurrent_fields_exact_after_fit'] for group in fitted['groups'])
                    and all(row['full_frozen_complement_exact'] for row in fitted['recoveries']),
                    'Actual fit/recover/next-fit/frozen/snapshot/fresh/resume contracts must pass.')
                b.write(root/name/'validation.json',dict(status='verified_fresh_BF16_S4_prefix_GPU_short',
                    source_sha256=plan['source_sha256'],all_seven_real_fits_and_recoveries=True,
                    real_next_category_preserved_prior_BF16=True,all_frozen_complements_exact=True,
                    full252_peak_groups=[2,10,45],midpoint_fresh_and_readout_then_resumed_update=True,
                    final_fresh=b.binding(final),elapsed_seconds=fitted['elapsed_seconds']))
                # All short payload readers have retired; formal starts again at HF.
                cleanup=b.cleanup_model(fitted['nodes'][-1]['post']['checkpoint']['path'],
                    'The complete short and fresh evidence is saved; formal consumes original HF, not the short model.')
                validation=b.read(root/name/'validation.json');validation['completed_payload_cleanup']=cleanup
                b.write(root/name/'validation.json',validation)
        formal=root/'formal';fitted=b.read(formal/'fit.json')
        checkpoint=fitted['nodes'][-1]['post']['checkpoint']['path']
        longbench(formal/'request.json',formal,checkpoint,report)
        report.update(status='complete_fresh_BF16_S4_algorithm_trial',exit_code=0);code=0
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.time()-report['started_unix'];b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--request',type=Path);parser.add_argument('--summarize',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare(args.prepare.resolve())
    elif args.summarize:summarize(args.request,args.summarize)
    else:execute(args.plan.resolve(strict=True))


if __name__=='__main__':main()
